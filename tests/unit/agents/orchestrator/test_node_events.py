"""Graph topology + node-invocation events for OrchestratorBot.

Covers:
  - get_graph_topology(): nodes/edges/entry of the compiled graph
  - astream_events: first event is ("graph", graph_topology) with a run_id
  - ("graph", node_invoked) signals: one per node execution, ordered seq
  - run_id: unique per run (date-prefixed, never repeats), consistent across
    topology, node events and BotResponse
  - arun/run: BotResponse.node_trace + BotResponse.run_id

Uses mocked LLM — no API keys needed.
"""

import asyncio
import re

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk

from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot

RUN_ID_RE = re.compile(r"^\d{8}T\d{12}-[0-9a-f]{32}$")


def _make_bot(mock_model, tools=None, **kwargs):
    lm = MagicMock()
    lm.model = mock_model
    lm.supports_thinking = False
    return OrchestratorBot(
        strong_model=lm,
        fast_model=lm,
        instructions="You are a test bot.",
        tools=tools or [],
        **kwargs,
    )


def _final_only_model(text="Hecho."):
    mock_model = MagicMock()

    async def fake_astream(messages):
        yield AIMessageChunk(content=text)

    mock_model.bind_tools.return_value = mock_model
    mock_model.astream = fake_astream
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content=text))
    return mock_model


def _tool_then_final_model():
    mock_model = MagicMock()
    chunk_tool = AIMessageChunk(
        content="", tool_calls=[{"id": "t1", "name": "test_tool", "args": {"x": 1}}]
    )
    chunk_final = AIMessageChunk(content="Hecho.")
    call_count = 0

    async def fake_astream(messages):
        nonlocal call_count
        call_count += 1
        yield chunk_tool if call_count == 1 else chunk_final

    mock_model.bind_tools.return_value = mock_model
    mock_model.astream = fake_astream
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hecho."))
    return mock_model


def _make_tool(name="test_tool"):
    tool = MagicMock()
    tool.name = name
    tool.description = "A test tool"
    tool.args_schema = None
    tool.ainvoke = AsyncMock(return_value="tool output")
    tool.risk_level = 0
    tool.risk_hint = 0
    return tool


async def _collect_events(bot, goal="Hola", mode="auto"):
    events = []
    async for stream_mode, payload in bot.astream_events(goal, mode=mode):
        events.append((stream_mode, payload))
    return events


def _graph_events(events, event_type=None):
    payloads = [p for m, p in events if m == "graph"]
    if event_type:
        payloads = [p for p in payloads if p.get("type") == event_type]
    return payloads


class TestGraphTopology:
    def test_topology_has_builtin_nodes_and_entry(self):
        bot = _make_bot(_final_only_model())
        topo = bot.get_graph_topology()
        assert topo["entry"] == "agent"
        assert {"__start__", "agent", "tools", "__end__"} <= set(topo["nodes"])

    def test_topology_edges_shape(self):
        bot = _make_bot(_final_only_model())
        topo = bot.get_graph_topology()
        assert topo["edges"], "expected at least one edge"
        for edge in topo["edges"]:
            assert set(edge) == {"source", "target", "conditional"}
        # agent → tools is conditional (router), tools → agent is not.
        assert any(
            e["source"] == "agent" and e["target"] == "tools" and e["conditional"]
            for e in topo["edges"]
        )
        assert any(
            e["source"] == "tools" and e["target"] == "agent" and not e["conditional"]
            for e in topo["edges"]
        )


@pytest.mark.asyncio
class TestStreamGraphEvents:
    async def test_first_event_is_graph_topology(self):
        bot = _make_bot(_final_only_model())
        events = await _collect_events(bot)
        stream_mode, payload = events[0]
        assert stream_mode == "graph"
        assert payload["type"] == "graph_topology"
        assert payload["entry"] == "agent"
        assert "agent" in payload["nodes"]
        assert payload["edges"]
        assert RUN_ID_RE.match(payload["run_id"])

    async def test_node_invoked_emitted_per_node_in_order(self):
        bot = _make_bot(_tool_then_final_model(), tools=[_make_tool()])
        events = await _collect_events(bot, goal="usa la tool")
        invoked = _graph_events(events, "node_invoked")
        assert [ev["node"] for ev in invoked] == ["agent", "tools", "agent"]
        assert [ev["seq"] for ev in invoked] == [1, 2, 3]
        assert all(isinstance(ev["ts"], float) for ev in invoked)

    async def test_run_id_consistent_across_run_events(self):
        bot = _make_bot(_final_only_model())
        events = await _collect_events(bot)
        graph_payloads = _graph_events(events)
        run_ids = {p["run_id"] for p in graph_payloads}
        assert len(run_ids) == 1

    async def test_run_id_never_repeats_between_runs(self):
        bot = _make_bot(_final_only_model())
        first = _graph_events(await _collect_events(bot), "graph_topology")[0]
        second = _graph_events(await _collect_events(bot), "graph_topology")[0]
        assert first["run_id"] != second["run_id"]

    async def test_updates_stream_still_emitted(self):
        """Backward compat: existing consumers of "updates" keep working."""
        bot = _make_bot(_final_only_model())
        events = await _collect_events(bot)
        update_nodes = [
            name
            for mode, payload in events
            if mode == "updates" and isinstance(payload, dict)
            for name in payload
        ]
        assert "agent" in update_nodes


@pytest.mark.asyncio
class TestNodeTraceInResponse:
    async def test_arun_returns_node_trace_and_run_id(self):
        bot = _make_bot(_tool_then_final_model(), tools=[_make_tool()])
        result = await bot.arun("usa la tool")
        assert RUN_ID_RE.match(result.run_id)
        assert [e["node"] for e in result.node_trace] == ["agent", "tools", "agent"]
        assert [e["seq"] for e in result.node_trace] == [1, 2, 3]
        assert all(e["run_id"] == result.run_id for e in result.node_trace)
        assert all(isinstance(e["ts"], float) for e in result.node_trace)

    async def test_arun_run_ids_unique(self):
        bot = _make_bot(_final_only_model())
        r1 = await bot.arun("Hola")
        r2 = await bot.arun("Hola")
        assert r1.run_id != r2.run_id


def _planning_model():
    """set_plan → step running + tool → step done → final (mirrors test_graph_planning)."""
    mock_model = MagicMock()
    chunks = [
        AIMessageChunk(content="", tool_calls=[
            {"id": "s1", "name": "set_plan",
             "args": {"steps": ["Buscar datos", "Generar reporte"]}}]),
        AIMessageChunk(content="Paso 1...", tool_calls=[
            {"id": "s2", "name": "update_step", "args": {"step": 1, "status": "running"}},
            {"id": "t1", "name": "test_tool", "args": {"x": 1}}]),
        AIMessageChunk(content="", tool_calls=[
            {"id": "s3", "name": "update_step", "args": {"step": 1, "status": "done"}}]),
        AIMessageChunk(content="Todo listo."),
    ]
    n = 0

    async def fake_astream(messages):
        nonlocal n
        idx = min(n, len(chunks) - 1)
        n += 1
        yield chunks[idx]

    mock_model.bind_tools.return_value = mock_model
    mock_model.astream = fake_astream
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Todo listo."))
    return mock_model


def _asking_model():
    """First turn calls ask_user; second turn answers with final text."""
    mock_model = MagicMock()
    ask_chunk = AIMessageChunk(content="", tool_calls=[{
        "id": "q1", "name": "ask_user",
        "args": {"questions": [{"id": "color", "text": "¿Qué color?", "type": "text"}]},
    }])
    n = 0

    async def fake_astream(messages):
        nonlocal n
        n += 1
        yield ask_chunk if n == 1 else AIMessageChunk(content="Perfecto, azul.")

    mock_model.bind_tools.return_value = mock_model
    mock_model.astream = fake_astream
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Perfecto, azul."))
    return mock_model


@pytest.mark.asyncio
class TestNodeDetail:
    """Node events carry the node's params (args) and output in `detail`."""

    async def test_tools_node_detail_has_args_and_output(self):
        bot = _make_bot(_tool_then_final_model(), tools=[_make_tool()])
        events = await _collect_events(bot, goal="usa la tool")
        tools_ev = next(e for e in _graph_events(events, "node_invoked")
                        if e["node"] == "tools")
        executed = tools_ev["detail"]["tools_executed"]
        assert executed[0]["tool_name"] == "test_tool"
        assert executed[0]["args"] == {"x": 1}
        assert executed[0]["status"] == "success"
        assert "tool output" in executed[0]["output"]

    async def test_agent_node_detail_has_tool_calls_and_output(self):
        bot = _make_bot(_tool_then_final_model(), tools=[_make_tool()])
        events = await _collect_events(bot, goal="usa la tool")
        agent_evs = [e for e in _graph_events(events, "node_invoked")
                     if e["node"] == "agent"]
        # First agent turn requested the tool (with args); last one emitted text.
        assert agent_evs[0]["detail"]["tool_calls"] == [
            {"name": "test_tool", "args": {"x": 1}}]
        assert agent_evs[-1]["detail"]["output"] == "Hecho."

    async def test_node_trace_entries_carry_detail(self):
        bot = _make_bot(_tool_then_final_model(), tools=[_make_tool()])
        result = await bot.arun("usa la tool")
        by_node = {e["node"]: e for e in result.node_trace}
        assert by_node["tools"]["detail"]["tools_executed"][0]["args"] == {"x": 1}
        assert by_node["agent"]["detail"]  # agent entries have detail too


@pytest.mark.asyncio
class TestPlanNode:
    """With enable_planning=True the plan is a real graph node."""

    async def test_plan_node_in_topology(self):
        bot = _make_bot(_planning_model(), tools=[_make_tool()], enable_planning=True)
        topo = bot.get_graph_topology()
        assert "plan" in topo["nodes"]
        assert any(e["source"] == "agent" and e["target"] == "plan"
                   for e in topo["edges"])
        assert any(e["source"] == "plan" and e["target"] == "tools"
                   for e in topo["edges"])
        assert any(e["source"] == "plan" and e["target"] == "agent"
                   for e in topo["edges"])

    async def test_plan_node_absent_when_disabled(self):
        bot = _make_bot(_final_only_model())
        assert "plan" not in bot.get_graph_topology()["nodes"]

    async def test_plan_node_invoked_with_detail(self):
        bot = _make_bot(_planning_model(), tools=[_make_tool()], enable_planning=True)
        events = await _collect_events(bot, goal="Haz dos cosas")
        plan_evs = [e for e in _graph_events(events, "node_invoked")
                    if e["node"] == "plan"]
        assert plan_evs, "plan node never invoked"
        # First plan-node run registers the snapshot.
        snapshot = plan_evs[0]["detail"]["plan"]
        assert [s["description"] for s in snapshot] == ["Buscar datos", "Generar reporte"]
        # A later run carries the running step event.
        assert any(
            {"step": 1, "status": "running"} in (e["detail"].get("step_events") or [])
            for e in plan_evs
        )

    async def test_node_sequence_shows_plan_steps(self):
        bot = _make_bot(_planning_model(), tools=[_make_tool()], enable_planning=True)
        events = await _collect_events(bot, goal="Haz dos cosas")
        seq = [e["node"] for e in _graph_events(events, "node_invoked")]
        assert seq == ["agent", "plan", "agent", "plan", "tools",
                       "agent", "plan", "agent"]

    async def test_arun_plan_still_populated(self):
        bot = _make_bot(_planning_model(), tools=[_make_tool()], enable_planning=True)
        result = await bot.arun("Haz dos cosas")
        assert result.content == "Todo listo."
        statuses = {s["step"]: s["status"] for s in result.plan}
        assert statuses == {1: "done", 2: "pending"}


@pytest.mark.asyncio
class TestAskUserNode:
    """With enable_user_questions=True, ask_user is a real graph node."""

    async def test_ask_user_node_in_topology(self):
        bot = _make_bot(_asking_model(), enable_user_questions=True)
        topo = bot.get_graph_topology()
        assert "ask_user" in topo["nodes"]
        assert any(e["source"] == "ask_user" and e["target"] == "agent"
                   for e in topo["edges"])

    async def test_ask_user_node_absent_when_disabled(self):
        bot = _make_bot(_final_only_model())
        assert "ask_user" not in bot.get_graph_topology()["nodes"]

    async def test_interrupt_then_resume_runs_ask_user_node(self):
        bot = _make_bot(_asking_model(), enable_user_questions=True)
        thread = "t-ask-1"

        interrupted = False
        async for stream_mode, payload in bot.astream_events(
                "Pinta algo", mode="auto", thread_id=thread):
            if (stream_mode == "updates" and isinstance(payload, dict)
                    and "__interrupt__" in payload):
                interrupted = True
        assert interrupted, "run did not pause on the ask_user interrupt"

        bot.set_resume_command({"color": "azul"})
        nodes_after = []
        final = None
        async for stream_mode, payload in bot.astream_events(
                None, mode="auto", thread_id=thread):
            if stream_mode == "graph" and payload["type"] == "node_invoked":
                nodes_after.append(payload)
            if stream_mode == "updates" and isinstance(payload, dict):
                final = payload.get("agent", {}).get("final_report") or final

        assert [e["node"] for e in nodes_after] == ["ask_user", "agent"]
        ask_ev = nodes_after[0]
        assert "azul" in ask_ev["detail"]["output"]
        exec_rec = ask_ev["detail"]["tools_executed"][0]
        assert exec_rec["tool_name"] == "ask_user"
        assert exec_rec["args"]["questions"][0]["id"] == "color"
        assert final == "Perfecto, azul."

    async def test_arun_surfaces_questions_on_interrupt(self):
        bot = _make_bot(_asking_model(), enable_user_questions=True)
        result = await bot.arun("Pinta algo")
        assert result.needs_input is True
        assert result.questions[0]["id"] == "color"


def _mixed_ask_and_tool_model():
    """First turn: ask_user + a real tool in the SAME batch; then final."""
    mock_model = MagicMock()
    mixed_chunk = AIMessageChunk(content="", tool_calls=[
        {"id": "q1", "name": "ask_user",
         "args": {"questions": [{"id": "color", "text": "¿Color?", "type": "text"}]}},
        {"id": "t1", "name": "test_tool", "args": {"x": 1}},
    ])
    n = 0

    async def fake_astream(messages):
        nonlocal n
        n += 1
        yield mixed_chunk if n == 1 else AIMessageChunk(content="Listo con azul.")

    mock_model.bind_tools.return_value = mock_model
    mock_model.astream = fake_astream
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Listo con azul."))
    return mock_model


def _plan_and_ask_model():
    """First turn: update_step signal + ask_user in the SAME batch; then final."""
    mock_model = MagicMock()
    chunks = [
        AIMessageChunk(content="", tool_calls=[
            {"id": "s1", "name": "set_plan", "args": {"steps": ["Preguntar", "Hacer"]}}]),
        AIMessageChunk(content="", tool_calls=[
            {"id": "s2", "name": "update_step", "args": {"step": 1, "status": "running"}},
            {"id": "q1", "name": "ask_user",
             "args": {"questions": [{"id": "color", "text": "¿Color?", "type": "text"}]}},
        ]),
        AIMessageChunk(content="Hecho."),
    ]
    n = 0

    async def fake_astream(messages):
        nonlocal n
        idx = min(n, len(chunks) - 1)
        n += 1
        yield chunks[idx]

    mock_model.bind_tools.return_value = mock_model
    mock_model.astream = fake_astream
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hecho."))
    return mock_model


@pytest.mark.asyncio
class TestMixedBatches:
    """Batches mixing ask_user / plan signals / real tools stay consistent."""

    async def _run_with_resume(self, bot, goal, answers, thread):
        async for _ in bot.astream_events(goal, mode="auto", thread_id=thread):
            pass
        bot.set_resume_command(answers)
        nodes, final = [], None
        async for mode, payload in bot.astream_events(None, mode="auto", thread_id=thread):
            if mode == "graph" and payload["type"] == "node_invoked":
                nodes.append(payload["node"])
            if mode == "updates" and isinstance(payload, dict):
                final = (payload.get("agent") or {}).get("final_report") or final
        return nodes, final

    async def test_ask_user_wins_over_real_tool_and_defers_it(self):
        tool = _make_tool()
        bot = _make_bot(_mixed_ask_and_tool_model(), tools=[tool],
                        enable_user_questions=True)
        nodes, final = await self._run_with_resume(
            bot, "algo", {"color": "azul"}, "t-mixed-1")
        # ask_user ran; the real tool was deferred, NOT executed.
        assert nodes[0] == "ask_user"
        assert "tools" not in nodes
        tool.ainvoke.assert_not_called()
        assert final == "Listo con azul."

    async def test_every_tool_call_gets_a_toolmessage_answer(self):
        """History integrity: no dangling tool_call ids after the deferral."""
        from langchain_core.messages import ToolMessage
        bot = _make_bot(_mixed_ask_and_tool_model(), tools=[_make_tool()],
                        enable_user_questions=True)
        thread = "t-mixed-2"
        async for _ in bot.astream_events("algo", mode="auto", thread_id=thread):
            pass
        bot.set_resume_command({"color": "azul"})
        async for _ in bot.astream_events(None, mode="auto", thread_id=thread):
            pass
        state = bot.graph.get_state({"configurable": {"thread_id": thread}})
        messages = state.values["messages"]
        called_ids = {tc["id"] for m in messages
                      for tc in (getattr(m, "tool_calls", None) or [])}
        answered_ids = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
        assert called_ids == answered_ids

    async def test_plan_signal_and_ask_user_same_batch(self):
        bot = _make_bot(_plan_and_ask_model(), enable_planning=True,
                        enable_user_questions=True)
        thread = "t-plan-ask"
        first_nodes = []
        async for mode, payload in bot.astream_events(
                "algo", mode="auto", thread_id=thread):
            if mode == "graph" and payload["type"] == "node_invoked":
                first_nodes.append(payload["node"])
        # Batch 2 (update_step + ask_user) routed agent → plan → (interrupt).
        assert first_nodes == ["agent", "plan", "agent", "plan"]

        bot.set_resume_command({"color": "azul"})
        resumed, final = [], None
        async for mode, payload in bot.astream_events(
                None, mode="auto", thread_id=thread):
            if mode == "graph" and payload["type"] == "node_invoked":
                resumed.append(payload["node"])
            if mode == "updates" and isinstance(payload, dict):
                final = (payload.get("agent") or {}).get("final_report") or final
        assert resumed == ["ask_user", "agent"]
        assert final == "Hecho."
        # The plan progressed despite sharing the batch with ask_user.
        state = bot.graph.get_state({"configurable": {"thread_id": thread}})
        assert {s["step"]: s["status"] for s in state.values["plan"]}[1] == "running"


def _counting_model():
    """agent round 1 requests a tool, round 2+ finalizes. Exposes .calls so a
    test can prove the graph did NOT advance to the next agent round."""
    mock_model = MagicMock()
    chunk_tool = AIMessageChunk(
        content="", tool_calls=[{"id": "t1", "name": "test_tool", "args": {"x": 1}}]
    )
    chunk_final = AIMessageChunk(content="Hecho.")
    state = {"calls": 0}

    async def fake_astream(messages):
        state["calls"] += 1
        yield chunk_tool if state["calls"] == 1 else chunk_final

    mock_model.bind_tools.return_value = mock_model
    mock_model.astream = fake_astream
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hecho."))
    mock_model._calls = state
    return mock_model


class TestAbort:
    """bot.abort() actually halts the graph — not just the event stream."""

    @pytest.mark.asyncio
    async def test_abort_halts_graph_execution(self):
        # Model would run agent(round1) → tools → agent(round2). Aborting right
        # after the first event must stop the graph BEFORE the tools node and
        # the second agent round ever run.
        model = _counting_model()
        bot = _make_bot(model, tools=[_make_tool()])
        events = []
        async for stream_mode, payload in bot.astream_events("Hazlo", mode="auto"):
            events.append((stream_mode, payload))
            if len(events) == 1:      # first event is the graph_topology
                bot.abort()

        # Last event is the aborted signal, and it carries a run_id.
        assert events[-1] == ("graph", {"type": "aborted", "run_id": events[-1][1]["run_id"]})
        assert events[-1][1]["run_id"]

        # DECISIVE: the graph genuinely stopped. A full run calls the model
        # twice (round1 → tools → round2); after abort it was called exactly
        # once — the second agent round never happened.
        assert model._calls["calls"] == 1
        # The tools node never executed.
        assert all(n["node"] != "tools" for n in _graph_events(events, "node_invoked"))

        # Nothing keeps running in the background: after waiting, the model
        # call count is unchanged (execution is pull-driven and was cancelled).
        await asyncio.sleep(0.05)
        assert model._calls["calls"] == 1
        # Flag is reset so the bot is reusable.
        assert bot._abort_requested is False

    @pytest.mark.asyncio
    async def test_run_completes_normally_when_not_aborted(self):
        # Sanity: without abort the same setup finishes, runs the tools node,
        # calls the model twice, and never emits "aborted".
        model = _counting_model()
        bot = _make_bot(model, tools=[_make_tool()])
        events = await _collect_events(bot, goal="Hazlo")
        assert not _graph_events(events, "aborted")
        assert "tools" in [n["node"] for n in _graph_events(events, "node_invoked")]
        assert model._calls["calls"] == 2
