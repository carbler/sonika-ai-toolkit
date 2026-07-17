"""Graph topology + node-invocation events for OrchestratorBot.

Covers:
  - get_graph_topology(): nodes/edges/entry of the compiled graph
  - astream_events: first event is ("graph", graph_topology) with a run_id
  - ("graph", node_invoked) signals: one per node execution, ordered seq
  - run_id: unique per run (date-prefixed, never repeats), consistent across
    topology, node events and BotResponse
  - arun/run: BotResponse.node_trace + BotResponse.run_id
  - custom nodes appear in both topology and node events

Uses mocked LLM — no API keys needed.
"""

import re

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk

from sonika_ai_toolkit.agents.extensions import CustomNode
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

    def test_topology_includes_custom_nodes(self):
        bot = _make_bot(
            _final_only_model(),
            custom_nodes=[CustomNode(name="audit", node=lambda s: {}, position="start")],
        )
        topo = bot.get_graph_topology()
        assert "audit" in topo["nodes"]
        assert topo["entry"] == "audit"


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

    async def test_custom_node_emits_node_invoked(self):
        bot = _make_bot(
            _final_only_model(),
            custom_nodes=[CustomNode(name="audit", node=lambda s: {}, position="start")],
        )
        events = await _collect_events(bot)
        invoked = _graph_events(events, "node_invoked")
        assert [ev["node"] for ev in invoked][:2] == ["audit", "agent"]

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

    async def test_custom_node_recorded_in_trace(self):
        bot = _make_bot(
            _final_only_model(),
            custom_nodes=[CustomNode(name="audit", node=lambda s: {}, position="start")],
        )
        result = await bot.arun("Hola")
        assert [e["node"] for e in result.node_trace][:2] == ["audit", "agent"]
