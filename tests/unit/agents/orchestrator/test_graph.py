"""Behavioral tests for partial response emission in OrchestratorBot.

Uses mocked LLM — no API keys needed.
"""

import asyncio
import time

import pytest
from unittest.mock import MagicMock, AsyncMock

from langchain_core.messages import AIMessage, AIMessageChunk

from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot, _extract_text_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot(mock_model, tools=None, **kwargs):
    """Build an OrchestratorBot with a mocked ILanguageModel."""
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


def _ai_chunk(content, tool_calls=None):
    """Create an AIMessageChunk with optional tool_calls."""
    chunk = AIMessageChunk(content=content, tool_calls=tool_calls or [])
    return chunk


# ---------------------------------------------------------------------------
# Unit: _extract_text_content
# ---------------------------------------------------------------------------

class TestExtractTextContent:

    def test_plain_string(self):
        msg = AIMessage(content="Hello world")
        assert _extract_text_content(msg) == "Hello world"

    def test_list_with_thinking_filtered(self):
        msg = AIMessage(content=[
            {"type": "thinking", "thinking": "Let me think..."},
            {"type": "text", "text": "Here is the answer"},
        ])
        assert _extract_text_content(msg) == "Here is the answer"

    def test_empty_content(self):
        msg = AIMessage(content="")
        assert _extract_text_content(msg) == ""

    def test_mixed_string_and_dict(self):
        msg = AIMessage(content=[
            "prefix ",
            {"type": "text", "text": "suffix"},
        ])
        assert _extract_text_content(msg) == "prefix suffix"


# ---------------------------------------------------------------------------
# Integration: OrchestratorBot with mocked LLM
# ---------------------------------------------------------------------------

TOOL_CALL_1 = {"id": "tc_1", "name": "test_tool", "args": {"x": 1}}


@pytest.mark.asyncio
class TestPartialResponseEmission:

    async def test_agent_emits_partial_when_text_and_tools(self):
        """When the LLM returns text AND tool_calls, the update should contain partial_response."""
        mock_model = MagicMock()

        # First call: text + tool_calls (intermediate turn)
        chunk_with_tools = AIMessageChunk(
            content="Working on task 1...",
            tool_calls=[TOOL_CALL_1],
        )

        # Second call: text only (final turn)
        chunk_final = AIMessageChunk(content="All done!")

        # astream yields chunks; we use side_effect to return different iterators
        call_count = 0

        async def fake_astream(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield chunk_with_tools
            else:
                yield chunk_final

        mock_model.bind_tools.return_value = mock_model
        mock_model.astream = fake_astream
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="All done!"))

        # Create a simple tool mock
        tool = MagicMock()
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.args_schema = None
        tool.ainvoke = AsyncMock(return_value="tool output")
        tool.risk_level = 0
        tool.risk_hint = 0

        bot = _make_bot(mock_model, tools=[tool])

        # Collect updates from the stream
        updates = []
        async for event in bot.astream_events("Do two tasks", mode="auto"):
            updates.append(event)

        # Find agent updates with partial_responses
        agent_updates = []
        for stream_mode, payload in updates:
            if stream_mode == "updates":
                if isinstance(payload, dict) and "agent" in payload:
                    agent_updates.append(payload["agent"])

        # At least one agent update should have partial_responses
        partials = [u for u in agent_updates if u.get("partial_responses")]
        assert len(partials) >= 1
        assert partials[0]["partial_responses"] == ["Working on task 1..."]

    async def test_no_partial_on_final_turn(self):
        """When the LLM returns text only (no tool_calls), only final_report is set."""
        mock_model = MagicMock()

        chunk_final = AIMessageChunk(content="Here is the answer.")

        async def fake_astream(messages):
            yield chunk_final

        mock_model.bind_tools.return_value = mock_model
        mock_model.astream = fake_astream
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Here is the answer."))

        bot = _make_bot(mock_model)

        updates = []
        async for event in bot.astream_events("Simple question", mode="auto"):
            updates.append(event)

        agent_updates = []
        for stream_mode, payload in updates:
            if stream_mode == "updates":
                if isinstance(payload, dict) and "agent" in payload:
                    agent_updates.append(payload["agent"])

        assert len(agent_updates) >= 1
        # Final turn should have final_report, not partial_response
        final = agent_updates[-1]
        assert "final_report" in final
        assert final["final_report"] == "Here is the answer."
        assert not final.get("partial_responses")

    async def test_partial_responses_accumulate(self):
        """Multi-turn: 2 turns with text+tools, 1 final → state accumulates 2 partials."""
        mock_model = MagicMock()

        tc1 = {"id": "tc_1", "name": "test_tool", "args": {"x": 1}}
        tc2 = {"id": "tc_2", "name": "test_tool", "args": {"x": 2}}

        chunk1 = AIMessageChunk(content="Starting task 1...", tool_calls=[tc1])
        chunk2 = AIMessageChunk(content="Starting task 2...", tool_calls=[tc2])
        chunk_final = AIMessageChunk(content="All tasks complete.")

        call_count = 0

        async def fake_astream(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield chunk1
            elif call_count == 2:
                yield chunk2
            else:
                yield chunk_final

        mock_model.bind_tools.return_value = mock_model
        mock_model.astream = fake_astream
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="All tasks complete."))

        tool = MagicMock()
        tool.name = "test_tool"
        tool.description = "A test tool"
        tool.args_schema = None
        tool.ainvoke = AsyncMock(return_value="tool output")
        tool.risk_level = 0
        tool.risk_hint = 0

        bot = _make_bot(mock_model, tools=[tool])

        agent_updates = []
        async for stream_mode, payload in bot.astream_events("Do three things", mode="auto"):
            if stream_mode == "updates":
                if isinstance(payload, dict) and "agent" in payload:
                    agent_updates.append(payload["agent"])

        partials = [u for u in agent_updates if u.get("partial_responses")]
        assert len(partials) == 2
        assert partials[0]["partial_responses"] == ["Starting task 1..."]
        assert partials[1]["partial_responses"] == ["Starting task 2..."]

        # Final update should have final_report
        final = [u for u in agent_updates if "final_report" in u]
        assert len(final) >= 1
        assert final[-1]["final_report"] == "All tasks complete."


# ---------------------------------------------------------------------------
# Parallel tool execution (tools_node)
# ---------------------------------------------------------------------------


class _FakeTool:
    """Lightweight tool with an async ``ainvoke`` and a call counter.

    Registry only needs ``.name``; execution path prefers ``ainvoke``.
    """

    def __init__(self, name, coro=None, risk_level=0):
        self.name = name
        self.description = f"{name} tool"
        self.args_schema = None
        self.risk_level = risk_level
        self.risk_hint = risk_level
        self._coro = coro
        self.calls = 0

    async def ainvoke(self, args):
        self.calls += 1
        if self._coro is not None:
            return await self._coro(args)
        return f"{self.name}:ok"


def _make_streaming_model(chunks):
    """Mock model whose astream yields chunks[i] on the i-th agent turn.

    Clamps to the last chunk so extra turns keep returning the final one.
    """
    mock_model = MagicMock()
    state = {"i": 0}

    async def fake_astream(messages):
        i = state["i"]
        state["i"] += 1
        yield chunks[i] if i < len(chunks) else chunks[-1]

    mock_model.bind_tools.return_value = mock_model
    mock_model.astream = fake_astream
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="done"))
    return mock_model


def _tools_executed_from(events):
    """Flatten tools_executed records from collected ("updates", …) events."""
    executed = []
    for stream_mode, payload in events:
        if stream_mode == "updates" and isinstance(payload, dict) and "tools" in payload:
            executed.extend((payload["tools"] or {}).get("tools_executed", []) or [])
    return executed


def _tc(id_, name, args=None):
    return {"id": id_, "name": name, "args": args or {}}


@pytest.mark.asyncio
class TestParallelToolExecution:

    async def test_independent_tools_run_concurrently(self):
        """Two 0.25s tools in one turn finish in ~0.25s, not ~0.5s (serial)."""
        delay = 0.25

        async def slow(args):
            await asyncio.sleep(delay)
            return "slow done"

        t1 = _FakeTool("tool_a", slow)
        t2 = _FakeTool("tool_b", slow)
        chunk1 = AIMessageChunk(content="", tool_calls=[_tc("a", "tool_a"), _tc("b", "tool_b")])
        chunk_final = AIMessageChunk(content="Done.")
        bot = _make_bot(_make_streaming_model([chunk1, chunk_final]), tools=[t1, t2])

        start = time.perf_counter()
        async for _ in bot.astream_events("do both", mode="auto"):
            pass
        elapsed = time.perf_counter() - start

        # Serial would be >= 2*delay (0.5s); concurrent ~delay (0.25s) + overhead.
        assert elapsed < 2 * delay, f"tools ran serially: {elapsed:.3f}s"
        assert t1.calls == 1 and t2.calls == 1

    async def test_results_preserve_batch_order(self):
        """tools_executed follows the tool_call order, not completion order."""

        async def slow(args):
            await asyncio.sleep(0.15)
            return "slow"

        # tool_0 finishes last, tool_2 first — output order must still be 0,1,2.
        tools = [
            _FakeTool("tool_0", slow),
            _FakeTool("tool_1"),
            _FakeTool("tool_2"),
        ]
        tcs = [_tc(str(i), f"tool_{i}") for i in range(3)]
        chunk1 = AIMessageChunk(content="", tool_calls=tcs)
        chunk_final = AIMessageChunk(content="Done.")
        bot = _make_bot(_make_streaming_model([chunk1, chunk_final]), tools=tools)

        events = []
        async for ev in bot.astream_events("do three", mode="auto"):
            events.append(ev)

        executed = _tools_executed_from(events)
        assert [r["tool_name"] for r in executed] == ["tool_0", "tool_1", "tool_2"]

    async def test_error_isolation(self):
        """One failing tool does not prevent the others from succeeding."""

        async def boom(args):
            raise ValueError("kaboom")

        t_bad = _FakeTool("bad", boom)
        t_good = _FakeTool("good")
        tcs = [_tc("x", "bad"), _tc("y", "good")]
        chunk1 = AIMessageChunk(content="", tool_calls=tcs)
        chunk_final = AIMessageChunk(content="Done.")
        bot = _make_bot(_make_streaming_model([chunk1, chunk_final]), tools=[t_bad, t_good])

        events = []
        async for ev in bot.astream_events("do both", mode="auto"):
            events.append(ev)

        executed = {r["tool_name"]: r for r in _tools_executed_from(events)}
        assert executed["bad"]["status"] == "error"
        assert "kaboom" in executed["bad"]["output"]
        assert executed["good"]["status"] == "success"
        assert t_good.calls == 1

    async def test_ask_mode_safe_tool_executes_exactly_once(self):
        """Gate phase re-runs on interrupt resume, but execution happens once.

        Batch = [safe (risk 0), risky (risk 1)] in ask mode. The risky tool
        interrupts; on resume LangGraph re-runs the node from the top. Because
        the gate phase has no side effects, the safe tool must NOT be executed
        during the re-run — only once, in the post-resume execution phase.
        """
        safe = _FakeTool("safe", risk_level=0)
        risky = _FakeTool("danger", risk_level=1)
        tcs = [_tc("s", "safe"), _tc("d", "danger")]
        chunk1 = AIMessageChunk(content="", tool_calls=tcs)
        chunk_final = AIMessageChunk(content="Done.")
        bot = _make_bot(_make_streaming_model([chunk1, chunk_final]), tools=[safe, risky])
        thread = "t-idempotent"

        interrupted = False
        async for stream_mode, payload in bot.astream_events("go", mode="ask", thread_id=thread):
            if stream_mode == "updates" and isinstance(payload, dict) and "__interrupt__" in payload:
                interrupted = True
        assert interrupted, "run did not pause on the risky-tool interrupt"

        # Safe tool must not have executed while the run was suspended.
        assert safe.calls == 0

        bot.set_resume_command({"approved": True})
        async for _ in bot.astream_events(None, mode="ask", thread_id=thread):
            pass

        assert safe.calls == 1, "safe tool re-executed across interrupt resume"
        assert risky.calls == 1

    async def test_ask_mode_rejection_skips_tool(self):
        """Rejecting a risky tool marks it skipped and never executes it."""
        risky = _FakeTool("danger", risk_level=1)
        chunk1 = AIMessageChunk(content="", tool_calls=[_tc("d", "danger")])
        chunk_final = AIMessageChunk(content="Done.")
        bot = _make_bot(_make_streaming_model([chunk1, chunk_final]), tools=[risky])
        thread = "t-reject"

        async for _ in bot.astream_events("go", mode="ask", thread_id=thread):
            pass

        bot.set_resume_command({"approved": False})
        events = []
        async for ev in bot.astream_events(None, mode="ask", thread_id=thread):
            events.append(ev)

        executed = {r["tool_name"]: r for r in _tools_executed_from(events)}
        assert executed["danger"]["status"] == "skipped"
        assert risky.calls == 0

    async def test_plan_signal_calls_are_skipped(self):
        """set_plan signals in a batch never appear in tools_executed."""
        from sonika_ai_toolkit.tools.plan_tools import SET_PLAN_TOOL_NAME

        real = _FakeTool("real")
        tcs = [
            _tc("p", SET_PLAN_TOOL_NAME, {"steps": ["step one", "step two"]}),
            _tc("r", "real"),
        ]
        chunk1 = AIMessageChunk(content="", tool_calls=tcs)
        chunk_final = AIMessageChunk(content="Done.")
        bot = _make_bot(
            _make_streaming_model([chunk1, chunk_final]),
            tools=[real],
            enable_planning=True,
        )

        events = []
        async for ev in bot.astream_events("plan and act", mode="auto"):
            events.append(ev)

        executed = _tools_executed_from(events)
        names = [r["tool_name"] for r in executed]
        assert names == ["real"]
        assert SET_PLAN_TOOL_NAME not in names
        assert real.calls == 1
