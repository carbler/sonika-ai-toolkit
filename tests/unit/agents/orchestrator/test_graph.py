"""Behavioral tests for partial response emission in OrchestratorBot.

Uses mocked LLM — no API keys needed.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from langchain_core.messages import AIMessage, AIMessageChunk

from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot, _extract_text_content


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bot(mock_model, tools=None):
    """Build an OrchestratorBot with a mocked ILanguageModel."""
    lm = MagicMock()
    lm.model = mock_model
    lm.supports_thinking = False
    return OrchestratorBot(
        strong_model=lm,
        fast_model=lm,
        instructions="You are a test bot.",
        tools=tools or [],
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
