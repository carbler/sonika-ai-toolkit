"""Tests for passing an externally-managed conversation to OrchestratorBot.

Covers the `_build_history_messages` helper and the `history` param of
`arun`/`run`/`astream_events` (the model must actually receive the prior turns).
Uses a mocked LLM — no API keys needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot
from sonika_ai_toolkit.agents.orchestrator._graph_helpers import _build_history_messages


class _SonikaMsg:
    """Stand-in for the sonika Message dataclass (is_bot + content)."""

    def __init__(self, content, is_bot):
        self.content = content
        self.is_bot = is_bot


class TestBuildHistoryMessages:
    def test_empty_returns_empty(self):
        assert _build_history_messages(None) == []
        assert _build_history_messages([]) == []

    def test_dict_roles_map_to_message_types(self):
        msgs = _build_history_messages([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "be nice"},
        ])
        assert [type(m).__name__ for m in msgs] == [
            "HumanMessage", "AIMessage", "SystemMessage",
        ]
        assert msgs[0].content == "hi"
        assert msgs[1].content == "hello"

    def test_ai_bot_aliases(self):
        msgs = _build_history_messages([
            {"role": "ai", "content": "x"},
            {"role": "bot", "content": "y"},
        ])
        assert all(type(m).__name__ == "AIMessage" for m in msgs)

    def test_sonika_message_like(self):
        msgs = _build_history_messages([
            _SonikaMsg("q", is_bot=False),
            _SonikaMsg("a", is_bot=True),
        ])
        assert type(msgs[0]).__name__ == "HumanMessage"
        assert type(msgs[1]).__name__ == "AIMessage"

    def test_basemessage_passthrough(self):
        hm = HumanMessage(content="x")
        assert _build_history_messages([hm]) == [hm]


def _make_bot(mock_model):
    lm = MagicMock()
    lm.model = mock_model
    lm.supports_thinking = False
    return OrchestratorBot(
        strong_model=lm,
        fast_model=lm,
        instructions="You are a test bot.",
        tools=[],
        memory_path="/tmp/orch_history_test_mem",
    )


def _capturing_model():
    """A mocked model whose astream records the messages it was called with."""
    captured = {}

    async def fake_astream(messages):
        captured["messages"] = messages
        yield AIMessageChunk(content="ok")

    mock_model = MagicMock()
    mock_model.bind_tools.return_value = mock_model
    mock_model.astream = fake_astream
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
    return mock_model, captured


@pytest.mark.asyncio
async def test_arun_prepends_history_before_goal():
    mock_model, captured = _capturing_model()
    bot = _make_bot(mock_model)

    await bot.arun(
        "What is my name?",
        history=[
            {"role": "user", "content": "Hi, I'm Carlos."},
            {"role": "assistant", "content": "Hello Carlos!"},
        ],
    )

    contents = " ".join(str(getattr(m, "content", "")) for m in captured["messages"])
    # The model must actually see the prior conversation AND the current goal.
    assert "I'm Carlos" in contents
    assert "Hello Carlos" in contents
    assert "What is my name?" in contents


@pytest.mark.asyncio
async def test_arun_context_block_reaches_model():
    mock_model, captured = _capturing_model()
    bot = _make_bot(mock_model)

    await bot.arun("Continue", context="Background: the user is a VIP client.")

    contents = " ".join(str(getattr(m, "content", "")) for m in captured["messages"])
    assert "VIP client" in contents


@pytest.mark.asyncio
async def test_arun_without_history_still_works():
    mock_model, captured = _capturing_model()
    bot = _make_bot(mock_model)

    await bot.arun("Just the goal")

    contents = " ".join(str(getattr(m, "content", "")) for m in captured["messages"])
    assert "Just the goal" in contents
