"""Behavioral tests for custom node injection in OrchestratorBot.

Uses mocked LLM — no API keys needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk

from sonika_ai_toolkit.agents.extensions import CustomNode, validate_custom_nodes
from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot


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


async def _collect_nodes(bot, goal="Hola"):
    """Return the ordered node names seen in the updates stream."""
    seen = []
    async for stream_mode, payload in bot.astream_events(goal, mode="auto"):
        if stream_mode == "updates" and isinstance(payload, dict):
            seen.extend(payload.keys())
    return seen


class TestValidation:
    def test_reserved_name_rejected(self):
        with pytest.raises(ValueError, match="collides"):
            validate_custom_nodes(
                [CustomNode(name="agent", node=lambda s: {})], reserved_names={"agent", "tools"}
            )

    def test_duplicate_name_rejected(self):
        nodes = [
            CustomNode(name="x", node=lambda s: {}),
            CustomNode(name="x", node=lambda s: {}),
        ]
        with pytest.raises(ValueError, match="Duplicate"):
            validate_custom_nodes(nodes, reserved_names=set())

    def test_invalid_position_rejected(self):
        with pytest.raises(ValueError, match="position"):
            validate_custom_nodes(
                [CustomNode(name="x", node=lambda s: {}, position="middle")],
                reserved_names=set(),
            )

    def test_non_callable_rejected(self):
        with pytest.raises(ValueError, match="callable"):
            validate_custom_nodes(
                [CustomNode(name="x", node="not callable")], reserved_names=set()
            )

    def test_constructor_validates(self):
        with pytest.raises(ValueError):
            _make_bot(
                _final_only_model(),
                custom_nodes=[CustomNode(name="tools", node=lambda s: {})],
            )


@pytest.mark.asyncio
class TestWiring:

    async def test_default_graph_has_only_builtin_nodes(self):
        bot = _make_bot(_final_only_model())
        seen = await _collect_nodes(bot)
        assert set(seen) <= {"agent", "tools"}

    async def test_start_node_runs_before_agent(self):
        def audit(state):
            return {"session_log": [f"audit: {state.get('goal', '')}"]}

        bot = _make_bot(
            _final_only_model(),
            custom_nodes=[CustomNode(name="audit", node=audit, position="start")],
        )
        seen = await _collect_nodes(bot, goal="mi meta")
        assert "audit" in seen
        assert seen.index("audit") < seen.index("agent")

    async def test_end_node_runs_after_final_agent_turn(self):
        async def notify(state):
            return {"session_log": ["notified"]}

        bot = _make_bot(
            _final_only_model(),
            custom_nodes=[CustomNode(name="notify", node=notify, position="end")],
        )
        seen = await _collect_nodes(bot)
        assert "notify" in seen
        assert seen.index("notify") > seen.index("agent")

    async def test_after_tools_node_runs_between_tools_and_agent(self):
        def review(state):
            return {"session_log": ["reviewed"]}

        bot = _make_bot(
            _tool_then_final_model(),
            tools=[_make_tool()],
            custom_nodes=[CustomNode(name="review", node=review, position="after_tools")],
        )
        seen = await _collect_nodes(bot, goal="usa la tool")
        assert "review" in seen
        assert seen.index("tools") < seen.index("review")
        # The agent runs again after the review node.
        assert any(i > seen.index("review") for i, n in enumerate(seen) if n == "agent")

    async def test_multiple_nodes_same_position_chain_in_order(self):
        order = []

        def first(state):
            order.append("first")
            return {}

        def second(state):
            order.append("second")
            return {}

        bot = _make_bot(
            _final_only_model(),
            custom_nodes=[
                CustomNode(name="first", node=first, position="start"),
                CustomNode(name="second", node=second, position="start"),
            ],
        )
        await _collect_nodes(bot)
        assert order == ["first", "second"]

    async def test_custom_node_state_updates_reach_final_state(self):
        def audit(state):
            return {"session_log": ["from custom node"]}

        bot = _make_bot(
            _final_only_model(),
            custom_nodes=[CustomNode(name="audit", node=audit, position="start")],
        )
        result = await bot.arun("Hola")
        assert "from custom node" in result.logs
