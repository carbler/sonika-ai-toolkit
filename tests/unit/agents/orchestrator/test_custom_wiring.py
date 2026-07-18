"""Custom wiring overrides (CustomEdge / CustomRouter) for OrchestratorBot.

Covers:
  - validation: unknown sources/targets, opt-in nodes, unwired manual nodes,
    duplicate routers, __start__ → __end__
  - CustomEdge: replacing a built-in fixed edge, overriding the entry point
  - CustomRouter: full override, delegation to the default route
    (None / DEFAULT_ROUTE), early "__end__", interplay with enable_planning
  - topology reflects the overridden wiring

Uses mocked LLM — no API keys needed.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk

from sonika_ai_toolkit.agents.extensions import (
    DEFAULT_ROUTE,
    CustomEdge,
    CustomNode,
    CustomRouter,
)
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


async def _node_sequence(bot, goal="Hola"):
    seq = []
    async for stream_mode, payload in bot.astream_events(goal, mode="auto"):
        if stream_mode == "graph" and payload.get("type") == "node_invoked":
            seq.append(payload["node"])
    return seq


class TestWiringValidation:
    def test_unknown_edge_source_rejected(self):
        with pytest.raises(ValueError, match="not a node"):
            _make_bot(_final_only_model(),
                      custom_edges=[CustomEdge(source="ghost", target="agent")])

    def test_unknown_edge_target_rejected(self):
        with pytest.raises(ValueError, match="not a node"):
            _make_bot(_final_only_model(),
                      custom_edges=[CustomEdge(source="agent", target="ghost")])

    def test_optin_node_requires_flag(self):
        """'plan' is only a valid target when enable_planning=True."""
        with pytest.raises(ValueError, match="enable_planning"):
            _make_bot(_final_only_model(),
                      custom_edges=[CustomEdge(source="tools", target="plan")])

    def test_start_to_end_rejected(self):
        with pytest.raises(ValueError, match="__start__"):
            _make_bot(_final_only_model(),
                      custom_edges=[CustomEdge(source="__start__", target="__end__")])

    def test_unwired_manual_node_rejected(self):
        with pytest.raises(ValueError, match="never reach"):
            _make_bot(_final_only_model(),
                      custom_nodes=[CustomNode(name="orphan", node=lambda s: {},
                                               position=None)])

    def test_duplicate_router_source_rejected(self):
        with pytest.raises(ValueError, match="Duplicate CustomRouter"):
            _make_bot(_final_only_model(), custom_routers=[
                CustomRouter(source="agent", router=lambda s: None),
                CustomRouter(source="agent", router=lambda s: None),
            ])

    def test_non_callable_router_rejected(self):
        with pytest.raises(ValueError, match="callable"):
            _make_bot(_final_only_model(),
                      custom_routers=[CustomRouter(source="agent", router="nope")])


@pytest.mark.asyncio
class TestCustomEdges:
    async def test_edge_inserts_node_between_tools_and_agent(self):
        """tools → validator → agent replaces the built-in tools → agent."""
        def validator(state):
            return {"session_log": ["validated"]}

        bot = _make_bot(
            _tool_then_final_model(), tools=[_make_tool()],
            custom_nodes=[CustomNode(name="validator", node=validator, position=None)],
            custom_edges=[
                CustomEdge(source="tools", target="validator"),
                CustomEdge(source="validator", target="agent"),
            ],
        )
        topo = bot.get_graph_topology()
        assert not any(e["source"] == "tools" and e["target"] == "agent"
                       for e in topo["edges"])
        assert any(e["source"] == "tools" and e["target"] == "validator"
                   for e in topo["edges"])

        seq = await _node_sequence(bot, "usa la tool")
        assert seq == ["agent", "tools", "validator", "agent"]

    async def test_edge_overrides_entry_point(self):
        def preflight(state):
            return {"session_log": ["preflight"]}

        bot = _make_bot(
            _final_only_model(),
            custom_nodes=[CustomNode(name="preflight", node=preflight, position=None)],
            custom_edges=[
                CustomEdge(source="__start__", target="preflight"),
                CustomEdge(source="preflight", target="agent"),
            ],
        )
        assert bot.get_graph_topology()["entry"] == "preflight"
        seq = await _node_sequence(bot)
        assert seq == ["preflight", "agent"]


@pytest.mark.asyncio
class TestCustomRouters:
    async def test_router_delegating_default_keeps_flow_identical(self):
        bot = _make_bot(
            _tool_then_final_model(), tools=[_make_tool()],
            custom_routers=[CustomRouter(source="agent", router=lambda s: None)],
        )
        seq = await _node_sequence(bot, "usa la tool")
        assert seq == ["agent", "tools", "agent"]

    async def test_default_route_sentinel_equivalent_to_none(self):
        bot = _make_bot(
            _tool_then_final_model(), tools=[_make_tool()],
            custom_routers=[
                CustomRouter(source="agent", router=lambda s: DEFAULT_ROUTE)],
        )
        seq = await _node_sequence(bot, "usa la tool")
        assert seq == ["agent", "tools", "agent"]

    async def test_router_diverts_final_turn_to_custom_node(self):
        """Delegate tool turns; send the final (no tool_calls) turn through
        a summarize node instead of ending at agent."""
        def summarize(state):
            return {"session_log": ["summarized"]}

        def router(state):
            last = state["messages"][-1]
            if getattr(last, "tool_calls", None):
                return None            # default: tools/plan/ask_user
            return "summarize"         # instead of END

        bot = _make_bot(
            _tool_then_final_model(), tools=[_make_tool()],
            custom_nodes=[CustomNode(name="summarize", node=summarize, position=None)],
            custom_edges=[CustomEdge(source="summarize", target="__end__")],
            custom_routers=[CustomRouter(
                source="agent", router=router,
                targets=["tools", "summarize", "__end__"],
            )],
        )
        seq = await _node_sequence(bot, "usa la tool")
        assert seq == ["agent", "tools", "agent", "summarize"]
        result = await bot.arun("usa la tool")
        assert "summarized" in result.logs

    async def test_router_can_end_run_directly(self):
        bot = _make_bot(
            _tool_then_final_model(), tools=[_make_tool()],
            custom_routers=[CustomRouter(source="agent", router=lambda s: "__end__")],
        )
        seq = await _node_sequence(bot, "usa la tool")
        assert seq == ["agent"]  # first turn had tool_calls, but router ended it

    async def test_delegation_preserves_plan_routing(self):
        """A delegating router on agent must not break the plan node flow."""
        mock_model = MagicMock()
        chunks = [
            AIMessageChunk(content="", tool_calls=[
                {"id": "s1", "name": "set_plan", "args": {"steps": ["a"]}}]),
            AIMessageChunk(content="Listo."),
        ]
        n = 0

        async def fake_astream(messages):
            nonlocal n
            idx = min(n, len(chunks) - 1)
            n += 1
            yield chunks[idx]

        mock_model.bind_tools.return_value = mock_model
        mock_model.astream = fake_astream
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Listo."))

        bot = _make_bot(
            mock_model, enable_planning=True,
            custom_routers=[CustomRouter(source="agent", router=lambda s: None)],
        )
        seq = await _node_sequence(bot, "planea")
        assert seq == ["agent", "plan", "agent"]

    async def test_router_targets_shape_topology(self):
        def summarize(state):
            return {}

        bot = _make_bot(
            _final_only_model(),
            custom_nodes=[CustomNode(name="summarize", node=summarize, position=None)],
            custom_edges=[CustomEdge(source="summarize", target="__end__")],
            custom_routers=[CustomRouter(
                source="agent", router=lambda s: "summarize",
                targets=["tools", "summarize", "__end__"],
            )],
        )
        topo = bot.get_graph_topology()
        agent_targets = {e["target"] for e in topo["edges"] if e["source"] == "agent"}
        assert {"tools", "summarize", "__end__"} <= agent_targets
