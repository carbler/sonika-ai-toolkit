"""Behavioral tests for structured plan emission in OrchestratorBot.

Uses mocked LLM — no API keys needed. Mirrors the fake_astream pattern of
test_graph.py.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, AIMessageChunk

from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot
from sonika_ai_toolkit.skills import Skill


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


def _make_tool(name="test_tool"):
    tool = MagicMock()
    tool.name = name
    tool.description = "A test tool"
    tool.args_schema = None
    tool.ainvoke = AsyncMock(return_value="tool output")
    tool.risk_level = 0
    tool.risk_hint = 0
    return tool


def _planning_model():
    """Mock model scripting: set_plan → step 1 running + tool → step 1 done → final."""
    mock_model = MagicMock()

    chunks = [
        AIMessageChunk(
            content="",
            tool_calls=[{
                "id": "s1", "name": "set_plan",
                "args": {"steps": ["Buscar datos", "Generar reporte"]},
            }],
        ),
        AIMessageChunk(
            content="Empezando paso 1...",
            tool_calls=[
                {"id": "s2", "name": "update_step", "args": {"step": 1, "status": "running"}},
                {"id": "t1", "name": "test_tool", "args": {"x": 1}},
            ],
        ),
        AIMessageChunk(
            content="",
            tool_calls=[{
                "id": "s3", "name": "update_step",
                "args": {"step": 1, "status": "done"},
            }],
        ),
        AIMessageChunk(content="Todo listo."),
    ]
    call_count = 0

    async def fake_astream(messages):
        nonlocal call_count
        idx = min(call_count, len(chunks) - 1)
        call_count += 1
        yield chunks[idx]

    mock_model.bind_tools.return_value = mock_model
    mock_model.astream = fake_astream
    mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Todo listo."))
    return mock_model


@pytest.mark.asyncio
class TestPlanEmission:

    async def _collect(self, bot, goal="Haz dos cosas"):
        """Return the ordered list of (node_name, update) from the updates stream."""
        node_updates = []
        async for stream_mode, payload in bot.astream_events(goal, mode="auto"):
            if stream_mode == "updates" and isinstance(payload, dict):
                for node_name, update in payload.items():
                    node_updates.append((node_name, update))
        return node_updates

    async def test_plan_snapshot_emitted(self):
        bot = _make_bot(_planning_model(), tools=[_make_tool()], enable_planning=True)
        node_updates = await self._collect(bot)

        plans = [u.get("plan") for n, u in node_updates if n == "plan" and u and u.get("plan")]
        assert plans, "no plan-node update carried a plan snapshot"
        first_plan = plans[0]
        assert [s["description"] for s in first_plan] == ["Buscar datos", "Generar reporte"]
        assert all(s["status"] == "pending" for s in first_plan)

    async def test_step_events_ordered_around_tools(self):
        bot = _make_bot(_planning_model(), tools=[_make_tool()], enable_planning=True)
        node_updates = await self._collect(bot)

        running_idx = done_idx = tools_idx = None
        for i, (node, update) in enumerate(node_updates):
            if node == "plan" and update:
                for ev in update.get("step_events", []) or []:
                    if ev["status"] == "running" and running_idx is None:
                        running_idx = i
                    if ev["status"] == "done" and done_idx is None:
                        done_idx = i
            # Only tools updates with real executions count (signal-only
            # batches return an empty tools_executed list).
            if node == "tools" and tools_idx is None and update and update.get("tools_executed"):
                tools_idx = i

        assert running_idx is not None, "no running step event"
        assert tools_idx is not None, "no tools update"
        assert done_idx is not None, "no done step event"
        assert running_idx < tools_idx < done_idx

    async def test_final_report_still_set(self):
        bot = _make_bot(_planning_model(), tools=[_make_tool()], enable_planning=True)
        node_updates = await self._collect(bot)
        finals = [u["final_report"] for n, u in node_updates
                  if n == "agent" and u and "final_report" in u]
        assert finals and finals[-1] == "Todo listo."

    async def test_signal_calls_not_recorded_as_executed(self):
        """set_plan/update_step get acknowledgment ToolMessages but are never
        recorded in tools_executed — they are not real actions."""
        bot = _make_bot(_planning_model(), tools=[_make_tool()], enable_planning=True)
        node_updates = await self._collect(bot)
        executed = []
        for node, update in node_updates:
            if node == "tools" and update:
                executed.extend(t["tool_name"] for t in update.get("tools_executed", []))
        assert executed == ["test_tool"]

    async def test_arun_populates_botresponse_plan(self):
        bot = _make_bot(_planning_model(), tools=[_make_tool()], enable_planning=True)
        result = await bot.arun("Haz dos cosas")
        assert result.content == "Todo listo."
        assert [s["description"] for s in result.plan] == ["Buscar datos", "Generar reporte"]
        statuses = {s["step"]: s["status"] for s in result.plan}
        assert statuses[1] == "done"
        assert statuses[2] == "pending"


@pytest.mark.asyncio
class TestPlanningDisabled:

    async def test_no_plan_tools_registered_by_default(self):
        mock_model = MagicMock()

        async def fake_astream(messages):
            yield AIMessageChunk(content="Hecho.")

        mock_model.bind_tools.return_value = mock_model
        mock_model.astream = fake_astream
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="Hecho."))

        bot = _make_bot(mock_model)
        assert bot.registry.get("set_plan") is None
        assert bot.registry.get("update_step") is None

        agent_updates = []
        async for stream_mode, payload in bot.astream_events("Hola", mode="auto"):
            if stream_mode == "updates" and isinstance(payload, dict) and "agent" in payload:
                agent_updates.append(payload["agent"])

        for update in agent_updates:
            if update:
                assert "plan" not in update
                assert "step_events" not in update
                assert "plan_continue" not in update

        result = await bot.arun("Hola")
        assert result.plan == []


class TestOrchestratorSkills:

    def test_skills_on_demand_by_default(self):
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        skill_tool = _make_tool("facturar")
        skill = Skill(
            name="facturacion",
            instructions="Sabes generar facturas legales.",
            tools=[skill_tool],
        )
        bot = _make_bot(mock_model, skills=[skill])

        # Skill-owned tools always merge; default is on-demand: index only,
        # plus a load_skill tool registered for on-demand body loading.
        assert bot.registry.get("facturar") is skill_tool
        assert "facturacion" in bot._skills_prompt
        assert "Sabes generar facturas legales." not in bot._skills_prompt
        assert bot.registry.get("load_skill") is not None

    def test_skills_eager_injects_full_body(self):
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model

        skill = Skill(
            name="facturacion",
            instructions="Sabes generar facturas legales.",
        )
        bot = _make_bot(mock_model, skills=[skill], skills_eager=True)

        assert "Sabes generar facturas legales." in bot._skills_prompt
        assert bot.registry.get("load_skill") is None

    def test_no_skills_is_noop(self):
        mock_model = MagicMock()
        mock_model.bind_tools.return_value = mock_model
        bot = _make_bot(mock_model)
        assert bot.skills == []
        assert bot._skills_prompt == ""
