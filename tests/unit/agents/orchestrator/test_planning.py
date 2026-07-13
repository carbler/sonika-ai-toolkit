"""Unit tests for the pure plan-tracking helpers (orchestrator.planning)."""

from sonika_ai_toolkit.agents.orchestrator.planning import (
    PLANNING_PROTOCOL_PROMPT,
    apply_update_step,
    normalize_plan,
    render_plan_status,
    split_plan_signal_calls,
)


class TestNormalizePlan:
    def test_builds_pending_steps(self):
        plan = normalize_plan(["Buscar datos", "Generar reporte"])
        assert plan == [
            {"step": 1, "description": "Buscar datos", "status": "pending"},
            {"step": 2, "description": "Generar reporte", "status": "pending"},
        ]

    def test_empty_and_none(self):
        assert normalize_plan([]) == []
        assert normalize_plan(None) == []

    def test_coerces_descriptions_to_str(self):
        assert normalize_plan([42])[0]["description"] == "42"


class TestApplyUpdateStep:
    PLAN = [
        {"step": 1, "description": "a", "status": "pending"},
        {"step": 2, "description": "b", "status": "pending"},
    ]

    def test_updates_matching_step(self):
        updated = apply_update_step(self.PLAN, 1, "running")
        assert updated[0]["status"] == "running"
        assert updated[1]["status"] == "pending"
        # Original untouched (copy semantics).
        assert self.PLAN[0]["status"] == "pending"

    def test_unknown_step_is_noop(self):
        assert apply_update_step(self.PLAN, 99, "done") == self.PLAN

    def test_invalid_status_is_noop(self):
        assert apply_update_step(self.PLAN, 1, "volando") == self.PLAN


class TestRenderPlanStatus:
    def test_empty_plan_renders_nothing(self):
        assert render_plan_status([]) == ""

    def test_renders_steps_with_status(self):
        text = render_plan_status(
            [{"step": 1, "description": "Buscar", "status": "running"}]
        )
        assert "1. [running] Buscar" in text
        assert "set_plan" in text  # reminder not to re-register


class TestSplitPlanSignalCalls:
    def test_partition(self):
        calls = [
            {"id": "1", "name": "set_plan", "args": {"steps": ["a"]}},
            {"id": "2", "name": "search_web", "args": {}},
            {"id": "3", "name": "update_step", "args": {"step": 1, "status": "running"}},
        ]
        signal, real = split_plan_signal_calls(calls)
        assert [c["name"] for c in signal] == ["set_plan", "update_step"]
        assert [c["name"] for c in real] == ["search_web"]

    def test_empty(self):
        assert split_plan_signal_calls([]) == ([], [])
        assert split_plan_signal_calls(None) == ([], [])


class TestProtocolPrompt:
    def test_mentions_both_tools(self):
        assert "set_plan" in PLANNING_PROTOCOL_PROMPT
        assert "update_step" in PLANNING_PROTOCOL_PROMPT
