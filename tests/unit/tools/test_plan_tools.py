"""Unit tests for the set_plan / update_step signal tools."""

import pytest
from pydantic import ValidationError

from sonika_ai_toolkit.tools.plan_tools import (
    PLAN_SIGNAL_TOOL_NAMES,
    SET_PLAN_TOOL_NAME,
    UPDATE_STEP_TOOL_NAME,
    SetPlanTool,
    UpdateStepTool,
)


class TestNames:
    def test_tool_names(self):
        assert SetPlanTool().name == SET_PLAN_TOOL_NAME == "set_plan"
        assert UpdateStepTool().name == UPDATE_STEP_TOOL_NAME == "update_step"
        assert PLAN_SIGNAL_TOOL_NAMES == {"set_plan", "update_step"}


class TestRiskLevel:
    def test_signal_tools_are_never_risky(self):
        assert SetPlanTool().risk_level == 0
        assert UpdateStepTool().risk_level == 0


class TestSchemas:
    def test_set_plan_requires_steps(self):
        schema = SetPlanTool().args_schema
        assert "steps" in schema.model_fields
        with pytest.raises(ValidationError):
            schema()

    def test_update_step_rejects_bad_status(self):
        schema = UpdateStepTool().args_schema
        assert schema(step=1, status="running").status == "running"
        with pytest.raises(ValidationError):
            schema(step=1, status="volando")


class TestRunFallback:
    def test_set_plan_run_is_safe(self):
        out = SetPlanTool()._run(steps=["a", "b"])
        assert "2" in out

    def test_update_step_run_is_safe(self):
        out = UpdateStepTool()._run(step=3, status="done")
        assert "3" in out and "done" in out
