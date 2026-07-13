"""Unit tests for orchestrator risk-gate helpers (orchestrator.risk)."""

from sonika_ai_toolkit.agents.orchestrator.risk import (
    RISK_DESCRIPTIONS,
    format_approval_prompt,
    should_auto_approve,
)


class TestShouldAutoApprove:
    def test_below_threshold_approves(self):
        assert should_auto_approve(risk_level=0, risk_threshold=1) is True

    def test_at_threshold_approves(self):
        assert should_auto_approve(risk_level=1, risk_threshold=1) is True

    def test_above_threshold_requires_approval(self):
        assert should_auto_approve(risk_level=2, risk_threshold=1) is False


class TestFormatApprovalPrompt:
    def test_includes_step_details_and_risk_description(self):
        step = {
            "id": 3,
            "description": "Delete the temp file",
            "tool_name": "delete_file",
            "params": {"path": "/tmp/x"},
            "risk_level": 2,
        }
        prompt = format_approval_prompt(step)
        assert "Delete the temp file" in prompt
        assert "delete_file" in prompt
        assert "/tmp/x" in prompt
        assert RISK_DESCRIPTIONS[2] in prompt

    def test_unknown_risk_level_is_labelled(self):
        prompt = format_approval_prompt({"risk_level": 99})
        assert "Unknown risk level." in prompt

    def test_missing_fields_use_placeholders(self):
        prompt = format_approval_prompt({})
        assert "Step #?" in prompt
