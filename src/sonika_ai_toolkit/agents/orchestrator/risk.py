"""Risk gate helpers for OrchestratorBot."""

from typing import Dict, Any

RISK_DESCRIPTIONS = {
    0: "Safe — read-only, no side effects (e.g., read_file, list_dir, search_web).",
    1: "Low risk — writes or modifies local state (e.g., write_file, run_bash, call_api).",
    2: "Medium risk — destructive or irreversible local changes (e.g., delete_file).",
    3: "High risk — external system modifications, financial transactions, mass operations.",
}


def should_auto_approve(risk_level: int, risk_threshold: int) -> bool:
    """Return True if the step can proceed without human approval."""
    return risk_level <= risk_threshold


def format_approval_prompt(step: Dict[str, Any]) -> str:
    """Format a human-readable approval request for a risky step."""
    risk_level = step.get("risk_level", 0)
    risk_desc = RISK_DESCRIPTIONS.get(risk_level, "Unknown risk level.")
    return (
        f"⚠️  Human approval required\n"
        f"Step #{step.get('id', '?')}: {step.get('description', '')}\n"
        f"Tool: {step.get('tool_name', '?')}\n"
        f"Params: {step.get('params', {})}\n"
        f"Risk level: {risk_level} — {risk_desc}\n"
        f"Approve? [True/False]"
    )
