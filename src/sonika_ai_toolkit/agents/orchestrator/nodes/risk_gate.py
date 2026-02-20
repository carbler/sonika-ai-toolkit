"""risk_gate node — deterministic risk evaluation."""

import logging
from typing import Any, Dict

from sonika_ai_toolkit.agents.orchestrator.risk import should_auto_approve
from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState


class RiskGateNode:
    """
    Compares the current step's risk_level against risk_threshold.
    Sets awaiting_approval in state — the graph edges decide the route.
    No LLM calls.
    """

    def __init__(self, risk_threshold: int = 1, logger=None):
        self.risk_threshold = risk_threshold
        self.logger = logger or logging.getLogger(__name__)

    def __call__(self, state: OrchestratorState) -> Dict[str, Any]:
        plan = state.get("plan", [])
        index = state.get("current_step_index", 0)
        step = plan[index] if index < len(plan) else {}
        risk_level = step.get("risk_level", 0)

        auto = should_auto_approve(risk_level, self.risk_threshold)
        log_msg = (
            f"[risk_gate] Step {index + 1} risk={risk_level} "
            f"threshold={self.risk_threshold} → {'auto-approve' if auto else 'needs approval'}"
        )
        self.logger.info(log_msg)

        return {
            "awaiting_approval": not auto,
            "user_approved": None,
            "session_log": [log_msg],
        }
