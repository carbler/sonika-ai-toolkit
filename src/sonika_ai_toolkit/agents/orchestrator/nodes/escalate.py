"""escalate node — marks the current step as definitively failed."""

import logging
from typing import Any, Dict

from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState, update_step


class EscalateNode:
    """
    Marks the current step as 'failed' and sets should_advance=True so the
    step_dispatcher moves on to the next step.  No LLM calls.
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def __call__(self, state: OrchestratorState) -> Dict[str, Any]:
        plan = list(state.get("plan", []))
        index = state.get("current_step_index", 0)
        last_error = state.get("last_error", "Escalated after retry exhaustion.")

        if index < len(plan):
            new_plan = update_step(plan, index, status="failed", error=last_error)
        else:
            new_plan = plan

        log_msg = f"[escalate] Step {index + 1} marked failed: {last_error}"
        self.logger.warning(log_msg)

        return {
            "plan": new_plan,
            "should_advance": True,
            "retry_count": 0,
            "retry_strategy": None,
            "retry_history": [],
            "last_error": None,
            "session_log": [log_msg],
        }
