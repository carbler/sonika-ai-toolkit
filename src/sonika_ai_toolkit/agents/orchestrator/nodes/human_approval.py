"""human_approval node — calls on_human_approval callback."""

import logging
from typing import Any, Callable, Dict, Optional

from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState, update_step


class HumanApprovalNode:
    """
    Calls the on_human_approval callback with the current step dict.
    If the callback returns False, or if no callback is configured, the step
    is marked 'skipped' and should_advance is set so step_dispatcher moves on.
    No LLM calls.
    """

    def __init__(
        self,
        on_human_approval: Optional[Callable[[Dict], bool]] = None,
        logger=None,
    ):
        self.on_human_approval = on_human_approval
        self.logger = logger or logging.getLogger(__name__)

    def __call__(self, state: OrchestratorState) -> Dict[str, Any]:
        plan = list(state.get("plan", []))
        index = state.get("current_step_index", 0)
        step = plan[index] if index < len(plan) else {}
        log_lines = []

        if self.on_human_approval is None:
            # Auto-reject: no callback configured.
            log_lines.append(
                f"[human_approval] No callback configured — skipping step {index + 1}."
            )
            new_plan = update_step(plan, index, status="skipped", error="No approval callback")
            return {
                "plan": new_plan,
                "user_approved": False,
                "awaiting_approval": False,
                "should_advance": True,
                "session_log": log_lines,
            }

        try:
            approved: bool = bool(self.on_human_approval(step))
        except Exception as e:
            self.logger.warning(f"[human_approval] Callback raised exception: {e}")
            approved = False

        if approved:
            log_lines.append(f"[human_approval] Step {index + 1} approved by user.")
            return {
                "user_approved": True,
                "awaiting_approval": False,
                "session_log": log_lines,
            }
        else:
            log_lines.append(f"[human_approval] Step {index + 1} rejected — skipping.")
            new_plan = update_step(plan, index, status="skipped", error="Rejected by user")
            return {
                "plan": new_plan,
                "user_approved": False,
                "awaiting_approval": False,
                "should_advance": True,
                "session_log": log_lines,
            }
