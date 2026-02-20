"""step_dispatcher node — deterministic step advancement."""

import logging
from typing import Any, Callable, Dict, Optional

from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState, update_step


class StepDispatcherNode:
    """
    Advances the step index, skips non-pending steps, and validates that the
    required tool exists in the registry.  No LLM calls.
    """

    def __init__(
        self,
        tool_registry,
        on_step_start: Optional[Callable[[Dict], None]] = None,
        logger=None,
    ):
        self.registry = tool_registry
        self.on_step_start = on_step_start
        self.logger = logger or logging.getLogger(__name__)

    def __call__(self, state: OrchestratorState) -> Dict[str, Any]:
        plan = list(state.get("plan", []))
        index = state.get("current_step_index", 0)
        should_advance = state.get("should_advance", False)

        # Advance if requested
        if should_advance:
            index += 1

        # Skip steps that are already done (success/failed/skipped)
        while index < len(plan) and plan[index].get("status") not in ("pending", "running"):
            index += 1

        n = len(plan)
        log_lines = []

        if index >= n:
            # All steps processed → signal reporter
            log_lines.append("[step_dispatcher] All steps complete → reporter.")
            return {
                "current_step_index": index,
                "should_advance": False,
                "last_error": None,
                "retry_count": 0,
                "retry_history": [],
                "session_log": log_lines,
            }

        step = plan[index]
        tool_name = step.get("tool_name", "")
        description = step.get("description", "")

        log_lines.append(
            f"[STEP {index + 1}/{n}] {description} (tool: {tool_name})"
        )
        self.logger.info(log_lines[-1])

        # Validate tool exists
        if tool_name and tool_name != "noop" and not self.registry.has(tool_name):
            err = f"Tool not found: {tool_name}"
            log_lines.append(f"[step_dispatcher] {err}")
            return {
                "current_step_index": index,
                "should_advance": False,
                "last_error": err,
                "last_result": None,
                "retry_count": 0,
                "retry_history": [],
                "session_log": log_lines,
            }

        # Fire callback
        if self.on_step_start:
            try:
                self.on_step_start(step)
            except Exception:
                pass

        return {
            "current_step_index": index,
            "should_advance": False,
            "last_error": None,
            "retry_count": state.get("retry_count", 0),
            "retry_history": state.get("retry_history", []),
            "session_log": log_lines,
        }
