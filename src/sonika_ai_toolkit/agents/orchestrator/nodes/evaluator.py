"""evaluator node — fast_model judges step success and goal completion."""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from sonika_ai_toolkit.agents.orchestrator.prompts import EVALUATOR_PROMPT, PROMPT_A
from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState, update_step
from sonika_ai_toolkit.agents.orchestrator.utils import ainvoke_with_thinking


def _plan_summary(plan: List[Dict]) -> str:
    lines = []
    for s in plan:
        lines.append(
            f"  [{s.get('status', '?').upper()}] Step {s.get('id', '?')}: {s.get('description', '')}"
        )
    return "\n".join(lines)


def _accumulate(current: str, new: Optional[str]) -> str:
    if not new:
        return current
    return (current + "\n\n" + new).strip() if current else new


class EvaluatorNode:
    """Uses fast_model to evaluate whether a step succeeded and if the goal is complete."""

    def __init__(self, fast_model, on_thinking: Optional[Callable[[str], None]] = None, logger=None):
        self.fast_model = fast_model
        self.on_thinking = on_thinking
        self.logger = logger or logging.getLogger(__name__)

    async def __call__(self, state: OrchestratorState) -> Dict[str, Any]:
        plan = list(state.get("plan", []))
        index = state.get("current_step_index", 0)
        step = plan[index] if index < len(plan) else {}
        goal = state.get("goal", "")
        last_result = state.get("last_result") or ""
        last_error = state.get("last_error") or ""
        tool_output = last_result or last_error or "(no output)"

        prompt = EVALUATOR_PROMPT.format(
            prompt_a=PROMPT_A,
            goal=goal,
            step_description=step.get("description", ""),
            tool_output=tool_output,
            plan_summary=_plan_summary(plan),
        )

        try:
            response = await ainvoke_with_thinking(self.fast_model.model, prompt, self.on_thinking)
            raw = response.content   # always clean string
            evaluation = _parse_json(raw)
            thinking = response.additional_kwargs.get("_thinking")
        except Exception as e:
            self.logger.warning(f"[evaluator] LLM failed: {e}; defaulting to step_success=True.")
            evaluation = {
                "step_success": True,
                "reason": f"Evaluation error: {e}",
                "goal_complete": False,
                "goal_complete_reason": "",
            }
            thinking = None

        step_success = evaluation.get("step_success", True)
        goal_complete = evaluation.get("goal_complete", False)
        reason = evaluation.get("reason", "")

        log_msg = (
            f"[evaluator] step_success={step_success} goal_complete={goal_complete} — {reason}"
        )
        self.logger.info(log_msg)

        accumulated = _accumulate(state.get("thinking", ""), thinking)
        extra = {"thinking": accumulated} if accumulated else {}

        if step_success:
            new_plan = update_step(plan, index, status="success", result=last_result)
            return {
                "plan": new_plan,
                "should_advance": True,
                "last_error": None,
                "session_log": [log_msg],
                "_goal_complete": goal_complete,
                **extra,
            }
        else:
            new_plan = update_step(plan, index, status="failed", error=reason)
            return {
                "plan": new_plan,
                "last_error": reason,
                "should_advance": False,
                "session_log": [log_msg],
                "_goal_complete": False,
                **extra,
            }


def _parse_json(text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object from text."""
    if not isinstance(text, str):
        text = str(text)
    for pattern in [r"```json\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                pass
    try:
        return json.loads(text.strip())
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass
    return {}
