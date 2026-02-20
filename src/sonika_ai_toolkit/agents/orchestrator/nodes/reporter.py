"""reporter node — fast_model generates the final report."""

import logging
from typing import Any, Dict, List

from sonika_ai_toolkit.agents.orchestrator.prompts import REPORTER_PROMPT, PROMPT_A
from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState
from sonika_ai_toolkit.agents.orchestrator.utils import ainvoke_with_thinking


def _plan_summary(plan: List[Dict]) -> str:
    lines = []
    for s in plan:
        status = s.get("status", "?").upper()
        sid = s.get("id", "?")
        desc = s.get("description", "")
        err = s.get("error")
        result = s.get("result")
        line = f"  [{status}] Step {sid}: {desc}"
        if err:
            line += f" — ERROR: {err}"
        elif result:
            line += f" — {result[:100]}"
        lines.append(line)
    return "\n".join(lines)


def _tool_outputs_summary(outputs: List[Dict]) -> str:
    if not outputs:
        return "(no tool outputs)"
    parts = []
    for o in outputs:
        parts.append(f"Step {o.get('step_id', '?')} [{o.get('tool', '?')}]: {o.get('output', '')[:300]}")
    return "\n".join(parts)


class ReporterNode:
    """Generates the final_report using fast_model."""

    def __init__(
        self,
        fast_model,
        on_thinking=None,
        logger=None,
        prompt_template: str = REPORTER_PROMPT,
        core_prompt: str = PROMPT_A,
    ):
        self.fast_model = fast_model
        self.on_thinking = on_thinking
        self.logger = logger or logging.getLogger(__name__)
        self.prompt_template = prompt_template
        self.core_prompt = core_prompt

    async def __call__(self, state: OrchestratorState) -> Dict[str, Any]:
        plan = state.get("plan", [])
        tool_outputs = state.get("tool_outputs", [])
        goal = state.get("goal", "")

        prompt = self.prompt_template.format(
            prompt_a=self.core_prompt,
            goal=goal,
            plan_summary=_plan_summary(plan),
            tool_outputs=_tool_outputs_summary(tool_outputs),
        )

        try:
            response = await ainvoke_with_thinking(self.fast_model.model, prompt, self.on_thinking)
            report = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            report = f"Error generating report: {e}"
            self.logger.error(f"[reporter] {e}")

        self.logger.info(f"[reporter] Report generated ({len(report)} chars).")

        return {
            "final_report": report,
            "session_log": ["[reporter] Final report generated."],
        }
