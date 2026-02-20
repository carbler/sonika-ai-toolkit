"""save_memory node — fast_model summarises the session and persists it."""

import logging
from typing import Any, Dict, List

from sonika_ai_toolkit.agents.orchestrator.prompts import SAVE_MEMORY_PROMPT, PROMPT_A
from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState
from sonika_ai_toolkit.agents.orchestrator.utils import ainvoke_with_thinking


def _plan_summary(plan: List[Dict]) -> str:
    lines = []
    for s in plan:
        status = s.get("status", "?").upper()
        lines.append(f"  [{status}] Step {s.get('id', '?')}: {s.get('description', '')}")
    return "\n".join(lines)


class SaveMemoryNode:
    """Generates a 2-bullet summary with fast_model and appends it to MEMORY.md."""

    def __init__(
        self,
        fast_model,
        memory_manager,
        logger=None,
        prompt_template: str = SAVE_MEMORY_PROMPT,
        core_prompt: str = PROMPT_A,
    ):
        self.fast_model = fast_model
        self.memory_manager = memory_manager
        self.logger = logger or logging.getLogger(__name__)
        self.prompt_template = prompt_template
        self.core_prompt = core_prompt

    async def __call__(self, state: OrchestratorState) -> Dict[str, Any]:
        plan = state.get("plan", [])
        goal = state.get("goal", "")
        session_id = state.get("session_id", "unknown")
        session_log = state.get("session_log", [])

        prompt = self.prompt_template.format(
            prompt_a=self.core_prompt,
            goal=goal,
            plan_summary=_plan_summary(plan),
        )

        try:
            response = await ainvoke_with_thinking(self.fast_model.model, prompt)
            summary = response.content if hasattr(response, "content") else str(response)
            summary = summary.strip()
        except Exception as e:
            summary = f"- Session {session_id} completed.\n- Goal: {goal[:100]}"
            self.logger.warning(f"[save_memory] LLM failed: {e}; using fallback summary.")

        try:
            self.memory_manager.update_memory(summary)
        except Exception as e:
            self.logger.warning(f"[save_memory] Could not update MEMORY.md: {e}")

        try:
            self.memory_manager.save_session_log(session_id, session_log)
        except Exception as e:
            self.logger.warning(f"[save_memory] Could not save session log: {e}")

        return {
            "session_log": ["[save_memory] Memory updated."],
        }
