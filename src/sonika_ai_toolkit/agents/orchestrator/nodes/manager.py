"""manager node — routes between chat and orchestration."""

import json
import logging
import re
from typing import Any, Callable, Dict, Optional

from sonika_ai_toolkit.agents.orchestrator.prompts import MANAGER_PROMPT, PROMPT_A
from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState
from sonika_ai_toolkit.agents.orchestrator.utils import ainvoke_with_thinking, get_text


class ManagerNode:
    """
    Acts as the conversation manager.
    Decides if the goal needs orchestration (planning) or just a direct chat.
    Emits an explanation or direct response.
    """

    def __init__(
        self,
        fast_model,
        on_thinking: Optional[Callable[[str], None]] = None,
        on_message: Optional[Callable[[str], None]] = None,
        logger=None,
    ):
        self.fast_model = fast_model
        self.on_thinking = on_thinking
        self.on_message = on_message
        self.logger = logger or logging.getLogger(__name__)

    async def __call__(self, state: OrchestratorState) -> Dict[str, Any]:
        goal = state.get("goal", "")
        
        prompt = MANAGER_PROMPT.format(
            prompt_a=PROMPT_A,
            goal=goal,
        )

        try:
            response = await ainvoke_with_thinking(
                self.fast_model.model, prompt, self.on_thinking
            )
            raw = get_text(response.content)
            data = self._parse_json(raw)
            
            action = data.get("action", "plan")
            
            if action == "chat":
                content = data.get("content", "I'm here to help.")
                if self.on_message:
                    self.on_message(content)
                return {
                    "final_report": content,
                    "_goal_complete": True,
                    "session_log": ["[manager] Responded directly via chat."],
                }
            else:
                explanation = data.get("explanation", "I'll create a plan to help you with that.")
                if self.on_message:
                    self.on_message(explanation)
                return {
                    "session_log": [f"[manager] Routing to planner. Explanation: {explanation}"],
                }

        except Exception as e:
            self.logger.error(f"[manager] Failed: {e}")
            return {
                "session_log": [f"[manager] Error: {e}. Falling back to planner."],
            }

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            # Try to find JSON in code blocks or just bare
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            return json.loads(text)
        except Exception:
            return {"action": "plan", "explanation": "I'll start planning now."}
