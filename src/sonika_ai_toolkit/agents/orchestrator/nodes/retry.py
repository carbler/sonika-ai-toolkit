"""retry node — fast_model decides recovery strategy."""

import hashlib
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from sonika_ai_toolkit.agents.orchestrator.prompts import RETRY_PROMPT, PROMPT_A
from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState, update_step
from sonika_ai_toolkit.agents.orchestrator.utils import ainvoke_with_thinking


def _params_hash(params: Dict) -> str:
    try:
        serialized = json.dumps(params, sort_keys=True)
    except Exception:
        serialized = str(params)
    return hashlib.md5(serialized.encode()).hexdigest()[:8]


def _retry_history_str(history: List[Dict]) -> str:
    if not history:
        return "None"
    return "\n".join(
        f"  - attempt {i + 1}: strategy={h.get('strategy')} tool={h.get('tool_name')} params_hash={h.get('params_hash')}"
        for i, h in enumerate(history)
    )


def _parse_json(text: str) -> Dict[str, Any]:
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


def _accumulate(current: str, new: Optional[str]) -> str:
    if not new:
        return current
    return (current + "\n\n" + new).strip() if current else new


class RetryNode:
    """Uses fast_model to decide the recovery strategy after a step failure."""

    def __init__(self, fast_model, tool_registry, on_thinking: Optional[Callable[[str], None]] = None, logger=None):
        self.fast_model = fast_model
        self.registry = tool_registry
        self.on_thinking = on_thinking
        self.logger = logger or logging.getLogger(__name__)

    async def __call__(self, state: OrchestratorState) -> Dict[str, Any]:
        plan = list(state.get("plan", []))
        index = state.get("current_step_index", 0)
        step = plan[index] if index < len(plan) else {}
        retry_count = state.get("retry_count", 0)
        max_retries = state.get("max_retries", 3)
        retry_history: List[Dict] = list(state.get("retry_history", []))
        last_error = state.get("last_error", "Unknown error")
        goal = state.get("goal", "")
        log_lines = []

        # Hard limit
        if retry_count >= max_retries:
            log_lines.append(
                f"[retry] Max retries ({max_retries}) reached for step {index + 1} → escalate."
            )
            return {"retry_strategy": "escalate", "session_log": log_lines}

        prompt = RETRY_PROMPT.format(
            prompt_a=PROMPT_A,
            goal=goal,
            step_description=step.get("description", ""),
            error=last_error,
            tool_descriptions=self.registry.get_tool_descriptions(),
            retry_history=_retry_history_str(retry_history),
        )

        try:
            response = await ainvoke_with_thinking(self.fast_model.model, prompt, self.on_thinking)
            raw = response.content   # clean string
            decision = _parse_json(raw)
            thinking = response.additional_kwargs.get("_thinking")
        except Exception as e:
            self.logger.warning(f"[retry] LLM failed: {e} → escalate")
            return {
                "retry_strategy": "escalate",
                "session_log": [f"[retry] LLM error: {e} → escalate"],
            }

        strategy = decision.get("strategy", "escalate")
        new_tool = decision.get("tool_name", step.get("tool_name", ""))
        new_params = decision.get("params", step.get("params", {}))
        reasoning = decision.get("reasoning", "")

        # Anti-loop check
        ph = _params_hash(new_params)
        loop_key = (strategy, new_tool, ph)
        for h in retry_history:
            if (h.get("strategy"), h.get("tool_name"), h.get("params_hash")) == loop_key:
                log_lines.append("[retry] Anti-loop: same strategy+tool+params → escalate.")
                return {"retry_strategy": "escalate", "session_log": log_lines}

        if strategy == "escalate":
            log_lines.append(f"[retry] LLM chose escalate — {reasoning}")
            return {"retry_strategy": "escalate", "session_log": log_lines}

        new_plan = update_step(plan, index, tool_name=new_tool, params=new_params, status="pending")
        retry_history.append({"strategy": strategy, "tool_name": new_tool, "params_hash": ph})

        log_lines.append(
            f"[retry] Strategy={strategy} tool={new_tool} retry={retry_count + 1}/{max_retries} — {reasoning}"
        )

        accumulated = _accumulate(state.get("thinking", ""), thinking)
        extra = {"thinking": accumulated} if accumulated else {}

        return {
            "plan": new_plan,
            "retry_count": retry_count + 1,
            "retry_strategy": strategy,
            "retry_history": retry_history,
            "last_error": None,
            "session_log": log_lines,
            **extra,
        }
