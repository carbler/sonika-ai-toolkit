"""planner node — strong_model → structured plan."""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from sonika_ai_toolkit.agents.orchestrator.prompts import PLANNER_PROMPT, PROMPT_A
from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState, make_step
from sonika_ai_toolkit.agents.orchestrator.utils import ainvoke_with_thinking, get_text


class _StepSchema(BaseModel):
    id: int
    description: str
    tool_name: str
    params: Dict[str, str] = {}   # str values — avoids OpenAI strict-mode rejection
    risk_level: int = 0


class _PlanSchema(BaseModel):
    steps: List[_StepSchema]


class PlannerNode:
    """Uses strong_model to generate a structured plan from the goal."""

    def __init__(
        self,
        strong_model,
        tool_registry,
        memory_manager,
        instructions: str,
        on_plan_generated: Optional[Callable[[List], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        logger=None,
        prompt_template: str = PLANNER_PROMPT,
        core_prompt: str = PROMPT_A,
    ):
        self.strong_model = strong_model
        self.registry = tool_registry
        self.memory_manager = memory_manager
        self.instructions = instructions
        self.on_plan_generated = on_plan_generated
        self.on_thinking = on_thinking
        self.logger = logger or logging.getLogger(__name__)
        self.prompt_template = prompt_template
        self.core_prompt = core_prompt

        # Try to set up structured output; fall back to raw parsing if unavailable.
        try:
            self._structured_model = self.strong_model.model.with_structured_output(
                _PlanSchema, method="function_calling"
            )
        except Exception:
            self._structured_model = None

    async def __call__(self, state: OrchestratorState) -> Dict[str, Any]:
        goal = state.get("goal", "")
        context = state.get("context", "")
        history = state.get("history", [])
        memory_context = self.memory_manager.read_memory()
        tool_descriptions = self.registry.get_tool_descriptions()

        # Format conversation history for the prompt
        history_str = ""
        if history:
            history_str = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in history])

        prompt = self.prompt_template.format(
            prompt_a=self.core_prompt,
            instructions=self.instructions,
            tool_descriptions=tool_descriptions,
            memory_context=memory_context,
            goal=goal,
            context=f"Conversation History:\n{history_str}\n\nAdditional Context:\n{context}",
        )

        plan_steps: List[Dict[str, Any]] = []
        thinking_text: Optional[str] = None

        # ── Attempt 1: raw streaming + JSON extraction (Best for progress/thinking) ──
        # If on_thinking is set, we prefer streaming to show progress.
        if self.on_thinking:
            try:
                response = await ainvoke_with_thinking(
                    self.strong_model.model, prompt, self.on_thinking
                )
                thinking_text = response.additional_kwargs.get("_thinking")
                raw = response.content
                plan_steps = _parse_plan_from_text(raw)
            except Exception as e:
                self.logger.warning(f"[planner] Streaming attempt failed: {e}")

        # ── Attempt 2: structured output (Fallback or default if no thinking needed) ──
        if not plan_steps and self._structured_model is not None:
            try:
                result: _PlanSchema = await self._structured_model.ainvoke(prompt)
                if result.steps:
                    plan_steps = [
                        make_step(s.id, s.description, s.tool_name, s.params, s.risk_level)
                        for s in result.steps
                    ]
            except Exception as e:
                self.logger.warning(f"[planner] Structured output failed: {e}")

        # ── Attempt 3: raw fallback if structured was tried and failed ──
        if not plan_steps and not self.on_thinking:
            try:
                response = await ainvoke_with_thinking(
                    self.strong_model.model, prompt, self.on_thinking
                )
                thinking_text = response.additional_kwargs.get("_thinking")
                plan_steps = _parse_plan_from_text(response.content)
            except Exception as e:
                self.logger.error(f"[planner] Raw fallback also failed: {e}")

        if not plan_steps:
            self.logger.error("[planner] No plan generated after all attempts → noop.")
            plan_steps = [make_step(1, "No plan generated", "noop", {}, 0)]

        self.logger.info(f"[planner] Generated {len(plan_steps)} steps.")

        if self.on_plan_generated:
            try:
                self.on_plan_generated(plan_steps)
            except Exception:
                pass

        accumulated = _accumulate(state.get("thinking", ""), thinking_text)
        updates: Dict[str, Any] = {
            "plan": plan_steps,
            "current_step_index": 0,
            "should_advance": False,
            "session_log": [f"[planner] Plan created with {len(plan_steps)} steps."],
        }
        if accumulated:
            updates["thinking"] = accumulated
        return updates


# ── Helpers ────────────────────────────────────────────────────────────────

def _accumulate(current: str, new: Optional[str]) -> str:
    if not new:
        return current
    return (current + "\n\n" + new).strip() if current else new


def _parse_plan_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract a JSON plan object from model text output."""
    if not isinstance(text, str):
        text = get_text(text)

    # Try fenced code blocks first
    for pattern in [r"```json\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            steps = _try_parse_steps(match.group(1))
            if steps is not None:
                return steps

    # Try bare JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        steps = _try_parse_steps(match.group(0))
        if steps is not None:
            return steps

    return []


def _try_parse_steps(json_text: str) -> Optional[List[Dict[str, Any]]]:
    try:
        data = json.loads(json_text.strip())
        steps_raw = data.get("steps", [])
        if not steps_raw:
            return None
        return [
            make_step(
                s.get("id", i + 1),
                s.get("description", ""),
                s.get("tool_name", "noop"),
                s.get("params", {}),
                s.get("risk_level", 0),
            )
            for i, s in enumerate(steps_raw)
        ]
    except Exception:
        return None
