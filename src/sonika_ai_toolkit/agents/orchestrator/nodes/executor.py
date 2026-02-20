"""executor node — invoke a tool or synthesize one on-the-fly."""

import logging
from typing import Any, Callable, Dict, Optional

from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState, update_step
from sonika_ai_toolkit.agents.orchestrator.utils import find_missing_params


class ExecutorNode:
    """
    Runs the current step's tool.  If retry_strategy == 'synth_tool', it
    delegates to DynamicToolSynthesizer first, then runs the new tool.
    """

    def __init__(
        self,
        tool_registry,
        synthesizer=None,
        on_step_end: Optional[Callable[[Dict, str], None]] = None,
        logger=None,
    ):
        self.registry = tool_registry
        self.synthesizer = synthesizer
        self.on_step_end = on_step_end
        self.logger = logger or logging.getLogger(__name__)

    async def __call__(self, state: OrchestratorState) -> Dict[str, Any]:
        plan = list(state.get("plan", []))
        index = state.get("current_step_index", 0)
        step = dict(plan[index]) if index < len(plan) else {}
        retry_strategy = state.get("retry_strategy")
        log_lines = []

        tool_name = step.get("tool_name", "")
        params = step.get("params", {}) or {}

        # Mark step as running
        new_plan = update_step(plan, index, status="running")

        # ── synth_tool: generate a new tool then run it ────────────────────
        if retry_strategy == "synth_tool":
            if self.synthesizer is None:
                result_text = "Error: DynamicToolSynthesizer not configured."
                log_lines.append(f"[executor] synth_tool requested but synthesizer is None.")
                new_plan = update_step(new_plan, index, status="failed", error=result_text)
                return {
                    "plan": new_plan,
                    "last_result": None,
                    "last_error": result_text,
                    "retry_strategy": None,
                    "session_log": log_lines,
                    "tool_outputs": [],
                }
            try:
                description = params.get("description", f"Tool to accomplish: {step.get('description', '')}")
                log_lines.append(f"[executor] Synthesizing tool: {tool_name}")
                tool = await self.synthesizer.synthesize(tool_name, description)
                self.registry.register(tool)
                log_lines.append(f"[executor] Tool synthesized and registered: {tool_name}")
            except Exception as e:
                err = f"Synthesis failed for {tool_name}: {e}"
                self.logger.error(f"[executor] {err}")
                new_plan = update_step(new_plan, index, status="failed", error=err)
                return {
                    "plan": new_plan,
                    "last_result": None,
                    "last_error": err,
                    "retry_strategy": None,
                    "session_log": log_lines,
                    "tool_outputs": [],
                }

        # ── Normal execution ───────────────────────────────────────────────
        tool = self.registry.get(tool_name)
        if tool is None and tool_name != "noop":
            err = f"Tool not found in registry: {tool_name}"
            self.logger.error(f"[executor] {err}")
            new_plan = update_step(new_plan, index, status="failed", error=err)
            return {
                "plan": new_plan,
                "last_result": None,
                "last_error": err,
                "retry_strategy": None,
                "session_log": log_lines,
                "tool_outputs": [],
            }

        # noop: no-operation step
        if tool_name == "noop" or tool is None:
            result_text = "noop: no operation performed."
            log_lines.append(f"[executor] noop step {index + 1}.")
            new_plan = update_step(new_plan, index, status="success", result=result_text)
            return {
                "plan": new_plan,
                "last_result": result_text,
                "last_error": None,
                "retry_strategy": None,
                "session_log": log_lines,
                "tool_outputs": [{"step_id": step.get("id"), "tool": tool_name, "output": result_text}],
            }

        # ── Validate required params before invoking ──────────────────────
        missing = find_missing_params(tool, params)
        if missing:
            err = f"Missing required params for {tool_name}: {', '.join(missing)}"
            self.logger.warning(f"[executor] {err}")
            new_plan = update_step(new_plan, index, status="failed", error=err)
            return {
                "plan": new_plan,
                "last_result": None,
                "last_error": err,
                "retry_strategy": None,
                "session_log": log_lines + [f"[executor] {err}"],
                "tool_outputs": [],
            }

        try:
            log_lines.append(f"[executor] Running {tool_name} with params: {params}")
            output = await tool.ainvoke(params)
            result_text = str(output)
            self.logger.info(f"[executor] {tool_name} → {result_text[:200]}")
            new_plan = update_step(new_plan, index, status="success", result=result_text)

            tool_output_entry = {
                "step_id": step.get("id"),
                "tool": tool_name,
                "output": result_text,
            }
            log_lines.append(f"[executor] {tool_name} completed.")

            if self.on_step_end:
                try:
                    self.on_step_end(new_plan[index], result_text)
                except Exception:
                    pass

            return {
                "plan": new_plan,
                "last_result": result_text,
                "last_error": None,
                "retry_strategy": None,
                "session_log": log_lines,
                "tool_outputs": [tool_output_entry],
            }

        except Exception as e:
            err = f"{tool_name} raised: {e}"
            self.logger.error(f"[executor] {err}")
            new_plan = update_step(new_plan, index, status="failed", error=err)
            return {
                "plan": new_plan,
                "last_result": None,
                "last_error": err,
                "retry_strategy": None,
                "session_log": log_lines,
                "tool_outputs": [],
            }
