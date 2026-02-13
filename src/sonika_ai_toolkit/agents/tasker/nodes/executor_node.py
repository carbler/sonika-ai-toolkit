"""Executor Node - Executes ONE tool at a time for ReAct loop."""

from typing import Dict, Any, Optional, Callable, List
import json
import asyncio
from langchain_core.messages import ToolMessage
from sonika_ai_toolkit.agents.tasker.nodes.base_node import BaseNode


class ExecutorNode(BaseNode):
    """Executes a single tool and returns observation to agent."""

    def __init__(
        self,
        tools: List[Any],
        max_retries: int = 2,
        on_tool_start: Optional[Callable] = None,
        on_tool_end: Optional[Callable] = None,
        on_tool_error: Optional[Callable] = None,
        logger=None
    ):
        super().__init__(logger)
        self.tools = {tool.name: tool for tool in tools}
        self.max_retries = max_retries
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_tool_error = on_tool_error

    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool(s) specified by planner."""

        planner_output = state.get("planner_output", {})
        tool_calls = planner_output.get("tool_calls", [])

        # Fallback para backward compatibility
        if not tool_calls and planner_output.get("tool"):
             tool_calls = [{
                 "name": planner_output.get("tool"),
                 "args": planner_output.get("params", {}),
                 "id": planner_output.get("tool_call_id", "unknown")
             }]

        if not tool_calls:
            self.logger.error("No tool specified by planner")
            return self._create_error_output("No tool specified", None, "unknown")

        results_accumulated = []
        messages_accumulated = []
        logs_accumulated = []

        # Ejecutar todas las tool calls
        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            params = tool_call["args"]
            tool_call_id = tool_call["id"]

            # Log inicio
            self._add_log(state, f"Executing tool '{tool_name}'")

            # --- 1. VALIDACIÓN ESTRUCTURAL (Smart Executor) ---
            validation_error = self._validate_params(tool_name, params)
            if validation_error:
                # Si falla la validación, simulamos un error de ejecución
                # para forzar al Planner a corregir.
                self.logger.warning(f"Validation failed for {tool_name}: {validation_error}")
                error_res = self._create_single_error(
                    f"PRE-EXECUTION VALIDATION ERROR: {validation_error}",
                    tool_name,
                    tool_call_id
                )
                logs_accumulated.append(f"Tool '{tool_name}' blocked: {validation_error}")
                results_accumulated.append(error_res["result"])
                messages_accumulated.append(error_res["message"])
                continue

            # --- 2. VERIFICACIÓN DE EXISTENCIA ---
            if tool_name not in self.tools:
                self.logger.error(f"Tool {tool_name} not found")
                error_res = self._create_single_error(f"Tool {tool_name} not found", tool_name, tool_call_id)
                results_accumulated.append(error_res["result"])
                messages_accumulated.append(error_res["message"])
                continue

            # --- 3. EJECUCIÓN REAL ---
            success = False
            for attempt in range(self.max_retries + 1):
                try:
                    result = await self._execute_tool(tool_name, params)
                    output_str = result.get('output', '')

                    logs_accumulated.append(f"Tool '{tool_name}' completed.")

                    tool_msg = ToolMessage(
                        content=str(output_str),
                        tool_call_id=tool_call_id,
                        name=tool_name
                    )

                    results_accumulated.append(result)
                    messages_accumulated.append(tool_msg)
                    success = True
                    break

                except Exception as e:
                    self.logger.error(f"Tool execution failed (attempt {attempt + 1}): {e}")
                    if self.on_tool_error:
                        try:
                            self.on_tool_error(tool_name, str(e))
                        except:
                            pass

                    if attempt >= self.max_retries:
                         error_res = self._create_single_error(str(e), tool_name, tool_call_id)
                         logs_accumulated.append(f"Tool '{tool_name}' failed after retries: {e}")
                         results_accumulated.append(error_res["result"])
                         messages_accumulated.append(error_res["message"])

        # Generar actualización de logs final
        final_logs = []
        for log in logs_accumulated:
             log_dict = self._add_log(state, log)
             final_logs.extend(log_dict["logs"])

        return {
            "executor_output": {
                "status": "success",
                "tools_executed": results_accumulated
            },
            "tools_executed": results_accumulated,
            "messages": messages_accumulated,
            "logs": final_logs
        }

    def _validate_params(self, tool_name: str, params: Dict[str, Any]) -> Optional[str]:
        """
        Structural Validation Logic.
        Returns an error string if validation fails, None if passes.
        """
        # Regla general: Ningún parámetro debe ser string vacío o placeholder obvio
        for key, value in params.items():
            if isinstance(value, str):
                cleaned = value.strip().lower()
                if cleaned == "" or cleaned in ["unknown", "n/a", "undefined", "null", "none", "string"]:
                     return (
                         f"Parameter '{key}' has an invalid empty or placeholder value ('{value}'). "
                         f"You MUST ask the user for this information before calling the tool."
                     )
        return None

    async def _execute_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single tool."""
        tool = self.tools[tool_name]

        if self.on_tool_start:
            try:
                self.on_tool_start(tool_name, json.dumps(params))
            except:
                pass

        output = await tool.ainvoke(params)

        if self.on_tool_end:
            try:
                self.on_tool_end(tool_name, str(output))
            except:
                pass

        return {
            "tool_name": tool_name,
            "args": json.dumps(params),
            "output": str(output),
            "status": "success"
        }

    def _create_single_error(self, error: str, tool_name: str, tool_call_id: str):
         error_result = {
            "tool_name": tool_name or "unknown",
            "output": f"ERROR: {error}",
            "status": "failed"
        }
         tool_msg = ToolMessage(
            content=f"ERROR: {error}",
            tool_call_id=tool_call_id,
            name=tool_name or "unknown",
            status="error"
        )
         return {"result": error_result, "message": tool_msg}

    def _create_error_output(self, error: str, tool_name: str = None, tool_call_id: str = "unknown") -> Dict[str, Any]:
        """Create global error output."""
        res = self._create_single_error(error, tool_name, tool_call_id)
        return {
            "executor_output": {
                "status": "failed",
                "tools_executed": [res["result"]],
                "error": error
            },
            "tools_executed": [res["result"]],
            "messages": [res["message"]]
        }
