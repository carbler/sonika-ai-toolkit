"""Validator Node - Quality Control."""

from typing import Dict, Any, List
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from sonika_ai_toolkit.agents.tasker.nodes.base_node import BaseNode

class ValidatorNode(BaseNode):
    """
    Analyzes the session before letting the bot finish.
    If it finds missing tasks or hallucinations, it sends the workflow back to Planner.
    """

    def __init__(self, model, logger=None):
        super().__init__(logger)
        self.model = model

        # Load prompt
        self.base_path = os.path.dirname(os.path.dirname(__file__))
        self.system_prompt_template = self._load_prompt("validator_system.txt")

    def _load_prompt(self, filename: str) -> str:
        try:
            path = os.path.join(self.base_path, "prompts", filename)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error loading prompt {filename}: {e}")
            return ""

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Perform validation check."""

        user_input = state.get("user_input", "")
        planner_output = state.get("planner_output", {})
        tools_executed = state.get("tools_executed", [])

        # Build prompt inputs
        tools_summary = self._build_tools_summary(tools_executed)
        planner_reasoning = planner_output.get("reasoning", "")

        system_prompt = self.system_prompt_template

        analysis_input = f"""## USER REQUEST
{user_input}

## TOOLS ACTUALLY EXECUTED
{tools_summary}

## PLANNER INTENTION
The planner wants to finish. Reasoning: "{planner_reasoning}"

---
Verify this work now."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=analysis_input)
        ]

        try:
            response = self.model.invoke(messages, config={"temperature": 0.0})
            content = response.content.strip()

            # Simple parsing of structured output (could be improved with structured output parser)
            status = "approved"
            feedback = ""

            lines = content.split('\n')
            for line in lines:
                if "Status:" in line:
                    if "rejected" in line.lower():
                        status = "rejected"
                if "Feedback:" in line:
                    feedback = line.split("Feedback:", 1)[1].strip()

            # Fallback if parsing fails but text clearly indicates rejection
            if "rejected" in content.lower() and status == "approved":
                 status = "rejected"
                 feedback = content

            log_msg = f"Validación: {status.upper()}. Feedback: {feedback}"

            # Debug Log explícito (se verá en consola si se activan logs)
            self.logger.info(f"VALIDATOR DEBUG -> Input: {planner_reasoning[:50]}... | Decision: {status} | Feedback: {feedback}")

            log_update = self._add_log(state, log_msg)

            return {
                "validator_output": {
                    "status": status,
                    "feedback": feedback
                },
                **log_update
            }

        except Exception as e:
            self.logger.error(f"Validator failed: {e}")
            # Fail safe: approve to avoid infinite loops if validator crashes
            return {
                "validator_output": {
                    "status": "approved",
                    "feedback": "Validator error, bypassing."
                }
            }

    def _build_tools_summary(self, tools_executed: List[Dict[str, Any]]) -> str:
        if not tools_executed:
            return "No tools executed."

        summary = []
        for tool in tools_executed:
            name = tool.get("tool_name")
            status = tool.get("status")
            summary.append(f"- Tool: {name} | Status: {status}")
        return "\n".join(summary)
