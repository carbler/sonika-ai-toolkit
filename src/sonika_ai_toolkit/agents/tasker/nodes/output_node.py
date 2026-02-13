"""Output Node - Generates natural language response."""

from typing import Dict, Any, List
import os
from langchain_core.messages import SystemMessage, HumanMessage
from sonika_ai_toolkit.agents.tasker.nodes.base_node import BaseNode

class OutputNode(BaseNode):
    """Generates final response to user."""

    def __init__(self, model, logger=None):
        super().__init__(logger)
        self.model = model

        # Load prompt
        self.base_path = os.path.dirname(os.path.dirname(__file__))
        self.system_prompt_template = self._load_prompt("output_system.txt")

    def _load_prompt(self, filename: str) -> str:
        try:
            path = os.path.join(self.base_path, "prompts", filename)
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Error loading prompt {filename}: {e}")
            return ""

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate final response."""
        try:
            response_text = self._generate_response(state)

            preview = response_text[:80].replace('\n', ' ')
            log_update = self._add_log(state, f"Respuesta generada: {preview}...")

            return {
                "output_node_response": response_text,
                **log_update
            }

        except Exception as e:
            self.logger.error(f"Output generation failed: {e}")
            return {"output_node_response": "Disculpa, encontré un error al procesar tu solicitud."}

    def _generate_response(self, state: Dict[str, Any]) -> str:
        """Generate response based on planner reasoning and tools."""

        user_input = state.get("user_input", "")
        personality_tone = state.get("personality_tone", "")
        limitations = state.get("limitations", "")
        planner_output = state.get("planner_output", {})
        tools_executed = state.get("tools_executed", [])
        dynamic_info = state.get("dynamic_info", "")

        planner_reasoning = planner_output.get('reasoning', 'No reasoning provided')
        results_summary = self._build_results_summary(tools_executed)

        system_prompt = self.system_prompt_template.format(
            personality_tone=personality_tone,
            limitations=limitations
        )

        analysis_input = f"""## PLANNER REASONING
{planner_reasoning}

## DYNAMIC CONTEXT
{dynamic_info}

## TOOLS RESULTS
{results_summary}

## USER MESSAGE
{user_input}

---
Generate your response now:"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=analysis_input)
        ]

        response = self.model.invoke(messages, config={"temperature": 0.3})

        if hasattr(response, 'content'):
            return response.content.strip()
        return str(response).strip()

    def _build_results_summary(self, tools_executed: List[Dict[str, Any]]) -> str:
        """Build summary of tool results."""
        if not tools_executed:
            return "No tools were executed."

        summary = []
        for tool in tools_executed:
            tool_name = tool.get("tool_name", "unknown")
            output = tool.get("output", "No output")
            status = tool.get("status", "unknown")

            if status == "success":
                summary.append(f"✓ {tool_name}: {output}")
            else:
                summary.append(f"✗ {tool_name} failed: {output}")

        return "\n".join(summary)
