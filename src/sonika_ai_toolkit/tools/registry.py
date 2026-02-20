"""Tool registry for OrchestratorBot."""

import inspect
from typing import Dict, List, Optional
from langchain_core.tools import BaseTool


class ToolRegistry:
    """Central registry for all tools available to the orchestrator."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        return name in self._tools

    def list(self) -> List[str]:
        return list(self._tools.keys())

    def all(self) -> List[BaseTool]:
        return list(self._tools.values())

    def get_tool_descriptions(self) -> str:
        """Return formatted tool descriptions including exact parameter names."""
        if not self._tools:
            return "No tools available."
        lines = []
        for name, tool in self._tools.items():
            params = self._get_param_str(tool)
            lines.append(f"- {name}: {tool.description} | params: {params}")
        return "\n".join(lines)

    def _get_param_str(self, tool: BaseTool) -> str:
        """Extract parameter names from args_schema or _run signature."""
        params = []
        try:
            schema = getattr(tool, "args_schema", None)
            if schema is not None:
                # Pydantic v2
                if hasattr(schema, "model_fields"):
                    for fname, field in schema.model_fields.items():
                        ann = getattr(field.annotation, "__name__", str(field.annotation))
                        desc = ""
                        if hasattr(field, "description") and field.description:
                            desc = f" ({field.description})"
                        params.append(f"{fname}: {ann}{desc}")
                # Pydantic v1
                elif hasattr(schema, "__fields__"):
                    for fname in schema.__fields__:
                        params.append(fname)
                return ", ".join(params) if params else "no params"

            # Fallback: inspect _run signature
            if hasattr(tool, "_run"):
                sig = inspect.signature(tool._run)
                for pname, _ in sig.parameters.items():
                    if pname not in ("self", "args", "kwargs", "run_manager"):
                        params.append(pname)
        except Exception:
            pass
        return ", ".join(params) if params else "no params"
