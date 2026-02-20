"""DynamicToolSynthesizer — generates BaseTool subclasses at runtime via LLM."""

import importlib.util
import inspect
import logging
import os
import re
from typing import Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

_SYNTH_PROMPT = """You are a Python expert. Write a LangChain BaseTool subclass named `{class_name}` that implements: {tool_description}

Requirements:
- Inherit from langchain_core.tools.BaseTool
- Set class attributes: name = "{tool_name}", description = "<one sentence>"
- Define an args_schema as a Pydantic BaseModel inner class or standalone class
- Implement _run(self, **kwargs) -> str  (sync only, no async needed)
- Return a string result or error message
- Import everything you need at the top of the file
- The file must be self-contained (no external dependencies beyond stdlib + langchain + pydantic)
- Do NOT include if __name__ == "__main__" blocks

Output ONLY the Python code, no explanation, wrapped in a ```python block.
"""


class DynamicToolSynthesizer:
    """
    Generates a Python BaseTool subclass from a natural language description,
    writes it to disk, dynamically imports it, and returns the instance.
    """

    def __init__(self, code_model, memory_manager=None, skills_dir: str = "./memory/skills"):
        """
        Args:
            code_model: ILanguageModel used to generate the tool code.
            memory_manager: Optional MemoryManager to record the new skill.
            skills_dir: Directory where synthesized tool files are saved.
        """
        self.code_model = code_model
        self.memory_manager = memory_manager
        self.skills_dir = skills_dir

    async def synthesize(self, tool_name: str, tool_description: str) -> BaseTool:
        """
        Generate, write, import, and return a new BaseTool for tool_name.

        Args:
            tool_name: snake_case tool name (used as filename and tool.name).
            tool_description: Natural language description of what the tool does.

        Returns:
            A BaseTool instance ready to use.

        Raises:
            RuntimeError: If code generation or import fails.
        """
        class_name = _to_class_name(tool_name)
        prompt = _SYNTH_PROMPT.format(
            class_name=class_name,
            tool_name=tool_name,
            tool_description=tool_description,
        )

        logger.info(f"[DynamicToolSynthesizer] Generating tool: {tool_name}")

        # Generate code via LLM
        try:
            response = await self.code_model.model.ainvoke(prompt)
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            raise RuntimeError(f"LLM code generation failed for {tool_name}: {e}") from e

        # Extract python code block
        code = _extract_code_block(raw)
        if not code:
            raise RuntimeError(
                f"Could not extract Python code block from LLM response for {tool_name}. "
                f"Response was:\n{raw[:500]}"
            )

        # Write to disk
        os.makedirs(self.skills_dir, exist_ok=True)
        file_path = os.path.join(self.skills_dir, f"{tool_name}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"[DynamicToolSynthesizer] Wrote {file_path}")

        # Dynamically import
        tool_instance = _load_tool_from_file(file_path, class_name)

        # Record in SKILLS.md
        if self.memory_manager is not None:
            try:
                self.memory_manager.add_skill({
                    "name": tool_name,
                    "class": class_name,
                    "file": file_path,
                    "description": tool_description,
                })
            except Exception as e:
                logger.warning(f"[DynamicToolSynthesizer] Could not save skill: {e}")

        return tool_instance


def _to_class_name(tool_name: str) -> str:
    """Convert snake_case tool name to PascalCase class name."""
    return "".join(word.capitalize() for word in tool_name.split("_")) + "Tool"


def _extract_code_block(text: str) -> Optional[str]:
    """Extract the first ```python ... ``` block from text."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: try plain ``` block
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _load_tool_from_file(file_path: str, class_name: str) -> BaseTool:
    """Dynamically import a module from file_path and return an instance of class_name."""
    module_name = f"_synth_{os.path.splitext(os.path.basename(file_path))[0]}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create module spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"Error executing synthesized module {file_path}: {e}") from e

    # Find the target class
    tool_class = getattr(module, class_name, None)
    if tool_class is None:
        # Try to find any BaseTool subclass in the module
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, BaseTool) and obj is not BaseTool:
                tool_class = obj
                break

    if tool_class is None:
        raise RuntimeError(
            f"No BaseTool subclass '{class_name}' found in {file_path}. "
            f"Available names: {[n for n, _ in inspect.getmembers(module, inspect.isclass)]}"
        )

    try:
        return tool_class()
    except Exception as e:
        raise RuntimeError(f"Error instantiating {class_name} from {file_path}: {e}") from e
