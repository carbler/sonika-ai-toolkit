"""RunPythonTool — executes Python code in a subprocess."""

import subprocess
import sys
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class _RunPythonInput(BaseModel):
    code: str = Field(description="Python code to execute.")
    timeout: int = Field(default=30, description="Timeout in seconds.")


class RunPythonTool(BaseTool):
    name: str = "run_python"
    description: str = (
        "Execute Python code in a subprocess and return stdout + stderr. "
        "Useful for calculations, data transformations, and scripting tasks."
    )
    args_schema: Type[BaseModel] = _RunPythonInput
    risk_hint: int = 1

    def _run(self, code: str, timeout: int = 30) -> str:
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            output_parts = []
            if result.stdout:
                output_parts.append(f"stdout:\n{result.stdout.strip()}")
            if result.stderr:
                output_parts.append(f"stderr:\n{result.stderr.strip()}")
            if not output_parts:
                output_parts.append(f"(exit code {result.returncode}, no output)")
            return "\n".join(output_parts)
        except subprocess.TimeoutExpired:
            return f"Error: code timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"
