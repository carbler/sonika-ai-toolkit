"""RunBashTool — executes a shell command in a subprocess."""

import subprocess
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class _RunBashInput(BaseModel):
    command: str = Field(description="The shell command to execute.")
    timeout: int = Field(default=30, description="Timeout in seconds.")


class RunBashTool(BaseTool):
    name: str = "run_bash"
    description: str = (
        "Execute a shell command and return stdout + stderr. "
        "Use for file operations, CLI utilities, or any system command."
    )
    args_schema: Type[BaseModel] = _RunBashInput
    risk_hint: int = 1

    def _run(self, command: str, timeout: int = 30) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
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
            return f"Error: command timed out after {timeout}s"
        except Exception as e:
            return f"Error: {e}"
