"""File system tools: read, write, list, delete."""

import os
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class _ReadFileInput(BaseModel):
    path: str = Field(description="Absolute or relative path to the file to read.")


class ReadFileTool(BaseTool):
    name: str = "read_file"
    description: str = "Read the contents of a file at the given path."
    args_schema: Type[BaseModel] = _ReadFileInput
    risk_hint: int = 0

    def _run(self, path: str) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading {path}: {e}"


class _WriteFileInput(BaseModel):
    path: str = Field(description="Path where the file will be written.")
    content: str = Field(description="Content to write to the file.")


class WriteFileTool(BaseTool):
    name: str = "write_file"
    description: str = "Write content to a file, creating parent directories if needed."
    args_schema: Type[BaseModel] = _WriteFileInput
    risk_hint: int = 1

    def _run(self, path: str, content: str) -> str:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Written {len(content)} chars to {path}"
        except Exception as e:
            return f"Error writing {path}: {e}"


class _ListDirInput(BaseModel):
    path: str = Field(description="Directory path to list.")


class ListDirTool(BaseTool):
    name: str = "list_dir"
    description: str = "List files and subdirectories in a directory."
    args_schema: Type[BaseModel] = _ListDirInput
    risk_hint: int = 0

    def _run(self, path: str) -> str:
        try:
            entries = os.listdir(path)
            if not entries:
                return f"{path} is empty."
            lines = []
            for entry in sorted(entries):
                full = os.path.join(path, entry)
                tag = "[dir]" if os.path.isdir(full) else "[file]"
                lines.append(f"{tag} {entry}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing {path}: {e}"


class _DeleteFileInput(BaseModel):
    path: str = Field(description="Path of the file to delete.")


class DeleteFileTool(BaseTool):
    name: str = "delete_file"
    description: str = "Permanently delete a file. Use with caution."
    args_schema: Type[BaseModel] = _DeleteFileInput
    risk_hint: int = 2

    def _run(self, path: str) -> str:
        try:
            os.remove(path)
            return f"Deleted {path}"
        except Exception as e:
            return f"Error deleting {path}: {e}"
