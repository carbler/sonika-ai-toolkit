"""load_memory node — reads MEMORY.md + SKILLS.md and registers dynamic tools."""

import logging
import os
from typing import Dict, Any

from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState


class LoadMemoryNode:
    """
    Reads MEMORY.md and SKILLS.md, then dynamically loads any synthesized tools
    found in skills_dir so they are available to the planner and executor.
    """

    def __init__(self, memory_manager, tool_registry, logger=None):
        self.memory_manager = memory_manager
        self.registry = tool_registry
        self.logger = logger or logging.getLogger(__name__)

    def __call__(self, state: OrchestratorState) -> Dict[str, Any]:
        skills_dir = state.get("skills_dir", "")
        loaded_skills = []

        # Try to load any previously synthesized tools from skills_dir.
        if skills_dir and os.path.isdir(skills_dir):
            loaded_skills = self._load_skill_files(skills_dir)

        memory_content = self.memory_manager.read_memory()
        skill_count = len(self.memory_manager.read_skills())

        log_lines = [
            f"[load_memory] Memory loaded ({len(memory_content)} chars, "
            f"{skill_count} skills registered, {len(loaded_skills)} tools loaded from disk)"
        ]

        return {"session_log": log_lines}

    def _load_skill_files(self, skills_dir: str):
        """Import synthesized tool files and register them."""
        import importlib.util
        import inspect
        from langchain_core.tools import BaseTool

        loaded = []
        for fname in os.listdir(skills_dir):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(skills_dir, fname)
            try:
                module_name = f"_skill_{fname[:-3]}"
                spec = importlib.util.spec_from_file_location(module_name, fpath)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)  # type: ignore[attr-defined]
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseTool)
                        and obj is not BaseTool
                        and not self.registry.has(obj.name if isinstance(obj.name, str) else "")
                    ):
                        instance = obj()
                        self.registry.register(instance)
                        loaded.append(instance.name)
                        self.logger.info(f"[load_memory] Loaded skill tool: {instance.name}")
            except Exception as e:
                self.logger.warning(f"[load_memory] Could not load {fpath}: {e}")
        return loaded
