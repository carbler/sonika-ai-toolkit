"""load_skill — on-demand loader for folder-based skills (progressive disclosure).

When a bot runs in on-demand skills mode, only a lightweight index (name +
description, see :func:`sonika_ai_toolkit.skills.loader.render_skills_index`) is
injected into the system prompt. The model then calls ``load_skill(name)`` to
pull a single skill's full instructions into the conversation only when it is
relevant, instead of paying for every skill body on every turn.
"""

from typing import Dict, List, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from sonika_ai_toolkit.skills.loader import Skill

LOAD_SKILL_TOOL_NAME = "load_skill"


class LoadSkillSchema(BaseModel):
    name: str = Field(
        ...,
        description="Exact name of the skill to load, as listed in the SKILLS index.",
    )


def make_load_skill_tool(skills: List[Skill]) -> BaseTool:
    """Build a ``load_skill`` tool bound to the given skills.

    The returned tool closes over a ``{name: Skill}`` map and returns the full
    ``instructions`` body of the requested skill on demand. Unknown names return
    the list of available skills so the model can self-correct.
    """
    skill_map: Dict[str, Skill] = {s.name: s for s in skills}
    available = ", ".join(skill_map.keys()) or "(none)"

    class LoadSkillTool(BaseTool):
        """Load the full instructions of a skill by name, on demand."""

        name: str = LOAD_SKILL_TOOL_NAME
        description: str = (
            "Load the full instructions of a skill by its exact name (from the "
            "SKILLS index) when the current request relates to it. Returns the "
            "skill's instructions; follow them for the rest of the task."
        )
        args_schema: Type[BaseModel] = LoadSkillSchema
        # Informational load — never a risky action, never triggers approval.
        risk_level: int = 0

        def _run(self, name: str, **kwargs) -> str:
            skill = skill_map.get(name)
            if skill is None:
                return (
                    f"Unknown skill '{name}'. Available skills: {available}. "
                    "Call load_skill again with one of these exact names."
                )
            body = skill.instructions or "(this skill has no written instructions)"
            if skill.path:
                body += (
                    f"\n\nBundled files for this skill live under: {skill.path} — "
                    "read them with the file tools if the instructions reference them."
                )
            return body

        async def _arun(self, name: str, **kwargs) -> str:
            return self._run(name, **kwargs)

    return LoadSkillTool()
