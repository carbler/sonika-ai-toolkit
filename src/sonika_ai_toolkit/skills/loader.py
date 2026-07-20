"""Folder-based skills — reusable capability packs for agents.

A skill is a directory containing a ``SKILL.md`` file (instructions injected
into the bot's system prompt) and an optional ``tools.py`` module whose
``BaseTool`` subclasses are merged into the bot's tool list.

.. warning::
    ``tools.py`` is imported and executed as arbitrary Python code. Only load
    skill directories you trust.

Layout::

    skills/
    ├── facturacion/
    │   ├── SKILL.md      # instructions (optional --- frontmatter: name/description)
    │   └── tools.py      # optional BaseTool subclasses
    └── reportes/
        └── SKILL.md
"""

import importlib.util
import inspect
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

SKILL_FILE = "SKILL.md"
TOOLS_FILE = "tools.py"


@dataclass
class Skill:
    """A capability pack: instructions plus optional tools."""

    name: str
    instructions: str
    description: str = ""
    tools: List[BaseTool] = field(default_factory=list)
    path: Optional[str] = None

    @classmethod
    def from_dir(cls, path: str) -> "Skill":
        """Load a skill from a directory containing SKILL.md (+ optional tools.py).

        The SKILL.md may start with a ``---`` frontmatter block declaring
        ``name:`` and ``description:``; the folder name is the fallback name
        and the remaining markdown body becomes the instructions.
        """
        skill_md = os.path.join(path, SKILL_FILE)
        with open(skill_md, "r", encoding="utf-8") as f:
            raw = f.read()

        meta, body = _parse_frontmatter(raw)
        name = meta.get("name") or os.path.basename(os.path.normpath(path))

        tools: List[BaseTool] = []
        tools_py = os.path.join(path, TOOLS_FILE)
        if os.path.isfile(tools_py):
            tools = _load_tools_from_file(tools_py)

        return cls(
            name=name,
            description=meta.get("description", ""),
            instructions=body.strip(),
            tools=tools,
            path=path,
        )


def load_skills(skills_dir: str) -> List[Skill]:
    """Load every skill under ``skills_dir`` (subdirectories with a SKILL.md).

    Skills are loaded in sorted directory order for determinism. A broken
    skill is logged and skipped — it never aborts the rest.
    """
    if not os.path.isdir(skills_dir):
        logger.warning(f"[skills] Directory not found: {skills_dir}")
        return []

    skills: List[Skill] = []
    for entry in sorted(os.listdir(skills_dir)):
        sub = os.path.join(skills_dir, entry)
        if not os.path.isdir(sub) or not os.path.isfile(os.path.join(sub, SKILL_FILE)):
            continue
        try:
            skills.append(Skill.from_dir(sub))
        except Exception as e:
            logger.warning(f"[skills] Skipping broken skill '{entry}': {e}")
    return skills


def resolve_skills(
    skills: Optional[List[Skill]] = None,
    skills_dir: Optional[str] = None,
) -> List[Skill]:
    """Combine explicitly-passed skills with those loaded from a directory."""
    resolved = list(skills) if skills else []
    if skills_dir:
        resolved.extend(load_skills(skills_dir))
    return resolved


def render_skills_prompt(skills: List[Skill]) -> str:
    """Render skills as a markdown block to append to a system prompt."""
    if not skills:
        return ""
    blocks = [
        "## SKILLS",
        "You have the following additional skills. Apply their instructions "
        "whenever the request is related to them.",
    ]
    for skill in skills:
        header = f"### {skill.name}"
        if skill.description:
            header += f" — {skill.description}"
        blocks.append(f"{header}\n{skill.instructions}" if skill.instructions else header)
    return "\n\n".join(blocks)


def render_skills_index(skills: List[Skill]) -> str:
    """Render a lightweight skills index (name + description only).

    Progressive-disclosure counterpart of :func:`render_skills_prompt`: instead
    of injecting every skill's full body on every turn, this emits just one
    ``- name — description`` line per skill and tells the model to call the
    ``load_skill`` tool to fetch a skill's full instructions on demand.
    """
    if not skills:
        return ""
    blocks = [
        "## SKILLS",
        "You have the following skills available. When a request relates to one, "
        "call the `load_skill` tool with its exact name to load its full "
        "instructions BEFORE acting. Only load a skill when it is relevant.",
    ]
    for skill in skills:
        line = f"- {skill.name}"
        if skill.description:
            line += f" — {skill.description}"
        blocks.append(line)
    return "\n\n".join(blocks[:2]) + "\n" + "\n".join(blocks[2:])


def merge_skill_tools(base_tools: List[BaseTool], skills: List[Skill]) -> List[BaseTool]:
    """Merge skill tools into ``base_tools``, deduplicating by tool name.

    Explicitly-passed tools win over skill tools with the same name.
    """
    merged = list(base_tools)
    seen = {getattr(t, "name", None) for t in merged}
    for skill in skills:
        for tool in skill.tools:
            if tool.name in seen:
                logger.debug(
                    f"[skills] Tool '{tool.name}' from skill '{skill.name}' "
                    "already registered — keeping the existing one."
                )
                continue
            merged.append(tool)
            seen.add(tool.name)
    return merged


def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Split optional ``---`` frontmatter (``key: value`` lines) from the body."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    meta: Dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:])
            return meta, body
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
    # No closing --- found: treat the whole file as body
    return {}, text


def _load_tools_from_file(file_path: str) -> List[BaseTool]:
    """Import a tools.py module and instantiate every BaseTool subclass it defines.

    Classes merely *imported* by the module are excluded (``__module__`` check).
    """
    module_name = f"_skill_{uuid.uuid4().hex[:8]}_{os.path.basename(os.path.dirname(file_path))}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create module spec from {file_path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[attr-defined]
    except Exception as e:
        raise RuntimeError(f"Error executing skill module {file_path}: {e}") from e

    tools: List[BaseTool] = []
    for _, obj in inspect.getmembers(module, inspect.isclass):
        if (
            issubclass(obj, BaseTool)
            and obj is not BaseTool
            and obj.__module__ == module.__name__
        ):
            try:
                tools.append(obj())
            except Exception as e:
                raise RuntimeError(
                    f"Error instantiating {obj.__name__} from {file_path}: {e}"
                ) from e
    return tools
