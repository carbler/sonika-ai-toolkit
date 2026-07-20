"""Unit tests for the folder-based skills loader (skills.loader)."""


from langchain_core.tools import BaseTool

from sonika_ai_toolkit.skills import (
    Skill,
    load_skills,
    merge_skill_tools,
    render_skills_index,
    render_skills_prompt,
    resolve_skills,
)

TOOLS_PY = '''
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from sonika_ai_toolkit.tools.ask_user import AskUserQuestionTool  # imported, NOT defined here


class GreetSchema(BaseModel):
    name: str = Field(..., description="Who to greet")


class GreetTool(BaseTool):
    name: str = "greet"
    description: str = "Greets someone"
    args_schema: Type[BaseModel] = GreetSchema

    def _run(self, name: str) -> str:
        return f"Hola {name}"
'''


def _write_skill(root, folder, skill_md, tools_py=None):
    d = root / folder
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(skill_md, encoding="utf-8")
    if tools_py is not None:
        (d / "tools.py").write_text(tools_py, encoding="utf-8")
    return d


class TestFromDir:
    def test_frontmatter_parsed(self, tmp_path):
        d = _write_skill(
            tmp_path,
            "facturacion",
            "---\nname: billing\ndescription: Genera facturas\n---\n\nInstrucciones de facturación.",
        )
        skill = Skill.from_dir(str(d))
        assert skill.name == "billing"
        assert skill.description == "Genera facturas"
        assert skill.instructions == "Instrucciones de facturación."
        assert skill.tools == []
        assert skill.path == str(d)

    def test_no_frontmatter_uses_folder_name(self, tmp_path):
        d = _write_skill(tmp_path, "reportes", "Sabes generar reportes.")
        skill = Skill.from_dir(str(d))
        assert skill.name == "reportes"
        assert skill.description == ""
        assert skill.instructions == "Sabes generar reportes."

    def test_unclosed_frontmatter_treated_as_body(self, tmp_path):
        content = "---\nname: partial\nsin cierre"
        d = _write_skill(tmp_path, "raro", content)
        skill = Skill.from_dir(str(d))
        assert skill.name == "raro"
        assert skill.instructions == content

    def test_loads_tools_py(self, tmp_path):
        d = _write_skill(tmp_path, "saludos", "Saluda a la gente.", TOOLS_PY)
        skill = Skill.from_dir(str(d))
        assert len(skill.tools) == 1
        assert skill.tools[0].name == "greet"
        assert isinstance(skill.tools[0], BaseTool)
        # Classes merely imported by tools.py must not be instantiated.
        assert all(t.name != "ask_user" for t in skill.tools)


class TestLoadSkills:
    def test_loads_sorted_and_skips_non_skill_dirs(self, tmp_path):
        _write_skill(tmp_path, "beta", "b")
        _write_skill(tmp_path, "alpha", "a")
        (tmp_path / "no_skill_dir").mkdir()
        (tmp_path / "loose_file.md").write_text("x", encoding="utf-8")
        skills = load_skills(str(tmp_path))
        assert [s.name for s in skills] == ["alpha", "beta"]

    def test_broken_skill_is_skipped(self, tmp_path):
        _write_skill(tmp_path, "buena", "ok")
        _write_skill(tmp_path, "rota", "boom", tools_py="raise RuntimeError('broken skill')")
        skills = load_skills(str(tmp_path))
        assert [s.name for s in skills] == ["buena"]

    def test_missing_dir_returns_empty(self, tmp_path):
        assert load_skills(str(tmp_path / "nope")) == []


class TestResolveAndRender:
    def test_resolve_combines_list_and_dir(self, tmp_path):
        _write_skill(tmp_path, "desde_disco", "instr")
        explicit = Skill(name="en_codigo", instructions="hola")
        resolved = resolve_skills([explicit], str(tmp_path))
        assert [s.name for s in resolved] == ["en_codigo", "desde_disco"]

    def test_render_empty(self):
        assert render_skills_prompt([]) == ""

    def test_render_contains_names_and_instructions(self):
        skills = [
            Skill(name="uno", instructions="haz uno", description="primera"),
            Skill(name="dos", instructions="haz dos"),
        ]
        prompt = render_skills_prompt(skills)
        assert "## SKILLS" in prompt
        assert "### uno — primera" in prompt
        assert "haz uno" in prompt
        assert "### dos" in prompt
        assert "haz dos" in prompt

    def test_render_index_empty(self):
        assert render_skills_index([]) == ""

    def test_render_index_has_names_but_not_bodies(self):
        skills = [
            Skill(name="uno", instructions="CUERPO_UNO detallado", description="primera"),
            Skill(name="dos", instructions="CUERPO_DOS detallado"),
        ]
        index = render_skills_index(skills)
        # Lightweight index: names + descriptions, NEVER the full instructions.
        assert "## SKILLS" in index
        assert "load_skill" in index
        assert "- uno — primera" in index
        assert "- dos" in index
        assert "CUERPO_UNO" not in index
        assert "CUERPO_DOS" not in index


class TestMergeSkillTools:
    def _tool(self, name):
        from unittest.mock import MagicMock
        t = MagicMock()
        t.name = name
        return t

    def test_merges_and_dedupes_by_name(self):
        base = [self._tool("a")]
        skills = [
            Skill(name="s1", instructions="", tools=[self._tool("a"), self._tool("b")]),
            Skill(name="s2", instructions="", tools=[self._tool("b"), self._tool("c")]),
        ]
        merged = merge_skill_tools(base, skills)
        assert [t.name for t in merged] == ["a", "b", "c"]
        # Explicit tool wins over the skill's homonym.
        assert merged[0] is base[0]

    def test_does_not_mutate_base(self):
        base = [self._tool("a")]
        merge_skill_tools(base, [Skill(name="s", instructions="", tools=[self._tool("b")])])
        assert len(base) == 1
