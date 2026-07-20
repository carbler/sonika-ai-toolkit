"""Unit tests for the on-demand load_skill tool (tools.skill_tools)."""

from sonika_ai_toolkit.skills import Skill
from sonika_ai_toolkit.tools.skill_tools import (
    LOAD_SKILL_TOOL_NAME,
    make_load_skill_tool,
)


def _skills():
    return [
        Skill(
            name="pdf-export",
            instructions="STEP 1 do X\nSTEP 2 do Y",
            description="Export docs to PDF",
            path="/tmp/skills/pdf-export",
        ),
        Skill(name="cleanup", instructions="clean the csv"),
    ]


class TestLoadSkillTool:
    def test_tool_metadata(self):
        tool = make_load_skill_tool(_skills())
        assert tool.name == LOAD_SKILL_TOOL_NAME
        # Informational load — never triggers an approval interrupt.
        assert tool.risk_level == 0

    def test_returns_full_body_and_path(self):
        tool = make_load_skill_tool(_skills())
        out = tool.invoke({"name": "pdf-export"})
        assert "STEP 1 do X" in out
        assert "STEP 2 do Y" in out
        # Level-3 progressive disclosure: bundled path is surfaced.
        assert "/tmp/skills/pdf-export" in out

    def test_body_without_path(self):
        tool = make_load_skill_tool(_skills())
        out = tool.invoke({"name": "cleanup"})
        assert "clean the csv" in out
        assert "Bundled files" not in out

    def test_unknown_name_lists_available(self):
        tool = make_load_skill_tool(_skills())
        out = tool.invoke({"name": "does-not-exist"})
        assert "Unknown skill" in out
        assert "pdf-export" in out
        assert "cleanup" in out

    def test_skill_without_instructions(self):
        tool = make_load_skill_tool([Skill(name="empty", instructions="")])
        out = tool.invoke({"name": "empty"})
        assert "no written instructions" in out
