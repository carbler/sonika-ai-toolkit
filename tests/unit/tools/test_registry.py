"""Unit tests for ToolRegistry (tools.registry).

The registry is the central lookup the orchestrator uses; get_tool_descriptions()
must expose exact parameter names extracted from each tool's args_schema.
"""

from sonika_ai_toolkit.tools.registry import ToolRegistry


class TestRegistryBasics:
    def test_register_and_get(self, email_tool):
        reg = ToolRegistry()
        reg.register(email_tool)
        assert reg.get(email_tool.name) is email_tool

    def test_has(self, email_tool):
        reg = ToolRegistry()
        assert reg.has(email_tool.name) is False
        reg.register(email_tool)
        assert reg.has(email_tool.name) is True

    def test_get_missing_returns_none(self):
        assert ToolRegistry().get("nope") is None

    def test_list_and_all(self, all_tools):
        reg = ToolRegistry()
        for t in all_tools:
            reg.register(t)
        assert set(reg.list()) == {t.name for t in all_tools}
        assert {t.name for t in reg.all()} == {t.name for t in all_tools}

    def test_register_overwrites_same_name(self, email_tool):
        reg = ToolRegistry()
        reg.register(email_tool)
        reg.register(email_tool)
        assert len(reg.list()) == 1


class TestToolDescriptions:
    def test_empty_registry(self):
        assert ToolRegistry().get_tool_descriptions() == "No tools available."

    def test_includes_name_description_and_params(self, all_tools):
        reg = ToolRegistry()
        for t in all_tools:
            reg.register(t)
        desc = reg.get_tool_descriptions()
        for t in all_tools:
            assert t.name in desc
            assert t.description in desc
        assert "params:" in desc

    def test_param_names_from_args_schema(self, email_tool):
        reg = ToolRegistry()
        reg.register(email_tool)
        desc = reg.get_tool_descriptions()
        # Field names from the tool's Pydantic args_schema must be present.
        field_names = list(email_tool.args_schema.model_fields.keys())
        assert field_names
        assert any(fname in desc for fname in field_names)
