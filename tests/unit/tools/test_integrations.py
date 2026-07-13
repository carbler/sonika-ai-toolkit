"""
Unit tests for sonika_ai_toolkit.tools.integrations

Covers:
  - EmailTool: name, description, _run return value, required params
  - SaveContacto: name, description, _run return value, required params
  - Both tools are valid LangChain BaseTool subclasses
"""

import pytest
from langchain_community.tools import BaseTool

from sonika_ai_toolkit.tools.integrations import EmailTool, SaveContacto


# ---------------------------------------------------------------------------
# EmailTool
# ---------------------------------------------------------------------------

class TestEmailTool:
    def test_is_base_tool_subclass(self):
        assert issubclass(EmailTool, BaseTool)

    def test_name_attribute(self):
        tool = EmailTool()
        assert tool.name == "EmailTool"

    def test_description_is_not_empty(self):
        tool = EmailTool()
        assert len(tool.description) > 0

    def test_run_returns_success_string(self):
        tool = EmailTool()
        result = tool._run(
            to_email="test@example.com",
            subject="Hello",
            message="This is a test email",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_success_message_content(self):
        tool = EmailTool()
        result = tool._run(
            to_email="user@domain.com",
            subject="Test",
            message="Body text",
        )
        assert "éxito" in result or "exitoso" in result or "success" in result.lower() or "enviado" in result.lower()

    def test_can_be_instantiated_without_args(self):
        tool = EmailTool()
        assert tool is not None

    @pytest.mark.parametrize("to_email,subject,message", [
        ("a@b.com", "Hi", "Hello"),
        ("x@y.org", "Subject with spaces", "Multiline\nbody"),
        ("foo@bar.net", "Test", ""),
    ])
    def test_run_with_various_inputs(self, to_email, subject, message):
        tool = EmailTool()
        result = tool._run(to_email=to_email, subject=subject, message=message)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# SaveContacto
# ---------------------------------------------------------------------------

class TestSaveContacto:
    def test_is_base_tool_subclass(self):
        assert issubclass(SaveContacto, BaseTool)

    def test_name_attribute(self):
        tool = SaveContacto()
        assert tool.name == "SaveContact"

    def test_description_is_not_empty(self):
        tool = SaveContacto()
        assert len(tool.description) > 0

    def test_run_returns_success_string(self):
        tool = SaveContacto()
        result = tool._run(
            nombre="John Doe",
            correo="john@example.com",
            telefono="555-1234",
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_success_message_content(self):
        tool = SaveContacto()
        result = tool._run(nombre="Jane", correo="jane@x.com", telefono="123")
        assert "guardado" in result.lower() or "success" in result.lower()

    def test_can_be_instantiated_without_args(self):
        tool = SaveContacto()
        assert tool is not None

    @pytest.mark.parametrize("nombre,correo,telefono", [
        ("Alice", "alice@x.com", "111-2222"),
        ("Bob Smith", "bob@y.net", ""),
        ("", "noemail@domain.org", "999"),
    ])
    def test_run_with_various_inputs(self, nombre, correo, telefono):
        tool = SaveContacto()
        result = tool._run(nombre=nombre, correo=correo, telefono=telefono)
        assert isinstance(result, str)
