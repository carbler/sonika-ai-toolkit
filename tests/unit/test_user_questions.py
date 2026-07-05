"""Unit tests for the structured user-questions feature (ask_user).

Covers the shared contract, the ReactBot terminal-question flow, and the
BotResponse surface.  OrchestratorBot uses a native LangGraph interrupt for the
same feature; that async path is exercised in the e2e/interrupt suites.
"""

import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessageChunk

from sonika_ai_toolkit.agents.react import ReactBot
from sonika_ai_toolkit.utilities.questions import (
    ASK_USER_TOOL_NAME,
    AskUserSchema,
    questions_to_payload,
    questions_summary,
)
from sonika_ai_toolkit.tools.ask_user import AskUserQuestionTool

pytestmark = pytest.mark.unit


_ASK_ARGS = {
    "reason": "Necesito datos para continuar",
    "questions": [
        {
            "id": "color",
            "text": "¿Qué color prefieres?",
            "type": "single_choice",
            "options": [
                {"value": "r", "label": "Rojo"},
                {"value": "b", "label": "Azul"},
            ],
            "required": True,
        }
    ],
}


# ── Shared contract ────────────────────────────────────────────────────────

class TestContract:
    def test_schema_validates_and_normalizes(self):
        payload = questions_to_payload(_ASK_ARGS)
        assert payload["reason"] == _ASK_ARGS["reason"]
        assert payload["questions"][0]["id"] == "color"
        assert payload["questions"][0]["options"][0]["label"] == "Rojo"

    def test_schema_defaults(self):
        parsed = AskUserSchema.model_validate({"questions": [{"id": "n", "text": "Nombre?"}]})
        q = parsed.questions[0]
        assert q.type == "text"
        assert q.required is True
        assert q.options is None

    def test_summary_is_human_readable(self):
        text = questions_summary(questions_to_payload(_ASK_ARGS))
        assert "Necesito datos" in text
        assert "¿Qué color prefieres?" in text
        assert "Rojo" in text

    def test_tool_metadata(self):
        tool = AskUserQuestionTool()
        assert tool.name == ASK_USER_TOOL_NAME
        assert tool.risk_level == 0
        assert tool.args_schema is AskUserSchema


# ── ReactBot flow ──────────────────────────────────────────────────────────

def _model_that_asks(mock_raw_model: MagicMock) -> MagicMock:
    """Configure the mock so the first agent step calls ask_user."""
    chunk = AIMessageChunk(content="")
    chunk.tool_calls = [{"name": ASK_USER_TOOL_NAME, "args": _ASK_ARGS, "id": "call_1"}]
    mock_raw_model.stream.return_value = iter([chunk])
    return mock_raw_model


class TestReactBotAsks:
    def test_disabled_by_default_registers_no_tool(self, mock_language_model):
        bot = ReactBot(language_model=mock_language_model, instructions="x")
        assert all(t.name != ASK_USER_TOOL_NAME for t in bot.tools)

    def test_enable_registers_ask_tool(self, mock_language_model):
        bot = ReactBot(
            language_model=mock_language_model, instructions="x",
            enable_user_questions=True,
        )
        assert any(t.name == ASK_USER_TOOL_NAME for t in bot.tools)

    def test_enable_does_not_double_register(self, mock_language_model):
        bot = ReactBot(
            language_model=mock_language_model, instructions="x",
            tools=[AskUserQuestionTool()], enable_user_questions=True,
        )
        assert sum(t.name == ASK_USER_TOOL_NAME for t in bot.tools) == 1

    def test_get_response_surfaces_questions(self, mock_language_model, mock_raw_model):
        _model_that_asks(mock_raw_model)
        bot = ReactBot(
            language_model=mock_language_model, instructions="x",
            enable_user_questions=True,
        )
        result = bot.get_response(user_input="Quiero algo")

        assert result.needs_input is True
        assert len(result.questions) == 1
        assert result.questions[0]["id"] == "color"
        # No final answer yet, but content carries a readable summary.
        assert "color" in result.content.lower()
        # The ask tool must NOT have executed as a real action.
        assert all(t["tool_name"] != ASK_USER_TOOL_NAME or t.get("status") != "success"
                   for t in result.tools_executed) or not result.tools_executed

    def test_normal_answer_has_no_questions(self, mock_language_model, mock_raw_model):
        mock_raw_model.stream.return_value = iter([AIMessageChunk(content="Listo!")])
        bot = ReactBot(
            language_model=mock_language_model, instructions="x",
            enable_user_questions=True,
        )
        result = bot.get_response(user_input="Hola")
        assert result.needs_input is False
        assert result.questions == []
        assert result.content == "Listo!"

    def test_stream_emits_questions_event(self, mock_language_model, mock_raw_model):
        _model_that_asks(mock_raw_model)
        bot = ReactBot(
            language_model=mock_language_model, instructions="x",
            enable_user_questions=True,
        )
        events = list(bot.stream_response(user_message="Quiero algo", messages=[], logs=[]))
        q_events = [e for e in events if e.get("type") == "questions"]
        assert len(q_events) == 1
        assert q_events[0]["questions"][0]["id"] == "color"
        done = [e for e in events if e.get("type") == "done"]
        assert done and done[0]["result"].needs_input is True
