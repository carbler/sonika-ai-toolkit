"""
Integration tests for sonika_ai_toolkit.agents.react.ReactBot end-to-end workflow.

Tests the full bot flow with mocked LLM, including:
  - Initialization with tools
  - get_response with user message
  - Tool execution and callbacks
  - Token tracking
  - Response structure validation
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from sonika_ai_toolkit.agents.react import ReactBot, _InternalToolLogger
from sonika_ai_toolkit.utilities.types import Message


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_language_model_react():
    """Mock ILanguageModel for ReactBot tests."""
    mock_model = MagicMock()
    mock_model.invoke.return_value = AIMessage(content="I've processed your request.")
    mock_model.bind_tools.return_value = mock_model

    lm = MagicMock()
    lm.model = mock_model
    return lm


@pytest.fixture
def basic_tools():
    """Basic tools for testing."""
    from sonika_ai_toolkit.tools.integrations import EmailTool
    return [EmailTool()]


# ---------------------------------------------------------------------------
# ReactBot Initialization
# ---------------------------------------------------------------------------

class TestReactBotInit:
    def test_initializes_with_required_params(self, mock_language_model_react):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="You are helpful."
        )
        assert bot.language_model is not None
        assert bot.instructions == "You are helpful."

    def test_initializes_with_tools(self, mock_language_model_react, basic_tools):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Help me.",
            tools=basic_tools
        )
        assert len(bot.tools) == 1

    def test_initializes_with_callbacks(self, mock_language_model_react):
        on_start = MagicMock()
        on_end = MagicMock()
        on_error = MagicMock()

        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Test",
            on_tool_start=on_start,
            on_tool_end=on_end,
            on_tool_error=on_error
        )

        assert bot.on_tool_start is on_start
        assert bot.on_tool_end is on_end
        assert bot.on_tool_error is on_error

    def test_graph_is_compiled(self, mock_language_model_react):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Test"
        )
        assert bot.graph is not None


# ---------------------------------------------------------------------------
# ReactBot.get_response
# ---------------------------------------------------------------------------

class TestReactBotGetResponse:
    def test_returns_dict_with_required_keys(self, mock_language_model_react):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Help."
        )

        response = bot.get_response(
            user_input="Hello",
            messages=[],
            logs=[]
        )

        assert "content" in response
        assert "logs" in response
        assert "tools_executed" in response
        assert "token_usage" in response

    def test_response_content_is_string(self, mock_language_model_react):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Respond"
        )

        response = bot.get_response(
            user_input="test query",
            messages=[],
            logs=[]
        )

        assert isinstance(response["content"], str)

    def test_response_logs_include_user_message(self, mock_language_model_react):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Help"
        )

        response = bot.get_response(
            user_input="My question",
            messages=[],
            logs=[]
        )

        assert any("My question" in log for log in response["logs"])

    def test_token_usage_structure(self, mock_language_model_react):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Test"
        )

        response = bot.get_response(
            user_input="test",
            messages=[],
            logs=[]
        )

        usage = response["token_usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_with_conversation_history(self, mock_language_model_react):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Chat"
        )

        history = [
            Message(is_bot=False, content="Hi"),
            Message(is_bot=True, content="Hello!"),
        ]

        response = bot.get_response(
            user_input="How are you?",
            messages=history,
            logs=["[USER] Hi", "[BOT] Hello!"]
        )

        assert isinstance(response["content"], str)

    def test_with_tools_list_structure(self, mock_language_model_react, basic_tools):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Help",
            tools=basic_tools
        )

        response = bot.get_response(
            user_input="Send email",
            messages=[],
            logs=[]
        )

        assert isinstance(response["tools_executed"], list)


# ---------------------------------------------------------------------------
# ReactBot Message Conversion
# ---------------------------------------------------------------------------

class TestReactBotMessageConversion:
    def test_converts_message_objects_to_base_messages(self, mock_language_model_react):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Test"
        )

        messages = [
            Message(is_bot=False, content="User says"),
            Message(is_bot=True, content="Bot replies"),
        ]

        base_messages = bot._convert_message_to_base_message(messages)

        assert len(base_messages) == 2
        assert isinstance(base_messages[0], HumanMessage)
        assert isinstance(base_messages[1], AIMessage)

    def test_handles_empty_message_list(self, mock_language_model_react):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Test"
        )

        base_messages = bot._convert_message_to_base_message([])
        assert base_messages == []


# ---------------------------------------------------------------------------
# ReactBot Chat History Management
# ---------------------------------------------------------------------------

class TestReactBotChatHistory:
    def test_load_conversation_history(self, mock_language_model_react):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Chat"
        )

        messages = [
            Message(is_bot=False, content="Hi"),
            Message(is_bot=True, content="Hello!"),
        ]

        bot.load_conversation_history(messages)
        history = bot.get_chat_history()

        assert len(history) == 2

    def test_save_and_get_messages(self, mock_language_model_react):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Chat"
        )

        bot.save_messages("User input", "Bot response")
        history = bot.get_chat_history()

        assert len(history) == 2
        assert isinstance(history[0], HumanMessage)
        assert isinstance(history[1], AIMessage)

    def test_clear_memory(self, mock_language_model_react):
        bot = ReactBot(
            language_model=mock_language_model_react,
            instructions="Chat"
        )

        bot.save_messages("msg", "resp")
        bot.clear_memory()

        assert len(bot.get_chat_history()) == 0
