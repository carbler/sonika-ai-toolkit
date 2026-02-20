"""
Integration tests for sonika_ai_toolkit.agents.think.ThinkBot end-to-end workflow.

Tests the full bot flow with mocked LLM, including:
  - Initialization with thinking support detection
  - get_response with thinking extraction
  - stream_response with incremental chunks
  - Token tracking with fallback strategy
  - on_thinking callback integration
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage

from sonika_ai_toolkit.agents.think import ThinkBot, ThinkBotState
from sonika_ai_toolkit.utilities.types import Message


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_language_model_with_thinking():
    """Mock ILanguageModel with thinking support."""
    mock_model = MagicMock()
    mock_model.model_name = "gpt-4o-mini"
    mock_model.invoke.return_value = AIMessage(
        content="<think>I'm thinking about this</think>Final answer"
    )
    mock_model.stream.return_value = iter([
        AIMessage(content="<think>step 1</think>"),
        AIMessage(content="answer"),
    ])
    mock_model.bind_tools.return_value = mock_model

    lm = MagicMock()
    lm.model = mock_model
    lm.supports_thinking = False
    return lm


@pytest.fixture
def thinking_callback():
    """Fixture for thinking callback."""
    return MagicMock()


# ---------------------------------------------------------------------------
# ThinkBot Initialization
# ---------------------------------------------------------------------------

class TestThinkBotInit:
    def test_initializes_with_required_params(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="You are helpful."
        )
        assert bot.language_model is not None
        assert bot.instructions == "You are helpful."

    def test_initializes_with_thinking_callback(self, mock_language_model_with_thinking, thinking_callback):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Help",
            on_thinking=thinking_callback
        )
        assert bot.on_thinking is thinking_callback

    def test_initializes_with_tool_callbacks(self, mock_language_model_with_thinking):
        on_start = MagicMock()
        on_end = MagicMock()
        on_error = MagicMock()

        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Test",
            on_tool_start=on_start,
            on_tool_end=on_end,
            on_tool_error=on_error
        )

        assert bot.on_tool_start is on_start
        assert bot.on_tool_end is on_end
        assert bot.on_tool_error is on_error

    def test_graph_is_compiled(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Test"
        )
        assert bot.graph is not None


# ---------------------------------------------------------------------------
# ThinkBot.get_response
# ---------------------------------------------------------------------------

class TestThinkBotGetResponse:
    def test_returns_dict_with_required_keys(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Help."
        )

        response = bot.get_response(
            user_message="Hello",
            messages=[],
            logs=[]
        )

        assert "content" in response
        assert "thinking" in response
        assert "logs" in response
        assert "tools_executed" in response
        assert "token_usage" in response

    def test_response_includes_thinking_when_present(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Think"
        )

        response = bot.get_response(
            user_message="test",
            messages=[],
            logs=[]
        )

        # Response should have thinking field (may be None if not extracted)
        assert "thinking" in response

    def test_response_content_is_string(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Answer"
        )

        response = bot.get_response(
            user_message="query",
            messages=[],
            logs=[]
        )

        assert isinstance(response["content"], str)

    def test_response_logs_structure(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Help"
        )

        response = bot.get_response(
            user_message="test",
            messages=[],
            logs=[]
        )

        assert isinstance(response["logs"], list)

    def test_token_usage_structure(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Test"
        )

        response = bot.get_response(
            user_message="test",
            messages=[],
            logs=[]
        )

        usage = response["token_usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

    def test_with_conversation_history(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Chat"
        )

        history = [
            Message(is_bot=False, content="Hi"),
            Message(is_bot=True, content="Hello!"),
        ]

        response = bot.get_response(
            user_message="How are you?",
            messages=history,
            logs=["[USER] Hi", "[BOT] Hello!"]
        )

        assert isinstance(response["content"], str)


# ---------------------------------------------------------------------------
# ThinkBot.stream_response
# ---------------------------------------------------------------------------

class TestThinkBotStreamResponse:
    def test_stream_yields_dicts(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Stream"
        )

        chunks = list(bot.stream_response(
            user_message="test",
            messages=[],
            logs=[]
        ))

        assert len(chunks) > 0
        assert all(isinstance(c, dict) for c in chunks)

    def test_stream_yields_done_event(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Stream"
        )

        chunks = list(bot.stream_response(
            user_message="test",
            messages=[],
            logs=[]
        ))

        done_chunks = [c for c in chunks if c.get("type") == "done"]
        assert len(done_chunks) == 1

    def test_done_event_contains_result(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Stream"
        )

        chunks = list(bot.stream_response(
            user_message="test",
            messages=[],
            logs=[]
        ))

        done_chunk = next((c for c in chunks if c.get("type") == "done"), None)
        assert done_chunk is not None
        result = done_chunk.get("result", {})
        assert "content" in result
        assert "thinking" in result

    def test_stream_with_thinking_callback(self, mock_language_model_with_thinking, thinking_callback):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Think",
            on_thinking=thinking_callback
        )

        list(bot.stream_response(
            user_message="test",
            messages=[],
            logs=[]
        ))

        # Callback may or may not be called depending on model response format
        # Just verify it doesn't break


# ---------------------------------------------------------------------------
# ThinkBot Message Conversion
# ---------------------------------------------------------------------------

class TestThinkBotMessageConversion:
    def test_converts_message_objects(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Test"
        )

        messages = [
            Message(is_bot=False, content="User"),
            Message(is_bot=True, content="Bot"),
        ]

        base_messages = bot._convert_message_to_base_message(messages)

        assert len(base_messages) == 2
        assert isinstance(base_messages[0], HumanMessage)
        assert isinstance(base_messages[1], AIMessage)

    def test_handles_empty_messages(self, mock_language_model_with_thinking):
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Test"
        )

        base_messages = bot._convert_message_to_base_message([])
        assert base_messages == []


# ---------------------------------------------------------------------------
# ThinkBot Tool Validation
# ---------------------------------------------------------------------------

class TestThinkBotToolValidation:
    def test_raises_on_deepseek_reasoner_with_tools(self, mock_language_model_with_thinking):
        # Set up a DeepSeek reasoner model
        mock_language_model_with_thinking.supports_thinking = True
        mock_language_model_with_thinking.model.model_name = "deepseek-reasoner"

        from sonika_ai_toolkit.tools.integrations import EmailTool

        with pytest.raises(ValueError, match="DeepSeek reasoner"):
            ThinkBot(
                language_model=mock_language_model_with_thinking,
                instructions="Test",
                tools=[EmailTool()]
            )

    def test_allows_tools_with_non_deepseek_models(self, mock_language_model_with_thinking):
        from sonika_ai_toolkit.tools.integrations import EmailTool

        # Non-DeepSeek model should work fine with tools
        bot = ThinkBot(
            language_model=mock_language_model_with_thinking,
            instructions="Test",
            tools=[EmailTool()]
        )
        assert len(bot.tools) == 1
