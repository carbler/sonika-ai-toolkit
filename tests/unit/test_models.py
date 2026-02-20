"""
Unit tests for sonika_ai_toolkit.utilities.models

All LLM SDK constructors and network calls are patched so no real API keys
or HTTP requests are needed.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, call
from langchain_core.messages import AIMessage

from sonika_ai_toolkit.utilities.models import (
    OpenAILanguageModel,
    DeepSeekLanguageModel,
    GeminiLanguageModel,
    BedrockLanguageModel,
    _DeepSeekReasonerChatModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_chat_openai():
    return patch("sonika_ai_toolkit.utilities.models.ChatOpenAI", autospec=True)

def _patch_gemini():
    return patch("sonika_ai_toolkit.utilities.models.ChatGoogleGenerativeAI", autospec=True)

def _patch_bedrock():
    return patch("sonika_ai_toolkit.utilities.models.ChatBedrock", autospec=True)


# ---------------------------------------------------------------------------
# OpenAILanguageModel
# ---------------------------------------------------------------------------

class TestOpenAILanguageModel:
    def test_init_sets_supports_thinking_false(self):
        with _patch_chat_openai():
            model = OpenAILanguageModel(api_key="key")
            assert model.supports_thinking is False

    def test_init_default_model_name(self):
        with _patch_chat_openai() as MockChatOpenAI:
            OpenAILanguageModel(api_key="key")
            _, kwargs = MockChatOpenAI.call_args
            assert kwargs.get("model_name") == "gpt-4o-mini"

    def test_init_custom_model_name(self):
        with _patch_chat_openai() as MockChatOpenAI:
            OpenAILanguageModel(api_key="key", model_name="gpt-4o")
            _, kwargs = MockChatOpenAI.call_args
            assert kwargs.get("model_name") == "gpt-4o"

    def test_init_passes_temperature(self):
        with _patch_chat_openai() as MockChatOpenAI:
            OpenAILanguageModel(api_key="key", temperature=0.3)
            _, kwargs = MockChatOpenAI.call_args
            assert kwargs.get("temperature") == 0.3

    def test_init_stream_usage_enabled(self):
        with _patch_chat_openai() as MockChatOpenAI:
            OpenAILanguageModel(api_key="key")
            _, kwargs = MockChatOpenAI.call_args
            assert kwargs.get("stream_usage") is True

    def test_predict_delegates_to_model(self):
        with _patch_chat_openai():
            lm = OpenAILanguageModel(api_key="key")
            lm.model = MagicMock()
            lm.model.predict.return_value = "predicted"
            result = lm.predict("hello")
            lm.model.predict.assert_called_once_with("hello")
            assert result == "predicted"

    def test_invoke_delegates_to_model(self):
        with _patch_chat_openai():
            lm = OpenAILanguageModel(api_key="key")
            lm.model = MagicMock()
            lm.model.invoke.return_value = AIMessage(content="invoked")
            result = lm.invoke("hello")
            assert result == "invoked"

    def test_stream_response_yields_chunks(self):
        with _patch_chat_openai():
            lm = OpenAILanguageModel(api_key="key")
            lm.model = MagicMock()
            lm.model.stream.return_value = iter([
                MagicMock(content="chunk1"),
                MagicMock(content="chunk2"),
            ])
            chunks = list(lm.stream_response("hello"))
            assert chunks == ["chunk1", "chunk2"]


# ---------------------------------------------------------------------------
# DeepSeekLanguageModel
# ---------------------------------------------------------------------------

class TestDeepSeekLanguageModel:
    @pytest.mark.parametrize("model_name,expected", [
        ("deepseek-chat", False),
        ("deepseek-reasoner", True),
        ("deepseek-r1-lite", True),
        ("my-r1-model", True),
        ("deepseek-coder", False),
    ])
    def test_supports_thinking_detection(self, model_name, expected):
        with _patch_chat_openai():
            lm = DeepSeekLanguageModel(api_key="key", model_name=model_name)
            assert lm.supports_thinking is expected

    def test_reasoner_uses_custom_class(self):
        with patch("sonika_ai_toolkit.utilities.models._DeepSeekReasonerChatModel") as MockReasoner, \
             patch("sonika_ai_toolkit.utilities.models.ChatOpenAI"):
            DeepSeekLanguageModel(api_key="key", model_name="deepseek-reasoner")
            MockReasoner.assert_called_once()

    def test_non_reasoner_uses_chat_openai(self):
        with patch("sonika_ai_toolkit.utilities.models.ChatOpenAI") as MockChat, \
             patch("sonika_ai_toolkit.utilities.models._DeepSeekReasonerChatModel") as MockReasoner:
            DeepSeekLanguageModel(api_key="key", model_name="deepseek-chat")
            MockChat.assert_called_once()
            MockReasoner.assert_not_called()

    def test_base_url_is_deepseek(self):
        with _patch_chat_openai() as MockChat:
            DeepSeekLanguageModel(api_key="key", model_name="deepseek-chat")
            _, kwargs = MockChat.call_args
            assert kwargs.get("base_url") == "https://api.deepseek.com"


# ---------------------------------------------------------------------------
# GeminiLanguageModel
# ---------------------------------------------------------------------------

class TestGeminiLanguageModel:
    @pytest.mark.parametrize("model_name,expected", [
        ("gemini-2.5-flash", True),
        ("gemini-2.5-pro", True),
        ("gemini-2.0-flash-thinking-exp", True),
        ("gemini-pro-preview", True),
        ("gemini-3-flash-preview", False),  # "flash" alone no longer triggers it
        ("gemini-1.0-pro", False),
    ])
    def test_supports_thinking_detection(self, model_name, expected):
        with _patch_gemini():
            lm = GeminiLanguageModel(api_key="key", model_name=model_name)
            assert lm.supports_thinking is expected

    def test_thinking_model_forces_temperature_one(self, caplog):
        import logging
        with _patch_gemini() as MockGemini:
            with caplog.at_level(logging.WARNING):
                lm = GeminiLanguageModel(
                    api_key="key",
                    model_name="gemini-2.5-flash",
                    temperature=0.5,
                )
            _, kwargs = MockGemini.call_args
            assert kwargs.get("temperature") == 1.0

    def test_non_thinking_model_uses_provided_temperature(self):
        with _patch_gemini() as MockGemini:
            GeminiLanguageModel(api_key="key", model_name="gemini-1.0-pro", temperature=0.3)
            _, kwargs = MockGemini.call_args
            assert kwargs.get("temperature") == 0.3

    def test_thinking_model_passes_thinking_budget(self):
        with _patch_gemini() as MockGemini:
            GeminiLanguageModel(
                api_key="key",
                model_name="gemini-2.5-flash",
                thinking_budget=4096,
            )
            _, kwargs = MockGemini.call_args
            assert kwargs.get("thinking_budget") == 4096

    def test_thinking_model_default_budget_8192(self):
        with _patch_gemini() as MockGemini:
            GeminiLanguageModel(api_key="key", model_name="gemini-2.5-flash")
            _, kwargs = MockGemini.call_args
            assert kwargs.get("thinking_budget") == 8192

    def test_thinking_model_passes_include_thoughts(self):
        with _patch_gemini() as MockGemini:
            GeminiLanguageModel(api_key="key", model_name="gemini-2.5-flash")
            _, kwargs = MockGemini.call_args
            assert kwargs.get("include_thoughts") is True

    def test_non_thinking_model_no_thinking_kwargs(self):
        with _patch_gemini() as MockGemini:
            GeminiLanguageModel(api_key="key", model_name="gemini-1.0-pro")
            _, kwargs = MockGemini.call_args
            assert "thinking_budget" not in kwargs
            assert "include_thoughts" not in kwargs


# ---------------------------------------------------------------------------
# BedrockLanguageModel
# ---------------------------------------------------------------------------

class TestBedrockLanguageModel:
    def test_init_sets_supports_thinking_false(self):
        with _patch_bedrock():
            lm = BedrockLanguageModel(api_key="token", region_name="us-east-1")
            assert lm.supports_thinking is False

    def test_init_sets_env_variable(self):
        original = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        try:
            with _patch_bedrock():
                BedrockLanguageModel(api_key="my-bearer-token", region_name="us-east-1")
            assert os.environ["AWS_BEARER_TOKEN_BEDROCK"] == "my-bearer-token"
        finally:
            if original is None:
                os.environ.pop("AWS_BEARER_TOKEN_BEDROCK", None)
            else:
                os.environ["AWS_BEARER_TOKEN_BEDROCK"] = original

    def test_init_passes_region(self):
        with _patch_bedrock() as MockBedrock:
            BedrockLanguageModel(api_key="tok", region_name="eu-west-1")
            _, kwargs = MockBedrock.call_args
            assert kwargs.get("region_name") == "eu-west-1"

    def test_init_default_model(self):
        with _patch_bedrock() as MockBedrock:
            BedrockLanguageModel(api_key="tok", region_name="us-east-1")
            _, kwargs = MockBedrock.call_args
            assert kwargs.get("model_id") == "amazon.nova-micro-v1:0"


# ---------------------------------------------------------------------------
# _DeepSeekReasonerChatModel
# ---------------------------------------------------------------------------

class TestDeepSeekReasonerChatModel:
    def _make_response_dict(self, content: str, reasoning: str):
        return {
            "choices": [
                {
                    "message": {
                        "content": content,
                        "reasoning_content": reasoning,
                    }
                }
            ]
        }

    def test_create_chat_result_injects_reasoning_content(self):
        """reasoning_content must end up in additional_kwargs of the first generation."""
        model = _DeepSeekReasonerChatModel.__new__(_DeepSeekReasonerChatModel)

        # Build a minimal fake result from parent
        from langchain_core.outputs import ChatGeneration
        ai_msg = AIMessage(content="answer", additional_kwargs={})
        gen = ChatGeneration(message=ai_msg)

        # Use MagicMock instead of real ChatResult to avoid Pydantic validation
        fake_result = MagicMock()
        # The implementation expects generations to be a flat list of generations
        fake_result.generations = [gen]

        with patch.object(
            _DeepSeekReasonerChatModel.__bases__[0],  # ChatOpenAI
            "_create_chat_result",
            return_value=fake_result,
        ):
            response_dict = self._make_response_dict("answer", "I thought about it")
            result = model._create_chat_result(response_dict)

        reasoning = result.generations[0].message.additional_kwargs.get("reasoning_content")
        assert reasoning == "I thought about it"

    def test_create_chat_result_no_reasoning_content_is_safe(self):
        """Missing reasoning_content must not raise."""
        model = _DeepSeekReasonerChatModel.__new__(_DeepSeekReasonerChatModel)

        from langchain_core.outputs import ChatGeneration
        ai_msg = AIMessage(content="answer", additional_kwargs={})
        gen = ChatGeneration(message=ai_msg)

        # Use MagicMock instead of real ChatResult to avoid Pydantic validation
        fake_result = MagicMock()
        fake_result.generations = [gen]

        with patch.object(
            _DeepSeekReasonerChatModel.__bases__[0],
            "_create_chat_result",
            return_value=fake_result,
        ):
            response_dict = {"choices": [{"message": {"content": "answer"}}]}
            result = model._create_chat_result(response_dict)

        assert result.generations[0].message.additional_kwargs.get("reasoning_content") is None
