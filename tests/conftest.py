"""
Shared fixtures for the sonika-ai-toolkit test suite.

All external dependencies (LLMs, HTTP, filesystem) are mocked here so tests
are fast, deterministic, and CI-safe.
"""

import pytest
from typing import List, Optional
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from sonika_ai_toolkit.utilities.types import ILanguageModel, Message
from sonika_ai_toolkit.tools.integrations import EmailTool, SaveContacto


# ---------------------------------------------------------------------------
# Concrete stub for ILanguageModel so tests don't import real SDK classes
# ---------------------------------------------------------------------------

class _MockLanguageModel(ILanguageModel):
    """Lightweight ILanguageModel backed by a MagicMock."""

    def __init__(self, mock_model: MagicMock, supports_thinking: bool = False):
        self.model = mock_model
        self.supports_thinking = supports_thinking

    def predict(self, prompt: str) -> str:
        return self.model.predict(prompt)


@pytest.fixture
def mock_raw_model() -> MagicMock:
    """
    Underlying MagicMock that represents a LangChain ChatModel.

    Preconfigured so that:
      - bind_tools(tools) returns itself (model_with_tools is the same mock)
      - with_structured_output(cls) returns itself
      - invoke(messages) returns a plain 'Hello!' AIMessage
      - stream(messages) yields a single 'Hello!' AIMessageChunk
    """
    m = MagicMock(name="MockChatModel")
    m.bind_tools.return_value = m
    m.with_structured_output.return_value = m

    # Default invoke response - no tool calls
    m.invoke.return_value = AIMessage(content="Hello!")

    # Default stream response - one chunk, no tool calls
    default_chunk = AIMessageChunk(content="Hello!")
    m.stream.return_value = iter([default_chunk])

    return m


@pytest.fixture
def mock_language_model(mock_raw_model: MagicMock) -> _MockLanguageModel:
    """ILanguageModel wrapping mock_raw_model; supports_thinking=False."""
    return _MockLanguageModel(mock_raw_model, supports_thinking=False)


@pytest.fixture
def mock_thinking_language_model(mock_raw_model: MagicMock) -> _MockLanguageModel:
    """ILanguageModel wrapping mock_raw_model; supports_thinking=True (non-reasoner)."""
    lm = _MockLanguageModel(mock_raw_model, supports_thinking=True)
    # model_name attribute used by ThinkBot._is_deepseek_reasoner()
    mock_raw_model.model_name = "gemini-2.5-flash"
    return lm


@pytest.fixture
def email_tool() -> EmailTool:
    return EmailTool()


@pytest.fixture
def save_contact_tool() -> SaveContacto:
    return SaveContacto()


@pytest.fixture
def all_tools(email_tool, save_contact_tool) -> list:
    return [email_tool, save_contact_tool]


@pytest.fixture
def sample_messages() -> List[Message]:
    return [
        Message(is_bot=False, content="My name is Erley"),
        Message(is_bot=True, content="Nice to meet you, Erley!"),
    ]


@pytest.fixture
def empty_messages() -> List[Message]:
    return []


@pytest.fixture
def sample_logs() -> List[str]:
    return ["[USER] Hello", "[BOT] Hi!"]


@pytest.fixture
def empty_logs() -> List[str]:
    return []


# ---------------------------------------------------------------------------
# Pydantic models for TextClassifier tests
# ---------------------------------------------------------------------------

class SentimentLabel(BaseModel):
    """Simple classification schema for tests."""
    sentiment: str = Field(..., enum=["positive", "negative", "neutral"])
    confidence: float = Field(..., ge=0.0, le=1.0)


class LanguageLabel(BaseModel):
    language: str = Field(..., enum=["english", "spanish", "french"])


@pytest.fixture
def sentiment_model():
    return SentimentLabel


@pytest.fixture
def language_model_class():
    return LanguageLabel
