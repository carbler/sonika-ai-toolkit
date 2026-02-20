"""
Unit tests for sonika_ai_toolkit.utilities.types

Covers: Message, ResponseModel, abstract interfaces.
"""

import pytest
from abc import ABC

from sonika_ai_toolkit.utilities.types import (
    Message,
    ResponseModel,
    ILanguageModel,
    IEmbeddings,
    FileProcessorInterface,
)


# ---------------------------------------------------------------------------
# Message
# ---------------------------------------------------------------------------

class TestMessage:
    def test_create_human_message(self):
        msg = Message(is_bot=False, content="Hello")
        assert msg.is_bot is False
        assert msg.content == "Hello"

    def test_create_bot_message(self):
        msg = Message(is_bot=True, content="Hi there!")
        assert msg.is_bot is True
        assert msg.content == "Hi there!"

    def test_empty_content_allowed(self):
        msg = Message(is_bot=False, content="")
        assert msg.content == ""

    def test_multiline_content(self):
        content = "Line 1\nLine 2\nLine 3"
        msg = Message(is_bot=True, content=content)
        assert msg.content == content

    def test_unicode_content(self):
        content = "Hola 🌍 世界"
        msg = Message(is_bot=False, content=content)
        assert msg.content == content

    @pytest.mark.parametrize("is_bot,content", [
        (True, "bot reply"),
        (False, "user query"),
        (True, ""),
        (False, "a" * 10_000),
    ])
    def test_parametrized_creation(self, is_bot, content):
        msg = Message(is_bot=is_bot, content=content)
        assert msg.is_bot is is_bot
        assert msg.content == content


# ---------------------------------------------------------------------------
# ResponseModel
# ---------------------------------------------------------------------------

class TestResponseModel:
    def test_defaults_are_none(self):
        r = ResponseModel()
        assert r.user_tokens is None
        assert r.bot_tokens is None
        assert r.response is None

    def test_with_all_args(self):
        r = ResponseModel(user_tokens=10, bot_tokens=20, response="ok")
        assert r.user_tokens == 10
        assert r.bot_tokens == 20
        assert r.response == "ok"

    def test_repr_contains_values(self):
        r = ResponseModel(user_tokens=5, bot_tokens=15, response="hello")
        text = repr(r)
        assert "5" in text
        assert "15" in text
        assert "hello" in text

    def test_repr_with_none(self):
        r = ResponseModel()
        text = repr(r)
        assert "None" in text

    def test_partial_initialization(self):
        r = ResponseModel(user_tokens=100)
        assert r.user_tokens == 100
        assert r.bot_tokens is None
        assert r.response is None


# ---------------------------------------------------------------------------
# Abstract interfaces
# ---------------------------------------------------------------------------

class TestILanguageModel:
    def test_is_abstract(self):
        assert issubclass(ILanguageModel, ABC)

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ILanguageModel()  # type: ignore

    def test_concrete_subclass_must_implement_predict(self):
        class Incomplete(ILanguageModel):
            pass  # missing predict

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_works(self):
        class Concrete(ILanguageModel):
            def predict(self, prompt: str) -> str:
                return "ok"

        obj = Concrete()
        assert obj.predict("hello") == "ok"


class TestIEmbeddings:
    def test_is_abstract(self):
        assert issubclass(IEmbeddings, ABC)

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            IEmbeddings()  # type: ignore

    def test_concrete_subclass_must_implement_both_methods(self):
        class OnlyDocuments(IEmbeddings):
            def embed_documents(self, documents):
                return []
            # missing embed_query

        with pytest.raises(TypeError):
            OnlyDocuments()

    def test_concrete_subclass_works(self):
        class Concrete(IEmbeddings):
            def embed_documents(self, documents):
                return [[0.1] * 3 for _ in documents]

            def embed_query(self, query: str):
                return [0.1, 0.2, 0.3]

        obj = Concrete()
        assert obj.embed_query("test") == [0.1, 0.2, 0.3]


class TestFileProcessorInterface:
    def test_is_abstract(self):
        assert issubclass(FileProcessorInterface, ABC)

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            FileProcessorInterface()  # type: ignore
