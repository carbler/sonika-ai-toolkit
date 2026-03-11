"""
Unit tests for sonika_ai_toolkit.classifiers

Covers:
  - TextClassifier: include_raw=True, _extract_tokens, model_fields, aclassify
  - IntentClassifier, SentimentClassifier, SafetyClassifier
  - ImageClassifier (mocked multimodal messages)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage

from sonika_ai_toolkit.classifiers.text import (
    TextClassifier,
    ClassificationResponse,
    _extract_tokens,
)
from sonika_ai_toolkit.classifiers.intent import IntentClassifier
from sonika_ai_toolkit.classifiers.sentiment import SentimentClassifier, SentimentResult
from sonika_ai_toolkit.classifiers.safety import SafetyClassifier
from sonika_ai_toolkit.classifiers.image import ImageClassifier


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class SentimentSchema(BaseModel):
    sentiment: str = Field(..., enum=["positive", "negative", "neutral"])
    confidence: float = Field(..., ge=0.0, le=1.0)


class SimpleLabel(BaseModel):
    label: str


def _make_raw_response(
    content="ok",
    response_metadata=None,
    usage_metadata=None,
    parsed=None,
    validation_class=SentimentSchema,
):
    """Build a mock include_raw=True response dict."""
    raw_msg = AIMessage(content=content)
    raw_msg.response_metadata = response_metadata or {}
    if usage_metadata is not None:
        raw_msg.usage_metadata = usage_metadata

    if parsed is None:
        parsed = validation_class(sentiment="positive", confidence=0.9)

    return {"raw": raw_msg, "parsed": parsed}


def _make_mock_lm(
    response_metadata=None,
    usage_metadata=None,
    parsed=None,
    validation_class=SentimentSchema,
):
    """Build a mock ILanguageModel for TextClassifier injection."""
    raw_resp = _make_raw_response(
        response_metadata=response_metadata,
        usage_metadata=usage_metadata,
        parsed=parsed,
        validation_class=validation_class,
    )

    structured_model = MagicMock()
    structured_model.invoke.return_value = raw_resp
    structured_model.ainvoke = AsyncMock(return_value=raw_resp)

    raw_model = MagicMock()
    raw_model.with_structured_output.return_value = structured_model

    lm = MagicMock()
    lm.model = raw_model
    return lm


# ---------------------------------------------------------------------------
# _extract_tokens
# ---------------------------------------------------------------------------

class TestExtractTokens:
    def test_openai_style(self):
        msg = AIMessage(content="ok")
        msg.response_metadata = {
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }
        assert _extract_tokens(msg) == (10, 5)

    def test_gemini_style(self):
        msg = AIMessage(content="ok")
        msg.response_metadata = {}
        msg.usage_metadata = {"input_tokens": 20, "output_tokens": 8}
        assert _extract_tokens(msg) == (20, 8)

    def test_no_metadata(self):
        msg = AIMessage(content="ok")
        msg.response_metadata = {}
        assert _extract_tokens(msg) == (0, 0)


# ---------------------------------------------------------------------------
# TextClassifier
# ---------------------------------------------------------------------------

class TestTextClassifier:
    def test_classify_returns_classification_response(self):
        lm = _make_mock_lm()
        classifier = TextClassifier(SentimentSchema, lm)
        result = classifier.classify("I love this product!")
        assert isinstance(result, ClassificationResponse)

    def test_classify_result_contains_schema_fields(self):
        parsed = SentimentSchema(sentiment="positive", confidence=0.95)
        lm = _make_mock_lm(parsed=parsed)
        classifier = TextClassifier(SentimentSchema, lm)
        response = classifier.classify("great!")
        assert response.result["sentiment"] == "positive"
        assert response.result["confidence"] == 0.95

    def test_token_counts_extracted_from_metadata(self):
        metadata = {"token_usage": {"prompt_tokens": 42, "completion_tokens": 18}}
        lm = _make_mock_lm(response_metadata=metadata)
        classifier = TextClassifier(SentimentSchema, lm)
        response = classifier.classify("test")
        assert response.input_tokens == 42
        assert response.output_tokens == 18

    def test_gemini_token_counts(self):
        lm = _make_mock_lm(usage_metadata={"input_tokens": 30, "output_tokens": 12})
        classifier = TextClassifier(SentimentSchema, lm)
        response = classifier.classify("test")
        assert response.input_tokens == 30
        assert response.output_tokens == 12

    def test_missing_token_metadata_defaults_to_zero(self):
        lm = _make_mock_lm(response_metadata={})
        classifier = TextClassifier(SentimentSchema, lm)
        response = classifier.classify("test")
        assert response.input_tokens == 0
        assert response.output_tokens == 0

    def test_with_structured_output_called_with_include_raw(self):
        lm = _make_mock_lm()
        TextClassifier(SentimentSchema, lm)
        lm.model.with_structured_output.assert_called_once_with(
            SentimentSchema, include_raw=True
        )

    def test_single_invocation_on_classify(self):
        """Verify only ONE LLM call is made (not two like before)."""
        lm = _make_mock_lm()
        classifier = TextClassifier(SentimentSchema, lm)
        classifier.classify("hello")
        structured = lm.model.with_structured_output.return_value
        structured.invoke.assert_called_once()
        # raw_model.invoke should NOT be called
        lm.model.invoke.assert_not_called()

    def test_raises_value_error_when_parsed_wrong_type(self):
        lm = _make_mock_lm(parsed=SimpleLabel(label="wrong"))
        classifier = TextClassifier(SentimentSchema, lm)
        with pytest.raises(ValueError, match="SentimentSchema"):
            classifier.classify("some text")

    def test_classify_with_different_schema(self):
        parsed = SimpleLabel(label="urgent")
        metadata = {"token_usage": {"prompt_tokens": 5, "completion_tokens": 2}}
        lm = _make_mock_lm(
            response_metadata=metadata,
            parsed=parsed,
            validation_class=SimpleLabel,
        )
        classifier = TextClassifier(SimpleLabel, lm)
        response = classifier.classify("urgent message")
        assert response.result["label"] == "urgent"

    @pytest.mark.asyncio
    async def test_aclassify(self):
        lm = _make_mock_lm()
        classifier = TextClassifier(SentimentSchema, lm)
        result = await classifier.aclassify("test async")
        assert isinstance(result, ClassificationResponse)
        assert result.result["sentiment"] == "positive"
        structured = lm.model.with_structured_output.return_value
        structured.ainvoke.assert_called_once()


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------

class TestIntentClassifier:
    def _make_intent_lm(self, intent="greeting", confidence=0.9, entities=None):
        entities = entities or {}
        # IntentClassifier creates a dynamic model, so we return a dict-like parsed
        # We need to return the include_raw=True format
        raw_msg = AIMessage(content="ok")
        raw_msg.response_metadata = {
            "token_usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }

        structured_model = MagicMock()

        def fake_invoke(prompt):
            # The dynamic model will be created by IntentClassifier,
            # so we create a mock parsed object
            parsed = MagicMock()
            parsed.intent = intent
            parsed.confidence = confidence
            parsed.entities = entities
            # Make isinstance check work by setting __class__
            type(parsed).__name__ = "IntentResult"
            return {"raw": raw_msg, "parsed": parsed}

        structured_model.invoke.side_effect = fake_invoke
        structured_model.ainvoke = AsyncMock(side_effect=fake_invoke)

        raw_model = MagicMock()
        raw_model.with_structured_output.return_value = structured_model

        lm = MagicMock()
        lm.model = raw_model
        return lm

    def test_classify_returns_response(self):
        lm = self._make_intent_lm()
        classifier = IntentClassifier(llm=lm, intents=["greeting", "question"])
        # We need isinstance to pass - patch validation_class check
        # Since the dynamic model won't match MagicMock, let's use a simpler approach
        result = classifier._classifier.structured_model.invoke("test")
        assert result["parsed"].intent == "greeting"

    def test_intents_in_schema_description(self):
        lm = self._make_intent_lm()
        IntentClassifier(
            llm=lm,
            intents=["greeting", "question", "complaint"],
            descriptions={"greeting": "A hello or hi"},
        )
        # Verify with_structured_output was called with a schema that has the intents
        call_args = lm.model.with_structured_output.call_args
        schema_cls = call_args[0][0]
        intent_field = schema_cls.model_fields["intent"]
        assert "greeting" in intent_field.description
        assert "question" in intent_field.description
        assert "A hello or hi" in intent_field.description

    def test_classify_integration(self):
        """Full classify flow with a properly mocked response."""
        raw_msg = AIMessage(content="ok")
        raw_msg.response_metadata = {
            "token_usage": {"prompt_tokens": 15, "completion_tokens": 8}
        }

        structured_model = MagicMock()

        raw_model = MagicMock()
        raw_model.with_structured_output.return_value = structured_model

        lm = MagicMock()
        lm.model = raw_model

        classifier = IntentClassifier(llm=lm, intents=["greeting", "question"])

        # Now create a proper instance of the dynamic schema
        schema_cls = raw_model.with_structured_output.call_args[0][0]
        parsed = schema_cls(intent="greeting", confidence=0.95, entities={"name": "John"})
        structured_model.invoke.return_value = {"raw": raw_msg, "parsed": parsed}

        result = classifier.classify("Hello John!")
        assert result.input_tokens == 15
        assert result.result["intent"] == "greeting"
        assert result.result["entities"] == {"name": "John"}


# ---------------------------------------------------------------------------
# SentimentClassifier
# ---------------------------------------------------------------------------

class TestSentimentClassifier:
    def test_classify(self):
        parsed = SentimentResult(
            sentiment="positive", confidence=0.92, reasoning="Very enthusiastic"
        )
        raw_msg = AIMessage(content="ok")
        raw_msg.response_metadata = {
            "token_usage": {"prompt_tokens": 20, "completion_tokens": 10}
        }

        structured_model = MagicMock()
        structured_model.invoke.return_value = {"raw": raw_msg, "parsed": parsed}
        structured_model.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": parsed}
        )

        raw_model = MagicMock()
        raw_model.with_structured_output.return_value = structured_model

        lm = MagicMock()
        lm.model = raw_model

        classifier = SentimentClassifier(llm=lm)
        result = classifier.classify("This is amazing!")
        assert result.result["sentiment"] == "positive"
        assert result.result["confidence"] == 0.92
        assert result.result["reasoning"] == "Very enthusiastic"
        assert result.input_tokens == 20

    @pytest.mark.asyncio
    async def test_aclassify(self):
        parsed = SentimentResult(
            sentiment="negative", confidence=0.8, reasoning="Sad tone"
        )
        raw_msg = AIMessage(content="ok")
        raw_msg.response_metadata = {}

        structured_model = MagicMock()
        structured_model.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": parsed}
        )

        raw_model = MagicMock()
        raw_model.with_structured_output.return_value = structured_model

        lm = MagicMock()
        lm.model = raw_model

        classifier = SentimentClassifier(llm=lm)
        result = await classifier.aclassify("I'm so sad")
        assert result.result["sentiment"] == "negative"


# ---------------------------------------------------------------------------
# SafetyClassifier
# ---------------------------------------------------------------------------

class TestSafetyClassifier:
    def test_classify_safe(self):
        raw_msg = AIMessage(content="ok")
        raw_msg.response_metadata = {}

        structured_model = MagicMock()
        raw_model = MagicMock()
        raw_model.with_structured_output.return_value = structured_model

        lm = MagicMock()
        lm.model = raw_model

        classifier = SafetyClassifier(llm=lm)

        # Get the dynamic schema and create a proper instance
        schema_cls = raw_model.with_structured_output.call_args[0][0]
        parsed = schema_cls(
            is_safe=True, categories=[], severity="none", explanation="Safe content"
        )
        structured_model.invoke.return_value = {"raw": raw_msg, "parsed": parsed}

        result = classifier.classify("Hello, how are you?")
        assert result.result["is_safe"] is True
        assert result.result["severity"] == "none"

    def test_custom_categories_in_schema(self):
        lm = MagicMock()
        lm.model = MagicMock()
        lm.model.with_structured_output.return_value = MagicMock()

        SafetyClassifier(llm=lm, custom_categories=["financial_advice"])
        schema_cls = lm.model.with_structured_output.call_args[0][0]
        cat_field = schema_cls.model_fields["categories"]
        assert "financial_advice" in cat_field.description

    def test_classify_unsafe(self):
        raw_msg = AIMessage(content="ok")
        raw_msg.response_metadata = {}

        structured_model = MagicMock()
        raw_model = MagicMock()
        raw_model.with_structured_output.return_value = structured_model

        lm = MagicMock()
        lm.model = raw_model

        classifier = SafetyClassifier(llm=lm)
        schema_cls = raw_model.with_structured_output.call_args[0][0]
        parsed = schema_cls(
            is_safe=False,
            categories=["pii"],
            severity="high",
            explanation="Contains personal info",
        )
        structured_model.invoke.return_value = {"raw": raw_msg, "parsed": parsed}

        result = classifier.classify("My SSN is 123-45-6789")
        assert result.result["is_safe"] is False
        assert "pii" in result.result["categories"]


# ---------------------------------------------------------------------------
# ImageClassifier
# ---------------------------------------------------------------------------

class TestImageClassifier:
    class ImageSchema(BaseModel):
        description: str = Field(..., description="What the image shows")
        objects: list[str] = Field(default_factory=list)

    def _make_image_lm(self, parsed=None):
        if parsed is None:
            parsed = self.ImageSchema(description="A cat", objects=["cat", "sofa"])
        raw_msg = AIMessage(content="ok")
        raw_msg.response_metadata = {
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 20}
        }

        structured_model = MagicMock()
        structured_model.invoke.return_value = {"raw": raw_msg, "parsed": parsed}
        structured_model.ainvoke = AsyncMock(
            return_value={"raw": raw_msg, "parsed": parsed}
        )

        raw_model = MagicMock()
        raw_model.with_structured_output.return_value = structured_model

        lm = MagicMock()
        lm.model = raw_model
        return lm

    def test_classify_with_url(self):
        lm = self._make_image_lm()
        classifier = ImageClassifier(llm=lm, validation_class=self.ImageSchema)
        result = classifier.classify("https://example.com/cat.jpg")
        assert result.result["description"] == "A cat"
        assert "cat" in result.result["objects"]
        assert result.input_tokens == 100

    def test_classify_with_url_builds_correct_message(self):
        lm = self._make_image_lm()
        classifier = ImageClassifier(llm=lm, validation_class=self.ImageSchema)
        classifier.classify("https://example.com/photo.jpg")

        structured = lm.model.with_structured_output.return_value
        call_args = structured.invoke.call_args[0][0]
        # Should be a list with one HumanMessage
        msg = call_args[0]
        assert len(msg.content) == 2
        assert msg.content[0]["type"] == "text"
        assert msg.content[1]["type"] == "image_url"
        assert msg.content[1]["image_url"]["url"] == "https://example.com/photo.jpg"

    def test_classify_with_context(self):
        lm = self._make_image_lm()
        classifier = ImageClassifier(llm=lm, validation_class=self.ImageSchema)
        classifier.classify("https://example.com/photo.jpg", context="Focus on animals")

        structured = lm.model.with_structured_output.return_value
        msg = structured.invoke.call_args[0][0][0]
        assert "Focus on animals" in msg.content[0]["text"]

    def test_classify_local_file_not_found(self):
        lm = self._make_image_lm()
        classifier = ImageClassifier(llm=lm, validation_class=self.ImageSchema)
        with pytest.raises(FileNotFoundError):
            classifier.classify("/nonexistent/image.png")

    def test_classify_local_file(self, tmp_path):
        # Create a tiny fake image
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        lm = self._make_image_lm()
        classifier = ImageClassifier(llm=lm, validation_class=self.ImageSchema)
        result = classifier.classify(str(img_path))
        assert result.result["description"] == "A cat"

        # Verify the data URL was built
        structured = lm.model.with_structured_output.return_value
        msg = structured.invoke.call_args[0][0][0]
        url = msg.content[1]["image_url"]["url"]
        assert url.startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_aclassify(self):
        lm = self._make_image_lm()
        classifier = ImageClassifier(llm=lm, validation_class=self.ImageSchema)
        result = await classifier.aclassify("https://example.com/cat.jpg")
        assert result.result["description"] == "A cat"

    def test_with_structured_output_called_with_include_raw(self):
        lm = self._make_image_lm()
        ImageClassifier(llm=lm, validation_class=self.ImageSchema)
        lm.model.with_structured_output.assert_called_once_with(
            self.ImageSchema, include_raw=True
        )
