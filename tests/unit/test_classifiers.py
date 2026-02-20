"""
Unit tests for sonika_ai_toolkit.classifiers.text.TextClassifier

Covers:
  - Successful classification returns ClassificationResponse with correct result
  - Token counts are extracted from response_metadata
  - Missing token metadata defaults to 0
  - Validation failure raises ValueError
  - Structured output path is used for parsing
"""

import pytest
from unittest.mock import MagicMock, patch
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage

from sonika_ai_toolkit.classifiers.text import TextClassifier, ClassificationResponse


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class SentimentSchema(BaseModel):
    sentiment: str = Field(..., enum=["positive", "negative", "neutral"])
    confidence: float = Field(..., ge=0.0, le=1.0)


class SimpleLabel(BaseModel):
    label: str


def _make_mock_lm(raw_content="ok", response_metadata=None, structured_result=None, validation_class=SentimentSchema):
    """Build a mock ILanguageModel for TextClassifier injection."""
    raw_msg = AIMessage(content=raw_content)
    raw_msg.response_metadata = response_metadata or {}

    raw_model = MagicMock()
    raw_model.invoke.return_value = raw_msg

    structured_model = MagicMock()
    if structured_result is not None:
        structured_model.invoke.return_value = structured_result
    else:
        structured_model.invoke.return_value = validation_class(sentiment="positive", confidence=0.9)

    raw_model.with_structured_output.return_value = structured_model

    lm = MagicMock()
    lm.model = raw_model
    return lm


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
        structured = SentimentSchema(sentiment="positive", confidence=0.95)
        lm = _make_mock_lm(structured_result=structured)
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

    def test_missing_token_metadata_defaults_to_zero(self):
        lm = _make_mock_lm(response_metadata={})
        classifier = TextClassifier(SentimentSchema, lm)
        response = classifier.classify("test")
        assert response.input_tokens == 0
        assert response.output_tokens == 0

    def test_missing_response_metadata_attr_defaults_to_zero(self):
        raw_msg = AIMessage(content="ok")
        # Ensure no response_metadata attribute at all
        if hasattr(raw_msg, "response_metadata"):
            del raw_msg.__dict__["response_metadata"]

        raw_model = MagicMock()
        raw_model.invoke.return_value = raw_msg
        structured_model = MagicMock()
        structured_model.invoke.return_value = SentimentSchema(sentiment="neutral", confidence=0.5)
        raw_model.with_structured_output.return_value = structured_model

        lm = MagicMock()
        lm.model = raw_model

        classifier = TextClassifier(SentimentSchema, lm)
        response = classifier.classify("neutral text")
        assert response.input_tokens == 0
        assert response.output_tokens == 0

    def test_raises_value_error_when_structured_output_wrong_type(self):
        raw_model = MagicMock()
        raw_msg = AIMessage(content="ok")
        raw_msg.response_metadata = {}
        raw_model.invoke.return_value = raw_msg

        # Return wrong type from structured model
        structured_model = MagicMock()
        structured_model.invoke.return_value = SimpleLabel(label="wrong")
        raw_model.with_structured_output.return_value = structured_model

        lm = MagicMock()
        lm.model = raw_model

        classifier = TextClassifier(SentimentSchema, lm)

        with pytest.raises(ValueError, match="SentimentSchema"):
            classifier.classify("some text")

    def test_with_structured_output_called_with_validation_class(self):
        lm = _make_mock_lm()
        TextClassifier(SentimentSchema, lm)
        lm.model.with_structured_output.assert_called_once_with(SentimentSchema)

    def test_raw_model_invoke_called_on_classify(self):
        lm = _make_mock_lm()
        classifier = TextClassifier(SentimentSchema, lm)
        classifier.classify("hello")
        lm.model.invoke.assert_called_once()

    def test_structured_model_invoke_called_on_classify(self):
        lm = _make_mock_lm()
        classifier = TextClassifier(SentimentSchema, lm)
        classifier.classify("hello")
        lm.model.with_structured_output.return_value.invoke.assert_called_once()

    def test_classify_with_different_schema(self):
        raw_model = MagicMock()
        raw_msg = AIMessage(content="ok")
        raw_msg.response_metadata = {"token_usage": {"prompt_tokens": 5, "completion_tokens": 2}}
        raw_model.invoke.return_value = raw_msg

        expected = SimpleLabel(label="urgent")
        structured_model = MagicMock()
        structured_model.invoke.return_value = expected
        raw_model.with_structured_output.return_value = structured_model

        lm = MagicMock()
        lm.model = raw_model

        classifier = TextClassifier(SimpleLabel, lm)
        response = classifier.classify("urgent message")
        assert response.result["label"] == "urgent"
