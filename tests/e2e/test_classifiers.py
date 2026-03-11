"""
E2E tests for classifiers — require GOOGLE_API_KEY.

Run:
    pytest tests/e2e/test_classifiers.py -m e2e -s -v
"""

import struct
import zlib

import pytest
from pydantic import BaseModel, Field

from sonika_ai_toolkit.classifiers.text import TextClassifier
from sonika_ai_toolkit.classifiers.intent import IntentClassifier
from sonika_ai_toolkit.classifiers.sentiment import SentimentClassifier
from sonika_ai_toolkit.classifiers.safety import SafetyClassifier
from sonika_ai_toolkit.classifiers.image import ImageClassifier


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TopicSchema(BaseModel):
    topic: str = Field(..., description="Main topic of the text")
    language: str = Field(..., description="Language of the text (e.g. 'en', 'es')")


class ImageAnalysis(BaseModel):
    description: str = Field(..., description="What the image shows")
    colors: list[str] = Field(default_factory=list, description="Main colors in the image")


def _create_test_png(path: str, width: int = 4, height: int = 4) -> str:
    """Create a minimal valid PNG with colored pixels (red/blue checkerboard)."""
    raw_data = b""
    for y in range(height):
        raw_data += b"\x00"  # filter byte
        for x in range(width):
            if (x + y) % 2 == 0:
                raw_data += b"\xff\x00\x00"  # red
            else:
                raw_data += b"\x00\x00\xff"  # blue
    compressed = zlib.compress(raw_data)

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr_data)
    png += _chunk(b"IDAT", compressed)
    png += _chunk(b"IEND", b"")

    with open(path, "wb") as f:
        f.write(png)
    return path


# ---------------------------------------------------------------------------
# TextClassifier
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestTextClassifierE2E:
    def test_classify_with_gemini(self, gemini_model):
        classifier = TextClassifier(validation_class=TopicSchema, llm=gemini_model)
        result = classifier.classify(
            "Python es un lenguaje de programacion muy popular para ciencia de datos."
        )
        assert result.result["topic"]
        assert result.result["language"] in ("es", "spanish", "Spanish", "español")
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        print(f"  TextClassifier: {result.result}")
        print(f"  Tokens: in={result.input_tokens}, out={result.output_tokens}")

    @pytest.mark.asyncio
    async def test_aclassify_with_gemini(self, gemini_model):
        classifier = TextClassifier(validation_class=TopicSchema, llm=gemini_model)
        result = await classifier.aclassify("Machine learning is transforming healthcare.")
        assert result.result["topic"]
        assert result.input_tokens > 0
        print(f"  TextClassifier async: {result.result}")


# ---------------------------------------------------------------------------
# SentimentClassifier
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestSentimentClassifierE2E:
    def test_positive(self, gemini_model):
        classifier = SentimentClassifier(llm=gemini_model)
        result = classifier.classify("This product is absolutely amazing! Best purchase ever!")
        assert result.result["sentiment"] in ("positive", "Positive")
        assert result.result["confidence"] > 0.5
        assert result.result["reasoning"]
        assert result.input_tokens > 0
        print(f"  Sentiment: {result.result}")

    def test_negative(self, gemini_model):
        classifier = SentimentClassifier(llm=gemini_model)
        result = classifier.classify("Terrible experience. The worst service I've ever had.")
        assert result.result["sentiment"] in ("negative", "Negative")
        print(f"  Sentiment: {result.result}")


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestIntentClassifierE2E:
    def test_classify_intent(self, gemini_model):
        classifier = IntentClassifier(
            llm=gemini_model,
            intents=["greeting", "question", "complaint", "booking"],
            descriptions={"booking": "User wants to reserve or schedule something"},
        )
        result = classifier.classify("I'd like to book a flight to Paris for next Monday")
        assert result.result["intent"] in ("booking", "question")
        assert result.result["confidence"] > 0.0
        assert result.input_tokens > 0
        print(f"  Intent: {result.result}")

    def test_greeting_intent(self, gemini_model):
        classifier = IntentClassifier(
            llm=gemini_model,
            intents=["greeting", "question", "complaint"],
        )
        result = classifier.classify("Hello! How are you doing today?")
        assert result.result["intent"] == "greeting"
        print(f"  Intent: {result.result}")


# ---------------------------------------------------------------------------
# SafetyClassifier
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestSafetyClassifierE2E:
    def test_safe_content(self, gemini_model):
        classifier = SafetyClassifier(llm=gemini_model)
        result = classifier.classify("The weather is beautiful today. Let's go for a walk!")
        assert result.result["is_safe"] is True
        assert result.result["severity"] in ("none", "low")
        assert result.input_tokens > 0
        print(f"  Safety: {result.result}")

    def test_with_custom_categories(self, gemini_model):
        classifier = SafetyClassifier(
            llm=gemini_model, custom_categories=["financial_advice"]
        )
        result = classifier.classify(
            "You should invest all your savings in this cryptocurrency right now!"
        )
        assert result.result["explanation"]
        assert result.input_tokens > 0
        print(f"  Safety (custom): {result.result}")


# ---------------------------------------------------------------------------
# ImageClassifier
# ---------------------------------------------------------------------------

@pytest.mark.e2e
class TestImageClassifierE2E:
    def test_classify_local_image(self, gemini_model, tmp_path):
        img_path = _create_test_png(str(tmp_path / "test.png"))
        classifier = ImageClassifier(
            llm=gemini_model, validation_class=ImageAnalysis
        )
        result = classifier.classify(img_path, context="Describe the colors in this image")
        assert result.result["description"]
        assert result.input_tokens > 0
        assert result.output_tokens > 0
        print(f"  Image: {result.result}")
        print(f"  Tokens: in={result.input_tokens}, out={result.output_tokens}")

    @pytest.mark.asyncio
    async def test_aclassify_local_image(self, gemini_model, tmp_path):
        img_path = _create_test_png(str(tmp_path / "test_async.png"))
        classifier = ImageClassifier(
            llm=gemini_model, validation_class=ImageAnalysis
        )
        result = await classifier.aclassify(
            img_path, context="Describe the colors in this image"
        )
        assert result.result["description"]
        assert result.input_tokens > 0
        print(f"  Image async: {result.result}")
