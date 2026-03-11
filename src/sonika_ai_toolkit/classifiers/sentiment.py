from pydantic import BaseModel, Field
from sonika_ai_toolkit.utilities.types import ILanguageModel
from sonika_ai_toolkit.classifiers.text import TextClassifier, ClassificationResponse


class SentimentResult(BaseModel):
    """Schema for sentiment classification."""
    sentiment: str = Field(
        ...,
        description="The sentiment of the text. Must be one of: positive, negative, neutral, mixed",
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: str = Field(..., description="Brief explanation of the sentiment assessment")


class SentimentClassifier:
    """Zero-config sentiment classifier.

    Built on top of TextClassifier with a fixed SentimentResult schema.

    Args:
        llm: Language model instance.
    """

    def __init__(self, llm: ILanguageModel):
        self._classifier = TextClassifier(validation_class=SentimentResult, llm=llm)

    def classify(self, text: str) -> ClassificationResponse:
        """Classify the sentiment of the given text."""
        return self._classifier.classify(text)

    async def aclassify(self, text: str) -> ClassificationResponse:
        """Async version of classify."""
        return await self._classifier.aclassify(text)
