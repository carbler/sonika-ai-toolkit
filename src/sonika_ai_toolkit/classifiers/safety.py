from pydantic import BaseModel, Field, create_model
from typing import Optional
from sonika_ai_toolkit.utilities.types import ILanguageModel
from sonika_ai_toolkit.classifiers.text import TextClassifier, ClassificationResponse


DEFAULT_CATEGORIES = [
    "hate_speech",
    "violence",
    "sexual_content",
    "self_harm",
    "pii",
    "harassment",
    "illegal_activity",
]


class SafetyResult(BaseModel):
    """Default schema for safety classification."""
    is_safe: bool = Field(..., description="Whether the text is safe")
    categories: list[str] = Field(
        default_factory=list,
        description="List of safety categories that were flagged",
    )
    severity: str = Field(
        "none",
        description="Severity level: none, low, medium, high",
    )
    explanation: str = Field(..., description="Explanation of the safety assessment")


class SafetyClassifier:
    """Content safety classifier.

    Built on top of TextClassifier with a SafetyResult schema.
    Supports custom categories in addition to the defaults.

    Args:
        llm: Language model instance.
        custom_categories: Optional list of additional safety categories to check.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        custom_categories: Optional[list[str]] = None,
    ):
        all_categories = DEFAULT_CATEGORIES + (custom_categories or [])
        categories_desc = (
            f"List of flagged safety categories from: {', '.join(all_categories)}"
        )

        DynamicSafetyResult = create_model(
            "SafetyResult",
            is_safe=(bool, Field(..., description="Whether the text is safe")),
            categories=(
                list[str],
                Field(default_factory=list, description=categories_desc),
            ),
            severity=(
                str,
                Field("none", description="Severity level: none, low, medium, high"),
            ),
            explanation=(
                str,
                Field(..., description="Explanation of the safety assessment"),
            ),
        )
        self._schema = DynamicSafetyResult
        self._classifier = TextClassifier(validation_class=DynamicSafetyResult, llm=llm)

    def classify(self, text: str) -> ClassificationResponse:
        """Assess the safety of the given text."""
        return self._classifier.classify(text)

    async def aclassify(self, text: str) -> ClassificationResponse:
        """Async version of classify."""
        return await self._classifier.aclassify(text)
