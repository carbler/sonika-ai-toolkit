from pydantic import BaseModel, Field, create_model
from typing import Optional
from sonika_ai_toolkit.utilities.types import ILanguageModel
from sonika_ai_toolkit.classifiers.text import TextClassifier, ClassificationResponse


class IntentResult(BaseModel):
    """Default schema for intent classification."""
    intent: str = Field(..., description="The detected intent")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    entities: dict = Field(default_factory=dict, description="Extracted entities")


class IntentClassifier:
    """Classifies text into one of a set of predefined intents.

    Built on top of TextClassifier with a dynamic schema that constrains
    the ``intent`` field to the provided list of valid intents.

    Args:
        llm: Language model instance.
        intents: List of valid intent names.
        descriptions: Optional mapping of intent name → description for the LLM.
    """

    def __init__(
        self,
        llm: ILanguageModel,
        intents: list[str],
        descriptions: Optional[dict[str, str]] = None,
    ):
        self.intents = intents
        self.descriptions = descriptions or {}

        # Build a dynamic schema with intent description listing valid values
        intent_desc = f"The detected intent. Must be one of: {', '.join(intents)}"
        if self.descriptions:
            details = "; ".join(f"{k}: {v}" for k, v in self.descriptions.items())
            intent_desc += f". Descriptions: {details}"

        DynamicIntentResult = create_model(
            "IntentResult",
            intent=(str, Field(..., description=intent_desc)),
            confidence=(float, Field(..., ge=0.0, le=1.0, description="Confidence score")),
            entities=(dict, Field(default_factory=dict, description="Extracted entities")),
        )
        self._schema = DynamicIntentResult
        self._classifier = TextClassifier(validation_class=DynamicIntentResult, llm=llm)

    def classify(self, text: str) -> ClassificationResponse:
        """Classify text into one of the configured intents."""
        return self._classifier.classify(text)

    async def aclassify(self, text: str) -> ClassificationResponse:
        """Async version of classify."""
        return await self._classifier.aclassify(text)
