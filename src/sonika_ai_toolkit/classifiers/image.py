import base64
import mimetypes
import os
from typing import Type

from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from sonika_ai_toolkit.classifiers.text import (
    ClassificationResponse,
    _extract_tokens,
)
from sonika_ai_toolkit.utilities.types import ILanguageModel


class ImageClassifier:
    """Multimodal image classifier using vision-capable LLMs.

    Supports Gemini (all models) and OpenAI (gpt-4o, gpt-4o-mini).
    Accepts image URLs or local file paths.

    Args:
        llm: Language model instance (must support vision).
        validation_class: Pydantic model defining the output schema.
    """

    def __init__(self, llm: ILanguageModel, validation_class: Type[BaseModel]):
        self.llm = llm
        self.validation_class = validation_class
        self.structured_model = self.llm.model.with_structured_output(
            validation_class, include_raw=True
        )

    @staticmethod
    def _image_to_data_url(path: str) -> str:
        """Convert a local image file to a base64 data URL."""
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            mime_type = "image/png"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime_type};base64,{data}"

    @staticmethod
    def _is_url(source: str) -> bool:
        return source.startswith("http://") or source.startswith("https://")

    def _build_message(self, image_source: str, context: str = "") -> HumanMessage:
        if self._is_url(image_source):
            image_url = image_source
        else:
            if not os.path.isfile(image_source):
                raise FileNotFoundError(f"Image file not found: {image_source}")
            image_url = self._image_to_data_url(image_source)

        prompt = "Classify this image based on the properties defined in the validation class."
        if context:
            prompt += f"\n\nAdditional context: {context}"

        fields_desc = ", ".join(self.validation_class.model_fields.keys())
        prompt += f"\n\nExtract these properties: {fields_desc}"

        return HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        )

    def _parse_response(self, response: dict) -> ClassificationResponse:
        raw_message = response["raw"]
        parsed = response["parsed"]

        input_tokens, output_tokens = _extract_tokens(raw_message)

        if not isinstance(parsed, self.validation_class):
            raise ValueError(
                f"The response is not of type '{self.validation_class.__name__}'"
            )

        result_data = {
            field: getattr(parsed, field)
            for field in self.validation_class.model_fields.keys()
        }

        return ClassificationResponse(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            result=result_data,
        )

    def classify(self, image_source: str, context: str = "") -> ClassificationResponse:
        """Classify an image.

        Args:
            image_source: URL or local file path to the image.
            context: Optional additional context for the classification.

        Returns:
            ClassificationResponse with the classification result and token counts.
        """
        message = self._build_message(image_source, context)
        response = self.structured_model.invoke([message])
        return self._parse_response(response)

    async def aclassify(self, image_source: str, context: str = "") -> ClassificationResponse:
        """Async version of classify.

        Args:
            image_source: URL or local file path to the image.
            context: Optional additional context for the classification.

        Returns:
            ClassificationResponse with the classification result and token counts.
        """
        message = self._build_message(image_source, context)
        response = await self.structured_model.ainvoke([message])
        return self._parse_response(response)
