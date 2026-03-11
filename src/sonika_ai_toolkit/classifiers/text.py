from pydantic import BaseModel
from typing import Dict, Any, Type
from sonika_ai_toolkit.utilities.types import ILanguageModel


class ClassificationResponse(BaseModel):
    """Respuesta de clasificación con tokens utilizados"""
    input_tokens: int
    output_tokens: int
    result: Dict[str, Any]


def _extract_tokens(raw_message) -> tuple[int, int]:
    """Extract input/output token counts from an AIMessage.

    Handles both OpenAI-style (response_metadata.token_usage.prompt_tokens)
    and Gemini-style (usage_metadata.input_tokens) responses.
    """
    input_tokens = 0
    output_tokens = 0

    if hasattr(raw_message, "response_metadata"):
        meta = raw_message.response_metadata
        # OpenAI style
        token_usage = meta.get("token_usage", {})
        if token_usage:
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)

    if input_tokens == 0 and output_tokens == 0 and hasattr(raw_message, "usage_metadata"):
        usage = raw_message.usage_metadata
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0) or (usage.get("input_tokens", 0) if isinstance(usage, dict) else 0)
            output_tokens = getattr(usage, "output_tokens", 0) or (usage.get("output_tokens", 0) if isinstance(usage, dict) else 0)

    return input_tokens, output_tokens


class TextClassifier:
    def __init__(self, validation_class: Type[BaseModel], llm: ILanguageModel):
        self.llm = llm
        self.validation_class = validation_class
        self.structured_model = self.llm.model.with_structured_output(
            validation_class, include_raw=True
        )

    def _build_prompt(self, text: str) -> str:
        return f"""
        Classify the following text based on the properties defined in the validation class.

        Text: {text}

        Only extract the properties mentioned in the validation class.
        """

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

    def classify(self, text: str) -> ClassificationResponse:
        """
        Clasifica el texto según la clase de validación.

        Args:
            text: Texto a clasificar

        Returns:
            ClassificationResponse: Objeto con result, input_tokens y output_tokens
        """
        prompt = self._build_prompt(text)
        response = self.structured_model.invoke(prompt)
        return self._parse_response(response)

    async def aclassify(self, text: str) -> ClassificationResponse:
        """
        Async version of classify.

        Args:
            text: Texto a clasificar

        Returns:
            ClassificationResponse: Objeto con result, input_tokens y output_tokens
        """
        prompt = self._build_prompt(text)
        response = await self.structured_model.ainvoke(prompt)
        return self._parse_response(response)
