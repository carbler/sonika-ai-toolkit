import logging
import os
from typing import Generator, Optional

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_aws import ChatBedrock
from langchain_anthropic import ChatAnthropic

from sonika_ai_toolkit.utilities.types import ILanguageModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Temperature compatibility
# ---------------------------------------------------------------------------
# Some model families only accept the provider's *default* temperature and
# return an HTTP 400 if an explicit ``temperature`` value is sent. To let every
# model work out of the box we detect those families by name and simply omit the
# parameter (the provider then uses its own default). Callers can also force
# this behaviour for any model by passing ``temperature=None``.

# OpenAI reasoning families (o1/o3/o4/… and gpt-5*) reject custom temperatures.
_OPENAI_FIXED_TEMPERATURE_PREFIXES = ("o1", "o2", "o3", "o4", "o5", "gpt-5")

# Anthropic Opus 4.7+ reject an explicit ``temperature`` (observed HTTP 400).
_ANTHROPIC_FIXED_TEMPERATURE_MARKERS = ("opus-4-7", "opus-4-8", "opus-4-9")


def _openai_omits_temperature(model_name: str) -> bool:
    """True if an OpenAI/DeepSeek-compatible model rejects custom temperature."""
    name = (model_name or "").strip().lower()
    return name.startswith(_OPENAI_FIXED_TEMPERATURE_PREFIXES)


def _anthropic_omits_temperature(model_name: str) -> bool:
    """True if an Anthropic model rejects an explicit temperature."""
    name = (model_name or "").strip().lower()
    return any(marker in name for marker in _ANTHROPIC_FIXED_TEMPERATURE_MARKERS)


def _temperature_kwargs(temperature: Optional[float], *, omit: bool, model_name: str) -> dict:
    """Build the ``temperature`` kwarg, omitting it when unsupported.

    Returns ``{}`` (no temperature sent) when the caller passed ``None`` or the
    model belongs to a fixed-temperature family; otherwise ``{"temperature": x}``.
    """
    if temperature is None:
        return {}
    if omit:
        logger.info(
            "Model %s only supports its default temperature; omitting the "
            "explicit value %s.",
            model_name,
            temperature,
        )
        return {}
    return {"temperature": temperature}


class _DeepSeekReasonerChatModel(ChatOpenAI):
    """ChatOpenAI subclass that injects DeepSeek's ``reasoning_content`` field
    into ``additional_kwargs`` so callers can read it like any other provider."""

    def _create_chat_result(self, response, generation_info=None):
        result = super()._create_chat_result(response, generation_info)
        try:
            if not isinstance(response, dict):
                response = response.model_dump()
            for i, choice in enumerate(response.get("choices", [])):
                reasoning = (choice.get("message") or {}).get("reasoning_content")
                if reasoning and i < len(result.generations):
                    result.generations[i].message.additional_kwargs["reasoning_content"] = reasoning
        except Exception:
            pass
        return result


class OpenAILanguageModel(ILanguageModel):
    """
    Clase que implementa la interfaz ILanguageModel para interactuar con los modelos de lenguaje de OpenAI.
    Proporciona funcionalidades para generar respuestas y contar tokens.
    """

    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", temperature: Optional[float] = 0.7):
        """
        Inicializa el modelo de lenguaje de OpenAI.

        Args:
            api_key (str): Clave API de OpenAI
            model_name (str): Nombre del modelo a utilizar
            temperature (float, optional): Temperatura para la generación de respuestas.
                ``None`` (o un modelo de razonamiento o1/o3/o4/gpt-5) omite el
                parámetro y deja que el proveedor use su valor por defecto.
        """
        from pydantic import SecretStr
        temp_kwargs = _temperature_kwargs(
            temperature,
            omit=_openai_omits_temperature(model_name),
            model_name=model_name,
        )
        self.model = ChatOpenAI(
            model=model_name,
            api_key=SecretStr(api_key),
            stream_usage=True,
            **temp_kwargs,
        )
        self.supports_thinking = False

    def predict(self, prompt: str) -> str:
        """
        Genera una respuesta basada en el prompt proporcionado.
        
        Args:
            prompt (str): Texto de entrada para generar la respuesta
            
        Returns:
            str: Respuesta generada por el modelo
        """
        res = self.model.predict(prompt)
        return str(res)
    
    def invoke(self, prompt: str) -> str:
        """
        Invokes the language model with a given prompt and returns the generated response.

        Args:
            prompt (str): The input text to be processed by the language model.

        Returns:
            str: The response generated by the language model based on the provided prompt.
        """
        message = HumanMessage(content=prompt)
        response = self.model.invoke([message])
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    if p.get("type") != "thinking":
                        parts.append(str(p.get("text") or p.get("content") or ""))
            return "\n".join(parts).strip()
        return str(content)

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """
        Genera una respuesta en streaming basada en el prompt proporcionado.

        Args:
            prompt (str): Texto de entrada para generar la respuesta

        Yields:
            str: Fragmentos de la respuesta generada por el modelo en tiempo real
        """
        message = HumanMessage(content=prompt)
        for chunk in self.model.stream([message]):
            content = chunk.content
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, str):
                        parts.append(p)
                    elif isinstance(p, dict):
                        if p.get("type") != "thinking":
                            parts.append(str(p.get("text") or p.get("content") or ""))
                yield "\n".join(parts)
            else:
                yield str(content)



class BedrockLanguageModel(ILanguageModel):
    """
    Clase que implementa la interfaz ILanguageModel para interactuar con los modelos de Amazon Bedrock.
    Proporciona funcionalidades para generar respuestas y contar tokens.
    """

    def __init__(self, api_key: str, region_name: str, model_name: str = "amazon.nova-micro-v1:0", temperature: float = 0.7):
        """
        Inicializa el modelo de lenguaje de Amazon Bedrock.

        Args:
            api_key (str): API Key de Amazon Bedrock (AWS_BEARER_TOKEN_BEDROCK)
            region_name (str): AWS Region (ej: us-east-1)
            model_name (str): ID del modelo en Bedrock (ej: amazon.nova-micro-v1:0)
            temperature (float): Temperatura para la generación de respuestas
        """
        # Configurar la variable de entorno necesaria para langchain-aws
        os.environ["AWS_BEARER_TOKEN_BEDROCK"] = api_key
        self.model = ChatBedrock(
            model=model_name,
            region_name=region_name,
            model_kwargs={"temperature": temperature}
        )
        self.supports_thinking = False

    def predict(self, prompt: str) -> str:
        """
        Genera una respuesta basada en el prompt proporcionado.

        Args:
            prompt (str): Texto de entrada para generar la respuesta

        Returns:
            str: Respuesta generada por el modelo
        """
        return self.model.predict(prompt)

    def invoke(self, prompt: str) -> str:
        """
        Invokes the language model with a given prompt and returns the generated response.

        Args:
            prompt (str): The input text to be processed by the language model.

        Returns:
            str: The response generated by the language model based on the provided prompt.
        """
        message = HumanMessage(content=prompt)
        response = self.model.invoke([message])
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [str(p.get("text") or p.get("content") or p if isinstance(p, dict) else p) for p in content if not (isinstance(p, dict) and p.get("type") == "thinking")]
            return "\n".join(parts).strip()
        return str(content)

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """
        Genera una respuesta en streaming basada en el prompt proporcionado.

        Args:
            prompt (str): Texto de entrada para generar la respuesta

        Yields:
            str: Fragmentos de la respuesta generada por el modelo en tiempo real
        """
        message = HumanMessage(content=prompt)
        for chunk in self.model.stream([message]):
            content = chunk.content
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                parts = [str(p.get("text") or p.get("content") or p if isinstance(p, dict) else p) for p in content if not (isinstance(p, dict) and p.get("type") == "thinking")]
                yield "\n".join(parts)
            else:
                yield str(content)


class GeminiLanguageModel(ILanguageModel):
    """
    Clase que implementa la interfaz ILanguageModel para interactuar con los modelos de lenguaje de Gemini (Google).
    Proporciona funcionalidades para generar respuestas y contar tokens.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-3-flash-preview",
        temperature: Optional[float] = 0.7,
        thinking_budget: Optional[int] = None,
    ):
        """
        Inicializa el modelo de lenguaje de Gemini.

        Args:
            api_key (str): Clave API de Google Gemini
            model_name (str): Nombre del modelo a utilizar
            temperature (float, optional): Temperatura para la generación de respuestas.
                ``None`` omite el parámetro (usa el default del proveedor).
            thinking_budget (int, optional): Presupuesto de tokens para modelos con thinking nativo.
        """
        self.supports_thinking = any(
            marker in model_name.lower()
            for marker in ["-thinking", "thinking-exp", "gemini-2.5"]
        )

        effective_temperature = temperature
        if self.supports_thinking and temperature is not None and temperature != 1:
            logger.warning(
                "Gemini thinking models require temperature=1. Overriding provided value %s.",
                temperature,
            )
            effective_temperature = 1.0

        model_kwargs: dict = {}
        if self.supports_thinking:
            budget = thinking_budget if thinking_budget is not None else 8192
            model_kwargs["thinking_budget"] = budget
            model_kwargs["include_thoughts"] = True

        model_kwargs.update(
            _temperature_kwargs(effective_temperature, omit=False, model_name=model_name)
        )
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            **model_kwargs,
        )

    def predict(self, prompt: str) -> str:
        """
        Genera una respuesta basada en el prompt proporcionado.

        Args:
            prompt (str): Texto de entrada para generar la respuesta

        Returns:
            str: Respuesta generada por el modelo
        """
        return self.model.predict(prompt)

    def invoke(self, prompt: str) -> str:
        """
        Invokes the language model with a given prompt and returns the generated response.

        Args:
            prompt (str): The input text to be processed by the language model.

        Returns:
            str: The response generated by the language model based on the provided prompt.
        """
        message = HumanMessage(content=prompt)
        response = self.model.invoke([message])
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [str(p.get("text") or p.get("content") or p if isinstance(p, dict) else p) for p in content if not (isinstance(p, dict) and p.get("type") == "thinking")]
            return "\n".join(parts).strip()
        return str(content)

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """
        Genera una respuesta en streaming basada en el prompt proporcionado.

        Args:
            prompt (str): Texto de entrada para generar la respuesta

        Yields:
            str: Fragmentos de la respuesta generada por el modelo en tiempo real
        """
        message = HumanMessage(content=prompt)
        for chunk in self.model.stream([message]):
            content = chunk.content
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                parts = [str(p.get("text") or p.get("content") or p if isinstance(p, dict) else p) for p in content if not (isinstance(p, dict) and p.get("type") == "thinking")]
                yield "\n".join(parts)
            else:
                yield str(content)





class AnthropicLanguageModel(ILanguageModel):
    """
    Clase que implementa la interfaz ILanguageModel para interactuar con los modelos
    de lenguaje Claude de Anthropic. Proporciona funcionalidades para generar respuestas
    y contar tokens.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "claude-haiku-4-5",
        temperature: Optional[float] = 0.7,
        max_tokens: int = 4096,
        thinking_budget: Optional[int] = None,
    ):
        """
        Inicializa el modelo de lenguaje de Anthropic (Claude).

        Args:
            api_key (str): Clave API de Anthropic
            model_name (str): Nombre del modelo a utilizar (ej: claude-haiku-4-5, claude-sonnet-4-6, claude-opus-4-8)
            temperature (float, optional): Temperatura para la generación de respuestas.
                ``None`` (o un modelo Opus 4.7+ que rechaza el parámetro) lo omite
                y deja que el proveedor use su valor por defecto.
            max_tokens (int): Máximo de tokens de salida (Anthropic lo exige; default razonable)
            thinking_budget (int, optional): Presupuesto de tokens para extended thinking.
                Si se especifica, activa el razonamiento extendido y fuerza temperature=1.
        """
        from pydantic import SecretStr

        self.supports_thinking = thinking_budget is not None
        effective_temperature = temperature
        model_kwargs: dict = {}

        if self.supports_thinking:
            if temperature is not None and temperature != 1:
                logger.warning(
                    "Anthropic extended thinking requires temperature=1. Overriding provided value %s.",
                    temperature,
                )
            effective_temperature = 1.0
            # max_tokens debe ser mayor que el presupuesto de thinking
            if max_tokens <= thinking_budget:
                max_tokens = thinking_budget + 1024
            model_kwargs["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}

        model_kwargs.update(
            _temperature_kwargs(
                effective_temperature,
                omit=_anthropic_omits_temperature(model_name),
                model_name=model_name,
            )
        )
        self.model = ChatAnthropic(
            model=model_name,
            max_tokens=max_tokens,
            api_key=SecretStr(api_key),
            **model_kwargs,
        )

    def predict(self, prompt: str) -> str:
        """
        Genera una respuesta basada en el prompt proporcionado.

        Args:
            prompt (str): Texto de entrada para generar la respuesta

        Returns:
            str: Respuesta generada por el modelo
        """
        return self.invoke(prompt)

    def invoke(self, prompt: str) -> str:
        """
        Invokes the language model with a given prompt and returns the generated response.

        Args:
            prompt (str): The input text to be processed by the language model.

        Returns:
            str: The response generated by the language model based on the provided prompt.
        """
        message = HumanMessage(content=prompt)
        response = self.model.invoke([message])
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [str(p.get("text") or p.get("content") or p if isinstance(p, dict) else p) for p in content if not (isinstance(p, dict) and p.get("type") == "thinking")]
            return "\n".join(parts).strip()
        return str(content)

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """
        Genera una respuesta en streaming basada en el prompt proporcionado.

        Args:
            prompt (str): Texto de entrada para generar la respuesta

        Yields:
            str: Fragmentos de la respuesta generada por el modelo en tiempo real
        """
        message = HumanMessage(content=prompt)
        for chunk in self.model.stream([message]):
            content = chunk.content
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                parts = [str(p.get("text") or p.get("content") or p if isinstance(p, dict) else p) for p in content if not (isinstance(p, dict) and p.get("type") == "thinking")]
                yield "\n".join(parts)
            else:
                yield str(content)


class DeepSeekLanguageModel(ILanguageModel):
    """
    Clase que implementa la interfaz ILanguageModel para interactuar con los modelos de lenguaje de DeepSeek.
    Proporciona funcionalidades para generar respuestas y contar tokens.
    """

    def __init__(self, api_key: str, model_name: str = "deepseek-chat", temperature: Optional[float] = 0.7):
        """
        Inicializa el modelo de lenguaje de DeepSeek.

        Args:
            api_key (str): Clave API de DeepSeek
            model_name (str): Nombre del modelo a utilizar
            temperature (float, optional): Temperatura para la generación de respuestas.
                ``None`` (o ``deepseek-reasoner``, que no admite temperatura) la omite.
        """
        from pydantic import SecretStr
        is_reasoner = model_name == "deepseek-reasoner" or "r1" in model_name.lower()
        model_class = _DeepSeekReasonerChatModel if is_reasoner else ChatOpenAI
        # El reasoner (R1) no soporta ``temperature``; lo omitimos para evitar errores.
        temp_kwargs = _temperature_kwargs(
            temperature,
            omit=is_reasoner or _openai_omits_temperature(model_name),
            model_name=model_name,
        )
        self.model = model_class(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url="https://api.deepseek.com",
            **temp_kwargs,
        )
        self.supports_thinking = is_reasoner

    def predict(self, prompt: str) -> str:
        """
        Genera una respuesta basada en el prompt proporcionado.

        Args:
            prompt (str): Texto de entrada para generar la respuesta

        Returns:
            str: Respuesta generada por el modelo
        """
        return self.model.predict(prompt)

    def invoke(self, prompt: str) -> str:
        """
        Invokes the language model with a given prompt and returns the generated response.

        Args:
            prompt (str): The input text to be processed by the language model.

        Returns:
            str: The response generated by the language model based on the provided prompt.
        """
        message = HumanMessage(content=prompt)
        response = self.model.invoke([message])
        content = response.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [str(p.get("text") or p.get("content") or p if isinstance(p, dict) else p) for p in content if not (isinstance(p, dict) and p.get("type") == "thinking")]
            return "\n".join(parts).strip()
        return str(content)

    def stream_response(self, prompt: str) -> Generator[str, None, None]:
        """
        Genera una respuesta en streaming basada en el prompt proporcionado.

        Args:
            prompt (str): Texto de entrada para generar la respuesta

        Yields:
            str: Fragmentos de la respuesta generada por el modelo en tiempo real
        """
        message = HumanMessage(content=prompt)
        for chunk in self.model.stream([message]):
            content = chunk.content
            if isinstance(content, str):
                yield content
            elif isinstance(content, list):
                parts = [str(p.get("text") or p.get("content") or p if isinstance(p, dict) else p) for p in content if not (isinstance(p, dict) and p.get("type") == "thinking")]
                yield "\n".join(parts)
            else:
                yield str(content)
