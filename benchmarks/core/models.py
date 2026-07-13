"""Model registry — turn a CLI spec string into an ILanguageModel.

A model spec is ``provider:model_name`` e.g. ``openai:gpt-4o-mini`` or
``anthropic:claude-haiku-4-5``. API keys are read from the environment (load a
.env first). Pass several specs to compare models side by side.
"""

import os

from sonika_ai_toolkit.utilities.models import (
    AnthropicLanguageModel,
    DeepSeekLanguageModel,
    GeminiLanguageModel,
    OpenAILanguageModel,
)

# provider -> (wrapper class, env var holding the API key)
_PROVIDERS = {
    "openai": (OpenAILanguageModel, "OPENAI_API_KEY"),
    "gemini": (GeminiLanguageModel, "GEMINI_API_KEY"),
    "deepseek": (DeepSeekLanguageModel, "DEEPSEEK_API_KEY"),
    "anthropic": (AnthropicLanguageModel, "ANTHROPIC_API_KEY"),
}

# Sensible defaults if the spec omits the model name (``openai`` -> gpt-4o-mini)
_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
    "deepseek": "deepseek-chat",
    "anthropic": "claude-haiku-4-5",
}


def available_providers() -> list:
    return sorted(_PROVIDERS.keys())


def build_model(spec: str):
    """Build an ILanguageModel from ``provider:model_name``.

    Raises ValueError for an unknown provider or a missing API key.
    """
    provider, _, model_name = spec.partition(":")
    provider = provider.strip().lower()
    model_name = model_name.strip() or _DEFAULT_MODEL.get(provider, "")

    if provider not in _PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Available: {available_providers()}"
        )

    cls, env_var = _PROVIDERS[provider]
    api_key = os.environ.get(env_var)
    if not api_key:
        raise ValueError(
            f"Missing API key for '{provider}': set {env_var} (e.g. in .env)."
        )

    return cls(api_key=api_key, model_name=model_name)


def normalize_spec(spec: str) -> str:
    """Canonical ``provider:model_name`` label used in reports."""
    provider, _, model_name = spec.partition(":")
    provider = provider.strip().lower()
    model_name = model_name.strip() or _DEFAULT_MODEL.get(provider, "")
    return f"{provider}:{model_name}"
