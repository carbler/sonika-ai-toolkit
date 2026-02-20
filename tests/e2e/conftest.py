"""
E2E test fixtures — require real API keys.

Run all e2e tests:
    pytest tests/e2e/ -m e2e -s -v

Run a specific provider:
    pytest tests/e2e/ -m e2e -k openai -s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL CONFIGURATION — change the defaults once here, or set
env vars to override without editing the file:

    TEST_OPENAI_MODEL    (default: gpt-4o-mini-2024-07-18)
    TEST_GEMINI_MODEL    (default: gemini-2.5-flash)
    TEST_DEEPSEEK_MODEL  (default: deepseek-chat)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os

import pytest
from dotenv import load_dotenv

from sonika_ai_toolkit.utilities.models import (
    DeepSeekLanguageModel,
    GeminiLanguageModel,
    OpenAILanguageModel,
)

load_dotenv()

# ── Model names ────────────────────────────────────────────────────────────
# Change the defaults here to test with a different model across ALL e2e tests.
OPENAI_MODEL   = os.getenv("TEST_OPENAI_MODEL",   "gpt-4o-mini-2024-07-18")
GEMINI_MODEL   = os.getenv("TEST_GEMINI_MODEL",   "gemini-2.5-flash")
DEEPSEEK_MODEL = os.getenv("TEST_DEEPSEEK_MODEL", "deepseek-chat")


def _require(env_key: str) -> str:
    """Return env var value, or skip the test if it is not set."""
    value = os.getenv(env_key)
    if not value:
        pytest.skip(f"{env_key} not set — skipping e2e test")
    return value


@pytest.fixture(scope="session")
def openai_model():
    """OpenAI language model configured for e2e tests."""
    api_key = _require("OPENAI_API_KEY")
    return OpenAILanguageModel(api_key, model_name=OPENAI_MODEL, temperature=0)


@pytest.fixture(scope="session")
def gemini_model():
    """Gemini language model configured for e2e tests."""
    api_key = _require("GOOGLE_API_KEY")
    return GeminiLanguageModel(api_key, model_name=GEMINI_MODEL, temperature=1)


@pytest.fixture(scope="session")
def deepseek_model():
    """DeepSeek language model configured for e2e tests."""
    api_key = _require("DEEPSEEK_API_KEY")
    return DeepSeekLanguageModel(api_key, model_name=DEEPSEEK_MODEL, temperature=0)
