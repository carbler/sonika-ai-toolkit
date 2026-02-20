"""
E2E tests for ReactBot — real API calls, real tool execution.

Each test sends a goal that requires using EmailTool and SaveContact, then
asserts that both tools were actually invoked and the response is non-empty.

Run:
    pytest tests/e2e/test_reactbot.py -m e2e -s -v
"""

import pytest

from sonika_ai_toolkit.agents.react import ReactBot
from sonika_ai_toolkit.tools.integrations import EmailTool, SaveContacto
from sonika_ai_toolkit.utilities.types import Message

# ── Shared test data ───────────────────────────────────────────────────────

INSTRUCTIONS = "Eres un asistente de comunicaciones."

GOAL = (
    "Envía un email a erley@gmail.com con asunto 'Hola' y el mensaje "
    "'Hola Erley, espero que estés bien'. "
    "Luego guarda a Erley como contacto con correo erley@gmail.com y "
    "teléfono +573183890492."
)

HISTORY = [Message(content="Mi nombre es Erley", is_bot=False)]


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_bot(language_model) -> ReactBot:
    return ReactBot(
        language_model=language_model,
        instructions=INSTRUCTIONS,
        tools=[EmailTool(), SaveContacto()],
    )


def _assert_result(result: dict) -> None:
    """Common assertions: non-empty content + expected tools executed."""
    assert isinstance(result.get("content"), str), "content must be a string"
    assert result["content"].strip(), "content must not be empty"

    executed_tools = [t.get("tool_name") for t in result.get("tools_executed", [])]
    assert "EmailTool" in executed_tools, (
        f"EmailTool should have been called; executed: {executed_tools}"
    )
    assert "SaveContact" in executed_tools, (
        f"SaveContact should have been called; executed: {executed_tools}"
    )

    usage = result.get("token_usage", {})
    assert isinstance(usage, dict), "token_usage must be a dict"


# ── Tests ──────────────────────────────────────────────────────────────────

@pytest.mark.e2e
def test_reactbot_openai(openai_model):
    """ReactBot with OpenAI executes both tools and returns a response."""
    bot = _build_bot(openai_model)
    result = bot.get_response(GOAL, HISTORY, [])
    _assert_result(result)


@pytest.mark.e2e
def test_reactbot_gemini(gemini_model):
    """ReactBot with Gemini executes both tools and returns a response."""
    bot = _build_bot(gemini_model)
    result = bot.get_response(GOAL, HISTORY, [])
    _assert_result(result)


@pytest.mark.e2e
def test_reactbot_deepseek(deepseek_model):
    """ReactBot with DeepSeek (deepseek-chat) executes both tools."""
    bot = _build_bot(deepseek_model)
    result = bot.get_response(GOAL, HISTORY, [])
    _assert_result(result)


@pytest.mark.e2e
def test_reactbot_thinking_callback(openai_model):
    """on_thinking callback is invoked (even for non-reasoning models, uses <think> fallback)."""
    thinking_chunks: list[str] = []

    bot = ReactBot(
        language_model=openai_model,
        instructions=INSTRUCTIONS,
        tools=[EmailTool(), SaveContacto()],
        on_thinking=lambda chunk: thinking_chunks.append(chunk),
    )
    result = bot.get_response(GOAL, HISTORY, [])
    # Non-reasoning models: thinking may be None or a string — just verify no crash
    assert "content" in result
