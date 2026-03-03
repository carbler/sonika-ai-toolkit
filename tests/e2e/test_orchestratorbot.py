"""
E2E tests for OrchestratorBot — real API calls, real tool execution.

Each test sends a goal, then asserts:
  - content is a non-empty string
  - expected tools appear in tools_executed

Run:
    pytest tests/e2e/test_orchestratorbot.py -m e2e -s -v
"""

import pytest

from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot
from sonika_ai_toolkit.tools.integrations import EmailTool, SaveContacto

# ── Shared test data ───────────────────────────────────────────────────────

INSTRUCTIONS = (
    "Eres un asistente de comunicaciones. "
    "Usas las herramientas disponibles para enviar emails y gestionar contactos."
)

GOAL = (
    "Envía un email a erley@gmail.com con asunto 'Hola' y el mensaje "
    "'Hola Erley, espero que estés bien'. "
    "Luego guarda a Erley como contacto con correo erley@gmail.com y "
    "teléfono +573183890492."
)


# ── Helpers ────────────────────────────────────────────────────────────────

def _build_bot(language_model, memory_path: str) -> OrchestratorBot:
    return OrchestratorBot(
        strong_model=language_model,
        fast_model=language_model,
        instructions=INSTRUCTIONS,
        tools=[EmailTool(), SaveContacto()],
        memory_path=memory_path,
    )


def _assert_result(result) -> None:
    """Common assertions for OrchestratorBot results."""
    assert isinstance(result.content, str), "content must be a string"
    assert result.content.strip(), "content must not be empty"

    executed_tools = [t.get("tool_name") for t in result.tools_executed]
    assert "EmailTool" in executed_tools, (
        f"EmailTool should have been called; executed: {executed_tools}"
    )
    assert "SaveContacto" in executed_tools, (
        f"SaveContacto should have been called; executed: {executed_tools}"
    )


# ── Tests ──────────────────────────────────────────────────────────────────

@pytest.mark.e2e
def test_orchestratorbot_openai(openai_model):
    """OrchestratorBot with OpenAI executes both tools and returns a result."""
    bot = _build_bot(openai_model, "/tmp/e2e_orch_openai")
    result = bot.run(GOAL)
    _assert_result(result)


@pytest.mark.e2e
def test_orchestratorbot_gemini(gemini_model):
    """OrchestratorBot with Gemini executes both tools and returns a result."""
    bot = _build_bot(gemini_model, "/tmp/e2e_orch_gemini")
    result = bot.run(GOAL)
    _assert_result(result)


@pytest.mark.e2e
def test_orchestratorbot_deepseek(deepseek_model):
    """OrchestratorBot with DeepSeek Chat executes both tools and returns a result."""
    bot = _build_bot(deepseek_model, "/tmp/e2e_orch_deepseek")
    result = bot.run(GOAL)
    _assert_result(result)


@pytest.mark.e2e
def test_orchestratorbot_thinking_callback(openai_model):
    """on_thinking callback fires during execution (no crash expected)."""
    thinking_chunks: list[str] = []

    bot = OrchestratorBot(
        strong_model=openai_model,
        fast_model=openai_model,
        instructions=INSTRUCTIONS,
        tools=[EmailTool(), SaveContacto()],
        memory_path="/tmp/e2e_orch_thinking",
        on_thinking=lambda chunk: thinking_chunks.append(chunk),
    )
    result = bot.run(GOAL)
    assert result.content.strip(), "content must not be empty"


@pytest.mark.e2e
def test_orchestratorbot_auto_mode_no_interrupt(openai_model):
    """arun() runs in auto mode — no interrupt prompts, completes silently."""
    import asyncio
    bot = _build_bot(openai_model, "/tmp/e2e_orch_auto")
    result = asyncio.run(bot.arun(GOAL))
    assert result.content.strip(), "content must not be empty"


@pytest.mark.e2e
def test_orchestratorbot_stream_emits_messages(openai_model):
    """astream_events yields at least one 'messages' or 'updates' event."""
    import asyncio

    async def _collect():
        bot = _build_bot(openai_model, "/tmp/e2e_orch_stream")
        modes_seen = set()
        async for stream_mode, payload in bot.astream_events(GOAL, mode="auto"):
            modes_seen.add(stream_mode)
        return modes_seen

    modes = asyncio.run(_collect())
    assert modes, "stream must yield at least one event"
    assert "messages" in modes or "updates" in modes
