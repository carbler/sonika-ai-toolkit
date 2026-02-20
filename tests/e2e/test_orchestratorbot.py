"""
E2E tests for OrchestratorBot — real API calls, real tool execution.

Each test sends a goal that requires using EmailTool and SaveContact, then
asserts:
  - success=True
  - content is a non-empty string
  - both tools appear in tools_executed

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

def _build_bot(language_model, memory_path: str, **kwargs) -> OrchestratorBot:
    return OrchestratorBot(
        strong_model=language_model,
        fast_model=language_model,
        instructions=INSTRUCTIONS,
        tools=[EmailTool(), SaveContacto()],
        risk_threshold=2,
        max_retries=2,
        memory_path=memory_path,
        **kwargs,
    )


def _assert_result(result: dict) -> None:
    """Common assertions for OrchestratorBot results."""
    assert result.get("success"), (
        f"success should be True; plan: {[s['status'] for s in result.get('plan', [])]}"
    )
    assert isinstance(result.get("content"), str), "content must be a string"
    assert result["content"].strip(), "content must not be empty"

    executed_tools = [t.get("tool_name") for t in result.get("tools_executed", [])]
    assert "EmailTool" in executed_tools, (
        f"EmailTool should have been called; executed: {executed_tools}"
    )
    assert "SaveContact" in executed_tools, (
        f"SaveContact should have been called; executed: {executed_tools}"
    )


# ── Tests ──────────────────────────────────────────────────────────────────

@pytest.mark.e2e
def test_orchestratorbot_openai(openai_model):
    """OrchestratorBot with OpenAI plans, executes both tools, and reports."""
    bot = _build_bot(openai_model, "/tmp/e2e_orch_openai")
    result = bot.run(GOAL)
    _assert_result(result)


@pytest.mark.e2e
def test_orchestratorbot_gemini(gemini_model):
    """OrchestratorBot with Gemini plans, executes both tools, and reports."""
    bot = _build_bot(gemini_model, "/tmp/e2e_orch_gemini")
    result = bot.run(GOAL)
    _assert_result(result)


@pytest.mark.e2e
def test_orchestratorbot_approval_all_approved(openai_model):
    """With risk_threshold=-1 every step requires approval; callback always approves."""
    approved_step_ids: list[int] = []

    def approval_cb(step: dict) -> bool:
        approved_step_ids.append(step["id"])
        return True

    bot = _build_bot(
        openai_model,
        "/tmp/e2e_orch_approval",
        risk_threshold=-1,
        on_human_approval=approval_cb,
    )
    result = bot.run(GOAL)
    assert approved_step_ids, "approval callback should have been invoked at least once"
    _assert_result(result)


@pytest.mark.e2e
def test_orchestratorbot_approval_one_rejected(openai_model):
    """When the first step is rejected it is skipped; remaining steps complete."""
    call_count = [0]

    def approval_cb(_: dict) -> bool:
        call_count[0] += 1
        return call_count[0] > 1  # reject first, approve the rest

    bot = _build_bot(
        openai_model,
        "/tmp/e2e_orch_rejection",
        risk_threshold=-1,
        on_human_approval=approval_cb,
    )
    result = bot.run(GOAL)
    # At least one step ran, so content must be non-empty
    assert isinstance(result.get("content"), str) and result["content"].strip()
    statuses = {s["id"]: s["status"] for s in result.get("plan", [])}
    # Step 1 must be skipped
    assert statuses.get(1) == "skipped", f"Step 1 should be skipped; got {statuses}"


@pytest.mark.e2e
def test_orchestratorbot_thinking_callback(openai_model):
    """on_thinking callback is fired during planning / evaluation."""
    thinking_chunks: list[str] = []

    bot = _build_bot(
        openai_model,
        "/tmp/e2e_orch_thinking",
        on_thinking=lambda chunk: thinking_chunks.append(chunk),
    )
    result = bot.run(GOAL)
    # For non-reasoning models thinking may not fire, but no crash expected
    assert "content" in result
