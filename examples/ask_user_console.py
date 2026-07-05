"""Reference UI for structured user questions (`ask_user`).

A minimal console "interface" that renders the questions an agent emits and
collects typed answers. It shows both flows:

  * ReactBot        — stateless: the turn ends, we re-ask with the answers.
  * OrchestratorBot — stateful: the run pauses on an interrupt and resumes.

The `render_and_collect()` function is the only UI-specific piece — swap it for
a web form, chat widget, or mobile screen. Everything else is library API.

Run:
    OPENAI_API_KEY=... python examples/ask_user_console.py
"""

import asyncio
import os

from sonika_ai_toolkit import OrchestratorBot, OpenAILanguageModel
from sonika_ai_toolkit.agents.react import ReactBot
from sonika_ai_toolkit.utilities.types import Message


# ── The UI layer (the only part you replace for web/mobile) ─────────────────

def render_and_collect(questions: list, reason: str | None = None) -> dict:
    """Render structured questions in the console and return {id: answer}."""
    if reason:
        print(f"\n🤖 {reason}")
    answers: dict = {}
    for q in questions:
        print(f"\n❓ {q['text']}")
        qtype = q.get("type", "text")
        options = q.get("options") or []

        if qtype in ("single_choice", "multi_choice") and options:
            for i, opt in enumerate(options, 1):
                print(f"   {i}) {opt['label']}")
            raw = input("   > ").strip()
            picks = [p.strip() for p in raw.split(",") if p.strip()]
            chosen = [options[int(p) - 1]["value"] for p in picks if p.isdigit()]
            answers[q["id"]] = chosen if qtype == "multi_choice" else (chosen[0] if chosen else None)

        elif qtype == "boolean":
            answers[q["id"]] = input("   (s/n) > ").strip().lower().startswith("s")

        elif qtype == "number":
            raw = input("   (número) > ").strip()
            answers[q["id"]] = float(raw) if raw else None

        else:  # text
            answers[q["id"]] = input("   > ").strip()

    return answers


# ── ReactBot flow (stateless) ───────────────────────────────────────────────

def demo_reactbot(lm) -> None:
    print("\n" + "=" * 60)
    print("ReactBot — stateless questions")
    print("=" * 60)

    bot = ReactBot(
        language_model=lm,
        instructions=(
            "Eres un asistente de reservas. Si te falta información para reservar "
            "(destino, fechas, número de personas), usa la herramienta ask_user "
            "con preguntas estructuradas y opciones cuando aplique."
        ),
        enable_user_questions=True,
    )

    history: list = []
    user_input = "Quiero reservar un viaje"

    # Loop until the agent stops asking and gives a final answer.
    for _ in range(5):
        result = bot.get_response(user_input=user_input, messages=history)
        history.append(Message(is_bot=False, content=user_input))

        if result.needs_input:
            answers = render_and_collect(result.questions)
            # Feed the answers back as the next user turn.
            history.append(Message(is_bot=True, content=result.content))
            user_input = f"Respuestas: {answers}"
            continue

        print(f"\n✅ {result.content}")
        break


# ── OrchestratorBot flow (stateful, resumes the same run) ────────────────────

async def demo_orchestrator(lm) -> None:
    print("\n" + "=" * 60)
    print("OrchestratorBot — stateful questions (interrupt/resume)")
    print("=" * 60)

    bot = OrchestratorBot(
        strong_model=lm,
        fast_model=lm,
        instructions=(
            "Eres un orquestador de reservas. Si te falta información, usa la "
            "herramienta ask_user con preguntas estructuradas antes de continuar."
        ),
        enable_user_questions=True,
    )

    thread_id = "demo-orchestrator"
    goal: str | None = "Quiero reservar un hotel"

    # Keep streaming/resuming until the run completes without an interrupt.
    for _ in range(5):
        pending_answers = None
        async for stream_mode, payload in bot.astream_events(goal, mode="ask", thread_id=thread_id):
            if stream_mode == "updates" and isinstance(payload, dict) and "__interrupt__" in payload:
                interrupt = payload["__interrupt__"][0].value
                if interrupt.get("type") == "question_request":
                    pending_answers = render_and_collect(
                        interrupt["questions"], interrupt.get("reason")
                    )

        if pending_answers is not None:
            bot.set_resume_command(pending_answers)
            goal = None  # resume the paused run
            continue

        state = bot.graph.get_state({"configurable": {"thread_id": thread_id}})
        print(f"\n✅ {state.values.get('final_report')}")
        break


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY to run this example.")
    lm = OpenAILanguageModel(api_key, model_name="gpt-4o-mini", temperature=0)

    demo_reactbot(lm)
    asyncio.run(demo_orchestrator(lm))


if __name__ == "__main__":
    main()
