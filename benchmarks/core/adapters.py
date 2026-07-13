"""Agent adapters — run any agent behind one interface.

The three agents have different constructors and entrypoints:

    ReactBot        get_response(user_input, messages, logs)
    TaskerBot       get_response(user_input, messages, logs)  (+ embeddings, tone…)
    OrchestratorBot run(goal, context)                        (+ strong/fast model, memory_path)

Each adapter normalizes construction + invocation and returns the unified
``BotResponse`` (dict-like with content / tools_executed / token_usage), which
is the common currency the evaluator scores.
"""

import shutil
import tempfile

from sonika_ai_toolkit.agents.react import ReactBot
from sonika_ai_toolkit.agents.tasker.tasker_bot import TaskerBot
from sonika_ai_toolkit.utilities.types import Message

from benchmarks.tools.support_tools import build_tools

# Shared system instructions so all agents get the same brief — differences in
# behavior then come from the model/architecture, not the prompt.
INSTRUCTIONS = (
    "You are a customer-support agent for a digital bank. "
    "Use the available tools to look up data and perform operations before answering. "
    "Never invent account data — always call a tool to obtain it. "
    "Be concise and state clearly what action you took."
)

AGENT_KINDS = ("react", "tasker", "orchestrator")


def _history_messages(scenario):
    return [Message(is_bot=(t.role == "assistant"), content=t.content)
            for t in scenario.history]


def _history_context(scenario) -> str:
    if not scenario.history:
        return ""
    lines = [f"{t.role}: {t.content}" for t in scenario.history]
    return "Prior conversation:\n" + "\n".join(lines)


def run_react(model, scenario):
    bot = ReactBot(
        language_model=model,
        instructions=INSTRUCTIONS,
        tools=build_tools(scenario.tools),
    )
    return bot.get_response(
        user_input=scenario.goal,
        messages=_history_messages(scenario),
        logs=[],
    )


def run_tasker(model, scenario):
    bot = TaskerBot(
        language_model=model,
        embeddings=None,                 # not used at runtime
        function_purpose=INSTRUCTIONS,
        personality_tone="Professional and concise.",
        limitations="Only act on the current customer.",
        dynamic_info="",
        tools=build_tools(scenario.tools),
    )
    return bot.get_response(
        user_input=scenario.goal,
        messages=_history_messages(scenario),
        logs=[],
    )


def run_orchestrator(model, scenario):
    # Import here so the whole module doesn't require the orchestrator's deps
    # just to run ReactBot benchmarks.
    from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot

    memory_path = tempfile.mkdtemp(prefix="bench_orch_")
    try:
        bot = OrchestratorBot(
            strong_model=model,
            fast_model=model,
            instructions=INSTRUCTIONS,
            tools=build_tools(scenario.tools),
            memory_path=memory_path,
        )
        return bot.run(goal=scenario.goal, context=_history_context(scenario))
    finally:
        shutil.rmtree(memory_path, ignore_errors=True)


_RUNNERS = {
    "react": run_react,
    "tasker": run_tasker,
    "orchestrator": run_orchestrator,
}


def run_agent(agent_kind: str, model, scenario):
    """Run one scenario on one agent and return its BotResponse."""
    try:
        runner = _RUNNERS[agent_kind]
    except KeyError:
        raise ValueError(f"Unknown agent '{agent_kind}'. Available: {AGENT_KINDS}")
    return runner(model, scenario)
