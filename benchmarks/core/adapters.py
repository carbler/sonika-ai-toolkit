"""Agent adapters — run any agent behind one interface.

    OrchestratorBot run(goal, context)   (+ strong/fast model, memory_path)

Each adapter normalizes construction + invocation and returns the unified
``BotResponse`` (dict-like with content / tools_executed / token_usage), which
is the common currency the evaluator scores.
"""

import shutil
import tempfile

from benchmarks.tools.support_tools import build_tools

# Shared system instructions so all agents get the same brief — differences in
# behavior then come from the model/architecture, not the prompt.
INSTRUCTIONS = (
    "You are a customer-support agent for a digital bank. "
    "Use the available tools to look up data and perform operations before answering. "
    "Never invent account data — always call a tool to obtain it. "
    "Be concise and state clearly what action you took."
)

AGENT_KINDS = ("orchestrator",)


def _history_turns(scenario):
    """The scenario's prior conversation as {role, content} dicts for the bot."""
    return [{"role": t.role, "content": t.content} for t in scenario.history]


def run_orchestrator(model, scenario):
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
        return bot.run(goal=scenario.goal, history=_history_turns(scenario))
    finally:
        shutil.rmtree(memory_path, ignore_errors=True)


_RUNNERS = {
    "orchestrator": run_orchestrator,
}


def run_agent(agent_kind: str, model, scenario):
    """Run one scenario on one agent and return its BotResponse."""
    try:
        runner = _RUNNERS[agent_kind]
    except KeyError:
        raise ValueError(f"Unknown agent '{agent_kind}'. Available: {AGENT_KINDS}")
    return runner(model, scenario)
