"""
Unit tests for OrchestratorBot prompt injection.

Covers:
  - OrchestratorPrompts defaults match module-level constants
  - Partial overrides: only the modified field changes
  - Full override: all fields replaced
  - Nodes receive the injected templates (no LLM calls needed)
  - OrchestratorBot wires prompts through to every node
"""

import pytest
from unittest.mock import MagicMock

from sonika_ai_toolkit.agents.orchestrator.prompts import (
    OrchestratorPrompts,
    PROMPT_A,
    PLANNER_PROMPT,
    EVALUATOR_PROMPT,
    RETRY_PROMPT,
    REPORTER_PROMPT,
    SAVE_MEMORY_PROMPT,
)
from sonika_ai_toolkit.agents.orchestrator.nodes.planner import PlannerNode
from sonika_ai_toolkit.agents.orchestrator.nodes.evaluator import EvaluatorNode
from sonika_ai_toolkit.agents.orchestrator.nodes.retry import RetryNode
from sonika_ai_toolkit.agents.orchestrator.nodes.reporter import ReporterNode
from sonika_ai_toolkit.agents.orchestrator.nodes.save_memory import SaveMemoryNode
from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_lm(model_name: str = "gpt-4o-mini") -> MagicMock:
    lm = MagicMock()
    lm.model = MagicMock()
    lm.model.model_name = model_name
    lm.model.with_structured_output.return_value = lm.model
    lm.model_name = model_name
    return lm


def _make_bot(prompts=None):
    """Build an OrchestratorBot with fully mocked models (no API calls)."""
    return OrchestratorBot(
        strong_model=_mock_lm("gpt-4o"),
        fast_model=_mock_lm("gpt-4o-mini"),
        instructions="Test instructions",
        tools=[],
        memory_path="/tmp/test_memory",
        prompts=prompts,
    )


# ---------------------------------------------------------------------------
# OrchestratorPrompts — default values
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorPromptsDefaults:
    def test_core_default(self):
        p = OrchestratorPrompts()
        assert p.core == PROMPT_A

    def test_planner_default(self):
        p = OrchestratorPrompts()
        assert p.planner == PLANNER_PROMPT

    def test_evaluator_default(self):
        p = OrchestratorPrompts()
        assert p.evaluator == EVALUATOR_PROMPT

    def test_retry_default(self):
        p = OrchestratorPrompts()
        assert p.retry == RETRY_PROMPT

    def test_reporter_default(self):
        p = OrchestratorPrompts()
        assert p.reporter == REPORTER_PROMPT

    def test_save_memory_default(self):
        p = OrchestratorPrompts()
        assert p.save_memory == SAVE_MEMORY_PROMPT

    def test_two_instances_are_independent(self):
        """Each instance gets a fresh copy — no shared mutable state."""
        p1 = OrchestratorPrompts()
        p2 = OrchestratorPrompts()
        p1.core = "overridden"
        assert p2.core == PROMPT_A


# ---------------------------------------------------------------------------
# OrchestratorPrompts — partial overrides
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorPromptsOverride:
    def test_override_core_only(self):
        custom_core = "You are a financial compliance bot."
        p = OrchestratorPrompts(core=custom_core)
        assert p.core == custom_core
        # Other fields keep their defaults
        assert p.planner == PLANNER_PROMPT
        assert p.evaluator == EVALUATOR_PROMPT

    def test_override_planner_only(self):
        custom_planner = "{prompt_a}\nCustom planner: {goal}\n{tool_descriptions}\n{memory_context}\n{context}\n{instructions}"
        p = OrchestratorPrompts(planner=custom_planner)
        assert p.planner == custom_planner
        assert p.core == PROMPT_A

    def test_override_multiple_fields(self):
        p = OrchestratorPrompts(
            core="Custom core.",
            reporter="Custom reporter: {prompt_a} {goal} {plan_summary} {tool_outputs}",
        )
        assert p.core == "Custom core."
        assert "Custom reporter" in p.reporter
        assert p.planner == PLANNER_PROMPT  # unchanged

    def test_override_all_fields(self):
        custom = OrchestratorPrompts(
            core="core",
            planner="planner {prompt_a} {instructions} {tool_descriptions} {memory_context} {goal} {context}",
            evaluator="evaluator {prompt_a} {goal} {step_description} {tool_output} {plan_summary}",
            retry="retry {prompt_a} {goal} {step_description} {error} {tool_descriptions} {retry_history}",
            reporter="reporter {prompt_a} {goal} {plan_summary} {tool_outputs}",
            save_memory="save {prompt_a} {goal} {plan_summary}",
        )
        assert custom.core == "core"
        assert custom.planner.startswith("planner")
        assert custom.evaluator.startswith("evaluator")
        assert custom.retry.startswith("retry")
        assert custom.reporter.startswith("reporter")
        assert custom.save_memory.startswith("save")


# ---------------------------------------------------------------------------
# Node-level: prompt_template and core_prompt are stored and used
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNodePromptInjection:
    def test_planner_node_stores_default_templates(self):
        node = PlannerNode(
            strong_model=_mock_lm(),
            tool_registry=MagicMock(),
            memory_manager=MagicMock(),
            instructions="",
        )
        assert node.prompt_template == PLANNER_PROMPT
        assert node.core_prompt == PROMPT_A

    def test_planner_node_stores_custom_templates(self):
        custom_tmpl = "{prompt_a}\nGoal: {goal}\n{instructions}\n{tool_descriptions}\n{memory_context}\n{context}"
        custom_core = "Custom rules."
        node = PlannerNode(
            strong_model=_mock_lm(),
            tool_registry=MagicMock(),
            memory_manager=MagicMock(),
            instructions="",
            prompt_template=custom_tmpl,
            core_prompt=custom_core,
        )
        assert node.prompt_template == custom_tmpl
        assert node.core_prompt == custom_core

    def test_evaluator_node_stores_custom_templates(self):
        custom_tmpl = "Evaluate: {prompt_a} {goal} {step_description} {tool_output} {plan_summary}"
        node = EvaluatorNode(
            fast_model=_mock_lm(),
            prompt_template=custom_tmpl,
            core_prompt="Custom core.",
        )
        assert node.prompt_template == custom_tmpl
        assert node.core_prompt == "Custom core."

    def test_retry_node_stores_custom_templates(self):
        custom_tmpl = "Retry: {prompt_a} {goal} {step_description} {error} {tool_descriptions} {retry_history}"
        node = RetryNode(
            fast_model=_mock_lm(),
            tool_registry=MagicMock(),
            prompt_template=custom_tmpl,
        )
        assert node.prompt_template == custom_tmpl
        assert node.core_prompt == PROMPT_A  # default kept

    def test_reporter_node_stores_custom_templates(self):
        custom_tmpl = "Report: {prompt_a} {goal} {plan_summary} {tool_outputs}"
        node = ReporterNode(
            fast_model=_mock_lm(),
            prompt_template=custom_tmpl,
            core_prompt="My core.",
        )
        assert node.prompt_template == custom_tmpl
        assert node.core_prompt == "My core."

    def test_save_memory_node_stores_custom_templates(self):
        custom_tmpl = "Save: {prompt_a} {goal} {plan_summary}"
        node = SaveMemoryNode(
            fast_model=_mock_lm(),
            memory_manager=MagicMock(),
            prompt_template=custom_tmpl,
        )
        assert node.prompt_template == custom_tmpl


# ---------------------------------------------------------------------------
# OrchestratorBot wiring — prompts reach the internal graph nodes
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorBotPromptWiring:
    def test_default_prompts_when_none_passed(self):
        bot = _make_bot(prompts=None)
        assert isinstance(bot.prompts, OrchestratorPrompts)
        assert bot.prompts.core == PROMPT_A
        assert bot.prompts.planner == PLANNER_PROMPT

    def test_custom_prompts_stored_on_bot(self):
        custom = OrchestratorPrompts(core="My custom core.")
        bot = _make_bot(prompts=custom)
        assert bot.prompts.core == "My custom core."
        assert bot.prompts.planner == PLANNER_PROMPT

    def test_custom_core_reaches_planner_node(self):
        """
        OrchestratorBot stores node references in self._nodes so the injected
        templates can be inspected without relying on LangGraph internals.
        """
        custom = OrchestratorPrompts(core="Domain-specific core rules.")
        bot = _make_bot(prompts=custom)

        planner_node = bot._nodes["planner"]
        assert planner_node.core_prompt == "Domain-specific core rules."
        assert planner_node.prompt_template == PLANNER_PROMPT  # not overridden

    def test_custom_planner_template_reaches_node(self):
        custom_planner = (
            "{prompt_a}\n## Tools\n{tool_descriptions}\n"
            "## Objective\n{goal}\n{instructions}\n{memory_context}\n{context}\n"
            'Return JSON: {{"steps": []}}'
        )
        custom = OrchestratorPrompts(planner=custom_planner)
        bot = _make_bot(prompts=custom)

        assert bot._nodes["planner"].prompt_template == custom_planner

    def test_custom_reporter_template_reaches_node(self):
        custom_reporter = "Custom report: {prompt_a} {goal} {plan_summary} {tool_outputs}"
        custom = OrchestratorPrompts(reporter=custom_reporter)
        bot = _make_bot(prompts=custom)

        assert bot._nodes["reporter"].prompt_template == custom_reporter

    def test_custom_evaluator_template_reaches_node(self):
        custom_ev = "Eval: {prompt_a} {goal} {step_description} {tool_output} {plan_summary}"
        custom = OrchestratorPrompts(evaluator=custom_ev)
        bot = _make_bot(prompts=custom)

        assert bot._nodes["evaluator"].prompt_template == custom_ev

    def test_custom_retry_template_reaches_node(self):
        custom_retry = (
            "Retry: {prompt_a} {goal} {step_description} "
            "{error} {tool_descriptions} {retry_history}"
        )
        custom = OrchestratorPrompts(retry=custom_retry)
        bot = _make_bot(prompts=custom)

        assert bot._nodes["retry"].prompt_template == custom_retry

    def test_custom_save_memory_template_reaches_node(self):
        custom_sm = "Save: {prompt_a} {goal} {plan_summary}"
        custom = OrchestratorPrompts(save_memory=custom_sm)
        bot = _make_bot(prompts=custom)

        assert bot._nodes["save_memory"].prompt_template == custom_sm

    def test_all_nodes_share_same_core_prompt(self):
        """Overriding core propagates to every LLM node."""
        shared_core = "Shared domain rules for all stages."
        custom = OrchestratorPrompts(core=shared_core)
        bot = _make_bot(prompts=custom)

        for node_name, node in bot._nodes.items():
            assert node.core_prompt == shared_core, (
                f"Node '{node_name}' has core_prompt={node.core_prompt!r}, expected {shared_core!r}"
            )
