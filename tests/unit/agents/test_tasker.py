"""Unit tests for TaskerBot (agents.tasker).

The full Planner→Executor→Validator→Output graph is exercised e2e with real
APIs; here we unit-test the parts that don't need an LLM: construction, the
BotResponse mapping in get_response() (graph mocked), and history trimming.
"""

from unittest.mock import AsyncMock, MagicMock


from sonika_ai_toolkit.agents.tasker.tasker_bot import TaskerBot
from sonika_ai_toolkit.utilities.types import BotResponse


def _make_bot(mock_language_model, **overrides):
    kwargs = dict(
        language_model=mock_language_model,
        embeddings=MagicMock(),
        function_purpose="assist",
        personality_tone="neutral",
        limitations="none",
        dynamic_info="",
    )
    kwargs.update(overrides)
    return TaskerBot(**kwargs)


class TestConstruction:
    def test_builds_compiled_graph(self, mock_language_model):
        bot = _make_bot(mock_language_model)
        assert bot.graph is not None
        assert bot.tools == []

    def test_is_conversation_bot(self, mock_language_model):
        from sonika_ai_toolkit.agents.base import IConversationBot
        assert isinstance(_make_bot(mock_language_model), IConversationBot)


class TestGetResponse:
    def test_maps_graph_result_to_botresponse(self, mock_language_model):
        bot = _make_bot(mock_language_model)
        bot.graph = MagicMock()
        bot.graph.ainvoke = AsyncMock(return_value={
            "output_node_response": "Final answer",
            "logs": ["old", "new-1", "new-2"],
            "tools_executed": [{"tool_name": "email"}],
            "token_usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        })

        result = bot.get_response(user_input="hi", messages=[], logs=["old"])

        assert isinstance(result, BotResponse)
        assert result.content == "Final answer"
        # Only logs appended during the run are returned (existing "old" trimmed).
        assert result.logs == ["new-1", "new-2"]
        assert result.tools_executed == [{"tool_name": "email"}]


class TestHistoryLimits:
    def test_limit_messages_keeps_tail(self, mock_language_model):
        bot = _make_bot(mock_language_model, max_messages=3)
        msgs = list(range(10))
        assert bot._limit_messages(msgs) == [7, 8, 9]

    def test_limit_messages_below_cap_unchanged(self, mock_language_model):
        bot = _make_bot(mock_language_model, max_messages=5)
        assert bot._limit_messages([1, 2]) == [1, 2]

    def test_limit_logs_keeps_tail(self, mock_language_model):
        bot = _make_bot(mock_language_model, max_logs=2)
        assert bot._limit_logs(["a", "b", "c"]) == ["b", "c"]


class TestSkills:
    """Folder/programmatic skills: tool merge + planner prompt injection."""

    def _skill(self):
        from sonika_ai_toolkit.skills import Skill
        tool = MagicMock()
        tool.name = "facturar"
        return Skill(name="facturacion", instructions="Sabes facturar.", tools=[tool])

    def test_skill_tools_merged(self, mock_language_model):
        bot = _make_bot(mock_language_model, skills=[self._skill()])
        assert any(t.name == "facturar" for t in bot.tools)

    def test_skills_prompt_rendered(self, mock_language_model):
        bot = _make_bot(mock_language_model, skills=[self._skill()])
        assert "facturacion" in bot._skills_prompt
        assert "Sabes facturar." in bot._skills_prompt

    def test_no_skills_keeps_empty_tools(self, mock_language_model):
        bot = _make_bot(mock_language_model)
        assert bot.tools == []
        assert bot._skills_prompt == ""

    def test_planner_receives_extra_instructions(self, mock_language_model):
        from sonika_ai_toolkit.agents.tasker.nodes.planner_node import PlannerNode
        planner = PlannerNode(
            model=MagicMock(),
            tools=[],
            extra_instructions="## SKILLS\n### facturacion\nSabes facturar.",
        )
        assert planner.extra_instructions.startswith("## SKILLS")


class TestNodeOverrides:
    """Custom node instances replace the shipped ones; topology stays fixed."""

    def test_override_nodes_are_wired_and_run(self, mock_language_model):
        def fake_planner(state):
            return {"planner_output": {"decision": "finish", "reasoning": "ok"}}

        def fake_validator(state):
            return {"validator_output": {"status": "approved"}}

        def fake_output(state):
            return {"output_node_response": "respuesta de nodos custom"}

        def fake_logger(state):
            return {"logger_output": "done"}

        bot = _make_bot(
            mock_language_model,
            planner_node=fake_planner,
            validator_node=fake_validator,
            output_node=fake_output,
            logger_node=fake_logger,
        )
        result = bot.get_response(user_input="hola", messages=[], logs=[])
        assert result.content == "respuesta de nodos custom"

    def test_defaults_used_when_no_overrides(self, mock_language_model):
        bot = _make_bot(mock_language_model)
        assert bot._planner_node_override is None
        assert bot.graph is not None
