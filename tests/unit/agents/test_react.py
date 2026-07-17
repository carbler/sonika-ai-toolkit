"""
Unit tests for sonika_ai_toolkit.agents.react.

Covers:
  _InternalToolLogger
    - on_llm_start: appends "[AGENT] Thinking..." log
    - on_llm_end: logs tool call names or "Generated response"
    - on_tool_start/end/error: execution records + user callbacks
    - Callback resilience: exceptions in user callbacks are swallowed
  ReactBot ask_user flow
    - enable_user_questions registration and terminal-question surfacing in
      BotResponse (needs_input / questions) and in the stream ("questions" event)
"""

import re as _re

from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration
from langchain_core.tools import tool as _lc_tool

from sonika_ai_toolkit.agents.react import ReactBot, _InternalToolLogger
from sonika_ai_toolkit.tools.ask_user import AskUserQuestionTool
from sonika_ai_toolkit.utilities.questions import ASK_USER_TOOL_NAME


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logger(**kwargs) -> _InternalToolLogger:
    return _InternalToolLogger(**kwargs)


def _make_llm_end_response(tool_names=None):
    """Build a minimal LLM response object, optionally with tool calls."""
    msg = AIMessage(content="response")
    if tool_names:
        msg.tool_calls = [{"name": n, "args": {}, "id": f"id_{i}"} for i, n in enumerate(tool_names)]

    # Return a MagicMock that mimics ChatResult structure
    response = MagicMock()
    response.generations = [[ChatGeneration(message=msg)]]
    return response


# ---------------------------------------------------------------------------
# on_llm_start
# ---------------------------------------------------------------------------

class TestOnLlmStart:
    def test_appends_thinking_log(self):
        logger = _make_logger()
        logger.on_llm_start({}, ["prompt"])
        assert "[AGENT] Thinking..." in logger.execution_logs

    def test_multiple_calls_append_multiple_logs(self):
        logger = _make_logger()
        logger.on_llm_start({}, ["p1"])
        logger.on_llm_start({}, ["p2"])
        thinking_logs = [line for line in logger.execution_logs if "Thinking" in line]
        assert len(thinking_logs) == 2


# ---------------------------------------------------------------------------
# on_llm_end
# ---------------------------------------------------------------------------

class TestOnLlmEnd:
    def test_logs_generated_response_when_no_tool_calls(self):
        logger = _make_logger()
        response = _make_llm_end_response()
        logger.on_llm_end(response)
        assert "[AGENT] Generated response" in logger.execution_logs

    def test_logs_tool_names_when_tool_calls_present(self):
        logger = _make_logger()
        response = _make_llm_end_response(tool_names=["email_tool", "save_contact"])
        logger.on_llm_end(response)
        log_entry = next((line for line in logger.execution_logs if "Decided to call tools" in line), None)
        assert log_entry is not None
        assert "email_tool" in log_entry
        assert "save_contact" in log_entry

    def test_handles_response_without_generations(self):
        logger = _make_logger()
        logger.on_llm_end(MagicMock(spec=[]))  # no .generations attr
        # Should not raise; still logs something or does nothing


# ---------------------------------------------------------------------------
# on_tool_start
# ---------------------------------------------------------------------------

class TestOnToolStart:
    def test_appends_execution_record(self):
        logger = _make_logger()
        logger.on_tool_start({"name": "my_tool"}, '{"param": "value"}')
        assert len(logger.tool_executions) == 1
        record = logger.tool_executions[0]
        assert record["tool_name"] == "my_tool"
        assert record["status"] == "started"

    def test_appends_execution_log_entries(self):
        logger = _make_logger()
        logger.on_tool_start({"name": "my_tool"}, "input data")
        assert any("my_tool" in line for line in logger.execution_logs)

    def test_calls_on_start_callback(self):
        cb = MagicMock()
        logger = _make_logger(on_start=cb)
        logger.on_tool_start({"name": "my_tool"}, "input")
        cb.assert_called_once_with("my_tool", "input")

    def test_on_start_callback_exception_is_swallowed(self):
        cb = MagicMock(side_effect=RuntimeError("boom"))
        logger = _make_logger(on_start=cb)
        # Should NOT raise
        logger.on_tool_start({"name": "my_tool"}, "input")

    def test_sets_current_tool_name(self):
        logger = _make_logger()
        logger.on_tool_start({"name": "tool_x"}, "in")
        assert logger.current_tool_name == "tool_x"

    def test_unknown_tool_name_fallback(self):
        logger = _make_logger()
        logger.on_tool_start({}, "input")  # no 'name' key
        assert logger.tool_executions[0]["tool_name"] == "unknown"


# ---------------------------------------------------------------------------
# on_tool_end
# ---------------------------------------------------------------------------

class TestOnToolEnd:
    def _start_tool(self, logger, name="my_tool"):
        logger.on_tool_start({"name": name}, "input")

    def test_updates_execution_status_to_success(self):
        logger = _make_logger()
        self._start_tool(logger)
        logger.on_tool_end("tool output")
        assert logger.tool_executions[-1]["status"] == "success"

    def test_stores_output_in_execution_record(self):
        logger = _make_logger()
        self._start_tool(logger)
        logger.on_tool_end("result text")
        assert logger.tool_executions[-1]["output"] == "result text"

    def test_handles_output_with_content_attribute(self):
        logger = _make_logger()
        self._start_tool(logger)
        output = MagicMock()
        output.content = "content result"
        logger.on_tool_end(output)
        assert logger.tool_executions[-1]["output"] == "content result"

    def test_calls_on_end_callback(self):
        cb = MagicMock()
        logger = _make_logger(on_end=cb)
        self._start_tool(logger)
        logger.on_tool_end("output")
        cb.assert_called_once_with("my_tool", "output")

    def test_on_end_callback_exception_is_swallowed(self):
        cb = MagicMock(side_effect=ValueError("oops"))
        logger = _make_logger(on_end=cb)
        self._start_tool(logger)
        logger.on_tool_end("output")  # Should NOT raise

    def test_clears_current_tool_name(self):
        logger = _make_logger()
        self._start_tool(logger)
        logger.on_tool_end("done")
        assert logger.current_tool_name is None

    def test_appends_completion_log(self):
        logger = _make_logger()
        self._start_tool(logger)
        logger.on_tool_end("done")
        assert any("completed successfully" in line for line in logger.execution_logs)


# ---------------------------------------------------------------------------
# on_tool_error
# ---------------------------------------------------------------------------

class TestOnToolError:
    def _start_tool(self, logger, name="my_tool"):
        logger.on_tool_start({"name": name}, "input")

    def test_updates_execution_status_to_error(self):
        logger = _make_logger()
        self._start_tool(logger)
        logger.on_tool_error(RuntimeError("network failure"))
        assert logger.tool_executions[-1]["status"] == "error"

    def test_stores_error_message(self):
        logger = _make_logger()
        self._start_tool(logger)
        logger.on_tool_error(RuntimeError("network failure"))
        assert "network failure" in logger.tool_executions[-1]["error"]

    def test_calls_on_error_callback(self):
        cb = MagicMock()
        logger = _make_logger(on_error=cb)
        self._start_tool(logger)
        logger.on_tool_error(ValueError("bad param"))
        cb.assert_called_once_with("my_tool", "bad param")

    def test_on_error_callback_exception_is_swallowed(self):
        cb = MagicMock(side_effect=Exception("meta error"))
        logger = _make_logger(on_error=cb)
        self._start_tool(logger)
        logger.on_tool_error(ValueError("original error"))  # Should NOT raise

    def test_clears_current_tool_name(self):
        logger = _make_logger()
        self._start_tool(logger)
        logger.on_tool_error(Exception("err"))
        assert logger.current_tool_name is None

    def test_appends_failure_log(self):
        logger = _make_logger()
        self._start_tool(logger)
        logger.on_tool_error(Exception("oops"))
        assert any("failed" in line for line in logger.execution_logs)


# ---------------------------------------------------------------------------
# No-callback construction
# ---------------------------------------------------------------------------

class TestNoCallbackLogger:
    def test_no_callbacks_does_not_raise(self):
        logger = _InternalToolLogger()
        logger.on_tool_start({"name": "tool"}, "in")
        logger.on_tool_end("out")
        logger.on_tool_error(Exception("err"))

    def test_initial_state_is_empty(self):
        logger = _InternalToolLogger()
        assert logger.tool_executions == []
        assert logger.execution_logs == []
        assert logger.current_tool_name is None


# ---------------------------------------------------------------------------
# ReactBot — ask_user (structured user questions) flow
# ---------------------------------------------------------------------------

_ASK_ARGS = {
    "reason": "Necesito datos para continuar",
    "questions": [
        {
            "id": "color",
            "text": "¿Qué color prefieres?",
            "type": "single_choice",
            "options": [
                {"value": "r", "label": "Rojo"},
                {"value": "b", "label": "Azul"},
            ],
            "required": True,
        }
    ],
}


def _model_that_asks(mock_raw_model: MagicMock) -> MagicMock:
    """Configure the mock so the first agent step calls ask_user."""
    chunk = AIMessageChunk(content="")
    chunk.tool_calls = [{"name": ASK_USER_TOOL_NAME, "args": _ASK_ARGS, "id": "call_1"}]
    mock_raw_model.stream.return_value = iter([chunk])
    return mock_raw_model


class TestReactBotAsks:
    def test_disabled_by_default_registers_no_tool(self, mock_language_model):
        bot = ReactBot(language_model=mock_language_model, instructions="x")
        assert all(t.name != ASK_USER_TOOL_NAME for t in bot.tools)

    def test_enable_registers_ask_tool(self, mock_language_model):
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       enable_user_questions=True)
        assert any(t.name == ASK_USER_TOOL_NAME for t in bot.tools)

    def test_enable_does_not_double_register(self, mock_language_model):
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       tools=[AskUserQuestionTool()], enable_user_questions=True)
        assert sum(t.name == ASK_USER_TOOL_NAME for t in bot.tools) == 1

    def test_get_response_surfaces_questions(self, mock_language_model, mock_raw_model):
        _model_that_asks(mock_raw_model)
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       enable_user_questions=True)
        result = bot.get_response(user_input="Quiero algo")

        assert result.needs_input is True
        assert len(result.questions) == 1
        assert result.questions[0]["id"] == "color"
        assert "color" in result.content.lower()
        # The ask tool must NOT have executed as a real action.
        assert all(t["tool_name"] != ASK_USER_TOOL_NAME or t.get("status") != "success"
                   for t in result.tools_executed) or not result.tools_executed

    def test_normal_answer_has_no_questions(self, mock_language_model, mock_raw_model):
        mock_raw_model.stream.return_value = iter([AIMessageChunk(content="Listo!")])
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       enable_user_questions=True)
        result = bot.get_response(user_input="Hola")
        assert result.needs_input is False
        assert result.questions == []
        assert result.content == "Listo!"

    def test_stream_emits_questions_event(self, mock_language_model, mock_raw_model):
        _model_that_asks(mock_raw_model)
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       enable_user_questions=True)
        events = list(bot.stream_response(user_message="Quiero algo", messages=[], logs=[]))
        q_events = [e for e in events if e.get("type") == "questions"]
        assert len(q_events) == 1
        assert q_events[0]["questions"][0]["id"] == "color"
        done = [e for e in events if e.get("type") == "done"]
        assert done and done[0]["result"].needs_input is True


class TestReactBotSkills:
    """Folder/programmatic skills: prompt injection + tool merge."""

    def _skill(self, name="facturacion", tool_name="facturar"):
        from sonika_ai_toolkit.skills import Skill
        tool = MagicMock()
        tool.name = tool_name
        return Skill(name=name, instructions="Sabes generar facturas.", tools=[tool])

    def test_skill_instructions_in_system_prompt(self, mock_language_model):
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       skills=[self._skill()])
        prompt = bot._build_system_prompt(include_fallback_think=False)
        assert "facturacion" in prompt
        assert "Sabes generar facturas." in prompt

    def test_skill_tools_merged(self, mock_language_model):
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       skills=[self._skill()])
        assert any(t.name == "facturar" for t in bot.tools)

    def test_explicit_tool_wins_on_name_collision(self, mock_language_model):
        mine = MagicMock()
        mine.name = "facturar"
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       tools=[mine], skills=[self._skill()])
        matching = [t for t in bot.tools if t.name == "facturar"]
        assert matching == [mine]

    def test_caller_tool_list_not_mutated(self, mock_language_model):
        caller_tools = []
        ReactBot(language_model=mock_language_model, instructions="x",
                 tools=caller_tools, skills=[self._skill()],
                 enable_user_questions=True)
        assert caller_tools == []

    def test_ask_user_still_single_with_skills(self, mock_language_model):
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       skills=[self._skill()], enable_user_questions=True)
        assert sum(t.name == ASK_USER_TOOL_NAME for t in bot.tools) == 1

    def test_no_skills_is_noop(self, mock_language_model):
        bot = ReactBot(language_model=mock_language_model, instructions="x")
        assert bot.skills == []
        assert bot._skills_prompt == ""
        prompt = bot._build_system_prompt(include_fallback_think=False)
        assert "## SKILLS" not in prompt

    def test_skills_dir_loaded(self, mock_language_model, tmp_path):
        d = tmp_path / "reportes"
        d.mkdir()
        (d / "SKILL.md").write_text("Sabes hacer reportes.", encoding="utf-8")
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       skills_dir=str(tmp_path))
        assert [s.name for s in bot.skills] == ["reportes"]
        assert "Sabes hacer reportes." in bot._skills_prompt


# ---------------------------------------------------------------------------
# ReactBot graph topology + node events (run_id / node_trace)
# ---------------------------------------------------------------------------

_RUN_ID_RE = _re.compile(r"^\d{8}T\d{12}-[0-9a-f]{32}$")


@_lc_tool
def _echo_tool(x: str) -> str:
    """Echo the input back."""
    return f"echo:{x}"


def _stream_side_effect(*chunk_lists):
    """Return a stream side_effect yielding one chunk list per model call."""
    iters = [list(chunks) for chunks in chunk_lists]

    def _stream(*args, **kwargs):
        return iter(iters.pop(0) if iters else [AIMessageChunk(content="")])

    return _stream


class TestReactBotGraphEvents:
    """Graph topology, per-node signals and run_id/node_trace in responses."""

    def test_topology_without_tools_only_agent(self, mock_language_model):
        bot = ReactBot(language_model=mock_language_model, instructions="x")
        topo = bot.get_graph_topology()
        assert topo["entry"] == "agent"
        assert set(topo["nodes"]) == {"__start__", "agent", "__end__"}

    def test_topology_with_tools_has_tools_and_ask_user(self, mock_language_model):
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       tools=[_echo_tool])
        topo = bot.get_graph_topology()
        assert {"agent", "tools", "ask_user"} <= set(topo["nodes"])
        for edge in topo["edges"]:
            assert set(edge) == {"source", "target", "conditional"}
        assert any(e["source"] == "tools" and e["target"] == "agent"
                   for e in topo["edges"])

    def test_get_response_carries_run_id_and_node_trace(
            self, mock_language_model, mock_raw_model):
        mock_raw_model.stream.side_effect = _stream_side_effect(
            [AIMessageChunk(content="Listo!")])
        bot = ReactBot(language_model=mock_language_model, instructions="x")
        result = bot.get_response(user_input="Hola")
        assert _RUN_ID_RE.match(result.run_id)
        assert [e["node"] for e in result.node_trace] == ["agent"]
        entry = result.node_trace[0]
        assert entry["run_id"] == result.run_id
        assert entry["seq"] == 1
        assert isinstance(entry["ts"], float)

    def test_run_id_never_repeats(self, mock_language_model, mock_raw_model):
        mock_raw_model.stream.side_effect = _stream_side_effect(
            [AIMessageChunk(content="a")], [AIMessageChunk(content="b")])
        bot = ReactBot(language_model=mock_language_model, instructions="x")
        r1 = bot.get_response(user_input="1")
        r2 = bot.get_response(user_input="2")
        assert r1.run_id != r2.run_id

    def test_node_trace_records_tool_round_trip(
            self, mock_language_model, mock_raw_model):
        tool_chunk = AIMessageChunk(content="")
        tool_chunk.tool_calls = [
            {"name": "_echo_tool", "args": {"x": "hola"}, "id": "call_1"}]
        mock_raw_model.stream.side_effect = _stream_side_effect(
            [tool_chunk], [AIMessageChunk(content="Hecho.")])
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       tools=[_echo_tool])
        result = bot.get_response(user_input="usa la tool")
        assert [e["node"] for e in result.node_trace] == ["agent", "tools", "agent"]
        assert [e["seq"] for e in result.node_trace] == [1, 2, 3]

    def test_stream_first_event_is_graph_topology(
            self, mock_language_model, mock_raw_model):
        mock_raw_model.stream.side_effect = _stream_side_effect(
            [AIMessageChunk(content="Listo!")])
        bot = ReactBot(language_model=mock_language_model, instructions="x")
        events = list(bot.stream_response(user_message="Hola", messages=[], logs=[]))
        assert events[0]["type"] == "graph"
        assert events[0]["entry"] == "agent"
        assert "agent" in events[0]["nodes"]
        assert _RUN_ID_RE.match(events[0]["run_id"])

    def test_stream_emits_node_events_in_order(
            self, mock_language_model, mock_raw_model):
        tool_chunk = AIMessageChunk(content="")
        tool_chunk.tool_calls = [
            {"name": "_echo_tool", "args": {"x": "hola"}, "id": "call_1"}]
        mock_raw_model.stream.side_effect = _stream_side_effect(
            [tool_chunk], [AIMessageChunk(content="Hecho.")])
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       tools=[_echo_tool])
        events = list(bot.stream_response(
            user_message="usa la tool", messages=[], logs=[]))

        node_events = [e for e in events if e["type"] == "node"]
        assert [e["node"] for e in node_events] == ["agent", "tools", "agent"]
        assert [e["seq"] for e in node_events] == [1, 2, 3]
        run_id = events[0]["run_id"]
        assert all(e["run_id"] == run_id for e in node_events)

        # Existing chunk types still flow, and the final result carries the trace.
        assert any(e["type"] == "content" for e in events)
        done = [e for e in events if e["type"] == "done"][0]
        assert [t["node"] for t in done["result"].node_trace] == ["agent", "tools", "agent"]
        assert done["result"].run_id == run_id


class TestReactBotNodeDetail:
    """Node events and node_trace carry params/output in `detail`."""

    def test_stream_node_events_carry_detail(self, mock_language_model, mock_raw_model):
        tool_chunk = AIMessageChunk(content="")
        tool_chunk.tool_calls = [
            {"name": "_echo_tool", "args": {"x": "hola"}, "id": "call_1"}]
        mock_raw_model.stream.side_effect = _stream_side_effect(
            [tool_chunk], [AIMessageChunk(content="Hecho.")])
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       tools=[_echo_tool])
        events = list(bot.stream_response(
            user_message="usa la tool", messages=[], logs=[]))
        node_events = {e["seq"]: e for e in events if e["type"] == "node"}

        # agent (seq 1): requested the tool with its args
        assert node_events[1]["detail"]["tool_calls"] == [
            {"name": "_echo_tool", "args": {"x": "hola"}}]
        # tools (seq 2): executed output visible
        assert "echo:hola" in node_events[2]["detail"]["output"]
        # final agent (seq 3): emitted text
        assert node_events[3]["detail"]["output"] == "Hecho."

    def test_get_response_node_trace_carries_detail(
            self, mock_language_model, mock_raw_model):
        tool_chunk = AIMessageChunk(content="")
        tool_chunk.tool_calls = [
            {"name": "_echo_tool", "args": {"x": "hola"}, "id": "call_1"}]
        mock_raw_model.stream.side_effect = _stream_side_effect(
            [tool_chunk], [AIMessageChunk(content="Hecho.")])
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       tools=[_echo_tool])
        result = bot.get_response(user_input="usa la tool")
        details = {e["node"]: e["detail"] for e in result.node_trace}
        assert details["agent"]  # at least tool_calls or output present
        assert "echo:hola" in details["tools"]["output"]

    def test_ask_user_node_detail_has_questions(
            self, mock_language_model, mock_raw_model):
        _model_that_asks(mock_raw_model)
        bot = ReactBot(language_model=mock_language_model, instructions="x",
                       enable_user_questions=True)
        result = bot.get_response(user_input="Quiero algo")
        ask_entry = next(e for e in result.node_trace if e["node"] == "ask_user")
        assert ask_entry["detail"]["questions"][0]["id"] == "color"
