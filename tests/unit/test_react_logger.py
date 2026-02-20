"""
Unit tests for sonika_ai_toolkit.agents.react._InternalToolLogger

Covers:
  - on_llm_start: appends "[AGENT] Thinking..." log
  - on_llm_end: logs tool call names or "Generated response"
  - on_tool_start: appends execution record, calls on_start callback
  - on_tool_end: updates execution record, calls on_end callback
  - on_tool_error: updates execution record, calls on_error callback
  - Callback resilience: exceptions in user callbacks are swallowed
"""

import pytest
from unittest.mock import MagicMock, call
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from sonika_ai_toolkit.agents.react import _InternalToolLogger


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
        thinking_logs = [l for l in logger.execution_logs if "Thinking" in l]
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
        log_entry = next((l for l in logger.execution_logs if "Decided to call tools" in l), None)
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
        assert any("my_tool" in l for l in logger.execution_logs)

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
        assert any("completed successfully" in l for l in logger.execution_logs)


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
        assert any("failed" in l for l in logger.execution_logs)


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
