"""Contract tests for the sonika-ai-toolkit public interfaces.

These tests act as executable documentation.  If anyone breaks IOrchestratorBot,
IConversationBot, or the stream event types, these tests fail immediately.
"""

import inspect
import pytest
from abc import ABC

from sonika_ai_toolkit.agents.orchestrator.interface import IOrchestratorBot
from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot
from sonika_ai_toolkit.agents.orchestrator.events import (
    AgentUpdate,
    ToolsUpdate,
    StatusEvent,
    ToolRecord,
)
from sonika_ai_toolkit.utilities.types import BotResponse, ILanguageModel
from sonika_ai_toolkit.interfaces.base import BaseInterface


# ── IBot hierarchy ────────────────────────────────────────────────────────────

class TestIBotHierarchy:
    """Verify the interface hierarchy is correct."""

    def test_ibot_is_abstract(self):
        from sonika_ai_toolkit.agents.base import IBot, IConversationBot
        assert issubclass(IBot, ABC)
        assert issubclass(IConversationBot, IBot)

    def test_iconversationbot_is_abstract(self):
        from sonika_ai_toolkit.agents.base import IConversationBot
        with pytest.raises(TypeError):
            IConversationBot()

    def test_iorchestrator_inherits_ibot(self):
        from sonika_ai_toolkit.agents.base import IBot
        assert issubclass(IOrchestratorBot, IBot)

    def test_reactbot_implements_iconversationbot(self):
        from sonika_ai_toolkit.agents.react import ReactBot
        from sonika_ai_toolkit.agents.base import IConversationBot
        assert issubclass(ReactBot, IConversationBot)

    def test_taskerbot_implements_iconversationbot(self):
        from sonika_ai_toolkit.agents.tasker.tasker_bot import TaskerBot
        from sonika_ai_toolkit.agents.base import IConversationBot
        assert issubclass(TaskerBot, IConversationBot)

    def test_orchestratorbot_implements_iorchestrator(self):
        assert issubclass(OrchestratorBot, IOrchestratorBot)


# ── IConversationBot contract ─────────────────────────────────────────────────

class TestIConversationBotContract:

    @pytest.mark.parametrize("method_name", ["get_response"])
    def test_required_methods_exist(self, method_name):
        from sonika_ai_toolkit.agents.base import IConversationBot
        assert hasattr(IConversationBot, method_name)

    def test_get_response_signature(self):
        from sonika_ai_toolkit.agents.base import IConversationBot
        sig = inspect.signature(IConversationBot.get_response)
        params = list(sig.parameters.keys())
        assert "user_input" in params
        assert "messages" in params
        assert "logs" in params


# ── IOrchestratorBot contract ─────────────────────────────────────────────────

class TestIOrchestratorBotContract:

    def test_is_abstract_base_class(self):
        assert issubclass(IOrchestratorBot, ABC)

    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            IOrchestratorBot()

    def test_orchestratorbot_implements_interface(self):
        assert issubclass(OrchestratorBot, IOrchestratorBot)

    @pytest.mark.parametrize("method_name", [
        "astream_events", "arun", "run", "set_resume_command", "a_prewarm",
    ])
    def test_required_methods_exist(self, method_name):
        assert hasattr(IOrchestratorBot, method_name)

    def test_astream_events_signature(self):
        sig = inspect.signature(IOrchestratorBot.astream_events)
        params = list(sig.parameters.keys())
        assert "goal" in params
        assert "mode" in params
        assert "thread_id" in params

    def test_arun_returns_botresponse_type_hint(self):
        sig = inspect.signature(IOrchestratorBot.arun)
        assert sig.return_annotation is not inspect.Parameter.empty


# ── Stream event TypedDicts ───────────────────────────────────────────────────

class TestStreamEventTypes:
    """Verify the stream event TypedDicts have the required keys."""

    def test_status_event_required_keys(self):
        ev: StatusEvent = {
            "type": "retrying",
            "reason": "rate_limit",
            "attempt": 1,
            "wait_s": 2.0,
        }
        assert ev["type"] == "retrying"
        assert isinstance(ev["attempt"], int)
        assert isinstance(ev["wait_s"], float)

    def test_tool_record_required_keys(self):
        rec: ToolRecord = {
            "tool_name": "run_bash",
            "args": {"command": "ls"},
            "status": "success",
            "output": "file.txt",
        }
        assert rec["status"] in ("success", "error", "skipped")

    def test_agent_update_accepts_status_events(self):
        update: AgentUpdate = {
            "final_report": "Done",
            "status_events": [
                {"type": "retrying", "reason": "rate_limit", "attempt": 1, "wait_s": 2.0}
            ],
        }
        assert len(update["status_events"]) == 1


# ── BotResponse contract ──────────────────────────────────────────────────────

class TestBotResponseContract:
    """Verify BotResponse always exposes required properties."""

    def test_required_properties_exist(self):
        r = BotResponse(content="ok", logs=[], tools_executed=[], token_usage={})
        assert isinstance(r.content, str)
        assert isinstance(r.logs, list)
        assert isinstance(r.tools_executed, list)
        assert isinstance(r.token_usage, dict)

    def test_empty_botresponse_has_safe_defaults(self):
        r = BotResponse()
        assert r.content == ""
        assert r.logs == []
        assert r.tools_executed == []
        assert r.thinking is None

    def test_is_dict_compatible(self):
        r = BotResponse(content="hello")
        assert r["content"] == "hello"
        assert r.get("missing_key", "default") == "default"


# ── BaseInterface contract ────────────────────────────────────────────────────

class TestBaseInterfaceContract:
    """Verify BaseInterface has all required UI methods."""

    @pytest.mark.parametrize("method_name", [
        "on_thought", "on_tool_start", "on_tool_end",
        "on_error", "on_interrupt", "on_result", "on_retry",
    ])
    def test_required_methods_exist(self, method_name):
        assert hasattr(BaseInterface, method_name)

    def test_on_retry_has_default_implementation(self):
        """on_retry must NOT be abstract — backward compatible with existing subclasses."""

        class MinimalImpl(BaseInterface):
            def on_thought(self, chunk): pass
            def on_tool_start(self, tool_name, params): pass
            def on_tool_end(self, tool_name, result): pass
            def on_error(self, tool_name, error): pass
            def on_interrupt(self, data): return True
            def on_result(self, result): pass

        impl = MinimalImpl()
        impl.on_retry(1, 2.0)  # Must not raise


# ── Public API smoke test ─────────────────────────────────────────────────────

class TestPublicAPI:
    """Verify top-level imports work — catches __init__.py regressions."""

    def test_orchestratorbot_importable_from_top_level(self):
        from sonika_ai_toolkit import OrchestratorBot
        assert OrchestratorBot is not None

    def test_interfaces_importable_from_top_level(self):
        from sonika_ai_toolkit import IOrchestratorBot, BaseInterface
        assert IOrchestratorBot is not None
        assert BaseInterface is not None

    def test_event_types_importable_from_top_level(self):
        from sonika_ai_toolkit import AgentUpdate, StatusEvent, ToolRecord
        assert AgentUpdate is not None

    def test_core_tools_importable_from_top_level(self):
        from sonika_ai_toolkit import FindFileTool, RunBashTool
        assert FindFileTool is not None

    def test_ibot_hierarchy_importable_from_top_level(self):
        from sonika_ai_toolkit import IBot, IConversationBot
        assert IBot is not None
        assert IConversationBot is not None
