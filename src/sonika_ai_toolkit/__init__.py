"""sonika-ai-toolkit — Public API."""

# Core orchestrator
from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot
from sonika_ai_toolkit.agents.orchestrator.interface import IOrchestratorBot

# Agent base interfaces
from sonika_ai_toolkit.agents.base import IBot, IConversationBot

# Stream event types
from sonika_ai_toolkit.agents.orchestrator.events import (
    AgentUpdate,
    ToolsUpdate,
    ToolRecord,
    StatusEvent,
    PartialResponseEvent,
)

# Response type
from sonika_ai_toolkit.utilities.types import BotResponse, ILanguageModel

# Language model implementations
from sonika_ai_toolkit.utilities.models import (
    GeminiLanguageModel,
    OpenAILanguageModel,
    BedrockLanguageModel,
    DeepSeekLanguageModel,
)

# UI interface contract
from sonika_ai_toolkit.interfaces.base import BaseInterface

# Core tools
from sonika_ai_toolkit.tools.core import (
    RunBashTool,
    BashSafeTool,
    ReadFileTool,
    WriteFileTool,
    ListDirTool,
    DeleteFileTool,
    FindFileTool,
    CallApiTool,
    SearchWebTool,
    RunPythonTool,
    FetchWebPageTool,
    GetDateTimeTool,
    EmailSMTPTool,
    SQLiteTool,
    PostgreSQLTool,
    MySQLTool,
    RedisTool,
)

__all__ = [
    # Orchestrator
    "OrchestratorBot",
    "IOrchestratorBot",
    # Agent interfaces
    "IBot",
    "IConversationBot",
    # Event types
    "AgentUpdate",
    "ToolsUpdate",
    "ToolRecord",
    "StatusEvent",
    "PartialResponseEvent",
    # Types
    "BotResponse",
    "ILanguageModel",
    # Models
    "GeminiLanguageModel",
    "OpenAILanguageModel",
    "BedrockLanguageModel",
    "DeepSeekLanguageModel",
    # UI
    "BaseInterface",
    # Tools
    "RunBashTool",
    "BashSafeTool",
    "ReadFileTool",
    "WriteFileTool",
    "ListDirTool",
    "DeleteFileTool",
    "FindFileTool",
    "CallApiTool",
    "SearchWebTool",
    "RunPythonTool",
    "FetchWebPageTool",
    "GetDateTimeTool",
    "EmailSMTPTool",
    "SQLiteTool",
    "PostgreSQLTool",
    "MySQLTool",
    "RedisTool",
]
