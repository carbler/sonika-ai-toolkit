"""Typed event payloads emitted by the OrchestratorBot stream.

Consumers import these TypedDicts to process events without coupling to
implementation details.  This file is the single source of truth for the
stream contract.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict


class StatusEvent(TypedDict):
    """A lifecycle / diagnostic event emitted inside an agent update."""
    type: Literal["retrying", "warning", "info"]
    reason: str        # e.g. "rate_limit", "timeout"
    attempt: int
    wait_s: float


class ToolRecord(TypedDict):
    """A record of a single tool execution captured by the tools node."""
    tool_name: str
    args: Dict[str, Any]
    status: Literal["success", "error", "skipped"]
    output: str


class AgentUpdate(TypedDict, total=False):
    """Payload emitted by the ``agent`` node in the ``updates`` stream."""
    messages: List[Any]          # List[BaseMessage]
    final_report: Optional[str]
    thinking: str
    status_events: List[StatusEvent]  # rate-limit retries, warnings, …


class ToolsUpdate(TypedDict, total=False):
    """Payload emitted by the ``tools`` node in the ``updates`` stream."""
    messages: List[Any]
    tools_executed: List[ToolRecord]
