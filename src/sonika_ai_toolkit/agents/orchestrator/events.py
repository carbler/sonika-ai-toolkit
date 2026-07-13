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


class PartialResponseEvent(TypedDict):
    """Intermediate text emitted by the agent while it continues working."""
    text: str
    turn: int


class QuestionOptionEvent(TypedDict, total=False):
    """A selectable choice inside a QuestionItem."""
    value: str
    label: str


class QuestionItem(TypedDict, total=False):
    """A single structured question surfaced to the consumer."""
    id: str
    text: str
    type: str  # text | single_choice | multi_choice | boolean | number
    options: List[QuestionOptionEvent]
    required: bool


class QuestionEvent(TypedDict, total=False):
    """Payload of a ``question_request`` LangGraph interrupt.

    Emitted when the agent calls the ``ask_user`` tool: it needs structured input
    from the caller before it can continue.  Resume with ``set_resume_command()``
    passing a dict of ``{question_id: answer}`` (or plain text for a single ask).
    """
    type: Literal["question_request"]
    questions: List[QuestionItem]
    reason: Optional[str]


class PlanStep(TypedDict):
    """One step of the structured plan registered via the ``set_plan`` tool."""
    step: int          # 1-based index
    description: str
    status: Literal["pending", "running", "done", "skipped", "error"]


class StepEvent(TypedDict):
    """A progress transition for one plan step (``update_step`` tool)."""
    step: int
    status: Literal["running", "done", "skipped", "error"]


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
    partial_response: Optional[str]  # intermediate text when agent continues working
    thinking: str
    status_events: List[StatusEvent]  # rate-limit retries, warnings, …
    plan: List[PlanStep]              # plan snapshot (enable_planning=True only)
    step_events: List[StepEvent]      # step progress deltas of this turn


class ToolsUpdate(TypedDict, total=False):
    """Payload emitted by the ``tools`` node in the ``updates`` stream."""
    messages: List[Any]
    tools_executed: List[ToolRecord]
