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


class GraphEdgeSpec(TypedDict):
    """One directed edge of the compiled graph topology."""
    source: str
    target: str
    conditional: bool  # True when the edge is taken via a router function


class GraphTopologyEvent(TypedDict):
    """First event of a run: the full node/edge layout of the compiled graph.

    Emitted as ``("graph", payload)`` by ``OrchestratorBot.astream_events`` and
    as ``{"type": "graph", ...}`` by ``ReactBot.stream_response`` so consumers
    can draw the graph before any node runs.  Nodes include the virtual
    ``__start__`` / ``__end__`` markers.
    """
    type: Literal["graph_topology"]
    run_id: str        # unique id of this run (process)
    entry: str         # first real node executed (e.g. "agent")
    nodes: List[str]
    edges: List[GraphEdgeSpec]


class NodeDetail(TypedDict, total=False):
    """Params/output summary of one node execution (all keys optional).

    Carried by ``NodeInvokedEvent.detail`` and ``NodeTraceEntry.detail``:
      tool_calls     — tools the node's message requests: [{name, args}, …]
      tools_executed — tools actually run: [{tool_name, args, status, output}, …]
      output         — text the node emitted (truncated to 500 chars)
      plan           — plan snapshot after this node (plan node)
      step_events    — step transitions of this node (plan node)
      questions      — structured questions raised (ask_user node, ReactBot)
    """
    tool_calls: List[Dict[str, Any]]
    tools_executed: List[Dict[str, Any]]
    output: str
    plan: List[Dict[str, Any]]
    step_events: List[Dict[str, Any]]
    questions: List[Dict[str, Any]]


class NodeInvokedEvent(TypedDict):
    """A signal that one graph node just executed.

    Emitted as ``("graph", payload)`` (OrchestratorBot) / ``{"type": "node"}``
    (ReactBot) once per node execution, in order — replay them over the
    topology to animate the path the bot took. ``detail`` carries the node's
    params/output summary.
    """
    type: Literal["node_invoked"]
    run_id: str
    node: str          # node name ("agent", "tools", "plan", "ask_user", custom, …)
    seq: int           # 1-based execution order within the run
    ts: float          # epoch seconds when the node finished
    detail: NodeDetail


class NodeTraceEntry(TypedDict):
    """One entry of ``BotResponse.node_trace`` — the recorded execution path."""
    node: str
    run_id: str
    seq: int
    ts: float
    detail: NodeDetail


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
