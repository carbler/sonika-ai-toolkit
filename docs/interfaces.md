# Interfaces

## BaseInterface

ABC for UI layers that interact with the OrchestratorBot. Implement this to build custom CLIs, web UIs, or GUI frontends.

```python
from sonika_ai_toolkit import BaseInterface

class MyUI(BaseInterface):
    def on_thought(self, chunk: str) -> None:
        print(f"Thinking: {chunk}")

    def on_tool_start(self, tool_name: str, params: dict) -> None:
        print(f"Running {tool_name}...")

    def on_tool_end(self, tool_name: str, result: str) -> None:
        print(f"{tool_name} done: {result[:100]}")

    def on_error(self, tool_name: str, error: str) -> None:
        print(f"Error in {tool_name}: {error}")

    def on_interrupt(self, data: dict) -> bool:
        return input(f"Allow {data}? (y/n): ").lower() == "y"

    def on_result(self, result: str) -> None:
        print(f"Result: {result}")
```

### Required Methods

| Method | Description |
|--------|-------------|
| `on_thought(chunk)` | Render a chunk of thinking/reasoning |
| `on_tool_start(tool_name, params)` | Render the start of a tool execution |
| `on_tool_end(tool_name, result)` | Render successful tool completion |
| `on_error(tool_name, error)` | Render a tool execution error |
| `on_interrupt(data) → bool` | Handle a LangGraph interrupt (permission prompt) |
| `on_result(result)` | Render the final result |

### Optional Methods (default no-op)

| Method | Description |
|--------|-------------|
| `on_retry(attempt, wait_s, reason)` | Called on rate-limit retry |
| `on_partial_response(text)` | Called when agent emits intermediate text |

## Stream Event Types

TypedDicts for processing `astream_events()` payloads:

```python
from sonika_ai_toolkit import (
    StatusEvent,           # rate-limit retry event
    PartialResponseEvent,  # intermediate text while agent continues
    AgentUpdate,           # "agent" node payload
    ToolsUpdate,           # "tools" node payload
    ToolRecord,            # individual tool execution record
    QuestionEvent,         # ask_user interrupt payload
    QuestionItem,          # a single structured question
    QuestionOptionEvent,   # a choice inside a QuestionItem
    GraphTopologyEvent,    # node/edge layout, first event of a run
    NodeInvokedEvent,      # signal fired every time a node executes
    AbortedEvent,          # last event when a run is stopped by bot.abort()
    NodeDetail,            # params/output summary inside a node event
    GraphEdgeSpec,         # one edge of the topology
    NodeTraceEntry,        # one entry of BotResponse.node_trace
)
```

### StatusEvent

```python
{"type": "retrying", "reason": "rate_limit", "attempt": 1, "wait_s": 2.0}
```

### QuestionEvent

Payload of a `question_request` interrupt, emitted when the agent calls the
`ask_user` tool (see [Agents — Structured User Questions](agents.md#structured-user-questions-ask_user)).
Resume with `set_resume_command({question_id: answer})`.

```python
{
    "type": "question_request",
    "reason": "Necesito datos para continuar",
    "questions": [
        {
            "id": "color",
            "text": "¿Qué color prefieres?",
            "type": "single_choice",   # text | single_choice | multi_choice | boolean | number
            "options": [{"value": "r", "label": "Rojo"}],
            "required": True,
        }
    ],
}
```

### PartialResponseEvent

```python
{"text": "I found the following information...", "turn": 2}
```

### ToolRecord

```python
{"tool_name": "search_web", "args": {"query": "..."}, "status": "success", "output": "..."}
```

### AgentUpdate

Payload from the `agent` node in `"updates"` stream mode:

```python
{
    "messages": [...],
    "final_report": "The result is...",
    "partial_response": "Searching...",
    "thinking": "Let me analyze...",
    "status_events": [{"type": "retrying", ...}],
    # Only when the bot was built with enable_planning=True:
    "plan": [{"step": 1, "description": "...", "status": "pending"}, ...],
    "step_events": [{"step": 1, "status": "running"}],
}
```

### PlanStep

One step of the structured plan (emitted when `enable_planning=True`):

```python
{"step": 1, "description": "Buscar datos", "status": "pending"}
# status: pending | running | done | skipped | error
```

### StepEvent

A progress transition for one plan step:

```python
{"step": 1, "status": "running"}
# status: running | done | skipped | error
```

### ToolsUpdate

Payload from the `tools` node in `"updates"` stream mode:

```python
{
    "messages": [...],
    "tools_executed": [{"tool_name": "...", "args": {...}, "status": "success", "output": "..."}],
}
```

### GraphTopologyEvent

First event of every run, in the `"graph"` stream mode (OrchestratorBot) or as
a `{"type": "graph"}` chunk (ReactBot) — the full graph layout, for drawing it
before any node runs. See
[Agents — Graph Topology & Node Events](agents.md#graph-topology-node-events-both-bots).

```python
{
    "type": "graph_topology",
    "run_id": "20260717T175049977691-3bf4f982c8bf4a209a9f9459e7cdaa28",
    "entry": "agent",
    "nodes": ["__start__", "agent", "tools", "__end__"],
    "edges": [{"source": "agent", "target": "tools", "conditional": True}, ...],
}
```

### NodeInvokedEvent

One per node execution, in order — replay over the topology to animate the
path the bot took. `run_id` is the unique, never-repeating process id
(UTC timestamp + full UUID4) shared by all events of the same run. `detail`
carries the node's params/output summary.

```python
{
    "type": "node_invoked", "run_id": "...", "node": "tools",
    "seq": 2, "ts": 1752774649.97,
    "detail": {"tools_executed": [
        {"tool_name": "get_datetime", "args": {"tz": "local"},
         "status": "success", "output": "14:44"},
    ]},
}
```

### AbortedEvent

The last event of a run stopped by `bot.abort()`. Emitted as `("graph",
payload)` by `OrchestratorBot.astream_events` and as `{"type": "aborted", ...}`
by `ReactBot.stream_response`, right before the stream stops. State up to the
last completed node is preserved in the checkpointer (`thread_id`); work in
progress at the moment of abort is discarded. See
[Aborting a run](agents.md#aborting-a-run).

```python
{"type": "aborted", "run_id": "..."}
```

### NodeDetail

Params/output summary inside `NodeInvokedEvent.detail` and
`NodeTraceEntry.detail` — all keys optional:

```python
{
    "tool_calls":     [{"name": "...", "args": {...}}],  # tools the node requested
    "tools_executed": [{"tool_name", "args", "status", "output"}],  # tools it ran
    "output":         "text the node emitted (truncated to 500 chars)",
    "plan":           [...],   # plan node: snapshot after this run
    "step_events":    [...],   # plan node: step transitions of this run
    "questions":      [...],   # ask_user node (ReactBot): questions raised
}
```

## BotResponse

Unified response type returned by all agents. See [Agents — BotResponse](agents.md#botresponse) for full documentation.
