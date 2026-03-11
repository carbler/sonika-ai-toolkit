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
)
```

### StatusEvent

```python
{"type": "retrying", "reason": "rate_limit", "attempt": 1, "wait_s": 2.0}
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
}
```

### ToolsUpdate

Payload from the `tools` node in `"updates"` stream mode:

```python
{
    "messages": [...],
    "tools_executed": [{"tool_name": "...", "args": {...}, "status": "success", "output": "..."}],
}
```

## BotResponse

Unified response type returned by all agents. See [Agents — BotResponse](agents.md#botresponse) for full documentation.
