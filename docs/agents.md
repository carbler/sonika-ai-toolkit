# Agents

Sonika AI Toolkit provides two agent architectures for different use cases. All agents return `BotResponse` — a `dict` subclass with typed property accessors.

## Overview

| Agent | Interface | Use Case |
|-------|-----------|----------|
| **ReactBot** | `IConversationBot` | Single-turn conversation + tools |
| **OrchestratorBot** | `IOrchestratorBot` | Autonomous goal-driven agent |

## ReactBot

Standard ReAct loop via LangGraph. Handles tool execution, token tracking, and callback logging.

```python
from sonika_ai_toolkit.agents.react import ReactBot
from sonika_ai_toolkit.utilities.types import Message
from sonika_ai_toolkit.utilities.models import OpenAILanguageModel

llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
bot = ReactBot(llm, instructions="You are a helpful assistant", tools=[])

messages = [Message(content="My name is Erley", is_bot=False)]
response = bot.get_response("What is my name?", messages, logs=[])
print(response.content)
```

## OrchestratorBot

Autonomous ReAct-based orchestration with persistent memory, LangGraph native interrupts, rate-limit retry with event propagation, and async-first streaming API.

### How the Graph Works

The compiled graph has **two always-present nodes** — `agent` and `tools` —
wired as a loop, plus **two opt-in nodes** that appear when their feature is
enabled: `plan` (with `enable_planning=True`) and `ask_user` (with
`enable_user_questions=True`):

```
        ┌─────────┐  plan signals?   ┌─────────┐
 START ─▶  agent   ├─────────────────▶  plan    │ (enable_planning=True)
        └────┬────┘                  └──┬──┬──┬─┘
             │ ask_user?   ┌────────────┘  │  │ no real calls
             ├─────────────▼──┐            │  └──────────► back to agent
             │           ask_user ◄────────┘ ask_user?
             │  (enable_user_ │
             │  questions)    └──────────► back to agent (after answers)
             │ real tool_calls    ┌─────────┐
             ├────────────────────▶  tools   ├──► back to agent
             │  no tool_calls     └─────────┘
             ▼
            END
```

- **`agent`** builds the system prompt (instructions + memory + skills +
  mode-specific text), calls the LLM, and routes: plan-signal calls go to
  `plan` first, an `ask_user` call goes to `ask_user`, real tool calls go to
  `tools`, and no calls at all ends the run (text becomes `final_report`).
- **`plan`** (opt-in) answers the `set_plan`/`update_step` signal calls with
  acknowledgment ToolMessages and applies them to the plan snapshot; its
  state updates stream as `("updates", {"plan": {...}})`. Real calls in the
  same batch continue to `tools` (or `ask_user`) afterwards.
- **`ask_user`** (opt-in) pauses via a native LangGraph interrupt and waits
  for `set_resume_command()`. Asking wins: any real tool call in the same
  batch is answered with a deferred ToolMessage (never executed) so the model
  re-issues it with the user's answers in hand.
- **`tools`** executes the real tool calls — including `mode="ask"` risk
  interrupts — then loops back to `agent` with the results.

With both features off the topology is exactly the classic two-node ReAct
loop. The graph wiring is fixed and not customizable.

### Sync Usage

```python
from sonika_ai_toolkit import OrchestratorBot, OpenAILanguageModel

llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
bot = OrchestratorBot(
    strong_model=llm,
    fast_model=llm,
    instructions="You are a helpful assistant.",
    tools=[],
    memory_path="/tmp/bot_memory",
)

result = bot.run("Summarize the latest AI news")
print(result.content)
print(result.tools_executed)
print(result.token_usage)
```

### Async Streaming

```python
import asyncio

async def main():
    llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
    bot = OrchestratorBot(
        strong_model=llm, fast_model=llm,
        instructions="You are a helpful assistant.",
        tools=[],
        memory_path="/tmp/bot_memory",
    )

    async for stream_mode, payload in bot.astream_events("Hello!", mode="auto"):
        if stream_mode == "updates":
            for node_name, update in payload.items():
                if node_name == "agent":
                    for ev in update.get("status_events", []):
                        if ev["type"] == "retrying":
                            print(f"Rate limit — retry {ev['attempt']}, wait {ev['wait_s']}s")
                    for partial in update.get("partial_responses", []):
                        print("Progress:", partial)
                    if update.get("final_report"):
                        print("Result:", update["final_report"])

asyncio.run(main())
```

### Modes

| Mode | Behavior |
|------|----------|
| `"auto"` | Executes all tools without interrupting |
| `"ask"` | Pauses on risky tool calls via LangGraph interrupt |
| `"plan"` | Forces the model to return a plan as text (no tool execution) |

### Key APIs

| Method | Description |
|--------|-------------|
| `bot.run(goal, context="")` | Synchronous execution |
| `await bot.arun(goal)` | Async execution |
| `bot.astream_events(goal, mode)` | Async streaming with interrupt support |
| `bot.set_resume_command(data)` | Resume after an interrupt |
| `bot.abort()` | Stop the in-flight `astream_events` run at the next boundary (see [Aborting a run](#aborting-a-run)) |
| `await bot.a_prewarm()` | Pre-warm TCP/TLS connection |

### Structured Plan & Step Progress (`enable_planning`)

**When does it activate?** All of the following must be true — miss any one
and the graph behaves exactly as if planning didn't exist:

1. The bot was built with `enable_planning=True` (default `False` — opt-in).
   This is the only thing that registers the `set_plan`/`update_step` tools
   and appends the planning protocol to the system prompt; with the default,
   the model never even sees those tools exist.
2. The current turn's `mode` is **not** `"plan"`. `mode="plan"` is a separate,
   older, mutually-exclusive behavior (free-text plan, all tools stripped) —
   the two never combine.
3. The model **itself decides** to call `set_plan` or `update_step` — the
   graph never forces this. A simple one-shot question typically gets no
   plan at all, even with `enable_planning=True`, because the model judges a
   plan unnecessary. Multi-step goals are what the protocol prompt asks the
   model to plan for.

**How it flows through the graph** (via the dedicated `plan` node, added to
the topology only when `enable_planning=True`):

1. `agent` runs with the planning protocol + current plan snapshot in the
   prompt. The model calls `set_plan(steps=[...])`, alone or alongside real
   tool calls for the first step.
2. The router sends the batch to the **`plan` node** first. It applies the
   signal calls to the plan snapshot (all steps `"pending"` after `set_plan`)
   and answers them with acknowledgment ToolMessages (so the conversation
   stays a normal tool round-trip). Signal calls are **excluded from
   `tools_executed`** — they performed no real action.
3. From `plan`, any real tool calls in the same batch continue to `tools`
   (or `ask_user`); with none, the flow returns straight to `agent`.
4. Back in `agent`, the updated plan snapshot renders into the prompt so the
   model sees step 1 is already registered and doesn't re-announce it. The
   model now calls `update_step(1, "running")` alongside the real tool(s) for
   that step, then later `update_step(1, "done")`, then moves to step 2, and
   so on until all steps are done and the model returns final text with no
   more tool calls — ending the run at `agent` as usual.

Each plan-node run surfaces through the `"updates"` stream **under the
`"plan"` node name** (and as `"plan"` node events in the `"graph"` stream):

```python
bot = OrchestratorBot(..., enable_planning=True)

async for stream_mode, payload in bot.astream_events(goal, mode="auto"):
    if stream_mode == "updates":
        for node, update in payload.items():
            if node == "plan":
                if update.get("plan"):
                    for s in update["plan"]:
                        print(f'{s["step"]}. [{s["status"]}] {s["description"]}')
                for ev in update.get("step_events", []):
                    print(f'Paso {ev["step"]} → {ev["status"]}')
```

Stream shape:

```python
("updates", {"agent": {...}})              # model called set_plan
("updates", {"plan": {"plan": [
    {"step": 1, "description": "Buscar datos", "status": "pending"},
    {"step": 2, "description": "Generar reporte", "status": "pending"},
]}})
("updates", {"agent": {...}})              # model called update_step + real tool
("updates", {"plan": {"plan": [...], "step_events": [{"step": 1, "status": "running"}]}})
("updates", {"tools": {...}})
("updates", {"plan": {"plan": [...], "step_events": [{"step": 1, "status": "done"}]}})
```

> Migration note: in releases up to v0.3.14 the plan snapshot and step events
> streamed under the `"agent"` update. They now stream under the dedicated
> `"plan"` node.

Non-streaming callers get the final snapshot in `result.plan` (from
`run`/`arun`), with each step's final status. The feature is **opt-in**: with
the default `enable_planning=False` nothing changes — no extra tools, no new
stream keys, and `result.plan` stays `[]`. The text-only `mode="plan"` is
unaffected.

## Graph Topology & Node Events (both bots)

Both **ReactBot** and **OrchestratorBot** expose the graph they run and signal
every node execution, so a UI can **draw the graph up front and animate the
path the bot takes** through it.

Three pieces, same semantics in both bots:

1. **Topology** — `bot.get_graph_topology()` returns the static layout of the
   compiled graph. The same payload is also the **first stream event** of
   every run:

   ```python
   {
       "entry": "agent",                       # first real node executed
       "nodes": ["__start__", "agent", "tools", "__end__"],
       "edges": [
           {"source": "__start__", "target": "agent", "conditional": False},
           {"source": "agent", "target": "tools", "conditional": True},
           {"source": "agent", "target": "__end__", "conditional": True},
           {"source": "tools", "target": "agent", "conditional": False},
       ],
   }
   ```

   `conditional: True` marks edges taken via a router (the agent's
   tool-calls decision). `plan` (`enable_planning=True`) and
   `ask_user` (`enable_user_questions=True`) appear as regular nodes.

2. **Node signals** — one event per node execution, in order, each carrying
   the node name, a 1-based `seq`, an epoch `ts`, the run's `run_id`, and a
   **`detail`** payload with the node's params and output
   (`NodeDetail`, all keys optional):

   ```python
   {
       "tool_calls":     [{"name": "get_datetime", "args": {"tz": "local"}}],
       "tools_executed": [{"tool_name": "get_datetime", "args": {...},
                           "status": "success", "output": "14:44"}],
       "output":         "text the node emitted (truncated to 500 chars)",
       "plan":           [...],   # plan node: snapshot after this run
       "step_events":    [...],   # plan node: transitions of this run
       "questions":      [...],   # ask_user (ReactBot): questions raised
   }
   ```

   For `agent`, `tool_calls` are the tools it *requested* (with args) and
   `output` the text it emitted; for `tools`, `tools_executed` carries the
   args and (truncated) output of every executed tool. Replay the signals
   over the topology to paint the steps the bot took, with their data.

3. **`run_id` (process id)** — every run (`run` / `arun` / `astream_events` /
   `get_response` / `stream_response`) gets a **globally unique id that never
   repeats**: a UTC timestamp (microsecond precision) plus a full UUID4, e.g.
   `20260717T175049977691-3bf4f982c8bf4a209a9f9459e7cdaa28`. The date prefix
   makes ids sortable; the untruncated UUID4 makes collisions impossible in
   practice. The same `run_id` appears in the topology event, every node
   signal, and the final `BotResponse`.

### OrchestratorBot — the `"graph"` stream mode

`astream_events` yields a third stream mode alongside `"messages"` and
`"updates"` (existing consumers that filter by mode are unaffected):

```python
async for stream_mode, payload in bot.astream_events(goal, mode="auto"):
    if stream_mode == "graph":
        if payload["type"] == "graph_topology":
            draw_graph(payload["nodes"], payload["edges"])   # first event
        elif payload["type"] == "node_invoked":
            highlight_node(payload["node"])                  # animate the step
            print(f'#{payload["seq"]} {payload["node"]}: {payload["detail"]}')
```

Stream shape of a run that uses one tool:

```python
("graph", {"type": "graph_topology", "run_id": "...", "entry": "agent",
           "nodes": [...], "edges": [...]})
("graph", {"type": "node_invoked", "run_id": "...", "node": "agent", "seq": 1, "ts": ...,
           "detail": {"tool_calls": [{"name": "get_datetime", "args": {...}}]}})
("updates", {"agent": {...}})
("graph", {"type": "node_invoked", "run_id": "...", "node": "tools", "seq": 2, "ts": ...,
           "detail": {"tools_executed": [{"tool_name": "get_datetime", "args": {...},
                                          "status": "success", "output": "14:44"}]}})
("updates", {"tools": {...}})
("graph", {"type": "node_invoked", "run_id": "...", "node": "agent", "seq": 3, "ts": ...,
           "detail": {"output": "La hora actual es 14:44."}})
("updates", {"agent": {"final_report": "..."}})
```

The topology event is emitted only when a `goal` starts a new run (not on
interrupt resumes); `node_invoked` events keep the original run's `run_id`
across resumes. The TypedDicts are `GraphTopologyEvent` and
`NodeInvokedEvent` (importable from `sonika_ai_toolkit`).

### ReactBot — `"graph"` / `"node"` stream chunks

`stream_response` yields the same information as plain dict chunks:

```python
for ev in bot.stream_response(user_message, messages=[], logs=[]):
    if ev["type"] == "graph":       # first chunk: topology + run_id
        draw_graph(ev["nodes"], ev["edges"])
    elif ev["type"] == "node":      # one per node execution
        highlight_node(ev["node"])  # ev["seq"], ev["ts"], ev["run_id"]
        print(ev["detail"])         # params/output of this node run
```

### Non-streaming: `BotResponse.node_trace`

`run` / `arun` / `get_response` return the recorded path in the response:

```python
result = bot.get_response("usa la tool")   # or await orchestrator.arun(goal)
result.run_id       # "20260717T175052294436-1375b198..."
result.node_trace   # [{"node": "agent", "seq": 1, "ts": ..., "run_id": ...,
                    #   "detail": {"tool_calls": [{"name": "...", "args": {...}}]}},
                    #  {"node": "tools", "seq": 2, ...,
                    #   "detail": {"tools_executed": [{..., "output": "..."}]}},
                    #  {"node": "agent", "seq": 3, ..., "detail": {"output": "..."}}]
```

## Aborting a run

Both bots expose `abort()` to **stop an in-flight streaming run**. It is meant
to be called from a **different task/thread** than the one consuming the stream
(a UI button, a websocket handler, a cancel signal) while the bot is running.

```python
# Task A — consumes the stream
async for stream_mode, payload in bot.astream_events(goal, mode="auto"):
    if stream_mode == "graph" and payload["type"] == "aborted":
        print("Run stopped by the user")
        break
    render(stream_mode, payload)

# Task B — the library user, at any moment
bot.abort()
```

ReactBot works the same way with its sync stream (call `abort()` from another
thread):

```python
for chunk in bot.stream_response(user_message, messages=[], logs=[]):
    if chunk["type"] == "aborted":
        break
    render(chunk)
# ...meanwhile, from another thread:  bot.abort()
```

**What happens on abort:**

- The stream yields one final event and stops: `("graph", {"type": "aborted",
  "run_id": ...})` for OrchestratorBot, `{"type": "aborted", "run_id": ...}`
  for ReactBot. ReactBot emits **no** `done` chunk when aborted. The typed
  contract is `AbortedEvent` (importable from `sonika_ai_toolkit`).
- **It genuinely halts the graph** — not just the event stream. Streaming is
  pull-driven, so breaking cancels the underlying LangGraph run; no work
  continues in the background. State up to the last completed node is preserved
  in the checkpointer under the run's `thread_id`.
- The abort flag is **reset at the start of every run**, so the bot is
  immediately reusable.

**Granularity — the one thing to know:** a node that is already running is not
cancelled mid-execution; the abort applies at the **next boundary**.

- Aborting while the `agent` is reasoning takes effect almost immediately — the
  stream yields on every LLM token, so the check runs constantly.
- Aborting while a **tool is executing** only applies once that tool returns (a
  running node cannot be killed halfway).

**Not affected:** the non-streaming `run` / `arun` / `get_response` paths use
`ainvoke` and do not observe the flag — cancel their asyncio task to stop them.

## BotResponse

All agents return `BotResponse`, a `dict` subclass with typed properties:

```python
result = bot.run(...)

# Dict-style access (backward compatible)
result["content"]
result.get("thinking")

# Property-style access (typed)
result.content          # str
result.thinking         # Optional[str]
result.logs             # List[str]
result.tools_executed   # List[dict]
result.token_usage      # {prompt_tokens, completion_tokens, total_tokens}
result.success          # bool
result.plan             # List[dict]
result.session_id       # Optional[str]
result.goal             # Optional[str]
result.questions        # List[dict]  — structured questions the agent asks
result.needs_input      # bool        — True when waiting for answers
result.run_id           # Optional[str] — unique process id (never repeats)
result.node_trace       # List[dict]  — ordered node executions {node, seq, ts, run_id}
```

## Structured User Questions (`ask_user`)

Both **ReactBot** and **OrchestratorBot** can pause and ask the caller
*structured* questions — instead of burying a question in free text — so a UI can
render inputs, radio buttons, or checkboxes and send back typed answers.

The mechanism is a **signal tool** (`ask_user`): the model calls it with a
schema-validated payload; the bot intercepts the call and surfaces the questions
rather than executing an action. Enable it with `enable_user_questions=True`.

### Question schema

```python
from sonika_ai_toolkit import Question, QuestionOption, AskUserSchema

# Each question the model emits:
# {
#   "id":       "color",                 # stable key; answers are keyed by it
#   "text":     "¿Qué color prefieres?",
#   "type":     "single_choice",         # text | single_choice | multi_choice | boolean | number
#   "options":  [{"value": "r", "label": "Rojo"}, ...],  # for *_choice types
#   "required": True,
# }
```

### ReactBot — stateless (ends the turn, resume on the next call)

```python
from sonika_ai_toolkit.agents.react import ReactBot

bot = ReactBot(
    language_model=lm,
    instructions="...",
    enable_user_questions=True,   # registers the ask_user tool
)

result = bot.get_response(user_input="Quiero reservar un vuelo")

if result.needs_input:
    answers = render_and_collect(result.questions)   # your UI draws the form
    # Feed the answers back as the next user turn (with history):
    followup = bot.get_response(
        user_input=f"Mis respuestas: {answers}",
        messages=history,
    )
```

`stream_response()` also emits a dedicated event when the agent asks:

```python
for event in bot.stream_response(user_message="...", messages=[], logs=[]):
    if event["type"] == "questions":
        render_and_collect(event["questions"])   # {"type","questions","reason"}
```

### OrchestratorBot — stateful (pauses via interrupt, resumes the *same* run)

With `enable_user_questions=True` the questions run through a **dedicated
`ask_user` graph node** (visible in the topology and in the `"graph"` node
events). It fires a native LangGraph interrupt, so the run keeps all context
and continues where it left off once you provide the answers. Asking wins
over other tool calls in the same batch: those are answered with a deferred
ToolMessage (never executed) and the model re-issues them after the answers
arrive.

```python
from sonika_ai_toolkit import OrchestratorBot

bot = OrchestratorBot(
    strong_model=lm, fast_model=lm,
    instructions="...",
    enable_user_questions=True,
)

async for stream_mode, payload in bot.astream_events(goal, mode="ask", thread_id="t1"):
    if stream_mode == "updates" and "__interrupt__" in payload:
        interrupt = payload["__interrupt__"][0].value       # QuestionEvent
        if interrupt["type"] == "question_request":
            answers = render_and_collect(interrupt["questions"])   # {id: answer}
            bot.set_resume_command(answers)

# Resume the same run — the model now has the answers in context:
async for stream_mode, payload in bot.astream_events(None, mode="ask", thread_id="t1"):
    ...
```

`run()` / `arun()` (non-streaming) also expose `result.questions` and
`result.needs_input` when the model asks.

> **Difference by design.** OrchestratorBot resumes without losing state (same
> run). ReactBot is stateless, so it ends the turn and resumes on the next
> `get_response()` call with the answers passed back as input.

See [`examples/ask_user_console.py`](https://github.com/carbler/sonika-ai-toolkit/blob/main/examples/ask_user_console.py)
for a runnable console UI that renders the questions and collects typed answers
for both agents.

## Interface Hierarchy

```
IBot (ABC)
├── IConversationBot    — ReactBot
│     get_response(user_input, messages, logs) → BotResponse
└── IOrchestratorBot    — OrchestratorBot
      astream_events(goal, mode, thread_id) → AsyncGenerator
      arun(goal, context, thread_id) → BotResponse
      run(goal, context, thread_id) → BotResponse
```

Type-hint against interfaces, not concrete classes:

```python
from sonika_ai_toolkit import IConversationBot, IOrchestratorBot
```
