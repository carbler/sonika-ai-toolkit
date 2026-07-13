# Agents

Sonika AI Toolkit provides three agent architectures for different use cases. All agents return `BotResponse` — a `dict` subclass with typed property accessors.

## Overview

| Agent | Interface | Use Case |
|-------|-----------|----------|
| **ReactBot** | `IConversationBot` | Single-turn conversation + tools |
| **TaskerBot** | `IConversationBot` | Multi-step planner-executor |
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

## TaskerBot

Planner → Executor → Validator → Output → Logger graph for complex multi-step tasks.

```python
from sonika_ai_toolkit.agents.tasker import TaskerBot

bot = TaskerBot(llm, instructions="You are a project manager", tools=[])
response = bot.get_response("Create a plan for launching a product", [], logs=[])
print(response.content)
print(response.plan)
```

## OrchestratorBot

Autonomous ReAct-based orchestration with persistent memory, LangGraph native interrupts, rate-limit retry with event propagation, and async-first streaming API.

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
| `await bot.a_prewarm()` | Pre-warm TCP/TLS connection |

### Structured Plan & Step Progress (`enable_planning`)

With `enable_planning=True` the orchestrator registers two internal *signal*
tools (`set_plan`, `update_step`) and instructs the model to announce a plan
before working and report per-step progress. The calls perform no real action
(they get a no-op acknowledgment and never appear in `tools_executed`); the
plan and its progress surface through the `"updates"` stream:

```python
bot = OrchestratorBot(..., enable_planning=True)

async for stream_mode, payload in bot.astream_events(goal, mode="auto"):
    if stream_mode == "updates":
        for node, update in payload.items():
            if node == "agent":
                if update.get("plan"):
                    for s in update["plan"]:
                        print(f'{s["step"]}. [{s["status"]}] {s["description"]}')
                for ev in update.get("step_events", []):
                    print(f'Paso {ev["step"]} → {ev["status"]}')
```

Stream shape:

```python
("updates", {"agent": {"plan": [
    {"step": 1, "description": "Buscar datos", "status": "pending"},
    {"step": 2, "description": "Generar reporte", "status": "pending"},
]}})
("updates", {"agent": {"step_events": [{"step": 1, "status": "running"}]}})
("updates", {"tools": {...}})
("updates", {"agent": {"step_events": [{"step": 1, "status": "done"}]}})
```

Non-streaming callers get the final snapshot in `result.plan` (from
`run`/`arun`), with each step's final status. The feature is **opt-in**: with
the default `enable_planning=False` nothing changes — no extra tools, no new
stream keys, and `result.plan` stays `[]`. The text-only `mode="plan"` is
unaffected.

### Custom Nodes (`custom_nodes`)

Inject your own LangGraph nodes into the orchestrator graph without forking:

```python
from sonika_ai_toolkit import CustomNode

def audit(state):  # sync or async; receives OrchestratorState
    return {"session_log": [f"goal: {state.get('goal')}"]}  # partial state update

bot = OrchestratorBot(..., custom_nodes=[
    CustomNode(name="audit", node=audit, position="start"),
])
```

Positions:

| Position | Where it runs |
|----------|---------------|
| `"start"` | Between the entry point and the agent (once per run) |
| `"after_tools"` | On the tools → agent edge (every tool loop) |
| `"end"` | After the agent's final turn, before END (once per run) |

Multiple nodes at the same position chain in list order. Node names must not
collide with the built-ins (`agent`, `tools`); their state updates stream as
`("updates", {"<name>": {...}})`.

**TaskerBot** supports a different mechanism — *node overrides*. The topology
(planner → executor → validator → output → logger) stays fixed, but each node
implementation can be swapped with any callable honoring that node's state
contract:

```python
bot = TaskerBot(..., planner_node=my_planner, output_node=my_output)
```

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

The orchestrator reuses the native LangGraph interrupt, so it keeps all context
and continues where it left off once you provide the answers.

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
├── IConversationBot    — ReactBot, TaskerBot
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
