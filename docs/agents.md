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
```

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
