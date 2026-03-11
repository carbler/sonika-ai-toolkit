# Sonika AI Toolkit

A robust Python library for building conversational AI agents using **LangChain** and **LangGraph**, with multi-provider LLM support.

## Features

- **Multi-Model Support** — OpenAI, DeepSeek, Google Gemini, Amazon Bedrock
- **Three Agent Architectures** — ReactBot, TaskerBot, OrchestratorBot
- **Classifiers** — Text, Intent, Sentiment, Safety, and Image classification
- **18 Built-in Tools** — Bash, file I/O, API calls, web search, databases, and more
- **Typed Interfaces** — `IConversationBot`, `IOrchestratorBot` ABCs with `BotResponse`
- **Async Streaming** — Real-time events, interrupts, and partial responses
- **Document Processing** — PDF, DOCX, XLSX, PPTX extraction and chunking

## Quick Install

```bash
pip install sonika-ai-toolkit
```

## Quick Example

```python
from sonika_ai_toolkit import OrchestratorBot, OpenAILanguageModel

llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
bot = OrchestratorBot(
    strong_model=llm, fast_model=llm,
    instructions="You are a helpful assistant.",
    tools=[],
    memory_path="/tmp/bot_memory",
)

result = bot.run("What is the capital of France?")
print(result.content)
```

## Next Steps

- [Getting Started](getting-started.md) — Installation, API keys, first bot
- [Agents](agents.md) — ReactBot, TaskerBot, OrchestratorBot
- [Classifiers](classifiers.md) — Text, Intent, Sentiment, Safety, Image
- [Models](models.md) — Provider configuration and gotchas
- [Tools](tools.md) — Built-in tools and custom tool creation
- [Interfaces](interfaces.md) — BaseInterface, BotResponse, stream events
