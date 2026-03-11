# Sonika AI Toolkit <a href="https://pepy.tech/projects/sonika-ai-toolkit"><img src="https://static.pepy.tech/badge/sonika-ai-toolkit" alt="PyPI Downloads"></a>

A robust Python library designed to build state-of-the-art conversational agents and AI tools. It leverages `LangChain` and `LangGraph` to create autonomous bots capable of complex reasoning and tool execution.

**[Documentation](https://sonika-technologies.github.io/sonika-ai-toolkit/)**

## Installation

```bash
pip install sonika-ai-toolkit
```

## Prerequisites

You'll need the following API keys depending on the model you wish to use:

- OpenAI API Key
- DeepSeek API Key (Optional)
- Google Gemini API Key (Optional)
- AWS Bedrock API Key (Optional, for Bedrock)

Create a `.env` file in the root of your project with the following variables:

```env
OPENAI_API_KEY=your_openai_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
GOOGLE_API_KEY=your_gemini_key_here
AWS_BEARER_TOKEN_BEDROCK=your_bedrock_api_key_here
AWS_REGION=us-east-1
```

## Key Features

- **Multi-Model Support**: Agnostic integration with OpenAI, DeepSeek, Google Gemini, and Amazon Bedrock.
- **Conversational Agent**: Robust agent (`ReactBot`) with native tool execution and LangGraph state management.
- **Tasker Agent**: Planner-executor agent (`TaskerBot`) for complex multi-step tasks.
- **Orchestrator Agent**: Autonomous goal-driven agent (`OrchestratorBot`) with async streaming, persistent memory, LangGraph interrupts for human-in-the-loop, and rate-limit retry with progress events.
- **Formal Interface Contracts**: `IConversationBot` and `IOrchestratorBot` ABCs ensure stable APIs across agent implementations.
- **Typed Stream Events**: `StatusEvent`, `PartialResponseEvent`, `AgentUpdate`, `ToolsUpdate` TypedDicts decouple consumers from implementation details.
- **Partial/Intermediate Responses**: The orchestrator emits structured `partial_responses` when the agent produces text while continuing to call tools, enabling real-time progress feedback.
- **Classifiers**: Text, Intent, Sentiment, Safety, and Image classification with structured outputs.
- **Document Processing**: Utilities for processing PDFs, DOCX, and other formats with intelligent chunking.
- **Custom Tools**: Easy integration of custom tools via Pydantic and LangChain.

## Basic Usage

### Conversational Agent with Tools

```python
import os
from dotenv import load_dotenv
from sonika_ai_toolkit.tools.integrations import EmailTool
from sonika_ai_toolkit.agents.react import ReactBot
from sonika_ai_toolkit.utilities.types import Message
from sonika_ai_toolkit.utilities.models import OpenAILanguageModel

load_dotenv()

language_model = OpenAILanguageModel(os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini")
bot = ReactBot(language_model, instructions="You are a helpful assistant", tools=[EmailTool()])

messages = [Message(content="My name is Erley", is_bot=False)]
response = bot.get_response("Send an email to erley@gmail.com saying hello", messages, logs=[])

print(response["content"])
```

### Autonomous Orchestrator (sync)

```python
import os
from dotenv import load_dotenv
from sonika_ai_toolkit import OrchestratorBot, OpenAILanguageModel
from sonika_ai_toolkit.tools.integrations import EmailTool, SaveContacto

load_dotenv()

llm = OpenAILanguageModel(os.getenv("OPENAI_API_KEY"), model_name="gpt-4o-mini")
bot = OrchestratorBot(
    strong_model=llm,
    fast_model=llm,
    instructions="You are a communications assistant.",
    tools=[EmailTool(), SaveContacto()],
    memory_path="/tmp/my_bot_memory",
)

result = bot.run("Send a hello email to erley@gmail.com and save him as a contact.")
print(result.content)
print("Tools used:", [t["tool_name"] for t in result.tools_executed])
```

### Autonomous Orchestrator (async streaming)

```python
import asyncio
from sonika_ai_toolkit import OrchestratorBot, OpenAILanguageModel, StatusEvent
from sonika_ai_toolkit.tools.integrations import EmailTool

async def main():
    llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
    bot = OrchestratorBot(
        strong_model=llm, fast_model=llm,
        instructions="You are a helpful assistant.",
        tools=[EmailTool()],
        memory_path="/tmp/bot_memory",
    )

    async for stream_mode, payload in bot.astream_events("Send hello to erley@gmail.com", mode="auto"):
        if stream_mode == "updates":
            for node_name, update in payload.items():
                if node_name == "agent":
                    # Show rate-limit retry progress
                    for ev in update.get("status_events", []):
                        if ev["type"] == "retrying":
                            print(f"↻ Rate limit — retry {ev['attempt']}, wait {ev['wait_s']}s")
                    # Show intermediate progress
                    for partial in update.get("partial_responses", []):
                        print("Progress:", partial)
                    if update.get("final_report"):
                        print("Result:", update["final_report"])

asyncio.run(main())
```

### Classifiers

#### Text Classification (custom schema)

```python
from pydantic import BaseModel, Field
from sonika_ai_toolkit import TextClassifier, OpenAILanguageModel

class TicketClassification(BaseModel):
    category: str = Field(..., description="The ticket category")
    priority: str = Field(..., description="Priority: low, medium, high, critical")

llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
classifier = TextClassifier(llm=llm, validation_class=TicketClassification)
result = classifier.classify("My server is down!")
print(result.result)  # {'category': 'infrastructure', 'priority': 'critical'}
```

#### Intent Classification

```python
from sonika_ai_toolkit import IntentClassifier, OpenAILanguageModel

llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
classifier = IntentClassifier(
    llm=llm,
    intents=["book_flight", "cancel_booking", "check_status"],
    descriptions={"book_flight": "User wants to book a new flight"},
)
result = classifier.classify("I need to fly to London next Friday")
print(result.result)  # {'intent': 'book_flight', 'confidence': 0.95, 'entities': {...}}
```

#### Sentiment Analysis

```python
from sonika_ai_toolkit import SentimentClassifier, OpenAILanguageModel

classifier = SentimentClassifier(llm=OpenAILanguageModel("sk-..."))
result = classifier.classify("This product is amazing!")
print(result.result)  # {'sentiment': 'positive', 'confidence': 0.92, 'reasoning': '...'}
```

#### Safety Classification

```python
from sonika_ai_toolkit import SafetyClassifier, OpenAILanguageModel

classifier = SafetyClassifier(llm=OpenAILanguageModel("sk-..."))
result = classifier.classify("I love sunny days at the park.")
print(result.result)  # {'is_safe': True, 'categories': [], 'severity': 'none', ...}

# With custom categories
classifier = SafetyClassifier(llm=llm, custom_categories=["misinformation", "spam"])
```

#### Image Classification

```python
from pydantic import BaseModel, Field
from sonika_ai_toolkit import ImageClassifier, OpenAILanguageModel

class SceneAnalysis(BaseModel):
    description: str = Field(..., description="Brief description")
    objects: list[str] = Field(..., description="Main objects detected")

llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
classifier = ImageClassifier(llm=llm, validation_class=SceneAnalysis)
result = classifier.classify("https://example.com/photo.jpg")
print(result.result)
```

## Available Components

### Agents

| Agent | Class | Interface | Use Case |
|-------|-------|-----------|----------|
| **ReactBot** | `agents.react.ReactBot` | `IConversationBot` | Single-turn conversation + tools |
| **TaskerBot** | `agents.tasker.TaskerBot` | `IConversationBot` | Multi-step planner-executor |
| **OrchestratorBot** | `agents.orchestrator.graph.OrchestratorBot` | `IOrchestratorBot` | Autonomous goal-driven agent |

All agents return `BotResponse` — a `dict` subclass with typed property accessors (`.content`, `.thinking`, `.tools_executed`, `.token_usage`).

### Classifiers

| Classifier | Description | Schema |
|------------|-------------|--------|
| **TextClassifier** | Custom schema classification | User-defined Pydantic model |
| **IntentClassifier** | Intent detection + entities | `intent`, `confidence`, `entities` |
| **SentimentClassifier** | Sentiment analysis | `sentiment`, `confidence`, `reasoning` |
| **SafetyClassifier** | Content safety moderation | `is_safe`, `categories`, `severity`, `explanation` |
| **ImageClassifier** | Multimodal image classification | User-defined Pydantic model |

### Interfaces

```python
from sonika_ai_toolkit.agents.base import IBot, IConversationBot
from sonika_ai_toolkit.agents.orchestrator.interface import IOrchestratorBot
```

### Stream Event Types

```python
from sonika_ai_toolkit.agents.orchestrator.events import (
    StatusEvent,           # rate-limit retry event
    PartialResponseEvent,  # intermediate text while agent continues working
    AgentUpdate,           # "agent" node payload in "updates" stream
    ToolsUpdate,           # "tools" node payload in "updates" stream
    ToolRecord,            # individual tool execution record
)
```

### Language Models

```python
from sonika_ai_toolkit.utilities.models import (
    OpenAILanguageModel,    # OpenAI (gpt-4o, gpt-4o-mini, ...)
    GeminiLanguageModel,    # Google Gemini (gemini-2.5-flash, ...)
    DeepSeekLanguageModel,  # DeepSeek (deepseek-chat, deepseek-reasoner, ...)
    BedrockLanguageModel,   # Amazon Bedrock (amazon.nova-micro-v1:0, ...)
)
```

### Utilities

- **`ILanguageModel`**: Unified interface for LLM providers (`predict`, `invoke`, `stream_response`).
- **`BotResponse`**: Unified response type — dict-compatible + typed properties.
- **`BaseInterface`**: ABC for UI layers — implement `on_thought`, `on_tool_start`, `on_tool_end`, `on_error`, `on_interrupt`, `on_result`. Optional: `on_retry`, `on_partial_response`.
- **`DocumentProcessor`**: Text extraction and chunking for PDF, DOCX, XLSX, PPTX.

### Top-Level Imports

```python
from sonika_ai_toolkit import (
    # Agents
    OrchestratorBot, IOrchestratorBot,
    IBot, IConversationBot,
    # Events
    AgentUpdate, ToolsUpdate, ToolRecord, StatusEvent, PartialResponseEvent,
    # Types
    BotResponse, ILanguageModel,
    # Models
    GeminiLanguageModel, OpenAILanguageModel,
    BedrockLanguageModel, DeepSeekLanguageModel,
    # UI
    BaseInterface,
    # Classifiers
    TextClassifier, ClassificationResponse,
    IntentClassifier, SentimentClassifier,
    SafetyClassifier, ImageClassifier,
    # Tools
    RunBashTool, BashSafeTool,
    ReadFileTool, WriteFileTool, ListDirTool, DeleteFileTool, FindFileTool,
    CallApiTool, SearchWebTool, FetchWebPageTool,
    RunPythonTool, GetDateTimeTool,
    EmailSMTPTool, SQLiteTool, PostgreSQLTool, MySQLTool, RedisTool,
)
```

## Project Structure

```
src/sonika_ai_toolkit/
├── agents/
│   ├── base.py              # IBot, IConversationBot ABCs
│   ├── react.py             # ReactBot(IConversationBot)
│   ├── tasker/              # TaskerBot(IConversationBot)
│   └── orchestrator/
│       ├── graph.py         # OrchestratorBot(IOrchestratorBot)
│       ├── interface.py     # IOrchestratorBot ABC
│       ├── events.py        # Stream event TypedDicts
│       ├── state.py         # OrchestratorState (LangGraph)
│       └── memory.py        # MemoryManager (MEMORY.md)
├── classifiers/
│   ├── __init__.py          # Public exports
│   ├── text.py              # TextClassifier (base, custom schema)
│   ├── intent.py            # IntentClassifier (predefined intents)
│   ├── sentiment.py         # SentimentClassifier (zero-config)
│   ├── safety.py            # SafetyClassifier (content moderation)
│   └── image.py             # ImageClassifier (multimodal, vision LLMs)
├── document_processing/     # PDF and document tools
├── interfaces/
│   └── base.py              # BaseInterface ABC for UI layers
├── tools/
│   ├── core/                # 16 built-in tools
│   ├── integrations.py      # EmailTool, SaveContacto
│   └── registry.py          # ToolRegistry
└── utilities/
    ├── models.py            # LLM provider wrappers
    └── types.py             # BotResponse, ILanguageModel, Message
```

## License

This project is licensed under the MIT License.
