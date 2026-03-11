# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in development mode
pip install -e .

# Run unit + integration tests (no API keys needed — always fast)
pytest tests/unit tests/integration -q

# Run unit tests only
pytest -m unit

# Run integration tests only
pytest -m integration

# Run e2e tests (require real API keys in .env)
pytest tests/e2e/ -m e2e -s -v

# Run e2e for one provider only
pytest tests/e2e/ -m e2e -k openai -s

# Run a specific test file
pytest tests/unit/test_models.py

# Run specific test
pytest tests/unit/test_models.py::TestOpenAILanguageModel::test_init_default_model_name

# Run contract tests
pytest tests/unit/test_orchestrator_contract.py -v

# Run stress test suite (banking scenario, interactive)
python tests/ultimate/banking_operations/stress_test_runner.py

# Lint
ruff check .

# Documentation — local preview
pip install mkdocs-material && mkdocs serve

# Documentation — manual deploy (normally auto via GitHub Actions)
mkdocs gh-deploy --force
```

## Documentation (MkDocs + GitHub Pages)

The project uses **MkDocs Material** for documentation, hosted on GitHub Pages.

- **Live site**: https://carbler.github.io/sonika-ai-toolkit/
- **Config**: `mkdocs.yml` (site metadata, theme, nav, markdown extensions)
- **Source**: `docs/` directory

### Documentation Structure

```
docs/
├── index.md               # Overview + quick example + navigation links
├── getting-started.md     # Installation, API keys, first agent, first classifier
├── agents.md              # ReactBot, TaskerBot, OrchestratorBot (modes, streaming, BotResponse)
├── classifiers.md         # TextClassifier, Intent, Sentiment, Safety, Image — with examples
├── models.md              # OpenAI, Gemini, DeepSeek, Bedrock — config + gotchas
├── tools.md               # 18 built-in tools + custom tool creation with Pydantic
└── interfaces.md          # BaseInterface, BotResponse, stream event TypedDicts
```

### Auto-Deploy Workflow

`.github/workflows/docs.yml` runs on every push to `main`:
1. Checkout → Setup Python 3.12 → Install mkdocs-material
2. `mkdocs gh-deploy --force` → publishes to `gh-pages` branch

**GitHub Pages config**: Settings → Pages → Source: Deploy from `gh-pages` branch, `/ (root)`.

### When to Update Docs

- Adding a new agent, classifier, tool, or event type → update the relevant `docs/*.md` page
- Changing public API (`__init__.py`) → update `docs/getting-started.md` and relevant page
- Adding a new top-level component → add nav entry in `mkdocs.yml`

## Testing

### Test Structure

```
tests/
├── conftest.py          # Shared mocks — no API keys needed
├── unit/                # ~310 tests — isolated, ~5s
│   ├── test_models.py                  # LLM wrapper tests (all providers)
│   ├── test_classifiers.py            # Classifier tests (28 tests)
│   └── test_orchestrator_contract.py  # Interface contract tests (37 tests)
├── integration/         # Tests with mocked component interaction
├── e2e/                 # Real API calls, skip if key missing
│   ├── conftest.py      # ← MODEL CONFIGURATION (change model name here)
│   ├── test_reactbot.py
│   ├── test_orchestratorbot.py
│   └── test_classifiers.py           # Classifier e2e tests (10 tests)
└── ultimate/            # Stress test suite (standalone runner, not pytest)
    └── banking_operations/
        ├── batch_runner.py      # Run: python batch_runner.py
        └── stress_test_runner.py
```

### Changing the test model (one place)

Edit `tests/e2e/conftest.py` — three lines:
```python
OPENAI_MODEL   = "gpt-4o-mini-2024-07-18"   # change here
GEMINI_MODEL   = "gemini-2.5-flash"
DEEPSEEK_MODEL = "deepseek-chat"
```
Or set env vars: `TEST_OPENAI_MODEL`, `TEST_GEMINI_MODEL`, `TEST_DEEPSEEK_MODEL`.

### Test Markers
```bash
pytest -m unit        # Fast, isolated, mocked
pytest -m integration # Component-interaction, mocked
pytest -m e2e         # Real API, skip if no key
pytest -m slow        # Slower tests
```

### What e2e tests assert
- `result.content` is a non-empty string
- Both `EmailTool` and `SaveContacto` appear in `result.tools_executed`
- Stream events yield at least one `"messages"` or `"updates"` event (OrchestratorBot)
- Classifier e2e: result has expected keys, token counts > 0

## Architecture

**sonika-ai-toolkit** is a Python library for building conversational AI agents using LangChain and LangGraph, with multi-provider LLM support (OpenAI, DeepSeek, Google Gemini, Amazon Bedrock).

### Interface Hierarchy (`agents/base.py`, `agents/orchestrator/interface.py`)

All agents share a common ABC lineage:

```
IBot (ABC)
├── IConversationBot    — ReactBot, TaskerBot
│     get_response(user_input, messages, logs) → BotResponse
└── IOrchestratorBot    — OrchestratorBot
      astream_events(goal, mode, thread_id) → AsyncGenerator
      arun(goal, context, thread_id) → BotResponse
      run(goal, context, thread_id) → BotResponse
      set_resume_command(resume_data) → None
      a_prewarm() → None
```

Consumers should type-hint against interfaces, not concrete classes.

### Unified Response Type (`BotResponse`)

All agents return a `BotResponse` — a `dict` subclass fully backward-compatible with plain dict access:

```python
from sonika_ai_toolkit.utilities.types import BotResponse

result = bot.get_response(...)   # ReactBot / TaskerBot
result = bot.run(...)            # OrchestratorBot

# dict-style (existing code unchanged)
result["content"]
result.get("thinking")

# property-style (typed)
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

### Stream Event Types (`agents/orchestrator/events.py`)

TypedDicts forming the stable contract for `astream_events()` payloads:

```python
from sonika_ai_toolkit.agents.orchestrator.events import (
    AgentUpdate, ToolsUpdate, StatusEvent, ToolRecord
)

# StatusEvent — emitted when a rate-limit retry occurs
ev: StatusEvent = {"type": "retrying", "reason": "rate_limit", "attempt": 1, "wait_s": 2.0}

# AgentUpdate — payload of "agent" node in "updates" stream mode
update: AgentUpdate = {"final_report": "...", "status_events": [...]}

# ToolsUpdate — payload of "tools" node in "updates" stream mode
update: ToolsUpdate = {"tools_executed": [...]}
```

Import these types in consumers instead of hardcoding dict keys.

### Core Abstractions (`src/sonika_ai_toolkit/utilities/`)

- `types.py`: `BotResponse` (unified response), `ILanguageModel`, `Message`, `ResponseModel`, `IEmbeddings`, `FileProcessorInterface`
- `models.py`: `OpenAILanguageModel`, `GeminiLanguageModel`, `BedrockLanguageModel`, `DeepSeekLanguageModel`. All expose `predict()`, `invoke()`, and `stream_response()`. All provider imports (`ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatBedrock`) are at module level — required for correct `unittest.mock.patch` behavior in tests.

### Agents (`src/sonika_ai_toolkit/agents/`)

Three architectures for different use cases:

1. **ReactBot** (`react.py`): Standard ReAct loop via LangGraph. Implements `IConversationBot`. Handles tool execution, token tracking, `_InternalToolLogger` callback. Returns `BotResponse`.

2. **TaskerBot** (`tasker/`): Planner→Executor→Validator→Output→Logger graph for complex multi-step tasks. Implements `IConversationBot`. Nodes under `tasker/nodes/`. Returns `BotResponse`.

3. **OrchestratorBot** (`orchestrator/`): Autonomous ReAct-based orchestration with persistent memory, LangGraph native interrupts, rate-limit retry with event propagation, and async-first streaming API. Implements `IOrchestratorBot`.

### OrchestratorBot — How it works

The orchestrator runs a **compact ReAct loop** compiled as a LangGraph state machine:

```
agent_node ──(tool_calls?)──► tools_node ──► agent_node
     │                                              │
     └─────────────── (no tool calls) ─────────── END
```

Each `run(goal)` call:

1. **`agent_node`** — streams the LLM response via `astream`, accumulates tool calls and thinking content. A tenacity retry decorator wraps the stream call; on 429/rate-limit it waits exponentially (up to 5 attempts) and emits `StatusEvent` objects into `state["status_events"]` so consumers can show progress.
2. **`tools_node`** — iterates tool calls; if `mode="ask"` and `tool.risk_level > 0`, fires a **native LangGraph interrupt** and waits for `set_resume_command()` before executing.
3. Loop continues until the model stops calling tools, then final text is stored as `final_report`.

**Key APIs:**
- `bot.run(goal, context="")` — synchronous (wraps `arun`)
- `await bot.arun(goal)` — async, runs in `"auto"` mode (no interrupts)
- `async for stream_mode, payload in bot.astream_events(goal, mode="ask")` — streaming with interrupt support
- `bot.set_resume_command(resume_data)` — provide approval data after an interrupt
- `await bot.a_prewarm()` — pre-warm TCP/TLS connection
- `memory_path` — directory for MEMORY.md and session logs

**Mode parameter:**
- `"ask"` (default) — pauses on risky tool calls via LangGraph interrupt
- `"auto"` — executes all tools without interrupting
- `"plan"` — forces the model to return a plan as text (tool calls stripped)

**Retry with rate-limit events:**

```python
# Consuming status events from the stream
async for stream_mode, payload in bot.astream_events(goal):
    if stream_mode == "updates":
        for node_name, update in payload.items():
            if node_name == "agent":
                for ev in update.get("status_events", []):
                    if ev["type"] == "retrying":
                        print(f"Rate limit — retry {ev['attempt']}, wait {ev['wait_s']}s")
```

### Classifiers (`src/sonika_ai_toolkit/classifiers/`)

Five classifiers for structured text and image classification. All return `ClassificationResponse` with `result` (dict), `input_tokens`, and `output_tokens`. All support sync (`classify`) and async (`aclassify`).

- **`TextClassifier`** (`text.py`): Base classifier — user provides a Pydantic `validation_class` defining the output schema. Uses `with_structured_output(include_raw=True)` for token extraction.
- **`IntentClassifier`** (`intent.py`): Classifies text into predefined intents. Accepts `intents` list and optional `descriptions` dict. Uses `create_model()` to build a dynamic schema constraining the `intent` field. Returns `{intent, confidence, entities}`.
- **`SentimentClassifier`** (`sentiment.py`): Zero-config sentiment analysis. Fixed `SentimentResult` schema. Returns `{sentiment, confidence, reasoning}`.
- **`SafetyClassifier`** (`safety.py`): Content safety moderation. Default categories: `hate_speech`, `violence`, `sexual_content`, `self_harm`, `pii`, `harassment`, `illegal_activity`. Accepts `custom_categories`. Returns `{is_safe, categories, severity, explanation}`.
- **`ImageClassifier`** (`image.py`): Multimodal image classification. Accepts image URLs or local file paths (auto-converts to base64 data URL). Requires vision-capable LLM (OpenAI gpt-4o/gpt-4o-mini or Gemini).

**Token extraction** (`_extract_tokens` in `text.py`): Handles both OpenAI-style (`response_metadata.token_usage.prompt_tokens`) and Gemini-style (`usage_metadata.input_tokens`).

### Tools (`src/sonika_ai_toolkit/tools/`)

- `registry.py`: `ToolRegistry` — register/get/list; `get_tool_descriptions()` includes param names for LLM prompts
- `core/`: 16 built-in tools — `RunBashTool`, `BashSafeTool`, `ReadFileTool`, `WriteFileTool`, `ListDirTool`, `DeleteFileTool`, `FindFileTool`, `RunPythonTool`, `CallApiTool`, `SearchWebTool`, `FetchWebPageTool`, `GetDateTimeTool`, `EmailSMTPTool`, `SQLiteTool`, `PostgreSQLTool`, `MySQLTool`, `RedisTool`
- `integrations.py`: `EmailTool`, `SaveContacto` — must have `args_schema` (Pydantic) for correct LLM param generation

### Other Components

- `document_processing/`: `DocumentProcessor` for PDF, DOCX, XLSX, PPTX, TXT
- `interfaces/base.py`: `BaseInterface` — ABC for UI layers. Implement `on_thought`, `on_tool_start`, `on_tool_end`, `on_error`, `on_interrupt`, `on_result`. `on_retry(attempt, wait_s, reason)` and `on_partial_response(text)` have default no-ops — override to show feedback.

### Provider-Specific Gotchas

- **Gemini**: Thinking models (`gemini-2.5-*`, `*-thinking`, `*thinking-exp*`) require `temperature=1.0` — automatically overridden with a warning. `response.content` is a list when `include_thoughts=True`.
- **Gemini thinking detection**: Only `gemini-2.5-*`, markers `"-thinking"`, `"thinking-exp"` trigger thinking mode. `gemini-3-*` models are **not** treated as thinking models.
- **DeepSeek reasoner** (`deepseek-reasoner`, or any model with `r1` in the name): Uses `_DeepSeekReasonerChatModel` (module-level subclass of `ChatOpenAI`). `reasoning_content` is injected into `additional_kwargs` via `_create_chat_result` override.
- **LangGraph**: Requires a checkpointer (`MemorySaver`) for state persistence. `status_events` uses `operator.add` reducer in `OrchestratorState`.
- **Bedrock**: `BedrockLanguageModel.__init__` sets `AWS_BEARER_TOKEN_BEDROCK` env var automatically.
- **Mock patching**: All provider imports are module-level in `models.py` — `patch("sonika_ai_toolkit.utilities.models.ChatOpenAI")` works correctly.

### Public API (`src/sonika_ai_toolkit/__init__.py`)

Top-level imports for the most common components:

```python
from sonika_ai_toolkit import (
    # Orchestrator
    OrchestratorBot, IOrchestratorBot,
    # Agent interfaces
    IBot, IConversationBot,
    # Stream event types
    AgentUpdate, ToolsUpdate, ToolRecord, StatusEvent, PartialResponseEvent,
    # Response type
    BotResponse, ILanguageModel,
    # LLM providers
    GeminiLanguageModel, OpenAILanguageModel,
    BedrockLanguageModel, DeepSeekLanguageModel,
    # UI contract
    BaseInterface,
    # Classifiers
    TextClassifier, ClassificationResponse,
    IntentClassifier, SentimentClassifier,
    SafetyClassifier, ImageClassifier,
    # Core tools
    RunBashTool, BashSafeTool,
    ReadFileTool, WriteFileTool, ListDirTool, DeleteFileTool, FindFileTool,
    CallApiTool, SearchWebTool, FetchWebPageTool,
    RunPythonTool, GetDateTimeTool,
    EmailSMTPTool, SQLiteTool, PostgreSQLTool, MySQLTool, RedisTool,
)
```
