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
pytest tests/unit/utilities/test_models.py

# Run specific test
pytest tests/unit/utilities/test_models.py::TestOpenAILanguageModel::test_init_default_model_name

# Run contract tests
pytest tests/unit/agents/orchestrator/test_contract.py -v

# Run the model/agent benchmark (real API keys — standalone, not pytest)
python benchmarks/run.py --agents react --models openai:gpt-4o-mini
python benchmarks/run.py --list   # discover agents / providers / scenarios

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
├── agents.md              # ReactBot, OrchestratorBot (modes, streaming, BotResponse)
├── classifiers.md         # TextClassifier, Intent, Sentiment, Safety, Image — with examples
├── models.md              # OpenAI, Gemini, DeepSeek, Bedrock — config + gotchas
├── tools.md               # 18 built-in tools + custom tool creation with Pydantic
├── skills.md              # Folder-based skills (SKILL.md + tools.py) for all bots
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

The `tests/unit/` tree mirrors `src/sonika_ai_toolkit/` 1:1 — one test file per
module. Markers (`unit`/`integration`/`e2e`) are applied automatically by
directory via a `pytest_collection_modifyitems` hook in `tests/conftest.py`, so
files don't declare `pytestmark`.

```
tests/
├── conftest.py                       # Shared mocks + auto-marking hook — no API keys needed
├── unit/                             # ~370 tests — isolated, ~5s
│   ├── utilities/
│   │   ├── test_models.py            # LLM wrappers (all providers)
│   │   ├── test_types.py             # BotResponse, ILanguageModel, Message
│   │   └── test_questions.py         # ask_user contract (schema/payload/summary)
│   ├── classifiers/
│   │   └── test_classifiers.py       # Text/Intent/Sentiment/Safety/Image
│   ├── tools/
│   │   ├── test_core_tools.py        # bash, files, http, python, search, web, datetime, email
│   │   ├── test_database_tools.py    # SQLite, PostgreSQL, MySQL, Redis
│   │   ├── test_integrations.py      # EmailTool, SaveContacto
│   │   ├── test_ask_user.py          # AskUserQuestionTool
│   │   ├── test_plan_tools.py        # set_plan / update_step signal tools
│   │   ├── test_registry.py          # ToolRegistry
│   │   └── test_synthesizer.py       # DynamicToolSynthesizer
│   ├── skills/
│   │   └── test_loader.py            # Skill.from_dir, load_skills, merge/render helpers
│   ├── agents/
│   │   ├── test_react.py             # _InternalToolLogger + ReactBot ask_user flow + skills
│   │   └── orchestrator/
│   │       ├── test_contract.py      # Interface contract tests
│   │       ├── test_graph.py         # agent/tools graph, partial responses
│   │       ├── test_graph_planning.py # enable_planning: plan snapshot + step events + arun plan
│   │       ├── test_node_events.py   # graph topology + node_invoked events + run_id/node_trace
│   │       ├── test_planning.py      # pure plan helpers (normalize/apply/render/split)
│   │       ├── test_risk.py          # risk-gate helpers
│   │       └── test_memory.py        # MemoryManager
│   └── document_processing/
│       └── test_processor.py         # DocumentProcessor (tokens, extract, chunks)
├── integration/         # Tests with mocked component interaction
│   └── test_reactbot_flow.py
├── e2e/                 # Real API calls, skip if key missing
│   ├── conftest.py      # ← MODEL CONFIGURATION (change model name here)
│   ├── test_reactbot.py
│   ├── test_orchestratorbot.py
│   └── test_classifiers.py           # Classifier e2e tests (10 tests)
└── fixtures/            # Custom test fixtures (if needed)
```

The former `tests/ultimate/` stress suite was removed. Measurable model/agent
comparison now lives in the standalone **`benchmarks/`** harness (real API keys,
not pytest) — see `benchmarks/README.md`. Run it with `python benchmarks/run.py`.

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
├── IConversationBot    — ReactBot
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

result = bot.get_response(...)   # ReactBot
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
result.run_id           # Optional[str] — unique process id (UTC timestamp + full UUID4, never repeats)
result.node_trace       # List[dict]  — ordered node executions [{node, run_id, seq, ts, detail}, …]
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

# GraphTopologyEvent — first event of a run: node/edge layout, for drawing the graph
ev: GraphTopologyEvent = {"type": "graph_topology", "run_id": "...", "entry": "agent",
                          "nodes": [...], "edges": [{"source", "target", "conditional"}]}

# NodeInvokedEvent — fired once per node execution, in order (animate the path)
ev: NodeInvokedEvent = {"type": "node_invoked", "run_id": "...", "node": "tools", "seq": 2, "ts": 175.0,
                        "detail": {"tools_executed": [{"tool_name": "...", "args": {...}, "status": "success", "output": "..."}]}}
```

Import these types in consumers instead of hardcoding dict keys.

### Core Abstractions (`src/sonika_ai_toolkit/utilities/`)

- `types.py`: `BotResponse` (unified response), `ILanguageModel`, `Message`, `ResponseModel`, `IEmbeddings`, `FileProcessorInterface`
- `models.py`: `OpenAILanguageModel`, `GeminiLanguageModel`, `BedrockLanguageModel`, `DeepSeekLanguageModel`, `AnthropicLanguageModel`. All expose `predict()`, `invoke()`, and `stream_response()`. All provider imports (`ChatOpenAI`, `ChatGoogleGenerativeAI`, `ChatBedrock`, `ChatAnthropic`) are at module level — required for correct `unittest.mock.patch` behavior in tests.

### Agents (`src/sonika_ai_toolkit/agents/`)

Two architectures for different use cases:

1. **ReactBot** (`react.py`): Standard ReAct loop via LangGraph. Implements `IConversationBot`. Handles tool execution, token tracking, `_InternalToolLogger` callback. Returns `BotResponse`.

2. **OrchestratorBot** (`orchestrator/`): Autonomous ReAct-based orchestration with persistent memory, LangGraph native interrupts, rate-limit retry with event propagation, and async-first streaming API. Implements `IOrchestratorBot`.

### OrchestratorBot — How it works

The orchestrator runs a **compact ReAct loop** compiled as a LangGraph state
machine. Two always-present nodes plus two opt-in nodes:

```
agent ──(plan signals?)──► plan ──┐ (enable_planning=True)
  │                          │    └─(no real calls)─► agent
  ├──(ask_user?)──► ask_user ◄┘ (enable_user_questions=True) ─► agent
  ├──(real tool_calls?)──► tools ──► agent
  └──(no tool calls)──► END
```

Each `run(goal)` call:

1. **`agent_node`** — streams the LLM response via `astream`, accumulates tool calls and thinking content. A tenacity retry decorator wraps the stream call; on 429/rate-limit it waits exponentially (up to 5 attempts) and emits `StatusEvent` objects into `state["status_events"]` so consumers can show progress. Routing (`should_continue`): plan signals → `plan`, ask_user call → `ask_user`, real calls → `tools`, none → END.
2. **`plan` node** (opt-in) — answers `set_plan`/`update_step` with acknowledgment ToolMessages and applies them to the plan snapshot; then routes remaining real calls to `tools`/`ask_user`, or straight back to `agent`.
3. **`ask_user` node** (opt-in) — fires the **question_request interrupt** and waits for `set_resume_command()`; asking wins: real calls in the same batch get a deferred ToolMessage (never executed). The interrupt is the node's first side effect, so resuming re-executes nothing else.
4. **`tools_node`** — iterates the real tool calls (finds the batch via `_last_tool_call_message` — after plan/ask_user, `messages[-1]` is a ToolMessage); if `mode="ask"` and `tool.risk_level > 0`, fires a **native LangGraph interrupt** and waits for `set_resume_command()` before executing. Skips plan-signal calls (already answered by `plan`).
5. Loop continues until the model stops calling tools, then final text is stored as `final_report`.

**Key APIs:**
- `bot.run(goal, context="")` — synchronous (wraps `arun`)
- `await bot.arun(goal)` — async, runs in `"auto"` mode (no interrupts)
- `async for stream_mode, payload in bot.astream_events(goal, mode="ask")` — streaming with interrupt support
- `bot.set_resume_command(resume_data)` — provide approval data after an interrupt
- `await bot.a_prewarm()` — pre-warm TCP/TLS connection
- `bot.get_graph_topology()` — static node/edge layout of the compiled graph
- `memory_path` — directory for MEMORY.md and session logs

**Graph topology + node events (both bots):**

Every run gets a unique `run_id` (UTC timestamp + full UUID4 — never repeats).
All graph nodes are wrapped by `_wrap_node_traced` (react.py) so each execution
appends to the `node_trace` state channel (`operator.add`). OrchestratorBot's
`astream_events` yields a third stream mode `"graph"`: first a
`GraphTopologyEvent` (only when a `goal` starts a new run), then one
`NodeInvokedEvent` per node execution (synthesized from the `"updates"`
payloads; `run_id`/`ts`/`detail` come from the node's `node_trace` delta so
they stay consistent across interrupt resumes). Each event carries a `detail`
(`NodeDetail`) with the node's params/output — `tool_calls` (name+args
requested), `tools_executed` (args + truncated output), `output` (text),
`plan`/`step_events`/`questions` — built generically from the node's state
delta by `_node_detail` (react.py). ReactBot's `stream_response` yields the
same as `{"type": "graph"}` / `{"type": "node"}` chunks (its graph.stream now
uses `stream_mode=["updates", "values"]`). Non-streaming (`run`/`arun`/
`get_response`) return `BotResponse.node_trace` + `BotResponse.run_id`.
Topology comes from `compiled_graph.get_graph()` via `_graph_topology`
(react.py).

**Mode parameter:**
- `"ask"` (default) — pauses on risky tool calls via LangGraph interrupt
- `"auto"` — executes all tools without interrupting
- `"plan"` — forces the model to return a plan as text (tool calls stripped)

**Structured plan + step progress (`enable_planning=True`, opt-in):**

Planning runs through a **dedicated `plan` node**, added to the graph only
when `enable_planning=True`. It activates only when (1) `enable_planning=True`
at construction, (2) `mode != "plan"`, and (3) the model itself chooses to
call the signal tools (never forced by the graph). Registers two internal
*signal* tools (`set_plan`, `update_step` in `tools/plan_tools.py`) and
appends a planning protocol to the system prompt (`orchestrator/planning.py`).
The router sends batches containing signal calls to the `plan` node, which
applies them and emits state deltas that surface in the `"updates"` stream:
`{"plan": {"plan": [PlanStep, ...]}}` and `{"plan": {"step_events":
[StepEvent, ...]}}` (up to v0.3.14 these streamed under `"agent"`). The calls
stay in the message: the `plan` node answers them with a no-op acknowledgment
ToolMessage (excluded from `tools_executed`) so the history remains a normal
tool round-trip — models return empty when the conversation ends on a bare AI
message, so do NOT strip the calls; `tools_node` skips them. `run`/`arun`
return the final snapshot in `BotResponse.plan`. With
`enable_planning=False` (default) the node does not exist and stream payloads
are unchanged. Text-only `mode="plan"` is unaffected (planning protocol not
appended there).

The graph wiring is fixed (built in `_build_workflow`) and not customizable
by consumers — there is no mechanism to inject nodes or override node routing.

**Skills (both bots — `skills=[Skill, ...]` and/or `skills_dir="./skills"`):**

Folder-based capability packs (`src/sonika_ai_toolkit/skills/loader.py`): each
subfolder has a `SKILL.md` (optional `---` frontmatter `name:`/`description:`;
body = instructions) and optional `tools.py` (BaseTool subclasses defined in
the file are instantiated — imported classes ignored). Instructions render as
a `## SKILLS` block appended to the system prompt (ReactBot
`_build_system_prompt`, Orchestrator `agent_node`);
tools merge into the bot's list **before** bind_tools, deduped by name with
explicitly-passed tools winning. Broken skills are logged and skipped.
`tools.py` executes arbitrary Python — only trusted directories. NOTE:
OrchestratorBot's `self.skills_dir` *attribute* (memory-derived, for
DynamicToolSynthesizer) is unrelated to the `skills_dir` constructor param.

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
- **Anthropic** (`AnthropicLanguageModel`, `ChatAnthropic` from `langchain-anthropic`): Default model `claude-haiku-4-5`. Anthropic **requires** `max_tokens` (default 4096). Pass `thinking_budget=N` to enable extended thinking — this forces `temperature=1` (with a warning) and bumps `max_tokens` above the budget. Like Gemini, thinking responses return `content` as a list of blocks; `invoke()`/`stream_response()` strip `type == "thinking"` blocks.
- **Mock patching**: All provider imports are module-level in `models.py` — `patch("sonika_ai_toolkit.utilities.models.ChatOpenAI")` (and `ChatAnthropic`) works correctly.

### Public API (`src/sonika_ai_toolkit/__init__.py`)

Top-level imports for the most common components:

```python
from sonika_ai_toolkit import (
    # Orchestrator
    OrchestratorBot, IOrchestratorBot,
    # Agent interfaces
    IBot, IConversationBot,
    # Skills + custom nodes + wiring overrides
    Skill, load_skills,
    # Stream event types
    AgentUpdate, ToolsUpdate, ToolRecord, StatusEvent, PartialResponseEvent,
    PlanStep, StepEvent,
    GraphTopologyEvent, NodeInvokedEvent, GraphEdgeSpec, NodeTraceEntry, NodeDetail,
    # Response type
    BotResponse, ILanguageModel,
    # LLM providers
    GeminiLanguageModel, OpenAILanguageModel,
    BedrockLanguageModel, DeepSeekLanguageModel,
    AnthropicLanguageModel,
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
