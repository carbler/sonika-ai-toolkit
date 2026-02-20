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

# Run stress test suite (banking scenario, interactive)
python tests/ultimate/banking_operations/batch_runner.py

# Lint
ruff check .
```

## Testing

### Test Structure

```
tests/
├── conftest.py          # Shared mocks — no API keys needed
├── unit/                # 153 tests — isolated, ~3s
│   └── test_orchestrator_prompts.py  # Prompt injection tests (26 tests)
├── integration/         # 15 tests — mocked, ~3s
├── e2e/                 # 9 tests — real API calls, skip if key missing
│   ├── conftest.py      # ← MODEL CONFIGURATION (change model name here)
│   ├── test_reactbot.py
│   └── test_orchestratorbot.py
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
- Both `EmailTool` and `SaveContact` appear in `result.tools_executed`
- `result.success` is `True` (OrchestratorBot)
- Approval callbacks are invoked when `risk_threshold=-1`

## Architecture

**sonika-ai-toolkit** is a Python library for building conversational AI agents using LangChain and LangGraph, with multi-provider LLM support (OpenAI, DeepSeek, Google Gemini, Amazon Bedrock).

### Unified Response Type (`BotResponse`)

All agents return a `BotResponse` — a `dict` subclass that is **fully backward-compatible** with plain dict access and also exposes typed properties:

```python
from sonika_ai_toolkit.utilities.types import BotResponse

result = bot.get_response(...)   # ReactBot / TaskerBot
result = bot.run(...)            # OrchestratorBot

# dict-style (existing code unchanged)
result["content"]
result.get("thinking")

# property-style (new, typed)
result.content          # str
result.thinking         # Optional[str]
result.logs             # List[str]
result.tools_executed   # List[dict]
result.token_usage      # {prompt_tokens, completion_tokens, total_tokens}
result.success          # bool (OrchestratorBot; defaults True for others)
result.plan             # List[dict] (OrchestratorBot)
result.session_id       # Optional[str] (OrchestratorBot)
result.goal             # Optional[str] (OrchestratorBot)
```

### Core Abstractions (`src/sonika_ai_toolkit/utilities/`)

- `types.py`: `BotResponse` (unified response), `ILanguageModel`, `Message`, `ResponseModel`, `IEmbeddings`, `FileProcessorInterface`
- `models.py`: `OpenAILanguageModel`, `GeminiLanguageModel`, `BedrockLanguageModel`, `DeepSeekLanguageModel`. All expose `predict()`, `invoke()`, and `stream_response()`.

### Agents (`src/sonika_ai_toolkit/agents/`)

Four architectures for different use cases:

1. **ReactBot** (`react.py`): Standard ReAct loop via LangGraph. Handles tool execution, token tracking, `_InternalToolLogger` callback. Returns `BotResponse`.

2. **ThinkBot** (`think.py`): Extends ReactBot with explicit reasoning separation. Extracts thinking from Gemini native thinking, DeepSeek R1 reasoner output, or fallback `<think>` tags. *(module not currently bundled — import-skipped automatically)*

3. **TaskerBot** (`tasker/`): Planner→Executor→Validator→Output→Logger graph for complex multi-step tasks. Nodes under `tasker/nodes/`. Returns `BotResponse`.

4. **OrchestratorBot** (`orchestrator/`): Autonomous orchestration with persistent memory (MEMORY.md), per-step risk gate + human approval callback, multi-model routing (strong/fast/code), structured retry with anti-loop protection, and dynamic tool synthesis. Returns `BotResponse`.
   - `bot.run(goal, context="")` — synchronous entry point
   - `memory_path` — directory for MEMORY.md, SKILLS.md, session logs
   - `risk_threshold=-1` forces all steps through `on_human_approval` callback
   - `prompts=OrchestratorPrompts(...)` — injectable prompt templates (see below)

### OrchestratorBot — How it works

The orchestrator runs a **LangGraph state machine** with 11 nodes. Each `run(goal)` call:

```
load_memory → planner → step_dispatcher → risk_gate → [human_approval] → executor
                                    ↑                                         ↓
                             escalate ← retry ←──────────────────────── evaluator
                                                                              ↓
                                                             reporter → save_memory → END
```

**Execution flow per step:**
1. `load_memory` — reads MEMORY.md + SKILLS.md, registers any previously synthesized tools
2. `planner` (strong_model) — decomposes the goal into a JSON step list
3. `step_dispatcher` — picks the next pending step; routes to `reporter` when all done
4. `risk_gate` — compares `step.risk_level` vs `risk_threshold`; routes to `human_approval` if above threshold
5. `human_approval` — calls `on_human_approval(step)` callback; skips step if returns False
6. `executor` — invokes the tool; supports `synth_tool` strategy (generates a new tool at runtime)
7. `evaluator` (fast_model) — judges step success and whether the overall goal is already met
8. `retry` (fast_model) — on failure: picks `retry_params | alt_tool | synth_tool | escalate`; anti-loop hash check
9. `escalate` — marks step as failed, resets retry counters, advances to next step
10. `reporter` (fast_model) — writes the final 2-5 paragraph report
11. `save_memory` (fast_model) — generates 2-bullet summary, appends to MEMORY.md

**Model routing:**
- `strong_model` — used only by `planner`
- `fast_model` — used by `evaluator`, `retry`, `reporter`, `save_memory`
- `code_model` — used by `DynamicToolSynthesizer` when a `synth_tool` strategy is chosen

**Risk levels:** 0=safe (read-only), 1=low, 2=medium, 3=high (external/financial). Steps at or below `risk_threshold` are auto-approved.

### OrchestratorBot — Prompt Injection

All LLM prompts are externalised in `orchestrator/prompts.py` and injectable via `OrchestratorPrompts`:

```python
from sonika_ai_toolkit.agents.orchestrator.prompts import OrchestratorPrompts
from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot

# Override only the core rules — all other prompts stay default
prompts = OrchestratorPrompts(
    core="You are a financial compliance bot. Never approve transfers above $10k.",
)

# Override a specific stage template (must keep all {placeholders} the template uses)
prompts = OrchestratorPrompts(
    reporter="""
{prompt_a}

## Objective
{goal}

## Steps executed
{plan_summary}

Write a ONE-paragraph executive summary only.
{tool_outputs}
""",
)

bot = OrchestratorBot(
    strong_model=...,
    fast_model=...,
    instructions="...",
    prompts=prompts,
)
```

**`OrchestratorPrompts` fields and their placeholders:**

| Field | Used by node | Required placeholders |
|---|---|---|
| `core` | injected as `{prompt_a}` into all templates | — |
| `planner` | `PlannerNode` | `{prompt_a}` `{instructions}` `{tool_descriptions}` `{memory_context}` `{goal}` `{context}` |
| `evaluator` | `EvaluatorNode` | `{prompt_a}` `{goal}` `{step_description}` `{tool_output}` `{plan_summary}` |
| `retry` | `RetryNode` | `{prompt_a}` `{goal}` `{step_description}` `{error}` `{tool_descriptions}` `{retry_history}` |
| `reporter` | `ReporterNode` | `{prompt_a}` `{goal}` `{plan_summary}` `{tool_outputs}` |
| `save_memory` | `SaveMemoryNode` | `{prompt_a}` `{goal}` `{plan_summary}` |

Node references after construction: `bot._nodes["planner"]`, `bot._nodes["evaluator"]`, etc.

### Tools (`src/sonika_ai_toolkit/tools/`)

- `registry.py`: `ToolRegistry` — register/get/list; `get_tool_descriptions()` includes param names for LLM prompts
- `synthesizer.py`: `DynamicToolSynthesizer` — LLM generates Python `BaseTool` code at runtime, writes to `skills_dir/`
- `core/`: `RunBashTool`, `ReadFileTool`, `WriteFileTool`, `ListDirTool`, `DeleteFileTool`, `CallApiTool`, `SearchWebTool` (opt-in, not auto-registered)
- `integrations.py`: `EmailTool`, `SaveContacto` — must have `args_schema` (Pydantic) for correct LLM param generation

### Other Components

- `classifiers/text.py`: `TextClassifier` — structured output classification
- `document_processing/`: `DocumentProcessor` for PDF, DOCX, XLSX, PPTX, TXT

### Provider-Specific Gotchas

- **Gemini**: System message must be first; messages must alternate User/AI. Thinking models require `temperature=1.0`. `response.content` is a list when `include_thoughts=True` — always use `get_text()` or `ainvoke_with_thinking()`.
- **DeepSeek reasoner** (`deepseek-reasoner`): Does not support tool calling. `reasoning_content` is injected via `_create_chat_result` override.
- **LangGraph**: Requires a checkpointer (e.g., `MemorySaver`) for state persistence.
- **Bedrock**: Configurable via region and model ID parameters.
- **OrchestratorBot**: All LLM nodes use `ainvoke_with_thinking()` from `orchestrator/utils.py` — always returns `AIMessage(content=clean_string)`.

### Package Exports

`src/sonika_ai_toolkit/agents/__init__.py` exports `ReactBot`, `ThinkBot` (None if not installed), `TaskerBot`, `OrchestratorBot`.
`src/sonika_ai_toolkit/utilities/__init__.py` exports `BotResponse`.
