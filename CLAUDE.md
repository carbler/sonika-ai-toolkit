# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in development mode
pip install -e .

# Run all tests (unit + integration)
pytest

# Run unit tests only (fast, ~20-30s)
pytest -m unit

# Run integration tests only
pytest -m integration

# Run a single test file
pytest tests/unit/test_models.py

# Run with coverage report
pytest --cov=src/sonika_ai_toolkit --cov-report=html

# Run specific test
pytest tests/unit/test_models.py::TestOpenAILanguageModel::test_init_default_model_name

# Build distribution
python setup.py sdist bdist_wheel

# Lint
ruff check .
```

## Testing

### Test Structure
- **164 total tests**: 131 unit tests + 33 integration tests
- **100% mocked**: No real API calls; all external dependencies (OpenAI, DeepSeek, Gemini, Bedrock) are patched
- **Fixture-based**: Shared mocks in `tests/conftest.py` reduce duplication
- **Marked by category**: `@pytest.mark.unit` and `@pytest.mark.integration` for filtering

### Key Test Files
- `tests/unit/test_models.py` (24 tests): LLM initialization (temperature, token tracking, thinking detection)
- `tests/unit/test_types.py` (8 tests): Type system and interfaces
- `tests/unit/test_think_helpers.py` (24 tests): Thinking extraction from Gemini list format, DeepSeek reasoning_content, and `<think>` tags
- `tests/unit/test_react_logger.py` (19 tests): Tool callback handler resilience
- `tests/unit/test_classifiers.py` (8 tests): TextClassifier with structured output
- `tests/unit/test_tools.py` (14 tests): Tool definitions (EmailTool, SaveContact)
- `tests/integration/test_reactbot_flow.py` (15 tests): Full ReactBot workflow with history and callbacks
- `tests/integration/test_thinkbot_flow.py` (18 tests): Full ThinkBot workflow including streaming and token extraction fallback

### Mocking Patterns
Tests use `unittest.mock.patch` to intercept LLM SDK imports:
```python
# In conftest.py: pre-built fixtures
@pytest.fixture
def mock_language_model(mock_raw_model):
    return _MockLanguageModel(mock_raw_model)

# In tests: use mocks without real keys
lm = OpenAILanguageModel(api_key="test-key")  # Patched, no real call
```

### Test Markers
```bash
pytest -m unit        # Fast, isolated tests
pytest -m integration # Component-interaction tests (still mocked)
pytest -m slow        # Slower tests
```

For detailed testing guide, see `tests/README.md`.

## Architecture

**sonika-ai-toolkit** is a Python library for building conversational AI agents using LangChain and LangGraph, with multi-provider LLM support (OpenAI, DeepSeek, Google Gemini, Amazon Bedrock).

### Core Abstractions (`src/sonika_ai_toolkit/utilities/`)

- `types.py`: Core interfaces — `ILanguageModel` (abstract base for all LLMs), `Message`, `ResponseModel` (wraps token usage + response), `IEmbeddings`, `FileProcessorInterface`
- `models.py`: Concrete LLM implementations — `OpenAILanguageModel`, `GeminiLanguageModel`, `BedrockLanguageModel`, `DeepSeekLanguageModel`. All expose `predict()`, `invoke()`, and `stream_response()`.

### Agents (`src/sonika_ai_toolkit/agents/`)

Three architectures for different use cases:

1. **ReactBot** (`react.py`): Standard ReAct loop via LangGraph. Handles tool execution, token tracking, and an internal `_InternalToolLogger` callback. The reference implementation for understanding the state/graph pattern.

2. **ThinkBot** (`think.py`): Extends ReactBot with explicit reasoning separation. Extracts thinking from Gemini native thinking, DeepSeek R1 reasoner output, or fallback `<think>` tags. Supports streaming with chunked output.

3. **TaskerBot** (`tasker/`): Planner→Executor→Validator→Output→Logger graph for complex multi-step tasks. Has dedicated node files under `tasker/nodes/` and prompt templates under `tasker/prompts/`. Uses `ChatState` (defined in `tasker/state.py`) with planning/execution attempt tracking and recursion limits.

### Other Components

- `classifiers/text.py`: `TextClassifier` — structured LLM output classification returning `ClassificationResponse` with typed results
- `document_processing/`: `DocumentProcessor` for PDF, DOCX, XLSX, PPTX, TXT with token counting (tiktoken) and chunking
- `tools/integrations.py`: Custom LangChain tools (`EmailTool`, `SaveContact`) using `BaseTool` + Pydantic `BaseModel`

### Provider-Specific Gotchas

- **Gemini**: System message must be first; messages must alternate User/AI. Thinking models require `temperature=1.0`.
- **DeepSeek reasoner** (`deepseek-reasoner`): Does not support tool calling.
- **LangGraph**: Requires a checkpointer (e.g., `MemorySaver`) for state persistence.
- **Bedrock**: Configurable via region and model ID parameters.

### Package Exports

Public API is controlled through `__init__.py` files at each layer. The top-level `src/sonika_ai_toolkit/agents/__init__.py` exports `ReactBot`, `ThinkBot`, and `TaskerBot`.
