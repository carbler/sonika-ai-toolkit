# Test Suite for sonika-ai-toolkit

Comprehensive, CI-ready test suite for the sonika-ai-toolkit library. All external dependencies (LLMs, HTTP, filesystem) are mocked to ensure fast, deterministic, and isolated tests.

## Quick Start

### Run All Tests
```bash
pytest
```

### Run Tests by Category
```bash
# Unit tests only (fast, ~10-20s)
pytest -m unit

# Integration tests only
pytest -m integration

# Specific test file
pytest tests/unit/test_models.py

# Specific test class
pytest tests/unit/test_models.py::TestOpenAILanguageModel

# Specific test
pytest tests/unit/test_models.py::TestOpenAILanguageModel::test_init_default_model_name
```

### Run with Coverage
```bash
pytest --cov=src/sonika_ai_toolkit --cov-report=html
open htmlcov/index.html  # macOS
```

### Run in Watch Mode
```bash
pytest-watch
```

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures and utilities
├── unit/
│   ├── test_models.py            # Language model initialization (52 tests)
│   ├── test_types.py             # Type definitions and interfaces (14 tests)
│   ├── test_think_helpers.py     # ThinkBot helpers and extraction (24 tests)
│   ├── test_react_logger.py      # Tool logger callback handling (19 tests)
│   ├── test_classifiers.py       # TextClassifier with structured output (8 tests)
│   └── test_tools.py             # Tool definitions (14 tests)
├── integration/
│   ├── test_reactbot_flow.py     # ReactBot end-to-end workflow (15 tests)
│   └── test_thinkbot_flow.py     # ThinkBot end-to-end workflow (18 tests)
├── fixtures/                      # Custom test fixtures (if needed)
├── pytest.ini                      # Pytest configuration
└── README.md                       # This file
```

## Test Coverage by Component

### Unit Tests (131 tests)

| Module | Coverage | Tests |
|--------|----------|-------|
| `sonika_ai_toolkit.utilities.models` | 100% | 24 tests |
| `sonika_ai_toolkit.utilities.types` | 100% | 8 tests |
| `sonika_ai_toolkit.agents.think` | 100% | 24 tests |
| `sonika_ai_toolkit.agents.react` | 100% | 19 tests |
| `sonika_ai_toolkit.classifiers.text` | 100% | 8 tests |
| `sonika_ai_toolkit.tools.integrations` | 100% | 14 tests |

### Integration Tests (33 tests)

| Module | Tests |
|--------|-------|
| ReactBot workflow | 15 tests |
| ThinkBot workflow | 18 tests |

## Key Features

### ✅ No Real API Calls
All LLM SDKs (OpenAI, DeepSeek, Google Generative AI, Bedrock) are patched with `unittest.mock`. Tests run without API keys.

### ✅ Deterministic Results
Mocks return pre-configured responses. No network I/O means tests are fast and reproducible.

### ✅ Fixture-Based Design
`conftest.py` provides reusable fixtures:
- `mock_language_model`: Basic mock ILanguageModel
- `mock_thinking_language_model`: Mock with thinking support
- `email_tool`, `save_contact_tool`: Tool fixtures
- `sentiment_model`, `language_model_class`: Pydantic models for classifier tests

### ✅ Comprehensive Coverage
- **Type system**: Message, ResponseModel, abstract interfaces (ILanguageModel, IEmbeddings, FileProcessorInterface)
- **Language models**: OpenAI, DeepSeek, Gemini, Bedrock initialization and parameter passing
- **Thinking extraction**: Gemini list format, DeepSeek reasoning_content, fallback <think> tags
- **Tool execution**: Tool logging, callback execution, error handling
- **Text classification**: Structured output, token counting, validation
- **Bot workflows**: Full agent-loop with tools, streaming, token tracking

## Test Markers

Tests use pytest markers for filtering:

```bash
pytest -m unit        # Run only unit tests
pytest -m integration # Run only integration tests
pytest -m slow        # Run only slow tests
```

Defined in `pytest.ini`:
- `unit`: Fast, isolated tests (no I/O)
- `integration`: Multi-component tests (still fully mocked)
- `slow`: Tests that take longer to run

## Configuration

### pytest.ini
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers --no-header
markers =
    unit: Unit tests (fast, isolated, no I/O)
    integration: Integration tests (component interaction, still mocked)
    slow: Tests that take longer to run
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
    ignore::UserWarning
```

## Writing New Tests

### Basic Test Template
```python
import pytest
from unittest.mock import MagicMock

class TestMyComponent:
    def test_something(self, mock_language_model):
        """Test description."""
        # Arrange
        component = MyComponent(mock_language_model)

        # Act
        result = component.do_something()

        # Assert
        assert result == expected_value
```

### Using Fixtures
```python
def test_with_fixture(self, mock_language_model, email_tool, sentiment_model):
    """Use multiple fixtures in one test."""
    pass
```

### Parametrized Tests
```python
@pytest.mark.parametrize("model_name,expected", [
    ("gpt-4o-mini", False),
    ("deepseek-reasoner", True),
])
def test_supports_thinking(self, model_name, expected):
    """Test with multiple inputs."""
    assert detect_thinking(model_name) == expected
```

## Mocking Patterns

### Patch External SDK Classes
```python
with patch("sonika_ai_toolkit.utilities.models.ChatOpenAI", autospec=True) as MockChat:
    model = OpenAILanguageModel(api_key="key")
    MockChat.assert_called_once()
```

### Mock Tool Callbacks
```python
on_start = MagicMock()
logger = _InternalToolLogger(on_start=on_start)
logger.on_tool_start({"name": "my_tool"}, "input")
on_start.assert_called_once_with("my_tool", "input")
```

### Mock LLM Response with Metadata
```python
msg = AIMessage(content="answer")
msg.usage_metadata = {"input_tokens": 50, "output_tokens": 30}
mock_model.invoke.return_value = msg
```

## Common Issues & Solutions

### "ModuleNotFoundError: No module named 'sonika_ai_toolkit'"
Install in development mode:
```bash
pip install -e .
```

### "ImportError: cannot import name 'X' from sonika_ai_toolkit"
Ensure the module exists and `__init__.py` files are in place:
```bash
find src -name "__init__.py" | head -10
```

### Test Hangs / Timeout
Check if any test is making real network calls:
- Verify all LLM SDKs are patched in `conftest.py`
- Check for uncaught asyncio calls (use `asyncio.run()` in tests)

### Assertion Fails Due to Mock Call Args
Print the actual call to debug:
```python
print(MockChat.call_args)  # Prints ((args), {kwargs})
_, kwargs = MockChat.call_args
```

## Performance

Typical test suite execution times:
- Unit tests: ~20-30 seconds (no I/O)
- Integration tests: ~10-15 seconds (still mocked)
- Full suite with coverage: ~40-50 seconds

Run time depends on your machine and pytest-xdist parallelization.

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest --cov=src/sonika_ai_toolkit
```

## Best Practices

1. **Mock at SDK boundaries**: Patch LangChain/OpenAI imports, not your code
2. **Use fixtures for reusable setup**: Avoid duplicating mock creation
3. **Test behavior, not implementation**: Assert on outcomes, not mock call counts (unless verifying integration)
4. **Keep tests focused**: One assertion or one clear behavior per test
5. **Use descriptive names**: `test_extract_thinking_from_gemini_list_content` > `test_extract`
6. **Parametrize similar cases**: Use `@pytest.mark.parametrize` for multiple inputs
7. **Isolate side effects**: Reset mock call counts if testing sequential calls

## Extending the Test Suite

### Adding a New Module Test
1. Create `tests/unit/test_new_module.py`
2. Import fixtures from `conftest.py`
3. Use mocking patterns from existing tests
4. Run: `pytest tests/unit/test_new_module.py -v`

### Adding a New Integration Test
1. Create `tests/integration/test_new_flow.py`
2. Use mocked language models (don't hit real APIs)
3. Test full workflow with multiple components
4. Mark with `@pytest.mark.integration`

## References

- [pytest documentation](https://docs.pytest.org/)
- [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
- [LangChain testing](https://python.langchain.com/docs/modules/agents/tools/testing_tools/)
