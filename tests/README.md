# Test Suite for sonika-ai-toolkit

CI-ready test suite. Every external dependency (LLM SDKs, HTTP, SMTP, DB drivers)
is mocked, so `unit` and `integration` tests are fast, deterministic, and need no
API keys. Only `e2e` tests hit real providers.

## Quick Start

```bash
# Unit + integration (no keys, ~5s)
pytest tests/unit tests/integration -q

# By category (markers are applied automatically by directory)
pytest -m unit
pytest -m integration
pytest -m e2e            # real API keys required (see e2e/conftest.py)

# A single module / class / test
pytest tests/unit/utilities/test_models.py
pytest tests/unit/utilities/test_models.py::TestOpenAILanguageModel
pytest tests/unit/tools/test_registry.py::TestToolDescriptions::test_empty_registry

# Lint
ruff check tests/unit tests/integration
```

## Structure

`tests/unit/` mirrors `src/sonika_ai_toolkit/` 1:1 — one test file per module.

```
tests/
├── conftest.py                       # Shared fixtures + auto-marking hook
├── unit/
│   ├── utilities/
│   │   ├── test_models.py            # OpenAI / DeepSeek / Gemini / Bedrock / Anthropic wrappers
│   │   ├── test_types.py             # BotResponse, ILanguageModel, Message, ResponseModel
│   │   └── test_questions.py         # ask_user contract (schema / payload / summary)
│   ├── classifiers/
│   │   └── test_classifiers.py       # Text / Intent / Sentiment / Safety / Image
│   ├── tools/
│   │   ├── test_core_tools.py        # bash, files, http, python, search, web, datetime, email
│   │   ├── test_database_tools.py    # SQLite, PostgreSQL, MySQL, Redis
│   │   ├── test_integrations.py      # EmailTool, SaveContacto
│   │   ├── test_ask_user.py          # AskUserQuestionTool
│   │   ├── test_registry.py          # ToolRegistry
│   │   └── test_synthesizer.py       # DynamicToolSynthesizer
│   ├── agents/
│   │   ├── test_react.py             # _InternalToolLogger + ReactBot ask_user flow
│   │   ├── test_tasker.py            # TaskerBot construction / get_response / limits
│   │   └── orchestrator/
│   │       ├── test_contract.py      # Interface contract (IBot / IConversationBot / IOrchestratorBot)
│   │       ├── test_graph.py         # agent/tools graph, partial-response filtering
│   │       ├── test_risk.py          # risk-gate helpers (should_auto_approve, format_approval_prompt)
│   │       └── test_memory.py        # MemoryManager (MEMORY.md / SKILLS.md / sessions)
│   └── document_processing/
│       └── test_processor.py         # DocumentProcessor (count_tokens, extract, chunks)
├── integration/
│   └── test_reactbot_flow.py         # ReactBot end-to-end (mocked LLM)
├── e2e/                              # Real API calls — skipped when keys are missing
│   ├── conftest.py                   # ← model configuration lives here
│   ├── test_reactbot.py
│   ├── test_orchestratorbot.py
│   └── test_classifiers.py
└── ultimate/                        # Standalone stress runners (not pytest)
```

## Markers

Markers are **not** declared per-file. `tests/conftest.py` has a
`pytest_collection_modifyitems` hook that marks every test by its location:
`unit/` → `unit`, `integration/` → `integration`, `e2e/` → `e2e`. Adding a new
test under the right directory is all that's needed for `pytest -m <marker>` to
pick it up.

## Shared fixtures (`conftest.py`)

- `mock_raw_model` — MagicMock mimicking a LangChain ChatModel (`bind_tools`,
  `with_structured_output`, `invoke`, `stream` preconfigured)
- `mock_language_model` — `ILanguageModel` wrapping `mock_raw_model`
- `email_tool`, `save_contact_tool`, `all_tools` — tool fixtures
- `sample_messages`, `empty_messages`, `sample_logs`, `empty_logs`
- `sentiment_model`, `language_model_class` — Pydantic schemas for classifier tests

Real I/O in unit tests is confined to pytest's `tmp_path`.

## Conventions

1. **One test file per source module**, mirroring the `src/` path.
2. **Mock at SDK boundaries** — patch `sonika_ai_toolkit.utilities.models.ChatOpenAI`
   (imports are module-level for exactly this reason), driver modules via
   `patch.dict("sys.modules", ...)`, and network via `patch("requests.get", ...)`.
3. **Behavior over implementation** — assert on outcomes; verify mock call args
   only when the call itself is the contract (e.g. `starttls`, `sendmail`).
4. **Descriptive names** — `test_set_with_ttl_uses_setex` over `test_set2`.
5. **Parametrize** similar cases with `@pytest.mark.parametrize`.

## References

- [pytest](https://docs.pytest.org/) · [unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
