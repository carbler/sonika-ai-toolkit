# Benchmarks

A standalone harness to compare **agents** (ReactBot, OrchestratorBot)
and **models** (OpenAI, Gemini, DeepSeek, Anthropic) under **measurable,
reproducible** conditions. It is **not** part of the pytest suite — it makes real
API calls and needs keys.

## What it measures

| Dimension | How |
|---|---|
| **Task success** | Declarative `checks` per scenario → pass/fail + score (fraction of checks passed). |
| **Tool-call accuracy** | Name-based precision / recall / **F1** of `tools_executed` vs `expected_tools`. |
| Tokens / latency / steps | Captured for free from `BotResponse.token_usage` and the wall clock (informational). |

Every tool response is **deterministic** (`tools/support_tools.py`) — no network,
randomness or clock — so a run's outcome is a function of the *model's decisions
alone*. That's what makes results comparable across models and agents.

## Usage

```bash
# One agent, one model (defaults: react × openai:gpt-4o-mini × all scenarios)
python benchmarks/run.py

# Compare several models on the same agent
python benchmarks/run.py --models openai:gpt-4o-mini,anthropic:claude-haiku-4-5

# Full matrix: every agent × two models
python benchmarks/run.py \
    --agents react,orchestrator \
    --models openai:gpt-4o-mini,gemini:gemini-2.5-flash

# Run a subset of scenarios
python benchmarks/run.py --scenarios profile_lookup,fraud_block

# Discover options
python benchmarks/run.py --list
```

Model spec is `provider:model_name` (provider alone uses a default model).
Providers: `openai`, `gemini`, `deepseek`, `anthropic`. Keys come from the
environment / `.env`: `OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`,
`ANTHROPIC_API_KEY`.

## Output

Each run writes a timestamped trio to `benchmarks/results/` (git-ignored):

- **`report_*.md`** — a **comparative summary table** (agent × model: pass rate,
  avg score, avg tool F1, tokens, latency) plus per-scenario detail.
- **`report_*.html`** — the same, self-contained (inline CSS, no external assets)
  with pass/fail/error badges and score bars — open it in a browser.
- **`report_*.json`** — machine-readable, for diffing runs or CI.

## Layout

```
benchmarks/
├── run.py                       # CLI entrypoint (agent × model × scenario matrix)
├── core/
│   ├── models.py                # "provider:model" -> ILanguageModel
│   ├── adapters.py              # normalize React/Orchestrator behind one call
│   ├── scenario.py              # Scenario / RunResult dataclasses
│   ├── checks.py                # declarative check factories (called, content_contains, …)
│   ├── runner.py                # matrix loop + precision/recall/F1
│   └── report.py                # Markdown + JSON writers
├── tools/support_tools.py       # deterministic mock tools
└── scenarios/customer_support.py# scenario definitions
```

## Extending

- **New scenario** — add a `Scenario(...)` to `scenarios/customer_support.py`
  (or a new module) with `goal`, `tools`, `expected_tools` and `checks`.
- **New check** — add a factory to `core/checks.py` that returns
  `(response) -> CheckOutcome`. Keep it based on `content` / `tools_executed`
  so it scores identically across all agents.
- **New model provider** — add it to `_PROVIDERS` in `core/models.py`.
- **New tool** — add a `BaseTool` (with `args_schema`) to
  `tools/support_tools.py`; keep `_run` deterministic.

### Note on tool-arg checks
Tool-call accuracy is **name-based** on purpose: ReactBot's `tools_executed`
records don't include call `args`, while OrchestratorBot does. Scoring on
names keeps the comparison fair across both agents.
