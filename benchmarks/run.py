#!/usr/bin/env python
"""Benchmark CLI — compare agents and models on measurable scenarios.

Examples
--------
    # One agent, one model (defaults)
    python benchmarks/run.py

    # Compare several models on ReactBot
    python benchmarks/run.py --models openai:gpt-4o-mini,anthropic:claude-haiku-4-5

    # Full matrix: every agent × two models
    python benchmarks/run.py \
        --agents react,orchestrator \
        --models openai:gpt-4o-mini,gemini:gemini-2.5-flash

Model spec is ``provider:model_name`` (provider alone uses a default model).
Providers: openai, gemini, deepseek, anthropic. API keys come from the
environment / .env (OPENAI_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY,
ANTHROPIC_API_KEY).
"""

import argparse
import os
import sys

# Make ``benchmarks`` importable when run directly (python benchmarks/run.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from benchmarks.core.adapters import AGENT_KINDS  # noqa: E402
from benchmarks.core.models import available_providers  # noqa: E402
from benchmarks.core.report import to_markdown, write_reports  # noqa: E402
from benchmarks.core.runner import run_matrix  # noqa: E402
from benchmarks.scenarios.customer_support import SCENARIOS  # noqa: E402

_DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def _csv(value: str) -> list:
    return [item.strip() for item in value.split(",") if item.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", type=_csv, default=["react"],
                        help=f"Comma-separated agents to run. Options: {','.join(AGENT_KINDS)}")
    parser.add_argument("--models", type=_csv, default=["openai:gpt-4o-mini"],
                        help="Comma-separated model specs 'provider:model_name'.")
    parser.add_argument("--scenarios", type=_csv, default=["all"],
                        help="Scenario ids to run, or 'all'.")
    parser.add_argument("--out", default=_DEFAULT_OUT,
                        help="Directory for the generated reports.")
    parser.add_argument("--list", action="store_true",
                        help="List available agents / providers / scenarios and exit.")
    args = parser.parse_args(argv)

    if args.list:
        print("Agents:    ", ", ".join(AGENT_KINDS))
        print("Providers: ", ", ".join(available_providers()))
        print("Scenarios: ", ", ".join(s.id for s in SCENARIOS))
        return 0

    # Validate agent selection early.
    bad = [a for a in args.agents if a not in AGENT_KINDS]
    if bad:
        parser.error(f"Unknown agent(s): {bad}. Options: {AGENT_KINDS}")

    # Resolve scenarios.
    if args.scenarios == ["all"]:
        scenarios = SCENARIOS
    else:
        by_id = {s.id: s for s in SCENARIOS}
        missing = [sid for sid in args.scenarios if sid not in by_id]
        if missing:
            parser.error(f"Unknown scenario id(s): {missing}. "
                         f"Available: {sorted(by_id)}")
        scenarios = [by_id[sid] for sid in args.scenarios]

    total = len(args.agents) * len(args.models) * len(scenarios)
    print(f"Running {total} combinations "
          f"({len(args.agents)} agents × {len(args.models)} models × {len(scenarios)} scenarios)...\n")

    counter = {"n": 0}

    def progress(agent, model, scenario_id):
        counter["n"] += 1
        print(f"  [{counter['n']}/{total}] {agent} · {model} · {scenario_id}")

    try:
        results = run_matrix(args.agents, args.models, scenarios, on_progress=progress)
    except ValueError as exc:  # bad model spec / missing key — fail clearly
        parser.error(str(exc))

    print("\n" + to_markdown(results, scenarios))

    paths = write_reports(results, scenarios, args.out)
    print(f"\nReports written:\n  {paths['markdown']}\n  {paths['html']}\n  {paths['json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
