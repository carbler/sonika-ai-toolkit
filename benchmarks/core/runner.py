"""Benchmark runner — the agent × model × scenario matrix."""

import time

from benchmarks.core.adapters import run_agent
from benchmarks.core.models import build_model, normalize_spec
from benchmarks.core.scenario import RunResult


def _prf(predicted, expected):
    """Name-based precision / recall / F1 over tool-call sets."""
    pred, exp = set(predicted), set(expected)
    if not exp:
        # No tools expected: recall is trivially satisfied; precision (and thus
        # F1) is 1.0 only if the agent correctly called nothing.
        clean = not pred
        return (1.0 if clean else 0.0), 1.0, (1.0 if clean else 0.0)
    if not pred:
        return 0.0, 0.0, 0.0
    hits = len(pred & exp)
    precision = hits / len(pred)
    recall = hits / len(exp)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _evaluate(scenario, response, agent, model_label, latency_s) -> RunResult:
    tools_executed = response.get("tools_executed") or []
    predicted = [t.get("tool_name") for t in tools_executed]

    outcomes = [check(response) for check in scenario.checks]
    passed = sum(1 for o in outcomes if o.passed)
    total = len(outcomes)

    precision, recall, f1 = _prf(predicted, scenario.expected_tools)

    token_usage = response.get("token_usage") or {}

    return RunResult(
        scenario_id=scenario.id,
        agent=agent,
        model=model_label,
        success=(total > 0 and passed == total),
        score=(passed / total) if total else 0.0,
        checks=outcomes,
        predicted_tools=predicted,
        expected_tools=list(scenario.expected_tools),
        tool_precision=precision,
        tool_recall=recall,
        tool_f1=f1,
        tokens=token_usage.get("total_tokens", 0),
        latency_s=latency_s,
        num_tool_calls=len(predicted),
    )


def run_matrix(agents, model_specs, scenarios, on_progress=None):
    """Run every (agent, model, scenario) combination.

    Models are built once per spec and reused across agents/scenarios. A crash
    in one combination is captured as an errored RunResult; the matrix keeps going.
    """
    # Build models up front so a bad spec / missing key fails fast.
    models = {}
    for spec in model_specs:
        label = normalize_spec(spec)
        models[label] = build_model(spec)

    results = []
    for agent in agents:
        for label, model in models.items():
            for scenario in scenarios:
                if on_progress:
                    on_progress(agent, label, scenario.id)
                t0 = time.perf_counter()
                try:
                    response = run_agent(agent, model, scenario)
                    latency = time.perf_counter() - t0
                    result = _evaluate(scenario, response, agent, label, latency)
                except Exception as exc:  # noqa: BLE001 — record and continue
                    result = RunResult(
                        scenario_id=scenario.id, agent=agent, model=label,
                        latency_s=time.perf_counter() - t0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                results.append(result)
    return results
