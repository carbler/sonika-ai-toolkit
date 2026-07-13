"""Report writers — comparative Markdown, HTML and machine-readable JSON."""

import html
import json
import os
from collections import defaultdict
from datetime import datetime


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _aggregate(results):
    """Aggregate per (agent, model): pass rate, avg score, avg tool F1, totals."""
    groups = defaultdict(list)
    for r in results:
        groups[(r.agent, r.model)].append(r)

    rows = []
    for (agent, model), rs in sorted(groups.items()):
        n = len(rs)
        errors = sum(1 for r in rs if r.error)
        rows.append({
            "agent": agent,
            "model": model,
            "scenarios": n,
            "pass_rate": _mean(1.0 if r.success else 0.0 for r in rs),
            "avg_score": _mean(r.score for r in rs),
            "avg_tool_f1": _mean(r.tool_f1 for r in rs),
            "total_tokens": sum(r.tokens for r in rs),
            "avg_latency_s": _mean(r.latency_s for r in rs),
            "errors": errors,
        })
    return rows


def to_markdown(results, scenarios) -> str:
    scenario_desc = {s.id: s.description for s in scenarios}
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    out = [f"# Benchmark report — {ts}", ""]

    # ── Comparative summary (the headline table) ────────────────────────────
    out += ["## Summary (agent × model)", "",
            "| Agent | Model | Pass rate | Avg score | Avg tool F1 | Tokens | Avg latency (s) | Errors |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in _aggregate(results):
        out.append(
            f"| {row['agent']} | {row['model']} | "
            f"{row['pass_rate']:.0%} | {row['avg_score']:.2f} | {row['avg_tool_f1']:.2f} | "
            f"{row['total_tokens']} | {row['avg_latency_s']:.2f} | {row['errors']} |"
        )
    out.append("")

    # ── Per-scenario detail, grouped by agent ───────────────────────────────
    by_agent = defaultdict(list)
    for r in results:
        by_agent[r.agent].append(r)

    for agent in sorted(by_agent):
        out += [f"## {agent} — per scenario", "",
                "| Scenario | Model | Success | Score | Tool F1 | Tools called | Tokens | Latency (s) | Error |",
                "|---|---|:---:|---:|---:|---|---:|---:|---|"]
        for r in sorted(by_agent[agent], key=lambda x: (x.scenario_id, x.model)):
            tools = ", ".join(r.predicted_tools) or "—"
            success = "✅" if r.success else ("💥" if r.error else "❌")
            err = (r.error or "").replace("|", "/")[:60]
            out.append(
                f"| {r.scenario_id} | {r.model} | {success} | {r.score:.2f} | "
                f"{r.tool_f1:.2f} | {tools} | {r.tokens} | {r.latency_s:.2f} | {err} |"
            )
        out.append("")

    # ── Scenario legend ─────────────────────────────────────────────────────
    out += ["## Scenarios", ""]
    for sid, desc in scenario_desc.items():
        out.append(f"- **{sid}** — {desc}")
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# HTML — self-contained, no external assets
# ---------------------------------------------------------------------------

_HTML_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       margin: 0; padding: 2rem; background: #0f1115; color: #e6e6e6; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 .5rem; border-bottom: 1px solid #2a2f3a; padding-bottom: .3rem; }
.meta { color: #8a93a3; font-size: .85rem; margin-bottom: 1.5rem; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0 1rem; font-size: .88rem; }
th, td { padding: .5rem .6rem; text-align: left; border-bottom: 1px solid #232833; white-space: nowrap; }
th { color: #9aa4b5; font-weight: 600; background: #161a22; position: sticky; top: 0; }
tbody tr:hover { background: #161b24; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge { display: inline-block; padding: .1rem .5rem; border-radius: 999px; font-size: .78rem; font-weight: 600; }
.ok { background: #123d1e; color: #58d67d; }
.fail { background: #3d1212; color: #f08585; }
.err { background: #3d2f12; color: #e6c35c; }
.bar { position: relative; background: #232833; border-radius: 4px; height: 1.1rem; min-width: 60px; overflow: hidden; }
.bar > span { position: absolute; inset: 0 auto 0 0; background: linear-gradient(90deg,#2b6cb0,#4299e1); }
.bar > em { position: relative; font-style: normal; padding-left: .4rem; line-height: 1.1rem; font-size: .78rem; }
.tools { color: #8a93a3; font-size: .82rem; white-space: normal; }
.errtext { color: #f08585; font-size: .8rem; white-space: normal; }
.legend { color: #b6bdca; font-size: .88rem; }
.legend li { margin: .2rem 0; }
code { background: #1b2029; padding: .05rem .3rem; border-radius: 4px; }
"""


def _bar(value: float, pct: bool = False) -> str:
    """A small inline bar cell (0..1)."""
    width = max(0.0, min(1.0, value)) * 100
    label = f"{value:.0%}" if pct else f"{value:.2f}"
    return (f'<div class="bar"><span style="width:{width:.0f}%"></span>'
            f'<em>{label}</em></div>')


def to_html(results, scenarios) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    esc = html.escape

    parts = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
             "<meta name='viewport' content='width=device-width, initial-scale=1'>",
             f"<title>Benchmark report — {ts}</title>",
             f"<style>{_HTML_CSS}</style></head><body>",
             "<h1>Benchmark report</h1>",
             f"<div class='meta'>Generated {ts} · {len(results)} runs</div>"]

    # ── Summary (agent × model) ─────────────────────────────────────────────
    parts.append("<h2>Summary (agent × model)</h2>")
    parts.append("<table><thead><tr>"
                 "<th>Agent</th><th>Model</th>"
                 "<th class='num'>Pass rate</th><th class='num'>Avg score</th>"
                 "<th class='num'>Avg tool F1</th><th class='num'>Tokens</th>"
                 "<th class='num'>Avg latency (s)</th><th class='num'>Errors</th>"
                 "</tr></thead><tbody>")
    for row in _aggregate(results):
        err_badge = (f"<span class='badge err'>{row['errors']}</span>"
                     if row['errors'] else "0")
        parts.append(
            "<tr>"
            f"<td>{esc(row['agent'])}</td><td><code>{esc(row['model'])}</code></td>"
            f"<td class='num'>{_bar(row['pass_rate'], pct=True)}</td>"
            f"<td class='num'>{_bar(row['avg_score'])}</td>"
            f"<td class='num'>{_bar(row['avg_tool_f1'])}</td>"
            f"<td class='num'>{row['total_tokens']}</td>"
            f"<td class='num'>{row['avg_latency_s']:.2f}</td>"
            f"<td class='num'>{err_badge}</td>"
            "</tr>")
    parts.append("</tbody></table>")

    # ── Per-scenario detail, grouped by agent ───────────────────────────────
    by_agent = defaultdict(list)
    for r in results:
        by_agent[r.agent].append(r)

    for agent in sorted(by_agent):
        parts.append(f"<h2>{esc(agent)} — per scenario</h2>")
        parts.append("<table><thead><tr>"
                     "<th>Scenario</th><th>Model</th><th>Result</th>"
                     "<th class='num'>Score</th><th class='num'>Tool F1</th>"
                     "<th>Tools called</th><th class='num'>Tokens</th>"
                     "<th class='num'>Latency (s)</th><th>Error</th>"
                     "</tr></thead><tbody>")
        for r in sorted(by_agent[agent], key=lambda x: (x.scenario_id, x.model)):
            if r.error:
                result = "<span class='badge err'>error</span>"
            elif r.success:
                result = "<span class='badge ok'>pass</span>"
            else:
                result = "<span class='badge fail'>fail</span>"
            tools = esc(", ".join(r.predicted_tools)) or "—"
            err = f"<span class='errtext'>{esc(r.error)}</span>" if r.error else ""
            parts.append(
                "<tr>"
                f"<td>{esc(r.scenario_id)}</td><td><code>{esc(r.model)}</code></td>"
                f"<td>{result}</td>"
                f"<td class='num'>{r.score:.2f}</td>"
                f"<td class='num'>{r.tool_f1:.2f}</td>"
                f"<td class='tools'>{tools}</td>"
                f"<td class='num'>{r.tokens}</td>"
                f"<td class='num'>{r.latency_s:.2f}</td>"
                f"<td>{err}</td>"
                "</tr>")
        parts.append("</tbody></table>")

    # ── Scenario legend ─────────────────────────────────────────────────────
    parts.append("<h2>Scenarios</h2><ul class='legend'>")
    for s in scenarios:
        parts.append(f"<li><b>{esc(s.id)}</b> — {esc(s.description)}</li>")
    parts.append("</ul></body></html>")
    return "".join(parts)


def write_reports(results, scenarios, out_dir: str) -> dict:
    """Write timestamped .md, .html and .json reports; return their paths."""
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    md_path = os.path.join(out_dir, f"report_{stamp}.md")
    html_path = os.path.join(out_dir, f"report_{stamp}.html")
    json_path = os.path.join(out_dir, f"report_{stamp}.json")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(to_markdown(results, scenarios))

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(to_html(results, scenarios))

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "summary": _aggregate(results),
            "runs": [r.as_dict() for r in results],
        }, f, indent=2, ensure_ascii=False)

    return {"markdown": md_path, "html": html_path, "json": json_path}
