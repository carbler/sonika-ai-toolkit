"""OrchestratorState TypedDict and helpers."""

import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated


class OrchestratorState(TypedDict):
    # ── Input ──────────────────────────────────────────────────────────────
    goal: str
    context: str

    # ── Plan ───────────────────────────────────────────────────────────────
    # Replaced whole when plan changes; each item is a TaskStep dict.
    plan: List[Dict[str, Any]]
    current_step_index: int

    # ── Flow control ───────────────────────────────────────────────────────
    should_advance: bool

    # ── Execution state ────────────────────────────────────────────────────
    # Accumulative: nodes return a *list delta*, LangGraph appends via operator.add.
    tool_outputs: Annotated[List[Dict], operator.add]
    last_result: Optional[str]
    last_error: Optional[str]

    # ── Retry ──────────────────────────────────────────────────────────────
    retry_count: int
    max_retries: int
    retry_strategy: Optional[str]
    # Anti-loop: list of {"strategy", "tool_name", "params_hash"} dicts.
    retry_history: List[Dict[str, Any]]

    # ── Human approval ─────────────────────────────────────────────────────
    awaiting_approval: bool
    user_approved: Optional[bool]

    # ── Output ─────────────────────────────────────────────────────────────
    final_report: Optional[str]
    _goal_complete: bool           # set by evaluator; triggers early reporter route

    # ── Session ────────────────────────────────────────────────────────────
    # Accumulative log lines.
    session_log: Annotated[List[str], operator.add]
    # Accumulative turn history: list of {"role": "user"|"assistant", "content": str}
    history: Annotated[List[Dict[str, str]], operator.add]
    model_used: str
    session_id: str
    skills_dir: str
    thinking: str                                        # accumulated reasoning across nodes


# ── TaskStep dict schema (documentation only) ──────────────────────────────
#
# {
#   "id": int,
#   "description": str,
#   "tool_name": str,
#   "params": dict,
#   "risk_level": int,   # 0-3
#   "status": "pending" | "running" | "success" | "failed" | "skipped",
#   "result": str | None,
#   "error": str | None,
#   "retries_used": int,
# }


def make_step(
    step_id: int,
    description: str,
    tool_name: str,
    params: Dict[str, Any],
    risk_level: int = 0,
) -> Dict[str, Any]:
    """Create a TaskStep dict with default values."""
    return {
        "id": step_id,
        "description": description,
        "tool_name": tool_name,
        "params": params,
        "risk_level": risk_level,
        "status": "pending",
        "result": None,
        "error": None,
        "retries_used": 0,
    }


def update_step(
    plan: List[Dict[str, Any]],
    step_index: int,
    **fields,
) -> List[Dict[str, Any]]:
    """Return a new plan list with step at step_index updated."""
    new_plan = [dict(s) for s in plan]
    new_plan[step_index].update(fields)
    return new_plan
