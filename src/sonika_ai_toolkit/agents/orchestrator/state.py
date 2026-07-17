"""OrchestratorState TypedDict and helpers."""

import operator
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class OrchestratorState(TypedDict):
    """Simplified state for the new fast ReAct-based Orchestrator."""
    messages: Annotated[List[BaseMessage], add_messages]
    
    # ── Input ──────────────────────────────────────────────────────────────
    goal: str
    mode: str # "plan", "ask", "auto"
    
    # ── Session & Memory ───────────────────────────────────────────────────
    session_id: str
    run_id: str  # unique id of this run (one per run/arun/astream_events call)
    skills_dir: str
    thinking: str # Accumulated reasoning
    
    # ── Outputs ────────────────────────────────────────────────────────────
    final_report: Optional[str]
    partial_responses: Annotated[List[str], operator.add]
    session_log: Annotated[List[str], operator.add]
    tools_executed: Annotated[List[Dict[str, Any]], operator.add]
    status_events: Annotated[List[Dict[str, Any]], operator.add]

    # ── Structured plan (only populated when enable_planning=True) ─────────
    plan: List[Dict[str, Any]]  # snapshot of PlanStep dicts (last-write-wins)
    step_events: Annotated[List[Dict[str, Any]], operator.add]

    # ── Node execution trace (one entry per node run, in order) ────────────
    node_trace: Annotated[List[Dict[str, Any]], operator.add]
