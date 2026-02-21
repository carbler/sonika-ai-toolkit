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
    skills_dir: str
    thinking: str # Accumulated reasoning
    
    # ── Outputs ────────────────────────────────────────────────────────────
    final_report: Optional[str]
    session_log: Annotated[List[str], operator.add]
    tools_executed: Annotated[List[Dict[str, Any]], operator.add]
