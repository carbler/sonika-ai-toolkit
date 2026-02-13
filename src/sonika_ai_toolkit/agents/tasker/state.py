"""Estado compartido para el workflow multi-nodo de LangGraph."""

from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage
from operator import add


class ChatState(TypedDict):
    """Estado compartido entre todos los nodos del workflow."""

    # ENTRADA
    user_input: str

    # HISTORIAL DE CONVERSACIÓN (Acumulativo)
    messages: Annotated[List[BaseMessage], add]

    # LOGS DEL SISTEMA (Acumulativo)
    logs: Annotated[List[str], add]

    dynamic_info: str

    # INSTRUCCIONES
    function_purpose: str
    personality_tone: str
    limitations: str

    # OUTPUTS DE CADA NODO
    router_decision: Optional[str]
    thinking_output: Optional[str]
    planner_output: Optional[Dict[str, Any]]
    executor_output: Optional[Dict[str, Any]]
    validator_output: Optional[Dict[str, Any]]
    output_node_response: Optional[str]
    logger_output: Optional[List[str]]

    # CONTROL DE FLUJO
    planning_attempts: int
    execution_attempts: int
    react_iteration: int

    # TRACKING (Acumulativo para análisis)
    tools_executed: Annotated[List[Dict[str, Any]], add]
    token_usage: Dict[str, int]
