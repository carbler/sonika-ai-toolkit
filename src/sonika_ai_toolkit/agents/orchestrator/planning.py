"""Pure helpers for structured plan tracking in the OrchestratorBot.

These functions have no LangGraph/LLM dependencies so they are unit-testable
in isolation. The graph's ``agent_node`` uses them to intercept the
``set_plan`` / ``update_step`` signal tools (see ``tools/plan_tools.py``).
"""

from typing import Any, Dict, List, Tuple

from sonika_ai_toolkit.tools.plan_tools import PLAN_SIGNAL_TOOL_NAMES

VALID_STEP_STATUSES = frozenset({"pending", "running", "done", "skipped", "error"})

PLANNING_PROTOCOL_PROMPT = (
    "[PLANIFICACIÓN] Para tareas de varios pasos sigue este protocolo:\n"
    "1. ANTES de empezar, llama a `set_plan` con la lista ordenada de pasos "
    "(descripciones cortas). Llámala UNA sola vez por tarea.\n"
    "2. Al iniciar un paso llama a `update_step(step, \"running\")` junto con las "
    "herramientas reales de ese paso; al terminarlo llama a "
    "`update_step(step, \"done\")` (o \"skipped\"/\"error\").\n"
    "3. Cuando todos los pasos estén completos, entrega tu respuesta final SIN "
    "llamar a set_plan ni update_step.\n"
    "Para preguntas simples que no requieren varios pasos, responde directamente "
    "sin plan."
)


def normalize_plan(steps: List[Any]) -> List[Dict[str, Any]]:
    """Build a plan snapshot (list of PlanStep dicts) from step descriptions."""
    return [
        {"step": i, "description": str(desc), "status": "pending"}
        for i, desc in enumerate(steps or [], start=1)
    ]


def apply_update_step(
    plan: List[Dict[str, Any]], step: Any, status: Any
) -> List[Dict[str, Any]]:
    """Return a copy of ``plan`` with ``step`` set to ``status``.

    Invalid statuses or unknown step numbers leave the plan untouched.
    """
    if status not in VALID_STEP_STATUSES:
        return list(plan)
    return [
        {**item, "status": status} if item.get("step") == step else item
        for item in plan
    ]


def render_plan_status(plan: List[Dict[str, Any]]) -> str:
    """Render the current plan snapshot for the system prompt."""
    if not plan:
        return ""
    lines = [
        "Plan actual (ya registrado — NO vuelvas a llamar a set_plan; "
        "usa update_step para reportar avance):"
    ]
    for item in plan:
        lines.append(
            f"{item.get('step')}. [{item.get('status')}] {item.get('description')}"
        )
    return "\n".join(lines)


def split_plan_signal_calls(
    tool_calls: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition tool calls into (plan signal calls, real tool calls)."""
    signal: List[Dict[str, Any]] = []
    real: List[Dict[str, Any]] = []
    for call in tool_calls or []:
        if call.get("name") in PLAN_SIGNAL_TOOL_NAMES:
            signal.append(call)
        else:
            real.append(call)
    return signal, real
