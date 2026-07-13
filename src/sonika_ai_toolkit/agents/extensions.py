"""Extension points for injecting custom LangGraph nodes into shipped agents."""

from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Optional, Set

CustomNodePosition = Literal["start", "after_tools", "end"]

_VALID_POSITIONS = ("start", "after_tools", "end")


@dataclass
class CustomNode:
    """A consumer-provided LangGraph node to insert into the orchestrator graph.

    Attributes:
        name: Unique node name (must not collide with built-in node names).
            Its state updates stream as ``("updates", {"<name>": {...}})``.
        node: Sync or async callable receiving the graph state (for the
            orchestrator, an ``OrchestratorState`` dict) and returning a
            partial state update dict.
        position: Where to insert the node:
            - ``"start"`` — between the entry point and the agent (runs once).
            - ``"after_tools"`` — on the tools → agent edge (runs every loop).
            - ``"end"`` — between the agent's final turn and END (runs once).
        Multiple nodes at the same position chain in list order.
    """

    name: str
    node: Callable
    position: CustomNodePosition = "start"


def validate_custom_nodes(
    custom_nodes: Optional[Iterable[CustomNode]],
    reserved_names: Set[str],
) -> List[CustomNode]:
    """Validate names, positions and callables; return the nodes as a list."""
    validated: List[CustomNode] = []
    seen: Set[str] = set()
    for cn in custom_nodes or []:
        if not cn.name or not isinstance(cn.name, str):
            raise ValueError("CustomNode.name must be a non-empty string.")
        if cn.name in reserved_names:
            raise ValueError(
                f"CustomNode name '{cn.name}' collides with a built-in node "
                f"(reserved: {sorted(reserved_names)})."
            )
        if cn.name in seen:
            raise ValueError(f"Duplicate CustomNode name: '{cn.name}'.")
        if cn.position not in _VALID_POSITIONS:
            raise ValueError(
                f"CustomNode '{cn.name}' has invalid position '{cn.position}' "
                f"(valid: {_VALID_POSITIONS})."
            )
        if not callable(cn.node):
            raise ValueError(f"CustomNode '{cn.name}'.node must be callable.")
        seen.add(cn.name)
        validated.append(cn)
    return validated
