"""Extension points for injecting custom LangGraph nodes into shipped agents.

Three levels of graph customization for the OrchestratorBot:

1. ``CustomNode`` with a ``position`` — insert a node at a predefined spot
   (``start`` / ``after_tools`` / ``end``); the default wiring adapts.
2. ``CustomEdge`` — replace the *fixed* outgoing edge of any node (built-in or
   custom) with your own, including ``__start__`` (entry) and ``__end__``.
3. ``CustomRouter`` — replace the *conditional* routing of any node with your
   own function; return ``None``/``"__default__"`` to fall back to the
   built-in decision for that node, so you only override the cases you care
   about (e.g. keep the plan/ask_user/tools routing and add one extra branch).
"""

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Literal, Optional, Set

CustomNodePosition = Literal["start", "after_tools", "end"]

_VALID_POSITIONS = ("start", "after_tools", "end")

START_NODE = "__start__"
END_NODE = "__end__"

#: Sentinel a CustomRouter may return to delegate to the built-in routing.
DEFAULT_ROUTE = "__default__"


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
            - ``None`` — not auto-wired: connect it yourself with
              ``CustomEdge`` / ``CustomRouter`` (an unwired node is an error).
        Multiple nodes at the same position chain in list order.
    """

    name: str
    node: Callable
    position: Optional[CustomNodePosition] = "start"


@dataclass
class CustomEdge:
    """A fixed edge that OVERRIDES the default outgoing wiring of ``source``.

    Declaring an edge from ``source`` removes that node's built-in outgoing
    edge *and* its built-in router (if any) — you take over where the node
    goes next. Several edges from the same source fan out in parallel
    (standard LangGraph semantics).

    Attributes:
        source: Node the edge leaves from; ``"__start__"`` overrides the
            graph entry point.
        target: Node the edge enters; ``"__end__"`` ends the run.
    """

    source: str
    target: str


@dataclass
class CustomRouter:
    """A conditional router that OVERRIDES the default routing of ``source``.

    Attributes:
        source: Node whose outgoing routing is replaced (fixed edge or
            built-in router alike).
        router: ``(state) -> str`` returning the next node name,
            ``"__end__"`` to end the run, or ``None`` / ``"__default__"`` to
            delegate to the node's built-in routing decision (for ``agent``
            that keeps the plan / ask_user / tools / END logic intact).
        targets: Optional list of node names this router can return. Used to
            declare the conditional edges in the topology (what
            ``get_graph_topology()`` and the ``graph_topology`` event show);
            with ``None``, every node plus ``"__end__"`` becomes a potential
            target — correct, but noisier to draw.
    """

    source: str
    router: Callable
    targets: Optional[List[str]] = field(default=None)


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
        if cn.position is not None and cn.position not in _VALID_POSITIONS:
            raise ValueError(
                f"CustomNode '{cn.name}' has invalid position '{cn.position}' "
                f"(valid: {_VALID_POSITIONS} or None for manual wiring)."
            )
        if not callable(cn.node):
            raise ValueError(f"CustomNode '{cn.name}'.node must be callable.")
        seen.add(cn.name)
        validated.append(cn)
    return validated


def validate_custom_wiring(
    custom_edges: Optional[Iterable[CustomEdge]],
    custom_routers: Optional[Iterable[CustomRouter]],
    node_names: Set[str],
    unwired_nodes: Set[str],
) -> tuple:
    """Validate edges/routers against the actual node set.

    Args:
        node_names: Every real node of the graph (built-ins that exist for
            this configuration + custom nodes).
        unwired_nodes: Custom nodes with ``position=None`` — each must be
            referenced by the custom wiring or the graph can never reach it.

    Returns:
        ``(edges, routers)`` as validated lists.
    """
    edges: List[CustomEdge] = list(custom_edges or [])
    routers: List[CustomRouter] = list(custom_routers or [])
    valid_sources = node_names | {START_NODE}
    valid_targets = node_names | {END_NODE}

    def _check(kind: str, name: str, valid: Set[str], role: str) -> None:
        if name not in valid:
            raise ValueError(
                f"{kind} {role} '{name}' is not a node of this graph "
                f"(valid: {sorted(valid)}). Note: 'plan' and 'ask_user' exist "
                f"only with enable_planning / enable_user_questions."
            )

    for ce in edges:
        _check("CustomEdge", ce.source, valid_sources, "source")
        _check("CustomEdge", ce.target, valid_targets, "target")
        if ce.source == START_NODE and ce.target == END_NODE:
            raise ValueError("CustomEdge from '__start__' cannot target '__end__'.")
    router_sources: Set[str] = set()
    for cr in routers:
        _check("CustomRouter", cr.source, valid_sources, "source")
        if not callable(cr.router):
            raise ValueError(f"CustomRouter for '{cr.source}'.router must be callable.")
        if cr.source in router_sources:
            raise ValueError(f"Duplicate CustomRouter for source '{cr.source}'.")
        router_sources.add(cr.source)
        for t in cr.targets or []:
            _check("CustomRouter", t, valid_targets, "target")

    referenced = {ce.target for ce in edges}
    referenced.update(t for cr in routers for t in (cr.targets or []))
    for name in unwired_nodes:
        if name not in referenced:
            raise ValueError(
                f"CustomNode '{name}' has position=None but no CustomEdge/"
                f"CustomRouter targets it — the graph can never reach it. "
                f"Wire it (e.g. CustomEdge(source='tools', target='{name}'))."
            )
    return edges, routers
