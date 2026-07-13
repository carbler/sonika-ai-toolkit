"""Scenario and result data structures for the benchmark."""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class CheckOutcome:
    """Result of a single declarative check against an agent response."""
    label: str
    passed: bool
    detail: str = ""


# A Check is a pure function: (BotResponse) -> CheckOutcome
Check = Callable[[Any], CheckOutcome]


@dataclass
class Turn:
    """One prior conversation turn used to seed context/memory."""
    role: str          # "user" | "assistant"
    content: str


@dataclass
class Scenario:
    """A single measurable task.

    goal            the user request the agent must handle
    tools           tool names made available to the agent
    expected_tools  tools a correct solution is expected to call (name-based
                    precision/recall is computed against this set)
    checks          declarative task-success assertions on the response
    history         optional prior turns (for memory/context scenarios)
    """
    id: str
    description: str
    goal: str
    tools: List[str] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)
    checks: List[Check] = field(default_factory=list)
    history: List[Turn] = field(default_factory=list)


@dataclass
class RunResult:
    """Outcome of running one (agent, model, scenario) combination."""
    scenario_id: str
    agent: str
    model: str

    # Task success
    success: bool = False              # every check passed
    score: float = 0.0                 # fraction of checks passed
    checks: List[CheckOutcome] = field(default_factory=list)

    # Tool-call accuracy (name-based, fair across agents)
    predicted_tools: List[str] = field(default_factory=list)
    expected_tools: List[str] = field(default_factory=list)
    tool_precision: float = 0.0
    tool_recall: float = 0.0
    tool_f1: float = 0.0

    # Informational metrics (captured for free from BotResponse / wall clock)
    tokens: int = 0
    latency_s: float = 0.0
    num_tool_calls: int = 0

    error: Optional[str] = None        # set if the agent raised

    def as_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "agent": self.agent,
            "model": self.model,
            "success": self.success,
            "score": round(self.score, 3),
            "checks": [{"label": c.label, "passed": c.passed, "detail": c.detail}
                       for c in self.checks],
            "predicted_tools": self.predicted_tools,
            "expected_tools": self.expected_tools,
            "tool_precision": round(self.tool_precision, 3),
            "tool_recall": round(self.tool_recall, 3),
            "tool_f1": round(self.tool_f1, 3),
            "tokens": self.tokens,
            "latency_s": round(self.latency_s, 2),
            "num_tool_calls": self.num_tool_calls,
            "error": self.error,
        }
