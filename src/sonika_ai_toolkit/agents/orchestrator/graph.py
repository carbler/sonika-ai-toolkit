"""OrchestratorBot — autonomous orchestration engine."""

import asyncio
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END

from sonika_ai_toolkit.utilities.types import BotResponse, ILanguageModel
from sonika_ai_toolkit.tools.registry import ToolRegistry
from sonika_ai_toolkit.tools.synthesizer import DynamicToolSynthesizer
from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState
from sonika_ai_toolkit.agents.orchestrator.memory import MemoryManager
from sonika_ai_toolkit.agents.orchestrator.nodes.load_memory import LoadMemoryNode
from sonika_ai_toolkit.agents.orchestrator.nodes.manager import ManagerNode
from sonika_ai_toolkit.agents.orchestrator.nodes.planner import PlannerNode
from sonika_ai_toolkit.agents.orchestrator.nodes.step_dispatcher import StepDispatcherNode
from sonika_ai_toolkit.agents.orchestrator.nodes.risk_gate import RiskGateNode
from sonika_ai_toolkit.agents.orchestrator.nodes.human_approval import HumanApprovalNode
from sonika_ai_toolkit.agents.orchestrator.nodes.executor import ExecutorNode
from sonika_ai_toolkit.agents.orchestrator.nodes.evaluator import EvaluatorNode
from sonika_ai_toolkit.agents.orchestrator.nodes.retry import RetryNode
from sonika_ai_toolkit.agents.orchestrator.nodes.escalate import EscalateNode
from sonika_ai_toolkit.agents.orchestrator.nodes.reporter import ReporterNode
from sonika_ai_toolkit.agents.orchestrator.nodes.save_memory import SaveMemoryNode


class OrchestratorBot:
    """
    Autonomous orchestration engine with:
    - Persistent memory (MEMORY.md / SKILLS.md)
    - Per-step human approval gate
    - Multi-model routing (strong / fast / code)
    - Structured retry with anti-loop protection
    - Dynamic tool synthesis at runtime
    - Thinking/reasoning extraction (same as ReactBot)
    """

    def __init__(
        self,
        strong_model: ILanguageModel,
        fast_model: ILanguageModel,
        instructions: str,
        tools: Optional[List[BaseTool]] = None,
        memory_path: str = "./memory",
        risk_threshold: int = 1,
        max_retries: int = 3,
        code_model: Optional[ILanguageModel] = None,
        on_human_approval: Optional[Callable[[Dict], bool]] = None,
        on_step_start: Optional[Callable[[Dict], None]] = None,
        on_step_end: Optional[Callable[[Dict, str], None]] = None,
        on_plan_generated: Optional[Callable[[List], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
        on_message: Optional[Callable[[str], None]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.strong_model = strong_model
        self.fast_model = fast_model
        self.code_model = code_model or strong_model
        self.instructions = instructions
        self.risk_threshold = risk_threshold
        self.max_retries = max_retries
        self.on_human_approval = on_human_approval
        self.on_step_start = on_step_start
        self.on_step_end = on_step_end
        self.on_plan_generated = on_plan_generated
        self.on_thinking = on_thinking
        self.on_message = on_message

        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())

        self.memory_manager = MemoryManager(memory_path)
        self.skills_dir = self.memory_manager.sessions_dir.replace("sessions", "skills")

        self.registry = ToolRegistry()
        for tool in tools or []:
            self.registry.register(tool)

        self.synthesizer = DynamicToolSynthesizer(
            code_model=self.code_model,
            memory_manager=self.memory_manager,
            skills_dir=self.skills_dir,
        )

        self.graph = self._build_workflow()

    # ── Public API ─────────────────────────────────────────────────────────

    def run(self, goal: str, context: str = "") -> Dict[str, Any]:
        """
        Run the orchestrator synchronously.

        Returns a dict with the same structure as ReactBot.get_response():
            content       — final report text
            thinking      — accumulated reasoning across all nodes (or None)
            tools_executed — list of {tool_name, args, status, output}
            logs          — session log lines
            token_usage   — placeholder (orchestrator doesn't track tokens yet)
            success       — True if at least one step succeeded
            plan          — full step list with statuses
            session_id    — unique run identifier
            goal          — original goal
        """
        session_id = str(uuid.uuid4())[:8]

        initial_state: OrchestratorState = {
            "goal": goal,
            "context": context,
            "plan": [],
            "current_step_index": 0,
            "should_advance": False,
            "tool_outputs": [],
            "last_result": None,
            "last_error": None,
            "retry_count": 0,
            "max_retries": self.max_retries,
            "retry_strategy": None,
            "retry_history": [],
            "awaiting_approval": False,
            "user_approved": None,
            "final_report": None,
            "_goal_complete": False,
            "session_log": [],
            "model_used": getattr(self.strong_model, "model_name", "unknown"),
            "session_id": session_id,
            "skills_dir": self.skills_dir,
            "thinking": "",
        }

        result = asyncio.run(
            self.graph.ainvoke(
                initial_state,
                config={"recursion_limit": 100},
            )
        )

        plan = result.get("plan", [])
        final_report = result.get("final_report", "")
        session_log = result.get("session_log", [])
        thinking = result.get("thinking") or None
        success = any(s.get("status") == "success" for s in plan)

        # Build tools_executed matching ReactBot format
        tools_executed = [
            {
                "tool_name": s["tool_name"],
                "args": str(s.get("params", {})),
                "status": s.get("status", "unknown"),
                "output": s.get("result") or s.get("error") or "",
            }
            for s in plan
            if s.get("status") not in ("pending",)
        ]

        return BotResponse(
            content=final_report,
            thinking=thinking,
            tools_executed=tools_executed,
            logs=session_log,
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            success=success,
            plan=plan,
            session_id=session_id,
            goal=goal,
        )

    # ── Graph construction ─────────────────────────────────────────────────

    def _build_workflow(self) -> StateGraph:
        load_memory = LoadMemoryNode(
            memory_manager=self.memory_manager,
            tool_registry=self.registry,
            logger=self.logger,
        )
        manager = ManagerNode(
            fast_model=self.fast_model,
            on_thinking=self.on_thinking,
            on_message=self.on_message,
            logger=self.logger,
        )
        planner = PlannerNode(
            strong_model=self.strong_model,
            tool_registry=self.registry,
            memory_manager=self.memory_manager,
            instructions=self.instructions,
            on_plan_generated=self.on_plan_generated,
            on_thinking=self.on_thinking,
            logger=self.logger,
        )
        step_dispatcher = StepDispatcherNode(
            tool_registry=self.registry,
            on_step_start=self.on_step_start,
            logger=self.logger,
        )
        risk_gate = RiskGateNode(
            risk_threshold=self.risk_threshold,
            logger=self.logger,
        )
        human_approval = HumanApprovalNode(
            on_human_approval=self.on_human_approval,
            logger=self.logger,
        )
        executor = ExecutorNode(
            tool_registry=self.registry,
            synthesizer=self.synthesizer,
            on_step_end=self.on_step_end,
            logger=self.logger,
        )
        evaluator = EvaluatorNode(
            fast_model=self.fast_model,
            on_thinking=self.on_thinking,
            logger=self.logger,
        )
        retry = RetryNode(
            fast_model=self.fast_model,
            tool_registry=self.registry,
            on_thinking=self.on_thinking,
            logger=self.logger,
        )
        escalate = EscalateNode(logger=self.logger)
        reporter = ReporterNode(
            fast_model=self.fast_model,
            on_thinking=self.on_thinking,
            logger=self.logger,
        )
        save_memory = SaveMemoryNode(
            fast_model=self.fast_model,
            memory_manager=self.memory_manager,
            logger=self.logger,
        )

        workflow = StateGraph(OrchestratorState)
        workflow.add_node("load_memory", load_memory)
        workflow.add_node("manager", manager)
        workflow.add_node("planner", planner)
        workflow.add_node("step_dispatcher", step_dispatcher)
        workflow.add_node("risk_gate", risk_gate)
        workflow.add_node("human_approval", human_approval)
        workflow.add_node("executor", executor)
        workflow.add_node("evaluator", evaluator)
        workflow.add_node("retry", retry)
        workflow.add_node("escalate", escalate)
        workflow.add_node("reporter", reporter)
        workflow.add_node("save_memory", save_memory)

        workflow.set_entry_point("load_memory")
        workflow.add_edge("load_memory", "manager")
        workflow.add_conditional_edges(
            "manager",
            self._route_after_manager,
            {"planner": "planner", "save_memory": "save_memory"},
        )
        workflow.add_edge("planner", "step_dispatcher")

        workflow.add_conditional_edges(
            "step_dispatcher",
            self._route_after_dispatcher,
            {"risk_gate": "risk_gate", "retry": "retry", "reporter": "reporter"},
        )
        workflow.add_conditional_edges(
            "risk_gate",
            self._route_after_risk_gate,
            {"executor": "executor", "human_approval": "human_approval"},
        )
        workflow.add_conditional_edges(
            "human_approval",
            self._route_after_approval,
            {"executor": "executor", "step_dispatcher": "step_dispatcher"},
        )
        workflow.add_edge("executor", "evaluator")
        workflow.add_conditional_edges(
            "evaluator",
            self._route_after_evaluator,
            {"step_dispatcher": "step_dispatcher", "reporter": "reporter", "retry": "retry"},
        )
        workflow.add_conditional_edges(
            "retry",
            self._route_after_retry,
            {"executor": "executor", "escalate": "escalate"},
        )
        workflow.add_edge("escalate", "step_dispatcher")
        workflow.add_edge("reporter", "save_memory")
        workflow.add_edge("save_memory", END)

        return workflow.compile()

    # ── Routing functions ──────────────────────────────────────────────────

    def _route_after_manager(self, state: OrchestratorState) -> str:
        return "save_memory" if state.get("_goal_complete") else "planner"

    def _route_after_dispatcher(self, state: OrchestratorState) -> str:
        plan = state.get("plan", [])
        index = state.get("current_step_index", 0)
        last_error = state.get("last_error")
        if index >= len(plan):
            return "reporter"
        if last_error and last_error.startswith("Tool not found:"):
            return "retry"
        return "risk_gate"

    def _route_after_risk_gate(self, state: OrchestratorState) -> str:
        return "human_approval" if state.get("awaiting_approval") else "executor"

    def _route_after_approval(self, state: OrchestratorState) -> str:
        return "executor" if state.get("user_approved") else "step_dispatcher"

    def _route_after_evaluator(self, state: OrchestratorState) -> str:
        plan = state.get("plan", [])
        index = state.get("current_step_index", 0)
        step = plan[index] if index < len(plan) else {}
        if step.get("status") != "success" or state.get("last_error"):
            return "retry"
        if state.get("_goal_complete"):
            return "reporter"
        return "step_dispatcher"

    def _route_after_retry(self, state: OrchestratorState) -> str:
        return "escalate" if state.get("retry_strategy") == "escalate" else "executor"
