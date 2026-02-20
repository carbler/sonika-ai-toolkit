"""Tasker Bot - The robust successor to MultiNodeBot."""

from typing import List, Dict, Any, Optional, Callable
import logging
import asyncio
from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langchain_community.callbacks.manager import get_openai_callback

from sonika_ai_toolkit.utilities.types import BotResponse
from sonika_ai_toolkit.agents.tasker.state import ChatState
from sonika_ai_toolkit.agents.tasker.nodes.planner_node import PlannerNode
from sonika_ai_toolkit.agents.tasker.nodes.executor_node import ExecutorNode
from sonika_ai_toolkit.agents.tasker.nodes.output_node import OutputNode
from sonika_ai_toolkit.agents.tasker.nodes.logger_node import LoggerNode
from sonika_ai_toolkit.agents.tasker.nodes.validator_node import ValidatorNode


class TaskerBot:
    """
    Bot with enhanced ReAct pattern and robust instruction following.
    Drop-in replacement for MultiNodeBot but with separate internal architecture.
    """

    def __init__(
        self,
        language_model,
        embeddings,
        function_purpose: str,
        personality_tone: str,
        limitations: str,
        dynamic_info: str,
        tools: Optional[List[BaseTool]] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        max_messages: int = 100,
        max_logs: int = 20,
        max_iterations: int = 15,
        recursion_limit: int = 100,
        executor_max_retries: int = 2,
        on_planner_update: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_tool_start: Optional[Callable[[str, str], None]] = None,
        on_tool_end: Optional[Callable[[str, str], None]] = None,
        on_tool_error: Optional[Callable[[str, str], None]] = None,
        on_logs_generated: Optional[Callable[[List[str]], None]] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())

        self.language_model = language_model
        self.embeddings = embeddings
        self.function_purpose = function_purpose
        self.personality_tone = personality_tone
        self.limitations = limitations
        self.dynamic_info = dynamic_info
        self.tools = tools or []

        if mcp_servers:
            self._initialize_mcp(mcp_servers)

        self.max_messages = max_messages
        self.max_logs = max_logs
        self.max_iterations = max_iterations
        self.recursion_limit = recursion_limit
        self.executor_max_retries = executor_max_retries

        # Callbacks
        self.on_planner_update = on_planner_update
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_tool_error = on_tool_error
        self.on_logs_generated = on_logs_generated

        self.model = language_model.model
        self.graph = self._build_workflow()

    def _initialize_mcp(self, mcp_servers: Dict[str, Any]):
        """Initialize MCP (Model Context Protocol)."""
        try:
            # Importación lazy para evitar crash si no está instalado
            from langchain_mcp_adapters.client import MultiServerMCPClient
            self.mcp_client = MultiServerMCPClient(mcp_servers)
            mcp_tools = asyncio.run(self.mcp_client.get_tools())
            self.tools.extend(mcp_tools)
        except ImportError:
             self.logger.warning("langchain_mcp_adapters not installed. MCP servers ignored.")
             self.mcp_client = None
        except Exception as e:
            self.logger.error(f"Error initializing MCP: {e}")
            self.mcp_client = None

    def _build_workflow(self) -> StateGraph:
        """Build the Planner -> Executor -> Output workflow."""

        # 1. Planner Node (The Brain)
        planner = PlannerNode(
            model=self.model,
            tools=self.tools,
            max_iterations=self.max_iterations,
            on_planner_update=self.on_planner_update,
            logger=self.logger
        )

        # 2. Executor Node (The Hands)
        executor = ExecutorNode(
            tools=self.tools,
            max_retries=self.executor_max_retries,
            on_tool_start=self.on_tool_start,
            on_tool_end=self.on_tool_end,
            on_tool_error=self.on_tool_error,
            logger=self.logger
        )

        # 3. Output Node (The Voice)
        output = OutputNode(
            model=self.model,
            logger=self.logger
        )

        # 4. Logger Node (The Recorder)
        logger_node = LoggerNode(
            on_logs_generated=self.on_logs_generated,
            logger=self.logger
        )

        # 5. Validator Node (The Quality Control)
        validator = ValidatorNode(
            model=self.model,
            logger=self.logger
        )

        # Build Graph
        workflow = StateGraph(ChatState)

        workflow.add_node("planner", planner)
        workflow.add_node("executor", executor)
        workflow.add_node("output", output)
        workflow.add_node("logger", logger_node)
        workflow.add_node("validator", validator)

        # Start at Planner
        workflow.set_entry_point("planner")

        # Conditional Edge: Planner -> Executor OR Validator
        def route_after_planner(state: ChatState) -> str:
            planner_output = state.get("planner_output", {})
            decision = planner_output.get("decision", "finish")

            if decision == "execute_tool":
                return "executor"
            return "validator"

        workflow.add_conditional_edges(
            "planner",
            route_after_planner,
            {
                "executor": "executor",
                "validator": "validator"
            }
        )

        # Loop: Executor -> Planner
        workflow.add_edge("executor", "planner")

        # Conditional Edge: Validator -> Output OR Planner (Retry)
        def route_after_validator(state: ChatState) -> str:
            validator_output = state.get("validator_output", {})
            status = validator_output.get("status", "approved")

            if status == "rejected":
                return "planner"
            return "output"

        workflow.add_conditional_edges(
            "validator",
            route_after_validator,
            {
                "planner": "planner",
                "output": "output"
            }
        )

        # End: Output -> Logger -> END
        workflow.add_edge("output", "logger")
        workflow.add_edge("logger", END)

        return workflow.compile()

    def get_response(
        self,
        user_input: str,
        messages: List[BaseMessage],
        logs: List[str],
    ) -> Dict[str, Any]:

        limited_messages = self._limit_messages(messages)
        limited_logs = self._limit_logs(logs)

        initial_state: ChatState = {
            "user_input": user_input,
            "messages": limited_messages,
            "logs": limited_logs,
            "dynamic_info": self.dynamic_info,
            "function_purpose": self.function_purpose,
            "personality_tone": self.personality_tone,
            "limitations": self.limitations,
            "planner_output": None,
            "executor_output": None,
            "validator_output": None,
            "output_node_response": None,
            "logger_output": None,
            "planning_attempts": 0,
            "execution_attempts": 0,
            "tools_executed": [],
            "react_iteration": 0,
            "token_usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

        with get_openai_callback() as cb:
            result = asyncio.run(self.graph.ainvoke(
                initial_state,
                config={"recursion_limit": self.recursion_limit}
            ))
            result["token_usage"] = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens
            }

        content = result.get("output_node_response", "")
        # logs and tools_executed in the result are now the cumulative lists
        full_logs = result.get("logs", [])
        original_log_count = len(limited_logs)
        new_logs_slice = full_logs[original_log_count:]

        tools_executed = result.get("tools_executed", [])
        token_usage = result.get("token_usage", {})

        return BotResponse(
            content=content,
            logs=new_logs_slice,
            tools_executed=tools_executed,
            token_usage=token_usage,
        )

    def _limit_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """Limit historical messages."""
        if len(messages) <= self.max_messages:
            return messages
        return messages[-self.max_messages:]

    def _limit_logs(self, logs: List[str]) -> List[str]:
        """Limit historical logs."""
        if len(logs) <= self.max_logs:
            return logs
        return logs[-self.max_logs:]
