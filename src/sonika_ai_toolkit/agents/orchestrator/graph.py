"""OrchestratorBot — autonomous orchestration engine (Fast ReAct Edition)."""

import asyncio
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt

from sonika_ai_toolkit.utilities.types import BotResponse, ILanguageModel
from sonika_ai_toolkit.tools.registry import ToolRegistry
from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState
from sonika_ai_toolkit.agents.orchestrator.memory import MemoryManager
from sonika_ai_toolkit.agents.react import extract_thinking


class OrchestratorBot:
    """
    Fast, Singleton, ReAct-based Orchestrator with native LangGraph interrupts.
    """

    def __init__(
        self,
        strong_model: ILanguageModel,
        fast_model: ILanguageModel, # Kept for signature compatibility
        instructions: str,
        tools: Optional[List[BaseTool]] = None,
        memory_path: str = "./memory",
        risk_threshold: int = 1, # Kept for signature compatibility
        max_retries: int = 3, # Kept for signature compatibility
        code_model: Optional[ILanguageModel] = None, # Kept for signature compatibility
        on_human_approval: Optional[Callable[[Dict], bool]] = None, # Legacy
        on_step_start: Optional[Callable[[Dict], None]] = None, # Legacy
        on_step_end: Optional[Callable[[Dict, str], None]] = None, # Legacy
        on_plan_generated: Optional[Callable[[List], None]] = None, # Legacy
        on_thinking: Optional[Callable[[str], None]] = None, # Legacy
        on_message: Optional[Callable[[str], None]] = None, # Legacy
        logger: Optional[logging.Logger] = None,
        prompts: Optional[Any] = None, # Legacy
        checkpointer: Any = None,
    ):
        self.model = strong_model
        self.instructions = instructions
        
        # Callbacks (kept for backward compatibility with old scripts)
        self.on_thinking = on_thinking
        self.on_step_start = on_step_start
        self.on_step_end = on_step_end
        self.on_message = on_message
        
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())

        self.memory_manager = MemoryManager(memory_path)
        self.skills_dir = self.memory_manager.sessions_dir.replace("sessions", "skills")

        self.registry = ToolRegistry()
        self.tools = []
        for tool in tools or []:
            self.registry.register(tool)
            self.tools.append(tool)

        self.model_with_tools = self.model.model.bind_tools(self.tools) if self.tools else self.model.model
        self.checkpointer = checkpointer or MemorySaver()
        
        # Singleton compiled graph
        self.graph = self._build_workflow().compile(checkpointer=self.checkpointer)
        self._last_resume_command = None

    async def a_prewarm(self):
        """Send a dummy request to the LLM to open the TCP/TLS connection early."""
        try:
            # Send a fast ping to models that support it, or a very cheap request
            await self.model.model.ainvoke([HumanMessage(content="1")])
            self.logger.debug("LLM Pre-warmed successfully.")
        except Exception as e:
            self.logger.warning(f"Failed to pre-warm LLM: {e}")

    def _build_workflow(self) -> StateGraph:
        async def agent_node(state: OrchestratorState) -> Dict[str, Any]:
            mode = state.get("mode", "ask")
            goal = state.get("goal", "")
            
            # Load memory dynamically (simplified)
            memory_context = self.memory_manager.get_summary()
            
            system_prompt = self.instructions + f"\n\nContexto de memoria:\n{memory_context}"
            
            if mode == "plan":
                system_prompt += "\n\n[MODO PLAN] Ignora todas tus herramientas. Devuelve ÚNICAMENTE un plan detallado paso a paso en texto plano/markdown para resolver la petición del usuario. NO EJECUTES HERRAMIENTAS."

            messages = [SystemMessage(content=system_prompt)] + state.get("messages", [])
            
            # Stream response to capture thinking
            accumulated_chunk = None
            thinking_emitted = False
            
            try:
                async for chunk in self.model_with_tools.astream(messages):
                    if isinstance(chunk.content, list):
                        for part in chunk.content:
                            if isinstance(part, dict) and part.get("type") == "thinking":
                                t = part.get("thinking", "")
                                if t:
                                    thinking_emitted = True
                                    if self.on_thinking:
                                        self.on_thinking(t)
                    accumulated_chunk = chunk if accumulated_chunk is None else (accumulated_chunk + chunk)
            except Exception as e:
                # Fallback to invoke if streaming fails
                accumulated_chunk = await self.model_with_tools.ainvoke(messages)

            if accumulated_chunk is None:
                response = AIMessage(content="")
            else:
                tc = getattr(accumulated_chunk, "tool_calls", []) or []
                
                # If mode is plan, explicitly strip tool calls to force the end
                if mode == "plan":
                    tc = []
                    
                response = AIMessage(
                    content=accumulated_chunk.content,
                    tool_calls=tc,
                    additional_kwargs=getattr(accumulated_chunk, "additional_kwargs", {}),
                )
                
            new_thinking = extract_thinking(response)
            if new_thinking and self.on_thinking and not thinking_emitted:
                self.on_thinking(new_thinking)
                
            accumulated_thinking = state.get("thinking", "")
            if new_thinking:
                accumulated_thinking = (accumulated_thinking + "\n" + new_thinking).strip()

            # Record final report if no tools are called
            final_report = state.get("final_report", "")
            if not response.tool_calls:
                # Clean content
                text = response.content
                if isinstance(text, list):
                    text = "\n".join(str(p.get("text", "")) for p in text if isinstance(p, dict) and p.get("type") != "thinking")
                final_report = str(text)

            return {
                "messages": [response],
                "thinking": accumulated_thinking,
                "final_report": final_report
            }

        async def tools_node(state: OrchestratorState) -> Dict[str, Any]:
            last_message = state["messages"][-1]
            results = []
            tools_executed = []
            
            if not getattr(last_message, "tool_calls", None):
                return {}
                
            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_instance = self.registry.get(tool_name)
                
                if not tool_instance:
                    results.append(ToolMessage(tool_call_id=tool_call["id"], content=f"Error: Tool '{tool_name}' not found."))
                    continue
                
                risk_level = getattr(tool_instance, "risk_level", getattr(tool_instance, "risk_hint", 0))
                
                if state.get("mode", "ask") == "ask" and risk_level > 0:
                    preview_data = {}
                    if hasattr(tool_instance, "preview"):
                        try:
                            preview_data["diff"] = tool_instance.preview(tool_args)
                        except Exception:
                            pass
                    
                    # Interrupción nativa de LangGraph
                    approval = interrupt({
                        "type": "permission_request",
                        "tool": tool_name,
                        "params": tool_args,
                        **preview_data
                    })
                    
                    # Use langgraph's Command object or literal bool
                    if isinstance(approval, dict) and "approved" in approval:
                        approved = approval["approved"]
                    else:
                        approved = bool(approval)
                        
                    if not approved:
                        results.append(ToolMessage(tool_call_id=tool_call["id"], content="Acción cancelada por el usuario."))
                        tools_executed.append({
                            "tool_name": tool_name, "args": tool_args, "status": "skipped", "output": "Rejected by user"
                        })
                        continue
                        
                # Execute tool
                if self.on_step_start:
                    self.on_step_start({"tool_name": tool_name, "params": tool_args})
                    
                try:
                    import asyncio
                    if hasattr(tool_instance, "ainvoke"):
                        output = await tool_instance.ainvoke(tool_args)
                    elif hasattr(tool_instance, "invoke"):
                        output = await asyncio.to_thread(tool_instance.invoke, tool_args)
                    else:
                        output = await asyncio.to_thread(tool_instance._run, **tool_args)
                        
                    results.append(ToolMessage(tool_call_id=tool_call["id"], content=str(output)))
                    tools_executed.append({
                        "tool_name": tool_name,
                        "args": tool_args,
                        "status": "success",
                        "output": str(output)[:500]
                    })
                    
                    if self.on_step_end:
                        self.on_step_end({"tool_name": tool_name, "params": tool_args}, str(output))
                        
                except Exception as e:
                    results.append(ToolMessage(tool_call_id=tool_call["id"], content=f"Error: {e}"))
                    tools_executed.append({
                        "tool_name": tool_name,
                        "args": tool_args,
                        "status": "error",
                        "output": str(e)
                    })
                    
                    if self.on_step_end:
                        self.on_step_end({"tool_name": tool_name, "params": tool_args}, f"Error: {e}")
            
            return {"messages": results, "tools_executed": tools_executed}

        def should_continue(state: OrchestratorState) -> str:
            last_message = state["messages"][-1]
            if getattr(last_message, "tool_calls", None):
                return "tools"
            return END

        workflow = StateGraph(OrchestratorState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tools_node)
        
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
        workflow.add_edge("tools", "agent")

        return workflow

    # ── Public API ─────────────────────────────────────────────────────────

    async def astream_events(self, goal: str, mode: str = "ask", thread_id: str = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        New Streaming API that yields typed events, completely decoupling logic from the UI.
        If `goal` is None/empty, it assumes we are resuming an interrupted state (from `Command`).
        """
        run_id = str(uuid.uuid4())[:8]
        current_thread = thread_id or run_id
        
        config = {"configurable": {"thread_id": current_thread}}
        
        from langgraph.types import Command
        
        if goal:
            # First turn: provide initial state
            input_state = {
                "goal": goal,
                "mode": mode,
                "messages": [HumanMessage(content=goal)],
                "session_id": current_thread,
                "skills_dir": self.skills_dir,
                "session_log": [],
                "tools_executed": []
            }
            # Add backward compatibility for old scripts using callbacks
            if self.on_message:
                self.on_message(goal)
                
            stream_input = input_state
        elif getattr(self, "_last_resume_command", None):
            # We are resuming from an interrupt!
            stream_input = self._last_resume_command
            self._last_resume_command = None
        else:
            raise ValueError("Must provide either a goal or resume from an interrupt via astream_events(None, command).")

        async for event in self.graph.astream(stream_input, config=config, stream_mode=["messages", "updates"]):
            # Yield structured events for the CLI
            yield event

    async def arun(self, goal: str, context: str = "", thread_id: str = None) -> BotResponse:
        """
        Legacy Async API for compatibility with `chat.py`.
        Consumes the stream silently until the end, ignoring interrupts (auto mode).
        """
        current_thread = thread_id or str(uuid.uuid4())[:8]
        config = {"configurable": {"thread_id": current_thread}}
        
        initial_state = {
            "goal": goal,
            "mode": "auto", # Force auto so we don't interrupt old scripts
            "messages": [HumanMessage(content=goal)],
            "session_id": current_thread,
            "skills_dir": self.skills_dir,
            "session_log": [],
            "tools_executed": []
        }
        
        # In `arun` (used by older tests/chat.py), we just want the final result
        final_state = await self.graph.ainvoke(initial_state, config=config)
        
        return BotResponse(
            content=final_state.get("final_report", ""),
            thinking=final_state.get("thinking", None),
            tools_executed=final_state.get("tools_executed", []),
            logs=final_state.get("session_log", []),
            token_usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            success=True,
            plan=[],
            session_id=current_thread,
            goal=goal
        )

    def run(self, goal: str, context: str = "", thread_id: str = None) -> BotResponse:
        """Legacy Sync API."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            pass
        return asyncio.run(self.arun(goal, context, thread_id))

    def set_resume_command(self, resume_data: Any):
        from langgraph.types import Command
        self._last_resume_command = Command(resume=resume_data)
