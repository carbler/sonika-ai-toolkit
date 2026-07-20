"""OrchestratorBot — autonomous orchestration engine (Fast ReAct Edition)."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_community.callbacks.manager import get_openai_callback
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt
from tenacity import (
    retry,
    retry_if_exception,
    wait_exponential,
    stop_after_attempt,
)

from sonika_ai_toolkit.utilities.types import BotResponse, ILanguageModel
from sonika_ai_toolkit.utilities.questions import (
    ASK_USER_TOOL_NAME,
    questions_to_payload,
    questions_summary,
)
from sonika_ai_toolkit.tools.registry import ToolRegistry
from sonika_ai_toolkit.tools.plan_tools import (
    PLAN_SIGNAL_TOOL_NAMES,
    SET_PLAN_TOOL_NAME,
    SetPlanTool,
    UpdateStepTool,
)
from sonika_ai_toolkit.skills import (
    Skill,
    merge_skill_tools,
    render_skills_prompt,
    resolve_skills,
)
from sonika_ai_toolkit.agents.orchestrator.state import OrchestratorState
from sonika_ai_toolkit.agents.orchestrator.memory import MemoryManager
from sonika_ai_toolkit.agents.orchestrator.events import StatusEvent
from sonika_ai_toolkit.agents.orchestrator.planning import (
    PLANNING_PROTOCOL_PROMPT,
    apply_update_step,
    normalize_plan,
    render_plan_status,
    split_plan_signal_calls,
)
from sonika_ai_toolkit.agents.orchestrator.interface import IOrchestratorBot
from sonika_ai_toolkit.agents.react import (
    extract_thinking,
    _has_image_content,
    _build_user_message,
    _graph_topology,
    _new_run_id,
    _wrap_node_traced,
)


def _is_rate_limit(exc: Exception) -> bool:
    """Return True when the exception represents a 429 / quota-exhausted error."""
    try:
        from google.api_core.exceptions import ResourceExhausted
        if isinstance(exc, ResourceExhausted):
            return True
    except ImportError:
        pass
    return "429" in str(exc) or "quota" in str(exc).lower() or "rate" in str(exc).lower()


def _extract_text_content(response) -> str:
    """Extract clean text content from an AIMessage, filtering out thinking blocks."""
    text = response.content
    if isinstance(text, list):
        parts = []
        for p in text:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type") != "thinking":
                parts.append(str(p.get("text", "") or p.get("content", "")))
        text = "".join(parts)
    return str(text).strip()


def _last_tool_call_message(messages) -> Optional[AIMessage]:
    """Return the most recent AIMessage that carries tool_calls.

    After the ``plan``/``ask_user`` nodes append their ToolMessage answers,
    ``messages[-1]`` is no longer the AIMessage that requested the batch —
    walk backwards to find it.
    """
    for msg in reversed(messages or []):
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            return msg
    return None


class OrchestratorBot(IOrchestratorBot):
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
        vision_model: Optional[ILanguageModel] = None, # Model used for image (vision) turns
        logger: Optional[logging.Logger] = None,
        prompts: Optional[Any] = None, # Legacy
        checkpointer: Any = None,
        enable_user_questions: bool = False,
        enable_planning: bool = False,
        skills: Optional[List[Skill]] = None,
        skills_dir: Optional[str] = None,
    ):
        self.model = strong_model
        # Model used for image (vision) turns. Falls back to the strong model so
        # existing callers keep working; pass a vision-capable model here to let
        # the orchestrator "see" images with a model of the caller's choosing.
        self.vision_model = vision_model or strong_model
        self.instructions = instructions

        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())

        self.memory_manager = MemoryManager(memory_path)
        # NOTE: this attribute predates folder-based skills — it points to the
        # memory-derived dir used by DynamicToolSynthesizer, NOT to the
        # `skills_dir` constructor param (folder-based skills below).
        self.skills_dir = self.memory_manager.sessions_dir.replace("sessions", "skills")

        # Folder/programmatic skills: instructions are appended to the system
        # prompt; skill tools are merged into the tool list (explicitly-passed
        # tools win on name collision).
        self.skills = resolve_skills(skills, skills_dir)
        self._skills_prompt = render_skills_prompt(self.skills)

        self.registry = ToolRegistry()
        self.tools = []
        for tool in merge_skill_tools(list(tools or []), self.skills):
            self.registry.register(tool)
            self.tools.append(tool)

        # Register the structured-question tool so the model can pause and ask the
        # caller via a native LangGraph interrupt (see tools_node interception).
        self.enable_user_questions = enable_user_questions
        if enable_user_questions and not self.registry.get(ASK_USER_TOOL_NAME):
            from sonika_ai_toolkit.tools.ask_user import AskUserQuestionTool
            ask_tool = AskUserQuestionTool()
            self.registry.register(ask_tool)
            self.tools.append(ask_tool)

        # Structured plan signal tools: registered only when planning is
        # enabled; their calls are intercepted in agent_node, never executed.
        self.enable_planning = enable_planning
        if enable_planning:
            for plan_tool in (SetPlanTool(), UpdateStepTool()):
                if not self.registry.get(plan_tool.name):
                    self.registry.register(plan_tool)
                    self.tools.append(plan_tool)

        self.model_with_tools = self.model.model.bind_tools(self.tools) if self.tools else self.model.model
        self.checkpointer = checkpointer or MemorySaver()
        
        # Singleton compiled graph
        self.graph = self._build_workflow().compile(checkpointer=self.checkpointer)
        self._last_resume_command = None
        # Set by abort() to stop the in-flight astream_events run at the next
        # event boundary. Reset at the start of every astream_events call.
        self._abort_requested = False

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

            # Load memory dynamically (simplified)
            memory_context = self.memory_manager.read_memory()

            system_prompt = self.instructions + f"\n\nContexto de memoria:\n{memory_context}"

            if self._skills_prompt:
                system_prompt += "\n\n" + self._skills_prompt

            if mode == "plan":
                system_prompt += "\n\n[MODO PLAN] Ignora todas tus herramientas. Devuelve ÚNICAMENTE un plan detallado paso a paso en texto plano/markdown para resolver la petición del usuario. NO EJECUTES HERRAMIENTAS."
            elif self.enable_planning:
                system_prompt += "\n\n" + PLANNING_PROTOCOL_PROMPT
                plan_snapshot = state.get("plan") or []
                if plan_snapshot:
                    system_prompt += "\n\n" + render_plan_status(plan_snapshot)

            # Vision turn: with tools bound, some models (e.g. gpt-4o-mini) refuse an
            # image question ("no puedo ayudar") when no tool applies. So for image
            # turns we call the model WITHOUT tools — vision Q&A rarely needs them.
            has_image = _has_image_content(state.get("messages", []))
            if has_image:
                system_prompt += (
                    "\n\n[IMAGEN] El usuario compartió una imagen. Analízala y responde "
                    "directamente sobre lo que muestra de forma clara y útil."
                )
            # On image turns use the configured vision model (without tools);
            # otherwise the regular tool-bound strong model.
            active_model = self.vision_model.model if has_image else self.model_with_tools

            messages = [SystemMessage(content=system_prompt)] + state.get("messages", [])

            # Stream response to capture thinking
            accumulated_chunk = None
            retry_events: List[StatusEvent] = []

            def _record_retry(retry_state) -> None:
                wait_s = getattr(retry_state.next_action, "sleep", 0) or 0
                retry_events.append(StatusEvent(
                    type="retrying",
                    reason="rate_limit",
                    attempt=retry_state.attempt_number,
                    wait_s=round(float(wait_s), 1),
                ))

            @retry(
                retry=retry_if_exception(_is_rate_limit),
                wait=wait_exponential(multiplier=1, min=2, max=60),
                stop=stop_after_attempt(5),
                before_sleep=_record_retry,
                reraise=True,
            )
            async def _call_model_stream():
                nonlocal accumulated_chunk
                accumulated_chunk = None
                async for chunk in active_model.astream(messages):
                    accumulated_chunk = chunk if accumulated_chunk is None else (accumulated_chunk + chunk)

            try:
                await _call_model_stream()
            except Exception:
                # Fallback to invoke if streaming or retries fail
                accumulated_chunk = await active_model.ainvoke(messages)

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

            accumulated_thinking = state.get("thinking", "")
            if new_thinking:
                accumulated_thinking = (accumulated_thinking + "\n" + new_thinking).strip()

            # Extract clean text content
            text_content = _extract_text_content(response)

            result: Dict[str, Any] = {
                "messages": [response],
                "thinking": accumulated_thinking,
                "status_events": retry_events,
            }

            if not response.tool_calls:
                # Final turn — set final_report
                final_report = state.get("final_report", "")
                if text_content:
                    final_report = text_content
                else:
                    # Fallback: model generated empty content after tool execution
                    # Use output from most recent ToolMessage(s) in history
                    messages = state.get("messages", [])
                    tool_outputs = []
                    for msg in reversed(messages):
                        if isinstance(msg, ToolMessage):
                            tool_outputs.insert(0, msg.content)
                        elif getattr(msg, "tool_calls", None):
                            break  # Stop at AIMessage that originated this batch
                    if tool_outputs:
                        final_report = "\n\n".join(tool_outputs)
                result["final_report"] = final_report
            elif text_content:
                # Intermediate turn with text + tool_calls → partial response
                result["partial_response"] = text_content
                result["partial_responses"] = [text_content]

            return result

        async def plan_node(state: OrchestratorState) -> Dict[str, Any]:
            """Dedicated node for the set_plan / update_step signal calls.

            Applies the signals to the plan snapshot and answers them with
            acknowledgment ToolMessages (the history must remain a normal tool
            round-trip — models misbehave when the conversation ends on a bare
            AI message). Signal calls are NOT real actions: they never appear
            in tools_executed. Any real tool calls in the same batch are
            executed afterwards by the tools node (see routing).
            """
            last_message = _last_tool_call_message(state.get("messages"))
            signal_calls, _real = split_plan_signal_calls(
                getattr(last_message, "tool_calls", None) or []
            )
            plan = list(state.get("plan") or [])
            step_events: List[Dict[str, Any]] = []
            results = []
            for call in signal_calls:
                args = call.get("args") or {}
                if call.get("name") == SET_PLAN_TOOL_NAME:
                    plan = normalize_plan(args.get("steps"))
                else:
                    plan = apply_update_step(plan, args.get("step"), args.get("status"))
                    step_events.append({"step": args.get("step"), "status": args.get("status")})
                tool_instance = self.registry.get(call.get("name"))
                try:
                    ack = tool_instance._run(**args) if tool_instance else "ok"
                except Exception as e:
                    ack = f"Error: {e}"
                results.append(ToolMessage(tool_call_id=call["id"], content=str(ack)))
            update: Dict[str, Any] = {"messages": results, "plan": plan}
            if step_events:
                update["step_events"] = step_events
            return update

        async def ask_user_node(state: OrchestratorState) -> Dict[str, Any]:
            """Dedicated node for structured user questions (ask_user tool).

            Pauses via a native LangGraph interrupt and waits for the caller's
            answers (delivered through set_resume_command()); the loop then
            continues in the SAME run with the answers in context. Asking wins
            over any other tool call in the batch: real calls are answered
            with a deferred ToolMessage (never executed) so the model re-issues
            them with the user's answers in hand. The interrupt is the first
            side effect, so resuming re-executes nothing else.
            """
            last_message = _last_tool_call_message(state.get("messages"))
            calls = getattr(last_message, "tool_calls", None) or []
            ask_call = next(c for c in calls if c.get("name") == ASK_USER_TOOL_NAME)
            payload = questions_to_payload(ask_call.get("args") or {})
            answers = interrupt({"type": "question_request", **payload})
            answer_text = (
                answers if isinstance(answers, str)
                else json.dumps(answers, ensure_ascii=False, default=str)
            )
            results = [ToolMessage(
                tool_call_id=ask_call["id"],
                content=f"User answers: {answer_text}",
            )]
            for call in calls:
                if call is ask_call or call.get("name") in PLAN_SIGNAL_TOOL_NAMES:
                    continue  # plan signals were already answered by plan_node
                results.append(ToolMessage(
                    tool_call_id=call["id"],
                    content=(
                        "Not executed: the user's answers arrived first. "
                        "Call the tool again now if it is still needed."
                    ),
                ))
            return {
                "messages": results,
                "tools_executed": [{
                    "tool_name": ASK_USER_TOOL_NAME,
                    "args": ask_call.get("args") or {},
                    "status": "success",
                    "output": str(answer_text)[:500],
                }],
            }

        async def tools_node(state: OrchestratorState) -> Dict[str, Any]:
            last_message = _last_tool_call_message(state.get("messages"))
            results = []
            tools_executed = []

            if last_message is None:
                return {}

            for tool_call in last_message.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # Plan signals were applied + acknowledged by plan_node.
                if tool_name in PLAN_SIGNAL_TOOL_NAMES:
                    continue

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

                except Exception as e:
                    results.append(ToolMessage(tool_call_id=tool_call["id"], content=f"Error: {e}"))
                    tools_executed.append({
                        "tool_name": tool_name,
                        "args": tool_args,
                        "status": "error",
                        "output": str(e)
                    })

            return {"messages": results, "tools_executed": tools_executed}

        def should_continue(state: OrchestratorState) -> str:
            """Route after agent: plan signals first, then ask_user, then tools."""
            last_message = state["messages"][-1]
            calls = getattr(last_message, "tool_calls", None) or []
            if not calls:
                return END
            if self.enable_planning and split_plan_signal_calls(calls)[0]:
                return "plan"
            if self.enable_user_questions and any(
                c.get("name") == ASK_USER_TOOL_NAME for c in calls
            ):
                return "ask_user"
            return "tools"

        def route_after_plan(state: OrchestratorState) -> str:
            """After the plan node: dispatch the remaining (real) calls."""
            last_message = _last_tool_call_message(state.get("messages"))
            _signals, real_calls = split_plan_signal_calls(
                getattr(last_message, "tool_calls", None) or []
            )
            if not real_calls:
                return "agent"
            if self.enable_user_questions and any(
                c.get("name") == ASK_USER_TOOL_NAME for c in real_calls
            ):
                return "ask_user"
            return "tools"

        workflow = StateGraph(OrchestratorState)
        # Every node (built-in and custom) is wrapped so its execution is
        # recorded in `node_trace` — the source of the ("graph", node_invoked)
        # stream events and of BotResponse.node_trace.
        workflow.add_node("agent", _wrap_node_traced("agent", agent_node))
        workflow.add_node("tools", _wrap_node_traced("tools", tools_node))
        if self.enable_planning:
            workflow.add_node("plan", _wrap_node_traced("plan", plan_node))
        if self.enable_user_questions:
            workflow.add_node("ask_user", _wrap_node_traced("ask_user", ask_user_node))

        # ── Fixed wiring (classic ReAct graph) ─────────────────────────────
        workflow.set_entry_point("agent")

        agent_routes = {"tools": "tools", END: END}
        if self.enable_planning:
            agent_routes["plan"] = "plan"
        if self.enable_user_questions:
            agent_routes["ask_user"] = "ask_user"
        workflow.add_conditional_edges("agent", should_continue, agent_routes)

        if self.enable_planning:
            plan_routes = {"agent": "agent", "tools": "tools"}
            if self.enable_user_questions:
                plan_routes["ask_user"] = "ask_user"
            workflow.add_conditional_edges("plan", route_after_plan, plan_routes)
        if self.enable_user_questions:
            workflow.add_edge("ask_user", "agent")

        workflow.add_edge("tools", "agent")

        return workflow

    # ── Public API ─────────────────────────────────────────────────────────

    def get_graph_topology(self) -> Dict[str, Any]:
        """Static node/edge layout of the compiled graph, for drawing it.

        Returns ``{"entry": str, "nodes": [str], "edges": [{"source", "target",
        "conditional"}]}``. Nodes include the virtual ``__start__``/``__end__``
        markers and any custom nodes. The same payload (plus ``run_id``) is
        emitted as the first ``("graph", …)`` event of ``astream_events``.
        """
        return _graph_topology(self.graph)

    async def astream_events(self, goal: str, mode: str = "ask", thread_id: str = None, images: Optional[List[str]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        New Streaming API that yields typed events, completely decoupling logic from the UI.
        If `goal` is None/empty, it assumes we are resuming an interrupted state (from `Command`).

        Besides LangGraph's ``("messages", …)`` / ``("updates", …)`` tuples,
        yields a third stream mode ``"graph"``:
          - ``("graph", GraphTopologyEvent)`` — first event of a new run: the
            node/edge layout plus the unique ``run_id`` of this process.
          - ``("graph", NodeInvokedEvent)``  — one per node execution, in
            order, so consumers can animate the path taken over the topology.
        """
        run_id = _new_run_id()
        current_thread = thread_id or str(uuid.uuid4())[:8]

        config = {"configurable": {"thread_id": current_thread}}

        # Reset before any yield so an abort() the caller fires while consuming
        # the very first event (the topology) is not clobbered below.
        self._abort_requested = False

        if goal:
            # First turn: provide initial state
            input_state = {
                "goal": goal,
                "mode": mode,
                "messages": [_build_user_message(goal, images)],
                "session_id": current_thread,
                "run_id": run_id,
                "skills_dir": self.skills_dir,
                "session_log": [],
                "tools_executed": [],
                "status_events": [],
                "partial_responses": [],
                "plan": [],
                "step_events": [],
                "node_trace": [],
            }

            stream_input = input_state

            # First event of the run: the graph layout, so consumers can draw
            # the full graph before any node executes.
            yield ("graph", {
                "type": "graph_topology",
                "run_id": run_id,
                **self.get_graph_topology(),
            })
        elif getattr(self, "_last_resume_command", None):
            # We are resuming from an interrupt!
            stream_input = self._last_resume_command
            self._last_resume_command = None
        else:
            raise ValueError("Must provide either a goal or resume from an interrupt via astream_events(None, command).")

        node_seq = 0
        async for stream_mode, payload in self.graph.astream(stream_input, config=config, stream_mode=["messages", "updates"]):
            # Cooperative abort: bot.abort() (from another task) flips the flag;
            # we stop at the next event boundary. Breaking closes the astream
            # generator, cancelling the run — state up to the last completed
            # node stays in the checkpointer (thread_id). "aborted" is the last
            # event yielded.
            if self._abort_requested:
                self._abort_requested = False
                yield ("graph", {"type": "aborted", "run_id": run_id})
                break
            # Synthesize a ("graph", node_invoked) signal per node execution.
            # The run_id/ts come from the node's own node_trace delta so they
            # stay consistent even across interrupt resumes.
            if stream_mode == "updates" and isinstance(payload, dict):
                for node_name, delta in payload.items():
                    if node_name.startswith("__"):
                        continue
                    node_seq += 1
                    trace = (delta or {}).get("node_trace") or [{}]
                    yield ("graph", {
                        "type": "node_invoked",
                        "run_id": trace[-1].get("run_id") or run_id,
                        "node": node_name,
                        "seq": node_seq,
                        "ts": trace[-1].get("ts") or time.time(),
                        "detail": trace[-1].get("detail", {}),
                    })
            # Yield structured events for the CLI
            yield (stream_mode, payload)

    async def arun(self, goal: str, context: str = "", thread_id: str = None, images: Optional[List[str]] = None) -> BotResponse:
        """
        Legacy Async API for compatibility with `chat.py`.
        Consumes the stream silently until the end, ignoring interrupts (auto mode).
        """
        current_thread = thread_id or str(uuid.uuid4())[:8]
        config = {"configurable": {"thread_id": current_thread}}

        run_id = _new_run_id()
        initial_state = {
            "goal": goal,
            "mode": "auto",  # Force auto so we don't interrupt old scripts
            "messages": [_build_user_message(goal, images)],
            "session_id": current_thread,
            "run_id": run_id,
            "skills_dir": self.skills_dir,
            "session_log": [],
            "tools_executed": [],
            "status_events": [],
            "partial_responses": [],
            "plan": [],
            "step_events": [],
            "node_trace": [],
        }

        # In `arun` (used by older tests/chat.py), we just want the final result.
        # Wrap in the OpenAI callback so token usage is tracked (works for
        # OpenAI-compatible providers: openai, deepseek). Others report 0.
        with get_openai_callback() as cb:
            final_state = await self.graph.ainvoke(initial_state, config=config)
        token_usage = {
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
            "total_tokens": cb.total_tokens,
        }

        # If the agent called ask_user, the graph paused on a native interrupt.
        # Surface the structured questions so non-streaming callers can react.
        question_payload = self._extract_question_interrupt(final_state)
        content = final_state.get("final_report", "")
        if question_payload and not content:
            content = questions_summary(question_payload)

        node_trace = [
            {**entry, "seq": i + 1}
            for i, entry in enumerate(final_state.get("node_trace") or [])
        ]

        return BotResponse(
            content=content,
            thinking=final_state.get("thinking", None),
            tools_executed=final_state.get("tools_executed", []),
            logs=final_state.get("session_log", []),
            token_usage=token_usage,
            success=True,
            plan=final_state.get("plan", []) or [],
            session_id=current_thread,
            goal=goal,
            questions=question_payload.get("questions", []) if question_payload else [],
            needs_input=bool(question_payload),
            run_id=run_id,
            node_trace=node_trace,
        )

    @staticmethod
    def _extract_question_interrupt(final_state: Any) -> Optional[Dict[str, Any]]:
        """Return the question_request payload if the run paused on an ask_user interrupt."""
        interrupts = None
        if isinstance(final_state, dict):
            interrupts = final_state.get("__interrupt__")
        if not interrupts:
            return None
        if not isinstance(interrupts, (list, tuple)):
            interrupts = [interrupts]
        for it in interrupts:
            value = getattr(it, "value", it)
            if isinstance(value, dict) and value.get("type") == "question_request":
                return value
        return None

    def run(self, goal: str, context: str = "", thread_id: str = None, images: Optional[List[str]] = None) -> BotResponse:
        """Legacy Sync API."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            pass
        return asyncio.run(self.arun(goal, context, thread_id, images))

    def set_resume_command(self, resume_data: Any):
        from langgraph.types import Command
        self._last_resume_command = Command(resume=resume_data)

    def abort(self):
        """Stop the in-flight ``astream_events`` run at the next event boundary.

        Meant to be called from a different task than the one consuming the
        stream (e.g. a UI/websocket handler) while the graph is running. The
        stream yields a final ``("graph", {"type": "aborted", ...})`` event and
        then stops. Because streaming yields on every LLM token, an abort during
        the agent's reasoning takes effect almost immediately; an abort while a
        tool is executing only applies once that tool returns (a running node
        cannot be cancelled mid-execution). State up to the last completed node
        is preserved in the checkpointer under the run's ``thread_id``; the
        non-streaming ``run``/``arun`` paths use ``ainvoke`` and are not affected
        by this flag — cancel their asyncio task to stop them.
        """
        self._abort_requested = True
