from typing import List, Optional, Dict, Any, TypedDict, Annotated, Callable, Union, get_origin, get_args, Generator
import asyncio
import logging
import inspect
import re
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage, SystemMessage
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.callbacks.manager import get_openai_callback
from langgraph.config import get_config

from sonika_ai_toolkit.utilities.types import BotResponse, ILanguageModel, Message


# ============= MODULE-LEVEL HELPERS =============

def _get_text_content(content) -> str:
    """Extract only the text (non-thinking) parts from a message content.

    Handles both plain strings and the list format that Gemini returns when
    include_thoughts=True: [{'type': 'thinking', 'thinking': '...'}, 'text...']
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") != "thinking":
                text = part.get("text") or part.get("content") or ""
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()
    return str(content) if content else ""


def extract_thinking(response: AIMessage) -> Optional[str]:
    """Extract thinking content from provider-specific responses.

    Args:
        response: AIMessage returned by the model.

    Returns:
        Extracted thinking text or None if not present.
    """
    try:
        # Gemini with include_thoughts=True: content is a list of parts
        if isinstance(response.content, list):
            for part in response.content:
                if isinstance(part, dict) and part.get("type") == "thinking":
                    thinking = part.get("thinking") or part.get("text") or part.get("content")
                    if thinking:
                        return str(thinking).strip() or None

        additional = response.additional_kwargs or {}

        # Gemini: direct thinking field in additional_kwargs
        gem_thinking = additional.get("thinking")
        if gem_thinking:
            if isinstance(gem_thinking, str):
                return gem_thinking.strip() or None
            if isinstance(gem_thinking, list):
                combined = "\n".join(str(part) for part in gem_thinking).strip()
                return combined or None

        # DeepSeek R1
        deepseek_reasoning = additional.get("reasoning_content")
        if deepseek_reasoning:
            return str(deepseek_reasoning).strip() or None

        # Fallback: parse <think> ... </think> block
        if response.content:
            text = _get_text_content(response.content)
            match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
            if match:
                captured = match.group(1).strip()
                if captured:
                    return captured
    except Exception:
        return None

    return None


# ============= STATE DEFINITION =============

class ChatState(TypedDict):
    """
    Chat state for LangGraph workflow.

    Attributes:
        messages: List of conversation messages with automatic message handling
        logs: Historical logs for context
        token_usage: Accumulated token usage across all model invocations
        thinking: Accumulated reasoning from the model (empty string for non-thinking models)
    """
    messages: Annotated[List[BaseMessage], add_messages]
    logs: List[str]
    token_usage: Dict[str, int]
    thinking: str


# ============= CALLBACK HANDLER =============

class _InternalToolLogger(BaseCallbackHandler):
    """
    Internal callback handler that bridges LangChain callbacks to user-provided functions.

    This class is used internally to forward tool execution events to the optional
    callback functions provided by the user during bot initialization.
    """

    def __init__(self,
                 on_start: Optional[Callable[[str, str], None]] = None,
                 on_end: Optional[Callable[[str, str], None]] = None,
                 on_error: Optional[Callable[[str, str], None]] = None):
        super().__init__()
        self.on_start_callback = on_start
        self.on_end_callback = on_end
        self.on_error_callback = on_error
        self.current_tool_name = None
        self.tool_executions = []
        self.execution_logs = []

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts processing."""
        self.execution_logs.append("[AGENT] Thinking...")

    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM finishes processing."""
        if hasattr(response, 'generations') and response.generations:
            for generation in response.generations:
                if hasattr(generation[0], 'message') and hasattr(generation[0].message, 'tool_calls'):
                    tool_calls = generation[0].message.tool_calls
                    if tool_calls:
                        tool_names = [tc.get('name', 'unknown') for tc in tool_calls]
                        self.execution_logs.append(f"[AGENT] Decided to call tools: {', '.join(tool_names)}")
                        return

        self.execution_logs.append("[AGENT] Generated response")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when a tool starts executing."""
        tool_name = serialized.get("name", "unknown")
        self.current_tool_name = tool_name

        self.tool_executions.append({
            "tool_name": tool_name,
            "args": input_str,
            "status": "started"
        })

        self.execution_logs.append(f"[TOOL] Executing {tool_name}")
        self.execution_logs.append(f"[TOOL] Input: {input_str[:100]}...")

        if self.on_start_callback:
            try:
                self.on_start_callback(tool_name, input_str)
            except Exception as e:
                logging.error(f"Error in on_tool_start callback: {e}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when a tool completes successfully."""
        tool_name = self.current_tool_name or "unknown"

        if hasattr(output, 'content'):
            output_str = output.content
        elif isinstance(output, str):
            output_str = output
        else:
            output_str = str(output)

        if self.tool_executions:
            self.tool_executions[-1]["status"] = "success"
            self.tool_executions[-1]["output"] = output_str

        self.execution_logs.append(f"[TOOL] {tool_name} completed successfully")
        self.execution_logs.append(f"[TOOL] Output: {output_str[:100]}...")

        if self.on_end_callback:
            try:
                self.on_end_callback(tool_name, output_str)
            except Exception as e:
                logging.error(f"Error in on_tool_end callback: {e}")

        self.current_tool_name = None

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Called when a tool encounters an error."""
        tool_name = self.current_tool_name or "unknown"
        error_message = str(error)

        if self.tool_executions:
            self.tool_executions[-1]["status"] = "error"
            self.tool_executions[-1]["error"] = error_message

        self.execution_logs.append(f"[TOOL] {tool_name} failed: {error_message}")

        if self.on_error_callback:
            try:
                self.on_error_callback(tool_name, error_message)
            except Exception as e:
                logging.error(f"Error in on_tool_error callback: {e}")

        self.current_tool_name = None


# ============= MAIN BOT CLASS =============

class ReactBot:
    """
    Modern LangGraph-based conversational bot with MCP support and optional thinking/reasoning.

    This implementation provides 100% API compatibility with existing ChatService
    while using modern LangGraph workflows and native tool calling internally.

    Features:
        - Native tool calling (no manual parsing)
        - MCP (Model Context Protocol) support
        - Complete token usage tracking across all model invocations
        - Thread-based conversation persistence
        - Tool execution callbacks for real-time monitoring
        - Thinking/reasoning extraction for Gemini, DeepSeek R1, and fallback <think> models
        - Streaming support via stream_response
        - Backward compatibility with legacy APIs
    """

    def __init__(
        self,
        language_model: ILanguageModel,
        instructions: str,
        tools: Optional[List[BaseTool]] = None,
        mcp_servers: Optional[Dict[str, Any]] = None,
        use_checkpointer: bool = False,
        thinking_budget: int = 8192,
        max_messages: int = 100,
        max_logs: int = 20,
        logger: Optional[logging.Logger] = None,
        on_tool_start: Optional[Callable[[str, str], None]] = None,
        on_tool_end: Optional[Callable[[str, str], None]] = None,
        on_tool_error: Optional[Callable[[str, str], None]] = None,
        on_thinking: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the ReactBot with optional MCP, thinking, and callback support.

        Args:
            language_model (ILanguageModel): The language model to use for generation
            instructions (str): System instructions for the bot
            tools (List[BaseTool], optional): LangChain tools to bind to the model
            mcp_servers (Dict[str, Any], optional): MCP server configurations
            use_checkpointer (bool): Enable conversation persistence via LangGraph checkpoints
            thinking_budget (int): Token budget for native thinking models (ignored otherwise)
            max_messages (int): Maximum number of messages to keep in history
            max_logs (int): Maximum number of logs to keep in history
            logger (Optional[logging.Logger]): Logger instance (silent by default if not provided)
            on_tool_start (Callable[[str, str], None], optional): Callback when a tool starts
            on_tool_end (Callable[[str, str], None], optional): Callback when a tool completes
            on_tool_error (Callable[[str, str], None], optional): Callback when a tool fails
            on_thinking (Callable[[str], None], optional): Callback for reasoning chunks
        """
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())

        self.language_model = language_model
        self.instructions = instructions
        self.thinking_budget = thinking_budget
        self.on_thinking = on_thinking

        self.max_messages = max_messages
        self.max_logs = max_logs

        self.chat_history: List[BaseMessage] = []
        self._current_logs: List[str] = []

        self.tools = tools or []
        self.mcp_client = None

        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_tool_error = on_tool_error

        if mcp_servers:
            self._initialize_mcp(mcp_servers)

        self.supports_native_thinking = bool(getattr(self.language_model, "supports_thinking", False))
        self.model_name = getattr(self.language_model.model, "model_name", "")

        if self.supports_native_thinking and self._is_deepseek_reasoner() and self.tools:
            raise ValueError(
                "DeepSeek reasoner does not support tool calling. Provide an empty tools list."
            )

        self.checkpointer = MemorySaver() if use_checkpointer else None

        self.model_with_tools = (
            self.language_model.model.bind_tools(self.tools)
            if self.tools
            else self.language_model.model
        )

        self.graph = self._create_workflow()

        self.conversation = None
        self.agent_executor = None

    # ============= INTERNAL HELPERS =============

    def _is_deepseek_reasoner(self) -> bool:
        name = (self.model_name or "").lower()
        return name == "deepseek-reasoner" or "r1" in name

    def _build_system_prompt(self, include_fallback_think: bool) -> str:
        system_content = self.instructions
        if self.tools:
            tool_names = ", ".join(tool.name for tool in self.tools)
            system_content += (
                "\n\nTOOL USAGE:\n"
                "- You MUST call the provided tools to execute the requested actions. Never describe or claim an action — execute it with a tool call.\n"
                "- If all required parameters are available in the conversation, call the tool immediately without asking for confirmation.\n"
                "- Only ask the user for a parameter if it is genuinely absent from the entire conversation. Never invent or assume values.\n"
                "- After tool outputs are received, give a concise final answer. Do NOT call tools again unless results explicitly require it.\n"
                f"- Available tools: {tool_names}"
            )
        if self._current_logs:
            logs_context = "\n".join(self._current_logs[-self.max_logs:])
            system_content += f"\n\nRecent logs:\n{logs_context}"
        if include_fallback_think:
            system_content += (
                "\n\nBefore giving your final answer, think step by step inside <think> tags."
                " Always share the final answer after the thinking block."
            )
        return system_content

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think>...</think> blocks from content (fallback thinking models)."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _clean_content(content) -> str:
        """Extract text from content, stripping thinking parts and <think> tags."""
        text = _get_text_content(content)
        return ReactBot._strip_think_tags(text)

    @staticmethod
    def _extract_token_usage(state: ChatState, cb) -> Dict[str, int]:
        """Return token usage from callback if available, else from message usage_metadata."""
        if cb.total_tokens > 0:
            return {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens,
            }
        prompt = completion = 0
        for msg in state.get("messages", []):
            meta = getattr(msg, "usage_metadata", None)
            if meta:
                prompt += meta.get("input_tokens", 0)
                completion += meta.get("output_tokens", 0)
        return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}

    def _finalize_response(
        self,
        state: ChatState,
        tool_logger: _InternalToolLogger,
        limited_logs: List[str],
        token_usage: Dict[str, int],
    ) -> Dict[str, Any]:
        final_response = ""
        for msg in reversed(state.get("messages", [])):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = self._clean_content(msg.content)
                break

        if final_response:
            tool_logger.execution_logs.append(f"[BOT] {final_response}")

        return BotResponse(
            content=final_response,
            thinking=state.get("thinking") or None,
            logs=limited_logs + tool_logger.execution_logs,
            tools_executed=tool_logger.tool_executions,
            token_usage=token_usage,
        )

    def _initialize_mcp(self, mcp_servers: Dict[str, Any]):
        """Initialize MCP (Model Context Protocol) connections and load available tools."""
        try:
            self.mcp_client = MultiServerMCPClient(mcp_servers)
            mcp_tools = asyncio.run(self.mcp_client.get_tools())
            self.tools.extend(mcp_tools)
            self.logger.info(f"MCP initialized with {len(mcp_tools)} tools")
        except Exception as e:
            self.logger.error(f"Error inicializando MCP: {e}")
            self.logger.exception("Traceback completo:")
            self.mcp_client = None

    def _extract_required_params(self, tool) -> Dict[str, List[str]]:
        """
        Extract the required parameters from a tool's schema.

        Supports:
        - LangChain BaseTool with args_schema (Pydantic v1/v2)
        - LangChain BaseTool with _run and type hints (Optional detection)
        - MCP Tools with inputSchema (JSON Schema)

        Returns:
            Dict with 'required' (list of required fields) and 'all' (all fields)
        """
        required = []
        all_params = []

        try:
            if hasattr(tool, 'inputSchema') and tool.inputSchema:
                schema = tool.inputSchema
                if isinstance(schema, dict):
                    all_params = list(schema.get('properties', {}).keys())
                    required = schema.get('required', [])
                    return {'required': required, 'all': all_params}

            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema

                if hasattr(schema, 'model_fields'):
                    for name, field in schema.model_fields.items():
                        all_params.append(name)
                        if field.is_required():
                            required.append(name)
                    return {'required': required, 'all': all_params}

                elif hasattr(schema, '__fields__'):
                    for name, field in schema.__fields__.items():
                        all_params.append(name)
                        if field.required:
                            required.append(name)
                    return {'required': required, 'all': all_params}

                elif isinstance(schema, dict):
                    all_params = list(schema.get('properties', {}).keys())
                    required = schema.get('required', [])
                    return {'required': required, 'all': all_params}

            if hasattr(tool, '_run'):
                sig = inspect.signature(tool._run)
                type_hints = {}
                try:
                    type_hints = tool._run.__annotations__
                except Exception:
                    pass

                for name, param in sig.parameters.items():
                    if name in ('self', 'kwargs', 'args'):
                        continue

                    all_params.append(name)

                    is_optional = False
                    if name in type_hints:
                        hint = type_hints[name]
                        origin = get_origin(hint)
                        if origin is Union:
                            args = get_args(hint)
                            if type(None) in args:
                                is_optional = True

                    has_default = param.default != inspect.Parameter.empty

                    if not is_optional and not has_default:
                        required.append(name)

        except Exception as e:
            self.logger.warning(f"Could not extract schema for {tool.name}: {e}")

        return {'required': required, 'all': all_params}

    def _build_conditional_rules(self) -> str:
        """Build conditional rules based on available tools."""
        if not self.tools:
            return ""

        tool_names = {tool.name for tool in self.tools}
        rules = []

        if 'search_knowledge_documents' in tool_names:
            rules.append(
                "\n## CORPORATE RULE — MANDATORY USE OF `search_knowledge_documents`\n"
                "If the user's query might be answered by internal documents:\n"
                "- ALWAYS call `search_knowledge_documents` FIRST before responding\n"
                "- Use the user's message as the query\n"
                "- Never invent information if it might exist in documents\n"
            )

        if 'accept_policies' in tool_names:
            rules.append(
                "\n## POLICY ACCEPTANCE HANDLING\n"
                "- On the FIRST user message of the conversation, ask if they accept the privacy policies.\n"
                "- Do NOT call the `accept_policies` tool automatically.\n"
                "- Wait for the user's explicit confirmation before calling the tool.\n"
                "- This rule is applied only once: after `accept_policies` executes, never ask again.\n"
            )

        if 'create_or_update_contact' in tool_names:
            rules.append(
                "\n## AUTOMATIC CONTACT UPDATE\n"
                "If the user provides contact information (name, email, phone):\n"
                "- ALWAYS call `create_or_update_contact` immediately\n"
                "- Execute this BEFORE any other action\n"
            )

        return "\n".join(rules) if rules else ""

    def tool_validator_node(self, state: ChatState) -> Dict[str, Any]:
        """
        Validates tool calls, executing valid ones and returning errors for invalid ones.
        This allows for partial success and gives the agent detailed feedback for self-correction.
        """
        last_message = state["messages"][-1]

        if not (isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls):
            return state

        tool_calls = last_message.tool_calls
        tools_by_name = {tool.name: tool for tool in self.tools}

        valid_calls = []
        invalid_call_messages = []

        for tool_call in tool_calls:
            tool_name = tool_call.get('name')
            tool_to_check = tools_by_name.get(tool_name)

            if not tool_to_check:
                invalid_call_messages.append(
                    ToolMessage(
                        content=f"Error: Tool '{tool_name}' not found.",
                        tool_call_id=tool_call['id']
                    )
                )
                continue

            params_info = self._extract_required_params(tool_to_check)
            required_params = params_info.get('required', [])
            provided_args = tool_call.get('args', {})
            missing_params = [p for p in required_params if not provided_args.get(p)]

            if missing_params:
                details = ", ".join(f"'{p}'" for p in missing_params)
                invalid_call_messages.append(
                    ToolMessage(
                        content=(
                            f"Tool call failed validation. Missing required parameters: {details}. "
                            "You must ask the user for this information before trying again."
                        ),
                        tool_call_id=tool_call['id']
                    )
                )
            else:
                valid_calls.append(tool_call)

        tool_results = []
        if valid_calls:
            temp_state = {"messages": [AIMessage(content="", tool_calls=valid_calls)]}
            tool_node = ToolNode(self.tools)
            tool_result_state = asyncio.run(tool_node.ainvoke(temp_state))
            tool_results = tool_result_state.get('messages', [])

        return {"messages": tool_results + invalid_call_messages}

    def _create_workflow(self) -> StateGraph:
        """
        Create the unified LangGraph workflow with optional thinking support.

        Returns:
            Compiled StateGraph workflow
        """

        def agent_node(state: ChatState) -> Dict[str, Any]:
            """Main agent node: invokes the model, extracts thinking, routes to tools or end."""
            include_fallback = not self.supports_native_thinking
            system_content = self._build_system_prompt(include_fallback_think=include_fallback)

            conditional_rules = self._build_conditional_rules()
            if conditional_rules:
                system_content += f"\n\n{conditional_rules}"

            is_post_tool_step = any(
                isinstance(msg, ToolMessage) for msg in state.get("messages", [])
            )

            messages: List[BaseMessage] = [SystemMessage(content=system_content)]

            # Inject accumulated reasoning so the model can build on it
            if state.get("thinking"):
                messages.append(SystemMessage(content=f"Prior reasoning:\n{state['thinking']}"))

            messages.extend(state.get("messages", []))

            # Meta-prompt reminder (HumanMessage for Gemini compatibility)
            if is_post_tool_step:
                messages.append(
                    HumanMessage(
                        content=(
                            "[SYSTEM REMINDER]\n"
                            "Tool results are shown above. DO NOT call any tools again. "
                            "Synthesize the results and provide your complete final answer to the user now."
                        )
                    )
                )
            else:
                messages.append(
                    HumanMessage(
                        content=(
                            "[SYSTEM REMINDER]\n"
                            "If all required tool parameters are present in the conversation, call the tool now. "
                            "Only ask the user if a parameter is truly absent."
                        )
                    )
                )

            try:
                # Get the current runnable config to propagate callbacks (e.g. token counting)
                try:
                    node_config = get_config()
                except Exception:
                    node_config = {}

                accumulated_thinking = state.get("thinking", "")
                thinking_emitted = False

                if self._is_deepseek_reasoner():
                    # DeepSeek R1: must use .invoke() for reasoning_content capture
                    response = self.model_with_tools.invoke(messages, node_config)
                else:
                    # All other models: use .stream() so on_thinking fires progressively
                    accumulated_chunk = None
                    for chunk in self.model_with_tools.stream(messages, node_config):
                        # Emit Gemini thinking chunks as they arrive
                        if isinstance(chunk.content, list):
                            for part in chunk.content:
                                if isinstance(part, dict) and part.get("type") == "thinking":
                                    t = part.get("thinking", "")
                                    if t:
                                        thinking_emitted = True
                                        if self.on_thinking:
                                            self.on_thinking(t)
                        accumulated_chunk = chunk if accumulated_chunk is None else (accumulated_chunk + chunk)

                    if accumulated_chunk is None:
                        response = AIMessage(content="")
                    else:
                        tc = getattr(accumulated_chunk, "tool_calls", []) or []
                        response = AIMessage(
                            content=accumulated_chunk.content,
                            tool_calls=tc,
                            additional_kwargs=getattr(accumulated_chunk, "additional_kwargs", {}),
                        )

                new_thinking = extract_thinking(response)

                # For DeepSeek R1 and fallback models (<think> tags), emit thinking now
                if new_thinking and self.on_thinking and not thinking_emitted:
                    self.on_thinking(new_thinking)

                if new_thinking:
                    accumulated_thinking = (accumulated_thinking + "\n" + new_thinking).strip()

                return {"messages": [response], "thinking": accumulated_thinking}

            except Exception as exc:
                self.logger.error(f"Error in agent_node: {exc}")
                self.logger.exception("Traceback completo:")
                fallback = AIMessage(
                    content="I apologize, but I encountered an error processing your request."
                )
                return {"messages": [fallback]}

        def should_continue(state: ChatState) -> str:
            """Determine if tools should be executed."""
            last_message = state["messages"][-1]
            if (isinstance(last_message, AIMessage) and
                    hasattr(last_message, 'tool_calls') and
                    last_message.tool_calls):
                return "tools"
            return "end"

        workflow = StateGraph(ChatState)
        workflow.add_node("agent", agent_node)

        if self.tools:
            workflow.add_node("tools", self.tool_validator_node)

        workflow.set_entry_point("agent")

        if self.tools:
            workflow.add_conditional_edges(
                "agent",
                should_continue,
                {
                    "tools": "tools",
                    "end": END
                }
            )
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)

        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        else:
            return workflow.compile()

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

    def _convert_message_to_base_message(self, messages: List[Message]) -> List[BaseMessage]:
        """
        Convert Message objects to BaseMessage objects.

        Args:
            messages: List of Message objects

        Returns:
            List of BaseMessage objects (HumanMessage or AIMessage)
        """
        base_messages = []
        for msg in messages:
            if msg.is_bot:
                base_messages.append(AIMessage(content=msg.content))
            else:
                base_messages.append(HumanMessage(content=msg.content))
        return base_messages

    # ===== PUBLIC API METHODS =====

    def get_response(
        self,
        user_input: str = None,
        messages: List[Message] = None,
        logs: List[str] = None,
        user_message: str = None,
    ) -> Dict[str, Any]:
        """
        Generate a response with logs and tool execution tracking.

        Accepts either `user_input` or `user_message` as the user's query
        (both are equivalent; `user_message` is provided for ThinkBot compatibility).

        Args:
            user_input (str): The user's message or query
            messages (List[Message]): Historical conversation messages
            logs (List[str]): Historical logs for context
            user_message (str): Alias for user_input (ThinkBot compatibility)

        Returns:
            dict: Structured response with content, thinking, logs, tools_executed, token_usage
        """
        actual_input = user_input if user_input is not None else user_message
        if messages is None:
            messages = []
        if logs is None:
            logs = []

        base_messages = self._convert_message_to_base_message(messages)

        limited_messages = self._limit_messages(base_messages)
        limited_logs = self._limit_logs(logs)
        self._current_logs = limited_logs

        tool_logger = _InternalToolLogger(
            on_start=self.on_tool_start,
            on_end=self.on_tool_end,
            on_error=self.on_tool_error
        )

        tool_logger.execution_logs.append(f"[USER] {actual_input}")

        initial_state: ChatState = {
            "messages": limited_messages + [HumanMessage(content=actual_input)],
            "logs": limited_logs,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "thinking": "",
        }

        config = {"callbacks": [tool_logger]}

        with get_openai_callback() as cb:
            result = asyncio.run(self.graph.ainvoke(initial_state, config=config))
            token_usage = self._extract_token_usage(result, cb)

        return self._finalize_response(result, tool_logger, limited_logs, token_usage)

    def stream_response(
        self,
        user_message: str,
        messages: List[Message],
        logs: List[str],
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream the response, yielding incremental chunks.

        Yields dicts with one of the following shapes:
            {"type": "thinking", "chunk": str}   — reasoning token (real-time for native models)
            {"type": "tool_call", "chunk": str}  — tool call being dispatched
            {"type": "content",  "chunk": str}   — response text token
            {"type": "done",     "result": dict} — full response (same structure as get_response)

        For native thinking models (DeepSeek R1, Gemini thinking): thinking and content
        tokens are streamed token-by-token via the "messages" event stream.
        For non-native models (GPT-4o, etc.): thinking (if any, via <think> fallback)
        and content appear as complete chunks when each node finishes.
        """
        base_messages = self._convert_message_to_base_message(messages)
        limited_messages = self._limit_messages(base_messages)
        limited_logs = self._limit_logs(logs)
        self._current_logs = limited_logs

        tool_logger = _InternalToolLogger(
            on_start=self.on_tool_start,
            on_end=self.on_tool_end,
            on_error=self.on_tool_error,
        )
        tool_logger.execution_logs.append(f"[USER] {user_message}")

        initial_messages = limited_messages + [HumanMessage(content=user_message)]
        initial_state: ChatState = {
            "messages": initial_messages,
            "logs": limited_logs,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "thinking": "",
        }
        config = {"callbacks": [tool_logger]}

        final_state: ChatState = initial_state
        previous_messages: List[BaseMessage] = list(initial_messages)
        previous_thinking = ""

        with get_openai_callback() as cb:
            for state in self.graph.stream(initial_state, config=config, stream_mode="values"):
                final_state = state

                # Emit new thinking (complete chunk per agent invocation)
                current_thinking = state.get("thinking", "")
                if current_thinking and current_thinking != previous_thinking:
                    diff = current_thinking[len(previous_thinking):]
                    if diff.strip():
                        yield {"type": "thinking", "chunk": diff}
                    previous_thinking = current_thinking

                # Emit new messages added by nodes
                current_messages = state.get("messages", [])
                if len(current_messages) > len(previous_messages):
                    new_msgs = current_messages[len(previous_messages):]
                    previous_messages = list(current_messages)
                    for msg in new_msgs:
                        if isinstance(msg, AIMessage):
                            if getattr(msg, "tool_calls", None):
                                yield {"type": "tool_call", "chunk": str(msg.tool_calls)}
                            elif msg.content:
                                clean = self._clean_content(msg.content)
                                if clean:
                                    yield {"type": "content", "chunk": clean}
                        elif isinstance(msg, ToolMessage):
                            tool_logger.execution_logs.append(f"[TOOL MESSAGE] {msg.content}")

            token_usage = self._extract_token_usage(final_state, cb)

        result = self._finalize_response(final_state, tool_logger, limited_logs, token_usage)
        yield {"type": "done", "result": result}

    def load_conversation_history(self, messages: List[Message]):
        """Load conversation history from Django model instances."""
        self.chat_history = self._convert_message_to_base_message(messages)

    def save_messages(self, user_message: str, bot_response: str):
        """Save messages to internal conversation history."""
        self.chat_history.append(HumanMessage(content=user_message))
        self.chat_history.append(AIMessage(content=bot_response))

    def clear_memory(self):
        """Clear conversation history."""
        self.chat_history.clear()

    def get_chat_history(self) -> List[BaseMessage]:
        """Retrieve a copy of the current conversation history."""
        return self.chat_history.copy()

    def set_chat_history(self, history: List[BaseMessage]):
        """Set the conversation history from a list of BaseMessage instances."""
        self.chat_history = history.copy()
