from typing import List, Optional, Dict, Any, TypedDict, Annotated, Callable, Union, get_origin, get_args
import asyncio
import logging
import inspect
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, ToolMessage
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.tools import BaseTool
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.callbacks.manager import get_openai_callback

from sonika_ai_toolkit.utilities.types import ILanguageModel, Message


# ============= STATE DEFINITION =============

class ChatState(TypedDict):
    """
    Chat state for LangGraph workflow.
    
    Attributes:
        messages: List of conversation messages with automatic message handling
        logs: Historical logs for context
        token_usage: Accumulated token usage across all model invocations
    """
    messages: Annotated[List[BaseMessage], add_messages]
    logs: List[str]
    token_usage: Dict[str, int]


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
    Modern LangGraph-based conversational bot with MCP support.
    
    This implementation provides 100% API compatibility with existing ChatService
    while using modern LangGraph workflows and native tool calling internally.
    
    Features:
        - Native tool calling (no manual parsing)
        - MCP (Model Context Protocol) support
        - Complete token usage tracking across all model invocations
        - Thread-based conversation persistence
        - Tool execution callbacks for real-time monitoring
        - Backward compatibility with legacy APIs
    """

    def __init__(self, 
                 language_model: ILanguageModel,
                 instructions: str, 
                 tools: Optional[List[BaseTool]] = None,
                 mcp_servers: Optional[Dict[str, Any]] = None,
                 use_checkpointer: bool = False,
                 max_messages: int = 100,
                 max_logs: int = 20,
                 logger: Optional[logging.Logger] = None,
                 on_tool_start: Optional[Callable[[str, str], None]] = None,
                 on_tool_end: Optional[Callable[[str, str], None]] = None,
                 on_tool_error: Optional[Callable[[str, str], None]] = None):
        """
        Initialize the modern LangGraph bot with optional MCP support and callbacks.

        Args:
            language_model (ILanguageModel): The language model to use for generation
            instructions (str): System instructions that will be modernized automatically
            tools (List[BaseTool], optional): Traditional LangChain tools to bind to the model
            mcp_servers (Dict[str, Any], optional): MCP server configurations for dynamic tool loading
            use_checkpointer (bool): Enable automatic conversation persistence using LangGraph checkpoints
            max_messages (int): Maximum number of messages to keep in history
            max_logs (int): Maximum number of logs to keep in history
            logger (Optional[logging.Logger]): Logger instance for error tracking (silent by default if not provided)
            on_tool_start (Callable[[str, str], None], optional): Callback when a tool starts.
                Receives (tool_name: str, input_data: str)
            on_tool_end (Callable[[str, str], None], optional): Callback when a tool completes successfully.
                Receives (tool_name: str, output: str)
            on_tool_error (Callable[[str, str], None], optional): Callback when a tool fails.
                Receives (tool_name: str, error_message: str)
        """
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())
        
        self.language_model = language_model
        self.instructions = instructions
        
        self.max_messages = max_messages
        self.max_logs = max_logs
        
        self.chat_history: List[BaseMessage] = []
        
        self.tools = tools or []
        self.mcp_client = None
        
        self.on_tool_start = on_tool_start
        self.on_tool_end = on_tool_end
        self.on_tool_error = on_tool_error
        
        if mcp_servers:
            self._initialize_mcp(mcp_servers)
        
        self.checkpointer = MemorySaver() if use_checkpointer else None
        
        self.model_with_tools = self.language_model.model.bind_tools(self.tools) if self.tools else self.language_model.model
        
        self.graph = self._create_workflow()
        
        self.conversation = None
        self.agent_executor = None

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
        Extrae los parámetros requeridos del schema de una tool.
        
        Soporta:
        - LangChain BaseTool con args_schema (Pydantic v1/v2)
        - LangChain BaseTool con _run y type hints (Optional detection)
        - MCP Tools con inputSchema (JSON Schema)
        - HTTPTool con args_schema dinámico
        
        Returns:
            Dict con 'required' (lista de campos requeridos) y 'all' (todos los campos)
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
            rules.append("""
## CORPORATE RULE — MANDATORY USE OF `search_knowledge_documents`
If the user's query might be answered by internal documents:
- ALWAYS call `search_knowledge_documents` FIRST before responding
- Use the user's message as the query
- Never invent information if it might exist in documents
""")
        
        if 'accept_policies' in tool_names:
            rules.append("""
## POLICY ACCEPTANCE HANDLING
- On the FIRST user message of the conversation, you MUST ask if they accept the privacy policies and terms of use.
- Do NOT call the `accept_policies` tool automatically.
- Wait for the user's explicit confirmation (e.g. "yes", "sí", "acepto", "ok").
- As soon as the user confirms, you MUST immediately call the `accept_policies` tool.
- Pass the user's confirmation inside the `user_message` parameter.
- If the user does NOT confirm, do not call the tool and continue waiting for acceptance.
- This rule is applied only once: after successfully executing `accept_policies`, NEVER ask for acceptance again.
""")
        
        if 'create_or_update_contact' in tool_names:
            rules.append("""
## AUTOMATIC CONTACT UPDATE
If the user provides contact information (name, email, phone):
- ALWAYS call `create_or_update_contact` immediately
- Include any information provided (don't wait for all fields)
- Execute this BEFORE any other action
""")
        
        return "\n".join(rules) if rules else ""

    def tool_validator_node(self, state: ChatState) -> ChatState:
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
            missing_params = []

            for req_param in required_params:
                if not provided_args.get(req_param):
                    missing_params.append(req_param)

            if missing_params:
                details = ", ".join(f"'{p}'" for p in missing_params)
                feedback_content = f"Tool call failed validation. Missing required parameters: {details}. You must ask the user for this information before trying again."
                invalid_call_messages.append(
                    ToolMessage(
                        content=feedback_content,
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

        all_feedback_messages = tool_results + invalid_call_messages
        
        return {"messages": all_feedback_messages}

    def _create_workflow(self) -> StateGraph:
        """
        Create standard LangGraph workflow.
        
        Returns:
            Compiled StateGraph workflow
        """
        
        def agent_node(state: ChatState) -> ChatState:
            """Main agent node for standard workflow."""
            last_user_message = None
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    last_user_message = msg.content
                    break
            
            if not last_user_message:
                return state
            
            system_content = self.instructions
            
            conditional_rules = self._build_conditional_rules()
            if conditional_rules:
                system_content += f"\n\n{conditional_rules}"
            
            if state.get("logs"):
                logs_context = "\n".join(state["logs"][-self.max_logs:])
                system_content += f"\n\nRecent logs:\n{logs_context}"
            
            messages = [{"role": "system", "content": system_content}]
            
            # --- MODIFICACIÓN 1: Flag para detectar estado ---
            is_post_tool_step = False 
            
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    if msg.tool_calls:
                        serialized_msg = {"role": "assistant", "content": msg.content or ""}
                        serialized_msg['tool_calls'] = msg.tool_calls
                        messages.append(serialized_msg)
                    else:
                        messages.append({"role": "assistant", "content": msg.content or ""})
                elif isinstance(msg, ToolMessage):
                    messages.append({
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": msg.tool_call_id,
                    })
                    # --- MODIFICACIÓN 2: Detectamos ejecución de tool ---
                    is_post_tool_step = True
            
            # --- MODIFICACIÓN 3: Inyección de Meta-Prompt (Sandwich) ---
            # Esto es lo que soluciona los tests sin cambiar tu lógica base.
            if is_post_tool_step:
                # El modelo ya tiene datos. Le recordamos mirar las reglas de formato/salida.
                meta_prompt = """
[SYSTEM REMINDER]
You have received data from a tool. Before generating your final response, REVIEW your initial System Instructions.
Ensure strict adherence to:
1. Defined Tone & Identity rules.
2. Output Formatting requirements (JSON, Footers, Dates).
3. Safety protocols based on the tool's output.
"""
                # Usamos "user" en lugar de "system" para evitar errores con modelos estrictos (Gemini)
                messages.append({"role": "user", "content": meta_prompt})
            else:
                # El modelo va a planificar. Le recordamos mirar las reglas de overrides/seguridad.
                meta_prompt = """
[SYSTEM REMINDER]
Before calling a tool or responding, REVIEW your System Instructions for:
1. Explicit Override Keywords (that forbid tool use).
2. Missing Information requirements (ask before acting).
3. User Sentiment protocols.
"""
                # Usamos "user" en lugar de "system" para evitar errores con modelos estrictos (Gemini)
                messages.append({"role": "user", "content": meta_prompt})

            try:
                response = self.model_with_tools.invoke(messages)
                return {"messages": [response]}
            except Exception as e:
                self.logger.error(f"Error en agent_node: {e}")
                self.logger.exception("Traceback completo:")
                import traceback
                traceback.print_exc()
                fallback_response = AIMessage(content="I apologize, but I encountered an error processing your request.")
                return {"messages": [fallback_response]}

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
        user_input: str,
        messages: List[Message],
        logs: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a response with logs and tool execution tracking.
        
        This method tracks ALL token usage across the entire workflow including:
        - Tool executions
        - Final response formatting
        
        Args:
            user_input (str): The user's message or query
            messages (List[Message]): Historical conversation messages (Message class)
            logs (List[str]): Historical logs for context
            
        Returns:
            dict: Structured response with content, logs, tools_executed, and token_usage
        """
        base_messages = self._convert_message_to_base_message(messages)
        
        limited_messages = self._limit_messages(base_messages)
        limited_logs = self._limit_logs(logs)
        
        tool_logger = _InternalToolLogger(
            on_start=self.on_tool_start,
            on_end=self.on_tool_end,
            on_error=self.on_tool_error
        )
        
        tool_logger.execution_logs.append(f"[USER] {user_input}")
        
        initial_state = {
            "messages": limited_messages + [HumanMessage(content=user_input)],
            "logs": limited_logs,
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }
        
        config = {"callbacks": [tool_logger]}
        
        with get_openai_callback() as cb:
            result = asyncio.run(self.graph.ainvoke(initial_state, config=config))
            
            result["token_usage"] = {
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_tokens": cb.total_tokens
            }
        
        final_response = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content
                break
        
        if final_response:
            tool_logger.execution_logs.append(f"[BOT] {final_response}")
        
        new_logs = limited_logs + tool_logger.execution_logs
        
        return {
            "content": final_response,
            "logs": new_logs,
            "tools_executed": tool_logger.tool_executions,
            "token_usage": result["token_usage"]
        }

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