"""Shared graph/message helpers for the orchestrator's LangGraph nodes.

These previously lived in ``agents/react.py``; they moved here when ReactBot was
removed, since OrchestratorBot is now their only consumer. All are pure helpers:
vision detection, user-message building, thinking extraction, per-node tracing,
run-id generation and graph topology.
"""

import inspect
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def _has_image_content(messages) -> bool:
    """True if any message carries multimodal image content (a vision turn).

    Vision turns send the user message as a list of parts, e.g.
    ``[{"type": "text", ...}, {"type": "image_url", "image_url": {...}}]``.
    """
    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    return True
    return False


def _build_user_message(text: str, images: Optional[List[str]] = None) -> HumanMessage:
    """Build the user turn — plain text, or multimodal (text + images) for vision.

    ``images`` is a list of image URLs / data-URLs. When present the message
    content becomes the list format the chat models expect for vision so the
    agent can actually "see" the image.
    """
    if not images:
        return HumanMessage(content=text)
    content: List[Dict[str, Any]] = []
    if text:
        content.append({"type": "text", "text": text})
    for url in images:
        content.append({"type": "image_url", "image_url": {"url": url}})
    return HumanMessage(content=content)


def _build_history_messages(history) -> List[BaseMessage]:
    """Convert prior conversation turns into LangChain messages.

    Lets the orchestrator receive an externally-managed conversation (the caller
    owns the history, e.g. from a DB) and prepend it before the current goal, so
    the model has the full context. Each item may be:

    - a LangChain ``BaseMessage`` — passed through unchanged;
    - a sonika ``Message`` (has ``is_bot`` + ``content``) — ``is_bot`` maps to an
      ``AIMessage``, otherwise a ``HumanMessage``;
    - a ``dict`` with ``role`` (``user``/``human``, ``assistant``/``ai``/``bot``,
      ``system``) and ``content``.

    Empty / unparseable items are skipped.
    """
    if not history:
        return []
    built: List[BaseMessage] = []
    for m in history:
        if isinstance(m, BaseMessage):
            built.append(m)
            continue
        # sonika Message dataclass: is_bot flag + content
        if not isinstance(m, dict) and hasattr(m, "is_bot"):
            content = getattr(m, "content", "") or ""
            built.append(AIMessage(content=content) if m.is_bot else HumanMessage(content=content))
            continue
        if isinstance(m, dict):
            content = m.get("content", "") or ""
            role = str(m.get("role", "") or "").lower()
            if role in ("assistant", "ai", "bot"):
                built.append(AIMessage(content=content))
            elif role == "system":
                built.append(SystemMessage(content=content))
            else:
                built.append(HumanMessage(content=content))
    return built


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


_DETAIL_TRUNC = 500  # chars kept for outputs inside node detail payloads


def _node_detail(result: Dict[str, Any]) -> Dict[str, Any]:
    """Compact params/output summary of a node execution, from its state delta.

    Keys (present only when applicable):
      tool_calls     — tools the node's message requests: [{name, args}, …]
      tools_executed — tools actually run: [{tool_name, args, status, output}, …]
      output         — text emitted by the node (truncated)
      plan / step_events / questions — planning + ask_user metadata
    """
    detail: Dict[str, Any] = {}
    tool_calls: List[Dict[str, Any]] = []
    texts: List[str] = []
    for msg in result.get("messages") or []:
        for tc in getattr(msg, "tool_calls", None) or []:
            tool_calls.append({"name": tc.get("name"), "args": tc.get("args")})
        content = getattr(msg, "content", None)
        text = _get_text_content(content) if content else ""
        if text:
            texts.append(text)
    if tool_calls:
        detail["tool_calls"] = tool_calls
    if result.get("tools_executed"):
        detail["tools_executed"] = [
            {
                "tool_name": t.get("tool_name"),
                "args": t.get("args"),
                "status": t.get("status"),
                "output": str(t.get("output", ""))[:_DETAIL_TRUNC],
            }
            for t in result["tools_executed"]
        ]
    if texts:
        detail["output"] = "\n".join(texts)[:_DETAIL_TRUNC]
    if result.get("plan"):
        detail["plan"] = result["plan"]
    if result.get("step_events"):
        detail["step_events"] = result["step_events"]
    pending = result.get("pending_questions")
    if pending and pending.get("questions"):
        detail["questions"] = pending["questions"]
    return detail


def _wrap_node_traced(name: str, fn: Callable) -> Callable:
    """Wrap a graph node so every execution appends a node-trace entry.

    The entry ``{"node", "run_id", "ts", "detail"}`` — ``detail`` carries the
    node's params/output summary (see ``_node_detail``) — rides in the node's
    own state delta under ``node_trace`` (an ``operator.add`` channel), so it
    surfaces both in the ``updates`` stream and in the final state. Works with
    sync and async node callables. If the node returns a full state (instead
    of a delta) any pre-existing ``node_trace`` key is replaced, never
    re-added.
    """
    def _annotate(state, result):
        if isinstance(result, dict):
            entry = {
                "node": name,
                "run_id": state.get("run_id", ""),
                "ts": time.time(),
                "detail": _node_detail(result),
            }
            result = {**result, "node_trace": [entry]}
        return result

    # Preserve sync/async nature: LangGraph runs sync nodes in a thread
    # executor (some rely on that, e.g. to call asyncio.run internally).
    if inspect.iscoroutinefunction(fn):
        async def traced(state):
            return _annotate(state, await fn(state))
    else:
        def traced(state):
            return _annotate(state, fn(state))

    traced.__name__ = f"traced_{name}"
    return traced


def _graph_topology(compiled_graph) -> Dict[str, Any]:
    """Return the static node/edge layout of a compiled LangGraph.

    Shape: ``{"entry": str, "nodes": [str], "edges": [{source, target,
    conditional}]}``. Nodes include the virtual ``__start__``/``__end__``.
    """
    g = compiled_graph.get_graph()
    edges = [
        {"source": e.source, "target": e.target, "conditional": bool(e.conditional)}
        for e in g.edges
    ]
    entry = next((e["target"] for e in edges if e["source"] == "__start__"), "")
    return {"entry": entry, "nodes": list(g.nodes.keys()), "edges": edges}


def _new_run_id() -> str:
    """Unique id for one run (process) of a bot — must never repeat.

    UTC timestamp (microsecond precision) + full UUID4: the date prefix makes
    ids sortable and human-readable, the untruncated UUID4 (122 bits) makes a
    collision impossible in practice even within the same microsecond.
    Example: ``20260717T153045123456-1f2a…``.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    return f"{stamp}-{uuid.uuid4().hex}"
