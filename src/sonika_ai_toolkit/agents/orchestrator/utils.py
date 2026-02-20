"""Shared utilities for orchestrator nodes."""

from typing import Callable, List, Optional
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool


def get_text(content) -> str:
    """
    Extract plain text from model content.
    Handles both str and Gemini's list format:
        [{"type": "thinking", "thinking": "..."}, "actual text"]
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


async def ainvoke_with_thinking(
    model,
    prompt: str,
    on_thinking: Optional[Callable[[str], None]] = None,
) -> AIMessage:
    """
    Stream a model call token-by-token.

    - Gemini 2.5: fires on_thinking for each thinking chunk as it arrives
    - DeepSeek R1: fires on_thinking once after full response (reasoning_content in additional_kwargs)
    - GPT-4o-mini / others: no native thinking → thinking stays None

    Always returns an AIMessage where `.content` is a CLEAN STRING (thinking parts stripped).
    Thinking text is available via `response.additional_kwargs.get("_thinking")`.
    """
    try:
        accumulated = None
        thinking_chunks: List[str] = []

        async for chunk in model.astream(prompt):
            # Gemini: thinking parts arrive as list items during stream
            if isinstance(chunk.content, list):
                for part in chunk.content:
                    if isinstance(part, dict) and part.get("type") == "thinking":
                        t = part.get("thinking", "")
                        if t:
                            thinking_chunks.append(t)
                            if on_thinking:
                                on_thinking(t)
            accumulated = chunk if accumulated is None else (accumulated + chunk)

        if accumulated is None:
            return AIMessage(content="")

        # Extract clean text content (strip thinking parts)
        clean = get_text(accumulated.content)

        # Build additional_kwargs with thinking for state accumulation
        extra = dict(getattr(accumulated, "additional_kwargs", None) or {})

        if thinking_chunks:
            extra["_thinking"] = "\n\n".join(thinking_chunks)
        else:
            # DeepSeek R1: reasoning_content arrives after full response
            deepseek = extra.get("reasoning_content")
            if deepseek:
                extra["_thinking"] = deepseek
                if on_thinking:
                    on_thinking(deepseek)

        return AIMessage(content=clean, additional_kwargs=extra)

    except Exception:
        # Fallback: blocking ainvoke (for models that don't support streaming)
        response = await model.ainvoke(prompt)
        clean = get_text(response.content if hasattr(response, "content") else "")
        extra = dict(getattr(response, "additional_kwargs", None) or {})

        # DeepSeek R1 fallback
        deepseek = extra.get("reasoning_content")
        if deepseek:
            extra["_thinking"] = deepseek
            if on_thinking:
                on_thinking(deepseek)

        # Gemini fallback: extract thinking from list content
        if not extra.get("_thinking") and isinstance(
            getattr(response, "content", None), list
        ):
            t_parts = [
                p.get("thinking", "")
                for p in response.content
                if isinstance(p, dict) and p.get("type") == "thinking"
            ]
            t_text = "\n\n".join(filter(None, t_parts))
            if t_text:
                extra["_thinking"] = t_text
                if on_thinking:
                    on_thinking(t_text)

        return AIMessage(content=clean, additional_kwargs=extra)


def find_missing_params(tool: BaseTool, params: dict) -> List[str]:
    """
    Return required param names that are absent or empty in params.
    Uses args_schema (Pydantic v2/v1) or _run signature as fallback.
    """
    import inspect

    missing: List[str] = []
    schema = getattr(tool, "args_schema", None)

    if schema is not None:
        # Pydantic v2
        if hasattr(schema, "model_fields"):
            for fname, field in schema.model_fields.items():
                if field.is_required():
                    val = params.get(fname)
                    if val is None or val == "":
                        missing.append(fname)
            return missing
        # Pydantic v1
        if hasattr(schema, "__fields__"):
            for fname, field in schema.__fields__.items():
                if field.required:
                    val = params.get(fname)
                    if val is None or val == "":
                        missing.append(fname)
            return missing

    # Fallback: inspect _run signature
    if hasattr(tool, "_run"):
        sig = inspect.signature(tool._run)
        for pname, param in sig.parameters.items():
            if pname in ("self", "args", "kwargs", "run_manager"):
                continue
            if param.default is inspect.Parameter.empty:
                val = params.get(pname)
                if val is None or val == "":
                    missing.append(pname)

    return missing
