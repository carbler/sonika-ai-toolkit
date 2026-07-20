"""Structured user-questions contract for OrchestratorBot.

Lets an agent ask the caller *structured* questions (that a UI can render as
inputs, radio buttons, checkboxes…) instead of burying a question in free text.

The ``AskUserQuestion`` tool (``tools/ask_user.py``) carries the :class:`AskUserSchema`
schema.  OrchestratorBot intercepts the tool call and surfaces the questions to
the consumer via a native LangGraph interrupt (``type == "question_request"``) —
instead of executing any action.  This module is the single source of truth for
that shape.
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# Name the model must use to call the ask tool. Both bots key their interception
# off this constant, so it must match the tool's ``name``.
ASK_USER_TOOL_NAME = "ask_user"

QuestionType = Literal["text", "single_choice", "multi_choice", "boolean", "number"]


class QuestionOption(BaseModel):
    """A selectable choice for single_choice / multi_choice questions."""

    value: str = Field(description="Machine value returned when this option is chosen.")
    label: str = Field(description="Human-readable text shown to the user.")


class Question(BaseModel):
    """A single structured question the agent needs answered."""

    id: str = Field(description="Stable identifier; the answer is keyed by it.")
    text: str = Field(description="The question shown to the user.")
    type: QuestionType = Field(
        default="text",
        description="How the UI should render and collect the answer.",
    )
    options: Optional[List[QuestionOption]] = Field(
        default=None,
        description="Choices for single_choice / multi_choice questions.",
    )
    required: bool = Field(default=True, description="Whether an answer is mandatory.")


class AskUserSchema(BaseModel):
    """Arguments for the ``ask_user`` tool: one or more structured questions."""

    questions: List[Question] = Field(description="The questions to ask the user.")
    reason: Optional[str] = Field(
        default=None,
        description="Optional context for the UI on why the information is needed.",
    )


def questions_to_payload(args: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize raw tool args into a plain, JSON-serializable question payload."""
    try:
        return AskUserSchema.model_validate(args).model_dump()
    except Exception:
        # Best-effort passthrough if the model produced a slightly loose shape.
        return {
            "questions": args.get("questions", []) if isinstance(args, dict) else [],
            "reason": args.get("reason") if isinstance(args, dict) else None,
        }


def questions_summary(payload: Dict[str, Any]) -> str:
    """Human-readable fallback text (for ``content`` / non-UI consumers)."""

    def _get(obj, key, default=None):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    lines: List[str] = []
    reason = _get(payload, "reason")
    if reason:
        lines.append(str(reason))
    for i, q in enumerate(_get(payload, "questions", []) or [], 1):
        lines.append(f"{i}. {_get(q, 'text', '')}")
        for opt in _get(q, "options") or []:
            lines.append(f"   - {_get(opt, 'label', _get(opt, 'value', ''))}")
    return "\n".join(lines).strip()
