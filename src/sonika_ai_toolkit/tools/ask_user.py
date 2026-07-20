"""AskUserQuestion — a *signal* tool that lets an agent ask the caller questions.

This tool performs no action. OrchestratorBot intercepts the call and surfaces
the structured questions to the UI (via a LangGraph interrupt) instead of
running ``_run``. The ``_run`` body is only a safe fallback for any path that
does execute it (returns a readable summary).
"""

from typing import List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel

from sonika_ai_toolkit.utilities.questions import (
    ASK_USER_TOOL_NAME,
    AskUserSchema,
    questions_summary,
    questions_to_payload,
)


class AskUserQuestionTool(BaseTool):
    """Ask the user one or more structured questions when information is missing."""

    name: str = ASK_USER_TOOL_NAME
    description: str = (
        "Ask the user one or more STRUCTURED questions when you need information "
        "you cannot obtain otherwise (a missing parameter, a choice between "
        "options, or a confirmation). Prefer this over asking in free text: give "
        "each question a stable `id`, a `type`, and — for choices — a list of "
        "`options` with value/label so the interface can render buttons. Call this "
        "ONLY when the information is genuinely absent from the conversation; never "
        "to ask permission to run another tool."
    )
    args_schema: Type[BaseModel] = AskUserSchema
    # Not a risky action — it is a request for input, never intercepted as approval.
    risk_level: int = 0

    def _run(self, questions: List, reason: Optional[str] = None, **kwargs) -> str:
        payload = questions_to_payload({"questions": questions, "reason": reason})
        return questions_summary(payload) or "Awaiting user input."

    async def _arun(self, questions: List, reason: Optional[str] = None, **kwargs) -> str:
        return self._run(questions, reason=reason, **kwargs)
