"""set_plan / update_step — *signal* tools for structured plan tracking.

These tools perform no real action. When ``enable_planning=True`` the
OrchestratorBot tracks their calls in ``agent_node`` (updating the ``plan`` /
``step_events`` graph state) and ``tools_node`` answers them with a no-op
acknowledgment ToolMessage — they are excluded from ``tools_executed``. The
``_run`` bodies produce that acknowledgment text.
"""

from typing import List, Literal, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

SET_PLAN_TOOL_NAME = "set_plan"
UPDATE_STEP_TOOL_NAME = "update_step"
PLAN_SIGNAL_TOOL_NAMES = frozenset({SET_PLAN_TOOL_NAME, UPDATE_STEP_TOOL_NAME})

StepStatus = Literal["running", "done", "skipped", "error"]


class SetPlanSchema(BaseModel):
    steps: List[str] = Field(
        ...,
        description="Ordered list of short step descriptions for the execution plan.",
    )


class UpdateStepSchema(BaseModel):
    step: int = Field(..., description="1-based index of the plan step to update.")
    status: StepStatus = Field(
        ..., description="New status for the step: running, done, skipped or error."
    )


class SetPlanTool(BaseTool):
    """Register the execution plan as an ordered list of steps."""

    name: str = SET_PLAN_TOOL_NAME
    description: str = (
        "Register your execution plan BEFORE starting a multi-step task. "
        "Pass an ordered list of short step descriptions. Call this exactly "
        "once per task; use update_step to report progress afterwards."
    )
    args_schema: Type[BaseModel] = SetPlanSchema
    # Signal tool — never a risky action, never triggers approval interrupts.
    risk_level: int = 0

    def _run(self, steps: List[str], **kwargs) -> str:
        return f"Plan registered with {len(steps)} steps."

    async def _arun(self, steps: List[str], **kwargs) -> str:
        return self._run(steps, **kwargs)


class UpdateStepTool(BaseTool):
    """Report progress on a previously registered plan step."""

    name: str = UPDATE_STEP_TOOL_NAME
    description: str = (
        "Report progress on a step of the plan you registered with set_plan: "
        "mark it as running when you start it and done/skipped/error when it "
        "finishes. Call it alongside the real tool calls of that step."
    )
    args_schema: Type[BaseModel] = UpdateStepSchema
    risk_level: int = 0

    def _run(self, step: int, status: str, **kwargs) -> str:
        return f"Step {step} marked as {status}."

    async def _arun(self, step: int, status: str, **kwargs) -> str:
        return self._run(step, status, **kwargs)
