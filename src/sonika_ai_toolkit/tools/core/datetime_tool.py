"""GetDateTimeTool — returns the current date and time."""

from datetime import datetime, timezone
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class _GetDateTimeInput(BaseModel):
    format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="strftime format string. Default: '%Y-%m-%d %H:%M:%S'.",
    )
    tz: str = Field(
        default="UTC",
        description="Timezone name (e.g. 'UTC', 'local'). Only 'UTC' and 'local' are supported.",
    )


class GetDateTimeTool(BaseTool):
    name: str = "get_datetime"
    description: str = (
        "Return the current date and time. "
        "Supports UTC or local time and custom strftime format strings."
    )
    args_schema: Type[BaseModel] = _GetDateTimeInput
    risk_hint: int = 0

    def _run(self, format: str = "%Y-%m-%d %H:%M:%S", tz: str = "UTC") -> str:
        try:
            tz_lower = tz.strip().lower()
            if tz_lower == "utc":
                now = datetime.now(timezone.utc)
            elif tz_lower == "local":
                now = datetime.now().astimezone()
            else:
                return (
                    f"Error: unsupported timezone '{tz}'. "
                    "Use 'UTC' or 'local'."
                )
            return now.strftime(format)
        except Exception as e:
            return f"Error: {e}"
