"""CallApiTool — HTTP GET/POST/PUT/DELETE with optional headers and body."""

import json
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool


class _CallApiInput(BaseModel):
    method: str = Field(description="HTTP method: GET, POST, PUT, or DELETE.")
    url: str = Field(description="Full URL to call.")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Optional HTTP headers.")
    body: Optional[Dict[str, Any]] = Field(default=None, description="Optional JSON body for POST/PUT.")


class CallApiTool(BaseTool):
    name: str = "call_api"
    description: str = (
        "Make an HTTP request (GET/POST/PUT/DELETE) to an API endpoint. "
        "Returns status code and response body."
    )
    args_schema: Type[BaseModel] = _CallApiInput
    risk_hint: int = 1

    def _run(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> str:
        try:
            import requests

            method = method.upper()
            kwargs: Dict[str, Any] = {"headers": headers or {}}
            if body is not None:
                kwargs["json"] = body

            response = requests.request(method, url, timeout=30, **kwargs)
            try:
                resp_body = response.json()
                resp_text = json.dumps(resp_body, ensure_ascii=False, indent=2)
            except Exception:
                resp_text = response.text

            return f"Status: {response.status_code}\n{resp_text}"
        except ImportError:
            return "Error: 'requests' package not installed. Run: pip install requests"
        except Exception as e:
            return f"Error calling {url}: {e}"
