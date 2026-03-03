"""FetchWebPageTool — fetches a URL and returns clean text."""

import re
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

_MAX_CHARS = 8000


class _FetchWebPageInput(BaseModel):
    url: str = Field(description="Full URL to fetch (http or https).")


class FetchWebPageTool(BaseTool):
    name: str = "fetch_web_page"
    description: str = (
        "Fetch a web page URL and return its text content (HTML stripped). "
        "Use this to read articles, documentation, or any public web page."
    )
    args_schema: Type[BaseModel] = _FetchWebPageInput
    risk_hint: int = 0

    def _run(self, url: str) -> str:
        try:
            import requests
        except ImportError:
            return "Error: 'requests' package not installed. Run: pip install requests"

        try:
            resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            text = self._extract_text(resp.text)
            if len(text) > _MAX_CHARS:
                text = text[:_MAX_CHARS] + f"\n\n[Truncated — showing first {_MAX_CHARS} chars]"
            return text or "(empty page)"
        except Exception as e:
            return f"Error fetching {url}: {e}"

    def _extract_text(self, html: str) -> str:
        # Remove scripts and styles
        html = re.sub(r"<(script|style)[^>]*>.*?</(script|style)>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # Remove all tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Decode common HTML entities
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&nbsp;", " ").replace("&quot;", '"').replace("&#39;", "'")
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text
