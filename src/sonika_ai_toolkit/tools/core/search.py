"""SearchWebTool — stub that warns if no API key is configured."""

import logging
import os
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class _SearchWebInput(BaseModel):
    query: str = Field(description="The search query.")


class SearchWebTool(BaseTool):
    name: str = "search_web"
    description: str = "Search the web for information using a query string."
    args_schema: Type[BaseModel] = _SearchWebInput
    risk_hint: int = 0

    def _run(self, query: str) -> str:
        api_key = os.environ.get("SERPER_API_KEY") or os.environ.get("SERPAPI_API_KEY")
        if not api_key:
            logger.warning(
                "SearchWebTool: no SERPER_API_KEY or SERPAPI_API_KEY found. "
                "Search is a no-op stub."
            )
            return (
                f"[SearchWebTool stub] No API key configured. "
                f"Query was: '{query}'. "
                "Set SERPER_API_KEY or SERPAPI_API_KEY to enable real search."
            )

        # Try serper.dev if key present
        if os.environ.get("SERPER_API_KEY"):
            return self._serper_search(query, os.environ["SERPER_API_KEY"])

        return f"[SearchWebTool] Search for '{query}' — configure a search provider."

    def _serper_search(self, query: str, api_key: str) -> str:
        try:
            import requests

            resp = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                json={"q": query, "num": 5},
                timeout=15,
            )
            data = resp.json()
            results = data.get("organic", [])
            if not results:
                return "No results found."
            lines = []
            for r in results[:5]:
                lines.append(f"- {r.get('title', '')}: {r.get('snippet', '')} ({r.get('link', '')})")
            return "\n".join(lines)
        except Exception as e:
            return f"Search error: {e}"
