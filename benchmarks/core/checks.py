"""Declarative check factories.

Each factory returns a Check: a pure function that inspects a BotResponse and
returns a CheckOutcome. Checks rely only on fields every agent exposes —
`content` (str) and `tools_executed` (list of {tool_name, ...}) — so the same
scenario scores identically across ReactBot, TaskerBot and OrchestratorBot.
"""

import re

from benchmarks.core.scenario import CheckOutcome


def _tool_names(response) -> list:
    return [t.get("tool_name") for t in (response.get("tools_executed") or [])]


def _content(response) -> str:
    return response.get("content") or ""


def called(tool_name: str):
    """Pass if the agent invoked `tool_name` at least once."""
    def check(response):
        names = _tool_names(response)
        ok = tool_name in names
        return CheckOutcome(f"called:{tool_name}", ok,
                            "" if ok else f"tools called: {names}")
    return check


def not_called(tool_name: str):
    """Pass if the agent did NOT invoke `tool_name` (guards against over-acting)."""
    def check(response):
        names = _tool_names(response)
        ok = tool_name not in names
        return CheckOutcome(f"not_called:{tool_name}", ok,
                            "" if ok else f"unexpectedly called {tool_name}")
    return check


def no_tools():
    """Pass if the agent answered without calling any tool."""
    def check(response):
        names = _tool_names(response)
        ok = len(names) == 0
        return CheckOutcome("no_tools", ok, "" if ok else f"called: {names}")
    return check


def content_contains(substring: str, ignore_case: bool = True):
    """Pass if the response text contains `substring`."""
    def check(response):
        content = _content(response)
        hay, needle = (content.lower(), substring.lower()) if ignore_case else (content, substring)
        ok = needle in hay
        return CheckOutcome(f"contains:{substring!r}", ok,
                            "" if ok else "substring not found")
    return check


def content_absent(substring: str, ignore_case: bool = True):
    """Pass if the response text does NOT contain `substring`."""
    def check(response):
        content = _content(response)
        hay, needle = (content.lower(), substring.lower()) if ignore_case else (content, substring)
        ok = needle not in hay
        return CheckOutcome(f"absent:{substring!r}", ok,
                            "" if ok else "unexpected substring present")
    return check


def content_matches(pattern: str, flags: int = re.IGNORECASE):
    """Pass if the response text matches the regex `pattern`."""
    compiled = re.compile(pattern, flags)
    def check(response):
        ok = bool(compiled.search(_content(response)))
        return CheckOutcome(f"matches:{pattern!r}", ok,
                            "" if ok else "pattern not matched")
    return check
