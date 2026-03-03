"""IOrchestratorBot — abstract contract for the OrchestratorBot.

Consumers (sonika CLI, test suites, etc.) should code against this interface
rather than the concrete OrchestratorBot so that the implementation can evolve
without breaking callers.
"""

from abc import abstractmethod
from typing import Any, AsyncGenerator, Optional, Tuple

from sonika_ai_toolkit.agents.base import IBot
from sonika_ai_toolkit.utilities.types import BotResponse


class IOrchestratorBot(IBot):
    """
    Contract for the OrchestratorBot: stateful, async-first, with interrupts
    and session management.
    """

    @abstractmethod
    async def astream_events(
        self,
        goal: Optional[str],
        mode: str = "ask",
        thread_id: Optional[str] = None,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        """
        Yield typed stream events.

        Yields:
            (stream_mode, payload) tuples where:
              - "messages"  → (AIMessageChunk, metadata)
              - "updates"   → {"agent": AgentUpdate} | {"tools": ToolsUpdate}
        """
        ...

    @abstractmethod
    async def arun(
        self,
        goal: str,
        context: str = "",
        thread_id: Optional[str] = None,
    ) -> BotResponse:
        """Async API: consume the stream and return a BotResponse."""
        ...

    @abstractmethod
    def run(
        self,
        goal: str,
        context: str = "",
        thread_id: Optional[str] = None,
    ) -> BotResponse:
        """Sync wrapper around arun()."""
        ...

    @abstractmethod
    def set_resume_command(self, resume_data: Any) -> None:
        """Store resume data so the next astream_events() call continues after an interrupt."""
        ...

    @abstractmethod
    async def a_prewarm(self) -> None:
        """Pre-warm the TCP/TLS connection to the LLM provider."""
        ...
