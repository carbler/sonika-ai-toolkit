"""Abstract base classes for all sonika-ai-toolkit agents."""

from abc import ABC, abstractmethod

from sonika_ai_toolkit.utilities.types import BotResponse


class IBot(ABC):
    """
    Root contract for all sonika-ai-toolkit agents.
    Defines the minimum common denominator: agents produce BotResponse.
    Concrete interfaces add the method signatures appropriate to each style.
    """

    @abstractmethod
    def abort(self) -> None:
        """
        Stop the in-flight streaming run at the next event boundary.

        Called from a different task/thread than the one consuming the stream
        (e.g. a UI/cancel handler). The stream yields a final "aborted" event
        and then stops; the underlying graph run is genuinely halted (streaming
        is pull-driven). A node already executing is not cancelled mid-run — the
        cut applies at the next boundary. The flag resets at the start of every
        run. Does not affect non-streaming paths (run/arun).
        """
        ...


class IConversationBot(IBot):
    """
    Contract for stateless conversational agents.
    Input: message + history.  Output: BotResponse.
    """

    @abstractmethod
    def get_response(
        self,
        user_input: str,
        messages: list,
        logs: list,
        **kwargs,
    ) -> BotResponse:
        """
        Process one conversation turn.

        Args:
            user_input: Text from the user.
            messages:   Conversation history.
            logs:       Prior execution logs.

        Returns:
            BotResponse with content, thinking, tools_executed, token_usage.
        """
        ...
