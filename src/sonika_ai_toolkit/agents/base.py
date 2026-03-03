"""Abstract base classes for all sonika-ai-toolkit agents."""

from abc import ABC, abstractmethod

from sonika_ai_toolkit.utilities.types import BotResponse


class IBot(ABC):
    """
    Root contract for all sonika-ai-toolkit agents.
    Defines the minimum common denominator: agents produce BotResponse.
    Concrete interfaces add the method signatures appropriate to each style.
    """


class IConversationBot(IBot):
    """
    Contract for stateless conversational agents (ReactBot, TaskerBot).
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
