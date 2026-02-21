from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseInterface(ABC):
    """
    Contrato base para cualquier interfaz que interactúe con el Motor de Sonika.
    Asegura el desacoplamiento entre la lógica de LangGraph y la presentación (CLI, Web, GUI).
    """

    @abstractmethod
    def on_thought(self, chunk: str) -> None:
        """Render a chunk of thinking/reasoning."""
        pass

    @abstractmethod
    def on_tool_start(self, tool_name: str, params: Dict[str, Any]) -> None:
        """Render the start of a tool execution."""
        pass

    @abstractmethod
    def on_tool_end(self, tool_name: str, result: str) -> None:
        """Render the successful completion of a tool."""
        pass

    @abstractmethod
    def on_error(self, tool_name: str, error: str) -> None:
        """Render an error that occurred during tool execution."""
        pass

    @abstractmethod
    def on_interrupt(self, data: Dict[str, Any]) -> bool:
        """
        Handle a LangGraph interrupt (e.g. permission required).
        Should prompt the user and return True/False or a structured Command response.
        """
        pass
        
    @abstractmethod
    def on_result(self, result: str) -> None:
        """Render the final result/report from the LLM."""
        pass
