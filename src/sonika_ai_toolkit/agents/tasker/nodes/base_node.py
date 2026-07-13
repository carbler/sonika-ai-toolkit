"""Clase base abstracta para todos los nodos."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime


class BaseNode(ABC):
    """Clase base para nodos del workflow."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        if logger is None:
            self.logger.addHandler(logging.NullHandler())

    @abstractmethod
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Procesa el estado."""
        pass

    def format_timestamp(self) -> str:
        """Genera timestamp consistente para logs."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _text_content(content) -> str:
        """Extrae el texto de un AIMessage.content.

        Maneja tanto strings planos como el formato de lista que devuelven los
        modelos "thinking" (p.ej. Gemini 2.5): [{'type': 'thinking', ...}, 'texto'].
        Sin esto, ``content.strip()`` revienta con 'list' object has no attribute 'strip'.
        """
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and part.get("type") != "thinking":
                    text = part.get("text") or part.get("content") or ""
                    if text:
                        parts.append(str(text))
            return "\n".join(parts).strip()
        return str(content).strip() if content else ""

    def _add_log(self, state: Dict[str, Any], message: str, node_name: str = None) -> Dict[str, Any]:
        """
        Retorna la actualización de logs para el estado.
        Al usar Annotated[List, add] en state, devolvemos solo la lista con el nuevo log.
        """
        timestamp = self.format_timestamp()
        node = node_name or self.__class__.__name__.replace('Node', '').upper()
        log_entry = f"[{timestamp}] [{node}] {message}"

        # En la arquitectura nueva con Annotated, devolvemos el cambio, no el estado mutado
        return {"logs": [log_entry]}
