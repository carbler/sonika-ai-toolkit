"""Logger Node - Tracks and logs events."""

from typing import Dict, Any, Optional, Callable
from sonika_ai_toolkit.agents.tasker.nodes.base_node import BaseNode

class LoggerNode(BaseNode):
    """Logs events and generates audit trail."""

    def __init__(
        self,
        on_logs_generated: Optional[Callable] = None,
        logger=None
    ):
        super().__init__(logger)
        self.on_logs_generated = on_logs_generated

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Consolida logs y notifica callbacks."""

        # En esta arquitectura con Annotated, 'logs' contiene TODOS los logs acumulados
        all_logs = state.get("logs", [])

        # Como este nodo corre al final, podemos simplemente pasar todos los logs
        # o intentar filtrar. Por simplicidad y robustez en esta nueva versión,
        # pasamos los logs que están en el estado.

        if self.on_logs_generated and all_logs:
            try:
                self.on_logs_generated(all_logs)
            except Exception as e:
                self.logger.warning(f"Callback failed: {e}")

        return {"logger_output": all_logs}
