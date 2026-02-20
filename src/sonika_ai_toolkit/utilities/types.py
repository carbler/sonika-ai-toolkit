from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BotResponse(dict):
    """
    Unified response returned by all sonika-ai-toolkit agents.

    Behaves exactly like a ``dict`` (backward-compatible with any code that
    does ``result["content"]`` or ``result.get("thinking")``) but also exposes
    typed properties for IDE autocomplete and static analysis.

    Common keys (all agents)
    -------------------------
    content        — final text response (str)
    thinking       — reasoning/chain-of-thought text, or None
    logs           — execution log lines (List[str])
    tools_executed — [{tool_name, args, status, output}, …] (List[dict])
    token_usage    — {prompt_tokens, completion_tokens, total_tokens}

    OrchestratorBot extras
    ----------------------
    success    — True if at least one step succeeded
    plan       — full step list with statuses
    session_id — unique run identifier
    goal       — original goal string
    """

    # ── Common ─────────────────────────────────────────────────────────────

    @property
    def content(self) -> str:
        return self.get("content", "")

    @property
    def thinking(self) -> Optional[str]:
        return self.get("thinking")

    @property
    def logs(self) -> List[str]:
        return self.get("logs", [])

    @property
    def tools_executed(self) -> List[Dict[str, Any]]:
        return self.get("tools_executed", [])

    @property
    def token_usage(self) -> Dict[str, int]:
        return self.get(
            "token_usage",
            {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    # ── OrchestratorBot extras ──────────────────────────────────────────────

    @property
    def success(self) -> bool:
        return self.get("success", True)

    @property
    def plan(self) -> List[Dict[str, Any]]:
        return self.get("plan", [])

    @property
    def session_id(self) -> Optional[str]:
        return self.get("session_id")

    @property
    def goal(self) -> Optional[str]:
        return self.get("goal")

    # ── Repr ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        preview = (self.content[:80] + "…") if len(self.content) > 80 else self.content
        return (
            f"BotResponse(content={preview!r}, "
            f"tools={len(self.tools_executed)}, "
            f"thinking={'yes' if self.thinking else 'no'})"
        )


class ResponseModel():
    def __init__(self, user_tokens=None, bot_tokens=None,  response = None):
        self.user_tokens = user_tokens
        self.bot_tokens = bot_tokens
        self.response = response
    def __repr__(self):
        return (f"ResponseModel(user_tokens={self.user_tokens}, "
                f"bot_tokens={self.bot_tokens}, response={self.response})")
        
# Definir la interfaz para procesar archivos
class FileProcessorInterface(ABC):
    @abstractmethod
    def getText(self):
        pass

class ILanguageModel(ABC):
    @abstractmethod
    def predict(self, prompt: str) -> str:
        pass

class IEmbeddings(ABC):
    @abstractmethod
    def embed_documents(self, documents: List[str]):
        pass

    @abstractmethod
    def embed_query(self, query: str):
        pass

class Message:
    """
    Clase para representar un mensaje con un indicador de si es del bot y su contenido.
    """
    def __init__(self, is_bot: bool, content: str):
        self.is_bot = is_bot
        self.content = content


        

