from abc import ABC, abstractmethod
from typing import List


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


        

