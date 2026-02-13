from langchain_community.document_loaders import PyPDFLoader
from sonika_ai_toolkit.utilities.types import FileProcessorInterface


class PDFProcessor(FileProcessorInterface):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def getText(self):
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        return documents