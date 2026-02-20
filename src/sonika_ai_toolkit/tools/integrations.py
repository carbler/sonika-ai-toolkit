from typing import Type
from langchain_community.tools import BaseTool
from pydantic import BaseModel, Field


class _EmailInput(BaseModel):
    to_email: str = Field(description="Dirección de correo del destinatario.")
    subject: str = Field(description="Asunto del correo.")
    message: str = Field(description="Cuerpo del mensaje.")


class EmailTool(BaseTool):
    name: str = "EmailTool"
    description: str = "Envía un correo electrónico al destinatario indicado."
    args_schema: Type[BaseModel] = _EmailInput

    def _run(self, to_email: str, subject: str, message: str) -> str:
        return "Correo enviado con éxito."


class _SaveContactoInput(BaseModel):
    nombre: str = Field(description="Nombre completo del contacto.")
    correo: str = Field(description="Correo electrónico del contacto.")
    telefono: str = Field(description="Número de teléfono del contacto.")


class SaveContacto(BaseTool):
    name: str = "SaveContact"
    description: str = "Guarda un contacto con nombre, correo y teléfono."
    args_schema: Type[BaseModel] = _SaveContactoInput

    def _run(self, nombre: str, correo: str, telefono: str) -> str:
        return "Contacto guardado"