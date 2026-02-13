from langchain_community.tools import BaseTool
# Crear una clase que herede de BaseTool
from pydantic import BaseModel

class EmailTool(BaseTool, BaseModel):
    name: str = "EmailTool"
    description: str = "Esta herramienta envía correos electrónicos."

    def _run(self, to_email: str, subject: str, message: str) -> str:
        
        if True:
            return "Correo enviado con éxito."
        else:
            return "No se pudo enviar el correo."
        

class SaveContacto(BaseTool, BaseModel):
    name: str = "SaveContact"
    description: str = "Esta herramienta guarda los contactos"

    def _run(self, nombre: str, correo: str, telefono: str) -> str:
        
        if True:
            return "Contacto guardado"
        else:
            return "No se pudo enviar el correo."