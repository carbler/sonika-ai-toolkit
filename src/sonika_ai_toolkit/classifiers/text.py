from pydantic import BaseModel
from typing import Dict, Any, Type
from sonika_ai_toolkit.utilities.types import ILanguageModel

class ClassificationResponse(BaseModel):
    """Respuesta de clasificación con tokens utilizados"""
    input_tokens: int
    output_tokens: int
    result: Dict[str, Any]

class TextClassifier:
    def __init__(self, validation_class: Type[BaseModel], llm: ILanguageModel):
        self.llm = llm
        self.validation_class = validation_class
        # Guardamos ambas versiones del modelo
        self.original_model = self.llm.model  # Sin structured output
        self.structured_model = self.llm.model.with_structured_output(validation_class)

    def classify(self, text: str) -> ClassificationResponse:
        """
        Clasifica el texto según la clase de validación.
        
        Args:
            text: Texto a clasificar
        
        Returns:
            ClassificationResponse: Objeto con result, input_tokens y output_tokens
        """
        prompt = f"""
        Classify the following text based on the properties defined in the validation class.
        
        Text: {text}
        
        Only extract the properties mentioned in the validation class.
        """
        
        # Primero invocamos el modelo ORIGINAL para obtener metadata de tokens
        raw_response = self.original_model.invoke(prompt)
        
        # Extraer información de tokens del AIMessage original
        input_tokens = 0
        output_tokens = 0
        
        if hasattr(raw_response, 'response_metadata'):
            token_usage = raw_response.response_metadata.get('token_usage', {})
            input_tokens = token_usage.get('prompt_tokens', 0)
            output_tokens = token_usage.get('completion_tokens', 0)
        
        # Ahora invocamos con structured output para obtener el objeto parseado
        response = self.structured_model.invoke(prompt)
        
        # Validar que el response es de la clase correcta
        if isinstance(response, self.validation_class):
            # Crear el resultado dinámicamente basado en los atributos
            result_data = {
                field: getattr(response, field) 
                for field in self.validation_class.__fields__.keys()
            }
            
            return ClassificationResponse(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                result=result_data
            )
        else:
            raise ValueError(f"The response is not of type '{self.validation_class.__name__}'")