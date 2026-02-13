
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import sys
# Añadir la carpeta 'src' al PYTHONPATH
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


from sonika_ai_toolkit.tools.integrations import EmailTool,SaveContacto
from sonika_ai_toolkit.agents.react import ReactBot
from sonika_ai_toolkit.classifiers.text import  TextClassifier
from sonika_ai_toolkit.utilities.types import Message, ResponseModel
from sonika_ai_toolkit.utilities.models import OpenAILanguageModel, DeepSeekLanguageModel, GeminiLanguageModel, BedrockLanguageModel
from pydantic import BaseModel, Field
import json


env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
load_dotenv(env_path)

 # Ejemplo 1: Con callbacks
def on_tool_start(tool_name: str, input_data: str):
    print(f"🔧 Ejecutando tool: {tool_name}")
    print(f"   Input: {input_data}")
def on_tool_end(tool_name: str, output: str):
    print(f"✅ Tool completada: {tool_name}")
    print(f"   Output: {output[:100]}...")  # Primeros 100 chars

def on_tool_error(tool_name: str, error: str):
    print(f"❌ Tool falló: {tool_name}")
    print(f"   Error: {error}")

# Callbacks
def on_reasoning_start(user_msg: str):
    print(f"\n🧠 RAZONAMIENTO INICIADO")
    print(f"Mensaje: {user_msg}")

def on_reasoning_update(plan: dict):
   print("razonamiento")
   print(plan)

def on_logs_generated(logs):
    print(f"\n📋 LOGS GENERADOS:")
    for log in logs:
        print(f"   {log}")


def bot_bdi():
    # Obtener claves de API desde el archivo .env
    api_key = os.getenv("OPENAI_API_KEY")

    language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=1)
    embeddings = OpenAIEmbeddings(api_key=api_key)

    tools = [EmailTool(), SaveContacto()]
    bot = ReactBot(
        language_model, 
        instructions="Eres un agente",
        tools=tools,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
        on_tool_error=on_tool_error
    )

    user_message = 'Envia un email con la tool a erley@gmail.com con el asunto Hola y el mensaje Hola Erley. Y guarda a erley como contacto'

    # Historial usando tu clase Message
    messages = [Message(content="Mi nombre es Erley", is_bot=False)]
    
    # Preparar logs históricos (vacío si es la primera vez)
    logs = []

    # Obtener la respuesta del bot
    response = bot.get_response(user_message, messages, logs)
    
    print("Bot response:", json.dumps(response, indent=2, ensure_ascii=False))
    
    # Opcional: Si quieres seguir la conversación, actualiza el historial
    # history.append(Message(content=user_message, is_bot=False))
    # history.append(Message(content=response["content"], is_bot=True))
    # logs = response["logs"]

def bot_bdi_deepseek():
    # Obtener claves de API desde el archivo .env
    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        print("⚠️ DEEPSEEK_API_KEY no encontrada en .env")
        return

    language_model = DeepSeekLanguageModel(api_key, model_name='deepseek-chat', temperature=1)

    tools = [EmailTool(), SaveContacto()]
    bot = ReactBot(
        language_model,
        instructions="Eres un agente",
        tools=tools,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
        on_tool_error=on_tool_error
    )

    user_message = 'Envia un email con la tool a erley@gmail.com con el asunto Hola y el mensaje Hola Erley. Y guarda a erley como contacto'

    # Historial usando tu clase Message
    messages = [Message(content="Mi nombre es Erley", is_bot=False)]

    # Preparar logs históricos (vacío si es la primera vez)
    logs = []

    # Obtener la respuesta del bot
    try:
        response = bot.get_response(user_message, messages, logs)
        print("Bot response (DeepSeek):", json.dumps(response, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error calling DeepSeek: {e}")

def bot_bdi_gemini():
    # Obtener claves de API desde el archivo .env
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("⚠️ GOOGLE_API_KEY no encontrada en .env")
        return

    language_model = GeminiLanguageModel(api_key, model_name='gemini-3-flash-preview', temperature=1)

    tools = [EmailTool(), SaveContacto()]
    bot = ReactBot(
        language_model,
        instructions="Eres un agente",
        tools=tools,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
        on_tool_error=on_tool_error
    )

    user_message = 'Envia un email con la tool a erley@gmail.com con el asunto Hola y el mensaje Hola Erley. Y guarda a erley como contacto'

    # Historial usando tu clase Message
    messages = [Message(content="Mi nombre es Erley", is_bot=False)]

    # Preparar logs históricos (vacío si es la primera vez)
    logs = []

    # Obtener la respuesta del bot
    try:
        response = bot.get_response(user_message, messages, logs)
        print("Bot response (Gemini):", json.dumps(response, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error calling Gemini: {e}")

def bot_bdi_bedrock():
    # Obtener claves de API desde el archivo .env
    api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    aws_region = os.getenv("AWS_REGION", "us-east-1")

    if not api_key:
        print("⚠️ AWS_BEARER_TOKEN_BEDROCK not found in .env")
        return

    # Usando Amazon Nova Micro como default
    language_model = BedrockLanguageModel(
        api_key=api_key,
        region_name=aws_region,
        model_name='amazon.nova-micro-v1:0',
        temperature=1
    )

    tools = [EmailTool(), SaveContacto()]
    bot = ReactBot(
        language_model,
        instructions="Eres un agente",
        tools=tools,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
        on_tool_error=on_tool_error
    )

    user_message = 'Envia un email con la tool a erley@gmail.com con el asunto Hola y el mensaje Hola Erley. Y guarda a erley como contacto'

    # Historial usando tu clase Message
    messages = [Message(content="Mi nombre es Erley", is_bot=False)]

    # Preparar logs históricos (vacío si es la primera vez)
    logs = []

    # Obtener la respuesta del bot
    try:
        response = bot.get_response(user_message, messages, logs)
        print("Bot response (Bedrock):", json.dumps(response, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error calling Bedrock: {e}")


def bot_bdi_streaming():
    # Obtener claves de API desde el archivo .env
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_tavily = os.getenv("TAVILY_API_KEY")

    language_model = OpenAILanguageModel(api_key, model_name='gpt-4o-mini-2024-07-18', temperature=1)
    embeddings = OpenAIEmbeddings(api_key=api_key)
    
    # Configuración de herramientas y bots    
    tools =[]
    bot = ReactBot(language_model, embeddings, instructions="Only answers in english", tools=tools)

    user_message = 'Hola como me llamo?'

    bot.load_conversation_history([Message(content="Mi nombre es Erley", is_bot=False)])
    for chunk in bot.get_response_stream(user_message):
        print(chunk)

    

# Definir la clase 'Classification' con Pydantic para validar la estructura
class Classification(BaseModel):
    intention: str = Field()
    sentiment: str = Field(..., enum=["feliz", "neutral", "triste", "excitado"])
    aggressiveness: int = Field(
        ...,
        description="describes how aggressive the statement is, the higher the number the more aggressive",
        enum=[1, 2, 3, 4, 5],
    )
    language: str = Field(
        ..., enum=["español", "ingles", "frances", "aleman", "italiano"]
    )

def clasification():
    api_key = os.getenv("OPENAI_API_KEY")
    model = OpenAILanguageModel(api_key=api_key)
    classifier = TextClassifier(llm=model, validation_class=Classification)
    result = classifier.classify("how are you?")
    print(result)

bot_bdi()
#bot_mutinodo()
#bot_bdi_streaming()
#clasification()
#bot_bdi_deepseek()
#bot_bdi_gemini()
#bot_bdi_bedrock()
