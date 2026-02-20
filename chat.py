import asyncio
import os
from dotenv import load_dotenv
from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot
from sonika_ai_toolkit.utilities.models import GeminiLanguageModel
from sonika_ai_toolkit.tools.registry import ToolRegistry
from sonika_ai_toolkit.tools.core.bash import RunBashTool
from sonika_ai_toolkit.tools.core.files import ReadFileTool, ListDirTool

load_dotenv()

async def main():
    # 1. Obtener la API Key del entorno
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\033[91m\n[Error] No se encontró la variable de entorno GOOGLE_API_KEY.\033[0m")
        print("Asegúrate de tener un archivo .env o haber ejecutado: export GOOGLE_API_KEY='tu_clave'\n")
        return

    # 2. Inicialización de modelo y herramientas
    # gemini-2.0-flash es el modelo más estable y balanceado
    model = GeminiLanguageModel(api_key=api_key, model_name="gemini-2.0-flash")
    tools = [RunBashTool(), ReadFileTool(), ListDirTool()]

    bot = OrchestratorBot(
        strong_model=model,
        fast_model=model,
        instructions="Eres un asistente útil que puede ejecutar comandos y leer archivos.",
        tools=tools,
        memory_path="./test_memory",
        # El callback on_thinking permite ver el razonamiento en gris
        on_thinking=lambda x: print(f"\033[90m{x}\033[0m", end="", flush=True)
    )

    predefined = [
        "Lista los archivos del proyecto y lee el setup.py para ver la versión.",
        "Ejecuta git status para ver qué archivos han cambiado.",
        "Busca todos los archivos .py en src/ y dime cuántos hay."
    ]

    print("\033[1m\n--- SONIKA INTERACTIVE TEST CONSOLE ---\033[0m")
    for i, p in enumerate(predefined):
        print(f"{i+1}. {p}")
    print("4. [Escribir prompt personalizado]")

    choice = input("\nSelecciona (1-4): ")
    
    if choice == "4":
        goal = input("Introduce tu prompt: ")
    elif choice in ["1", "2", "3"]:
        goal = predefined[int(choice)-1]
    else:
        print("Opción no válida.")
        return

    print(f"\n\033[1;34m[*] Objetivo:\033[0m {goal}\n")
    print("\033[1;32m[*] Pensando...\033[0m")
    print("\033[90m> [Razonamiento interno]:\033[0m")
    
    # arun() es la versión asíncrona recomendada para scripts async
    response = await bot.arun(goal)
    
    print("\n\n\033[1;34m--- RESULTADO FINAL ---\033[0m")
    print(response.content)
    
    if response.tools_executed:
        print("\n\033[1;33m--- HERRAMIENTAS UTILIZADAS ---\033[0m")
        for t in response.tools_executed:
            color = "\033[92m" if t['status'] == 'success' else "\033[91m"
            print(f" • {t['tool_name']}: {color}{t['status']}\033[0m")
    print("\n")

if __name__ == "__main__":
    asyncio.run(main())
