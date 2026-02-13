import sys
import importlib
import os
from dotenv import load_dotenv

# Asegurar que podemos importar desde el directorio actual
sys.path.append(os.path.dirname(__file__))

from stress_test_runner import UltimateStressTestRunner, AVAILABLE_BOTS

load_dotenv()

# ==========================================
# CONFIGURACIÓN DEL BATCH DE PRUEBAS
# ==========================================
# Define aquí las combinaciones de Bot y Modelo que deseas probar.
# bot_id: Corresponde a las claves en AVAILABLE_BOTS (1-6)
# model: Nombre del modelo (ej: gpt-4o-mini, deepseek-chat)
# provider: Proveedor del modelo (openai, deepseek). Default: openai

TEST_CONFIGS = [
    {"bot_id": "5", "model": "gpt-4o-mini", "provider": "openai"}, # TaskerBot
    {"bot_id": "6", "model": "gpt-4o-mini", "provider": "openai"}, # ReactBot
    {"bot_id": "6", "model": "deepseek-chat", "provider": "deepseek"}, # ReactBot with DeepSeek
    {"bot_id": "6", "model": "gemini-3-flash-preview", "provider": "gemini"}, # ReactBot with Gemini
    {"bot_id": "6", "model": "amazon.nova-micro-v1:0", "provider": "bedrock"}, # ReactBot with Bedrock
]

def resolve_bot_class(bot_id):
    """Resuelve la clase del bot a partir del ID."""
    if bot_id not in AVAILABLE_BOTS:
        print(f"⚠️  Advertencia: Bot ID '{bot_id}' no encontrado. Saltando.")
        return None, None

    bot_info = AVAILABLE_BOTS[bot_id]
    try:
        module = importlib.import_module(bot_info["module"])
        bot_class = getattr(module, bot_info["class"])
        return bot_class, bot_info["name"]
    except ImportError as e:
        print(f"❌ Error importando {bot_info['name']}: {e}")
        return None, None
    except AttributeError:
        print(f"❌ Clase {bot_info['class']} no encontrada en {bot_info['module']}")
        return None, None

def run_batch():
    print(f"🚀 INICIANDO BATCH STRESS TEST ({len(TEST_CONFIGS)} configuraciones)...")
    print("="*60)

    successful_runs = 0
    failed_runs = 0

    for i, config in enumerate(TEST_CONFIGS, 1):
        bot_id = config.get("bot_id")
        model_name = config.get("model", "gpt-4o-mini")
        provider = config.get("provider", "openai")

        bot_class, bot_name = resolve_bot_class(bot_id)

        if not bot_class:
            failed_runs += 1
            continue

        print(f"\n▶️  EJECUTANDO RUN {i}/{len(TEST_CONFIGS)}")
        print(f"   🤖 Bot: {bot_name}")
        print(f"   🧠 Modelo: {model_name}")
        print(f"   🌐 Proveedor: {provider}")
        print("-" * 30)

        try:
            # Instanciar y ejecutar el runner
            runner = UltimateStressTestRunner(bot_class, bot_name, model_name, provider=provider)
            runner.run_all_tests()
            successful_runs += 1
            print(f"✅ Run {i} completado exitosamente.")

        except Exception as e:
            print(f"❌ Run {i} falló con excepción: {e}")
            failed_runs += 1

        print("="*60)

    print("\n🏁 RESUMEN DEL BATCH")
    print(f"   Total Ejecutados: {len(TEST_CONFIGS)}")
    print(f"   Exitosos: {successful_runs}")
    print(f"   Fallidos: {failed_runs}")
    print("="*60)

if __name__ == "__main__":
    # Verificar keys según proveedores usados
    providers = {c.get("provider", "openai") for c in TEST_CONFIGS}

    # OpenAI check (provider or embeddings)
    if "openai" in providers or not os.getenv("OPENAI_API_KEY"):
         if not os.getenv("OPENAI_API_KEY"):
            print("❌ Error: OPENAI_API_KEY requerida (para provider 'openai' o embeddings).")
            sys.exit(1)

    # DeepSeek check
    if "deepseek" in providers:
        if not os.getenv("DEEPSEEK_API_KEY"):
            print("❌ Error: DEEPSEEK_API_KEY requerida para configs con provider 'deepseek'.")
            sys.exit(1)

    # Gemini check
    if "gemini" in providers:
        if not os.getenv("GOOGLE_API_KEY"):
            print("❌ Error: GOOGLE_API_KEY requerida para configs con provider 'gemini'.")
            sys.exit(1)

    # Bedrock check
    if "bedrock" in providers:
        if not os.getenv("AWS_BEARER_TOKEN_BEDROCK"):
            print("❌ Error: AWS_BEARER_TOKEN_BEDROCK requerida para configs con provider 'bedrock'.")
            sys.exit(1)

    run_batch()
