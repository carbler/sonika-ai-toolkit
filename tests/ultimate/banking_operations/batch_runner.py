"""
Batch stress-test runner.

Edit TEST_CONFIGS below to choose which Bot × Model combinations to run.
Then execute:

    python tests/ultimate/banking_operations/batch_runner.py

Bot IDs (defined in stress_test_runner.AVAILABLE_BOTS):
    5 = TaskerBot
    6 = ReactBot
    7 = ThinkBot
    8 = OrchestratorBot
"""

import importlib
import os
import sys

from dotenv import load_dotenv

# Ensure imports work when executed directly from any working directory
sys.path.append(os.path.dirname(__file__))

from stress_test_runner import UltimateStressTestRunner, AVAILABLE_BOTS

load_dotenv()

# ==========================================
# CONFIGURACIÓN DEL BATCH
# ==========================================
# Descomenta las combinaciones que quieras ejecutar.

TEST_CONFIGS = [
    # ReactBot
    # {"bot_id": "6", "model": "gpt-4o-mini", "provider": "openai"},
    # {"bot_id": "6", "model": "deepseek-chat", "provider": "deepseek"},
    # {"bot_id": "6", "model": "gemini-2.5-flash", "provider": "gemini"},
    # {"bot_id": "6", "model": "amazon.nova-micro-v1:0", "provider": "bedrock"},

    # ThinkBot (reasoning models)
    # {"bot_id": "7", "model": "gpt-4o-mini", "provider": "openai"},

    # TaskerBot
    # {"bot_id": "5", "model": "gpt-4o-mini", "provider": "openai"},

    # OrchestratorBot
    {"bot_id": "8", "model": "gpt-4o-mini", "provider": "openai"},
    # {"bot_id": "8", "model": "gemini-2.5-flash", "provider": "gemini"},
    # {"bot_id": "8", "model": "deepseek-chat", "provider": "deepseek"},
]


# ── Helpers ────────────────────────────────────────────────────────────────

def resolve_bot_class(bot_id):
    if bot_id not in AVAILABLE_BOTS:
        print(f"⚠️  Bot ID '{bot_id}' no encontrado en AVAILABLE_BOTS. Saltando.")
        return None, None

    bot_info = AVAILABLE_BOTS[bot_id]
    try:
        module    = importlib.import_module(bot_info["module"])
        bot_class = getattr(module, bot_info["class"])
        return bot_class, bot_info["name"]
    except ImportError as e:
        print(f"❌ Error importando {bot_info['name']}: {e}")
        return None, None
    except AttributeError:
        print(f"❌ Clase {bot_info['class']} no encontrada en {bot_info['module']}")
        return None, None


# ── Batch runner ───────────────────────────────────────────────────────────

def run_batch():
    print(f"🚀 INICIANDO BATCH STRESS TEST ({len(TEST_CONFIGS)} configuraciones)...")
    print("=" * 60)

    successful_runs = 0
    failed_runs     = 0

    for i, config in enumerate(TEST_CONFIGS, 1):
        bot_id     = config.get("bot_id")
        model_name = config.get("model", "gpt-4o-mini")
        provider   = config.get("provider", "openai")

        bot_class, bot_name = resolve_bot_class(bot_id)
        if not bot_class:
            failed_runs += 1
            continue

        print(f"\n▶️  RUN {i}/{len(TEST_CONFIGS)}")
        print(f"   🤖 Bot:      {bot_name}")
        print(f"   🧠 Modelo:   {model_name}")
        print(f"   🌐 Provider: {provider}")
        print("-" * 30)

        try:
            runner = UltimateStressTestRunner(bot_class, bot_name, model_name, provider=provider)
            runner.run_all_tests()
            successful_runs += 1
            print(f"✅ Run {i} completado exitosamente.")
        except Exception as e:
            print(f"❌ Run {i} falló: {e}")
            failed_runs += 1

        print("=" * 60)

    print("\n🏁 RESUMEN DEL BATCH")
    print(f"   Total:    {len(TEST_CONFIGS)}")
    print(f"   Exitosos: {successful_runs}")
    print(f"   Fallidos: {failed_runs}")
    print("=" * 60)


if __name__ == "__main__":
    providers = {c.get("provider", "openai") for c in TEST_CONFIGS}

    checks = {
        "openai":   ("OPENAI_API_KEY",           "provider 'openai'"),
        "deepseek": ("DEEPSEEK_API_KEY",          "provider 'deepseek'"),
        "gemini":   ("GOOGLE_API_KEY",            "provider 'gemini'"),
        "bedrock":  ("AWS_BEARER_TOKEN_BEDROCK",  "provider 'bedrock'"),
    }

    for provider, (env_key, label) in checks.items():
        if provider in providers and not os.getenv(env_key):
            print(f"❌ Error: {env_key} requerida para {label}.")
            sys.exit(1)

    run_batch()
