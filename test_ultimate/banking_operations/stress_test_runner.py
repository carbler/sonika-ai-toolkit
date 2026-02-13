import os
import sys
import time
import importlib
from datetime import datetime
from dotenv import load_dotenv

# --- CONFIGURACIÓN DE RUTAS ---
# Ajustar para importar desde src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from sonika_ai_toolkit.utilities.models import OpenAILanguageModel, DeepSeekLanguageModel, GeminiLanguageModel, BedrockLanguageModel
from sonika_ai_toolkit.utilities.types import Message

# Importar componentes locales
from tools import (
    GetUserProfile, TransactionTool, CreateTicket, BlockAccountTool,
    RefundTool, GetTransactionHistory, VerifyIdentityDocument,
    ApplyPromoCode, CheckFraudScore, UpdateAccountTier,
    ScheduleCallback, AdjustCreditLimit
)
from instructions import PERSONALITY_TONE, LIMITATIONS, FUNCTION_PURPOSE
from test_cases import tests_data

load_dotenv()

# ==========================================
# CONFIGURACIÓN DE BOTS DISPONIBLES
# ==========================================

AVAILABLE_BOTS = {
    "5": {
        "name": "TaskerBot",
        "module": "sonika_ai_toolkit.agents.tasker.tasker_bot",
        "class": "TaskerBot"
    },
    "6": {
        "name": "ReactBot",
        "module": "sonika_ai_toolkit.agents.react",
        "class": "ReactBot"
    }
}

# ==========================================
# TEST RUNNER CON SCORING Y REPORTE
# ==========================================

class UltimateStressTestRunner:
    def __init__(self, bot_class, bot_name, model_name, provider="openai"):
        self.bot_class = bot_class
        self.bot_name = bot_name
        self.model_name = model_name
        self.provider = provider

        if provider == "deepseek":
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("❌ DEEPSEEK_API_KEY no encontrada")
            self.llm = DeepSeekLanguageModel(api_key, model_name=model_name, temperature=0)
        elif provider == "gemini":
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("❌ GOOGLE_API_KEY no encontrada")
            self.llm = GeminiLanguageModel(api_key, model_name=model_name, temperature=0)
        elif provider == "bedrock":
            api_key = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            if not api_key:
                raise ValueError("❌ AWS_BEARER_TOKEN_BEDROCK no encontrada")
            self.llm = BedrockLanguageModel(api_key, region_name=aws_region, model_name=model_name, temperature=0)
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("❌ OPENAI_API_KEY no encontrada")
            self.llm = OpenAILanguageModel(api_key, model_name=model_name, temperature=0)

        self.total_score = 0
        self.max_score = 0
        self.test_results = []
        self.start_time = None
        self.end_time = None

        # Configurar directorio de reportes
        self.report_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
        os.makedirs(self.report_dir, exist_ok=True)

        # Nombre del archivo dinámico
        safe_bot_name = self.bot_name.replace(" ", "_").replace("(", "").replace(")", "")
        safe_model_name = self.model_name.replace(":", "-")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"report_{safe_bot_name}_{safe_model_name}_{timestamp}.txt"
        self.report_file = os.path.join(self.report_dir, filename)
        self.log_buffer = []

    def log(self, message):
        """Imprime en consola y guarda en buffer para archivo"""
        print(message)
        self.log_buffer.append(message)

    def save_report(self):
        """Escribe todo el log en el archivo"""
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.log_buffer))
        print(f"\n📄 Reporte guardado en: {self.report_file}")

    def build_conversation_history(self, messages_data):
        """Convierte lista de tuplas (content, is_bot) a objetos Message"""
        return [Message(content=msg[0], is_bot=msg[1]) for msg in messages_data]

    def run_all_tests(self):
        self.start_time = time.time()
        for t_num, t_name, t_hist, t_input, t_val in tests_data:
            self.run_test(
                test_num=t_num,
                test_name=t_name,
                conversation_history=t_hist,
                user_input=t_input,
                validation_fn=t_val
            )
        self.end_time = time.time()
        self.print_final_report()

    def run_test(self, test_num, test_name, conversation_history, user_input, validation_fn, max_points=100):
        self.log(f"\n{'='*90}")
        self.log(f"🧪 TEST #{test_num}: {test_name}")
        self.log(f"   💯 Max Score: {max_points} points")
        self.log(f"   📜 Conversación previa: {len(conversation_history)} mensajes")
        self.log(f"   📥 Input Usuario: '{user_input}'")

        try:
            tools = [
                GetUserProfile(), TransactionTool(), CreateTicket(), BlockAccountTool(),
                RefundTool(), GetTransactionHistory(), VerifyIdentityDocument(),
                ApplyPromoCode(), CheckFraudScore(), UpdateAccountTier(),
                ScheduleCallback(), AdjustCreditLimit()
            ]

            # Instanciación dinámica
            if self.bot_name == "ReactBot":
                # Adaptador para ReactBot que espera 'instructions' concatenadas
                combined_instructions = f"{FUNCTION_PURPOSE}\n\n{PERSONALITY_TONE}\n\n{LIMITATIONS}"
                bot = self.bot_class(
                    language_model=self.llm,
                    instructions=combined_instructions,
                    tools=tools,
                    on_tool_start=lambda x, y: None,
                    on_tool_end=lambda x, y: None,
                    on_tool_error=lambda x, y: None
                )
            elif self.bot_name == "TaskerBot":
                # TaskerBot con parámetros mejorados para evitar recursion limit
                bot = self.bot_class(
                    language_model=self.llm,
                    function_purpose=FUNCTION_PURPOSE,
                    personality_tone=PERSONALITY_TONE,
                    limitations=LIMITATIONS,
                    dynamic_info='',
                    tools=tools,
                    max_iterations=15,
                    recursion_limit=100,
                    on_tool_start=lambda x, y: None,
                    on_tool_end=lambda x, y: None,
                    on_tool_error=lambda x, y: None
                )
            else:
                # Estándar para otros bots (si quedan)
                bot = self.bot_class(
                    language_model=self.llm,
                    function_purpose=FUNCTION_PURPOSE,
                    personality_tone=PERSONALITY_TONE,
                    limitations=LIMITATIONS,
                    dynamic_info='',
                    tools=tools,
                    on_tool_start=lambda x, y: None,
                    on_tool_end=lambda x, y: None,
                    on_tool_error=lambda x, y: None
                )

            history_messages = self.build_conversation_history(conversation_history)

            start_time = time.time()
            response = bot.get_response(user_input=user_input, logs=[], messages=history_messages)
            execution_time = time.time() - start_time

        except Exception as e:
            self.log(f"   ❌ CRASH: {e}")
            import traceback
            traceback.print_exc()
            self.record_result(test_num, test_name, 0, max_points, f"CRASH: {str(e)}", execution_time=0)
            return

        bot_content = response.get('content', '')
        tools_executed = response.get('tools_executed', [])

        adapted_logs = [{"name": t.get('tool_name'), "input": str(t.get('args'))} for t in tools_executed]
        tool_names = [t['name'] for t in adapted_logs]

        # Validación
        try:
            score, feedback = validation_fn(adapted_logs, bot_content, conversation_history)
        except Exception as e:
            self.log(f"   ❌ Error en validación: {e}")
            score = 0
            feedback = f"Error en validación: {str(e)}"

        passed = score >= (max_points * 0.7)  # 70% es passing
        status = "✅ PASSED" if passed else "❌ FAILED"

        self.log(f"   {status} - Score: {score}/{max_points} ({int(score/max_points*100)}%)")
        self.log(f"   ⏱️  Execution Time: {execution_time:.2f}s")
        self.log(f"   📊 Feedback: {feedback}")
        self.log(f"\n   🔍 DEBUG INFO:")
        self.log(f"   🤖 Bot Response: \"{bot_content[:200]}...\"" if len(bot_content) > 200 else f"   🤖 Bot Response: \"{bot_content}\"")
        self.log(f"   🔧 Tools Executed: {tool_names if tool_names else '[NINGUNA]'}")
        self.log(f"{'='*90}")

        self.record_result(test_num, test_name, score, max_points, feedback, execution_time)

    def record_result(self, test_num, test_name, score, max_score, feedback, execution_time):
        self.total_score += score
        self.max_score += max_score
        self.test_results.append({
            "test_num": test_num,
            "name": test_name,
            "score": score,
            "max_score": max_score,
            "percentage": int(score/max_score*100) if max_score > 0 else 0,
            "feedback": feedback,
            "execution_time": execution_time
        })

    def print_final_report(self):
        duration = self.end_time - self.start_time if self.end_time else 0
        minutes, seconds = divmod(duration, 60)

        self.log(f"\n\n{'='*90}")
        self.log(f"📈 REPORTE FINAL")
        self.log(f"🤖 Bot: {self.bot_name}")
        self.log(f"🧠 Model: {self.model_name}")
        self.log(f"⏱️  Total Duration: {int(minutes)}m {int(seconds)}s")
        self.log(f"{'='*90}")
        self.log(f"Total Score: {self.total_score}/{self.max_score} ({int(self.total_score/self.max_score*100)}%)")
        self.log(f"\nDetalle por Test:")
        for result in self.test_results:
            emoji = "✅" if result['percentage'] >= 70 else "❌"
            self.log(f"{emoji} Test #{result['test_num']}: {result['score']}/{result['max_score']} ({result['percentage']}%) - {result['name']}")
        self.log(f"{'='*90}\n")

        self.save_report()

def select_bot():
    print("\n🤖 SELECCIONE EL BOT A PROBAR:")
    for key, bot_info in AVAILABLE_BOTS.items():
        print(f"  [{key}] {bot_info['name']}")

    choice = input("\nOpción (default 1): ").strip()
    if not choice:
        choice = "1"

    if choice not in AVAILABLE_BOTS:
        print("❌ Opción inválida. Usando default (1).")
        choice = "1"

    selected = AVAILABLE_BOTS[choice]

    try:
        module = importlib.import_module(selected["module"])
        bot_class = getattr(module, selected["class"])
        return bot_class, selected["name"]
    except ImportError as e:
        print(f"❌ Error importando {selected['name']}: {e}")
        sys.exit(1)
    except AttributeError:
        print(f"❌ Clase {selected['class']} no encontrada en {selected['module']}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 INICIANDO ULTIMATE STRESS TEST DE NEOFIN AI...")

    # Selección interactiva
    try:
        bot_class, bot_name = select_bot()

        provider = input("\n🌐 Ingrese proveedor (openai/deepseek) [default: openai]: ").strip().lower()
        if not provider:
            provider = "openai"

        default_model = "gpt-4o-mini" if provider == "openai" else "deepseek-chat"
        model_name = input(f"🧠 Ingrese nombre del modelo (default: {default_model}): ").strip()
        if not model_name:
            model_name = default_model

        runner = UltimateStressTestRunner(bot_class, bot_name, model_name, provider=provider)
        runner.run_all_tests()

    except ValueError as e:
        print(f"❌ Error de configuración: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🚫 Ejecución cancelada por el usuario.")
        sys.exit(0)
