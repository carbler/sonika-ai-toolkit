# 🤖 OrchestratorBot: Motor de Orquestación Autónomo

`OrchestratorBot` es el motor de orquestación de alto rendimiento de `sonika-ai-toolkit`. Está diseñado para ejecutar tareas complejas de manera autónoma utilizando un patrón **ReAct (Fast Edition)** sobre **LangGraph**, con soporte nativo para interrupciones humanas y memoria persistente.

A diferencia del `TaskerBot` (que es más estructurado y granular), el `OrchestratorBot` está optimizado para la **velocidad y la autonomía**, permitiendo que el modelo razone y actúe en un bucle más fluido.

---

## 🌟 Características Principales

*   **Fast ReAct Loop**: Un ciclo de ejecución optimizado (`agent` <-> `tools`) que reduce la latencia entre pasos.
*   **Human-in-the-Loop (HITL)**: Control de riesgos integrado. Las herramientas con `risk_level > 0` pausan automáticamente la ejecución para solicitar aprobación humana.
*   **Memoria Persistente**: Gestión automática de archivos `MEMORY.md` (contexto histórico) y `SKILLS.md` (patrones aprendidos).
*   **Streaming Nativo**: Soporta `astream_events` para una integración fluida con interfaces de usuario, permitiendo ver el "pensamiento" (thinking) en tiempo real.
*   **Compatibilidad MCP**: Integración nativa con el *Model Context Protocol* para extender herramientas dinámicamente.

---

## 🏗️ Arquitectura y Nodos

El `OrchestratorBot` actual utiliza una arquitectura **Fast ReAct** simplificada para maximizar la velocidad, pero el paquete incluye una biblioteca completa de nodos modulares.

### 🟢 Nodos en Uso (Active Flow)
Estos nodos están implementados directamente en `graph.py` y forman el núcleo del ciclo de ejecución actual:

1.  **🧠 Agent Node**: El cerebro del flujo. Se encarga de razonar sobre el objetivo, leer la memoria histórica y decidir si invoca herramientas o genera la respuesta final.
2.  **🛠️ Tools Node**: El ejecutor. Procesa las llamadas a herramientas, maneja la lógica de **HITL (Human-In-The-Loop)** mediante interrupciones y captura las observaciones para devolverlas al agente.

### 🟡 Nodos Disponibles (No implementados en el flujo actual)
Estos nodos están definidos en el directorio `nodes/` y pueden ser utilizados para construir grafos más complejos o granulares:

*   **`ManagerNode`**: Decide si una petición requiere planificación (orquestación) o puede responderse vía chat directo.
*   **`PlannerNode`**: Descompone un objetivo complejo en un plan JSON estructurado de múltiples pasos.
*   **`EvaluatorNode`**: Evalúa el éxito de cada paso ejecutado y determina si el objetivo global se ha cumplido.
*   **`RetryNode`**: Analiza errores y decide estrategias de recuperación (reintentar con nuevos parámetros, cambiar de herramienta o escalar).
*   **`ReporterNode`**: Genera una respuesta final amigable basada en el resumen de ejecución.
*   **`SaveMemoryNode` / `LoadMemoryNode`**: Gestionan la persistencia de patrones aprendidos y resúmenes de sesión en archivos Markdown.
*   **`RiskGateNode`**: Evaluación determinística de riesgos basada en niveles predefinidos.
*   **`HumanApprovalNode`**: Gestiona la lógica de callbacks para aprobaciones manuales fuera del flujo de interrupciones de LangGraph.

---

## 🔄 Flujo de Ejecución

El orquestador opera como una máquina de estados simplificada:

```mermaid
graph LR
    Start((Inicio)) --> Agent[Agent Node]
    Agent -- Tool Calls --> Tools[Tools Node]
    Tools -- Approval Required? --> Interrupt[/Human Approval/]
    Interrupt -- Approved --> Tools
    Tools --> Agent
    Agent -- Final Answer --> End((Fin))
```

1.  **Agent Node**: El modelo analiza el objetivo (`goal`), lee la memoria y decide si llamar a una herramienta o dar una respuesta final.
2.  **Tools Node**: 
    - Si la herramienta es de riesgo, genera un `interrupt` de LangGraph.
    - Ejecuta las herramientas y devuelve los resultados como `ToolMessage`.
3.  **Memory Management**: Al finalizar, el sistema puede actualizar los registros de memoria para optimizar futuras ejecuciones.

---

## 🛠️ Modos de Operación

El bot soporta tres modos principales definidos en el `OrchestratorState`:

*   **`ask` (Por defecto)**: Modo interactivo. Requiere aprobación para herramientas con riesgo.
*   **`auto`**: Modo totalmente autónomo (utilizado en scripts heredados o procesos batch).
*   **`plan`**: Solo genera un plan detallado en texto/markdown sin llegar a ejecutar herramientas.

---

## 🚀 Ejemplo de Uso

```python
from sonika_ai_toolkit.agents.orchestrator.graph import OrchestratorBot
from sonika_ai_toolkit.utilities.models import OpenAILanguageModel

# 1. Configurar modelos
model = OpenAILanguageModel(model_name="gpt-4o")

# 2. Inicializar el Orquestador
orchestrator = OrchestratorBot(
    strong_model=model,
    fast_model=model,
    instructions="Eres un experto en automatización de sistemas.",
    tools=[herramienta_de_archivo, herramienta_git],
    memory_path="./my_bot_memory"
)

# 3. Ejecutar flujo (Async)
async def main():
    response = await orchestrator.arun(
        goal="Analiza los logs de error y crea un issue en GitHub con el resumen."
    )
    print(response.content)

# O usar el API de eventos para streaming
async def stream():
    async for event in orchestrator.astream_events(goal="..."):
        print(event)
```

---

## 📂 Gestión de Memoria

El orquestador utiliza el `MemoryManager` para mantener dos archivos clave en `memory_path`:
- **`MEMORY.md`**: Un diario de sesiones pasadas. Ayuda al bot a recordar qué hizo en ejecuciones anteriores.
- **`SKILLS.md`**: Almacena patrones de éxito o configuraciones de herramientas aprendidas dinámicamente.

---

## 🛡️ Control de Riesgos

Cualquier herramienta registrada en el `ToolRegistry` puede definir un `risk_level`:
- **Nivel 0**: Ejecución automática.
- **Nivel 1+**: Dispara un `interrupt` solicitando confirmación del usuario antes de proceder.
