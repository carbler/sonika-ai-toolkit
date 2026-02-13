# AI Agents Documentation (`AGENTS.md`)

This document serves as the primary knowledge base for AI Agents (such as OpenDevin, Claude Code, GitHub Copilot Workspace, etc.) working on this repository. It defines the project context, architecture, available skills, and development standards.

## 🧠 Project Context & Mission

**sonika-ai-toolkit** is a robust Python library designed to build state-of-the-art conversational agents and AI tools. It leverages `LangChain` and `LangGraph` to create autonomous bots capable of:
1.  **Complex Reasoning**: Using ReAct patterns and graph-based workflows.
2.  **Tool Execution**: Interacting with external systems (Email, CRM, etc.) via structured tool definitions.
3.  **Multi-Model Support**: Agnostic integration with **OpenAI**, **DeepSeek**, and **Google Gemini**.

The goal is to provide a standardized, scalable framework for banking and customer service bots that is easy to extend and stress-test.

---

## 🤖 Bot Architectures

The project features two primary bot implementations:

### 1. `ReactBot` (Standard Agent)
*   **Path**: `src/sonika_ai_toolkit/agents/react.py`
*   **Architecture**: Uses `LangGraph` state graph (`agent` -> `tools` -> `agent`).
*   **Key Components**:
    *   `ChatState`: TypedDict managing messages, logs, and token usage.
    *   `_InternalToolLogger`: Custom callback handler for tool execution tracking.
    *   `ILanguageModel`: Unified interface for model switching.
*   **Features**: Streaming response, native tool calling, robust error handling, and token usage tracking.

### 2. `TaskerBot` (Advanced Planner)
*   **Path**: `src/sonika_ai_toolkit/agents/tasker/`
*   **Architecture**: Enhanced ReAct pattern with explicit `Planner`, `Executor`, and `Validator` nodes.
*   **Workflow**:
    1.  **Planner**: Breaks down user request into sub-tasks.
    2.  **Executor**: Executes the current sub-task using available tools.
    3.  **Validator**: Checks if the task is complete or needs more steps.
*   **Features**: Iterative problem solving with recursion limits and separation of concerns.

---

## 🛠 Skills & Tools

Agents working on this repo should be aware of the "Skills" (Tools) available to the bots. These are defined in `src/sonika_ai_toolkit/tools/` and other modules.

### Core Skills
| Skill / Tool Name | Class Name | Description | Inputs |
| :--- | :--- | :--- | :--- |
| **Email Sender** | `EmailTool` | Sends emails to users. | `to_email` (str), `subject` (str), `message` (str) |
| **Contact Saver** | `SaveContact` | Saves/Updates contact info in CRM. | `nombre` (str), `correo` (str), `telefono` (str) |

---

## 🌐 Supported Models

This project implements a unified `ILanguageModel` interface in `src/sonika_ai_toolkit/utilities/types.py`.

| Provider | Class Name | Config File | Env Variable |
| :--- | :--- | :--- | :--- |
| **OpenAI** | `OpenAILanguageModel` | `utilities/models.py` | `OPENAI_API_KEY` |
| **DeepSeek** | `DeepSeekLanguageModel` | `utilities/models.py` | `DEEPSEEK_API_KEY` |
| **Google Gemini** | `GeminiLanguageModel` | `utilities/models.py` | `GOOGLE_API_KEY` |
| **Amazon Bedrock** | `BedrockLanguageModel` | `utilities/models.py` | `AWS_BEARER_TOKEN_BEDROCK` |

---

## 💻 Development Standards

### 1. Commands
| Task | Command |
| :--- | :--- |
| **Environment** | `python -m venv venv && source venv/bin/activate` |
| **Install** | `pip install -e .` |
| **Run All Tests** | `pytest` or `python test/test.py` |
| **Single Test** | `pytest test/test.py::test_function_name` |
| **Stress Test** | `python test_ultimate/banking_operations/batch_runner.py` |
| **Build** | `python setup.py sdist bdist_wheel` |
| **Linting** | `ruff check .` or `flake8 src` |
| **Type Check** | `mypy src` (Recommended) |

### 2. Code Style Guidelines
*   **Imports**: Organize imports in three groups: 
    1. Standard library imports (e.g., `os`, `sys`, `typing`).
    2. Third-party library imports (e.g., `langchain`, `pydantic`, `langgraph`).
    3. Local application/library specific imports (`sonika_ai_toolkit`).
    *Use absolute imports for local modules.*
*   **Formatting**: Strictly follow PEP 8. Use 4 spaces for indentation. Use double blank lines between classes and top-level functions. Limit line length to 88-100 characters.
*   **Naming**: 
    *   Classes: `PascalCase` (e.g., `ReactBot`).
    *   Functions/Variables: `snake_case` (e.g., `get_response`).
    *   Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TEMPERATURE`).
    *   Private members: Prefix with underscore (e.g., `_internal_method`).
*   **Types**: Use type hints for ALL function signatures and class attributes. Utilize `typing` (List, Dict, Optional, etc.) and `Annotated` for LangGraph states.
*   **Documentation**: Google-style docstrings for all public classes and methods.
    ```python
    def method(arg1: int) -> str:
        """Description.
        Args:
            arg1: Explanation.
        Returns:
            Explanation.
        """
    ```

### 3. Error Handling & Validation
*   **Tool Execution**: Wrap tool calls in try-except blocks. Use logging to capture failures without crashing the agent.
*   **LLM Failures**: Implement retry logic or graceful degradation if the LLM fails to return a valid response.
*   **Validation**: Use Pydantic's `BaseModel` for structured input/output validation. Ensure all inputs to the bot are validated before processing.

### 4. Common Gotchas
*   **Gemini Message Order**: Gemini requires a specific message order (System, then User/AI pairs). Check `agents/react.py` for how this is handled.
*   **State Persistence**: `LangGraph` requires a `checkpointer` (e.g., `MemorySaver`) to maintain state between turns.
*   **Package Path**: When running tests, ensure `src` is in `PYTHONPATH` or use `pip install -e .`.

### 5. Agent Workflow
1.  **Read Context**: Review `AGENTS.md` and `README.md`.
2.  **Environment**: Ensure `.env` is configured with necessary API keys.
3.  **Implementation**: Write clean, typed code. Ensure multi-model compatibility.
4.  **Verification**: 
    *   Run `test/test.py` for functional validation.
    *   Run `test_ultimate/banking_operations/batch_runner.py` for stress testing before core changes.
5.  **Documentation**: Update docstrings and this file if architecture changes.


---

## 📂 Project Structure Map

*   `src/sonika_ai_toolkit/`: Core library code.
    *   `agents/`: Bot implementations.
        *   `react.py`: Main `ReactBot` implementation.
        *   `tasker/`: `TaskerBot` implementation.
    *   `classifiers/`: Text classification tools.
    *   `document_processing/`: PDF and document tools.
    *   `tools/`: Tool definitions.
    *   `utilities/`: Models and common types.
*   `test/`: Unit and functional tests.
*   `test_ultimate/`: Stress testing framework for complex workflows.

