# Tools

Sonika AI Toolkit includes 18 built-in tools and supports custom tool creation via Pydantic.

## Built-in Tools

### Core File & System Tools

| Tool | Description |
|------|-------------|
| `RunBashTool` | Execute shell commands |
| `BashSafeTool` | Execute shell commands (sandboxed) |
| `ReadFileTool` | Read file contents |
| `WriteFileTool` | Write content to files |
| `ListDirTool` | List directory contents |
| `DeleteFileTool` | Delete files |
| `FindFileTool` | Case-insensitive glob file search |
| `RunPythonTool` | Execute Python code |

### Network & API Tools

| Tool | Description |
|------|-------------|
| `CallApiTool` | Make HTTP API calls |
| `SearchWebTool` | Search the web |
| `FetchWebPageTool` | Fetch and extract web page content |
| `EmailSMTPTool` | Send emails via SMTP |

### Database Tools

| Tool | Description |
|------|-------------|
| `SQLiteTool` | Query SQLite databases |
| `PostgreSQLTool` | Query PostgreSQL databases |
| `MySQLTool` | Query MySQL databases |
| `RedisTool` | Interact with Redis |

### Integration Tools

| Tool | Description |
|------|-------------|
| `EmailTool` | Send emails (simplified) |
| `SaveContacto` | Save contact information |

## Using Tools with Agents

```python
from sonika_ai_toolkit import (
    OrchestratorBot, OpenAILanguageModel,
    RunBashTool, ReadFileTool, WriteFileTool, SearchWebTool,
)

llm = OpenAILanguageModel("sk-...", model_name="gpt-4o-mini")
bot = OrchestratorBot(
    strong_model=llm,
    fast_model=llm,
    instructions="You are a helpful assistant.",
    tools=[RunBashTool(), ReadFileTool(), WriteFileTool(), SearchWebTool()],
    memory_path="/tmp/bot_memory",
)

result = bot.run("List all Python files in the current directory")
```

## Creating Custom Tools

Define a custom tool using Pydantic for the argument schema:

```python
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class WeatherInput(BaseModel):
    city: str = Field(..., description="City name")
    units: str = Field("celsius", description="Temperature units: celsius or fahrenheit")

class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get current weather for a city"
    args_schema: type = WeatherInput

    def _run(self, city: str, units: str = "celsius") -> str:
        # Your implementation here
        return f"Weather in {city}: 22°C, sunny"

# Use with any agent
bot = ReactBot(llm, instructions="...", tools=[WeatherTool()])
```

!!! important "args_schema required"
    Tools **must** have an `args_schema` (Pydantic model) for the LLM to correctly generate parameters.

## Tool Registry

```python
from sonika_ai_toolkit.tools.registry import ToolRegistry

registry = ToolRegistry()
registry.register(WeatherTool())
registry.get("get_weather")
registry.list_tools()

# Get tool descriptions formatted for LLM prompts
descriptions = registry.get_tool_descriptions()
```

## Top-Level Imports

```python
from sonika_ai_toolkit import (
    RunBashTool, BashSafeTool,
    ReadFileTool, WriteFileTool, ListDirTool, DeleteFileTool, FindFileTool,
    CallApiTool, SearchWebTool, FetchWebPageTool,
    RunPythonTool, GetDateTimeTool,
    EmailSMTPTool, SQLiteTool, PostgreSQLTool, MySQLTool, RedisTool,
)
```
