# Skills

Skills are **folder-based capability packs** that teach agents new abilities:
markdown instructions that get injected into the system prompt, plus optional
tools that get merged into the agent's tool list. Both agents support them:
**ReactBot** and **OrchestratorBot**.

## Folder Layout

```
skills/
├── facturacion/
│   ├── SKILL.md      # instructions injected into the system prompt
│   └── tools.py      # optional BaseTool subclasses
└── reportes/
    └── SKILL.md
```

### SKILL.md

Plain markdown with optional `---` frontmatter (`name` and `description`).
Without frontmatter, the folder name is used as the skill name:

```markdown
---
name: facturacion
description: Genera facturas legales colombianas
---

Cuando el usuario pida una factura:
1. Valida NIT y razón social.
2. Usa la herramienta `generar_factura`.
3. Confirma el número de factura generado.
```

### tools.py (optional)

Any `BaseTool` subclass **defined in the file** is instantiated and merged
into the agent's tool list (classes merely imported are ignored):

```python
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class FacturaSchema(BaseModel):
    nit: str = Field(..., description="NIT del cliente")
    total: float = Field(..., description="Total de la factura")


class GenerarFacturaTool(BaseTool):
    name: str = "generar_factura"
    description: str = "Genera una factura legal"
    args_schema: Type[BaseModel] = FacturaSchema

    def _run(self, nit: str, total: float) -> str:
        return f"Factura generada para {nit} por ${total}"
```

!!! warning "Security"
    `tools.py` is imported and executed as **arbitrary Python code**. Only
    load skill directories you trust.

## Using Skills

Every bot accepts `skills` (a list of `Skill` objects) and/or `skills_dir`
(a directory scanned for skill folders). Both can be combined:

```python
from sonika_ai_toolkit import OrchestratorBot, Skill, load_skills
from sonika_ai_toolkit.agents.react import ReactBot

# Load everything under a directory
bot = OrchestratorBot(
    strong_model=model,
    fast_model=model,
    instructions="Eres un asistente de negocio.",
    skills_dir="./skills",
)

# Or load/build skills explicitly
skill = Skill.from_dir("./skills/facturacion")
programmatic = Skill(
    name="saludos",
    instructions="Saluda siempre con el nombre del usuario.",
    tools=[],  # optional BaseTool instances
)
bot = ReactBot(
    language_model=model,
    instructions="Eres un asistente.",
    skills=[skill, programmatic],
)
```

## How Skills Interact With the Graph

Skills are **not** a node and don't add any conditional routing — they are
resolved once, at bot construction, and then apply unconditionally on every
turn for the rest of that bot's life:

1. **At construction** (`__init__`), `skills=`/`skills_dir=` are loaded once
   via `resolve_skills()`. Their instructions are pre-rendered into one string
   (`self._skills_prompt`) and their tools are merged into the bot's tool list
   **before** `bind_tools()` — so skill tools are indistinguishable from any
   other tool by the time the graph is built.
2. **Every turn**, the `agent` node (OrchestratorBot) or the prompt builder
   (ReactBot) appends `self._skills_prompt` to the system prompt — right after
   the base `instructions` (and, for the orchestrator, after the memory
   context, before mode-specific text like `[MODO PLAN]`). This happens
   regardless of mode, `enable_planning`, or what the user asked — the model
   always sees the skill instructions.
3. **Skill tools run through the normal `tools` node** like any built-in
   tool: no interception, no acknowledgment step. If the model calls a skill
   tool it executes for real and shows up in `tools_executed`, exactly like
   `RunBashTool` or any other tool would.

So "when does a skill act?" has a simpler answer than planning: the
instructions are **always** in context; a skill's *tool* only runs when the
model decides to call it — the same as any other tool. There's no separate
flag to enable skills at runtime (unlike `enable_planning`) — passing
`skills=`/`skills_dir=` at construction is the only switch.

## Semantics

- **Instructions** are rendered as a `## SKILLS` block appended to the system
  prompt (after the bot's own `instructions`).
- **Tools** are merged before the model is bound, deduplicating by tool name.
  Tools passed explicitly via `tools=` **win** over a skill tool with the same
  name.
- **Broken skills** (unreadable SKILL.md, tools.py that fails to import) are
  logged and skipped — they never abort bot construction.
- Loading is deterministic: skill folders load in sorted order.

## API Reference

```python
from sonika_ai_toolkit import Skill, load_skills

Skill(name, instructions, description="", tools=[], path=None)
Skill.from_dir(path)      # load one skill folder
load_skills(skills_dir)   # load every skill folder under a directory
```
