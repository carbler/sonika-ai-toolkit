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
