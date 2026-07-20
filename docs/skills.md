# Skills

Skills are **folder-based capability packs** that teach agents new abilities:
markdown instructions that get injected into the system prompt, plus optional
tools that get merged into the agent's tool list. **OrchestratorBot** supports
them.

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

OrchestratorBot accepts `skills` (a list of `Skill` objects) and/or `skills_dir`
(a directory scanned for skill folders). Both can be combined:

```python
from sonika_ai_toolkit import OrchestratorBot, Skill, load_skills

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
bot = OrchestratorBot(
    strong_model=model,
    fast_model=model,
    instructions="Eres un asistente.",
    skills=[skill, programmatic],
)
```

## On-demand loading (progressive disclosure) — the default

By default, skills load **on demand** so you don't pay for every skill's full
body on every turn. This follows the progressive-disclosure model:

1. **Level 1 — index (always in the prompt):** only a lightweight `## SKILLS`
   block with one `- name — description` line per skill is injected. This is
   what the model reads to decide *whether* a skill is relevant — a few tokens
   per skill instead of the whole body.
2. **Level 2 — body on demand:** a built-in `load_skill` tool is registered
   automatically. When the model judges a skill relevant, it calls
   `load_skill("<name>")` and receives that single skill's full instructions as
   a tool result, then follows them. Only the skill it actually needs is loaded.
3. **Level 3 — bundled files:** `load_skill` also returns the skill's folder
   `path`, so the model can read extra files in the skill (scripts, templates)
   with the normal file tools when the instructions reference them.

This costs one extra round-trip **only when a skill is relevant**; unrelated
requests never pay for skill bodies at all.

### Escape hatch: `skills_eager=True`

Pass `skills_eager=True` to either bot to restore the legacy behavior — every
skill's full body is injected into the system prompt on every turn and no
`load_skill` tool is registered. Useful for a tiny set of small skills where
the extra round-trip isn't worth it:

```python
bot = OrchestratorBot(..., skills_dir="./skills", skills_eager=True)
```

## How Skills Interact With the Graph

Skills are **not** a node and don't add any conditional routing — they are
resolved once, at bot construction:

1. **At construction** (`__init__`), `skills=`/`skills_dir=` are loaded once
   via `resolve_skills()`. The prompt block is pre-rendered into one string
   (`self._skills_prompt`) — the lightweight index by default, or the full
   bodies with `skills_eager=True`. Skill **tools** are merged into the bot's
   tool list **before** `bind_tools()` (plus the `load_skill` tool in on-demand
   mode) — so they're indistinguishable from any other tool once the graph is
   built.
2. **Every turn**, the `agent` node appends `self._skills_prompt` to the system
   prompt — right after the base `instructions` and the memory context, before
   mode-specific text like `[MODO PLAN]`.
3. **Skill tools (and `load_skill`) run through the normal `tools` node** like
   any built-in tool: no interception, no acknowledgment step. They execute for
   real and show up in `tools_executed`, exactly like `RunBashTool` would.

There's no separate flag to enable skills at runtime (unlike `enable_planning`)
— passing `skills=`/`skills_dir=` at construction is the only switch; the
on-demand vs eager choice is `skills_eager`.

## Semantics

- **Instructions** render as a `## SKILLS` block appended to the system prompt
  (after the bot's own `instructions`): a lightweight name+description index by
  default, or the full bodies under `skills_eager=True`.
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
