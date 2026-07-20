---
name: clean-tree-before-push
description: Guard against leaving dead/floating code before pushing sonika-ai-toolkit — no unused modules (like the old orchestrator `nodes/` folder), no orphaned files, no unreferenced imports/exports, tests green, lint clean. Use before any git push, before cutting a release, or whenever asked to "leave the library perfect", "no dejar código flotando", "limpiar antes de subir", or "verify nothing is dead before pushing".
---

# Leave the library perfect before pushing

The library must ship **clean**: no dead code, no floating files, no half-removed
features. A real example of what NOT to leave: the orchestrator's `nodes/` folder
stopped being imported in v0.3.0 (the ReAct rewrite) but lingered as dead code
for ~5 months until it was finally deleted. **Do not let that happen again** — when
a feature is removed or replaced, remove *everything* it left behind in the same
change, and verify it before the push.

## The checklist — run ALL of it before `git push`

### 1. No dead / unreferenced code

- **No orphaned modules or folders.** Anything you stopped using must be
  deleted, not left behind. Find modules that nothing imports:
  ```bash
  # For a suspect module `foo.py`, confirm it is actually imported somewhere:
  grep -rEn "import +foo\b|from .*foo import|\.foo\b" src/ tests/
  ```
  If a file under `src/` is imported by nothing (and isn't a documented public
  entry point), it is dead — delete it.
- **No unreferenced imports or exports.** Every name added to
  `src/sonika_ai_toolkit/__init__.py` (`__all__` + the import block) must
  actually exist and be used; every import in a module must be used. Ruff
  (`F401`, `F811`, `F841`) catches most of this — see step 4.
- **When you remove a feature, remove its whole footprint:** the module, its
  tests, its exports in `__init__.py`, its mentions in `CLAUDE.md` and `docs/`,
  and any nav entry in `mkdocs.yml`. Grep the name across the repo to be sure:
  ```bash
  grep -rIn "RemovedThing" src/ tests/ docs/ CLAUDE.md mkdocs.yml
  ```

### 2. No floating / stray files

- No scratch files, `.tmp*`, ad-hoc debug scripts, or `__pycache__` noise staged
  for commit. Check what would be committed:
  ```bash
  git status --porcelain          # nothing unexpected
  git status --ignored --porcelain # sanity-check ignored files too
  ```
- Temporary work belongs in the scratchpad directory, **never** in the repo.

### 3. Tests are green (mirrored 1:1)

- The `tests/unit/` tree mirrors `src/sonika_ai_toolkit/` one file per module.
  A new/changed module needs its test file updated in the same change; a removed
  module's test file must be removed too.
  ```bash
  pytest tests/unit tests/integration -q   # must pass, no skips you introduced
  ```

### 4. Lint is clean

```bash
ruff check .        # must print "All checks passed!"
```
Ruff is the primary automated guard for unused imports (`F401`), redefinitions
(`F811`), and unused locals (`F841`) — a green ruff is a hard requirement.

### 5. Public API + docs stay in sync

- If you changed `src/sonika_ai_toolkit/__init__.py`, update
  `docs/getting-started.md` and the relevant `docs/*.md` page.
- If you added/removed an agent, classifier, tool, or event type, update the
  matching `docs/*.md` page (and `CLAUDE.md`), plus `mkdocs.yml` nav for a new
  top-level component. Docs auto-deploy on push to `main`, so stale docs ship
  immediately.

## The rule

> If you removed or replaced something, its footprint is gone **completely** in
> the same change. If you added something, it is exported, tested, documented,
> and lint-clean. Only then push.

Run this skill's checks (or the `verify` / `code-review` skills) before every
push, and always as the pre-flight gate of `release-to-prod`.
