---
name: release-to-prod
description: How to cut a production release of sonika-ai-toolkit — bump version, commit, push to main, and push the vX.Y.Z tag. The GitHub Release (which triggers the PyPI publish) is created MANUALLY by the maintainer, never by this automation. Use whenever asked to "release", "publish", "cut a version", "subir a prod", "sacar una versión", or "deploy the library".
---

# Releasing sonika-ai-toolkit to production

The release process is **intentionally minimal**. Do **only** these steps and
nothing more. Do NOT create the GitHub Release, do NOT touch PyPI, do NOT run
any publish command — the maintainer publishes the GitHub Release by hand, and
that manual publish is what triggers the automated PyPI upload.

## The exact steps (do only these)

1. **Bump the version.** Edit `version="X.Y.Z"` in `setup.py`. Follow semver:
   - patch (`X.Y.Z+1`) — bug fixes, docs, internal refactors with no API change
   - minor (`X.Y+1.0`) — new backward-compatible features
   - major (`X+1.0.0`) — breaking changes to the public API (`__init__.py`)
2. **Commit** the change — a `chore(release): X.Y.Z` message is the convention
   (see `git log` for the exact style used in this repo).
3. **Push to `main`.**
4. **Create and push the annotated tag** `vX.Y.Z`:
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

**That's it — push to main + create the tag. Nothing else.**

## What you MUST NOT do

- ❌ Do **not** create the GitHub Release (`gh release create ...`). The
  maintainer does this manually from the pushed tag.
- ❌ Do **not** run `twine`, `python -m build`, `mkdocs gh-deploy`, or any
  publish/upload command. Publishing the GitHub Release triggers
  `.github/workflows/python-publish.yml`, which uploads to PyPI on its own.
- ❌ Do **not** bump the version in any file other than `setup.py`.

## Before you release — the pre-flight gate

A release must never ship a broken or messy tree. **Run the `clean-tree-before-push`
skill first** (or, at minimum, do its checks): the working tree must be clean of
dead/floating code, tests must pass, and lint must be green.

```bash
pytest tests/unit tests/integration -q   # must pass
ruff check .                              # must be clean
git status                               # no stray/untracked files that shouldn't ship
```

If any of these fail, **stop and fix before releasing** — do not tag a red tree.

## After you push the tag

Tell the user plainly: the tag `vX.Y.Z` is pushed and CI is green; the
**maintainer now needs to publish the GitHub Release manually** from that tag to
trigger the PyPI upload. Do not attempt to do this step yourself.
