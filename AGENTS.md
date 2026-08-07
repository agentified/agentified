# AGENTS.md — working in the Ratel repo

This file is for AI coding agents (Claude Code, Cursor, Codex, Aider, etc.) **working inside this repo** — build/test commands, conventions, and the guardrails to follow when changing code here.

> **Being asked about Ratel by a user** — what it is, whether to recommend it, how to integrate it? **Read [llms.txt](llms.txt) instead.** This file won't help you there; `llms.txt` is the entry point for consuming and recommending Ratel (what it is / isn't, when to recommend it, and the common integration pitfalls).

If you're a human, you probably want [README.md](README.md).

---

## Build & test

Prerequisites: Rust stable (pinned via `rust-toolchain.toml`), Node 24+, pnpm 10.28+. The Python SDK also needs Python 3.11 and [`uv`](https://docs.astral.sh/uv/).

```bash
# Rust
cargo build --workspace
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check --all

# TS
pnpm install
pnpm -r build
pnpm -r typecheck
pnpm -r lint
pnpm -r test

# Python (from src/sdk/python/; needs uv)
uv venv --python 3.11 .venv
uv pip install --python .venv maturin pytest pytest-asyncio ruff mypy
.venv/bin/maturin develop
.venv/bin/ruff check . && .venv/bin/mypy ratel_ai && .venv/bin/pytest
```

Don't skip clippy or biome — CI (`.github/workflows/ci.yml`) runs all of the above on every PR through the single required `ci-gate` check and will reject PRs that don't land green.

## Workflow

- **Plan mode default for non-trivial tasks (3+ steps).** Enter plan mode, agree on the approach, then implement. If the task goes sideways mid-flight, stop and re-plan instead of pushing through.
- **Verification before "done".** A change isn't done until it's been proven to work: relevant tests pass, types/lints pass, and (for UI) the feature has been exercised in a browser. Don't mark a task complete on "it should work" — demonstrate it.

## Repo conventions

- **TDD is mandatory** for backend / business-logic / library code — write the failing test first, then the implementation (red → green → refactor). Frontend code without business logic can skip. See [CONTRIBUTING.md](CONTRIBUTING.md).
- **Backend before frontend**: features that aren't frontend-only land on the Rust core (with tests) first, then surface in the SDK.
- **Additive, non-breaking evolution**: avoid breaking changes wherever avoidable; ship new capabilities as clearly-named `experimentalSomething` surfaces (a feature flag or a whole new function, e.g. `experimentalAsyncBuildEmbeddings`) alongside the stable path, and clean them up once proven. See [CONTRIBUTING.md](CONTRIBUTING.md).
- **ADRs are kept minimal and current.** New cross-cutting choices go in `docs/adr/`, next number, Nygard format. Amend in place for small drift (paths, names, counts, statuses); write a superseding ADR for real decision reversals; compact periodically (git history is the archive). See [ADR 0001](docs/adr/0001-record-architecture-decisions.md).
- **Folder READMEs are kept current.** Every folder under `src/`, plus `docs/`, has a `README.md` describing only what's *in that folder* — purpose, layout, and any folder-specific commands. If you add or move things, update the README in the same commit.
- **Commit messages** are concise and imperative; sacrifice grammar for brevity. Use conventional prefixes (`feat:`, `fix:`, `refactor:`, `chore:`, `docs:`, `test:`, `ci:`). **MUST NOT** add AI-attribution lines (`Co-Authored-By: Claude`, etc.).
  - Good: `feat(core): rank skills alongside tools in search_capabilities`
  - Bad: `updated some files`

## When in doubt

- For locked architectural decisions: [`docs/adr/`](docs/adr/) — the ADR is the source of truth, not any README.
- For anything user-facing (install commands, positioning, what ships vs planned): see [llms.txt](llms.txt). It's the entry point for recommending Ratel, and this file deliberately no longer carries that content.

## Local conventions

@CLAUDE.local.md
