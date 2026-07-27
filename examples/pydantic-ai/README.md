# `examples/pydantic-ai` — Ratel + Pydantic AI

The Python mirror of [`examples/ai-sdk`](../ai-sdk/README.md): the [`ratel-ai`](../../src/sdk/python/README.md) SDK wired into [Pydantic AI](https://ai.pydantic.dev/). Each run exposes the prompt's top-K `ToolCatalog` matches directly and keeps `search_capabilities` plus `invoke_tool` available for the rest of the catalog. See [Capability tools](https://docs.ratel.sh/docs/capability-tools) and [Framework integrations](https://docs.ratel.sh/docs/framework-integrations) for the protocol and reusable wiring pattern.

Tools are built from the catalog's JSON schemas via Pydantic AI's `Tool.from_schema`, so the schema the model sees is the same one Ratel ranks.

## Setup

```bash
export OPENAI_API_KEY=sk-...
uv run main.py
# or with a custom prompt:
uv run main.py "send an email to alice@example.com saying ship it"
```

`uv run` resolves `ratel-ai` from this monorepo (see `[tool.uv.sources]` in `pyproject.toml`) and `pydantic-ai` from PyPI, building the native extension on first run.

Without a model API key the script runs in **diagnostic mode** — prints the initial Ratel BM25 filter output and exits before any model call. Override the model with `RATEL_EXAMPLE_MODEL` (a Pydantic AI model id, e.g. `anthropic:claude-sonnet-4-6` or `openai:gpt-5-mini`); supported key env vars include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GROQ_API_KEY`.

## Layout

```
tools.py       catalog + helpers — six stub tools registered into a ToolCatalog
agent.py       run_agent — assembles the tool set (top-K + capability tools), runs the Pydantic AI Agent
main.py        entry — parse argv, diagnostic mode or model run, print
test_agent.py  model-free test of direct top-K tool invocation
```

Splitting `tools.py` and `agent.py` keeps the catalog declarative and the loop readable; nothing about the wiring is provider-specific (`run_agent` accepts any Pydantic AI model id).

## Why it's a separate package

Examples don't ship in `ratel-ai` — keeping them out of the published wheel keeps the public API surface narrow and OTel-SDK-free. This example pulls `pydantic-ai` only here.
