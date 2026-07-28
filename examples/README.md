# `examples/`

Runnable demos of Ratel wired into real agent frameworks and protocol surfaces. Each entry is a private workspace package (not published).

## Layout

```
adaptive-ranking-ts/     Ratel adaptive usage ranking — learn from invocations, persist/reload the graph (TS)
adaptive-ranking-python/ Ratel adaptive usage ranking — the Python mirror, plus model-swap recovery
ai-sdk/           Ratel + Vercel AI SDK — top-K filtering + capability tools in ToolLoopAgent.generate
mastra/           Ratel + Mastra — capability tools + per-turn recall processor via @ratel-ai/mastra
mcp-chat/         Interactive REPL against an MCP-backed agent (Vercel AI SDK + OpenAI)
pydantic-ai/      Ratel + Pydantic AI (Python) — top-K filtering + capability tools in the agent loop
telemetry-ts/     Ratel telemetry — emit ratel.* spans via the OpenTelemetry JS SDK
telemetry-python/ Ratel telemetry — emit ratel.* spans via the OpenTelemetry Python SDK
```

## Conventions

- TypeScript examples are private pnpm workspace packages; Python examples are uv projects with checked-in lockfiles.
- Keep framework dependencies in the example, not in a published Ratel package.
- Provide one end-to-end entry point: a `start` script for pnpm packages or an `uv run` command for Python projects.
- Each example's README documents the exact env vars (model API keys) and the wiring pattern it illustrates.
