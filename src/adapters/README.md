# `src/adapters/`

Framework adapter packages: one complementary package per host framework that plugs into the `@ratel-ai/sdk` framework-adapter SPI (`ratel(config).adaptTo(adapter)`, [ADR-0013](../../docs/adr/0013-framework-adapter-spi.md)) so Ratel speaks that framework's native tool and message shapes. The core owns all state and guards; each adapter has three required codecs, optional extension hooks, and the framework's idioms.

Directories are language-prefixed (`ts-*`, `py-*`) since a framework gets one adapter per SDK language.

## Layout

- `ts-vercel-ai-sdk/` — [`@ratel-ai/vercel-ai-sdk`](ts-vercel-ai-sdk/README.md): the [Vercel AI SDK](https://sdk.vercel.ai) (`ai@^5 || ^6 || ^7`) adapter, plus `RatelOtelIntegration` on a separate `./otel` entrypoint (`ai@7` telemetry).
- `ts-mastra/` — [`@ratel-ai/mastra`](ts-mastra/README.md): the [Mastra](https://mastra.ai) (`@mastra/core`) adapter, plus experimental retrieval-experiment joins on its separate `./observability` entrypoint.

## Build & test

Each package is a member of the pnpm workspace; build and test it from the repo root:

```bash
pnpm --filter @ratel-ai/vercel-ai-sdk build
pnpm --filter @ratel-ai/vercel-ai-sdk typecheck
pnpm --filter @ratel-ai/vercel-ai-sdk lint
pnpm --filter @ratel-ai/vercel-ai-sdk test
```
