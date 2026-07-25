/**
 * `@ratel-ai/vercel-ai-sdk` — the Vercel AI SDK adapter for Ratel. Implements
 * the `@ratel-ai/sdk` framework-adapter SPI (ADR-0013) for `ai@^5 || ^6 || ^7`, so
 * `ratel(config).adaptTo(aiSdk())` speaks the AI SDK's native `Tool` and
 * `ModelMessage` shapes and adds the SDK-idiomatic per-turn recall helpers.
 *
 * The `ai@7` telemetry integration lives at `@ratel-ai/vercel-ai-sdk/otel`, not
 * here: it needs `@ai-sdk/otel`, which pins its own `ai@7`. Re-exporting it from
 * this entrypoint would drag that second `ai` copy into every `ai@5`/`ai@6`
 * host's type graph.
 *
 * @packageDocumentation
 */

export { type AiSdkExt, aiSdk } from "./aisdk.js";
