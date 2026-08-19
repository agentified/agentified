# `src/sdk/`

Language SDKs that wrap `ratel-ai-core` so agents written in their respective host languages can drop the context engineering platform in with one dependency.

Each SDK bundles the core (binding strategy varies per language; see the relevant ADR) and exposes an idiomatic API for that ecosystem.

## Layout

```
ts/        @ratel-ai/sdk — TypeScript SDK
python/    ratel-ai — Python SDK
```

## `ts/` — `@ratel-ai/sdk`

The TypeScript SDK. Part of the pnpm workspace; bundles `ratel-ai-core` via a NAPI-RS native binding under [`ts/native/`](ts/native/README.md). See [`ts/README.md`](ts/README.md) for usage.

Its definition-overlay seam follows register-up/attach-down ownership: local registration syncs
definitions up, while externally-authored Retrieval descriptions apply down only with
`useDefinitionOverrides: true`. See the TypeScript SDK README for the attach example and shadowing
rules.

Binding strategy and tool-injection mode are documented in [ADR 0006](../../docs/adr/0006-native-ffi-bindings.md) and [ADR 0004](../../docs/adr/0004-retrieval-and-tool-selection.md).

## `python/` — `ratel-ai`

The Python SDK. Bundles `ratel-ai-core` via a [PyO3](https://pyo3.rs) native binding under [`python/native/`](python/native/README.md), distributed as prebuilt `abi3` wheels. Full feature parity with the TS SDK. See [`python/README.md`](python/README.md) for usage.

Binding strategy is documented in [ADR 0006](../../docs/adr/0006-native-ffi-bindings.md).
