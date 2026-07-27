# `ratel-ai-telemetry` (Rust)

The `ratel.*` telemetry vocabulary as typed Rust constants: attribute keys, span
and Logs EventRecord names, and the value enums (`Origin`, `SearchTarget`, `AuthOutcome`).
Codifies the Tier 2 overlay of [`../CONVENTIONS.md`](../CONVENTIONS.md) so callers
emit the convention without stringly-typed keys, plus the pinned semconv version
(`SEMCONV_VERSION`) and the content-capture gate env var.

Emitting the vocabulary is done through the standard OpenTelemetry SDK; this crate
adds no transport, no FFI, no schema ([ADR-0007](../../../docs/adr/0007-telemetry-two-streams.md)).

## Library shape

- Crate name: `ratel-ai-telemetry`
- Library name: `ratel_ai_telemetry`
- No dependencies; released under the `telemetry-core-v*` tag prefix ([ADR-0008](../../../docs/adr/0008-release-engineering.md))
- MIT ([ADR-0009](../../../docs/adr/0009-licensing.md)); member of the root Cargo workspace

## Build & test

From the repo root:

```bash
cargo build  -p ratel-ai-telemetry
cargo test   -p ratel-ai-telemetry
cargo clippy -p ratel-ai-telemetry --all-targets -- -D warnings
```

The tests are the contract-against-the-pin conformance for the Rust helper: each
constant and enum value is asserted against the vocabulary pinned in
[`../CONVENTIONS.md`](../CONVENTIONS.md).
