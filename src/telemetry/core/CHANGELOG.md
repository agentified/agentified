# Changelog

All notable changes to `ratel-ai-telemetry` (the Rust telemetry constants crate) are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this package adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] - 2026-08-17

### Added

- `Origin::Baseline` (`baseline`) mirrors the new local-trace origin. `ratel.origin` now takes `direct | agent | baseline`.

### Changed

- **BREAKING:** `Origin` is now `#[non_exhaustive]`, matching `ratel-ai-core`'s.

## [0.2.0] - 2026-08-11

### Added

- `SearchTarget::Fact` — the third `ratel.search.target` value, beside `Tool` and `Skill`, for the fact retrieval the SDKs' grounding path emits. The vocabulary spec (`CONVENTIONS.md`) and the shared conformance fixtures widened with it; `SEMCONV_VERSION` is unchanged.

### Changed

- **BREAKING (0.x minor):** `SearchTarget` is not `#[non_exhaustive]`, so a downstream `match` over it that has no wildcard arm must add one for `Fact`. Nothing else about the type changed.

## [0.1.1] - 2026-07-26

### Added

- `GEN_AI_SYSTEM_INSTRUCTIONS`, `GEN_AI_INPUT_MESSAGES`, `GEN_AI_OUTPUT_MESSAGES`, and
  `RATEL_TOOL_EXECUTION_DETAILS` EventRecord constants.

### Changed

- Clarify that content events use the OpenTelemetry Logs Event API and that inference output messages require `finish_reason`.
- `RATEL_ORIGIN` is now specified for third-party `gen_ai.*` spans that a framework adapter overlays, not just Ratel's own search/invoke spans. The constant, its `direct | agent` values, and `SEMCONV_VERSION` are unchanged; only the shared vocabulary spec widened (`CONVENTIONS.md`).

## [0.1.0] - 2026-07-06

### Added

- The telemetry vocabulary (ADR-0015): the full `ratel.*` constants (attribute keys, span/event names, `gen_ai.*` interop keys, and the `Origin`/`SearchTarget`/`AuthOutcome` value enums) pinned to OpenTelemetry semconv `gen_ai` v1.42.0, as zero-dependency `&str` constants and enums.
- The `contract-against-the-pin` test suite: every constant is asserted against its pinned wire key, so the vocabulary cannot drift from `CONVENTIONS.md`.

### Changed

- Released as an independent crates.io unit under the `telemetry-core-v*` tag prefix.

This crate is constants-only; the TS and Python helpers ship `init()`.
