# Contributing to Ratel

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md) v2.1. By participating, you're agreeing to abide by it. Report incidents to `conduct@ratel.sh`.

## Prerequisites

- **Rust** stable — pinned via `rust-toolchain.toml`; rustup picks it up automatically.
- **Node** 24+
- **pnpm** 10.28+ (see `packageManager` in `package.json`)

## Build & test

```bash
# Rust
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check --all
cargo test --workspace

# TS
pnpm install
pnpm -r build
pnpm -r typecheck
pnpm -r lint
pnpm -r test
```

CI (`.github/workflows/{rust,ts}.yml`) runs all of the above on every PR. PRs are expected to land green.

## Branching

- `main` — stable
- `revamp` — v1 line, current development target until cutover
- Branch off `revamp` for new work; PR back into `revamp`

## Backend before frontend

For features that aren't frontend-only, implement on the Rust core first (with tests), then surface in the TS SDK.

## TDD

Backend / business-logic / library code: TDD is the default — write the failing test first, make it pass, refactor. Frontend changes don't require TDD unless they encode business logic.

## Additive, non-breaking evolution

Avoid breaking changes wherever avoidable. Ship new capabilities as additive, clearly-named experimental surfaces — an `experimentalSomething` function/method or a feature flag (e.g. `experimentalAsyncBuildEmbeddings`) — alongside the stable path, so existing callers keep working while the new behavior is proven out. Once it's proven, promote it to stable (drop the `experimental` prefix) or remove it; don't leave `experimental*` surfaces to rot.

## Architecture decisions

For cross-cutting choices, write an ADR in `docs/adr/` — Nygard format (`Status` / `Context` / `Decision` / `Consequences`), next available number, kebab-cased title. The set is kept minimal and current: amend in place for small drift (paths, names, counts), supersede for real decision reversals, compact periodically (git history is the archive). See [ADR 0001](docs/adr/0001-record-architecture-decisions.md) for the full convention.

## Commit messages

- Concise, imperative mood; sacrifice grammar for brevity
- Conventional-commits-ish prefixes (`feat:`, `fix:`, `chore:`, `docs:`, `refactor:`, `ci:`) where useful — the prefix picks the CHANGELOG section (`feat`→Added, `fix`→Fixed, `perf`/`refactor`→Changed; `docs`/`chore`/`ci`/`build`/`test` are skipped). See Releases
- A scope (`feat(core):`, `fix(sdk):`) is cosmetic — it's shown in the entry but does not decide which unit's CHANGELOG a commit lands in. That routing is by the files a commit touches (git-cliff scopes each unit to its own paths), so keep a commit within one unit's tree where practical
- No tool-attribution lines (no `Co-Authored-By` for AI assistants)

## Releases

Independently-versioned units publish from this repo (ADR-0008): `ratel-ai-core` (crates.io, `core-v*`), `@ratel-ai/sdk` + its per-OS packages (npm, `sdk-ts-v*`), `ratel-ai` (PyPI, `sdk-py-v*`), the three telemetry units — `ratel-ai-telemetry` (crates.io, `telemetry-core-v*`), `@ratel-ai/telemetry` (npm, `telemetry-ts-v*`), and `ratel-ai-telemetry` (PyPI, `telemetry-py-v*`) — and the two framework adapters, `@ratel-ai/vercel-ai-sdk` (npm, `vercel-ai-sdk-v*`) and `@ratel-ai/mastra` (npm, `mastra-v*`). Eight in total; [`scripts/release-units.mjs`](scripts/release-units.mjs) is the single source of truth. Each has a `CHANGELOG.md` in its package directory and its own tag prefix. `@ratel-ai/mcp-server` publishes independently from [ratel-ai/ratel-mcp](https://github.com/ratel-ai/ratel-mcp).

Two rules the unit list doesn't show:

- **Publish in dependency order.** `@ratel-ai/telemetry` before `@ratel-ai/sdk`, and `@ratel-ai/sdk` before either adapter. `workspace:^` specifiers are rewritten to a concrete `^X.Y.Z` at pack time, so a tag cut out of order publishes an immutable version whose dependency is not on the registry yet. The publish jobs check the pinned range against npm and fail before publishing rather than after.
- **At 0.x, a breaking change takes a MINOR bump** (`0.4.1` → `0.5.0`), never a major and never a patch. Removing or narrowing a public surface is what makes a change breaking; adding to it is not, however large the addition.

To cut a release — one unit at a time; see [RELEASING.md](RELEASING.md) for the full flow:

1. `node scripts/releasable.mjs` to see which units have unreleased commits.
2. Bump that unit's version in its manifest(s) — the release workflow validates the tag against every manifest the unit owns.
3. Run the `/changelog` skill (`.claude/skills/changelog/`) for the unit. It drafts entries with [git-cliff](https://git-cliff.org), lets you curate, and writes the unit's CHANGELOG; for GA versions it collapses the unit's `## [X.Y.Z-rc.*]` sections into a single `## [X.Y.Z]`.
4. Commit the bump + CHANGELOG together (`release: <unit>-vX.Y.Z`), tag `<unit>-vX.Y.Z`, push.

The release workflow's `tag-version-check` job rejects any tag whose unit CHANGELOG lacks a `## [<version>]` heading. See [ADR 0008](docs/adr/0008-release-engineering.md) (per-unit versions, tags, and CHANGELOG gates) for the rationale.

## Pull requests

- Keep PRs focused — one logical change per PR
- Update `README.md` / `CONTRIBUTING.md` in the same PR if the change affects them
- Tag `@claude` to invoke automated review on GitHub if useful

## License

The engine (`ratel-ai-core`) is licensed under [Apache-2.0](LICENSE-APACHE); every other component (SDKs, telemetry helpers, examples) is [MIT](LICENSE.md) — see [ADR-0009](docs/adr/0009-licensing.md). By submitting a PR you agree your contribution is licensed under the terms governing the component it touches.
