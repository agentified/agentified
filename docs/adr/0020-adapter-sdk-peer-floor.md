# 20. Adapter `@ratel-ai/sdk` peer is a 0.x floor range, not a caret of the current minor

Date: 2026-08-14

## Status

Accepted

Builds on [ADR-0008](0008-release-engineering.md) and [ADR-0013](0013-framework-adapter-spi.md).

## Context

`@ratel-ai/vercel-ai-sdk` and `@ratel-ai/mastra` declare `@ratel-ai/sdk` as `workspace:^` in
source. At publish, `release.yml` used to rewrite that to `'^' + <in-repo SDK version>`. On
`0.x`, a caret pins the **minor**: `^0.6.0` means `>=0.6.0 <0.7.0`. Every SDK minor therefore
orphaned both adapters until they were re-released (`ERESOLVE`).

The floor must match current adapter source. A one-off measurement against
`vercel-ai-sdk@0.4.0-rc.2` / `mastra@0.3.0-rc.1` found vercel shipped code (`src/aisdk.ts`)
needs `ExperimentalPassthroughToolExposure` / `experimentalExposePassthrough`, first exported
in SDK 0.9.1 (94/94 tests). Mastra passed 52/52 on 0.9.1; its shipped code still compiles at
0.6.0. Publishing `>=0.6.0` would be a false compatibility claim: npm would resolve, then
vercel would fail on a missing export.

## Decision

**Publish the adapter SDK peer as a floor range, never a caret** — a lower bound that moves
with each adapter release, and a `<1.0.0` upper bound that does not.

| Half | Where it lives | Does it change? |
|---|---|---|
| floor | `SDK_ADAPTER_PEER_FLOOR` in `scripts/release-units.mjs` | **Yes** — bump it to the SDK the adapters are built against when cutting an adapter release. Never lower it. |
| ceiling | literal `<1.0.0`, same file | **No** — this mechanism is 0.x-only and is deleted at SDK 1.0.0 rather than raised. |

As of this ADR's date that resolves to `>=0.9.1 <1.0.0`. **Every concrete range in this ADR is
a snapshot, not a constant** — read the live value from `SDK_ADAPTER_PEER_FLOOR`.
Source manifests stay `workspace:^`; `scripts/pin-adapter-sdk-peer.mjs` applies the range at
publish.

**This whole mechanism is 0.x-only. Delete it when the SDK reaches 1.0.0** — at `>=1.0.0` a
caret already means `>=1.2.0 <2.0.0`, which is the range this ADR spells out by hand. Do not
carry the floor forward into 1.x: the ceiling is a literal `<1.0.0`, so a 1.x floor would
publish the empty range `>=1.2.0 <1.0.0`. The tripwire is
`scripts/pin-adapter-sdk-peer.test.mjs`, which asserts the produced range verbatim and fails
on any floor change.

The floor is true by construction only while it matches what the adapters actually build
against; the `<1.0.0` ceiling is the additive-evolution policy (AGENTS.md).

The adapter SDK peer **no longer names the in-repo SDK version**, so the old
`npm view @ratel-ai/sdk@^${sdk_ver}` publish-order guard is gone. Telemetry still pins to
`'^' +` the in-repo telemetry version; those `npm view` guards stay.

## Consequences

- Adapter minors no longer ship in lockstep with every SDK 0.x minor.
- Lowering the floor requires making vercel's passthrough SPI usage optional first.
- Narrowing the published peer is breaking under the 0.x rule in CONTRIBUTING.md (it drops
  SDK 0.6–0.8 support), so the release carrying it takes a MINOR bump.
- After an adapter GA publish, move the npm `rc` dist-tag onto that GA version. OIDC Trusted
  Publishing does not cover `npm dist-tag`, so it is a manual post-publish step; left alone,
  `@rc` keeps serving a prerelease whose SDK peer is the old caret.
