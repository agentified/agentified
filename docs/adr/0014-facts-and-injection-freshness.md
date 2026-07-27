# 14. Facts: constant grounding content with a content-presence re-injection gate

Date: 2026-07-27

## Status

Accepted

Builds on first-class skills (ADR-0005), the selectable retrieval engines (ADR-0004,
ADR-0011), the two telemetry streams (ADR-0007), and the framework-adapter SPI (ADR-0013):
facts reuse the skill registry pattern wholesale and surface through the same `recall`/adapter
seam.

## Context

Everything Ratel puts in the context window is **pull-based**: a query ranks the tool and skill
catalogs and only the winners are surfaced (`search_capabilities`, `recall`). That is the right
default for large catalogs where a fraction is relevant per turn — but it has no home for the
opposite need. A customer running a barbershop agent has a handful of *constant* facts the model
must always work from: the shop's address, its opening hours, the brand's voice. These are not
"discovered by relevance" — the address must never fail to be present because the user's message
didn't lexically match "address." They also want to *manage* these like skills: register them,
see which are used, edit them when the shop moves.

Two forces shape the design:

1. **Constant information spans two tiers.** A few facts are genuinely always-on (address,
   hours). Many more are constant-but-conditional (per-service pricing, cancellation policy,
   each barber's specialty) — injecting all of them every turn is exactly the token waste and
   noise Ratel exists to remove. So "constant info" is not one thing; it is an always-on tier
   plus a retrieval-gated tier.

2. **A transcript message persists.** Once a fact's body is injected as a message it stays in the
   history on every later turn. Naively re-appending an always-on fact each turn duplicates it N
   times — pure token waste, and the model reads the repetition as significant. The question
   "should I inject this fact now?" therefore has a real answer beyond "always": inject only when
   it isn't already fresh in the context.

The naive framing — "a second recall that also searches constant info" — is wrong for tier 1: it
re-gates the always-on facts behind relevance, reinventing the very thing that already exists.

## Decision

**Introduce `Fact` as a third registered primitive, parallel to `Tool` and `Skill`, split into an
always-on and a retrieval-gated tier by a `pin` mode; and gate its injection with a stateless
content-presence check so a fact is (re-)injected only when it is absent from the window or has
changed — never merely because a turn elapsed.**

### Facts as a third registry

A `Fact` is `{ id, name, description, tags, metadata, body, pin }` — a `Skill` minus `tools`, plus
`pin: "always" | "retrieved"` (default `retrieved`). `name`/`description`/`tags` are indexed
(both tiers are query-rankable, so a pinned fact stays discoverable); `body` is the injected
content, never indexed. It gets its own `FactRegistry` in `ratel-ai-core` — a near-clone of
`SkillRegistry` (BM25/semantic/hybrid, replace-in-place, `IndexMap` insertion order) — and its own
`FactCatalog` / native binding in each SDK. Per ADR-0007's "a parallel type lets each capability's
telemetry stand on its own," facts emit their own `fact_search` / `fact_churn` / `fact_inject` /
`fact_inject_skip` / `fact_snapshot` events rather than borrowing the skill stream.

- **Pinned (`always`)** facts are the push tier: injected every applicable turn, bypassing
  ranking. Kept small — they are paid for on every injection.
- **Retrieved** (the default) facts are ranked like skills and surfaced only when a query pulls
  them in. They ride the **host-driven** `recall()` path as a third bucket in
  `runCapabilitiesSearch`, alongside `tools` and `skills`, with the `body` inline (facts are
  small — no second `get_*_content` round-trip). Two boundaries hold: the search is skipped
  entirely when the fact catalog is empty, so a host that registered no facts sees zero new
  behavior (no extra search, no extra `ratel.search` span); and the **model-facing
  `search_capabilities` tool is unchanged in both SDKs** — facts never enter that stable tool's
  schema while they are experimental.

### The re-injection freshness gate: content presence

Injection is a pure decision over three cases, computed by `planInjection`:

| Trigger | Meaning | Detection |
|---|---|---|
| `never` | not in the window, never injected this session | body absent + id unseen |
| `evicted` | injected earlier, now gone (trimmed/compacted) | body absent + same body previously injected |
| `mutated` | body edited since injection | current body absent + differs from the one last injected |

The presence signal is **the fact's own body text**: a fact is "fresh" (skipped) when its body
appears verbatim — a literal substring check, no regex, no parsing — **within a single transcript
message**. The scan is per message, not over a joined transcript, because injections render the
whole body into one message; joining would false-positive when two unrelated messages happen to
end/start with the halves of a multi-line body. There is no marker, no tag, no extra token: the
injected content is its own record. Content presence is who-put-it-there agnostic: if the
assistant echoed the fact verbatim, or a summarizing compaction preserved it, the information *is*
in context and skipping is correct.

The gate stays in-process with no conversation store: compaction dropping the text naturally
re-arms injection, and an edited body (no longer found verbatim) naturally re-injects the new
version. The only session state is a small map of last-injected body per id, used solely to
classify an absent body as `evicted`/`mutated` rather than `never`. The one contract on hosts:
render `body` **verbatim** in the appended message — decorate around it, never rewrite it.

#### Rejected alternatives

- **A transcript marker** (`⟦ratel:fact id=… v=hash⟧` beside each injected body). Rejected: for
  short facts the ~20-token marker rivals the fact itself, it exposes odd glyphs to the model, and
  it answers the wrong question ("did *we* tag this?") instead of the right one ("is this
  information in the window?") — mis-handling exactly the assistant-echo and verbatim-summary
  cases content presence gets right.
- **A `stale` verdict** ("present but buried too far back," with a tunable freshness window).
  Rejected: the window has no principled value, and re-injecting on an append-only transcript
  duplicates the buried copy rather than moving it. Recency is `groundSnapshot`'s job — it places
  facts near the end of every call by construction.

### Two injection modes: `ground` vs `groundSnapshot`

Grounding ships in two modes, deliberately mirroring the recall idiom's persist-vs-per-call split
(`appendRecall` vs `prepareStep` in the AI SDK adapter) so the SDK has one injection philosophy,
not two:

- **`ground(query, transcript)`** — the *persist* mode above: facts enter the durable history, the
  presence gate dedupes across turns, and the stable prefix accrues prompt-cache credit. For
  long-lived multi-turn agents that store their messages.
- **`groundSnapshot(query)`** — the *per-call* mode: the full grounding set (always-on plus
  query-ranked facts) recomputed fresh each call, **no gate, no state, no transcript argument** —
  rendered into a per-call message override and discarded with it. For one-shot or stateless calls,
  or hosts that keep synthetic content out of their stored history. Traced per fact as
  `fact_snapshot`.

Both modes assemble candidates identically (one shared code path), so they can never disagree on
*which* facts apply — only on how the injection lives.

## Consequences

- **The always-on tier becomes cache-friendly, not cache-hostile.** Skipping re-injection keeps
  the transcript prefix stable across turns, so the feature *improves* prompt-cache hit rates
  instead of churning them. "Pay every turn" becomes "pay once per lifetime-in-window."
- **Facts reuse ~everything from skills.** The registry, native bindings, catalog facade, and
  telemetry plumbing are mechanical parallels — low new surface, and the retrieval behavior is
  identical and already proven.
- **The planner is a reusable primitive.** `planInjection` is pure and framework-agnostic;
  adapters render, the core decides. Nothing about the gate is fact-specific in principle — it is
  injection-deduplication that could later apply to any pushed content.
- **Known limits, accepted.** (1) The gate depends on hosts rendering `body` verbatim — a host
  that rewrites the injected text (translation, aggressive reformatting) defeats detection and the
  fact re-injects each turn (safe, just wasteful). (2) Across process restarts the injected-body
  map is empty, so an absent body reads as `never` rather than `evicted`/`mutated` — the action
  (inject) is identical either way; only the telemetry reason coarsens. (3) A summarizing
  compaction that *rewords* (rather than preserves) a fact's text causes a redundant re-inject;
  one that preserves it verbatim is correctly detected as still present.
  (4) Fact ids are constrained to `[A-Za-z0-9._:-]+`; they ride in trace events and structured
  injection payloads — validated at the catalog boundary.
- **Deciding the tier is the author's job.** The `pin` flag — an enum (`Pin.Always` /
  `Pin.ALWAYS`, wire strings still accepted) — is the whole UX; new facts default to `retrieved`
  and are promoted to `always` deliberately. A size cap on the always-on tier is left to a
  follow-up.

## Rollout: behind an `experimental` namespace

Facts ship **experimental** — a new, unproven API we want to trial in the open before committing to
stability. The whole surface (catalog, registry, planner, `Pin`, the grounding types) is reachable
only through an `experimental` namespace (`experimental.FactCatalog` in TS via `export * as
experimental`; `ratel_ai.experimental` in Python), never the root export, so any dependence on it is
explicit at the import site. Constructing a `FactCatalog` logs a one-time warning (silence:
`RATEL_EXPERIMENTAL_SILENCE`). The `ratel()` touchpoints that can't move off the stable object
(`r.facts`, `r.ground`, `r.groundSnapshot`, `RatelConfig.factsTopK`, the recall `facts` bucket)
are tagged "⚠️ Experimental" in their docs. The freshness gate lives on `FactCatalog.ground` (it
owns the fact state); `r.ground`/`r.groundSnapshot` are thin delegates. Adapter ergonomics — an
`appendGrounding` and facts riding `prepareStep` — are deliberately deferred to graduation, so
experimental API never enters the released adapter packages. Graduation is non-breaking: add the
stable root export, keep the `experimental.*` alias as a deprecated shim, and measure adoption in
between.
