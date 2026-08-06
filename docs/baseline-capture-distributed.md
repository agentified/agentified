# Baseline capture in a distributed host

How to collect the evidence for [adaptive usage ranking](adr/0014-adaptive-usage-ranking.md) when your agent runs as a **process-per-request server across many instances** — the shape where the search and the invocation that follows are different requests, on possibly different machines.

The single-process path is the demo: [`examples/configurable-adaptive-ranking-ts`](../examples/configurable-adaptive-ranking-ts/README.md). Read that first — it explains *why* to seed before you rank. This page covers only what changes when one process cannot see a whole turn.

## The setup

Ratel serves **no retrieval**. Whatever ranks your capabilities today keeps ranking them; the SDK is loaded purely as a recorder. That is the condition seeding exists for — evidence Ratel had no hand in producing.

Collection is close to free. A recorder needs no registered tools, no embedder, and no graph: `experimentalRecordBaselineTurn` is sugar over `recordEvent`, nothing embeds, and `experimentalAdaptiveRankingStatus` reads `inactive` throughout.

```ts
const recorder = new ToolCatalog({
  trace: {
    kind: "callback",
    sessionId: turnId,
    onEvent: (line) => queueForPersistence(line),
  },
});
```

## Two things the single-process API assumes

### 1. One process owns the turn

`experimentalBaselineTurn(query).invoked(id).record()` holds its buffer in a closure. It cannot span requests. Reassemble the turn from your own storage — stash the query when the search happens, replay it when the invocation arrives — and hand it over whole:

```ts
catalog.experimentalRecordBaselineTurn({
  query: storedQuery,
  invoked: [operationId],
});
```

Do **not** record one turn per invocation. The graph counts one observation per turn, so a search followed by three invocations recorded as three turns counts the query three times. Support is what scales the boost and gates the flip, so inflating it corrupts the one number you are trying to read.

The atomic call also cannot interleave: it emits its events back to back, where the builder lets you `await` between `invoked()` calls and let another turn's events land in the middle, breaking the search-then-invoke adjacency the graph pairs on.

### 2. The SDK owns the destination

`"noop"`, `"memory"`, and `"jsonl"` are all wrong for this deployment — ephemeral disk, split across instances, gone on recycle. The `"callback"` sink hands you each envelope as a JSON line and gets out of the way.

The line is **byte-identical to what the `"jsonl"` sink would have written**. That is the contract worth relying on: collect lines from every instance into a table, select the ones you want, `join("\n")`, and pass the result straight to `experimentalBuildIntentGraph` — no re-derivation, no wire form of your own.

Two properties to design around:

- **Delivery is asynchronous.** Recording queues the line and returns; the callback runs on a later turn of the event loop, with no ordering guarantee against your own microtasks or `setImmediate`. Flush before a request handler resolves or a process exits, or you lose the tail of a capture.
- **Delivery is lossy.** Per [ADR-0007](adr/0007-telemetry-two-streams.md) a trace sink may drop events under backpressure but must never block the agent loop, so a queue that cannot keep up drops silently. Treat a capture window as best-effort, and keep `onEvent` cheap — enqueue, don't await.

## Session ids are yours to assign

Replay pairs a search with the invokes that follow it **per session** — deliberately, so interleaved sessions in one log cannot cross-pair. The sink's `sessionId` is therefore a *default, not an identity*: it is fixed when the sink is built, which a process-wide catalog serving concurrent requests cannot make unique.

So either build the sink per turn, or — simpler — restamp `session_id` on each line when you re-serialize from your storage. Any id works as long as **concurrent turns never share one**. A per-turn id is safest; a per-client/actor/workspace unit id is usually enough and groups more naturally for analysis.

## What to record, and what that evidence is worth

Record every turn you would be willing to learn from; nothing reaches the log unless you call the method, so the quality gate is simply whether you call it.

The harder question is what your evidence *means*, and it depends on what sits upstream of the invocation:

- **The agent chose from its own tool list.** The cleanest evidence there is, and what the demo models.
- **A ranker returned a shortlist and the agent picked from it.** Common, and weaker than it looks: you are recording that ranker's judgement, not the agent's. Seeding on it teaches Ratel to reproduce a ranking that already exists — including its mistakes.

If a ranker is upstream, log enough alongside each turn to separate the two later — whether a model arbitrated, and whether the agent took the top-ranked option or overrode it. Turns where the agent disagreed with its ranker are the most informative ones you will collect, and they are indistinguishable afterwards if you did not record the distinction at the time.

Capturing everything and filtering at build time is the right default. Filtering happens in **your** query, before you serialize — `experimentalBuildIntentGraph`'s `origins` / `provenance` options select by trace origin, not by anything only you know.

## Building and inspecting

Unchanged from the single-process path:

```ts
const graph = await serving.experimentalBuildIntentGraph(lines.join("\n"));
```

The build is a pure function of (log, policy) and returns a **detached** graph, so rebuilding as often as you like costs nothing but time, and nothing switches on until you pass it to `experimentalEnableAdaptiveRanking`. Rebuilding is O(whole log) — fine nightly, wasteful per turn.

Gate on **coverage**: held-out queries that match a cluster, where the held-out set is real traffic that was never captured. The demo's README explains why the other numbers rise whether or not the graph generalises.

## Caveats that carry over

Every one of the demo's caveats still applies — most importantly that **nothing in a trace says whether a turn went well**, so every recorded invocation is treated as good evidence. Seed from an agent you already trust.

One caveat is *reduced* here: if your query is what an agent wrote to a search tool, it is drawn from the same distribution as the queries the graph will later be matched against — which is not true when the query is a user's turn text.

One is *added*: a graph's cluster members are raw query strings. Collecting them from real traffic into a durable store, and later into a graph blob, moves user text somewhere new. Settle that before the first capture, not after.
