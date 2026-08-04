# `examples/configurable-adaptive-ranking-ts` — seed adaptive ranking from a baseline capture

Shows the **seed-first** path for [adaptive usage ranking](../../docs/adr/0014-adaptive-usage-ranking.md): record what an agent invokes while Ratel serves nothing, build an intent graph from that log offline, inspect it, and only then switch ranking on. **No model or API key** — a pure-Ratel feature demo over BM25.

The plain adaptive-ranking demo ([`examples/adaptive-ranking-ts`](../adaptive-ranking-ts/README.md)) learns live, starting from an empty graph. This one starts from evidence Ratel had no hand in. The Python mirror is [`examples/configurable-adaptive-ranking-python`](../configurable-adaptive-ranking-python/README.md).

## Why seed first

Adaptive ranking learns from what the agent invokes. But once Ratel is ranking, what the agent invokes is partly Ratel's own doing — ADR-0014 concedes the loop: *"boosting used capabilities makes them more used."* And a fresh deployment gets no boost at all until enough pairs accumulate.

Capturing first fixes both. An agent choosing from its own full tool list, with Ratel nowhere in the ranking path, produces the cleanest evidence available — and enough of it to be useful on day one.

The catch: a graph is keyed on query text, and a run where nobody searches has no query. So the host records each turn's text alongside the invocations, and Ratel stays a recorder.

## Setup

```bash
pnpm install
pnpm -F @ratel-ai/example-configurable-adaptive-ranking start
```

Expected output — the graph matures as turns are captured, then the flip:

```
query: "why is the build broken"
  cold (BM25 only) : docker_build > gh_run_list
  ranking status   : inactive

A. collecting — Ratel records every invocation; the graph is scored after each

  turn 1  gh_run_list   clusters=1 support=1/3 obs=1 fromBaseline=1
  turn 2  gh_run_list   clusters=1 support=2/3 obs=2 fromBaseline=2
  turn 3  gh_run_list   clusters=1 support=3/3 obs=3 fromBaseline=3
  turn 4  vault_rotate  clusters=2 support=1/3 obs=4 fromBaseline=4

  log -> /tmp/ratel-baseline-XXXX/telemetry.jsonl

B. built graph:
  "why is the build broken"
    observations  : 3 (3 from this capture)
    invoked       : gh_run_list x3
    phrasings     : 3
  "rotate the signing key"
    observations  : 1 (1 from this capture)
    invoked       : vault_rotate x1
    phrasings     : 1
  ranking status  : inactive

C. after seeding   : gh_run_list > docker_build
   ranking status  : active

persist with graph.toJson() — rev=4 marks what to save.
```

BM25 ranks `docker_build` first for *"why is the build broken"* on the token *build*. Three real turns say people reach for `gh_run_list`, and the seeded graph closes the gap — with no live learning in between.

## The four phases

### A. Collect

Ratel is a tape recorder: no graph attached, no learner, no embedder, no search on the turn path. `ranking status: inactive` throughout.

```ts
const capture = await buildCatalog({ kind: "jsonl", sessionId: "session-1", path: logPath });

capture.experimentalRecordBaselineQuery(turn);                 // the turn's text
capture.recordEvent({ type: "invoke_start", tool_id: invoked, args_size_bytes: 0 });
```

Two rules:

- **Query before invokes.** Invocations attribute to the session's most recent query, so a call landing after the next turn's query is credited to the wrong question.
- **Every invocation counts.** There is no per-turn filter, because success is not observable from a trace. Seed from an agent you already trust.

### B. Initialize

```ts
const graph = await serving.experimentalInitializeIntentGraph(readFileSync(logPath, "utf8"), {
  origins: "baseline",    // only observed turns count, not Ratel's own searches
  provenance: "seeded",   // record where this evidence came from
});
```

Every distinct query is embedded up front, so clusters form at the **dense** tier — the same tier the live path uses. That is why this lives on the catalog: a model-free replay would cluster on word overlap, and `experimentalRebuildIntentGraph` cannot repair it later (it replaces centroids without revisiting cluster boundaries).

One call covers both catalogs — a log carrying tool *and* skill events fills both edge maps.

### C. Inspect

The returned graph is **detached**. Building never switches ranking on, so inspecting first is the default rather than something you opt into. The example prints each cluster's label, its observations, and which tools it remembers.

### D. Serve

```ts
serving.experimentalEnableAdaptiveRanking(graph);
```

From here the live learner keeps adding to the same graph. `support` grows while `seeded_support` stays put, so the gap between them tells you how much of each cluster still rests on the baseline versus what live traffic has since confirmed.

## Policy options

`experimentalInitializeIntentGraph` takes the same three knobs everywhere; each defaults to reproducing live behavior.

| Option | Values | Default |
|---|---|---|
| `origins` | `any` \| `direct` \| `agent` \| `baseline` | `any` |
| `confirmation` | `attempted` \| `succeeded` | `attempted` |
| `provenance` | `live` \| `seeded` | `live` |

`confirmation: "succeeded"` counts only tool calls that *completed*, so a wrong-tool call that failed on its arguments never becomes an edge — stricter evidence, worth it for a seeding pass.

Unknown values are rejected rather than silently defaulting: a policy is a deliberate configuration, and reading `"seedd"` as `"live"` would produce a graph with no provenance and no error.

## Watching it mature

`experimentalInitializeIntentGraph` is a **pure function of (log, policy)** returning a detached graph, so you can rebuild from the log so far as often as you like while capture continues. The demo does exactly that — it scores after every captured turn, so the progression is visible inline rather than in a separate script. That makes the "when do we flip?" decision measurable rather than a guess.

| column | meaning |
|---|---|
| `clusters` | distinct intents found so far |
| `support` | observations behind the cluster **this turn landed in**, out of the 3 that reach full strength — below that the boost is scaled down proportionally |
| `obs` | confirmed observations across every cluster |
| `fromBaseline` | how many of those came from this capture rather than live traffic; after the flip it stays put while `obs` keeps growing |

Treat them as a report for a person to read, not an auto-trigger.

### Rough edges

- **`SUPPORT_FULL` is not exposed.** The demo hardcodes `3`. The threshold is Ratel's, so you should not have to know it — a first-class readiness API is still to come.
- **Rebuilding is O(whole log).** Fine nightly or every few hundred turns; wasteful per turn. There is no incremental "add these envelopes" path yet.

## Caveats worth knowing

- **Every invocation is evidence, and the graph assumes it is good evidence.** Nothing in a trace says whether a turn went well, so nothing is filtered. This demo seeds from an agent that already performs well, which is the precondition the mode rests on.

  It matters more than the support ramp suggests. Edge weights inside a cluster set only their *order*, never their magnitude — so `gh_run_list x3` against `docker_build x1` is arm rank 0 against rank 1, a difference of `0.5/60` vs `0.5/61`. Measured on this catalog, adding a single wrong invocation of `docker_build` moves it from `0.016667` to `0.024863` and puts it back above `gh_run_list` at `0.024727`. A mistake that names the tool the base ranker already favours — the common case, since that is *why* the agent got it wrong — is close to free, and more good data does not dislodge it.
- **Turn text is not agent query text.** Members here are what a *user* wrote; after the flip, queries are what the *agent* writes when calling `search_capabilities`. The dense tier is what bridges that gap, so use `"semantic"` or `"hybrid"` for a real deployment — this demo runs on BM25 because near-repeat phrasings cluster without a model.
- **Tool ids must match.** An id recorded during capture that no longer exists in the serving catalog is dropped at rank time. The `usage_boost` trace event reports `dropped` so that shows up rather than looking like a coverage gap.

## Files

```
src/tools.ts   the catalog and the baseline turns with their success flags
src/index.ts   everything: collect + score, inspect, serve
```
