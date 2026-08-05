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

A. collecting — scoring after each turn against held-out: "is the build broken today", "rotate the key", "read the file", "is CI green on my branch"

  turn  1  gh_run_list   clusters=1 support=1/3       obs=1  fromBaseline=1 coverage=1/4
  turn  2  gh_run_list   clusters=1 support=2/3       obs=2  fromBaseline=2 coverage=1/4
  turn  3  gh_run_list   clusters=1 support=3 (full)  obs=3  fromBaseline=3 coverage=1/4
  turn  4  vault_rotate  clusters=2 support=1/3       obs=4  fromBaseline=4 coverage=2/4
  turn  5  gh_run_list   clusters=2 support=4 (full)  obs=5  fromBaseline=5 coverage=2/4
  turn  6  vault_rotate  clusters=2 support=2/3       obs=6  fromBaseline=6 coverage=2/4
  turn  7  read_file     clusters=3 support=1/3       obs=7  fromBaseline=7 coverage=3/4
  turn  8  vault_rotate  clusters=3 support=3 (full)  obs=8  fromBaseline=8 coverage=3/4
  turn  9  read_file     clusters=3 support=2/3       obs=9  fromBaseline=9 coverage=3/4
  turn 10  gh_run_list   clusters=3 support=5 (full)  obs=10 fromBaseline=10 coverage=3/4

  log -> /tmp/ratel-baseline-XXXX/telemetry.jsonl

  coverage stops at 3/4: "is CI green on my branch" never matches. It
  means the same as a captured turn but shares no words with one, and
  this demo runs on BM25. Bridging that is what the dense tier is for —
  a real deployment wants method: "semantic".

B. built graph:
  "the build is broken"
    observations  : 5 (5 from this capture)
    invoked       : gh_run_list x5
    phrasings     : 5
  "rotate the signing key"
    observations  : 3 (3 from this capture)
    invoked       : vault_rotate x3
    phrasings     : 3
  "read a file from disk"
    observations  : 2 (2 from this capture)
    invoked       : read_file x2
    phrasings     : 2
  ranking status  : inactive

C. after seeding   : gh_run_list > docker_build
   ranking status  : active

persist with graph.toJson() — rev=10 marks what to save.
```

BM25 ranks `docker_build` first for *"why is the build broken"* on the token *build*. Ten turns across three intents say people reach for `gh_run_list` on build questions, and the seeded graph closes the gap — with no live learning in between.

The intents interleave, so you can watch clusters form and reach full strength at different points: the build cluster at turn 3, the key-rotation one at turn 8, and file reading still ramping when capture ends.

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
// Both knobs default to seeding, so the common call passes nothing.
const graph = await serving.experimentalBuildIntentGraph(readFileSync(logPath, "utf8"));
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

`experimentalBuildIntentGraph` takes the same two knobs everywhere; each defaults to reproducing live behavior.

| Option | Values | Default |
|---|---|---|
| `origins` | `any` \| `agent` \| `baseline` | `baseline` |
| `provenance` | `live` \| `seeded` | `seeded` |

Unknown values are rejected rather than silently defaulting: a policy is a deliberate configuration, and reading `"seedd"` as `"live"` would produce a graph with no provenance and no error.

## Watching it mature

`experimentalBuildIntentGraph` is a **pure function of (log, policy)** returning a detached graph, so you can rebuild from the log so far as often as you like while capture continues. The demo does exactly that — it scores after every captured turn, so the progression is visible inline rather than in a separate script. That makes the "when do we flip?" decision measurable rather than a guess.

| column | meaning |
|---|---|
| `clusters` | distinct intents found so far |
| `support` | observations behind the cluster **this turn landed in**, out of the 3 that reach full strength — below that the boost is scaled down proportionally |
| `obs` | confirmed observations across every cluster |
| `fromBaseline` | how many of those came from this capture rather than live traffic; after the flip it stays put while `obs` keeps growing |
| `coverage` | held-out queries that matched a cluster — **none of them a captured turn**. The one to gate on: the others rise whether or not the graph generalises, so a healthy-looking graph can still fire on none of your traffic |

**Gate on coverage.** Measured on a real embedding model against invented agent-style queries, a graph seeded from user turn text matched 9 of 13 — and the misses clustered entirely in one intent, where users described a *symptom* ("why is the build broken") while the agent searched by *action* ("list ci workflow runs"). Whether your traffic looks like that is not predictable from outside, and no other column tells you.

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
