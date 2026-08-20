# 15. Whole-catalog skill reload: `replaceAll` mutates the catalog in place

Date: 2026-07-27

## Status

Accepted

Implements the catalog-hydration half of [ADR-0003](0003-catalog-source-interface.md), which
accepted a source loader that "pulls a published catalog and hydrates the local registries" but
left the mutation primitive unspecified. ADR-0003 is unchanged and remains Accepted; this ADR
adds the missing primitive and fixes its semantics.

## Context

A catalog source — the managed cloud client first — reloads the full skill catalog periodically
and hands it to the SDK. ADR-0003 already settled that sync is a whole-catalog pull rather than
a delta protocol, so what arrives is always the complete set.

The SDK had no way to apply it. `SkillCatalog` was append-only: `register` replaces an id in
place, and nothing removes one. A reload could only ever grow the corpus, so a skill deleted
upstream stayed searchable forever — and an agent could still load its body through
`get_skill_content`.

Replacing the `SkillCatalog` *object* is not available as a fix. `ratel()` creates it once and
closes over it in `r.skills`, in each `adaptTo` view's `base.skills`, in every capability tool
returned by `modelTools()`, and in `recall()`; hosts hold `r.skills` directly as the documented
escape hatch. Rebinding the reference would strand all of them.

Two properties of the existing design make in-place mutation safe, and are the reason this can
be a small change rather than an architecture:

- **The model-facing payload is already pinned.** `ratel()` builds `search_capabilities` with
  `advertiseSkills: true`, so the tool description does not depend on whether the catalog is
  empty. A reload cannot perturb the prompt cache.
- **The registries already refuse a racing mutation.** A corpus swap is never observed torn and
  two overlapping reloads cannot interleave, because a mutation racing an in-flight operation is
  rejected rather than applied. Two details are worth stating precisely, because both are easy
  to get wrong:

  *Where the guard lives differs by surface.* The napi binding holds the registry in an
  `Arc<RwLock<_>>` and goes through `write_registry`, which checks the pending-dense counter and
  then takes the lock; the pyo3 binding owns its registry unlocked, and the equivalent check
  (`_raise_if_busy` against `_dense_pending`) lives in the Python facade above it, not in the
  native layer.

  *What contends is broader than dense work.* The pending-dense counter covers dense operations,
  but `write_registry` also fails when any in-flight `search_async` still holds the read lock —
  including a plain BM25 one, which takes no dense permit. So an ordinary read can refuse a
  reload, non-deterministically, depending on whether the worker has taken the lock yet. This is
  inherited, not introduced here: `register` contends the same way and did so before this ADR.
  It is called out because the retry guidance below depends on it — a source must treat
  `registry busy` as retryable on any traffic, not just dense traffic.

## Decision

**`SkillCatalog.replaceAll(skills)` / `replace_all(skills)`: the batch becomes the entire
catalog, applied in place.** Ids absent from the batch are removed; the rest are added or
updated. Object identity is preserved, so every existing holder observes the reload.

### Replace means replace

Removal is unconditional. Skills registered in-process are dropped like any others, because the
batch *is* the catalog. A host that wants local skills to survive a remote reload composes the
batch itself:

```ts
await r.skills.replaceAll([...localSkills, ...remoteSkills]);
```

Ownership policy therefore lives in the host, not in the core. The rejected alternative — an
owner tag per skill so a source only replaces its own partition — pushes the core into
arbitrating between sources for a need no caller has yet expressed, and can be added later
without changing this primitive.

### Two-phase, exactly like `register`

The corpus swap commits synchronously; the embedding pass is the returned awaitable. On failure
the new corpus is live and BM25 ranks it, while semantic search reports `EmbeddingsNotBuilt`
until a later pass succeeds. This is the contract `register` already has, so there is one rule
to learn rather than two.

The rejected alternative — staging the corpus, embedding, then committing both together — would
keep semantic search serving the *old* catalog through a failed reload. It was declined because
it requires the whole operation to move onto a worker task, which forfeits the property that a
forgotten `await` can never leave a half-applied change.

### The diff is the point

`replace_all` compares the incoming batch against the live corpus and touches the dense cache
only where it must: a removed id's vector is dropped, an id whose indexed text
(`name`/effective searchable description/`tags`) changed is invalidated, and everything else
keeps its vector — including an id whose `body`, `tools`, or `metadata` changed, since none of
those are embedded.
Reloading an unchanged catalog therefore costs **zero** embedding calls, which is the common
case for a polling source.

Dropping a removed id's vector is load-bearing rather than tidiness. The dense cache's
built-ness guard compares *counts*, so a stale vector left behind could offset a new,
unembedded id, let the guard pass, and make a semantic search silently omit that skill. That
interacts directly with the two-phase decision above: it is exactly the state a failed embedding
pass leaves behind.

Churn telemetry falls out of the same diff — `SkillChurn{add}` / `SkillChurn{remove}` for real
changes only, reusing the existing `ChurnKind::Remove` variant. A no-op reload emits nothing.

The call returns a `ReplaceOutcome` (`added` / `removed` / `updated` / `unchanged`) so a source
can report what a reload did without draining the trace stream. The counts ride *on* the returned
awaitable rather than through it: they are final once the swap commits, so they stay readable when
the embedding pass fails — which is precisely the state a source most needs to report on. Awaiting
still resolves to the same counts, or raises, exactly as `register` does.

### Skills only

`ToolCatalog` keeps no equivalent. A cloud-sourced tool carries no local executor, so
whole-catalog tool replacement is a separate question and is not answered here.

## Consequences

- A source can keep the catalog honest: upstream deletions actually disappear, from both
  retrieval and `get_skill_content`.
- Reload cost scales with what changed, not with catalog size.
- Hosts that mix in-process and sourced skills must compose the batch on every reload. This is
  the deliberate cost of keeping ownership out of the core, and it is visible in the method
  name.
- A reload that races an in-flight operation — dense work, but also an ordinary BM25
  `search_async` holding the read lock — raises rather than applying partially; a periodic source
  must tolerate that and retry on its next tick.
- The failure mode of a botched reload is degraded ranking (BM25 only), never an empty catalog —
  provided the source does not call `replaceAll([])` on a fetch error. Guarding that is the
  source's job; an empty batch is a legitimate way to clear the catalog.

## Rejected

- **Per-id `upsert`/`remove` as the primitive:** more surface and a diff burden on every source,
  to express what a whole-catalog pull already delivers in one shot. Nothing in ADR-0003's sync
  model produces per-id deltas.
- **Swapping the `SkillCatalog` object:** strands every closure and holder described above, and
  discards the dense cache wholesale on each reload.
- **An owner tag per skill so a source replaces only its own partition:** see above; deferred
  until a caller needs it.
