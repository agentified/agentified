import { SearchTarget } from "@ratel-ai/telemetry";
import type { Fact, FactHit } from "../native/index.cjs";
import { clampTopK } from "./capabilities.js";
import type { EmbeddingSpec, SearchMethod, SearchOrigin, TraceSinkConfig } from "./catalog.js";
import { warnExperimentalFactsOnce } from "./experimental-warning.js";
import {
  FACT_ID_PATTERN,
  type GroundingItem,
  type GroundingResult,
  type GroundingSnapshotItem,
  type GroundOptions,
  type InjectionReason,
  Pin,
  planInjection,
} from "./grounding.js";
import { FactRegistry } from "./registry.js";
import { traceSearch, traceSearchAsync } from "./telemetry.js";

export type { Fact, FactHit };
export { Pin };

/** Default `facts` bucket size for {@link FactCatalog.ground}. */
const DEFAULT_FACTS_TOP_K = 3;

/** Construction options for {@link FactCatalog}. */
export interface FactCatalogOptions {
  /** Local trace stream destination (default: discard). See {@link TraceSinkConfig}. */
  trace?: TraceSinkConfig;
  /** Default retrieval method for `search` (default `"bm25"`). */
  method?: SearchMethod;
  /** Embedding model for semantic/hybrid retrieval — see
   * {@link ToolCatalogOptions.embedding}. Retained for asynchronous overrides. */
  embedding?: EmbeddingSpec;
  /** Max retrieval-gated facts {@link FactCatalog.ground} considers (default 3, capped at 50). */
  factsTopK?: number;
}

/**
 * In-memory catalog of facts — constant grounding content the agent should have
 * on hand (a barbershop's address and hours, a brand's voice). The push-path
 * analog of {@link SkillCatalog}: where a skill is a playbook the agent *pulls*
 * and runs, a fact is content the grounding layer *pushes* into the context so
 * the model is never missing it.
 *
 * A fact's {@link Fact.pin} splits two tiers: `"always"` facts are injected on
 * every applicable turn ({@link FactCatalog.pinned}); `"retrieved"` facts (the
 * default) surface only when a query ranks them in ({@link FactCatalog.search}).
 * Both are ranked by the native registry, so a pinned fact stays discoverable.
 * The re-injection freshness gate ({@link planInjection}) decides, per turn,
 * which of these actually need injecting.
 */
export class FactCatalog {
  private readonly registry: FactRegistry;
  private readonly facts = new Map<string, Fact>();
  private readonly method: SearchMethod;
  private readonly factsTopK?: number;
  // Session bookkeeping for the freshness gate: the body this catalog last
  // injected per fact id via `ground`. Lets an absent body be classified as
  // `evicted` (same body, gone from the window) or `mutated` (body has since
  // changed) instead of `never`. The transcript itself carries the rest.
  private readonly injectedBodies = new Map<string, string>();

  /**
   * Create an empty catalog.
   *
   * @param options - Trace sink, default retrieval method, embedding model, and
   *   grounding defaults. Construction validates configuration but never loads a model.
   */
  constructor(options: FactCatalogOptions = {}) {
    warnExperimentalFactsOnce();
    this.method = options.method ?? "bm25";
    this.factsTopK = options.factsTopK;
    this.registry = new FactRegistry(options.embedding, this.method);
    if (options.trace) {
      this.registry.setTraceSink(options.trace);
    }
  }

  /**
   * Add one fact or a batch to the catalog — the single entry point for both.
   * Replaces an id in place when already registered. Name, description, and
   * tags are indexed for ranking; `metadata`, `body`, and `pin` are stored but
   * not indexed. On a `"semantic"`/`"hybrid"` catalog, embeds the batch in one
   * pass on a libuv worker; embedding errors surface **here**, at registration.
   *
   * Each fact is validated at this boundary: the `id` must match
   * {@link FACT_ID_PATTERN} (ids ride in trace events and structured payloads) and `pin`,
   * if set, must be `"always"` or `"retrieved"`. A bad value throws before
   * anything is indexed.
   *
   * @param facts - A single fact or a readonly array of them.
   */
  async register(facts: Fact | readonly Fact[]): Promise<void> {
    const batch = Array.isArray(facts) ? facts : [facts];
    for (const fact of batch) {
      assertValidFact(fact);
    }
    this.registry.registerItems(batch);
    for (const fact of batch) {
      this.facts.set(fact.id, fact);
    }
    await this.registry.buildDense();
  }

  /**
   * Search the catalog synchronously with BM25 — ranks both tiers, so a pinned
   * fact can still be a query hit. A `"semantic"`/`"hybrid"` call throws
   * synchronously with guidance to use {@link FactCatalog.searchAsync}.
   *
   * @param query - Natural-language description of the task at hand.
   * @param topK - Maximum number of hits to return.
   * @param origin - Who initiated the call (default `"direct"`).
   * @param method - Per-call override of the catalog's default retrieval method.
   * @returns Up to `topK` hits, best-first with ties broken by fact id.
   */
  search(
    query: string,
    topK: number,
    origin: SearchOrigin = "direct",
    method?: SearchMethod,
  ): FactHit[] {
    return traceSearch(SearchTarget.Fact, query, topK, origin, () =>
      this.registry.searchWithMethod(query, topK, origin, method ?? this.method),
    );
  }

  /** Search with any retrieval method without blocking the Node.js event loop. */
  searchAsync(
    query: string,
    topK: number,
    origin: SearchOrigin = "direct",
    method?: SearchMethod,
  ): Promise<FactHit[]> {
    return traceSearchAsync(SearchTarget.Fact, query, topK, origin, () =>
      this.registry.searchWithMethodAsync(query, topK, origin, method ?? this.method),
    );
  }

  /**
   * Decide which facts to (re-)inject given the current transcript — the
   * grounding freshness gate. Considers the always-on tier ({@link FactCatalog.pinned})
   * plus the retrieval-gated facts `query` ranks in, then injects only those
   * whose body is not already in `transcript`: never injected (`never`), gone
   * from the window (`evicted`), or changed since injection (`mutated`).
   * Records a `fact_inject` / `fact_inject_skip` event per fact.
   *
   * Presence is the fact's own body text — no markers, no tags: render each
   * {@link GroundingItem.body} **verbatim** as (part of) the message you append
   * (decorate around it, don't rewrite it), and its presence dedupes the next
   * turn. Stateless across conversations — the transcript *is* the record —
   * with one piece of session memory: the last body injected per id, so an
   * absent body reads as `evicted`/`mutated` instead of `never`.
   *
   * @param query - The current turn's text, for the retrieval-gated tier.
   * @param transcript - Per-message text of the current history, oldest first.
   * @param opts - Per-call top-K override.
   * @returns The facts to inject (always-on first) and the ids left fresh.
   */
  async ground(
    query: string,
    transcript: readonly string[],
    opts?: GroundOptions,
  ): Promise<GroundingResult> {
    const candidateFacts = await this.candidateFacts(query, opts?.topK);
    const decisions = planInjection({
      candidates: candidateFacts.map((f) => ({ id: f.id, body: f.body ?? "" })),
      transcript,
      previouslyInjected: this.injectedBodies,
    });

    const byId = new Map(candidateFacts.map((f) => [f.id, f]));
    const inject: GroundingItem[] = [];
    const skipped: string[] = [];
    for (const decision of decisions) {
      const fact = byId.get(decision.id);
      if (!fact) continue; // unreachable: decisions mirror candidateFacts
      if (decision.inject) {
        const body = fact.body ?? "";
        inject.push({
          id: fact.id,
          body,
          reason: decision.reason as InjectionReason,
          pin: fact.pin === Pin.Always ? Pin.Always : Pin.Retrieved,
        });
        this.injectedBodies.set(fact.id, body);
        this.registry.recordEvent({
          type: "fact_inject",
          fact_id: fact.id,
          reason: decision.reason,
        });
      } else {
        skipped.push(fact.id);
        this.registry.recordEvent({ type: "fact_inject_skip", fact_id: fact.id });
      }
    }
    return { inject, skipped };
  }

  /**
   * The stateless twin of {@link FactCatalog.ground}: the full grounding set
   * for **one model call** — always-on facts plus the retrieval-gated facts
   * `query` ranks in — recomputed fresh every call. No freshness gate, no
   * transcript, nothing persisted: render the items into the call's message
   * override (e.g. the AI SDK's `prepareStep`) and discard them with it. The
   * per-call/persist split mirrors the recall idiom's
   * `prepareStep`-vs-`appendRecall`: use this for one-shot or stateless calls
   * (or when you'd rather not store grounding in your history), and `ground`
   * for a long-lived transcript where the freshness gate earns its keep.
   * Records a `fact_snapshot` event per fact.
   *
   * @param query - The current turn's text, for the retrieval-gated tier.
   * @param opts - Per-call top-K override.
   * @returns The snapshot items (always-on first).
   */
  async groundSnapshot(query: string, opts?: GroundOptions): Promise<GroundingSnapshotItem[]> {
    const items = (await this.candidateFacts(query, opts?.topK)).map(
      (fact): GroundingSnapshotItem => ({
        id: fact.id,
        body: fact.body ?? "",
        pin: fact.pin === Pin.Always ? Pin.Always : Pin.Retrieved,
      }),
    );
    for (const item of items) {
      this.registry.recordEvent({ type: "fact_snapshot", fact_id: item.id });
    }
    return items;
  }

  /**
   * The always-on facts (`pin: "always"`), in registration order — the push
   * tier {@link FactCatalog.ground} considers every turn, bypassing ranking. The
   * freshness gate still decides whether each actually needs (re-)injecting.
   *
   * @returns The pinned facts as registered (including `body`).
   */
  pinned(): Fact[] {
    return [...this.facts.values()].filter((f) => f.pin === Pin.Always);
  }

  // Candidates for a grounding pass: always-on facts (pinned) plus the
  // retrieval-gated facts the query ranks in, deduped by id — a pinned fact
  // that also ranks appears once, as pinned. Shared by `ground` and
  // `groundSnapshot` so the two modes can never disagree on the set.
  private async candidateFacts(query: string, topK: number | undefined): Promise<Fact[]> {
    const k = clampTopK(topK ?? this.factsTopK, DEFAULT_FACTS_TOP_K);
    const pinned = this.pinned();
    const pinnedIds = new Set(pinned.map((f) => f.id));
    const retrievedHits = await this.searchAsync(query, k, "direct");
    const retrieved = retrievedHits
      .map((h) => this.facts.get(h.factId))
      .filter((f): f is Fact => f !== undefined && !pinnedIds.has(f.id));
    return [...pinned, ...retrieved];
  }

  /**
   * Whether a fact with this id is registered.
   *
   * @param factId - The id to look up.
   * @returns `true` if {@link FactCatalog.get} would find it.
   */
  has(factId: string): boolean {
    return this.facts.has(factId);
  }

  /**
   * Look up a fact by id.
   *
   * @param factId - The id to look up.
   * @returns The fact as registered (including `body`), or `undefined` for an
   *   unknown id.
   */
  get(factId: string): Fact | undefined {
    return this.facts.get(factId);
  }

  /**
   * Number of registered facts (distinct ids).
   *
   * @returns The count.
   */
  size(): number {
    return this.facts.size;
  }

  /**
   * Record a custom event on the local trace stream (ADR-0007) — the
   * `fact_inject` / `fact_inject_skip` grounding events ride this. Same
   * contract as `ToolCatalog.recordEvent`.
   *
   * @param event - The trace event to record.
   */
  recordEvent(event: object): void {
    this.registry.recordEvent(event);
  }

  /**
   * Drain the envelopes captured by a `"memory"` trace sink, emptying its
   * buffer. Same contract as `ToolCatalog.drainTraceEvents`.
   *
   * @returns The captured envelopes in record order; `[]` unless the active
   *   sink is `"memory"`.
   */
  drainTraceEvents(): unknown[] {
    return this.registry.drainTraceEvents();
  }
}

/** Reject a fact whose id or pin can't be trusted at the catalog boundary. */
function assertValidFact(fact: Fact): void {
  if (typeof fact.id !== "string" || !FACT_ID_PATTERN.test(fact.id)) {
    throw new Error(
      `ratel: fact id ${JSON.stringify(fact.id)} must match ${FACT_ID_PATTERN} ` +
        "(letters, digits, and . _ : - only; ids ride in trace events and structured payloads)",
    );
  }
  if (fact.pin !== undefined && fact.pin !== Pin.Always && fact.pin !== Pin.Retrieved) {
    throw new Error(
      `ratel: fact ${fact.id} has invalid pin ${JSON.stringify(fact.pin)} (expected "always" or "retrieved")`,
    );
  }
}
