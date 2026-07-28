import { SearchTarget } from "@ratel-ai/telemetry";
import type { ReplaceOutcome, Skill, SkillHit } from "../native/index.cjs";
import type { EmbeddingSpec, SearchMethod, SearchOrigin, TraceSinkConfig } from "./catalog.js";
import { type IntentGraph, SkillRegistry } from "./registry.js";
import { traceSearch, traceSearchAsync, traceSkillLoad } from "./telemetry.js";

export type { ReplaceOutcome, Skill, SkillHit };

/**
 * What {@link SkillCatalog.replaceAll} returns: the {@link ReplaceOutcome}
 * counts, readable immediately, over a promise for the embedding pass.
 *
 * The corpus swap commits synchronously, so the counts are final the moment
 * `replaceAll` returns — read them without awaiting. Awaiting drives the
 * embedding pass and resolves to the same counts, or rejects if embedding
 * fails; a source that must report what a reload changed can therefore do so
 * even when the embedding pass failed.
 */
export type PendingReplace = Promise<ReplaceOutcome> & ReplaceOutcome;

/** Construction options for {@link SkillCatalog}. */
export interface SkillCatalogOptions {
  /** Local trace stream destination (default: discard). See {@link TraceSinkConfig}. */
  trace?: TraceSinkConfig;
  /** Default retrieval method for `search` (default `"bm25"`). */
  method?: SearchMethod;
  /** Embedding model for semantic/hybrid retrieval — see
   * {@link ToolCatalogOptions.embedding}. Retained for asynchronous overrides. */
  embedding?: EmbeddingSpec;
}

/**
 * In-memory catalog of skills, ranked by the native `SkillRegistry` retrieval
 * engine. The on-demand analog of {@link ToolCatalog}: registered skills are
 * searched by relevance; the matching body is fetched only on
 * {@link SkillCatalog.invoke}.
 */
export class SkillCatalog {
  private readonly registry: SkillRegistry;
  private readonly skills = new Map<string, Skill>();
  private readonly method: SearchMethod;

  /**
   * Create an empty catalog.
   *
   * @param options - Trace sink, default retrieval method, and embedding model.
   *   Construction validates configuration but never loads a model.
   */
  constructor(options: SkillCatalogOptions = {}) {
    this.method = options.method ?? "bm25";
    this.registry = new SkillRegistry(options.embedding, this.method);
    if (options.trace) {
      this.registry.setTraceSink(options.trace);
    }
  }

  /**
   * Add one skill or a batch to the catalog — the single entry point for
   * both. Replaces an id in place when already registered. Name,
   * description, and tags are indexed for ranking; `tools`, `metadata`, and
   * `body` are stored but not indexed (`body` is the dispatch payload,
   * fetched by {@link SkillCatalog.invoke}). On a `"semantic"`/`"hybrid"`
   * catalog, embeds the batch in one pass on a libuv worker after metadata is
   * indexed — embedding errors surface **here**, at registration. A
   * `"bm25"` catalog never loads a model.
   *
   * A model or dimension change is not recovered in place — construct a new
   * catalog and re-register.
   *
   * @param skills - A single skill or a readonly array of them. Pass the
   *   whole batch at once for a single embedding request.
   */
  async register(skills: Skill | readonly Skill[]): Promise<void> {
    const batch = Array.isArray(skills) ? skills : [skills];
    this.registry.registerItems(batch);
    for (const skill of batch) {
      this.skills.set(skill.id, skill);
    }
    await this.registry.buildDense();
  }

  /**
   * Make `skills` the entire content of the catalog: ids absent from the batch
   * are removed, the rest are added or updated. The whole-catalog counterpart
   * to {@link SkillCatalog.register}, for a source that reloads a catalog
   * rather than pushing individual changes. Mutates in place, so every holder
   * of this catalog — `ratel()`'s `r.skills`, each adapted view, and the
   * capability tools taken from `modelTools()` — observes the reload without
   * being rebuilt.
   *
   * **Removal is unconditional.** Skills registered in-process are dropped just
   * like any others, since the batch *is* the catalog. A host that keeps local
   * skills alongside a remote source composes the batch itself:
   * `await catalog.replaceAll([...localSkills, ...remoteSkills])`.
   *
   * Embedding follows the same two-phase contract as
   * {@link SkillCatalog.register}: the corpus swap lands first, then the batch
   * is embedded. Only genuinely new or re-worded skills are embedded — an
   * unchanged id keeps its vector, so reloading an unchanged catalog costs no
   * embedding calls at all. If the embedding pass fails, the new corpus is
   * already live and BM25 still ranks it; semantic search reports
   * `EmbeddingsNotBuilt` until a later pass succeeds.
   *
   * A concurrent operation refuses the call outright rather than applying it
   * half-way, so two overlapping reloads can never blend into one corpus.
   * Because the swap is the synchronous half, that refusal throws at the call
   * site rather than rejecting the returned promise. Note that any in-flight
   * {@link SkillCatalog.searchAsync} can trigger it, including a plain BM25 one
   * — `registry busy` is retryable and not exclusive to dense work.
   *
   * @param skills - The complete catalog contents. A repeated id keeps its last
   *   entry. An empty array clears the catalog.
   * @returns The counts, already final, over a promise for the embedding pass —
   *   see {@link PendingReplace}. Read `.added`/`.removed` without awaiting;
   *   `await` to drive the embedding pass.
   */
  replaceAll(skills: readonly Skill[]): PendingReplace {
    const outcome = this.registry.replaceAllItems(skills);
    this.skills.clear();
    for (const skill of skills) {
      this.skills.set(skill.id, skill);
    }
    // The counts ride on the promise itself, so a failed embedding pass still
    // reports what the (already committed) swap changed.
    const embedded = this.registry.buildDense().then(() => outcome);
    return Object.assign(embedded, outcome);
  }

  /**
   * Search the catalog synchronously with BM25. A `"semantic"`/`"hybrid"`
   * call throws synchronously with guidance to use
   * {@link SkillCatalog.searchAsync}.
   *
   * @param query - Natural-language description of the task at hand.
   * @param topK - Maximum number of hits to return.
   * @param origin - Who initiated the call (default `"direct"`); recorded on
   *   the trace event and span, never affects ranking.
   * @param method - Per-call override of the catalog's default retrieval method.
   * @returns Up to `topK` BM25 hits, best-first with ties broken by skill id.
   *   Semantic/dense/hybrid methods throw migration guidance; use
   *   {@link SkillCatalog.searchAsync} for those methods.
   */
  search(
    query: string,
    topK: number,
    origin: SearchOrigin = "direct",
    method?: SearchMethod,
  ): SkillHit[] {
    return traceSearch(SearchTarget.Skill, query, topK, origin, () =>
      this.registry.searchWithMethod(query, topK, origin, method ?? this.method),
    );
  }

  /** Search with any retrieval method without blocking the Node.js event loop. */
  searchAsync(
    query: string,
    topK: number,
    origin: SearchOrigin = "direct",
    method?: SearchMethod,
  ): Promise<SkillHit[]> {
    return traceSearchAsync(SearchTarget.Skill, query, topK, origin, () =>
      this.registry.searchWithMethodAsync(query, topK, origin, method ?? this.method),
    );
  }

  /**
   * Whether a skill with this id is registered.
   *
   * @param skillId - The id to look up.
   * @returns `true` if {@link SkillCatalog.invoke} would find it.
   */
  has(skillId: string): boolean {
    return this.skills.has(skillId);
  }

  /**
   * Look up a skill by id.
   *
   * @param skillId - The id to look up.
   * @returns The skill as registered (including `body`), or `undefined` for an
   *   unknown id.
   */
  get(skillId: string): Skill | undefined {
    return this.skills.get(skillId);
  }

  /**
   * Number of registered skills (distinct ids). `searchCapabilitiesTool` uses
   * this to decide whether to advertise a `skills` bucket at all.
   *
   * @returns The count.
   */
  size(): number {
    return this.skills.size;
  }

  /**
   * Record a custom event on the local trace stream (ADR-0007). Same contract
   * as `ToolCatalog.recordEvent`: the event is a tagged wire-shape object
   * (`{ type: "...", ... }`, snake_case fields) and an unknown shape throws.
   *
   * @param event - The trace event to record.
   */
  recordEvent(event: object): void {
    this.registry.recordEvent(event);
  }

  /**
   * Turn on adaptive usage ranking against `graph` (ADR-0014): the catalog
   * ranks against what users have actually invoked after similar queries, and
   * keeps learning as it is used.
   *
   * Pass the same {@link IntentGraph} to a {@link ToolCatalog} so both learn
   * into one set of clusters.
   *
   * Set `rebuildOnModelChange` to auto-recover a model-mismatched graph on the
   * next dense (semantic/hybrid) search rather than staying paused until you
   * call {@link experimentalRebuildIntentGraph} yourself. Off by default — the rebuild is an
   * embedding pass (cost, possible `EmbedderError`, and it mutates the graph).
   */
  experimentalEnableAdaptiveRanking(
    graph: IntentGraph,
    options: { warnOnModelMismatch?: boolean; rebuildOnModelChange?: boolean } = {},
  ): void {
    this.registry.experimentalEnableAdaptiveRanking(graph, options);
  }

  /**
   * Re-embed the intent graph's members under the current model and replace its
   * centroids — call after changing the embedding model. Preserves members,
   * support, and edges. See {@link experimentalEnableAdaptiveRanking}.
   */
  async experimentalRebuildIntentGraph(): Promise<void> {
    await this.registry.experimentalRebuildIntentGraph();
  }

  /** Whether adaptive usage ranking is active, inactive, or paused by a model
   * change — see the native `AdaptiveRankingStatus`. */
  get experimentalAdaptiveRankingStatus() {
    return this.registry.experimentalAdaptiveRankingStatus;
  }

  /** Turn adaptive usage ranking off; the graph keeps what it learned. */
  experimentalDisableAdaptiveRanking(): void {
    this.registry.experimentalDisableAdaptiveRanking();
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

  /**
   * Return a skill's body for dispatch, recording a `skill_invoke` event.
   * Throws on an unknown id — callers at the capability-tool boundary translate that
   * into a structured error for the agent.
   *
   * @param skillId - Id of a registered skill.
   * @returns The skill's `body` (`""` when it was registered without one).
   */
  invoke(skillId: string): string {
    const skill = this.skills.get(skillId);
    if (!skill) {
      throw new Error(`unknown skillId: ${skillId}`);
    }
    return traceSkillLoad(skillId, () => {
      const started = Date.now();
      const body = skill.body ?? "";
      this.registry.recordEvent({
        type: "skill_invoke",
        skill_id: skillId,
        took_ms: Date.now() - started,
      });
      return body;
    });
  }
}
