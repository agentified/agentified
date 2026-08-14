import {
  type AdaptiveRankingStatus,
  type FactHit,
  IntentGraph,
  type EmbeddingConfig as NativeEmbeddingConfig,
  type NativeEventSubscription,
  FactRegistry as NativeFactRegistry,
  SkillRegistry as NativeSkillRegistry,
  ToolRegistry as NativeToolRegistry,
  type ReplaceOutcome,
  type SearchHit,
  type Skill,
  type SkillHit,
  type Tool,
} from "../native/index.cjs";
import { assertNotArtifactBusy } from "./artifact-source-warm.js";
import type { EmbeddingSpec, SearchMethod, SearchOrigin, TraceSinkConfig } from "./catalog.js";
import { mapArtifactBuildError, mapArtifactWarmError, mapEmbedderError } from "./errors.js";
import { assertValidFact, type Fact } from "./grounding.js";
import type { RuntimeEvent, RuntimeEventsOptions } from "./runtime-events.js";
import type { RuntimeEventProjection } from "./telemetry.js";

export { IntentGraph };

/** Normalize the public string|object form into the native config the binding
 * expects (a string is the local-path `spec`, validated in core). */
function toNativeEmbedding(
  embedding: EmbeddingSpec | undefined,
): NativeEmbeddingConfig | undefined {
  if (embedding === undefined) return undefined;
  return typeof embedding === "string" ? { spec: embedding } : embedding;
}

/**
 * Typed facade over the native tool registry: metadata-only indexing and
 * retrieval, with the SDK's public embedding config and an async, batch-aware
 * `register`. {@link ToolCatalog} layers executors, OTel spans, and defaults
 * on top; reach for this directly only when bare metadata (no executors) is
 * enough.
 */
export class ToolRegistry {
  private readonly native: NativeToolRegistry;
  #warnOnModelMismatch = true;
  #adaptiveWarned = false;
  #rebuildOnModelChange = false;
  private readonly eager: boolean;

  /**
   * Create a registry with an optional embedding model and retrieval method.
   *
   * @param embedding - Embedding model for semantic/hybrid retrieval; a bare
   *   string is a local model directory path. Validated at construction,
   *   never loaded eagerly here.
   * @param method - `"bm25"` (default, model-free) or `"semantic"`/`"hybrid"`,
   *   which makes {@link ToolRegistry.register} embed the batch inline.
   */
  constructor(embedding?: EmbeddingSpec, method: SearchMethod = "bm25") {
    this.native = new NativeToolRegistry(toNativeEmbedding(embedding));
    this.eager = method === "semantic" || method === "hybrid";
  }

  /**
   * Register one tool or a batch, replacing any existing id in place — the
   * corpus never holds a duplicate. On a `"semantic"`/`"hybrid"` registry,
   * embeds the whole batch in one pass on a libuv worker after metadata is
   * indexed, so the event loop is never blocked; awaiting surfaces embedding
   * errors here. A `"bm25"` registry resolves as soon as metadata is indexed
   * and never loads a model.
   *
   * @param item - A single {@link Tool} or a readonly array of them.
   */
  async register(item: Tool | readonly Tool[]): Promise<void> {
    this.registerItems(item);
    await this.buildDense();
  }

  /**
   * Index metadata only, without embedding. Exposed so {@link ToolCatalog}
   * can interleave its own executor bookkeeping between metadata
   * registration and the (possibly failing) embedding pass — metadata
   * persists even if a later {@link ToolRegistry.buildDense} throws.
   *
   * @internal
   */
  registerItems(item: Tool | readonly Tool[]): void {
    assertNotArtifactBusy(this);
    const items = Array.isArray(item) ? item : [item];
    this.native.registerMany([...items]);
  }

  /**
   * Embed any not-yet-embedded items on a libuv worker when this registry
   * was constructed for `"semantic"`/`"hybrid"`; a no-op on `"bm25"`.
   *
   * @internal
   */
  async buildDense(): Promise<void> {
    if (!this.eager) return;
    try {
      await this.native.buildEmbeddings();
    } catch (error) {
      throw mapEmbedderError(error);
    }
    this.#maybeWarnModelMismatch();
  }

  /**
   * Build a single-kind binary embedding artifact from this registry's
   * registered corpus (ADR-0018). Embedding for the artifact is independent of
   * the mutable dense cache — it does not consume or update cached vectors.
   * Returns a Node `Buffer` suitable for `fs.writeFile`. Independent of the
   * registry's search method (works with `"bm25"` as well as
   * `"semantic"`/`"hybrid"`). For a mixed Tool+Skill artifact, use the
   * top-level {@link experimentalBuildEmbeddingArtifact} module helper instead.
   *
   * @throws {EmbedderError} When embedding or backend resolution fails.
   * @throws {ArtifactError} When artifact construction fails for a non-embedder reason.
   */
  async experimentalBuildEmbeddingArtifact(): Promise<Buffer> {
    try {
      return await this.native.buildEmbeddingArtifact();
    } catch (error) {
      throw mapArtifactBuildError(error);
    }
  }

  /**
   * Warm the dense cache from a build-time embedding artifact (ADR-0018).
   * `onMiss` is `"error"` (fail with {@link ArtifactWarmError} / `"Incomplete"`
   * when some corpus ids are uncovered) or `"embed"` (embed only the missing
   * ids). Independent of the registry's search method.
   */
  async experimentalWarmEmbeddingsFromArtifact(
    bytes: Buffer,
    onMiss: "error" | "embed",
  ): Promise<void> {
    try {
      await this.native.warmEmbeddingsFromArtifact(bytes, onMiss);
    } catch (error) {
      throw mapArtifactWarmError(error);
    }
    this.#maybeWarnModelMismatch();
  }

  /**
   * Lexical BM25 search: up to `topK` hits, best-first with ties broken by
   * id. Model-free and infallible; records the query on the local trace
   * stream with origin `"direct"`.
   */
  search(query: string, topK: number): SearchHit[] {
    return this.native.search(query, topK);
  }

  /** BM25 search with an explicit trace origin; ranking is unaffected. */
  searchWithOrigin(query: string, topK: number, origin: SearchOrigin): SearchHit[] {
    return this.native.searchWithOrigin(query, topK, origin);
  }

  /**
   * Synchronous search restricted to BM25; `"semantic"`/`"hybrid"` throw with
   * guidance to use {@link ToolRegistry.searchWithMethodAsync}.
   */
  searchWithMethod(
    query: string,
    topK: number,
    origin: SearchOrigin,
    method: SearchMethod,
    projection?: RuntimeEventProjection,
  ): SearchHit[] {
    return this.native.searchWithMethod(query, topK, origin, method, projection);
  }

  /** Search on a libuv worker; supports `"bm25"`, `"semantic"`, and `"hybrid"`. */
  async searchWithMethodAsync(
    query: string,
    topK: number,
    origin: SearchOrigin,
    method: SearchMethod,
    projection?: RuntimeEventProjection,
  ): Promise<SearchHit[]> {
    try {
      // Guard the await behind the flag so the default path stays synchronous
      // up to the native call — the pending-dense counter must increment before
      // control yields, or a following register() would slip past serialization.
      if (method !== "bm25" && this.#rebuildOnModelChange) {
        await this.#maybeRebuildOnModelChange();
      }
      return await this.native.searchWithMethodAsync(query, topK, origin, method, projection);
    } catch (error) {
      throw mapEmbedderError(error);
    }
  }

  /**
   * Record a custom event on the local trace stream (ADR-0007). Throws on an
   * object that doesn't parse as a known trace event.
   */
  recordEvent(event: object, projection?: RuntimeEventProjection): void {
    if (projection) {
      this.native.recordEventWithContext(event, projection);
    } else {
      this.native.recordEvent(event);
    }
  }

  /** Replace the trace sink; subsequent events go to the new destination. */
  setTraceSink(config: TraceSinkConfig): void {
    this.native.setTraceSink(config);
  }

  /** @internal Attach one public runtime-event subscriber. */
  subscribeEvents(
    handler: (batch: RuntimeEvent[]) => void,
    options: Required<RuntimeEventsOptions>,
  ): NativeEventSubscription {
    return this.native.subscribeTraceEvents(handler, options);
  }

  /**
   * Turn on adaptive usage ranking against `graph` (ADR-0014).
   *
   * Wires both halves: the registry ranks against the graph, and its trace sink
   * is decorated with a learner that grows it from search-then-invoke pairs — a
   * capability the user actually invoked after a query becomes evidence for
   * similar queries later.
   *
   * Pass the **same** {@link IntentGraph} to the tool and skill registries. One
   * cluster holds both a tool and a skill edge map, so sharing gives one set of
   * clusters with all the evidence behind it; separate graphs duplicate every
   * cluster and split the evidence.
   *
   * Only queries that match a cluster are affected — anything else ranks exactly
   * as it would have. With a graph attached, `SearchHit.score` becomes a fusion
   * score rather than a raw BM25 score, so use `rank` for ordering and
   * `fused` to detect the scale, not the raw `score`.
   */
  experimentalEnableAdaptiveRanking(
    graph: IntentGraph,
    options: { warnOnModelMismatch?: boolean; rebuildOnModelChange?: boolean } = {},
  ): void {
    this.#warnOnModelMismatch = options.warnOnModelMismatch ?? true;
    this.#rebuildOnModelChange = options.rebuildOnModelChange ?? false;
    this.#adaptiveWarned = false;
    this.native.enableAdaptiveRanking(graph);
    this.#maybeWarnModelMismatch();
  }

  /**
   * Re-embed the intent graph's members under the current embedding model and
   * replace its centroids. Call after changing the model: a graph's centroids
   * are only comparable to queries from the model that built them, so on a swap
   * the usage arm pauses until this runs. Members, support, and edges are
   * preserved — only the centroids move to the new space.
   */
  async experimentalRebuildIntentGraph(): Promise<void> {
    try {
      await this.native.rebuildIntentGraph();
    } catch (error) {
      throw mapEmbedderError(error);
    }
    this.#adaptiveWarned = false;
    this.#maybeWarnModelMismatch();
  }

  /**
   * Whether adaptive usage ranking is contributing (`"active"`), off
   * (`"inactive"`), not yet determinable (`"unknown"`), or paused because the
   * embedding model changed (`"paused: dim mismatch"` / `"paused: model
   * mismatch"`). Gate on this instead of reading stderr if you prefer.
   */
  get experimentalAdaptiveRankingStatus(): AdaptiveRankingStatus {
    return this.native.adaptiveRankingStatus();
  }

  /** One-time stderr warning when the attached graph's model no longer matches
   * the catalog's. A dev-time config error that otherwise silently pauses
   * ranking — printed unless `warnOnModelMismatch: false`. */
  #maybeWarnModelMismatch(): void {
    if (this.#adaptiveWarned || !this.#warnOnModelMismatch) return;
    const s = this.native.adaptiveRankingStatus();
    if (!s.status.startsWith("paused")) return;
    this.#adaptiveWarned = true;
    const how = s.dimMismatch
      ? `built with a ${s.built}-dim embedding model but the active model outputs ${s.active} dims`
      : `built with embedding model '${s.built}' but the active model is '${s.active}'`;
    console.warn(
      `ratel: intent graph was ${how}. Adaptive usage ranking is PAUSED — ` +
        `call experimentalRebuildIntentGraph() to rebuild it with the current model.`,
    );
  }

  /** Opt-in auto-recovery (see {@link experimentalEnableAdaptiveRanking}): when the arm is
   * paused because the graph's model no longer matches, re-embed the graph under
   * the current model before the dense search. Re-checks each search, so once
   * rebuilt it stops; a failed rebuild throws `EmbedderError`, exactly as the
   * dense query itself would. */
  async #maybeRebuildOnModelChange(): Promise<void> {
    if (!this.#rebuildOnModelChange) return;
    if (this.native.adaptiveRankingStatus().status.startsWith("paused")) {
      await this.experimentalRebuildIntentGraph();
    }
  }

  /**
   * Turn adaptive usage ranking off: ranking returns to the base engine and the
   * graph stops growing. The graph keeps what it learned, so re-enabling
   * resumes rather than restarts.
   */
  experimentalDisableAdaptiveRanking(): void {
    this.#rebuildOnModelChange = false;
    this.native.disableAdaptiveRanking();
  }

  /** Drain captured envelopes from a `"memory"` sink; `[]` otherwise. */
  drainTraceEvents(): unknown[] {
    return this.native.drainTraceEvents();
  }
}

/**
 * Typed facade over the native skill registry — the skill twin of
 * {@link ToolRegistry}. {@link SkillCatalog} is the higher-level surface;
 * reach for this directly only when bare metadata is enough.
 */
export class SkillRegistry {
  private readonly native: NativeSkillRegistry;
  #warnOnModelMismatch = true;
  #adaptiveWarned = false;
  #rebuildOnModelChange = false;
  private readonly eager: boolean;

  /**
   * Create a registry with an optional embedding model and retrieval method.
   *
   * @param embedding - Embedding model for semantic/hybrid retrieval — see
   *   {@link ToolRegistry.constructor}.
   * @param method - `"bm25"` (default, model-free) or `"semantic"`/`"hybrid"`,
   *   which makes {@link SkillRegistry.register} embed the batch inline.
   */
  constructor(embedding?: EmbeddingSpec, method: SearchMethod = "bm25") {
    this.native = new NativeSkillRegistry(toNativeEmbedding(embedding));
    this.eager = method === "semantic" || method === "hybrid";
  }

  /**
   * Register one skill or a batch, replacing any existing id in place — see
   * {@link ToolRegistry.register} for the embed-inside contract.
   *
   * @param item - A single {@link Skill} or a readonly array of them.
   */
  async register(item: Skill | readonly Skill[]): Promise<void> {
    this.registerItems(item);
    await this.buildDense();
  }

  /**
   * Index metadata only, without embedding — see
   * {@link ToolRegistry.registerItems}.
   *
   * @internal
   */
  registerItems(item: Skill | readonly Skill[]): void {
    assertNotArtifactBusy(this);
    const items = Array.isArray(item) ? item : [item];
    this.native.registerMany([...items]);
  }

  /**
   * Swap the whole corpus for `items` without embedding — the metadata half of
   * {@link SkillCatalog.replaceAll}, the sibling of
   * {@link SkillRegistry.registerItems}. Ids absent from `items` are removed
   * along with their cached embeddings; ids whose indexed text changed are
   * invalidated for re-embedding, and unchanged ids keep the vector they have,
   * so the following {@link SkillRegistry.buildDense} only embeds real changes.
   *
   * @internal
   */
  replaceAllItems(items: readonly Skill[]): ReplaceOutcome {
    assertNotArtifactBusy(this);
    return this.native.replaceAll([...items]);
  }

  /**
   * Embed any not-yet-embedded items — see {@link ToolRegistry.buildDense}.
   *
   * @internal
   */
  async buildDense(): Promise<void> {
    if (!this.eager) return;
    try {
      await this.native.buildEmbeddings();
    } catch (error) {
      throw mapEmbedderError(error);
    }
    this.#maybeWarnModelMismatch();
  }

  /**
   * Build a single-kind binary embedding artifact from this registry's
   * registered corpus — see {@link ToolRegistry.experimentalBuildEmbeddingArtifact}.
   *
   * @throws {EmbedderError} When embedding or backend resolution fails.
   * @throws {ArtifactError} When artifact construction fails for a non-embedder reason.
   */
  async experimentalBuildEmbeddingArtifact(): Promise<Buffer> {
    try {
      return await this.native.buildEmbeddingArtifact();
    } catch (error) {
      throw mapArtifactBuildError(error);
    }
  }

  /**
   * Warm the dense cache from a build-time embedding artifact — see
   * {@link ToolRegistry.experimentalWarmEmbeddingsFromArtifact}.
   */
  async experimentalWarmEmbeddingsFromArtifact(
    bytes: Buffer,
    onMiss: "error" | "embed",
  ): Promise<void> {
    try {
      await this.native.warmEmbeddingsFromArtifact(bytes, onMiss);
    } catch (error) {
      throw mapArtifactWarmError(error);
    }
    this.#maybeWarnModelMismatch();
  }

  /** Lexical BM25 search over skills — see `ToolRegistry.search`. */
  search(query: string, topK: number): SkillHit[] {
    return this.native.search(query, topK);
  }

  /** BM25 search with an explicit trace origin. */
  searchWithOrigin(query: string, topK: number, origin: SearchOrigin): SkillHit[] {
    return this.native.searchWithOrigin(query, topK, origin);
  }

  /** Synchronous search restricted to BM25 — see `ToolRegistry.searchWithMethod`. */
  searchWithMethod(
    query: string,
    topK: number,
    origin: SearchOrigin,
    method: SearchMethod,
    projection?: RuntimeEventProjection,
  ): SkillHit[] {
    return this.native.searchWithMethod(query, topK, origin, method, projection);
  }

  /** Search on a libuv worker — see `ToolRegistry.searchWithMethodAsync`. */
  async searchWithMethodAsync(
    query: string,
    topK: number,
    origin: SearchOrigin,
    method: SearchMethod,
    projection?: RuntimeEventProjection,
  ): Promise<SkillHit[]> {
    try {
      // Guard the await behind the flag so the default path stays synchronous
      // up to the native call — the pending-dense counter must increment before
      // control yields, or a following register() would slip past serialization.
      if (method !== "bm25" && this.#rebuildOnModelChange) {
        await this.#maybeRebuildOnModelChange();
      }
      return await this.native.searchWithMethodAsync(query, topK, origin, method, projection);
    } catch (error) {
      throw mapEmbedderError(error);
    }
  }

  /** Record a custom event on the local trace stream (ADR-0007). */
  recordEvent(event: object, projection?: RuntimeEventProjection): void {
    if (projection) {
      this.native.recordEventWithContext(event, projection);
    } else {
      this.native.recordEvent(event);
    }
  }

  /** Replace the trace sink; subsequent events go to the new destination. */
  setTraceSink(config: TraceSinkConfig): void {
    this.native.setTraceSink(config);
  }

  /** @internal Attach one public runtime-event subscriber. */
  subscribeEvents(
    handler: (batch: RuntimeEvent[]) => void,
    options: Required<RuntimeEventsOptions>,
  ): NativeEventSubscription {
    return this.native.subscribeTraceEvents(handler, options);
  }

  /**
   * Turn on adaptive usage ranking against `graph` (ADR-0014).
   *
   * Wires both halves: the registry ranks against the graph, and its trace sink
   * is decorated with a learner that grows it from search-then-invoke pairs — a
   * capability the user actually invoked after a query becomes evidence for
   * similar queries later.
   *
   * Pass the **same** {@link IntentGraph} to the tool and skill registries. One
   * cluster holds both a tool and a skill edge map, so sharing gives one set of
   * clusters with all the evidence behind it; separate graphs duplicate every
   * cluster and split the evidence.
   *
   * Only queries that match a cluster are affected — anything else ranks exactly
   * as it would have. With a graph attached, `SearchHit.score` becomes a fusion
   * score rather than a raw BM25 score, so use `rank` for ordering and
   * `fused` to detect the scale, not the raw `score`.
   */
  experimentalEnableAdaptiveRanking(
    graph: IntentGraph,
    options: { warnOnModelMismatch?: boolean; rebuildOnModelChange?: boolean } = {},
  ): void {
    this.#warnOnModelMismatch = options.warnOnModelMismatch ?? true;
    this.#rebuildOnModelChange = options.rebuildOnModelChange ?? false;
    this.#adaptiveWarned = false;
    this.native.enableAdaptiveRanking(graph);
    this.#maybeWarnModelMismatch();
  }

  /**
   * Re-embed the intent graph's members under the current embedding model and
   * replace its centroids. Call after changing the model: a graph's centroids
   * are only comparable to queries from the model that built them, so on a swap
   * the usage arm pauses until this runs. Members, support, and edges are
   * preserved — only the centroids move to the new space.
   */
  async experimentalRebuildIntentGraph(): Promise<void> {
    try {
      await this.native.rebuildIntentGraph();
    } catch (error) {
      throw mapEmbedderError(error);
    }
    this.#adaptiveWarned = false;
    this.#maybeWarnModelMismatch();
  }

  /**
   * Whether adaptive usage ranking is contributing (`"active"`), off
   * (`"inactive"`), not yet determinable (`"unknown"`), or paused because the
   * embedding model changed (`"paused: dim mismatch"` / `"paused: model
   * mismatch"`). Gate on this instead of reading stderr if you prefer.
   */
  get experimentalAdaptiveRankingStatus(): AdaptiveRankingStatus {
    return this.native.adaptiveRankingStatus();
  }

  /** One-time stderr warning when the attached graph's model no longer matches
   * the catalog's. A dev-time config error that otherwise silently pauses
   * ranking — printed unless `warnOnModelMismatch: false`. */
  #maybeWarnModelMismatch(): void {
    if (this.#adaptiveWarned || !this.#warnOnModelMismatch) return;
    const s = this.native.adaptiveRankingStatus();
    if (!s.status.startsWith("paused")) return;
    this.#adaptiveWarned = true;
    const how = s.dimMismatch
      ? `built with a ${s.built}-dim embedding model but the active model outputs ${s.active} dims`
      : `built with embedding model '${s.built}' but the active model is '${s.active}'`;
    console.warn(
      `ratel: intent graph was ${how}. Adaptive usage ranking is PAUSED — ` +
        `call experimentalRebuildIntentGraph() to rebuild it with the current model.`,
    );
  }

  /** Opt-in auto-recovery (see {@link experimentalEnableAdaptiveRanking}): when the arm is
   * paused because the graph's model no longer matches, re-embed the graph under
   * the current model before the dense search. Re-checks each search, so once
   * rebuilt it stops; a failed rebuild throws `EmbedderError`, exactly as the
   * dense query itself would. */
  async #maybeRebuildOnModelChange(): Promise<void> {
    if (!this.#rebuildOnModelChange) return;
    if (this.native.adaptiveRankingStatus().status.startsWith("paused")) {
      await this.experimentalRebuildIntentGraph();
    }
  }

  /**
   * Turn adaptive usage ranking off: ranking returns to the base engine and the
   * graph stops growing. The graph keeps what it learned, so re-enabling
   * resumes rather than restarts.
   */
  experimentalDisableAdaptiveRanking(): void {
    this.#rebuildOnModelChange = false;
    this.native.disableAdaptiveRanking();
  }

  /** Drain captured envelopes from a `"memory"` sink; `[]` otherwise. */
  drainTraceEvents(): unknown[] {
    return this.native.drainTraceEvents();
  }
}

/**
 * Typed facade over the native fact registry — the fact twin of
 * {@link SkillRegistry}. {@link FactCatalog} is the higher-level surface;
 * reach for this directly only when bare metadata is enough.
 */
export class FactRegistry {
  private readonly native: NativeFactRegistry;
  private readonly eager: boolean;

  /**
   * Create a registry with an optional embedding model and retrieval method.
   *
   * @param embedding - Embedding model for semantic/hybrid retrieval — see
   *   {@link ToolRegistry.constructor}.
   * @param method - `"bm25"` (default, model-free) or `"semantic"`/`"hybrid"`,
   *   which makes {@link FactRegistry.register} embed the batch inline.
   */
  constructor(embedding?: EmbeddingSpec, method: SearchMethod = "bm25") {
    this.native = new NativeFactRegistry(toNativeEmbedding(embedding));
    this.eager = method === "semantic" || method === "hybrid";
  }

  /**
   * Register one fact or a batch, replacing any existing id in place — see
   * {@link ToolRegistry.register} for the embed-inside contract. A malformed
   * `id` or an unknown `pin` throws before anything is indexed — the returned
   * promise rejects, since the method is `async`.
   *
   * @param item - A single {@link Fact} or a readonly array of them.
   */
  async register(item: Fact | readonly Fact[]): Promise<void> {
    this.registerItems(item);
    await this.buildDense();
  }

  /**
   * Index metadata only, without embedding — see
   * {@link ToolRegistry.registerItems}. Validates the whole batch first
   * ({@link assertValidFact}), so a bad fact anywhere in it leaves the registry
   * untouched rather than half-populated.
   *
   * @internal
   */
  registerItems(item: Fact | readonly Fact[]): void {
    const items = Array.isArray(item) ? item : [item];
    for (const fact of items) assertValidFact(fact);
    this.native.registerMany([...items]);
  }

  /**
   * Embed any not-yet-embedded items — see {@link ToolRegistry.buildDense}.
   *
   * @internal
   */
  async buildDense(): Promise<void> {
    if (!this.eager) return;
    try {
      await this.native.buildEmbeddings();
    } catch (error) {
      throw mapEmbedderError(error);
    }
  }

  /** Lexical BM25 search over facts — see `ToolRegistry.search`. */
  search(query: string, topK: number): FactHit[] {
    return this.native.search(query, topK);
  }

  /** BM25 search with an explicit trace origin. */
  searchWithOrigin(query: string, topK: number, origin: SearchOrigin): FactHit[] {
    return this.native.searchWithOrigin(query, topK, origin);
  }

  /** Synchronous search restricted to BM25 — see `ToolRegistry.searchWithMethod`. */
  searchWithMethod(
    query: string,
    topK: number,
    origin: SearchOrigin,
    method: SearchMethod,
  ): FactHit[] {
    return this.native.searchWithMethod(query, topK, origin, method);
  }

  /** Search on a libuv worker — see `ToolRegistry.searchWithMethodAsync`. */
  async searchWithMethodAsync(
    query: string,
    topK: number,
    origin: SearchOrigin,
    method: SearchMethod,
  ): Promise<FactHit[]> {
    try {
      return await this.native.searchWithMethodAsync(query, topK, origin, method);
    } catch (error) {
      throw mapEmbedderError(error);
    }
  }

  /** Record a custom event on the local trace stream (ADR-0007). */
  recordEvent(event: object): void {
    this.native.recordEvent(event);
  }

  /** Replace the trace sink; subsequent events go to the new destination. */
  setTraceSink(config: TraceSinkConfig): void {
    this.native.setTraceSink(config);
  }

  /** Drain captured envelopes from a `"memory"` sink; `[]` otherwise. */
  drainTraceEvents(): unknown[] {
    return this.native.drainTraceEvents();
  }
}
