import {
  type Fact,
  type FactHit,
  type EmbeddingConfig as NativeEmbeddingConfig,
  FactRegistry as NativeFactRegistry,
  SkillRegistry as NativeSkillRegistry,
  ToolRegistry as NativeToolRegistry,
  type SearchHit,
  type Skill,
  type SkillHit,
  type Tool,
} from "../native/index.cjs";
import type { EmbeddingSpec, SearchMethod, SearchOrigin, TraceSinkConfig } from "./catalog.js";
import { mapEmbedderError } from "./errors.js";

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
  ): SearchHit[] {
    return this.native.searchWithMethod(query, topK, origin, method);
  }

  /** Search on a libuv worker; supports `"bm25"`, `"semantic"`, and `"hybrid"`. */
  async searchWithMethodAsync(
    query: string,
    topK: number,
    origin: SearchOrigin,
    method: SearchMethod,
  ): Promise<SearchHit[]> {
    try {
      return await this.native.searchWithMethodAsync(query, topK, origin, method);
    } catch (error) {
      throw mapEmbedderError(error);
    }
  }

  /**
   * Record a custom event on the local trace stream (ADR-0007). Throws on an
   * object that doesn't parse as a known trace event.
   */
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

/**
 * Typed facade over the native skill registry — the skill twin of
 * {@link ToolRegistry}. {@link SkillCatalog} is the higher-level surface;
 * reach for this directly only when bare metadata is enough.
 */
export class SkillRegistry {
  private readonly native: NativeSkillRegistry;
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
    const items = Array.isArray(item) ? item : [item];
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
  ): SkillHit[] {
    return this.native.searchWithMethod(query, topK, origin, method);
  }

  /** Search on a libuv worker — see `ToolRegistry.searchWithMethodAsync`. */
  async searchWithMethodAsync(
    query: string,
    topK: number,
    origin: SearchOrigin,
    method: SearchMethod,
  ): Promise<SkillHit[]> {
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
   * {@link ToolRegistry.register} for the embed-inside contract. An unknown
   * `pin` value throws here, synchronously, before the batch is indexed.
   *
   * @param item - A single {@link Fact} or a readonly array of them.
   */
  async register(item: Fact | readonly Fact[]): Promise<void> {
    this.registerItems(item);
    await this.buildDense();
  }

  /**
   * Index metadata only, without embedding — see
   * {@link ToolRegistry.registerItems}.
   *
   * @internal
   */
  registerItems(item: Fact | readonly Fact[]): void {
    const items = Array.isArray(item) ? item : [item];
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
