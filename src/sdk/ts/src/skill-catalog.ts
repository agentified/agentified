import { SearchTarget } from "@ratel-ai/telemetry";
import type { NativeEventSubscription, ReplaceOutcome, Skill, SkillHit } from "../native/index.cjs";
import { warmFromEmbeddingArtifactSource } from "./artifact-source-warm.js";
import type {
  EmbeddingSpec,
  ExperimentalBm25Params,
  ObservationPolicyOptions,
  SearchMethod,
  SearchOrigin,
  TraceSinkConfig,
} from "./catalog.js";
import {
  type DefinitionOverrideApplyOptions,
  withDefinitionOverride,
} from "./definition-overrides.js";
import {
  type ExperimentalEmbeddingArtifact,
  resolveEmbeddingArtifact,
} from "./embedding-artifact.js";
import { type IntentGraph, SkillRegistry } from "./registry.js";
import type { RuntimeEvent, RuntimeEventsOptions } from "./runtime-events.js";
import { traceSearch, traceSearchAsync, traceSkillLoad } from "./telemetry.js";

export type { ReplaceOutcome, Skill, SkillHit };

/**
 * What {@link SkillCatalog.replaceAll} returns: the {@link ReplaceOutcome}
 * counts, readable immediately, over a promise for dense preparation.
 *
 * The corpus swap commits synchronously, so the counts are final the moment
 * `replaceAll` returns and can be read before dense preparation settles — a
 * source that must report what a reload changed can therefore do so even when
 * that phase fails.
 *
 * **Always `await` (or `.catch()`) the value, even when you only want the
 * counts.** It is a real promise for dense preparation (artifact warm or
 * embedding), so leaving it unhandled turns a preparation failure into an
 * unhandled rejection — which terminates the process under Node's default
 * `--unhandled-rejections=throw`. Reading the counts is not a substitute for
 * handling it:
 *
 * ```ts
 * const reload = catalog.replaceAll(batch); // corpus live, counts final
 * try {
 *   await reload;
 * } catch {
 *   log.warn(`applied +${reload.added} -${reload.removed}, dense prep pending`);
 * }
 * ```
 */
export type PendingReplace = Promise<ReplaceOutcome> & ReplaceOutcome;

/** Serializable skill definition used by catalog snapshots (never the private body). */
export type SkillDefinition = Omit<Skill, "body">;

/** Construction options for {@link SkillCatalog}. */
export interface SkillCatalogOptions {
  /** Local trace stream destination (default: discard). See {@link TraceSinkConfig}. */
  trace?: TraceSinkConfig;
  /** Default retrieval method for `search` (default `"bm25"`). */
  method?: SearchMethod;
  /**
   * Share of the hybrid content score the dense (semantic) arm carries; BM25
   * takes the remainder. Default `0.7`. Read by `"hybrid"` only — the
   * single-arm methods have nothing to weigh.
   *
   * **Experimental.** The default suits catalogs of natural-language
   * descriptions, which is where it was measured (ADR-0024). A catalog keyed on
   * exact identifiers, error codes, or internal jargon gives the lexical arm
   * purchase those corpora do not have and wants a lower value. `0` is pure
   * lexical, `1` pure dense; anything outside `[0, 1]` throws rather than being
   * clamped, so a mistyped `70` is reported instead of silently searching at
   * `1`.
   *
   * It does not scale the adaptive-ranking arm, whose own share is a separate
   * guard (ADR-0014).
   */
  experimentalDenseWeight?: number;
  /**
   * BM25 `k1`/`b` override — see {@link ExperimentalBm25Params} for the
   * defaults, the evidence behind them, and when to reach for this.
   */
  experimentalBm25?: ExperimentalBm25Params;
  /** Embedding model for semantic/hybrid retrieval — see
   * {@link ToolCatalogOptions.embedding}. Retained for asynchronous overrides. */
  embedding?: EmbeddingSpec;
  /**
   * Build-time embedding artifact to warm on register/replaceAll (any method;
   * default `onMiss: "error"`). Each call re-resolves and re-warms over the
   * whole current corpus — intended for one batch at startup; incremental
   * register/replaceAll calls repeat I/O and id+hash matching.
   *
   * With the default fail-closed `onMiss: "error"`, warm fails when the
   * artifact is missing one or more ids from the catalog's current corpus.
   */
  experimentalEmbeddingArtifact?: ExperimentalEmbeddingArtifact;
}

/**
 * In-memory catalog of skills, ranked by the native `SkillRegistry` retrieval
 * engine. The on-demand analog of {@link ToolCatalog}: registered skills are
 * searched by relevance; the matching body is fetched only on
 * {@link SkillCatalog.invoke}.
 */
export class SkillCatalog {
  private readonly registry: SkillRegistry;
  private readonly localSkills = new Map<string, Skill>();
  private readonly skills = new Map<string, Skill>();
  private overrideSearchableDescriptions = new Map<string, string>();
  private readonly warnedShadowIds = new Set<string>();
  private readonly method: SearchMethod;
  private readonly embeddingArtifact: ExperimentalEmbeddingArtifact | undefined;

  /**
   * Create an empty catalog.
   *
   * @param options - Trace sink, default retrieval method, and embedding model.
   *   Construction validates configuration but never loads a model.
   */
  constructor(options: SkillCatalogOptions = {}) {
    this.method = options.method ?? "bm25";
    this.registry = new SkillRegistry(
      options.embedding,
      this.method,
      options.experimentalDenseWeight,
      options.experimentalBm25,
    );
    this.embeddingArtifact = options.experimentalEmbeddingArtifact;
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
   * catalog without an artifact, embeds after metadata. With
   * {@link SkillCatalogOptions.experimentalEmbeddingArtifact}, warms that
   * artifact first (any method). A `"bm25"` catalog without an artifact never
   * loads a model.
   *
   * A model or dimension change is not recovered in place — construct a new
   * catalog and re-register.
   *
   * @param skills - A single skill or a readonly array of them. Pass the
   *   whole batch at once for a single dense-preparation request.
   */
  async register(skills: Skill | readonly Skill[]): Promise<void> {
    const batch = Array.isArray(skills) ? skills : [skills];
    const localBatch = batch.map((skill) => structuredClone(skill));
    await this.registerEffective(
      localBatch.map((skill) => this.applyDefinitionOverride(skill)),
      localBatch,
    );
  }

  /** @internal Apply a complete definition overlay while retaining local definitions for restore. */
  async applyDefinitionOverrides(
    overrides: ReadonlyMap<string, string>,
    options: DefinitionOverrideApplyOptions = {},
  ): Promise<void> {
    const { adopt = true, emitDefinitions = true } = options;
    this.overrideSearchableDescriptions = new Map(overrides);
    if (adopt) this.registry.setUseDefinitionOverrides();
    await this.replaceAllEffective(
      [...this.localSkills.values()].map((skill) => this.applyDefinitionOverride(skill)),
      undefined,
      emitDefinitions,
    );
  }

  /** @internal Commit one-way definition-override ownership after a staged apply. */
  enableDefinitionOverrides(): void {
    this.registry.setUseDefinitionOverrides([...this.skills.values()]);
  }

  private async registerEffective(
    batch: readonly Skill[],
    localBatch?: readonly Skill[],
  ): Promise<void> {
    this.registry.registerItems(batch);
    if (localBatch) {
      for (const skill of localBatch) this.localSkills.set(skill.id, skill);
    }
    for (const skill of batch) {
      this.skills.set(skill.id, skill);
    }
    await this.ensureDenseReady();
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
   * Dense preparation follows the same two-phase contract as
   * {@link SkillCatalog.register}: corpus swap first, then artifact warm or
   * embed. Unchanged ids keep their vectors. If that phase fails, the new
   * corpus is already live and BM25 still ranks it; semantic search reports
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
   * @returns The counts, already final, over a promise for dense preparation —
   *   see {@link PendingReplace}. `.added`/`.removed` can be read before that
   *   phase settles, but the value must still always be awaited (or
   *   `.catch()`-ed), or a dense-preparation failure becomes an unhandled
   *   rejection.
   */
  replaceAll(skills: readonly Skill[]): PendingReplace {
    const localSkills = skills.map((skill) => structuredClone(skill));
    const localIds = new Set(localSkills.map((skill) => skill.id));
    for (const warnedId of this.warnedShadowIds) {
      if (!localIds.has(warnedId)) this.warnedShadowIds.delete(warnedId);
    }
    return this.replaceAllEffective(
      localSkills.map((skill) => this.applyDefinitionOverride(skill)),
      localSkills,
    );
  }

  private replaceAllEffective(
    skills: readonly Skill[],
    localSkills?: readonly Skill[],
    emitDefinitions = true,
  ): PendingReplace {
    const outcome = this.registry.replaceAllItems(skills, emitDefinitions);
    if (localSkills) {
      this.localSkills.clear();
      for (const skill of localSkills) this.localSkills.set(skill.id, skill);
    }
    this.skills.clear();
    for (const skill of skills) {
      this.skills.set(skill.id, skill);
    }
    // The counts ride on the promise itself, so a failed dense-preparation
    // phase still reports what the (already committed) swap changed.
    const embedded = this.ensureDenseReady().then(() => outcome);
    return Object.assign(embedded, outcome);
  }

  private applyDefinitionOverride(skill: Skill): Skill {
    return withDefinitionOverride(
      "skill",
      skill,
      this.overrideSearchableDescriptions,
      this.warnedShadowIds,
    );
  }

  private async ensureDenseReady(): Promise<void> {
    const artifact = this.embeddingArtifact;
    if (artifact) {
      await warmFromEmbeddingArtifactSource(this.registry, () =>
        resolveEmbeddingArtifact(artifact),
      );
      return;
    }
    await this.registry.buildDense();
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
    return traceSearch(SearchTarget.Skill, query, topK, origin, (projection) =>
      this.registry.searchWithMethod(query, topK, origin, method ?? this.method, projection),
    );
  }

  /** Search with any retrieval method without blocking the Node.js event loop. */
  searchAsync(
    query: string,
    topK: number,
    origin: SearchOrigin = "direct",
    method?: SearchMethod,
  ): Promise<SkillHit[]> {
    return traceSearchAsync(SearchTarget.Skill, query, topK, origin, (projection) =>
      this.registry.searchWithMethodAsync(query, topK, origin, method ?? this.method, projection),
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

  /** Complete, deterministic public skill definition set. */
  snapshot(): SkillDefinition[] {
    return [...this.localSkills.values()]
      .sort((left, right) => left.id.localeCompare(right.id))
      .map(({ body: _body, ...skill }) => structuredClone(skill));
  }

  /** @internal Attach one public runtime-event subscriber. */
  subscribeEvents(
    handler: (batch: RuntimeEvent[]) => void,
    options: Required<RuntimeEventsOptions>,
  ): NativeEventSubscription {
    return this.registry.subscribeEvents(handler, options);
  }

  /** @internal Enable experimental complete catalog-definition events. */
  experimentalEnableCatalogDefinitions(): void {
    this.registry.experimentalEnableCatalogDefinitions();
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
    options: {
      warnOnModelMismatch?: boolean;
      rebuildOnModelChange?: boolean;
    } & ObservationPolicyOptions = {},
  ): void {
    this.registry.experimentalEnableAdaptiveRanking(graph, options);
  }

  /**
   * Re-embed the intent graph's members under the current model and replace its
   * centroids — call after changing the embedding model. Preserves members,
   * support, and edges. See {@link experimentalEnableAdaptiveRanking}.
   *
   * Also the repair for an **over-merged** graph. Clustering compares a query
   * against a cluster's individual members, and those per-member vectors are
   * held in memory rather than persisted, so a graph loaded from storage — or
   * grown by an older version — matches on its centroid alone until a rebuild
   * refills them. A rebuild does not move cluster boundaries; replaying a trace
   * log, or relearning from scratch, is what re-clusters.
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
    return traceSkillLoad(skillId, (projection) => {
      const started = Date.now();
      const body = skill.body ?? "";
      this.registry.recordEvent(
        {
          type: "skill_invoke",
          skill_id: skillId,
          took_ms: Date.now() - started,
        },
        projection,
      );
      return body;
    });
  }
}
