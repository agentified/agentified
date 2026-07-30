import type { JSONSchema7 } from "json-schema";
import type { SearchHit, Tool } from "../native/index.cjs";
import {
  clampTopK,
  DEFAULT_TOP_K_TOOLS,
  INVOKE_TOOL_ID,
  invokeToolTool,
  runCapabilitiesSearch,
  SEARCH_CAPABILITIES_ID,
  type SearchCapabilitiesResult,
  searchCapabilitiesTool,
} from "./capabilities.js";
import {
  type EmbeddingSpec,
  type ExecutableTool,
  type InputValidator,
  runToolInvocation,
  type SearchMethod,
  ToolCatalog,
  type TraceSinkConfig,
} from "./catalog.js";
import type { ExperimentalEmbeddingArtifact } from "./embedding-artifact.js";
import { FactCatalog } from "./fact-catalog.js";
import type { GroundingResult, GroundingSnapshotItem, GroundOptions } from "./grounding.js";
import { isPackageInstalled } from "./package-resolution.js";
import { SkillCatalog } from "./skill-catalog.js";
import { GET_SKILL_CONTENT_ID, getSkillContentTool } from "./skill-tools.js";

/** Construction options for {@link ratel}. Shared by every adapter view of the core. */
export interface RatelConfig {
  /** Default retrieval method for the tool, skill, and fact catalogs (default `"bm25"`).
   * BM25 search is model-free, but an explicitly configured embedding artifact is
   * still warmed during Tool/Skill registration. */
  method?: SearchMethod;
  /** Embedding model backing `"semantic"`/`"hybrid"` retrieval, forwarded to the tool, skill, and fact catalogs
   * — see {@link ToolCatalogOptions.embedding}. A string is a local model
   * directory path; every other source is a keyed object (`{ huggingface }`,
   * `{ ollama }`, `{ url, model, apiKeyEnv }`). Omit to use the built-in default
   * model. `await r.tools.register(...)` awaits dense preparation (artifact warm
   * or embedding) and rejects if it fails, so errors surface at registration. */
  embedding?: EmbeddingSpec;
  /**
   * Build-time embedding artifact forwarded to the Tool and Skill catalogs (default `onMiss: "error"`).
   *
   * Under the default fail-closed policy, the shared artifact must cover every
   * non-empty Tool/Skill corpus that actually registers. A tool-only artifact is
   * valid when the Skill corpus stays empty.
   *
   * Remedies:
   * - Build one mixed artifact with {@link experimentalBuildEmbeddingArtifact}
   *   so it covers all expected Tool + Skill corpora.
   * - Or set `onMiss: "embed"` to infer uncovered current-kind entries at runtime.
   */
  experimentalEmbeddingArtifact?: ExperimentalEmbeddingArtifact;
  /** Max tools each host-driven `recall` returns: capped at 50; 0, negative, or
   * non-integer values fall back to the default 5. */
  recallTopK?: number;
  /** ⚠️ Experimental (facts, ADR-0017). Max retrieval-gated facts each
   * `ground`/`groundSnapshot` considers: capped at 50; invalid values fall back to 3. */
  factsTopK?: number;
  /** Local trace-stream destination for all catalogs (default: discard). */
  trace?: TraceSinkConfig;
}

/**
 * A framework tool ingested into the catalog: what an adapter's
 * {@link RatelAdapter.ingest} returns for an executable tool. The core registers
 * it verbatim (id/name are the app's tool key), so the schemas are the SDK's
 * public {@link JSONSchema7} spelling and adapters need no casts.
 */
export interface CatalogRegistration {
  /** Retrieval ranks on this; resolve dynamic descriptions at ingest time. */
  description: string;
  /** Input JSON Schema (the catalog's native spelling). */
  inputSchema: JSONSchema7;
  /** Output JSON Schema; defaults to `{ type: "object" }` when omitted. */
  outputSchema?: JSONSchema7;
  /**
   * Framework-native parser retained in the shared catalog. Capability-tool
   * exposure delegates to this live parser, so every adapted view observes
   * native replacements and root-level transformations.
   */
  validateInput?: InputValidator;
  /** Runs the tool through the capability funnel with args and optional opaque
   * adapter context. An adapter that supports live framework context tags it in
   * `expose` and validates that private tag here before unwrapping it; a missing
   * or foreign tag must take the framework's context-free fallback. */
  execute(input: unknown, context?: unknown): Promise<unknown> | unknown;
}

/** The identity of one synthetic recall call, handed to {@link RatelAdapter.recallMessages}. */
export interface RecallRef {
  /** Unique call id from the core's private counter (never a transcript position). */
  callId: string;
  /** The recall query (the last user turn's text, in the AI SDK adapter). */
  query: string;
}

/**
 * Core-owned invocation funnel supplied when an adapter exposes a passthrough.
 * Wrappers retain the framework tool's native execution contract while the SDK
 * records the same OTel span and local trace events as a catalog invocation.
 */
export interface ExperimentalPassthroughToolExposure {
  /** Id used by `execute_tool` telemetry and local invocation events. */
  readonly id: string;
  /** Run one native execution without changing its scalar/promise/stream shape. */
  invoke<Result>(input: unknown, execute: () => Result): Result;
}

/**
 * The framework boundary: one complementary package per framework implements
 * this so Ratel speaks that framework's native tool and message shapes. The
 * three codecs (`ingest` / `expose` / `recallMessages`) are the required
 * contract; `experimentalExposePassthrough` can instrument client-executed
 * native tools and `extend` adds framework idioms. The core owns all state and
 * guards.
 *
 * @typeParam TTool - The framework's tool type (e.g. AI SDK `Tool`).
 * @typeParam TMessage - The framework's message type (e.g. AI SDK `ModelMessage`).
 * @typeParam TExt - The framework-idiomatic helpers merged onto the adapted object.
 */
export interface RatelAdapter<
  TTool = unknown,
  TMessage = unknown,
  TExt extends object = Record<never, never>,
> {
  /** Names the adapter in error messages (and, once adapter packages emit it, in
   * the `ratel.adapter` telemetry attribute — stamping is deferred to them per ADR-0013). */
  readonly name: string;
  /**
   * Framework tool → catalog registration, or `"passthrough"` for tools the
   * catalog can't execute (provider-executed) that must stay eagerly exposed.
   */
  ingest(id: string, tool: TTool): CatalogRegistration | "passthrough";
  /** Ratel capability tool → framework tool. Context-aware adapters wrap the
   * framework's complete live execution context in a private, stable tag before
   * passing it as the capability executor's optional second argument. */
  expose(tool: ExecutableTool): TTool;
  /**
   * Optionally wrap a framework-native passthrough each time {@link AdaptedBase.modelTools}
   * exposes it. Return `tool` unchanged when it has no client-side executor.
   */
  experimentalExposePassthrough?(tool: TTool, exposure: ExperimentalPassthroughToolExposure): TTool;
  /** Synthetic recall pair in the framework's message shape. */
  recallMessages(ref: RecallRef, recall: SearchCapabilitiesResult): TMessage[];
  /** Framework-idiomatic helpers, merged onto the adapted object. */
  extend?(base: AdaptedBase<TTool, TMessage>): TExt;
}

/**
 * The core's handle over its shared {@link ToolCatalog} — registration and
 * lookup in the SDK's native shapes, callable at any time (also after
 * {@link Ratel.modelTools}: the capability tools search the live catalog at
 * invocation time). Guards live here: the reserved capability-tool ids throw,
 * and a framework-shaped tool throws an actionable install-the-adapter error.
 * Registration keeps the catalog's own replace-in-place semantics — the native
 * path is authoritative, unlike the first-wins adapted path — and prepares the
 * dense cache on a semantic/hybrid catalog (or when an embedding artifact is
 * configured).
 */
export interface ToolCollection {
  /**
   * Register native tools (replace-in-place on a duplicate id). Async: input is
   * validated synchronously (a missing `execute`, a reserved id, or a
   * framework-shaped tool throws *at the call site*, before the promise), then
   * the returned promise resolves when the batch is indexed and dense preparation
   * finishes (embed or configured artifact warm), rejecting if that phase fails.
   * `await` it before searching a dense core; dense-preparation errors surface as
   * the rejection. Pass the whole batch in one call for a single preparation pass.
   */
  register(...tools: ExecutableTool[]): Promise<void>;
  /** Whether a tool with this id is registered. */
  has(id: string): boolean;
  /** The tool's searchable metadata, or `undefined` when unregistered. */
  get(id: string): Tool | undefined;
  /**
   * Rank the catalog for `query` synchronously (host-driven, origin `"direct"`).
   * BM25 only: a `"semantic"`/`"hybrid"` catalog (or a per-call `method`
   * override to one) throws with a pointer to {@link searchAsync}, since dense
   * ranking runs against the prebuilt embedding cache off the event loop. `topK`
   * is clamped to `[1, 50]` (invalid values fall back to 5), like the capability
   * funnel — drop to {@link catalog} for an unclamped search.
   */
  search(query: string, topK: number, method?: SearchMethod): SearchHit[];
  /**
   * Rank the catalog for `query` with any retrieval method without blocking the
   * event loop (origin `"direct"`, same `topK` clamp as {@link search}). Ranks
   * whatever is embedded now — `await register(...)` first so a dense tool is in
   * the cache.
   */
  searchAsync(query: string, topK: number, method?: SearchMethod): Promise<SearchHit[]>;
  /** Execute a registered tool by id with the args object. */
  invoke(id: string, args: Record<string, unknown>): Promise<unknown>;
  /** The shared catalog itself — the unguarded driver-level escape hatch. */
  readonly catalog: ToolCatalog;
}

/**
 * An adapted view's handle over the same shared catalog, speaking the
 * framework's tool shape. Registration runs the adapter's `ingest` codec;
 * first registration of an id wins across every view of the core, including
 * ids claimed by framework-local passthroughs, so repeated calls are idempotent.
 *
 * `register`/`has` are framework-aware (`register` takes the framework's tool
 * shape; `has` also covers this view's passthroughs); `get`/`search`/`invoke`
 * are at parity with {@link ToolCollection} — they read and run the shared
 * catalog in its neutral shapes, so they're catalog-only: a passthrough is
 * provider-executed and un-indexed, so `has` reports it but `get` returns
 * `undefined`, `search` never ranks it, and `invoke` can't run it.
 */
export interface AdaptedToolCollection<TTool> {
  /**
   * Ingest framework tools (keyed by tool id) into the shared catalog. Async,
   * with the same semantics as {@link ToolCollection.register}: ids are validated
   * and ingested synchronously (a reserved id throws at the call site), then the
   * returned promise resolves when the batch is indexed and dense preparation
   * finishes (embed or configured artifact warm) — rejecting if that phase fails.
   * `await` it before a dense search.
   */
  register(tools: Record<string, TTool>): Promise<void>;
  /** Whether this id is registered — in the catalog or as this view's passthrough. */
  has(id: string): boolean;
  /** A catalog tool's searchable metadata, or `undefined` (incl. for a passthrough). */
  get(id: string): Tool | undefined;
  /**
   * Rank the shared catalog for `query` synchronously (host-driven, origin
   * `"direct"`). BM25 only — semantic/hybrid throws with a pointer to
   * {@link searchAsync}. `topK` is clamped to `[1, 50]` (invalid values fall
   * back to 5); passthroughs are never ranked. Drop to {@link catalog} for an
   * unclamped search.
   */
  search(query: string, topK: number, method?: SearchMethod): SearchHit[];
  /**
   * Rank the shared catalog for `query` with any retrieval method off the event
   * loop (origin `"direct"`). Same `topK` clamp as {@link search}; passthroughs
   * are never ranked. `await register(...)` first so a dense tool is embedded.
   */
  searchAsync(query: string, topK: number, method?: SearchMethod): Promise<SearchHit[]>;
  /** Execute a catalog tool by id with the args object (not a passthrough). */
  invoke(id: string, args: Record<string, unknown>): Promise<unknown>;
  /** The shared catalog itself — the unguarded driver-level escape hatch. */
  readonly catalog: ToolCatalog;
}

/**
 * The framework-shaped surface every adapter inherits from the core. Adapters
 * add their idioms via {@link RatelAdapter.extend}; universal capability lives
 * here.
 */
export interface AdaptedBase<TTool, TMessage> {
  /** This view's handle over the shared catalog, in the framework's tool shape. */
  readonly tools: AdaptedToolCollection<TTool>;
  /** The shared skill catalog (skills are framework-neutral). */
  readonly skills: SkillCatalog;
  /** ⚠️ Experimental (facts, ADR-0017). The shared fact catalog — framework-neutral grounding content. */
  readonly facts: FactCatalog;
  /**
   * The model-facing toolset in the framework's shape: this view's passthroughs
   * plus the three capability tools run through the adapter's `expose` codec.
   * Fresh objects per call — take it once per agent instance and reuse it, so
   * the prompt cache survives. Tools registered later are still discoverable
   * (the capability tools search the live catalog); only a *passthrough*
   * registered later needs a fresh `modelTools()` to reach the model.
   */
  modelTools(): Record<string, TTool>;
  /**
   * Rank `query` and return the synthetic `search_capabilities` message pair in
   * the framework's shape (origin `"direct"`), or `[]` when nothing matched
   * (spending no call id). Pure: it builds fresh messages and never mutates a
   * host array.
   */
  recall(query: string): Promise<TMessage[]>;
  /**
   * ⚠️ Experimental (facts, ADR-0017). Decide which facts to (re-)inject given
   * the current transcript — the grounding freshness gate. See
   * {@link Ratel.ground}. Returns structured items (body + reason); the
   * caller renders them into messages.
   */
  ground(
    query: string,
    transcript: readonly string[],
    opts?: GroundOptions,
  ): Promise<GroundingResult>;
  /**
   * ⚠️ Experimental (facts, ADR-0017). Stateless per-call twin of
   * {@link AdaptedBase.ground}: the full grounding set for one model call,
   * nothing persisted. See `experimental.FactCatalog.groundSnapshot`.
   */
  groundSnapshot(query: string, opts?: GroundOptions): Promise<GroundingSnapshotItem[]>;
}

/**
 * The object {@link Ratel.adaptTo} returns: the base surface plus the adapter's
 * `extend` helpers. An adapter whose `extend` returns a non-object degrades to
 * the bare base type here (instead of collapsing the whole view to `never`,
 * which would surface as a cryptic error far from the broken adapter).
 */
export type AdaptedRatel<A extends RatelAdapter> =
  A extends RatelAdapter<infer TTool, infer TMessage, infer TExt>
    ? AdaptedBase<TTool, TMessage> & (TExt extends object ? TExt : unknown)
    : never;

/**
 * One `ratel(config)` core: a single {@link ToolCatalog} + {@link SkillCatalog}
 * + recall-id counter shared by every {@link Ratel.adaptTo} view. Genuinely
 * usable standalone — register native tools on {@link Ratel.tools}, hand the
 * model {@link Ratel.modelTools}, rank with {@link Ratel.recall} —
 * and adaptable on top for a framework's native shapes.
 */
export interface Ratel {
  /** Handle over the shared tool catalog (native shapes, guarded). */
  readonly tools: ToolCollection;
  /** The shared skill catalog — exposed raw (skills are framework-neutral and
   * need no ingest codec), so it is the unguarded escape hatch at parity with
   * {@link ToolCollection.catalog}: its `search` top-K is not clamped. Use
   * {@link recall} for a clamped, capability-shaped skills ranking. */
  readonly skills: SkillCatalog;
  /** ⚠️ Experimental (facts, ADR-0017). The shared fact catalog — constant grounding content, injected via {@link Ratel.ground}. */
  readonly facts: FactCatalog;
  /**
   * The three capability tools (`search_capabilities`, `invoke_tool`,
   * `get_skill_content`) in native shape, for framework-free hosts. All three
   * are always advertised — the set never depends on registration order, so the
   * prompt cache survives; loading a skill from an empty catalog returns a
   * structured error, not a missing tool. Fresh objects per call: take it once
   * and reuse it. Tools and skills registered later are still discoverable.
   */
  modelTools(): Record<string, ExecutableTool>;
  /**
   * Rank `query` into the canonical `search_capabilities` result (origin
   * `"direct"`, top-K from `recallTopK`), or `null` when nothing matched. A
   * pure query: no call id is minted — ids exist only on the adapted views,
   * whose synthetic message pairs need them. Ranks whatever is registered and
   * (on a dense core) embedded now — `await r.tools.register(...)` first.
   */
  recall(query: string): Promise<SearchCapabilitiesResult | null>;
  /**
   * ⚠️ Experimental (facts, ADR-0017). Decide which facts to (re-)inject given
   * the current transcript — the grounding freshness gate. Considers the
   * always-on tier (`experimental.FactCatalog.pinned`) plus the retrieval-gated
   * facts `query` ranks in, then injects only those whose body is not already
   * present in `transcript`: never injected (`never`), gone from the window
   * (`evicted`), or changed since injection (`mutated`). Records a
   * `fact_inject` / `fact_inject_skip` event per fact.
   *
   * Stateless across conversations — the transcript *is* the record — but
   * session-aware within one core instance: it remembers the last body injected
   * per id, so an absent body reads as `evicted`/`mutated` rather than `never`.
   * Returns structured `experimental.GroundingItem`s (body + reason); the caller
   * renders each `body` verbatim into the framework's message shape — its
   * presence in the transcript is what dedupes the next turn.
   *
   * @param query - The current turn's text, for the retrieval-gated tier.
   * @param transcript - Per-message text of the current history, oldest first.
   * @param opts - Per-call top-K override.
   * @returns The facts to inject (always-on first) and the ids left fresh.
   */
  ground(
    query: string,
    transcript: readonly string[],
    opts?: GroundOptions,
  ): Promise<GroundingResult>;
  /**
   * ⚠️ Experimental (facts, ADR-0017). Stateless per-call twin of
   * {@link Ratel.ground}: the full grounding set (always-on plus query-matched
   * facts) for one model call — no freshness gate, no transcript, nothing persisted.
   * Render into a per-call message override (e.g. a `prepareStep`) and discard.
   */
  groundSnapshot(query: string, opts?: GroundOptions): Promise<GroundingSnapshotItem[]>;
  /** Adapt the core to a framework, inferring its tool/message types and helpers. */
  adaptTo<A extends RatelAdapter>(adapter: A): AdaptedRatel<A>;
}

// The capability-tool ids: an app tool may not shadow them (registration throws).
const RESERVED_TOOL_IDS: ReadonlySet<string> = new Set([
  SEARCH_CAPABILITIES_ID,
  INVOKE_TOOL_ID,
  GET_SKILL_CONTENT_ID,
]);

// Catalog default when an adapter's registration omits an output schema.
const DEFAULT_OUTPUT_SCHEMA: JSONSchema7 = { type: "object" };

// Frameworks probed (for error messages only — detection can't tell installed
// from in use) to point a framework-shaped registration at the exact adapter.
const KNOWN_FRAMEWORKS: readonly {
  readonly pkg: string;
  readonly adapter: string;
  readonly factory: string;
}[] = [
  { pkg: "ai", adapter: "@ratel-ai/vercel-ai-sdk", factory: "aiSdk" },
  { pkg: "@mastra/core", adapter: "@ratel-ai/mastra", factory: "mastra" },
];

/**
 * Create a framework-neutral Ratel core. It works standalone — register native
 * tools on `r.tools`, skills on `r.skills`, hand the model `r.modelTools()`, rank
 * with `r.recall(query)` — and {@link Ratel.adaptTo | adapts} to a framework
 * with a {@link RatelAdapter} for that framework's native shapes. The core owns
 * all state (the catalogs, the recall-id counter) and every
 * framework-independent guard — reserved capability-tool ids, top-K clamping,
 * first-registration-wins on the adapted path, passthrough of non-executable
 * tools — so adapters stay tiny. One core can back several adapter views (they
 * share the catalog, embeddings, and counter).
 *
 * @param config - Retrieval method, embedding model (for semantic/hybrid),
 *   recall budget, and trace sink.
 * @returns The standalone core; call `.adaptTo(adapter())` for a framework-shaped view.
 *
 * @example
 * ```ts
 * import { ratel } from "@ratel-ai/sdk";
 * import { aiSdk } from "@ratel-ai/vercel-ai-sdk";
 *
 * const r = ratel({ recallTopK: 5 }).adaptTo(aiSdk());
 * await r.tools.register(myTools);
 * const tools = r.modelTools(); // stable capability set for the model — take once, reuse
 * const messages = await r.appendRecall(history); // per-turn recall (AI SDK idiom)
 * ```
 */
export function ratel(config: RatelConfig = {}): Ratel {
  const catalogMethod: SearchMethod = config.method ?? "bm25";
  const embeddingArtifact = config.experimentalEmbeddingArtifact;
  const catalog = new ToolCatalog({
    method: config.method,
    embedding: config.embedding,
    trace: config.trace,
    experimentalEmbeddingArtifact: embeddingArtifact,
  });
  const skills = new SkillCatalog({
    method: config.method,
    embedding: config.embedding,
    trace: config.trace,
    experimentalEmbeddingArtifact: embeddingArtifact,
  });
  // The fact catalog owns the grounding freshness state (its injected-body map),
  // so `r.ground` is a thin delegate to `facts.ground`. Constructed **lazily**:
  // facts are experimental, and building one eagerly would fire the
  // experimental-API warning at every `ratel()` call — including for the vast
  // majority of hosts that never touch facts. First access to `r.facts` (or to
  // `ground`/`groundSnapshot`, which go through it) creates the single shared
  // instance; nothing on the stable path (`recall`, `modelTools`) touches it.
  let factsCatalog: FactCatalog | undefined;
  const facts = (): FactCatalog => {
    factsCatalog ??= new FactCatalog({
      method: config.method,
      embedding: config.embedding,
      trace: config.trace,
      factsTopK: config.factsTopK,
    });
    return factsCatalog;
  };
  // Framework-shaped passthrough values stay in their originating views, but
  // their ids must participate in the core-wide adapted-path first-wins guard.
  const claimedAdaptedToolIds = new Set<string>();
  // Recall call ids come from a private counter shared across every view of this
  // core: transcript positions are caller-owned (trimming/compaction repeats
  // them as tool-call ids), so they can't be the id source.
  let recallSeq = 0;

  // Shared by both handles. Sync `search` is BM25-only — dense methods rank
  // against the prebuilt embedding cache off the event loop, so they route
  // through `searchAsync`.
  const searchSync = (query: string, topK: number, method?: SearchMethod): SearchHit[] => {
    const effective = method ?? catalogMethod;
    if (effective !== "bm25") {
      throw new Error(
        `ratel: tools.search() is synchronous and ranks BM25 only; "${effective}" ranks against ` +
          "prebuilt embeddings — use tools.searchAsync().",
      );
    }
    return catalog.search(query, clampTopK(topK, DEFAULT_TOP_K_TOOLS), "direct", "bm25");
  };
  const searchAsync = (query: string, topK: number, method?: SearchMethod): Promise<SearchHit[]> =>
    catalog.searchAsync(query, clampTopK(topK, DEFAULT_TOP_K_TOOLS), "direct", method);

  const tools: ToolCollection = {
    catalog,
    // Validate synchronously (a missing execute / reserved id / framework-shaped
    // tool throws at the call site, before the promise), then embed the whole
    // batch in one pass; the returned promise rejects if that embedding fails.
    register(...items) {
      for (const tool of items) {
        assertNativeTool(tool);
        assertUnreservedId(tool.id);
      }
      return catalog.register(items);
    },
    has: (id) => catalog.has(id),
    get: (id) => catalog.get(id),
    search: searchSync,
    searchAsync,
    invoke: (id, args) => catalog.invoke(id, args),
  };

  function modelTools(): Record<string, ExecutableTool> {
    return {
      // advertiseSkills pins the skills clause of the description: the exposed
      // payload must be byte-identical whether skills register before or after.
      [SEARCH_CAPABILITIES_ID]: searchCapabilitiesTool(catalog, skills, {
        advertiseSkills: true,
      }),
      [INVOKE_TOOL_ID]: invokeToolTool(catalog),
      [GET_SKILL_CONTENT_ID]: getSkillContentTool(skills),
    };
  }

  async function recall(query: string): Promise<SearchCapabilitiesResult | null> {
    const result = await runCapabilitiesSearch(catalog, query, {
      topKTools: config.recallTopK, // capped/validated inside runCapabilitiesSearch
      skillCatalog: skills,
      origin: "direct",
    });
    return result.tools.groups.length === 0 && result.skills.length === 0 ? null : result;
  }

  // Grounding lives on the fact catalog (it owns the fact state); the core just
  // forwards to it, so `r.ground`/`r.groundSnapshot` and the catalog methods
  // are one path.
  const ground = (
    query: string,
    transcript: readonly string[],
    opts?: GroundOptions,
  ): Promise<GroundingResult> => facts().ground(query, transcript, opts);
  const groundSnapshot = (query: string, opts?: GroundOptions): Promise<GroundingSnapshotItem[]> =>
    facts().groundSnapshot(query, opts);

  function adaptTo<A extends RatelAdapter>(adapter: A): AdaptedRatel<A> {
    assertAdapter(adapter);
    // Provider- or client-executed tools: framework-shaped, so per view — a
    // Mastra view must never expose an AI SDK passthrough.
    const passthrough = new Map<string, unknown>();

    const adaptedTools: AdaptedToolCollection<unknown> = {
      catalog,
      has: (id) => catalog.has(id) || passthrough.has(id),
      get: (id) => catalog.get(id),
      search: searchSync,
      searchAsync,
      invoke: (id, args) => catalog.invoke(id, args),
      // Ingest synchronously (validation + adapter codec + passthrough routing),
      // then embed the ingested batch in one pass; the promise rejects on failure.
      // Both the passthroughs and the executables stage into locals and commit
      // only after the whole batch validates, so a mid-batch throw (a reserved
      // id, or an `ingest` that throws) leaves nothing committed — parity with
      // the native path, which validates the batch before touching the catalog.
      register(appTools) {
        const batch: ExecutableTool[] = [];
        const stagedClaims: string[] = [];
        const stagedPassthrough: [string, unknown][] = [];
        for (const [id, tool] of Object.entries(appTools)) {
          assertUnreservedId(id);
          // First registration of an id wins, across every view of the core
          // and all views' passthrough ids — repeated calls are idempotent.
          if (catalog.has(id) || claimedAdaptedToolIds.has(id)) continue;
          const registration = adapter.ingest(id, tool);
          stagedClaims.push(id);
          if (registration === "passthrough") {
            stagedPassthrough.push([id, tool]);
            continue;
          }
          batch.push({
            id,
            name: id,
            description: registration.description,
            inputSchema: registration.inputSchema,
            outputSchema: registration.outputSchema ?? DEFAULT_OUTPUT_SCHEMA,
            validateInput: registration.validateInput,
            execute: registration.execute,
          });
        }
        // The batch validated: commit the passthroughs, then index the executables.
        for (const [id, tool] of stagedPassthrough) passthrough.set(id, tool);
        for (const id of stagedClaims) claimedAdaptedToolIds.add(id);
        return catalog.register(batch);
      },
    };

    const baseWithoutFacts: Omit<AdaptedBase<unknown, unknown>, "facts"> = {
      tools: adaptedTools,
      skills,
      ground,
      groundSnapshot,
      modelTools() {
        const out: Record<string, unknown> = {};
        for (const [id, tool] of passthrough) {
          out[id] =
            adapter.experimentalExposePassthrough?.(tool, {
              id,
              invoke: (input, execute) => runToolInvocation(catalog, id, input, execute),
            }) ?? tool;
        }
        for (const [id, tool] of Object.entries(modelTools())) {
          out[id] = adapter.expose(tool);
        }
        return out;
      },
      async recall(query) {
        const result = await recall(query);
        if (result === null) return []; // nothing matched: don't spend a call id
        return adapter.recallMessages({ callId: `recall_${recallSeq++}`, query }, result);
      },
    };
    const base: AdaptedBase<unknown, unknown> = withLazyFacts(facts, baseWithoutFacts);
    // Generics are erased inside the implementation; the public signature keeps
    // them, so cast the base at the extend boundary and on return. `base`'s
    // `facts` is non-enumerable, so this spread does NOT construct the catalog;
    // re-attach the accessor to the new object rather than copying a value.
    const ext = adapter.extend ? adapter.extend(base as AdaptedBase<never, never>) : {};
    return withLazyFacts(facts, { ...base, ...ext }) as AdaptedRatel<A>;
  }

  return withLazyFacts(facts, {
    tools,
    skills,
    modelTools,
    recall,
    ground,
    groundSnapshot,
    adaptTo,
  });
}

/**
 * Attach the lazy, ⚠️ experimental `facts` accessor to a returned handle.
 *
 * **Non-enumerable on purpose.** Object spread, `Object.assign`, `JSON.stringify`,
 * and any logger that walks own enumerable keys all perform `[[Get]]` on every
 * enumerable property — which would construct the experimental {@link FactCatalog},
 * and fire its one-time warning, for a host that never touches facts. `adaptTo`
 * spreads its base, so an *enumerable* getter there quietly defeated the lazy
 * quarantine for every adapted host (and both showcase examples). Keeping it
 * off the enumerable keys is what makes the quarantine hold in practice; the
 * property is still readable, `in`-visible, and typed exactly as before.
 *
 * @param get - Builds (or returns the already-built) shared catalog.
 * @param target - The handle to attach to; returned for chaining.
 */
function withLazyFacts<T extends object>(
  get: () => FactCatalog,
  target: T,
): T & { readonly facts: FactCatalog } {
  Object.defineProperty(target, "facts", { get, enumerable: false, configurable: true });
  return target as T & { readonly facts: FactCatalog };
}

/** Reject a reserved capability-tool id (the funnel's vocabulary can't be shadowed). */
function assertUnreservedId(id: string): void {
  if (RESERVED_TOOL_IDS.has(id)) {
    throw new Error(`ratel: tool id "${id}" is reserved for the capability tools`);
  }
}

/**
 * Reject a native-registration input that isn't a well-formed native tool.
 * Fingerprints, not full validation: a zod-style schema or a dynamic (function)
 * description — the marks of a framework tool that belongs on an adapted view —
 * trigger the actionable {@link unadaptedError}. A missing id is a *malformed
 * native* tool, not a framework one (framework tools are keyed externally in a
 * `Record`, so their id-lessness is expected), so it gets its own plain error
 * rather than misdirecting the caller to install an adapter. A missing `execute`
 * is caught here too, so a malformed tool fails fast at the call site rather than
 * as a rejection of the `register` promise the caller may not be awaiting.
 * Everything else is the catalog's job to validate.
 */
function assertNativeTool(tool: ExecutableTool): void {
  const candidate = tool as {
    id?: unknown;
    description?: unknown;
    inputSchema?: unknown;
    execute?: unknown;
  } | null;
  if (candidate === null || typeof candidate !== "object") {
    throw new TypeError(`ratel: tools.register() expects ExecutableTools; got ${typeof tool}`);
  }
  const schema = candidate.inputSchema as { _def?: unknown; safeParse?: unknown } | undefined;
  const frameworkShaped =
    typeof candidate.description === "function" ||
    (typeof schema === "object" &&
      schema !== null &&
      (schema._def !== undefined || typeof schema.safeParse === "function"));
  if (frameworkShaped) throw unadaptedError(isPackageInstalled);
  if (typeof candidate.id !== "string") {
    throw new TypeError(
      `ratel: tools.register() expects each ExecutableTool to have a string \`id\`; got ${typeof candidate.id}`,
    );
  }
  if (typeof candidate.execute !== "function") {
    throw new TypeError(
      `ratel: tools.register() expects each ExecutableTool to have an \`execute\` function; got ${typeof candidate.execute}`,
    );
  }
}

/** Reject a non-adapter early (JS callers) with a message that names it via `adapter.name`. */
function assertAdapter(adapter: RatelAdapter): void {
  if (
    !adapter ||
    typeof adapter.ingest !== "function" ||
    typeof adapter.expose !== "function" ||
    typeof adapter.recallMessages !== "function"
  ) {
    const label =
      adapter && typeof adapter.name === "string" ? `"${adapter.name}"` : `${typeof adapter}`;
    throw new Error(
      `ratel: adaptTo() requires a RatelAdapter with ingest/expose/recallMessages; got ${label}`,
    );
  }
}

/**
 * Build the error thrown when a framework-shaped tool hits the native
 * `ratel().tools.register(...)` path. When a known framework is present in the
 * tree, name its exact adapter package and factory; otherwise a generic
 * adapt-first message. Exported (module-internal, not re-exported from the
 * package) so the detected-framework branch is testable with an injected probe.
 *
 * @param isInstalled - Package-resolution probe (the real {@link isPackageInstalled}).
 * @returns The error to throw.
 */
export function unadaptedError(isInstalled: (specifier: string) => boolean): Error {
  const intro =
    "ratel: this looks like a framework tool, not a native ExecutableTool. " +
    "Register framework tools on an adapted view: `ratel(config).adaptTo(adapter()).tools.register(...)`.";
  const detected = KNOWN_FRAMEWORKS.filter((f) => isInstalled(f.pkg));
  if (detected.length === 0) return new Error(intro);
  const lines = detected.map(
    (f) =>
      `  - ${f.pkg}: install \`${f.adapter}\` and register via \`ratel(config).adaptTo(${f.factory}()).tools.register(...)\``,
  );
  return new Error(`${intro} Detected in your dependencies:\n${lines.join("\n")}`);
}
