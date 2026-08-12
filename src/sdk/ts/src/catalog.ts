import { SearchTarget } from "@ratel-ai/telemetry";
import type { SearchHit, Tool } from "../native/index.cjs";
import { isAsyncIterable, isPromiseLike } from "./async.js";
import {
  type ExperimentalEmbeddingArtifact,
  resolveEmbeddingArtifact,
} from "./embedding-artifact.js";
import { type IntentGraph, ToolRegistry } from "./registry.js";
import {
  argsSizeBytes,
  errorMessage,
  traceExecuteTool,
  traceSearch,
  traceSearchAsync,
} from "./telemetry.js";

/**
 * The function that runs a tool. Receives the arguments object and an optional
 * opaque invocation context supplied by a framework adapter. It may return
 * a plain value, `Promise`, or `AsyncIterable`. {@link ToolCatalog.invoke}
 * absorbs synchronous values and throws into its promise contract;
 * {@link ToolCatalog.invokeRaw} preserves the immediate return shape when
 * validation is synchronous, while {@link ToolCatalog.invokeValidatedRaw}
 * guarantees that shape after a host has already validated the input.
 * One-argument executors remain valid; framework-neutral callers normally omit
 * `context`.
 */
// biome-ignore lint/suspicious/noExplicitAny: tool inputs are heterogeneous across the catalog
export type Executor = (input: any, context?: unknown) => Promise<unknown> | unknown;

/** Result returned by a framework-native input validator. */
export type InputValidationResult =
  | {
      /** The input was accepted. */
      success: true;
      /** Parsed value, including defaults or transformations. */
      value: unknown;
    }
  | {
      /** The input was rejected. */
      success: false;
      /** Framework-native validation failure. */
      error: Error;
    };

/**
 * Optional framework-native parser retained by the shared catalog. It may
 * validate, apply defaults, or transform the input before execution.
 */
export type InputValidator = (
  input: unknown,
) => InputValidationResult | PromiseLike<InputValidationResult>;

/**
 * A tool the catalog can both retrieve *and* run: the searchable metadata of a
 * `Tool` (id, name, description, schemas) plus its {@link Executor}. The unit
 * {@link ToolCatalog.register} accepts and {@link ToolCatalog.getExecutable}
 * returns.
 */
export interface ExecutableTool extends Tool {
  /** Shared parser used before execution and by model-facing host bridges. */
  validateInput?: InputValidator;
  /** Runs the tool. Called by {@link ToolCatalog.invoke} with args and optional context. */
  execute: Executor;
}

/**
 * Where the local trace stream (ADR-0007) goes. Distinct from the OTel spans in
 * {@link ToolCatalog}'s docs — this is the in-process channel drained via
 * {@link ToolCatalog.drainTraceEvents} or written to disk.
 *
 * - `"noop"` — discard every event (the default when no `trace` option is given).
 * - `"memory"` — buffer envelopes in-process; read them back with
 *   {@link ToolCatalog.drainTraceEvents}. `sessionId` is stamped on each envelope.
 * - `"jsonl"` — append one JSON envelope per line to the file at `path`
 *   (parent directories are created). `sessionId` is stamped on each envelope.
 */
export type TraceSinkConfig =
  | {
      /** Discard every event. */
      kind: "noop";
    }
  | {
      /** Buffer envelopes in-process for `drainTraceEvents`. */
      kind: "memory";
      /** Session id stamped on every envelope. */
      sessionId: string;
    }
  | {
      /** Append one JSON envelope per line to `path`. */
      kind: "jsonl";
      /** Session id stamped on every envelope. */
      sessionId: string;
      /** File to append to; parent directories are created. */
      path: string;
    };

/**
 * Who initiated a search: `"direct"` for host code calling the SDK itself
 * (pre-fetch helpers, benchmarks), `"agent"` for a call the model synthesized
 * through the capability tools (`search_capabilities`). Recorded on trace
 * events and the `ratel.origin` span attribute so consumers can separate the
 * two paths.
 */
export type SearchOrigin = "direct" | "agent";

/**
 * Retrieval engine for {@link ToolCatalog.search} (and the skill catalog's
 * `search`):
 *
 * - `"bm25"` — lexical ranking; model-free and infallible (the default).
 * - `"semantic"` — cosine similarity over prebuilt embeddings.
 * - `"hybrid"` — BM25 and semantic rankings fused with Reciprocal Rank Fusion
 *   (ADR-0011).
 *
 * `"semantic"`/`"hybrid"` need a prepared dense cache: registration builds it
 * (or warms a configured embedding artifact). Dense ranking uses
 * `searchAsync()`.
 */
export type SearchMethod = "bm25" | "semantic" | "hybrid";

type EmbeddingConfigKey =
  | "huggingface"
  | "local"
  | "ollama"
  | "url"
  | "model"
  | "revision"
  | "apiKeyEnv"
  | "pooling"
  | "download";

type ExclusiveEmbeddingFields<Allowed extends EmbeddingConfigKey> = {
  [Key in Exclude<EmbeddingConfigKey, Allowed>]?: never;
};

/** Object form of the embedding-model selection for semantic/hybrid retrieval.
 * Each variant accepts exactly one source; fields from other variants are
 * rejected at compile time. Use the bare string form only for a local model
 * directory path. */
export type EmbeddingModelConfig =
  | (ExclusiveEmbeddingFields<"huggingface" | "revision" | "pooling" | "download"> & {
      /** HuggingFace repo id (e.g. `"intfloat/e5-small-v2"`), loaded in-process via Candle. */
      huggingface: string;
      /** Git revision to pin; defaults to `main`. */
      revision?: string;
      /** Query-side prefix for asymmetric models (e.g. e5's `"query: "`). */
      queryPrefix?: string;
      /** Document-side prefix for asymmetric models (e.g. e5's `"passage: "`). */
      docPrefix?: string;
      /** `"cls"` | `"mean"` — overrides pooling auto-detection. */
      pooling?: "cls" | "mean";
      /** Opt in to downloading if not already cached (default false; Ratel
       * auto-downloads only the built-in default model). */
      download?: boolean;
    })
  | (ExclusiveEmbeddingFields<"local" | "pooling"> & {
      /** Path to a local model directory, loaded in-process via Candle. */
      local: string;
      /** Query-side prefix for asymmetric models. */
      queryPrefix?: string;
      /** Document-side prefix for asymmetric models. */
      docPrefix?: string;
      /** `"cls"` | `"mean"` — overrides pooling auto-detection. */
      pooling?: "cls" | "mean";
    })
  | (ExclusiveEmbeddingFields<"ollama"> & {
      /** Ollama model name, served via the local Ollama endpoint. */
      ollama: string;
      /** Query-side prefix for asymmetric models. */
      queryPrefix?: string;
      /** Document-side prefix for asymmetric models. */
      docPrefix?: string;
    })
  | (ExclusiveEmbeddingFields<"url" | "model" | "apiKeyEnv"> & {
      /** Full OpenAI-compatible `/embeddings` endpoint URL. */
      url: string;
      /** Model name sent in the request body. */
      model: string;
      /** Env var holding the bearer key; omit for no auth. */
      apiKeyEnv?: string;
      /** Query-side prefix for asymmetric models. */
      queryPrefix?: string;
      /** Document-side prefix for asymmetric models. */
      docPrefix?: string;
    });

/** Embedding-model selection: a bare string is a **local model directory path**;
 * every other source is an explicit {@link EmbeddingModelConfig} object. */
export type EmbeddingSpec = string | EmbeddingModelConfig;

/** Construction options for {@link ToolCatalog}. */
export interface ToolCatalogOptions {
  /** Local trace stream destination (default: discard). See {@link TraceSinkConfig}. */
  trace?: TraceSinkConfig;
  /** Default retrieval method for `search` (default `"bm25"`, model-free). A
   * per-call `method` argument overrides it. */
  method?: SearchMethod;
  /** Embedding model backing semantic/hybrid retrieval. A string is a local
   * model directory path (`"/opt/models/bge"`); every other source is a keyed
   * object: `{ huggingface: "BAAI/bge-base-en-v1.5" }`, `{ ollama: "…" }`, or
   * `{ url, model, apiKeyEnv }`. Chosen once, used for both document and query
   * embedding. Retained and validated even when the default method is `"bm25"`,
   * allowing a later asynchronous semantic override. */
  embedding?: EmbeddingSpec;
  /**
   * Build-time RAT1 to warm on register (any method; default `onMiss: "error"`).
   * Each `register` re-resolves and re-warms over the whole current corpus —
   * intended for one batch at startup; incremental register calls repeat I/O
   * and id+hash matching.
   *
   * With the default fail-closed `onMiss: "error"`, warm fails when the
   * artifact is missing one or more ids from the catalog's current corpus.
   */
  experimentalEmbeddingArtifact?: ExperimentalEmbeddingArtifact;
}

/**
 * In-process catalog of executable tools, ranked by the native Rust registry.
 * The SDK's central surface: {@link ToolCatalog.register} tools (or ingest an
 * MCP server's via `registerMcpServer`), {@link ToolCatalog.search} them by
 * relevance, and {@link ToolCatalog.invoke} the chosen one. Every operation
 * emits both an OTel span (to whatever provider is active — see telemetry.ts)
 * and a local trace event (to the sink from {@link ToolCatalogOptions.trace}),
 * per ADR-0007.
 *
 * @example
 * ```ts
 * import { ToolCatalog } from "@ratel-ai/sdk";
 * import { readFile } from "node:fs/promises";
 *
 * const catalog = new ToolCatalog();
 * await catalog.register({
 *   id: "read_file",
 *   name: "read_file",
 *   description: "Read a file from local disk and return its textual contents.",
 *   inputSchema: {
 *     type: "object",
 *     properties: { path: { type: "string", description: "absolute path to the file" } },
 *     required: ["path"],
 *   },
 *   outputSchema: { type: "object" },
 *   execute: async ({ path }) => ({ contents: await readFile(path, "utf8") }),
 * });
 *
 * const [hit] = catalog.search("read a file from disk", 5);
 * const result = await catalog.invoke(hit.toolId, { path: "/tmp/notes.txt" });
 * ```
 */
export class ToolCatalog {
  private readonly registry: ToolRegistry;
  private readonly executors = new Map<string, Executor>();
  private readonly inputValidators = new Map<string, InputValidator>();
  private readonly tools = new Map<string, Tool>();
  private readonly method: SearchMethod;
  private readonly embeddingArtifact: ExperimentalEmbeddingArtifact | undefined;

  /**
   * Create an empty catalog.
   *
   * @param options - Trace sink, default retrieval method, and embedding model.
   *   Construction validates configuration but never loads a model.
   */
  constructor(options: ToolCatalogOptions = {}) {
    this.method = options.method ?? "bm25";
    this.registry = new ToolRegistry(options.embedding, this.method);
    this.embeddingArtifact = options.experimentalEmbeddingArtifact;
    if (options.trace) {
      this.registry.setTraceSink(options.trace);
    }
  }

  /**
   * Add one tool or a batch to the catalog — the single entry point for
   * both. Replaces an id in place when already registered (metadata,
   * executor, and index entry; the corpus never holds a duplicate). On a
   * `"semantic"`/`"hybrid"` catalog without an artifact, embeds the batch after
   * metadata is indexed. With {@link ToolCatalogOptions.experimentalEmbeddingArtifact},
   * warms that artifact first (any method). Dense-preparation errors surface
   * **here**; metadata still persists if that phase fails. A `"bm25"` catalog
   * without an artifact never loads a model.
   *
   * A model or dimension change is not recovered in place — construct a new
   * catalog and re-register.
   *
   * @param tools - A single tool or a readonly array of tools; each
   *   `execute` must be set. Pass the whole batch at once for a single
   *   dense-preparation request — separate `register` calls prepare separately.
   * @throws {@link EmbedderError} when embedding fails;
   *   {@link ArtifactWarmError} when a configured artifact fails;
   *   plain `Error` if `execute` is missing.
   */
  async register(tools: ExecutableTool | readonly ExecutableTool[]): Promise<void> {
    const batch = Array.isArray(tools) ? tools : [tools];
    for (const tool of batch) {
      if (typeof tool.execute !== "function") {
        throw new Error(`tool ${tool.id} has no execute handler`);
      }
    }
    this.registry.registerItems(
      batch.map(({ execute: _execute, validateInput: _validateInput, ...metadata }) => metadata),
    );
    for (const tool of batch) {
      const { execute, validateInput, ...metadata } = tool;
      this.executors.set(tool.id, execute);
      if (validateInput) {
        this.inputValidators.set(tool.id, validateInput);
      } else {
        this.inputValidators.delete(tool.id);
      }
      this.tools.set(tool.id, metadata);
    }
    await this.ensureDenseReady();
  }

  private async ensureDenseReady(): Promise<void> {
    if (this.embeddingArtifact) {
      const { bytes, onMiss } = await resolveEmbeddingArtifact(this.embeddingArtifact);
      await this.registry.experimentalWarmEmbeddingsFromArtifact(bytes, onMiss);
      return;
    }
    await this.registry.buildDense();
  }

  /**
   * Search the catalog. `method` overrides the catalog default for this call.
   * `"semantic"`/`"hybrid"` rank against the prebuilt embedding cache and throw
   * synchronously with guidance to use {@link ToolCatalog.searchAsync}.
   *
   * @param query - Natural-language description of what the caller wants to do.
   * @param topK - Maximum number of hits to return.
   * @param origin - Who initiated the call (default `"direct"`); recorded on
   *   the trace event and span, never affects ranking.
   * @param method - Per-call override of the catalog's default retrieval method.
   * @returns Up to `topK` BM25 hits, best-first with ties broken by tool id.
   *   Semantic/dense/hybrid methods throw migration guidance; use
   *   {@link ToolCatalog.searchAsync} for those methods.
   */
  search(
    query: string,
    topK: number,
    origin: SearchOrigin = "direct",
    method?: SearchMethod,
  ): SearchHit[] {
    return traceSearch(SearchTarget.Tool, query, topK, origin, () =>
      this.registry.searchWithMethod(query, topK, origin, method ?? this.method),
    );
  }

  /** Search with any retrieval method without blocking the Node.js event loop. */
  searchAsync(
    query: string,
    topK: number,
    origin: SearchOrigin = "direct",
    method?: SearchMethod,
  ): Promise<SearchHit[]> {
    return traceSearchAsync(SearchTarget.Tool, query, topK, origin, () =>
      this.registry.searchWithMethodAsync(query, topK, origin, method ?? this.method),
    );
  }

  /**
   * Whether a tool with this id is registered.
   *
   * @param toolId - The id to look up.
   * @returns `true` if {@link ToolCatalog.invoke} would find an executor for it.
   */
  has(toolId: string): boolean {
    return this.executors.has(toolId);
  }

  /**
   * Look up a tool's searchable metadata (no executor attached).
   *
   * @param toolId - The id to look up.
   * @returns The metadata as registered, or `undefined` for an unknown id.
   */
  get(toolId: string): Tool | undefined {
    return this.tools.get(toolId);
  }

  /**
   * Look up a tool with its executor reattached.
   *
   * @param toolId - The id to look up.
   * @returns A copy of the registered tool including `execute`, or `undefined`
   *   for an unknown id.
   */
  getExecutable(toolId: string): ExecutableTool | undefined {
    const tool = this.tools.get(toolId);
    const execute = this.executors.get(toolId);
    if (!tool || !execute) return undefined;
    const validateInput = this.inputValidators.get(toolId);
    return validateInput ? { ...tool, validateInput, execute } : { ...tool, execute };
  }

  /**
   * Run a registered framework validator without executing the tool. A tool
   * without one succeeds with its input unchanged. The returned success value
   * is the exact input the executor should receive, including any defaults or
   * root-level transformation.
   */
  validateInput(
    toolId: string,
    input: unknown,
  ): InputValidationResult | Promise<InputValidationResult> {
    const validate = this.inputValidators.get(toolId);
    if (!validate) return { success: true, value: input };
    try {
      const result = validate(input);
      return isPromiseLike(result)
        ? Promise.resolve(result).catch((error) => ({
            success: false as const,
            error: error instanceof Error ? error : new Error(String(error)),
          }))
        : result;
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error : new Error(String(error)),
      };
    }
  }

  /**
   * Record a custom event on the local trace stream (ADR-0007), e.g. an
   * `upstream_register` from an ingestion layer. Delivered to the sink
   * configured at construction; a no-op sink discards it.
   *
   * @param event - A tagged trace event in wire shape: `{ type: "...", ... }`
   *   with snake_case fields. Throws if the object is not a known trace event.
   */
  recordEvent(event: object): void {
    this.registry.recordEvent(event);
  }

  /**
   * Turn on adaptive usage ranking against `graph` (ADR-0014): the catalog
   * ranks against what users have actually invoked after similar queries, and
   * keeps learning as it is used.
   *
   * Pass the same {@link IntentGraph} to a {@link SkillCatalog} so both learn
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
   * buffer.
   *
   * @returns The captured envelopes (`{ v, ts, session_id, type, ... }` — the
   *   event fields are flattened alongside the envelope stamp) in record
   *   order. Always `[]` unless the active sink is `"memory"`.
   */
  drainTraceEvents(): unknown[] {
    return this.registry.drainTraceEvents();
  }

  /**
   * Run a registered tool's executor. Sync-absorbing: a plain value or
   * `Promise` settles through the returned promise, and a synchronous `throw`
   * surfaces as a rejection — `invoke` never throws synchronously, including
   * for an unknown `toolId` (that rejects with `unknown toolId: …`). An
   * `AsyncIterable` is the resolved value; its instrumentation settles when
   * the iterator completes, is cancelled, or fails.
   *
   * The call is wrapped in an `execute_tool` OTel span and bracketed by
   * `invoke_start` / `invoke_end` (or `invoke_error`, with the error message)
   * events on the local trace stream, `took_ms` in wall-clock milliseconds.
   *
   * @param toolId - Id of a registered tool.
   * @param args - Arguments object validated and possibly transformed before execution.
   * @param context - Optional opaque invocation context forwarded unchanged.
   * @returns Whatever the executor returns (resolved if it returned a promise).
   */
  async invoke(toolId: string, args: Record<string, unknown>, context?: unknown): Promise<unknown> {
    return await this.invokeRaw(toolId, args, context);
  }

  /**
   * Run a registered tool without erasing its immediate return shape after
   * synchronous validation. A plain value stays plain, a promise stays a
   * promise, and an `AsyncIterable` stays synchronously iterable. An async
   * validator necessarily makes this method return a promise of the executor's
   * result. Instrumentation settles only when that returned shape settles
   * (including iterator completion, cancellation, or failure).
   *
   * Most callers should use {@link invoke}; capability-tool bridges use this
   * path so a host framework can observe streamed preliminary outputs.
   */
  invokeRaw(toolId: string, args: Record<string, unknown>, context?: unknown): unknown {
    if (!this.executors.has(toolId)) {
      throw new Error(`unknown toolId: ${toolId}`);
    }
    return runIfValid(this.validateInput(toolId, args), (validated) =>
      this.invokeValidatedRaw(toolId, validated, context),
    );
  }

  /**
   * Execute input already parsed by {@link validateInput}, preserving its
   * immediate scalar, promise, or `AsyncIterable` return shape. Framework
   * bridges call this only after their host has run the capability tool's live
   * validator; ordinary callers should use {@link invoke}.
   */
  invokeValidatedRaw(toolId: string, input: unknown, context?: unknown): unknown {
    const fn = this.executors.get(toolId);
    if (!fn) {
      throw new Error(`unknown toolId: ${toolId}`);
    }
    // The `execute_tool` OTel span wraps the local trace stream; both record the
    // same invocation, on their two independent channels (ADR-0007).
    return traceExecuteTool(toolId, input, () => {
      this.registry.recordEvent({
        type: "invoke_start",
        tool_id: toolId,
        args_size_bytes: argsSizeBytes(input),
      });
      const started = Date.now();

      const succeed = (_result: unknown): void => {
        this.registry.recordEvent({
          type: "invoke_end",
          tool_id: toolId,
          took_ms: Date.now() - started,
        });
      };
      const reject = (err: unknown): void => {
        this.registry.recordEvent({
          type: "invoke_error",
          tool_id: toolId,
          took_ms: Date.now() - started,
          error: errorMessage(err),
        });
      };

      try {
        const result = context === undefined ? fn(input) : fn(input, context);
        return observeInvocationResult(result, succeed, reject);
      } catch (err) {
        reject(err);
        throw err;
      }
    });
  }
}

/** Unwrap a (possibly async) validation result and continue with its value, or throw its error. */
function runIfValid(
  result: InputValidationResult | PromiseLike<InputValidationResult>,
  onSuccess: (value: unknown) => unknown,
): unknown {
  const settle = (validated: InputValidationResult): unknown => {
    if (!validated.success) throw validated.error;
    return onSuccess(validated.value);
  };
  return isPromiseLike(result) ? Promise.resolve(result).then(settle) : settle(result);
}

function observeInvocationResult(
  result: unknown,
  onSuccess: (result: unknown) => void,
  onError: (error: unknown) => void,
): unknown {
  if (isAsyncIterable(result)) {
    return observeAsyncIterable(result, onSuccess, onError);
  }
  if (isPromiseLike(result)) {
    return Promise.resolve(result).then(
      (value) => {
        onSuccess(value);
        return value;
      },
      (error) => {
        onError(error);
        throw error;
      },
    );
  }
  onSuccess(result);
  return result;
}

async function* observeAsyncIterable(
  iterable: AsyncIterable<unknown>,
  onSuccess: (result: unknown) => void,
  onError: (error: unknown) => void,
): AsyncGenerator<unknown> {
  let failed = false;
  let lastValue: unknown;
  try {
    for await (const value of iterable) {
      lastValue = value;
      yield value;
    }
  } catch (error) {
    failed = true;
    onError(error);
    throw error;
  } finally {
    if (!failed) onSuccess(lastValue);
  }
}
