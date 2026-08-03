import type {
  CompressedPrompt as NativeCompressedPrompt,
  CompressionStats as NativeCompressionStats,
} from "../native/index.cjs";
import { PromptCompressor as NativePromptCompressor, type WordScore } from "../native/index.cjs";

/**
 * Which model backs compression.
 *
 * A bare string is a **local model directory path**; a HuggingFace repo uses the
 * explicit object form, so the source is never guessed from an ambiguous string.
 * There is no endpoint form — nothing standard serves per-token classification
 * logits, so compression runs in-process only.
 */
export type CompressionModelSpec =
  | string
  | {
      /** HuggingFace repo id of a BERT token-classification checkpoint. */
      huggingface: string;
      /** Git revision; `main` when omitted. */
      revision?: string;
      /**
       * Opt in to downloading this model if it isn't already in the local
       * HuggingFace cache. Ratel auto-downloads only the built-in default.
       */
      download?: boolean;
    }
  | {
      /** Directory holding `config.json`, `tokenizer.json`, and weights. */
      local: string;
    };

/** A span of the input that must survive compression at any rate. */
export type ProtectPattern = string | RegExp;

/** Per-call compression options. Omitted fields take the documented defaults. */
export interface CompressionOptions {
  /**
   * Keep every unit the model rates at or above this. Default `0.5`.
   *
   * **A quality bar, not a size target** — the compression ratio is an *output*.
   * Redundant prose compresses hard; text where the model finds little filler
   * barely compresses, and never at the cost of dropping something it rated
   * important. Lower the bar to compress more aggressively; `0` keeps everything.
   */
  minImportance?: number;
  /**
   * Word count below which the input is returned verbatim. Checked **before**
   * any model load, so a short prompt costs nothing. Default `40`.
   */
  minWords?: number;
  /** Model-token count below which the input is returned verbatim. Default `50`. */
  minTokens?: number;
  /** Maximum encoder passes; beyond it, compression throws. Default `16`. */
  maxChunks?: number;
  /**
   * Spans that must survive at any rate. A `string` matches literally (regex
   * metacharacters mean themselves); a `RegExp` is applied as a pattern.
   *
   * Naming what matters is cheaper and more precise than
   * {@link CompressionOptions.protectNumbers} — protection is charged to the
   * budget first, so every protected span is budget not spent on prose.
   */
  protect?: ProtectPattern[];
  /**
   * Protect every unit containing a digit. Default **`false`**.
   *
   * Off by default because it is indiscriminate and measurably harmful: on the
   * reference transcript it protects 23 units — `Q3`, `12 terabytes`,
   * `November 2024` — and the budget they consume costs *more* critical facts
   * than it saves at every rate below `0.4`. Prefer naming the figures that
   * matter via {@link CompressionOptions.protect}.
   */
  protectNumbers?: boolean;
  /**
   * Protect negations, whose loss inverts a claim rather than shortening it.
   * Default `true` — it costs ~3 units on a page of prose and is what keeps
   * `doesn't support` from becoming `support` under an aggressive rate.
   */
  protectNegations?: boolean;
  /** Replace the built-in (English) negation list. */
  negationTerms?: string[];
  /** Keep blank lines as blank lines. Default `true`. */
  preserveParagraphs?: boolean;
  /**
   * Populate {@link CompressedPrompt.kept} / {@link CompressedPrompt.dropped}.
   * Default `true`.
   */
  explain?: boolean;
}

export type { WordScore };

/**
 * Why an input was returned verbatim.
 *
 * - `too_short_words` — fewer words than `minWords`; no model was loaded.
 * - `too_short_tokens` — fewer model tokens than `minTokens`.
 * - `keep_everything` — `minImportance <= 0`, so nothing could be removed.
 */
export type CompressionGate = "too_short_words" | "too_short_tokens" | "keep_everything";

/**
 * What compression cost and produced. Narrows the native `gate: string` to
 * {@link CompressionGate} so `stats.gate` can be switched on exhaustively.
 */
export interface CompressionStats extends Omit<NativeCompressionStats, "gate"> {
  /** Set when the input was returned verbatim; absent when it was compressed. */
  gate?: CompressionGate;
}

/** A compressed prompt plus the evidence for every decision made. */
export interface CompressedPrompt extends Omit<NativeCompressedPrompt, "stats"> {
  /** Counts and gating for this call. */
  stats: CompressionStats;
}

/** Options for {@link ExperimentalCompression}. */
export interface ExperimentalCompressionOptions {
  /** Model to compress with. Defaults to the pinned built-in LLMLingua-2 checkpoint. */
  model?: CompressionModelSpec;
  /** Defaults applied to every {@link ExperimentalCompression.compress} call. */
  defaults?: CompressionOptions;
}

/**
 * Normalize the public model spec into the flat shape the native layer takes.
 *
 * Every key present is forwarded rather than being selected by which variant
 * the type system thinks this is: **core is the single validator** (it is shared
 * with the Python SDK), and a plain-JS caller with no types can hand us a
 * conflicting object that must be rejected rather than silently half-read.
 */
function toNativeModel(spec: CompressionModelSpec | undefined) {
  if (spec === undefined) return undefined;
  if (typeof spec === "string") return { spec };
  const source = spec as Partial<
    Record<"huggingface" | "local" | "revision", string> & { download: boolean }
  >;
  return {
    huggingface: source.huggingface,
    local: source.local,
    revision: source.revision,
    download: source.download,
  };
}

/**
 * A `RegExp` crosses as its source pattern. Rust's `regex` crate is used
 * natively, which has no lookaround or backreferences — a JS pattern using
 * those throws from the native layer rather than silently matching differently.
 */
function toNativeOptions(options: CompressionOptions | undefined) {
  if (options === undefined) return undefined;
  return {
    minImportance: options.minImportance,
    minWords: options.minWords,
    minTokens: options.minTokens,
    maxChunks: options.maxChunks,
    protect: options.protect?.map((p) =>
      typeof p === "string" ? { literal: p } : { regex: p.source },
    ),
    protectNumbers: options.protectNumbers,
    protectNegations: options.protectNegations,
    negationTerms: options.negationTerms,
    preserveParagraphs: options.preserveParagraphs,
    explain: options.explain,
  };
}

/**
 * **Experimental.** Compresses prose you carry into the context — transcripts,
 * retrieved passages, documents — with an LLMLingua-2 token classifier plus a
 * policy layer that keeps the loss visible.
 *
 * Orthogonal to {@link ToolCatalog} and {@link SkillCatalog}: it holds no
 * catalog state, and constructing one touches no disk and no network.
 *
 * @example
 * ```ts
 * import { ExperimentalCompression } from "@ratel-ai/sdk";
 *
 * const compressor = new ExperimentalCompression();
 * await compressor.preload(); // ~700 MB, once, not inside a request
 *
 * const result = await compressor.compress(transcript, {
 *   minImportance: 0.5,
 *   protect: [/\$[\d,]+/],
 * });
 * console.log(result.text);
 * console.log(result.stats.modelTokensIn, "->", result.stats.modelTokensOut);
 * ```
 *
 * ## What to compress
 *
 * Compress the **context**; leave your instructions and the user's question
 * verbatim. Prompts shorter than `minTokens` come back untouched with
 * `stats.gate` saying which rule fired — short text has no redundancy to spend,
 * and raising the rate does not recover it.
 *
 * ## What it costs
 *
 * The model is ~700 MB and a 623-token document takes roughly 2 seconds on a
 * laptop CPU. Call {@link ExperimentalCompression.preload} at startup.
 *
 * ## What it does not promise
 *
 * Compression is lossy. `kept`/`dropped` scores and the gate make the loss
 * visible and steerable; they do not make it safe. Evaluate it on your own
 * data. Shipped behind an `Experimental` marker — the API may change until it
 * graduates. See ADR-0016.
 */
export class ExperimentalCompression {
  readonly #native: NativePromptCompressor;
  readonly #defaults: CompressionOptions;

  /**
   * @param options - Model selection and per-call defaults.
   * @throws If the model configuration is invalid — thrown here, at
   * construction, rather than on the first compression.
   */
  constructor(options: ExperimentalCompressionOptions = {}) {
    this.#native = new NativePromptCompressor(toNativeModel(options.model));
    this.#defaults = options.defaults ?? {};
  }

  /**
   * Load (and, for the default model, download) the weights now, so the first
   * real call is not charged the cold load. Idempotent and safe to race.
   *
   * Worth calling at startup in any server: a request that pays a ~700 MB load
   * inline stalls for seconds.
   */
  async preload(): Promise<void> {
    await this.#native.preload();
  }

  /**
   * Compress `text`, spending roughly `rate` of its tokens.
   *
   * Returns the input verbatim with `stats.gate` set when it is too short to
   * compress — and does so before loading a model.
   *
   * @param text - The prose to compress. Not your system prompt.
   * @param options - Overrides merged over this instance's defaults.
   */
  async compress(text: string, options?: CompressionOptions): Promise<CompressedPrompt> {
    const merged = { ...this.#defaults, ...options };
    return (await this.#native.compress(text, toNativeOptions(merged))) as CompressedPrompt;
  }

  /**
   * Per-unit importance with **no policy applied** — the raw signal, for
   * inspecting a compression or writing your own selection rule.
   */
  async score(text: string): Promise<WordScore[]> {
    return await this.#native.score(text);
  }
}
