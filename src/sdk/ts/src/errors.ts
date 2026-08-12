/**
 * Typed embedding errors — the TypeScript twin of the Python SDK's
 * `EmbedderError` / `DimensionMismatchError` (`ratel_ai/exceptions.py`).
 *
 * The native binding surfaces every failure as a plain {@link Error} whose
 * message is the core `EmbedderError` display string. {@link mapEmbedderError}
 * recognizes the embedding failures among them (by their stable message
 * signatures) and re-raises them as these typed classes so callers can branch on
 * `instanceof` / `code` instead of matching message text. Non-embedding errors
 * (registry-busy, lock-poison, the sync "use searchAsync" guard, and
 * construction-time config errors) are passed through unchanged.
 *
 * {@link ArtifactWarmError} / {@link mapArtifactWarmError} cover failures from
 * `experimentalWarmEmbeddingsFromArtifact` (ADR-0017), decoded from a private
 * native→TS envelope (not part of the public API).
 *
 * {@link ArtifactError} / {@link IncompatibleMergeError} / {@link mapArtifactError}
 * cover build-time artifact encode/decode/merge failures from
 * `experimentalBuildEmbeddingArtifact` and internal merge, decoded from a
 * separate private envelope.
 */

/**
 * An embedding model failed to load, download, or run — the base class for every
 * dense-retrieval failure raised from `register` / `searchAsync` on a
 * `"semantic"`/`"hybrid"` catalog. Mirrors Python's `EmbedderError`.
 */
export class EmbedderError extends Error {
  /**
   * Stable machine-readable discriminant — one of `"Load"`, `"Download"`,
   * `"NotCached"`, `"ModelMismatch"`, `"DimensionMismatch"`,
   * `"EmbeddingsNotBuilt"`, `"Inference"`, or `"CacheUnwritable"`. Prefer this (or
   * `instanceof`) over parsing {@link Error.message}.
   */
  readonly code: string;

  /**
   * @param message - The underlying failure description (the core error text).
   * @param code - The stable {@link EmbedderError.code} discriminant.
   */
  constructor(message: string, code: string) {
    super(message);
    this.name = "EmbedderError";
    this.code = code;
  }
}

/**
 * A vector's dimension did not match the embedding cache's — the model changed
 * under an existing corpus. A subclass of {@link EmbedderError} (so
 * `instanceof EmbedderError` still catches it), mirroring Python's
 * `DimensionMismatchError`. Its {@link EmbedderError.code} is `"DimensionMismatch"`.
 */
export class DimensionMismatchError extends EmbedderError {
  /**
   * @param message - The underlying dimension-mismatch description.
   */
  constructor(message: string) {
    super(message, "DimensionMismatch");
    this.name = "DimensionMismatchError";
  }
}

/**
 * Build-time embedding artifact encode/decode/merge failure (non-embedder
 * {@link ArtifactError} variants). Prefer {@link ArtifactError.code} over
 * parsing {@link Error.message}.
 */
export class ArtifactError extends Error {
  /**
   * Stable machine-readable discriminant — exact Rust variant name (for example
   * `"IncompatibleMerge"` or `"VectorNotNormalized"`). Prefer this (or
   * `instanceof`) over parsing {@link Error.message}.
   */
  readonly code:
    | "TooShort"
    | "InvalidMagic"
    | "UnsupportedFormatVersion"
    | "ChecksumMismatch"
    | "CorruptPayload"
    | "InconsistentVectorWidth"
    | "VectorNotNormalized"
    | "NonEmptyZeroDim"
    | "InvalidVector"
    | "IncompatibleMerge";

  /**
   * @param message - The underlying failure description (the core error text).
   * @param code - The stable {@link ArtifactError.code} discriminant.
   */
  constructor(message: string, code: ArtifactError["code"]) {
    super(message);
    this.name = "ArtifactError";
    this.code = code;
  }
}

/**
 * Valid RAT1 parts that cannot be merged (header mismatch or duplicate
 * kind+id). A subclass of {@link ArtifactError}; its {@link ArtifactError.code}
 * is `"IncompatibleMerge"`.
 */
export class IncompatibleMergeError extends ArtifactError {
  /**
   * @param message - Why the parts cannot be combined.
   */
  constructor(message: string) {
    super(message, "IncompatibleMerge");
    this.name = "IncompatibleMergeError";
  }
}

/**
 * Failure warming a dense cache from a build-time embedding artifact
 * (`experimentalWarmEmbeddingsFromArtifact`). Prefer {@link ArtifactWarmError.code} over
 * parsing {@link Error.message}; for `"Incomplete"`, use
 * {@link ArtifactWarmError.missing}.
 */
export class ArtifactWarmError extends Error {
  /**
   * Stable discriminant — `"Warm"` (parse / model mismatch during warm),
   * `"Incomplete"` (corpus ids not covered, `onMiss: "error"`), or `"Embedder"`
   * (follow-up embed failed under `onMiss: "embed"`).
   */
  readonly code: "Warm" | "Incomplete" | "Embedder";

  /**
   * Corpus ids not reused from the artifact. Set only when
   * {@link ArtifactWarmError.code} is `"Incomplete"`.
   */
  readonly missing?: string[];

  /**
   * @param message - The underlying failure description (the core error text).
   * @param code - The stable {@link ArtifactWarmError.code} discriminant.
   * @param missing - Missing corpus ids when `code` is `"Incomplete"`.
   */
  constructor(message: string, code: "Warm" | "Incomplete" | "Embedder", missing?: string[]) {
    super(message);
    this.name = "ArtifactWarmError";
    this.code = code;
    if (missing !== undefined) this.missing = missing;
  }
}

/** Appended to a "not built" error — the signature of a forgotten `await` on a
 * corpus mutation. Both mutations embed on await, so both can leave this state. */
const AWAIT_MUTATION_HINT =
  " — if you called register(...) or replaceAll(...) without awaiting it, the " +
  "dense preparation did not complete; await the call (`await catalog.register(...)` / " +
  "`await catalog.replaceAll(...)`) before a semantic/hybrid search";

/**
 * Classify a native error message by matching the core `EmbedderError` display
 * signatures. Returns the stable code, or `undefined` when the message is not an
 * embedding failure (so the caller passes it through untouched).
 */
function embedderCode(message: string): string | undefined {
  if (message.includes("embedding dimension mismatch")) return "DimensionMismatch";
  if (message.includes("embedding model mismatch")) return "ModelMismatch";
  if (message.includes("is not in the local HuggingFace cache")) return "NotCached";
  if (message.includes("not computed for semantic search")) return "EmbeddingsNotBuilt";
  if (message.startsWith("failed to load embedding model")) return "Load";
  if (message.startsWith("failed to download embedding model")) return "Download";
  if (message.startsWith("embedding model cache is not writable")) return "CacheUnwritable";
  if (message.startsWith("embedding failed:")) return "Inference";
  return undefined;
}

/**
 * Re-raise a native embedding failure as a typed {@link EmbedderError} /
 * {@link DimensionMismatchError}, preserving the original message (and appending
 * the await-register hint to a "not built" error). Any error that is not a
 * recognized embedding failure is returned unchanged.
 *
 * @param error - The error thrown by the native binding.
 * @returns The typed embedding error, or `error` unchanged when it is not one.
 */
export function mapEmbedderError(error: unknown): unknown {
  if (!(error instanceof Error)) return error;
  const code = embedderCode(error.message);
  if (code === undefined) return error;
  const message =
    code === "EmbeddingsNotBuilt" ? error.message + AWAIT_MUTATION_HINT : error.message;
  return code === "DimensionMismatch"
    ? new DimensionMismatchError(message)
    : new EmbedderError(message, code);
}

/** Private NAPI→TS transport prefix — must match native `ARTIFACT_WARM_ERROR_PREFIX`. */
const ARTIFACT_WARM_ERROR_PREFIX = "RATEL_ARTIFACT_WARM_ERROR:";

/** Private NAPI→TS transport prefix — must match native `ARTIFACT_ERROR_PREFIX`. */
const ARTIFACT_ERROR_PREFIX = "RATEL_ARTIFACT_ERROR:";

const ARTIFACT_ERROR_CODES: ReadonlySet<string> = new Set([
  "TooShort",
  "InvalidMagic",
  "UnsupportedFormatVersion",
  "ChecksumMismatch",
  "CorruptPayload",
  "InconsistentVectorWidth",
  "VectorNotNormalized",
  "NonEmptyZeroDim",
  "InvalidVector",
  "IncompatibleMerge",
]);

type ArtifactWarmCode = "Warm" | "Incomplete" | "Embedder";

function isStringArray(value: unknown): value is string[] {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

/**
 * Decode the private native artifact-warm envelope into a typed
 * {@link ArtifactWarmError}. Malformed envelopes and non-warm errors are
 * returned unchanged.
 *
 * @param error - The error thrown by the native binding.
 * @returns The typed warm error, or `error` unchanged when it is not one.
 */
export function mapArtifactWarmError(error: unknown): unknown {
  if (!(error instanceof Error)) return error;
  if (!error.message.startsWith(ARTIFACT_WARM_ERROR_PREFIX)) return error;
  const raw = error.message.slice(ARTIFACT_WARM_ERROR_PREFIX.length);
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return error;
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return error;
  const record = parsed as Record<string, unknown>;
  const code = record.code;
  if (code !== "Warm" && code !== "Incomplete" && code !== "Embedder") return error;
  if (typeof record.message !== "string") return error;

  if (code === "Incomplete") {
    if (!isStringArray(record.missing)) return error;
    return new ArtifactWarmError(record.message, "Incomplete", record.missing);
  }
  // Warm / Embedder: `missing` must be absent.
  if ("missing" in record) return error;
  return new ArtifactWarmError(record.message, code as ArtifactWarmCode);
}

/**
 * Decode the private native artifact-build envelope into a typed
 * {@link ArtifactError} / {@link IncompatibleMergeError}. Malformed envelopes
 * and non-artifact errors are returned unchanged.
 *
 * @param error - The error thrown by the native binding.
 * @returns The typed artifact error, or `error` unchanged when it is not one.
 */
export function mapArtifactError(error: unknown): unknown {
  if (!(error instanceof Error)) return error;
  if (!error.message.startsWith(ARTIFACT_ERROR_PREFIX)) return error;
  const raw = error.message.slice(ARTIFACT_ERROR_PREFIX.length);
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return error;
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return error;
  const record = parsed as Record<string, unknown>;
  const code = record.code;
  if (typeof code !== "string" || !ARTIFACT_ERROR_CODES.has(code)) return error;
  if (typeof record.message !== "string") return error;
  return code === "IncompatibleMerge"
    ? new IncompatibleMergeError(record.message)
    : new ArtifactError(record.message, code as ArtifactError["code"]);
}

/**
 * Re-raise native artifact-build failures: embedder errors first, then typed
 * {@link ArtifactError}. Any unrecognized error is returned unchanged.
 *
 * @param error - The error thrown by the native binding.
 * @returns The typed error, or `error` unchanged when it is not recognized.
 */
export function mapArtifactBuildError(error: unknown): unknown {
  const embedder = mapEmbedderError(error);
  if (embedder !== error) return embedder;
  return mapArtifactError(error);
}
