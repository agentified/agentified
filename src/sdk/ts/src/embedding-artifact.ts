import { readFile, writeFile } from "node:fs/promises";
import { mergeEmbeddingArtifacts, type Skill, type Tool } from "../native/index.cjs";
import type { EmbeddingSpec } from "./catalog.js";
import { mapArtifactError } from "./errors.js";
import { SkillRegistry, ToolRegistry } from "./registry.js";

/**
 * Build-time embedding artifact for catalog warm (ADR-0017). Exactly one of
 * `path` or `bytes`. Default `onMiss` is `"error"`.
 */
export type ExperimentalEmbeddingArtifact =
  | {
      /** Filesystem path to an embedding artifact. */
      readonly path: string;
      /** @ignore */
      readonly bytes?: never;
      /** `"error"` (default) or `"embed"` missing corpus ids only. */
      readonly onMiss?: "error" | "embed";
    }
  | {
      /** In-memory embedding artifact bytes. */
      readonly bytes: Uint8Array;
      /** @ignore */
      readonly path?: never;
      /** `"error"` (default) or `"embed"` missing corpus ids only. */
      readonly onMiss?: "error" | "embed";
    };

/** Options for {@link experimentalBuildEmbeddingArtifact}. */
export interface ExperimentalBuildEmbeddingArtifactOptions {
  /** Output path (`fs.writeFile`; parent must exist). */
  output: string;
  /** Same as catalogs; omit for the built-in default model. */
  embedding?: EmbeddingSpec;
  /** Tool metadata (no `execute`). Empty/omitted is valid. */
  tools?: readonly Tool[];
  /** Skill corpus. Empty/omitted is valid; both empty → empty RAT1. */
  skills?: readonly Skill[];
}

/**
 * Build one mixed Tool+Skill RAT1 and write it to `output`. Registers metadata
 * on BM25 registries so each document is embedded once (not via catalog
 * `buildDense`).
 */
export async function experimentalBuildEmbeddingArtifact(
  options: ExperimentalBuildEmbeddingArtifactOptions,
): Promise<void> {
  const tools = options.tools ?? [];
  const skills = options.skills ?? [];
  const embedding = options.embedding;

  const toolRegistry = new ToolRegistry(embedding, "bm25");
  if (tools.length > 0) {
    toolRegistry.registerItems(tools);
  }
  const toolBytes = await toolRegistry.experimentalBuildEmbeddingArtifact();

  const skillRegistry = new SkillRegistry(embedding, "bm25");
  if (skills.length > 0) {
    skillRegistry.registerItems(skills);
  }
  const skillBytes = await skillRegistry.experimentalBuildEmbeddingArtifact();

  const merged = (() => {
    try {
      return mergeEmbeddingArtifacts([toolBytes, skillBytes]);
    } catch (error) {
      throw mapArtifactError(error);
    }
  })();
  await writeFile(options.output, merged);
}

function validateEmbeddingArtifact(config: unknown): {
  path?: string;
  bytes?: Uint8Array;
  onMiss: "error" | "embed";
} {
  if (config === null || typeof config !== "object" || Array.isArray(config)) {
    throw new TypeError("experimentalEmbeddingArtifact must be an object");
  }
  const record = config as Record<string, unknown>;
  const unknown = Object.keys(record).filter(
    (key) => key !== "path" && key !== "bytes" && key !== "onMiss",
  );
  if (unknown.length > 0) {
    throw new TypeError(
      `experimentalEmbeddingArtifact has unknown keys: ${unknown.sort().join(", ")}`,
    );
  }
  const hasPath = Object.hasOwn(record, "path");
  const hasBytes = Object.hasOwn(record, "bytes");
  if (hasPath === hasBytes) {
    throw new TypeError("experimentalEmbeddingArtifact requires exactly one of 'path' or 'bytes'");
  }
  const onMissRaw = Object.hasOwn(record, "onMiss") ? record.onMiss : "error";
  if (onMissRaw !== "error" && onMissRaw !== "embed") {
    throw new Error(
      `unknown on-artifact-miss policy ${JSON.stringify(onMissRaw)} (expected "error" or "embed")`,
    );
  }
  if (hasPath) {
    if (typeof record.path !== "string") {
      throw new TypeError("experimentalEmbeddingArtifact path must be a string");
    }
    return { path: record.path, onMiss: onMissRaw };
  }
  const raw = record.bytes;
  if (!(raw instanceof Uint8Array)) {
    throw new TypeError("experimentalEmbeddingArtifact bytes must be a Uint8Array or Buffer");
  }
  return { bytes: raw, onMiss: onMissRaw };
}

export async function resolveEmbeddingArtifact(
  config: ExperimentalEmbeddingArtifact,
): Promise<{ bytes: Buffer; onMiss: "error" | "embed" }> {
  const validated = validateEmbeddingArtifact(config);
  if (validated.path !== undefined) {
    return { bytes: await readFile(validated.path), onMiss: validated.onMiss };
  }
  const bytes = validated.bytes;
  if (bytes === undefined) {
    throw new TypeError("experimentalEmbeddingArtifact requires exactly one of 'path' or 'bytes'");
  }
  return { bytes: Buffer.from(bytes), onMiss: validated.onMiss };
}
