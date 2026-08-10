import { readFile, writeFile } from "node:fs/promises";
import { mergeEmbeddingArtifacts, type Skill, type Tool } from "../native/index.cjs";
import type { EmbeddingSpec } from "./catalog.js";
import { mapEmbedderError } from "./errors.js";
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

  try {
    const toolRegistry = new ToolRegistry(embedding, "bm25");
    if (tools.length > 0) {
      toolRegistry.registerItems(tools);
    }
    const toolBytes = await toolRegistry.buildEmbeddingArtifact();

    const skillRegistry = new SkillRegistry(embedding, "bm25");
    if (skills.length > 0) {
      skillRegistry.registerItems(skills);
    }
    const skillBytes = await skillRegistry.buildEmbeddingArtifact();

    const merged = mergeEmbeddingArtifacts([toolBytes, skillBytes]);
    await writeFile(options.output, merged);
  } catch (error) {
    throw mapEmbedderError(error);
  }
}

export async function resolveEmbeddingArtifact(
  config: ExperimentalEmbeddingArtifact,
): Promise<{ bytes: Buffer; onMiss: "error" | "embed" }> {
  const onMiss = config.onMiss ?? "error";
  if (typeof config.path === "string") {
    return { bytes: await readFile(config.path), onMiss };
  }
  return { bytes: Buffer.from(config.bytes), onMiss };
}
