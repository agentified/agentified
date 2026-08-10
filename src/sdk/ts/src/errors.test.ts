import { describe, expect, it } from "vitest";
import { mapArtifactWarmError, mapEmbedderError } from "./errors.js";
import { ArtifactWarmError, DimensionMismatchError, EmbedderError } from "./index.js";

const PREFIX = "RATEL_ARTIFACT_WARM_ERROR:";

function envelope(payload: unknown): Error {
  return new Error(`${PREFIX}${JSON.stringify(payload)}`);
}

describe("mapEmbedderError", () => {
  // Each core `EmbedderError` display signature → its typed class + code.
  const cases: Array<[string, string, string]> = [
    ["Load", "failed to load embedding model /x: missing model.safetensors (hint: …)", "Load"],
    [
      "Download",
      "failed to download embedding model bge: connection refused (hint: …)",
      "Download",
    ],
    [
      "CacheUnwritable",
      "embedding model cache is not writable: permission denied (hint: …)",
      "CacheUnwritable",
    ],
    ["Inference", "embedding failed: tokenizer error (hint: …)", "Inference"],
    [
      "ModelMismatch",
      "embedding model mismatch: cache was built with A, active model is B (hint: …)",
      "ModelMismatch",
    ],
    [
      "NotCached",
      "embedding model bge-base is not in the local HuggingFace cache — download it first (hint: …)",
      "NotCached",
    ],
  ];

  for (const [name, message, code] of cases) {
    it(`maps a ${name} error to EmbedderError with code "${code}"`, () => {
      const mapped = mapEmbedderError(new Error(message));
      expect(mapped).toBeInstanceOf(EmbedderError);
      expect(mapped).not.toBeInstanceOf(DimensionMismatchError);
      expect((mapped as EmbedderError).code).toBe(code);
      expect((mapped as EmbedderError).message).toBe(message); // message preserved verbatim
    });
  }

  it("maps a dimension mismatch to DimensionMismatchError (a subclass of EmbedderError)", () => {
    const message = "embedding dimension mismatch for bge: expected 384, got 768 (hint: …)";
    const mapped = mapEmbedderError(new Error(message));
    expect(mapped).toBeInstanceOf(DimensionMismatchError);
    expect(mapped).toBeInstanceOf(EmbedderError);
    expect((mapped as DimensionMismatchError).code).toBe("DimensionMismatch");
    expect((mapped as DimensionMismatchError).message).toBe(message);
  });

  it("appends the await-register hint to a not-built error, keeping the original text", () => {
    const message = "embeddings are not computed for semantic search (hint: …)";
    const mapped = mapEmbedderError(new Error(message)) as EmbedderError;
    expect(mapped).toBeInstanceOf(EmbedderError);
    expect(mapped.code).toBe("EmbeddingsNotBuilt");
    expect(mapped.message).toContain("not computed for semantic"); // existing matcher
    expect(mapped.message).toContain("without awaiting it"); // added hint
  });

  it("passes non-embedding errors through unchanged", () => {
    for (const message of [
      "registry busy; await the active operation before registering more items",
      "tool registry lock poisoned",
      "semantic and hybrid search are asynchronous; use searchWithMethodAsync()",
      "embedding 'local' must not be blank (hint: …)", // construction-time config error
    ]) {
      const original = new Error(message);
      expect(mapEmbedderError(original)).toBe(original); // same reference, untouched
    }
    const notError = { message: "failed to load embedding model x" };
    expect(mapEmbedderError(notError)).toBe(notError);
  });
});

describe("mapArtifactWarmError", () => {
  it("maps Warm from the private native envelope", () => {
    const human = "embedding model mismatch: cache was built with A, active model is B (hint: …)";
    const original = envelope({ code: "Warm", message: human });
    const mapped = mapArtifactWarmError(original) as ArtifactWarmError;
    expect(mapped).toBeInstanceOf(ArtifactWarmError);
    expect(mapped.code).toBe("Warm");
    expect(mapped.missing).toBeUndefined();
    expect(mapped.message).toBe(human);
    expect(mapped.message).not.toContain(PREFIX);
  });

  it("maps Embedder from the private native envelope", () => {
    const human = "embedding failed: tokenizer error (hint: …)";
    const mapped = mapArtifactWarmError(
      envelope({ code: "Embedder", message: human }),
    ) as ArtifactWarmError;
    expect(mapped).toBeInstanceOf(ArtifactWarmError);
    expect(mapped.code).toBe("Embedder");
    expect(mapped.missing).toBeUndefined();
    expect(mapped.message).toBe(human);
  });

  it("maps Incomplete with multiple normal ids", () => {
    const human =
      "embedding artifact incomplete for the current corpus: 2 id(s) missing (delete_file, write_file)";
    const mapped = mapArtifactWarmError(
      envelope({ code: "Incomplete", message: human, missing: ["delete_file", "write_file"] }),
    ) as ArtifactWarmError;
    expect(mapped).toBeInstanceOf(ArtifactWarmError);
    expect(mapped.code).toBe("Incomplete");
    expect(mapped.missing).toEqual(["delete_file", "write_file"]);
    expect(mapped.message).toBe(human);
  });

  it("preserves delimiter-heavy Incomplete ids exactly", () => {
    const missing = ["foo, bar", "name (legacy)", "x) y"];
    const human =
      "embedding artifact incomplete for the current corpus: 3 id(s) missing (foo, bar, name (legacy), x) y)";
    const mapped = mapArtifactWarmError(
      envelope({ code: "Incomplete", message: human, missing }),
    ) as ArtifactWarmError;
    expect(mapped.code).toBe("Incomplete");
    expect(mapped.missing).toEqual(missing);
  });

  it("rejects Warm/Embedder envelopes that include missing", () => {
    for (const code of ["Warm", "Embedder"] as const) {
      const original = envelope({ code, message: "x", missing: [] });
      expect(mapArtifactWarmError(original)).toBe(original);
      const withNull = envelope({ code, message: "x", missing: null });
      expect(mapArtifactWarmError(withNull)).toBe(withNull);
    }
  });

  it("rejects Incomplete envelopes without a string[] missing", () => {
    const noMissing = envelope({ code: "Incomplete", message: "x" });
    expect(mapArtifactWarmError(noMissing)).toBe(noMissing);
    const badMissing = envelope({ code: "Incomplete", message: "x", missing: [1, 2] });
    expect(mapArtifactWarmError(badMissing)).toBe(badMissing);
  });

  it("returns the original Error for malformed JSON", () => {
    const original = new Error(`${PREFIX}{not-json`);
    expect(mapArtifactWarmError(original)).toBe(original);
  });

  it("returns the original Error for a wrong schema/code", () => {
    const badCode = envelope({ code: "Nope", message: "x" });
    expect(mapArtifactWarmError(badCode)).toBe(badCode);
    const noMessage = envelope({ code: "Warm" });
    expect(mapArtifactWarmError(noMessage)).toBe(noMessage);
  });

  it("passes unrelated errors through unchanged", () => {
    const original = new Error(
      "registry busy; await the active operation before registering more items",
    );
    expect(mapArtifactWarmError(original)).toBe(original);
  });

  it("passes non-Error values through unchanged", () => {
    const notError = { message: `${PREFIX}{"code":"Warm","message":"x"}` };
    expect(mapArtifactWarmError(notError)).toBe(notError);
  });
});
