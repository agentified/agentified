import { describe, expect, it } from "vitest";
import {
  ArtifactWarmError,
  EmbedderError,
  type Skill,
  SkillRegistry,
  type Tool,
  ToolRegistry,
} from "./index.js";
import { startDelayedEmbeddingServer } from "./test-support/delayed-embedding-server.js";

const readFile: Tool = {
  id: "read_file",
  name: "read_file",
  description: "Read a file from local disk and return its textual contents.",
  inputSchema: { properties: { path: { type: "string" } } },
  outputSchema: {},
};

const writeFile: Tool = {
  id: "write_file",
  name: "write_file",
  description: "Write textual contents to a file on local disk.",
  inputSchema: {
    properties: {
      path: { type: "string" },
      contents: { type: "string" },
    },
  },
  outputSchema: {},
};

const slides: Skill = {
  id: "frontend-slides",
  name: "frontend-slides",
  description: "Build animation-rich HTML presentations from scratch.",
  tags: ["frontend", "presentations"],
  body: "# Frontend Slides\n\nStep 1: pick an aesthetic.",
};

const apiDesign: Skill = {
  id: "api-design",
  name: "api-design",
  description: "REST API design patterns: resource naming, status codes, pagination.",
  tags: ["backend", "api"],
  body: "# API Design\n\nUse nouns for resources.",
};

describe("ToolRegistry embedding artifacts", () => {
  it("experimentalBuildEmbeddingArtifact returns a non-empty Buffer for a non-empty corpus", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const registry = new ToolRegistry({ url: server.url, model: "test-model" }, "bm25");
      await registry.register([readFile, writeFile]);
      const bytes = await registry.experimentalBuildEmbeddingArtifact();
      expect(Buffer.isBuffer(bytes)).toBe(true);
      expect(bytes.byteLength).toBeGreaterThan(0);
    } finally {
      await server.close();
    }
  });

  it("round-trips build then warm with onMiss error", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const source = new ToolRegistry(embedding, "bm25");
      await source.register([readFile, writeFile]);
      const bytes = await source.experimentalBuildEmbeddingArtifact();

      const target = new ToolRegistry(embedding, "bm25");
      await target.register([readFile, writeFile]);
      await expect(
        target.experimentalWarmEmbeddingsFromArtifact(bytes, "error"),
      ).resolves.toBeUndefined();
      const hits = await target.searchWithMethodAsync("read a file", 5, "direct", "semantic");
      expect(hits[0]?.toolId).toBe("read_file");
    } finally {
      await server.close();
    }
  });

  it("warm with partial coverage and onMiss error yields Incomplete + missing", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const source = new ToolRegistry(embedding, "bm25");
      await source.register(readFile);
      const bytes = await source.experimentalBuildEmbeddingArtifact();

      const target = new ToolRegistry(embedding, "bm25");
      await target.register([readFile, writeFile]);
      const error = await target.experimentalWarmEmbeddingsFromArtifact(bytes, "error").then(
        () => undefined,
        (e: unknown) => e,
      );
      expect(error).toBeInstanceOf(ArtifactWarmError);
      expect((error as ArtifactWarmError).code).toBe("Incomplete");
      expect((error as ArtifactWarmError).missing).toEqual(["write_file"]);
    } finally {
      await server.close();
    }
  });

  it("warm with partial coverage and onMiss embed completes the corpus", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const source = new ToolRegistry(embedding, "bm25");
      await source.register(readFile);
      const bytes = await source.experimentalBuildEmbeddingArtifact();

      const target = new ToolRegistry(embedding, "bm25");
      await target.register([readFile, writeFile]);
      await target.experimentalWarmEmbeddingsFromArtifact(bytes, "embed");
      // Fake endpoint vectors are not text-ranked; prove both ids are embedded.
      const hits = await target.searchWithMethodAsync("anything", 5, "direct", "semantic");
      expect(hits.map((h) => h.toolId).sort()).toEqual(["read_file", "write_file"]);
    } finally {
      await server.close();
    }
  });

  it("model mismatch surfaces as ArtifactWarmError with code Warm", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const source = new ToolRegistry({ url: server.url, model: "model-a" }, "bm25");
      await source.register(readFile);
      const bytes = await source.experimentalBuildEmbeddingArtifact();

      const target = new ToolRegistry({ url: server.url, model: "model-b" }, "bm25");
      await target.register(readFile);
      const error = await target.experimentalWarmEmbeddingsFromArtifact(bytes, "error").then(
        () => undefined,
        (e: unknown) => e,
      );
      expect(error).toBeInstanceOf(ArtifactWarmError);
      expect((error as ArtifactWarmError).code).toBe("Warm");
    } finally {
      await server.close();
    }
  });

  it("maps ArtifactError::Embedder through experimentalBuildEmbeddingArtifact as EmbedderError", async () => {
    const registry = new ToolRegistry(
      { local: "/definitely/missing/ratel-embedding-model" },
      "bm25",
    );
    await registry.register(readFile);
    const error = await registry.experimentalBuildEmbeddingArtifact().then(
      () => undefined,
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(EmbedderError);
    expect((error as EmbedderError).message).not.toMatch(/^embedding artifact build failed:/);
    // Fixture yields a load failure today; assert the mapped code matches that path.
    expect((error as EmbedderError).code).toBe("Load");
  });

  it("does not block concurrent semantic search while experimentalBuildEmbeddingArtifact is in flight", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      // Prefill dense cache so semantic search only needs a query embed (batch size 1).
      const registry = new ToolRegistry(embedding, "semantic");
      await registry.register([readFile, writeFile]);

      await server.armBatchHold(2);
      const build = registry.experimentalBuildEmbeddingArtifact();
      await server.waitForHeld();

      let buildSettled = false;
      void build.finally(() => {
        buildSettled = true;
      });

      const hits = await registry.searchWithMethodAsync("read a file", 5, "direct", "semantic");
      expect(buildSettled).toBe(false);
      expect(hits[0]?.toolId).toBe("read_file");

      server.releaseHeld();
      const bytes = await build;
      expect(bytes.byteLength).toBeGreaterThan(0);
    } finally {
      await server.close();
    }
  });

  it("rejects corpus mutation while experimentalBuildEmbeddingArtifact is pending", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const registry = new ToolRegistry({ url: server.url, model: "test-model" }, "bm25");
      await registry.register([readFile, writeFile]);
      await server.armBatchHold(2);
      const pending = registry.experimentalBuildEmbeddingArtifact();
      // Assert busy immediately after schedule (before any await) so the
      // rejection comes from DenseOperationPermit (pending_dense), not from the
      // worker already holding the registry read lock in compute().
      expect(() => registry.registerItems({ ...writeFile, id: "later" })).toThrow(
        /registry busy; await the active operation/,
      );
      await server.waitForHeld();
      server.releaseHeld();
      await pending;
    } finally {
      await server.close();
    }
  });
});

describe("SkillRegistry embedding artifacts", () => {
  it("experimentalBuildEmbeddingArtifact returns a non-empty Buffer for a non-empty corpus", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const registry = new SkillRegistry({ url: server.url, model: "test-model" }, "bm25");
      await registry.register([slides, apiDesign]);
      const bytes = await registry.experimentalBuildEmbeddingArtifact();
      expect(Buffer.isBuffer(bytes)).toBe(true);
      expect(bytes.byteLength).toBeGreaterThan(0);
    } finally {
      await server.close();
    }
  });

  it("round-trips build then warm with onMiss error", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const source = new SkillRegistry(embedding, "bm25");
      await source.register([slides, apiDesign]);
      const bytes = await source.experimentalBuildEmbeddingArtifact();

      const target = new SkillRegistry(embedding, "bm25");
      await target.register([slides, apiDesign]);
      await expect(
        target.experimentalWarmEmbeddingsFromArtifact(bytes, "error"),
      ).resolves.toBeUndefined();
      const hits = await target.searchWithMethodAsync("slides", 5, "direct", "semantic");
      expect(hits[0]?.skillId).toBe("frontend-slides");
    } finally {
      await server.close();
    }
  });

  it("warm with partial coverage and onMiss error yields Incomplete + missing", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const source = new SkillRegistry(embedding, "bm25");
      await source.register(slides);
      const bytes = await source.experimentalBuildEmbeddingArtifact();

      const target = new SkillRegistry(embedding, "bm25");
      await target.register([slides, apiDesign]);
      const error = await target.experimentalWarmEmbeddingsFromArtifact(bytes, "error").then(
        () => undefined,
        (e: unknown) => e,
      );
      expect(error).toBeInstanceOf(ArtifactWarmError);
      expect((error as ArtifactWarmError).code).toBe("Incomplete");
      expect((error as ArtifactWarmError).missing).toEqual(["api-design"]);
    } finally {
      await server.close();
    }
  });

  it("warm with partial coverage and onMiss embed completes the corpus", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const source = new SkillRegistry(embedding, "bm25");
      await source.register(slides);
      const bytes = await source.experimentalBuildEmbeddingArtifact();

      const target = new SkillRegistry(embedding, "bm25");
      await target.register([slides, apiDesign]);
      await target.experimentalWarmEmbeddingsFromArtifact(bytes, "embed");
      const hits = await target.searchWithMethodAsync("anything", 5, "direct", "semantic");
      expect(hits.map((h) => h.skillId).sort()).toEqual(["api-design", "frontend-slides"]);
    } finally {
      await server.close();
    }
  });

  it("model mismatch surfaces as ArtifactWarmError with code Warm", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const source = new SkillRegistry({ url: server.url, model: "model-a" }, "bm25");
      await source.register(slides);
      const bytes = await source.experimentalBuildEmbeddingArtifact();

      const target = new SkillRegistry({ url: server.url, model: "model-b" }, "bm25");
      await target.register(slides);
      const error = await target.experimentalWarmEmbeddingsFromArtifact(bytes, "error").then(
        () => undefined,
        (e: unknown) => e,
      );
      expect(error).toBeInstanceOf(ArtifactWarmError);
      expect((error as ArtifactWarmError).code).toBe("Warm");
    } finally {
      await server.close();
    }
  });
});

describe("experimental embedding artifact public surface", () => {
  it("exposes experimental registry methods and omits unprefixed names", () => {
    for (const Registry of [ToolRegistry, SkillRegistry]) {
      expect(Registry.prototype).toHaveProperty("experimentalBuildEmbeddingArtifact");
      expect(Registry.prototype).toHaveProperty("experimentalWarmEmbeddingsFromArtifact");
      expect(Registry.prototype).not.toHaveProperty("buildEmbeddingArtifact");
      expect(Registry.prototype).not.toHaveProperty("warmEmbeddingsFromArtifact");
    }
  });
});
