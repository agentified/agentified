import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { resolveEmbeddingArtifact } from "./embedding-artifact.js";
import * as sdk from "./index.js";
import {
  ArtifactError,
  ArtifactWarmError,
  EmbedderError,
  experimentalBuildEmbeddingArtifact,
  IncompatibleMergeError,
  IntentGraph,
  ratel,
  type Skill,
  SkillCatalog,
  SkillRegistry,
  type Tool,
  ToolCatalog,
  ToolRegistry,
} from "./index.js";
import { startDelayedEmbeddingServer } from "./test-support/delayed-embedding-server.js";

const readFileTool: Tool = {
  id: "read_file",
  name: "read_file",
  description: "Read a file from local disk and return its textual contents.",
  inputSchema: { properties: { path: { type: "string" } } },
  outputSchema: {},
};

const writeFileTool: Tool = {
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

const searchTool: Tool = {
  id: "search",
  name: "search",
  description: "Search the tool catalog for matching tools.",
  inputSchema: { properties: { q: { type: "string" } } },
  outputSchema: {},
};

const searchSkill: Skill = {
  id: "search",
  name: "search",
  description: "Search skill playbooks for matching guidance.",
  tags: ["search"],
  body: "# Search skill",
};

describe("experimentalBuildEmbeddingArtifact", () => {
  const dirs: string[] = [];

  afterEach(async () => {
    await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
  });

  async function tempDir(): Promise<string> {
    const dir = await mkdtemp(join(tmpdir(), "ratel-artifact-"));
    dirs.push(dir);
    return dir;
  }

  it("does not export mergeEmbeddingArtifacts from the package entry", () => {
    expect(sdk).not.toHaveProperty("mergeEmbeddingArtifacts");
  });

  it("exports ArtifactError and IncompatibleMergeError from the package entry", () => {
    expect(sdk).toHaveProperty("ArtifactError");
    expect(sdk).toHaveProperty("IncompatibleMergeError");
    expect(sdk.ArtifactError).toBe(ArtifactError);
    expect(sdk.IncompatibleMergeError).toBe(IncompatibleMergeError);
  });

  it("builds one mixed artifact that warms both registry kinds", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "catalog.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool, writeFileTool],
        skills: [slides, apiDesign],
      });
      const bytes = await readFile(output);
      expect(bytes.byteLength).toBeGreaterThan(0);

      const tools = new ToolRegistry(embedding, "bm25");
      tools.registerItems([readFileTool, writeFileTool]);
      await tools.experimentalWarmEmbeddingsFromArtifact(bytes, "error");
      const toolHits = await tools.searchWithMethodAsync("read a file", 5, "direct", "semantic");
      expect(toolHits[0]?.toolId).toBe("read_file");

      const skills = new SkillRegistry(embedding, "bm25");
      skills.registerItems([slides, apiDesign]);
      await skills.experimentalWarmEmbeddingsFromArtifact(bytes, "error");
      const skillHits = await skills.searchWithMethodAsync(
        "html presentations",
        5,
        "direct",
        "semantic",
      );
      expect(skillHits[0]?.skillId).toBe("frontend-slides");
    } finally {
      await server.close();
    }
  });

  it("embeds each document at most once during build", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      await experimentalBuildEmbeddingArtifact({
        output: join(dir, "once.ratel-embeddings"),
        embedding,
        tools: [readFileTool],
        skills: [slides],
      });
      // Tool batch + skill batch only (no catalog buildDense).
      expect(server.requests.length).toBe(2);
    } finally {
      await server.close();
    }
  });

  it("supports tool-only and skill-only builds", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const toolsOut = join(dir, "tools.ratel-embeddings");
      const skillsOut = join(dir, "skills.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output: toolsOut,
        embedding,
        tools: [readFileTool],
      });
      await experimentalBuildEmbeddingArtifact({
        output: skillsOut,
        embedding,
        skills: [slides],
      });
      expect((await readFile(toolsOut)).byteLength).toBeGreaterThan(0);
      expect((await readFile(skillsOut)).byteLength).toBeGreaterThan(0);
    } finally {
      await server.close();
    }
  });

  it("writes a valid empty RAT1 when tools and skills are empty", async () => {
    const dir = await tempDir();
    const output = join(dir, "empty.ratel-embeddings");
    await experimentalBuildEmbeddingArtifact({
      output,
      tools: [],
      skills: [],
    });
    const bytes = await readFile(output);
    expect(bytes.subarray(0, 4).toString("utf8")).toBe("RAT1");
    expect(bytes.byteLength).toBeGreaterThan(0);

    const tools = new ToolRegistry(undefined, "bm25");
    await expect(
      tools.experimentalWarmEmbeddingsFromArtifact(bytes, "error"),
    ).resolves.toBeUndefined();
  });

  it("overwrites an existing output file with fs.writeFile semantics", async () => {
    const dir = await tempDir();
    const output = join(dir, "overwrite.ratel-embeddings");
    await writeFile(output, "stale");
    await experimentalBuildEmbeddingArtifact({ output, tools: [], skills: [] });
    const bytes = await readFile(output);
    expect(bytes.subarray(0, 4).toString("utf8")).toBe("RAT1");
  });

  it("propagates missing parent directory as a Node filesystem error", async () => {
    await expect(
      experimentalBuildEmbeddingArtifact({
        output: join(tmpdir(), "ratel-missing-parent", "nope", "out.ratel-embeddings"),
        tools: [],
        skills: [],
      }),
    ).rejects.toMatchObject({ code: "ENOENT" });
  });

  it("raises IncompatibleMergeError when tool and skill batches stamp different models", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      await server.setResponseModels(["model-a", "model-b"]);
      const dir = await tempDir();
      const error = await experimentalBuildEmbeddingArtifact({
        output: join(dir, "merge-fail.ratel-embeddings"),
        embedding,
        tools: [readFileTool],
        skills: [slides],
      }).then(
        () => undefined,
        (e: unknown) => e,
      );
      expect(error).toBeInstanceOf(IncompatibleMergeError);
      expect(error).toBeInstanceOf(ArtifactError);
      expect((error as IncompatibleMergeError).code).toBe("IncompatibleMerge");
      expect(error).not.toBeInstanceOf(EmbedderError);
    } finally {
      await server.close();
    }
  });

  it("header model mismatch is ArtifactWarmError Warm naming the artifact", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const source = new ToolRegistry({ url: server.url, model: "model-a" }, "bm25");
      source.registerItems([readFileTool]);
      const bytes = await source.experimentalBuildEmbeddingArtifact();

      const target = new ToolRegistry({ url: server.url, model: "model-b" }, "bm25");
      target.registerItems([readFileTool]);
      const error = await target.experimentalWarmEmbeddingsFromArtifact(bytes, "error").then(
        () => undefined,
        (e: unknown) => e,
      );
      expect(error).toBeInstanceOf(ArtifactWarmError);
      expect((error as ArtifactWarmError).code).toBe("Warm");
      const message = String(error);
      expect(message).toContain("embedding artifact was built with");
      expect(message).toContain("rebuild the artifact");
    } finally {
      await server.close();
    }
  });
});

describe("catalog experimentalEmbeddingArtifact", () => {
  const dirs: string[] = [];

  afterEach(async () => {
    await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
  });

  async function tempDir(): Promise<string> {
    const dir = await mkdtemp(join(tmpdir(), "ratel-artifact-rt-"));
    dirs.push(dir);
    return dir;
  }

  function corpusInputs(requests: string[][]): Set<string> {
    return new Set(requests.flat());
  }

  function assertCorpusNotReembedded(corpus: Set<string>, after: string[][]): void {
    for (const batch of after) {
      for (const text of batch) {
        expect(corpus.has(text)).toBe(false);
      }
    }
  }

  it("warms from path with default onMiss error and avoids document inference", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "catalog.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
        skills: [slides],
      });
      const buildRequests = server.requests.length;

      const tools = new ToolCatalog({
        method: "hybrid",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      await tools.register({
        ...readFileTool,
        execute: async () => ({}),
      });
      expect(server.requests.length).toBe(buildRequests);

      const hits = await tools.searchAsync("read a file", 5);
      expect(hits[0]?.toolId).toBe("read_file");
    } finally {
      await server.close();
    }
  });

  it("accepts Uint8Array bytes and fails Incomplete on partial artifact with onMiss error", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "partial.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });
      const bytes = new Uint8Array(await readFile(output));

      const catalog = new ToolCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { bytes },
      });
      const error = await catalog
        .register([
          { ...readFileTool, execute: async () => ({}) },
          { ...writeFileTool, execute: async () => ({}) },
        ])
        .then(
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

  it("onMiss embed fills only missing ids", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "partial.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });
      const afterBuild = server.requests.length;
      const bytes = await readFile(output);

      const catalog = new ToolCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { bytes, onMiss: "embed" },
      });
      await catalog.register([
        { ...readFileTool, execute: async () => ({}) },
        { ...writeFileTool, execute: async () => ({}) },
      ]);
      expect(server.requests.length).toBe(afterBuild + 1);
      const hits = await catalog.searchAsync("write a file", 5);
      expect(hits.map((h) => h.toolId)).toContain("write_file");
      expect(hits.map((h) => h.toolId)).toContain("read_file");
    } finally {
      await server.close();
    }
  });

  it("one mixed artifact via ratel() serves ToolCatalog and SkillCatalog", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "mixed.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [searchTool],
        skills: [searchSkill],
      });
      const buildRequests = server.requests.length;

      const core = ratel({
        method: "hybrid",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      await core.tools.register({ ...searchTool, execute: async () => ({}) });
      await core.skills.register(searchSkill);
      expect(server.requests.length).toBe(buildRequests);

      const toolHits = await core.tools.searchAsync("matching tools", 5);
      expect(toolHits[0]?.toolId).toBe("search");
      const skillHits = await core.skills.searchAsync("matching guidance", 5);
      expect(skillHits[0]?.skillId).toBe("search");
    } finally {
      await server.close();
    }
  });

  it("ratel() shared artifact is fail-closed per kind corpus coverage (and recoverable)", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "tools-only.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool, writeFileTool],
        skills: [],
      });

      const expectedSkillId = slides.id;

      const core = ratel({
        method: "hybrid",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      await core.tools.register({ ...readFileTool, execute: async () => ({}) });

      let caught: unknown;
      try {
        await core.skills.register(slides);
      } catch (e: unknown) {
        caught = e;
      }
      expect(caught).toBeInstanceOf(ArtifactWarmError);
      expect((caught as ArtifactWarmError).code).toBe("Incomplete");
      expect((caught as ArtifactWarmError).missing).toEqual([expectedSkillId]);
      expect(caught).not.toBeInstanceOf(EmbedderError);

      const recovered = ratel({
        method: "hybrid",
        embedding,
        experimentalEmbeddingArtifact: { path: output, onMiss: "embed" },
      });
      await recovered.tools.register({ ...readFileTool, execute: async () => ({}) });
      await recovered.skills.register(slides);

      const skillHits = await recovered.skills.searchAsync(
        "html presentations",
        5,
        "direct",
        "semantic",
      );
      expect(skillHits[0]?.skillId).toBe(expectedSkillId);
    } finally {
      await server.close();
    }
  });

  it("bm25 catalog with explicit artifact still warms for semantic override", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "bm25.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });

      const catalog = new ToolCatalog({
        method: "bm25",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      await catalog.register({ ...readFileTool, execute: async () => ({}) });
      const hits = await catalog.searchAsync("read a file", 5, "direct", "semantic");
      expect(hits[0]?.toolId).toBe("read_file");
    } finally {
      await server.close();
    }
  });

  it("SkillCatalog.replaceAll warms the configured artifact", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "skills.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        skills: [slides, apiDesign],
      });
      const buildRequests = server.requests.length;

      const catalog = new SkillCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      const reload = catalog.replaceAll([slides, apiDesign]);
      expect(reload.added).toBe(2);
      await reload;
      expect(server.requests.length).toBe(buildRequests);
      const hits = await catalog.searchAsync("html presentations", 5);
      expect(hits[0]?.skillId).toBe("frontend-slides");
    } finally {
      await server.close();
    }
  });

  it("re-warms on every register with zero document inference (TS-1)", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "both-tools.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool, writeFileTool],
      });
      // Artifact holds A+B, but warm only applies entries in the current corpus.
      // First register ({A}) must not populate B; the second register must warm B
      // from the artifact (no document inference) — search for B proves it.
      const corpus = corpusInputs(server.requests);
      const afterBuild = server.requests.length;

      const catalog = new ToolCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      await catalog.register({ ...readFileTool, execute: async () => ({}) });
      assertCorpusNotReembedded(corpus, server.requests.slice(afterBuild));
      const hitsAfterA = await catalog.searchAsync("write a file", 5);
      expect(hitsAfterA.map((h) => h.toolId)).not.toContain("write_file");

      const afterFirst = server.requests.length;
      await catalog.register({ ...writeFileTool, execute: async () => ({}) });
      assertCorpusNotReembedded(corpus, server.requests.slice(afterFirst));

      const hits = await catalog.searchAsync("write a file", 5);
      expect(hits.map((h) => h.toolId)).toContain("write_file");
    } finally {
      await server.close();
    }
  });

  it("re-reads path-backed artifact bytes on every register (TS-2)", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "path-swap.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });
      const artifactConfig = { path: output };

      const catalog = new ToolCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: artifactConfig,
      });
      await catalog.register({ ...readFileTool, execute: async () => ({}) });
      const hitsAfterA = await catalog.searchAsync("write a file", 5);
      expect(hitsAfterA.map((h) => h.toolId)).not.toContain("write_file");

      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool, writeFileTool],
      });
      await catalog.register({ ...writeFileTool, execute: async () => ({}) });

      const hits = await catalog.searchAsync("write a file", 5);
      expect(hits.map((h) => h.toolId)).toContain("write_file");
    } finally {
      await server.close();
    }
  });

  it("incremental second register fails closed when coverage is exceeded (TS-3)", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "tool-a-only.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });

      const catalog = new ToolCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      await catalog.register({ ...readFileTool, execute: async () => ({}) });

      const error = await catalog.register({ ...writeFileTool, execute: async () => ({}) }).then(
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

  it("replaceAll twice re-reads path-backed artifact at the same path (TS-4)", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "skills-path-swap.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        skills: [slides],
      });
      const artifactConfig = { path: output };

      const catalog = new SkillCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: artifactConfig,
      });
      const reload1 = catalog.replaceAll([slides]);
      expect(reload1.added).toBe(1);
      expect(reload1.removed).toBe(0);
      await reload1;

      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        skills: [apiDesign],
      });
      const reload2 = catalog.replaceAll([apiDesign]);
      expect(reload2.added).toBe(1);
      expect(reload2.removed).toBe(1);
      await reload2;

      const hits = await catalog.searchAsync("REST API design", 5, "direct", "semantic");
      expect(hits[0]?.skillId).toBe("api-design");
    } finally {
      await server.close();
    }
  });

  it("rejects a corrupt artifact as ArtifactWarmError Warm", async () => {
    const catalog = new ToolCatalog({
      method: "semantic",
      embedding: { url: "http://127.0.0.1:9/v1/embeddings", model: "x" },
      experimentalEmbeddingArtifact: { bytes: new Uint8Array([1, 2, 3, 4]) },
    });
    const error = await catalog.register({ ...readFileTool, execute: async () => ({}) }).then(
      () => undefined,
      (e: unknown) => e,
    );
    expect(error).toBeInstanceOf(ArtifactWarmError);
    expect((error as ArtifactWarmError).code).toBe("Warm");
  });

  it("rejects missing execute before attempting artifact warm", async () => {
    const catalog = new ToolCatalog({
      method: "semantic",
      experimentalEmbeddingArtifact: { bytes: new Uint8Array([1, 2, 3, 4]) },
    });
    await expect(
      catalog.register({
        ...readFileTool,
        execute: undefined as never,
      }),
    ).rejects.toThrow(/no execute handler/);
  });

  function withExecute(tool: Tool) {
    return { ...tool, execute: async () => ({}) };
  }

  it("artifact register refuses a racing register with registry busy ({ path })", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "tools-path.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool, writeFileTool],
      });
      const catalog = new ToolCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      const settled = await Promise.allSettled([
        catalog.register(withExecute(readFileTool)),
        catalog.register(withExecute(writeFileTool)),
      ]);
      expect(settled.map((r) => r.status)).toEqual(["fulfilled", "rejected"]);
      expect(settled[1]).toMatchObject({
        status: "rejected",
        reason: expect.objectContaining({
          message: expect.stringMatching(/registry busy; await/),
        }),
      });
    } finally {
      await server.close();
    }
  });

  it("artifact register refuses a racing register with registry busy ({ bytes })", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "tools-bytes.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool, writeFileTool],
      });
      const bytes = await readFile(output);
      const catalog = new ToolCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { bytes },
      });
      const settled = await Promise.allSettled([
        catalog.register(withExecute(readFileTool)),
        catalog.register(withExecute(writeFileTool)),
      ]);
      expect(settled.map((r) => r.status)).toEqual(["fulfilled", "rejected"]);
      expect(settled[1]).toMatchObject({
        status: "rejected",
        reason: expect.objectContaining({
          message: expect.stringMatching(/registry busy; await/),
        }),
      });
    } finally {
      await server.close();
    }
  });

  it("SkillCatalog register refuses a racing register during artifact resolution", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "skills-path.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        skills: [slides, apiDesign],
      });
      const catalog = new SkillCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      const settled = await Promise.allSettled([
        catalog.register(slides),
        catalog.register(apiDesign),
      ]);
      expect(settled.map((r) => r.status)).toEqual(["fulfilled", "rejected"]);
      expect(settled[1]).toMatchObject({
        status: "rejected",
        reason: expect.objectContaining({
          message: expect.stringMatching(/registry busy; await/),
        }),
      });
    } finally {
      await server.close();
    }
  });

  it("SkillCatalog.replaceAll refuses a second reload during artifact resolution", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "skills-replace.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        skills: [slides],
      });
      const catalog = new SkillCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      const first = catalog.replaceAll([slides]);
      expect(() => catalog.replaceAll([apiDesign])).toThrow(/registry busy; await/);
      await first;
      expect(catalog.has("frontend-slides")).toBe(true);
      expect(catalog.has("api-design")).toBe(false);
      expect(catalog.size()).toBe(1);
    } finally {
      await server.close();
    }
  });

  it("first replaceAll counts describe the corpus that is actually live", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "skills-counts.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        skills: [slides],
      });
      const catalog = new SkillCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      const first = catalog.replaceAll([slides]);
      expect(first.added).toBe(1);
      expect(first.removed).toBe(0);
      expect(() => catalog.replaceAll([apiDesign])).toThrow(/registry busy; await/);
      await first;
      expect(first.added).toBe(1);
      expect(first.removed).toBe(0);
      expect(catalog.has("frontend-slides")).toBe(true);
      expect(catalog.has("api-design")).toBe(false);
      expect(catalog.size()).toBe(1);
    } finally {
      await server.close();
    }
  });

  it("racing register cannot contaminate the artifact warm corpus", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "tool-a-only-race.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });
      const catalog = new ToolCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      const settled = await Promise.allSettled([
        catalog.register(withExecute(readFileTool)),
        catalog.register(withExecute(writeFileTool)),
      ]);
      expect(settled.map((r) => r.status)).toEqual(["fulfilled", "rejected"]);
      expect(settled[1]).toMatchObject({
        status: "rejected",
        reason: expect.objectContaining({
          message: expect.stringMatching(/registry busy; await/),
        }),
      });
      expect(catalog.has("read_file")).toBe(true);
      expect(catalog.has("write_file")).toBe(false);
    } finally {
      await server.close();
    }
  });

  it("artifact-read failure releases the guard", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const missing = new ToolCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: "/nonexistent/ratel-artifact.rat1" },
      });
      await expect(missing.register(withExecute(readFileTool))).rejects.toThrow(/ENOENT/);
      await expect(missing.register(withExecute(readFileTool))).rejects.toThrow(/ENOENT/);

      const dir = await tempDir();
      const output = join(dir, "after-enoent.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });
      const catalog = new ToolCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      await expect(catalog.register(withExecute(readFileTool))).resolves.toBeUndefined();
    } finally {
      await server.close();
    }
  });

  it("Incomplete warm failure releases the guard", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "partial-cleanup.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });
      const catalog = new ToolCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      const error = await catalog
        .register([withExecute(readFileTool), withExecute(writeFileTool)])
        .then(
          () => undefined,
          (e: unknown) => e,
        );
      expect(error).toBeInstanceOf(ArtifactWarmError);
      expect((error as ArtifactWarmError).code).toBe("Incomplete");
      const second = await catalog.register(withExecute(readFileTool)).then(
        () => undefined,
        (e: unknown) => e,
      );
      expect(String(second)).not.toMatch(/registry busy/);
    } finally {
      await server.close();
    }
  });

  it("sequential register and replaceAll with an artifact succeed", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const toolsOut = join(dir, "seq-tools.ratel-embeddings");
      const skillsOut = join(dir, "seq-skills.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output: toolsOut,
        embedding,
        tools: [readFileTool, writeFileTool],
      });
      await experimentalBuildEmbeddingArtifact({
        output: skillsOut,
        embedding,
        skills: [slides, apiDesign],
      });
      const tools = new ToolCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: toolsOut },
      });
      await tools.register(withExecute(readFileTool));
      await tools.register(withExecute(writeFileTool));
      const skills = new SkillCatalog({
        method: "semantic",
        embedding,
        experimentalEmbeddingArtifact: { path: skillsOut },
      });
      const first = skills.replaceAll([slides]);
      await first;
      const second = skills.replaceAll([slides, apiDesign]);
      expect(second.added).toBe(1);
      await second;
      expect(skills.size()).toBe(2);
    } finally {
      await server.close();
    }
  });
});

/** 2-dim graph matching the stub endpoint width; `model` is a mismatch unless overridden. */
function stubDimGraph(kind: "tool" | "skill", model = "some-other-model"): IntentGraph {
  return IntentGraph.fromJson(
    JSON.stringify({
      v: 1,
      built_from_ts: 1,
      model,
      intents: [
        {
          id: "intent_0",
          label: "l",
          terms: [],
          members: ["read a file"],
          centroid: [1, 0],
          support: 9,
          tools: kind === "tool" ? { read_file: 1.0 } : {},
          skills: kind === "skill" ? { "frontend-slides": 1.0 } : {},
        },
      ],
    }),
  );
}

function withExecute(tool: Tool) {
  return { ...tool, execute: async () => ({}) };
}

describe("artifact warm adaptive-ranking warning", () => {
  const dirs: string[] = [];

  afterEach(async () => {
    await Promise.all(dirs.splice(0).map((dir) => rm(dir, { recursive: true, force: true })));
  });

  async function tempDir(): Promise<string> {
    const dir = await mkdtemp(join(tmpdir(), "ratel-artifact-warn-"));
    dirs.push(dir);
    return dir;
  }

  it("bm25 catalog with artifact warns after warm when the adaptive graph model differs", async () => {
    const server = await startDelayedEmbeddingServer();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "tools.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });
      const catalog = new ToolCatalog({
        method: "bm25",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      catalog.experimentalEnableAdaptiveRanking(stubDimGraph("tool"));
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("active");
      expect(warn).not.toHaveBeenCalled();

      await catalog.register(withExecute(readFileTool));
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("paused: model mismatch");
      expect(warn).toHaveBeenCalledOnce();
      expect(String(warn.mock.calls[0]?.[0])).toContain("experimentalRebuildIntentGraph()");
    } finally {
      warn.mockRestore();
      await server.close();
    }
  });

  it("plain bm25 without an artifact stays active and does not warn", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const catalog = new ToolCatalog({ method: "bm25" });
      catalog.experimentalEnableAdaptiveRanking(stubDimGraph("tool"));
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("active");
      await catalog.register(withExecute(readFileTool));
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("active");
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it("SkillCatalog register with artifact warns once after warm", async () => {
    const server = await startDelayedEmbeddingServer();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "skills.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        skills: [slides],
      });
      const catalog = new SkillCatalog({
        method: "bm25",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      catalog.experimentalEnableAdaptiveRanking(stubDimGraph("skill"));
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("active");
      expect(warn).not.toHaveBeenCalled();

      await catalog.register(slides);
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("paused: model mismatch");
      expect(warn).toHaveBeenCalledOnce();
      expect(String(warn.mock.calls[0]?.[0])).toContain("experimentalRebuildIntentGraph()");
    } finally {
      warn.mockRestore();
      await server.close();
    }
  });

  it("SkillCatalog replaceAll with artifact warns once after warm", async () => {
    const server = await startDelayedEmbeddingServer();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "skills-reload.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        skills: [slides, apiDesign],
      });
      const catalog = new SkillCatalog({
        method: "bm25",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      catalog.experimentalEnableAdaptiveRanking(stubDimGraph("skill"));
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("active");
      expect(warn).not.toHaveBeenCalled();

      const reload = catalog.replaceAll([slides, apiDesign]);
      await reload;
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("paused: model mismatch");
      expect(warn).toHaveBeenCalledOnce();
      expect(String(warn.mock.calls[0]?.[0])).toContain("experimentalRebuildIntentGraph()");
    } finally {
      warn.mockRestore();
      await server.close();
    }
  });

  it("failed incomplete warm does not warn", async () => {
    const server = await startDelayedEmbeddingServer();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "partial.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });
      const bytes = new Uint8Array(await readFile(output));
      const catalog = new ToolCatalog({
        method: "bm25",
        embedding,
        experimentalEmbeddingArtifact: { bytes },
      });
      catalog.experimentalEnableAdaptiveRanking(stubDimGraph("tool"));
      expect(warn).not.toHaveBeenCalled();

      const error = await catalog
        .register([withExecute(readFileTool), withExecute(writeFileTool)])
        .then(
          () => undefined,
          (e: unknown) => e,
        );
      expect(error).toBeInstanceOf(ArtifactWarmError);
      expect((error as ArtifactWarmError).code).toBe("Incomplete");
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
      await server.close();
    }
  });

  it("repeated successful warms warn at most once", async () => {
    const server = await startDelayedEmbeddingServer();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "repeat.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });
      const catalog = new ToolCatalog({
        method: "bm25",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      catalog.experimentalEnableAdaptiveRanking(stubDimGraph("tool"));
      await catalog.register(withExecute(readFileTool));
      expect(warn).toHaveBeenCalledOnce();
      await catalog.register(withExecute(readFileTool));
      expect(warn).toHaveBeenCalledOnce();
    } finally {
      warn.mockRestore();
      await server.close();
    }
  });

  it("stays silent when warnOnModelMismatch is false", async () => {
    const server = await startDelayedEmbeddingServer();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "silent.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });
      const catalog = new ToolCatalog({
        method: "bm25",
        embedding,
        experimentalEmbeddingArtifact: { path: output },
      });
      catalog.experimentalEnableAdaptiveRanking(stubDimGraph("tool"), {
        warnOnModelMismatch: false,
      });
      await catalog.register(withExecute(readFileTool));
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("paused: model mismatch");
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
      await server.close();
    }
  });

  it("matching graph model does not warn after artifact warm", async () => {
    const server = await startDelayedEmbeddingServer();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const embedding = { url: server.url, model: "test-model" };
      const dir = await tempDir();
      const output = join(dir, "match.ratel-embeddings");
      await experimentalBuildEmbeddingArtifact({
        output,
        embedding,
        tools: [readFileTool],
      });
      const bytes = await readFile(output);

      const probe = new ToolCatalog({
        method: "bm25",
        embedding,
        experimentalEmbeddingArtifact: { bytes },
      });
      probe.experimentalEnableAdaptiveRanking(stubDimGraph("tool"), {
        warnOnModelMismatch: false,
      });
      await probe.register(withExecute(readFileTool));
      const active = probe.experimentalAdaptiveRankingStatus.active;
      expect(active).toBeTruthy();

      const catalog = new ToolCatalog({
        method: "bm25",
        embedding,
        experimentalEmbeddingArtifact: { bytes },
      });
      catalog.experimentalEnableAdaptiveRanking(stubDimGraph("tool", active ?? ""));
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("active");
      expect(warn).not.toHaveBeenCalled();
      await catalog.register(withExecute(readFileTool));
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("active");
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
      await server.close();
    }
  });
});

describe("resolveEmbeddingArtifact", () => {
  it("rejects neither path nor bytes", async () => {
    await expect(resolveEmbeddingArtifact({} as never)).rejects.toThrow(
      /exactly one of 'path' or 'bytes'/,
    );
  });

  it("rejects both path and bytes", async () => {
    await expect(
      resolveEmbeddingArtifact({ path: "/tmp/a.rat1", bytes: new Uint8Array([1]) } as never),
    ).rejects.toThrow(/exactly one of 'path' or 'bytes'/);
  });

  it("treats explicit undefined as absence (path/bytes + onMiss)", async () => {
    const dir = await mkdtemp(join(tmpdir(), "ratel-artifact-validate-"));
    const output = join(dir, "a.rat1");
    try {
      await writeFile(output, "RAT1");
      const pathBytes = Buffer.from("RAT1");
      const validBytes = new Uint8Array([1, 2, 3]);

      const fromPath = await resolveEmbeddingArtifact({
        path: output,
        // Explicit undefined counts as “absent”.
        bytes: undefined,
      } as never);
      expect(fromPath.onMiss).toBe("error");
      expect(Buffer.compare(fromPath.bytes, pathBytes)).toBe(0);

      const fromBytes = await resolveEmbeddingArtifact({
        bytes: validBytes,
        // Explicit undefined counts as “absent”.
        path: undefined,
      } as never);
      expect(fromBytes.onMiss).toBe("error");
      expect(fromBytes.bytes).toEqual(Buffer.from(validBytes));

      const onMissDefaulted = await resolveEmbeddingArtifact({
        path: output,
        onMiss: undefined,
      } as never);
      expect(onMissDefaulted.onMiss).toBe("error");

      await expect(
        resolveEmbeddingArtifact({
          path: undefined,
          bytes: undefined,
        } as never),
      ).rejects.toThrow(/experimentalEmbeddingArtifact requires exactly one of 'path' or 'bytes'/);

      await expect(
        resolveEmbeddingArtifact({
          path: output,
          bytes: validBytes,
        } as never),
      ).rejects.toThrow(/experimentalEmbeddingArtifact requires exactly one of 'path' or 'bytes'/);

      await expect(
        resolveEmbeddingArtifact({
          path: output,
          onMiss: "nope",
        } as never),
      ).rejects.toThrow(/unknown on-artifact-miss policy/);

      // `null` is a defined invalid value: it must fail type checks (not be treated as absence).
      await expect(
        resolveEmbeddingArtifact({
          path: null,
          bytes: undefined,
        } as never),
      ).rejects.toThrow(/path must be a string/);

      await expect(
        resolveEmbeddingArtifact({
          bytes: null,
          path: undefined,
        } as never),
      ).rejects.toThrow(/bytes must be a Uint8Array or Buffer/);

      await expect(
        resolveEmbeddingArtifact({
          path: output,
          onMiss: null,
        } as never),
      ).rejects.toThrow(/unknown on-artifact-miss policy/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("rejects unknown keys", async () => {
    await expect(
      resolveEmbeddingArtifact({ path: "/tmp/a.rat1", extra: 1 } as never),
    ).rejects.toThrow(/unknown keys: extra/);
  });

  it("rejects invalid onMiss", async () => {
    await expect(
      resolveEmbeddingArtifact({ bytes: new Uint8Array([1]), onMiss: "retry" } as never),
    ).rejects.toThrow(/unknown on-artifact-miss policy/);
  });

  it("defaults onMiss to error for bytes", async () => {
    const resolved = await resolveEmbeddingArtifact({ bytes: new Uint8Array([1, 2, 3]) });
    expect(resolved.onMiss).toBe("error");
    expect(Buffer.compare(resolved.bytes, Buffer.from([1, 2, 3]))).toBe(0);
  });
});
