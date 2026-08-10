import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { resolveEmbeddingArtifact } from "./embedding-artifact.js";
import {
  ArtifactWarmError,
  experimentalBuildEmbeddingArtifact,
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
      await tools.warmEmbeddingsFromArtifact(bytes, "error");
      const toolHits = await tools.searchWithMethodAsync("read a file", 5, "direct", "semantic");
      expect(toolHits[0]?.toolId).toBe("read_file");

      const skills = new SkillRegistry(embedding, "bm25");
      skills.registerItems([slides, apiDesign]);
      await skills.warmEmbeddingsFromArtifact(bytes, "error");
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
    await expect(tools.warmEmbeddingsFromArtifact(bytes, "error")).resolves.toBeUndefined();
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
