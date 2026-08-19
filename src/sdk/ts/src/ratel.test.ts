import { describe, expect, it, vi } from "vitest";
import {
  type ExecutableTool,
  GET_SKILL_CONTENT_ID,
  INVOKE_TOOL_ID,
  type RatelAdapter,
  ratel,
  SEARCH_CAPABILITIES_ID,
  type SearchCapabilitiesResult,
  SkillCatalog,
  ToolCatalog,
} from "./index.js";
import { unadaptedError } from "./ratel.js";
import type {
  ExperimentalDefinitionOverridesRuntimeCatalog,
  InternalExperimentalDefinitionOverridesRuntimeCatalog,
} from "./runtime-events.js";
import { startDelayedEmbeddingServer } from "./test-support/delayed-embedding-server.js";
import {
  type FakeMessage,
  type FakeTool,
  makeExecutableTool,
  referenceAdapter,
} from "./testkit/reference-adapter.js";

const exec = (description: string): FakeTool => makeExecutableTool({ description });

const native = (id: string, description: string): ExecutableTool => ({
  id,
  name: id,
  description,
  inputSchema: { type: "object" },
  outputSchema: { type: "object" },
  execute: () => ({ ok: true }),
});

// `applyDefinitionOverrides` is deliberately off the public catalog interface (it
// bypasses the conditional-request protocol); tests drive it through the
// implementation-side type the runtime actually returns.
const internalCatalog = (
  catalog: ExperimentalDefinitionOverridesRuntimeCatalog,
): InternalExperimentalDefinitionOverridesRuntimeCatalog =>
  catalog as InternalExperimentalDefinitionOverridesRuntimeCatalog;

const CAPABILITY_IDS = [SEARCH_CAPABILITIES_ID, INVOKE_TOOL_ID, GET_SKILL_CONTENT_ID];

describe("ratel() standalone core", () => {
  it("attaches and refreshes override retrieval descriptions through an injected source", async () => {
    const r = ratel();
    await r.tools.register({
      ...native("deploy", "Deploy an application."),
      experimentalSearchableDescription: "localcanaryterm",
    });
    await r.skills.register({
      id: "deploy-skill",
      name: "deploy-skill",
      description: "Deploy an application.",
      experimentalSearchableDescription: "localskillterm",
    });
    const snapshotBeforeAttach = r.catalog.snapshot();
    const requests: Array<string | undefined> = [];
    const responses = [
      {
        status: 200 as const,
        etag: '"overlay-1"',
        body: {
          overrides: [
            {
              kind: "tool" as const,
              entryId: "deploy",
              searchableDescription: "overriderollbackterm",
            },
            {
              kind: "skill" as const,
              entryId: "deploy-skill",
              searchableDescription: "overrideskillterm",
            },
          ],
        },
      },
      { status: 304 as const },
      {
        status: 200 as const,
        etag: '"overlay-2"',
        body: { overrides: [] },
      },
    ];
    const attachment = await r.catalog.experimentalAttachDefinitionOverrides({
      source: {
        fetch: async (ifNoneMatch) => {
          requests.push(ifNoneMatch);
          const response = responses.shift();
          if (response === undefined) throw new Error("unexpected overlay fetch");
          return response;
        },
      },
    });

    expect(r.tools.search("overriderollbackterm", 5).map((hit) => hit.toolId)).toEqual(["deploy"]);
    expect(r.tools.search("localcanaryterm", 5)).toEqual([]);
    expect(r.skills.search("overrideskillterm", 5).map((hit) => hit.skillId)).toEqual([
      "deploy-skill",
    ]);
    expect(r.catalog.snapshot()).toEqual(snapshotBeforeAttach);

    await attachment.refresh();
    expect(r.tools.search("overriderollbackterm", 5).map((hit) => hit.toolId)).toEqual(["deploy"]);

    await attachment.refresh();
    expect(r.tools.search("localcanaryterm", 5).map((hit) => hit.toolId)).toEqual(["deploy"]);
    expect(r.tools.search("overriderollbackterm", 5)).toEqual([]);
    expect(requests).toEqual([undefined, '"overlay-1"', '"overlay-1"']);
  });

  it("keeps local retrieval descriptions when definition overrides are not enabled", async () => {
    const r = ratel();
    await r.tools.register({
      ...native("deploy", "Deploy an application."),
      experimentalSearchableDescription: "localcanaryterm",
    });

    expect(r.tools.search("localcanaryterm", 5).map((hit) => hit.toolId)).toEqual(["deploy"]);
    expect(r.tools.search("overriderollbackterm", 5)).toEqual([]);
  });

  it("applies override retrieval descriptions without changing local definitions", async () => {
    const r = ratel();
    await r.tools.register({
      ...native("deploy", "Deploy an application."),
      experimentalSearchableDescription: "localcanaryterm",
    });

    await internalCatalog(r.catalog).applyDefinitionOverrides([
      { kind: "tool", entryId: "deploy", searchableDescription: "overriderollbackterm" },
    ]);

    expect(r.tools.search("overriderollbackterm", 5).map((hit) => hit.toolId)).toEqual(["deploy"]);
    expect(r.tools.search("localcanaryterm", 5)).toEqual([]);
    expect(r.tools.get("deploy")?.description).toBe("Deploy an application.");
  });

  it("restores the local retrieval description when the source removes an override", async () => {
    const r = ratel();
    await r.tools.register({
      ...native("deploy", "Deploy an application."),
      experimentalSearchableDescription: "localcanaryterm",
    });
    await internalCatalog(r.catalog).applyDefinitionOverrides([
      { kind: "tool", entryId: "deploy", searchableDescription: "overriderollbackterm" },
    ]);

    await internalCatalog(r.catalog).applyDefinitionOverrides([]);

    expect(r.tools.search("localcanaryterm", 5).map((hit) => hit.toolId)).toEqual(["deploy"]);
    expect(r.tools.search("overriderollbackterm", 5)).toEqual([]);
  });

  it("restores accepted registrations until mutated values are explicitly re-registered", async () => {
    const r = ratel();
    const tool = {
      ...native("deploy", "Original tool"),
      experimentalSearchableDescription: "originaltoolterm",
      inputSchema: { type: "object", properties: { target: { type: "string" } } },
    };
    const skill = {
      id: "deploy-skill",
      name: "deploy-skill",
      description: "Original skill",
      experimentalSearchableDescription: "originalskillterm",
      tags: ["original"],
      metadata: { audiences: ["operators"] },
      body: "Original skill body",
    };
    const fact = {
      id: "deploy-fact",
      name: "deploy-fact",
      description: "Original fact",
      experimentalSearchableDescription: "originalfactterm",
      tags: ["original"],
      metadata: { regions: ["eu"] },
      body: "Original fact body",
    };
    await r.tools.register(tool);
    await r.skills.register(skill);
    await r.facts.register(fact);
    const overrides = [
      { kind: "tool" as const, entryId: tool.id, searchableDescription: "overridetoolterm" },
      { kind: "skill" as const, entryId: skill.id, searchableDescription: "overrideskillterm" },
      { kind: "fact" as const, entryId: fact.id, searchableDescription: "overridefactterm" },
    ];
    await internalCatalog(r.catalog).applyDefinitionOverrides(overrides);

    tool.description = "Mutated tool";
    tool.experimentalSearchableDescription = "mutatedtoolterm";
    tool.inputSchema.properties.target.type = "number";
    skill.description = "Mutated skill";
    skill.experimentalSearchableDescription = "mutatedskillterm";
    skill.tags.push("mutated");
    skill.metadata.audiences.push("developers");
    skill.body = "Mutated skill body";
    fact.description = "Mutated fact";
    fact.experimentalSearchableDescription = "mutatedfactterm";
    fact.tags.push("mutated");
    fact.metadata.regions.push("us");
    fact.body = "Mutated fact body";

    await internalCatalog(r.catalog).applyDefinitionOverrides([]);

    expect(r.tools.search("originaltoolterm", 5).map((hit) => hit.toolId)).toEqual([tool.id]);
    expect(r.skills.search("originalskillterm", 5).map((hit) => hit.skillId)).toEqual([skill.id]);
    expect(r.facts.search("originalfactterm", 5).map((hit) => hit.factId)).toEqual([fact.id]);
    expect(r.catalog.snapshot().tools[0]?.inputSchema).toEqual({
      type: "object",
      properties: { target: { type: "string" } },
    });
    expect(r.skills.get(skill.id)).toMatchObject({
      tags: ["original"],
      metadata: { audiences: ["operators"] },
      body: "Original skill body",
    });
    expect(r.facts.get(fact.id)).toMatchObject({
      tags: ["original"],
      metadata: { regions: ["eu"] },
      body: "Original fact body",
    });

    await internalCatalog(r.catalog).applyDefinitionOverrides(overrides);
    await r.tools.register(tool);
    await r.skills.register(skill);
    await r.facts.register(fact);
    await internalCatalog(r.catalog).applyDefinitionOverrides([]);

    expect(r.tools.search("mutatedtoolterm", 5).map((hit) => hit.toolId)).toEqual([tool.id]);
    expect(r.skills.search("mutatedskillterm", 5).map((hit) => hit.skillId)).toEqual([skill.id]);
    expect(r.facts.search("mutatedfactterm", 5).map((hit) => hit.factId)).toEqual([fact.id]);
    expect(r.catalog.snapshot().tools[0]?.inputSchema).toEqual({
      type: "object",
      properties: { target: { type: "number" } },
    });
    expect(r.skills.invoke(skill.id)).toBe("Mutated skill body");
    expect(r.facts.get(fact.id)?.body).toBe("Mutated fact body");
  });

  it("applies override retrieval descriptions to skills registered after attach", async () => {
    const r = ratel();
    await internalCatalog(r.catalog).applyDefinitionOverrides([
      { kind: "skill", entryId: "deploy", searchableDescription: "overriderollbackterm" },
    ]);

    await r.skills.register({
      id: "deploy",
      name: "deploy",
      description: "Deploy an application.",
      experimentalSearchableDescription: "localcanaryterm",
      body: "# Deploy",
    });

    expect(r.skills.search("overriderollbackterm", 5).map((hit) => hit.skillId)).toEqual([
      "deploy",
    ]);
    expect(r.skills.search("localcanaryterm", 5)).toEqual([]);
  });

  it("applies override retrieval descriptions to facts without eagerly creating the catalog", async () => {
    const r = ratel();
    await internalCatalog(r.catalog).applyDefinitionOverrides([
      { kind: "fact", entryId: "address", searchableDescription: "overridelocationterm" },
    ]);

    await r.facts.register({
      id: "address",
      name: "address",
      description: "Where the shop is.",
      experimentalSearchableDescription: "localaddressterm",
    });

    expect(r.facts.search("overridelocationterm", 5).map((hit) => hit.factId)).toEqual(["address"]);
    expect(r.facts.search("localaddressterm", 5)).toEqual([]);
  });

  it("warns once per continuous period that an override shadows a local retrieval description", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const r = ratel();
      await r.tools.register({
        ...native("deploy", "Deploy an application."),
        experimentalSearchableDescription: "localcanaryterm",
      });
      const override = {
        kind: "tool" as const,
        entryId: "deploy",
        searchableDescription: "overriderollbackterm",
      };

      await internalCatalog(r.catalog).applyDefinitionOverrides([override]);
      await internalCatalog(r.catalog).applyDefinitionOverrides([override]);
      await internalCatalog(r.catalog).applyDefinitionOverrides([]);
      await internalCatalog(r.catalog).applyDefinitionOverrides([override]);

      expect(warn).toHaveBeenCalledTimes(2);
      expect(warn).toHaveBeenCalledWith(
        'ratel: definition override for tool "deploy" shadows the local experimentalSearchableDescription while definition overrides are enabled',
      );
    } finally {
      warn.mockRestore();
    }
  });

  it("does not warn when an override has no matching explicit local value", async () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const r = ratel();
      await r.tools.register(native("deploy", "Deploy an application."));

      await internalCatalog(r.catalog).applyDefinitionOverrides([
        { kind: "tool", entryId: "deploy", searchableDescription: "overriderollbackterm" },
        { kind: "tool", entryId: "unknown", searchableDescription: "overrideunknownterm" },
      ]);

      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it("r.tools is a handle over the one shared catalog", async () => {
    const r = ratel();
    await r.tools.register(
      native("read_file", "Read a file from local disk."),
      native("write_file", "Write a file to local disk."),
    );
    expect(r.tools.has("read_file")).toBe(true);
    expect(r.tools.get("write_file")?.description).toBe("Write a file to local disk.");
    expect(r.tools.search("read a file", 5)[0]?.toolId).toBe("read_file");
    expect(r.tools.catalog).toBeInstanceOf(ToolCatalog);
    expect(r.tools.catalog.has("read_file")).toBe(true); // same instance, no drift
    expect(r.skills).toBeInstanceOf(SkillCatalog);
  });

  it("invokes registered tools through the handle", async () => {
    const r = ratel();
    await r.tools.register(native("ping", "Ping a host."));
    expect(await r.tools.invoke("ping", {})).toEqual({ ok: true });
  });

  it("native registration is authoritative: a duplicate id replaces in place", async () => {
    const r = ratel();
    await r.tools.register(native("dup", "first description"));
    await r.tools.register(native("dup", "second description"));
    expect(r.tools.get("dup")?.description).toBe("second description"); // unlike the adapted path
  });

  it("native replace-in-place resolves a duplicate id within one batch to the last", async () => {
    const r = ratel();
    // Both entries share an id in a single register() call; the native path is
    // replace-in-place (not first-wins like the adapted path), so the last wins.
    await r.tools.register(native("dup", "first description"), native("dup", "second description"));
    expect(r.tools.get("dup")?.description).toBe("second description");
  });

  it("modelTools() returns the three capability tools in native shape", async () => {
    const r = ratel();
    await r.tools.register(native("read_file", "Read a file from local disk."));
    const out = r.modelTools();
    expect(Object.keys(out).sort()).toEqual([...CAPABILITY_IDS].sort());
    expect(out[SEARCH_CAPABILITIES_ID]?.id).toBe(SEARCH_CAPABILITIES_ID); // native, not adapted
    const result = (await out[SEARCH_CAPABILITIES_ID]?.execute({
      query: "read a file",
    })) as SearchCapabilitiesResult;
    expect(result.tools.groups[0]?.hits[0]?.toolId).toBe("read_file");
  });

  it("modelTools() builds fresh capability-tool objects each call", async () => {
    const r = ratel();
    await r.tools.register(native("read_file", "Read a file from local disk."));
    const first = r.modelTools();
    const second = r.modelTools();
    // Fresh objects per call (parity with the adapted view), so a host taking
    // the set once and reusing it keeps the prompt cache stable.
    expect(second[SEARCH_CAPABILITIES_ID]).not.toBe(first[SEARCH_CAPABILITIES_ID]);
    expect(second[INVOKE_TOOL_ID]).not.toBe(first[INVOKE_TOOL_ID]);
    expect(second[GET_SKILL_CONTENT_ID]).not.toBe(first[GET_SKILL_CONTENT_ID]);
  });

  it("modelTools() always advertises get_skill_content, even with zero skills", async () => {
    const r = ratel();
    const out = r.modelTools();
    expect(Object.keys(out)).toContain(GET_SKILL_CONTENT_ID);
    // Empty skill catalog degrades to a structured error, never a missing tool.
    const res = (await out[GET_SKILL_CONTENT_ID]?.execute({ skillId: "nope" })) as {
      isError?: boolean;
    };
    expect(res.isError).toBe(true);
  });

  it("tools registered after modelTools() are discoverable through the exposed set", async () => {
    const r = ratel();
    const out = r.modelTools(); // take the set first…
    await r.tools.register(native("late_tool", "Deploy the app to production.")); // …register later
    const result = (await out[SEARCH_CAPABILITIES_ID]?.execute({
      query: "deploy production",
    })) as SearchCapabilitiesResult;
    expect(result.tools.groups[0]?.hits[0]?.toolId).toBe("late_tool");
  });

  it("loads a skill registered after modelTools() through the exposed get_skill_content", async () => {
    const r = ratel();
    const out = r.modelTools(); // taken once, before any skill exists
    await r.skills.register({
      id: "deploy",
      name: "deploy",
      description: "Deploy playbook: preview vs production, rollbacks.",
      tags: [],
      body: "# Deploy",
    });
    const res = (await out[GET_SKILL_CONTENT_ID]?.execute({ skillId: "deploy" })) as {
      body?: string;
    };
    expect(res.body).toBe("# Deploy");
  });

  it("keeps the exposed search_capabilities description byte-stable across skill registration", async () => {
    const r = ratel();
    const before = r.modelTools()[SEARCH_CAPABILITIES_ID]?.description;
    // The skills clause is always advertised — get_skill_content is always in
    // the set, so the description must never deny the bucket…
    expect(before).toContain("get_skill_content");
    await r.skills.register({
      id: "deploy",
      name: "deploy",
      description: "Deploy playbook: preview vs production, rollbacks.",
      tags: [],
      body: "# Deploy",
    });
    // …and re-exposing after the first skill must not bust the prompt cache.
    expect(r.modelTools()[SEARCH_CAPABILITIES_ID]?.description).toBe(before);
  });

  it("propagates a skills.replaceAll() reload through the capability tools taken earlier", async () => {
    const r = ratel();
    await r.skills.register([
      {
        id: "deploy",
        name: "deploy",
        description: "Deploy playbook: rollbacks.",
        body: "# Deploy",
      },
      { id: "auth", name: "auth", description: "Set up login and sessions.", body: "# Auth" },
    ]);
    // Taken once, as a host would, and reused across turns.
    const out = r.modelTools();
    const before = out[SEARCH_CAPABILITIES_ID]?.description;

    await r.skills.replaceAll([
      { id: "auth", name: "auth", description: "Set up login and sessions.", body: "# Auth" },
      {
        id: "migrations",
        name: "migrations",
        description: "Reversible DB migrations.",
        body: "# M",
      },
    ]);

    // The already-exposed tools read the live catalog: no rebuild, no cache bust.
    expect(r.modelTools()[SEARCH_CAPABILITIES_ID]?.description).toBe(before);

    const dropped = (await out[GET_SKILL_CONTENT_ID]?.execute({ skillId: "deploy" })) as {
      body?: string;
      error?: string;
      isError?: boolean;
    };
    expect(dropped.isError).toBe(true);
    expect(dropped.body).toBeUndefined();
    expect(dropped.error).toMatch(/unknown skillId: deploy/);

    const added = (await out[GET_SKILL_CONTENT_ID]?.execute({ skillId: "migrations" })) as {
      body?: string;
    };
    expect(added.body).toBe("# M");

    const recalled = await r.recall("roll back a deploy");
    expect(recalled?.skills.map((skill) => skill.skillId) ?? []).not.toContain("deploy");
  });

  it("throws when a tool shadows a reserved capability-tool id", () => {
    for (const id of CAPABILITY_IDS) {
      expect(() => ratel().tools.register(native(id, "impostor"))).toThrow(/reserved/);
    }
  });

  it("throws the actionable install-the-adapter error on framework-shaped tools", () => {
    const r = ratel();
    // A framework tool record: no id, zod-like schema, keyed externally.
    const zodLike = {
      description: "AI SDK style tool",
      inputSchema: { _def: {}, safeParse: () => ({}) },
      execute: () => ({}),
    };
    expect(() => r.tools.register(zodLike as unknown as ExecutableTool)).toThrow(/adaptTo/);
    // Dynamic (function) descriptions are a framework idiom too.
    const dynamicDescription = {
      ...native("x", "y"),
      description: () => "computed",
    };
    expect(() => r.tools.register(dynamicDescription as unknown as ExecutableTool)).toThrow(
      /adaptTo/,
    );
  });

  it("throws a plain TypeError, not the adapter hint, on non-object register input", () => {
    const r = ratel();
    expect(() => r.tools.register(null as unknown as ExecutableTool)).toThrow(
      /ExecutableTools; got object/,
    );
    expect(() => r.tools.register("read_file" as unknown as ExecutableTool)).toThrow(
      /ExecutableTools; got string/,
    );
  });

  it("throws its own error, not the adapter hint, on a native-shaped tool missing an id", () => {
    const r = ratel();
    // Plain native shape (string description, JSON-schema input) — just no id.
    // That's a malformed native tool, not a framework tool: the adapter hint
    // would send the caller off installing a package that can't help.
    const noId = {
      name: "orphan",
      description: "A plain native tool with a JSON schema but no id.",
      inputSchema: { type: "object" },
      outputSchema: { type: "object" },
      execute: () => ({ ok: true }),
    };
    expect(() => r.tools.register(noId as unknown as ExecutableTool)).toThrow(
      /expects each ExecutableTool to have a string `id`/,
    );
    expect(() => r.tools.register(noId as unknown as ExecutableTool)).not.toThrow(/adaptTo/);
  });

  it("rejects a native tool with no execute handler synchronously, committing nothing", () => {
    const r = ratel();
    // A JS caller (or a cast) can hand register() a native-shaped tool that is
    // missing its execute handler. The handle validates synchronously, so it
    // fails fast at the call site rather than as a rejection of a promise the
    // caller may not be awaiting.
    const noExecute = {
      id: "broken",
      name: "broken",
      description: "A native-shaped tool missing its execute handler.",
      inputSchema: { type: "object" },
      outputSchema: { type: "object" },
    };
    expect(() => r.tools.register(noExecute as unknown as ExecutableTool)).toThrow(/execute/);
    expect(r.tools.has("broken")).toBe(false); // nothing committed
    expect(() => r.tools.register(noExecute as unknown as ExecutableTool)).not.toThrow(/adaptTo/);
  });

  it("clamps tools.search topK the same as the funnel: caps at 50, falls back on invalid", async () => {
    const r = ratel();
    const many = Array.from({ length: 60 }, (_, i) =>
      native(`grep_${i}`, `Search files variant ${i}: grep ripgrep.`),
    );
    await r.tools.register(...many);
    // 60 tools match, so exact counts pin the contract: a stray-large topK caps
    // at 50 (not passed straight to the catalog), a negative one falls back to
    // the default 5 (not wrapping to an unbounded u32 set in the native layer).
    expect(r.tools.search("search files grep", 999).length).toBe(50);
    expect(r.tools.search("search files grep", -1).length).toBe(5);
  });

  it("recall() is a pure query: canonical result or null, no call ids", async () => {
    const r = ratel();
    expect(await r.recall("anything")).toBeNull(); // empty catalog

    await r.tools.register(native("deploy_app", "Deploy the app to production."));
    expect(await r.recall("zzzqqq totally unrelated")).toBeNull(); // no hits

    const result = await r.recall("deploy to production");
    expect(result?.tools.groups[0]?.hits[0]?.toolId).toBe("deploy_app");
    expect(result).not.toHaveProperty("callId"); // pure shape, id-free
  });

  it("recall() includes the skills bucket when a skill is registered", async () => {
    const r = ratel();
    await r.skills.register({
      id: "vercel-deploy",
      name: "vercel-deploy",
      description: "Deploy to Vercel: env vars, preview vs production, rollbacks.",
      tags: ["vercel"],
      body: "# Vercel",
    });
    expect((await r.recall("deploy to vercel"))?.skills.map((s) => s.skillId)).toContain(
      "vercel-deploy",
    );
  });

  it("caps recallTopK at 50 and falls back to the default on invalid values", async () => {
    const many = Array.from({ length: 60 }, (_, i) =>
      native(`grep_${i}`, `Search files variant ${i}: grep ripgrep.`),
    );

    const high = ratel({ recallTopK: 999 });
    await high.tools.register(...many);
    const hi = (await high.recall("search files grep"))?.tools.groups.reduce(
      (n, g) => n + g.hits.length,
      0,
    );
    expect(hi).toBe(50); // 999 honored past the default 5, then capped at exactly 50

    const low = ratel({ recallTopK: -1 });
    await low.tools.register(...many);
    const lo = (await low.recall("search files grep"))?.tools.groups.reduce(
      (n, g) => n + g.hits.length,
      0,
    );
    expect(lo).toBe(5); // invalid → exactly the default 5
  });

  it("derives server groups by the '__' prefix, treating a leading '__' as no prefix", async () => {
    const r = ratel();
    await r.tools.register(
      native("github__create_issue", "Open a GitHub issue tracker ticket."),
      native("__weird_internal", "An internal GitHub issue tracker tool with leading underscores."),
    );
    const names = (await r.recall("github issue tracker"))?.tools.groups.map((g) => g.server.name);
    expect(names).toContain("github"); // sep > 0 → "github"
    expect(names).toContain("__weird_internal"); // sep === 0 → whole id
  });

  it("records a direct-origin gateway_search on recall", async () => {
    const r = ratel({ trace: { kind: "memory", sessionId: "t" } });
    await r.tools.register(native("deploy_grep", "Deploy grep search tool."));
    r.tools.catalog.drainTraceEvents();
    await r.recall("deploy grep");
    const ev = (r.tools.catalog.drainTraceEvents() as Array<Record<string, unknown>>).find(
      (e) => e.type === "gateway_search",
    );
    expect(ev?.origin).toBe("direct");
  });
});

describe("ratel() semantic/hybrid config", () => {
  // Semantic/hybrid is a first-class config through the factory. These prove
  // method + embedding are forwarded to the tool, skill, and fact catalogs and that register is
  // async (awaits/surfaces the embedding pass) — offline except where a fake
  // embedding endpoint is needed for the end-to-end embed-then-search.
  it("forwards method: the handle's sync search is BM25-only and points to searchAsync", () => {
    expect(() => ratel().tools.search("q", 5)).not.toThrow(); // bm25 default: sync search works

    // A hybrid catalog can't be searched synchronously — the handle says so in
    // its own words, not by leaking the native "use searchWithMethodAsync()".
    expect(() => ratel({ method: "hybrid" }).tools.search("q", 5)).toThrow(/searchAsync/);
    // A per-call semantic override on a bm25 core is caught the same way.
    expect(() => ratel().tools.search("q", 5, "semantic")).toThrow(/searchAsync/);
  });

  it("forwards method + embedding so searchAsync ranks against real embeddings", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const r = ratel({ method: "semantic", embedding: { url: server.url, model: "test-model" } });
      await r.tools.register(native("read_file", "Read a file from local disk."));
      const hits = await r.tools.searchAsync("read a file", 5);
      expect(hits[0]?.toolId).toBe("read_file");
    } finally {
      await server.close();
    }
  });

  it("register awaits the embedding pass and rejects on failure, but commits metadata", async () => {
    const r = ratel({
      method: "semantic",
      embedding: { local: "/definitely/missing/ratel-embedding-model" },
    });
    // register() is async: the embedding failure is the rejection of the call
    // itself (no discarded promise, no unhandled rejection).
    await expect(
      r.tools.register(native("read_file", "Read a file from local disk.")),
    ).rejects.toThrow(/failed to load embedding model/);
    expect(r.tools.has("read_file")).toBe(true); // metadata is committed before the embed pass
  });

  it("a BM25 searchAsync override doesn't depend on the (failing) embedding pass", async () => {
    const r = ratel({
      method: "semantic",
      embedding: { local: "/definitely/missing/ratel-embedding-model" },
    });
    // The embedding pass fails, but the lexical corpus is still indexed, so a
    // per-call bm25 override ranks it.
    await r.tools.register(native("read_file", "Read a file from local disk.")).catch(() => {});
    const hits = await r.tools.searchAsync("read a file", 5, "bm25");
    expect(hits[0]?.toolId).toBe("read_file");
  });

  it("recall ranks a registered semantic tool", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const r = ratel({ method: "semantic", embedding: { url: server.url, model: "test-model" } });
      await r.tools.register(native("deploy_app", "Deploy the app to production."));
      const result = await r.recall("deploy to production");
      expect(result?.tools.groups[0]?.hits[0]?.toolId).toBe("deploy_app");
    } finally {
      await server.close();
    }
  });

  it("forwards method + embedding to the skill catalog too", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const r = ratel({ method: "semantic", embedding: { url: server.url, model: "test-model" } });
      await r.skills.register({
        id: "vercel-deploy",
        name: "vercel-deploy",
        description: "Deploy to Vercel: env vars, preview vs production, rollbacks.",
        tags: ["vercel"],
        body: "# Vercel",
      });
      const hits = await r.skills.searchAsync("deploy to vercel", 3);
      expect(hits[0]?.skillId).toBe("vercel-deploy");
    } finally {
      await server.close();
    }
  });
});

describe("ratel().adaptTo(adapter)", () => {
  it("merges the adapter's extend helpers onto the base surface", () => {
    const a = ratel().adaptTo(referenceAdapter());
    expect(a.label).toBe("adapted:true"); // TExt inferred and merged
    expect(typeof a.tools.register).toBe("function");
    expect(typeof a.modelTools).toBe("function");
    expect(typeof a.recall).toBe("function");
    expect(a.skills).toBeInstanceOf(SkillCatalog);
  });

  it("registers tools into the shared catalog; modelTools() returns only the capability set", async () => {
    const a = ratel().adaptTo(referenceAdapter());
    await a.tools.register({ read_file: exec("Read a file from local disk.") });
    expect(a.tools.has("read_file")).toBe(true);
    expect(a.tools.catalog.has("read_file")).toBe(true); // same shared catalog
    expect(Object.keys(a.modelTools()).sort()).toEqual([...CAPABILITY_IDS].sort());
  });

  it("forwards an adapter registration experimentalSearchableDescription into ranking", async () => {
    const a = ratel().adaptTo(referenceAdapter());
    await a.tools.register({
      tool: {
        ...exec("composedonlyterm"),
        experimentalSearchableDescription: "overrideonlyterm",
      },
    });

    expect(a.tools.search("overrideonlyterm", 5).map((hit) => hit.toolId)).toEqual(["tool"]);
    expect(a.tools.search("composedonlyterm", 5)).toEqual([]);
  });

  it("routes the capability tools through the adapter's expose codec", async () => {
    const a = ratel().adaptTo(referenceAdapter());
    await a.tools.register({ read_file: exec("Read a file from local disk.") });
    const search = a.modelTools()[SEARCH_CAPABILITIES_ID];
    // A FakeTool has no id/outputSchema — a raw ExecutableTool would. Their
    // absence proves modelTools() ran the value through the codec, not shipped it raw.
    expect(search).not.toHaveProperty("id");
    expect(search).not.toHaveProperty("outputSchema");
    const result = (await search?.execute?.({ query: "read a file" })) as SearchCapabilitiesResult;
    expect(result.tools.groups[0]?.hits[0]?.toolId).toBe("read_file");
  });

  it("always advertises get_skill_content, even before any skill is registered", () => {
    const a = ratel().adaptTo(referenceAdapter());
    expect(Object.keys(a.modelTools())).toContain(GET_SKILL_CONTENT_ID);
  });

  it("tools registered after modelTools() are discoverable through the exposed set", async () => {
    const a = ratel().adaptTo(referenceAdapter());
    const out = a.modelTools();
    await a.tools.register({ late_tool: exec("Deploy the app to production.") });
    const result = (await out[SEARCH_CAPABILITIES_ID]?.execute?.({
      query: "deploy production",
    })) as SearchCapabilitiesResult;
    expect(result.tools.groups[0]?.hits[0]?.toolId).toBe("late_tool");
  });

  it("keeps non-executable tools as per-view passthroughs, unregistered", async () => {
    const core = ratel();
    const a = core.adaptTo(referenceAdapter());
    const b = core.adaptTo(referenceAdapter());
    const provider: FakeTool = { description: "provider-run search", inputSchema: {} };
    await a.tools.register({ provider_search: provider });
    expect(a.modelTools().provider_search).toBe(provider); // eagerly exposed, untouched
    expect(a.tools.catalog.has("provider_search")).toBe(false); // never in the catalog
    expect(a.tools.has("provider_search")).toBe(true); // …but the view knows it
    expect(b.modelTools()).not.toHaveProperty("provider_search"); // framework-shaped → per view
  });

  it("a passthrough claims its id across adapter views without leaking its value", async () => {
    const core = ratel();
    const a = core.adaptTo(referenceAdapter());
    const b = core.adaptTo(referenceAdapter());
    const firstForExecutableCollision: FakeTool = { description: "provider a", inputSchema: {} };
    const firstForPassthroughCollision: FakeTool = { description: "provider a", inputSchema: {} };
    const latePassthrough: FakeTool = { description: "provider b", inputSchema: {} };

    await a.tools.register({
      executable_shadow: firstForExecutableCollision,
      passthrough_shadow: firstForPassthroughCollision,
    });
    await b.tools.register({
      executable_shadow: exec("late executable"),
      passthrough_shadow: latePassthrough,
    });

    expect(core.tools.has("executable_shadow")).toBe(false);
    expect(a.modelTools().executable_shadow).toBe(firstForExecutableCollision);
    expect(a.modelTools().passthrough_shadow).toBe(firstForPassthroughCollision);
    expect(b.modelTools()).not.toHaveProperty("executable_shadow");
    expect(b.modelTools()).not.toHaveProperty("passthrough_shadow");
  });

  it("exposes get/search/invoke on the adapted handle, at parity with the native ToolCollection", async () => {
    const a = ratel().adaptTo(referenceAdapter());
    await a.tools.register({ read_file: exec("Read a file from local disk.") });
    expect(a.tools.get("read_file")?.description).toBe("Read a file from local disk.");
    expect(a.tools.search("read a file", 5)[0]?.toolId).toBe("read_file");
    expect(await a.tools.invoke("read_file", {})).toEqual({ ok: true });
  });

  it("adapted searchAsync ranks a semantic catalog; sync search still points to searchAsync", async () => {
    const server = await startDelayedEmbeddingServer();
    try {
      const a = ratel({
        method: "semantic",
        embedding: { url: server.url, model: "test-model" },
      }).adaptTo(referenceAdapter());
      await a.tools.register({ read_file: exec("Read a file from local disk.") });
      expect(() => a.tools.search("read a file", 5)).toThrow(/searchAsync/);
      const hits = await a.tools.searchAsync("read a file", 5);
      expect(hits[0]?.toolId).toBe("read_file");
    } finally {
      await server.close();
    }
  });

  it("adapted get/search/invoke are catalog-only: a passthrough is known to has() but not to them", async () => {
    const a = ratel().adaptTo(referenceAdapter());
    const provider: FakeTool = { description: "provider-run search", inputSchema: {} };
    await a.tools.register({ provider_search: provider });
    expect(a.tools.has("provider_search")).toBe(true); // the view knows it
    expect(a.tools.get("provider_search")).toBeUndefined(); // …but it's not in the catalog
  });

  it("throws when an app tool shadows a reserved capability-tool id", () => {
    for (const id of CAPABILITY_IDS) {
      const a = ratel().adaptTo(referenceAdapter());
      expect(() => a.tools.register({ [id]: exec("x") })).toThrow(/reserved/);
    }
  });

  it("register is atomic: a reserved id mid-batch commits neither passthrough nor executable", () => {
    const a = ratel().adaptTo(referenceAdapter());
    const provider: FakeTool = { description: "provider-run search", inputSchema: {} };
    // A mixed batch whose reserved id throws after a passthrough and an
    // executable: staging means the earlier entries never land — no
    // half-committed, model-exposed passthrough while the executable is dropped.
    expect(() =>
      a.tools.register({
        provider_search: provider,
        read_file: exec("Read a file from local disk."),
        [SEARCH_CAPABILITIES_ID]: exec("impostor"),
      }),
    ).toThrow(/reserved/);
    expect(a.tools.has("provider_search")).toBe(false); // passthrough not committed
    expect(a.tools.catalog.has("read_file")).toBe(false); // executable not committed
    expect(a.modelTools()).not.toHaveProperty("provider_search"); // and not model-exposed
  });

  it("skips ingest entirely for an id that is already registered", async () => {
    const core = ratel();
    await core.tools.register(native("shared_tool", "Shared grep tool."));
    let ingestCalls = 0;
    const counting: RatelAdapter<FakeTool, FakeMessage> = {
      ...referenceAdapter(),
      ingest(id, tool) {
        ingestCalls++;
        return referenceAdapter().ingest(id, tool);
      },
    };
    const a = core.adaptTo(counting);
    await a.tools.register({ shared_tool: exec("would shadow") }); // native id → first-wins pre-check
    expect(ingestCalls).toBe(0);
    await a.tools.register({ fresh: exec("Fresh tool.") });
    await a.tools.register({ fresh: exec("Duplicate call.") });
    expect(ingestCalls).toBe(1); // ingest's user code must not re-run per repeated register
  });

  it("preserves an ingest-provided outputSchema and defaults an omitted one", async () => {
    const withOutput: RatelAdapter<FakeTool, FakeMessage> = {
      ...referenceAdapter(),
      ingest(_id, tool) {
        return {
          description: tool.description,
          inputSchema: tool.inputSchema,
          outputSchema: { type: "object", properties: { ok: { type: "boolean" } } },
          execute: tool.execute ?? (() => ({})),
        };
      },
    };
    const a = ratel().adaptTo(withOutput);
    await a.tools.register({ typed: exec("Tool with a declared output shape.") });
    expect(a.tools.catalog.get("typed")?.outputSchema).toEqual({
      type: "object",
      properties: { ok: { type: "boolean" } },
    });

    const b = ratel().adaptTo(referenceAdapter()); // referenceAdapter's ingest omits outputSchema
    await b.tools.register({ untyped: exec("Tool without an output shape.") });
    expect(b.tools.catalog.get("untyped")?.outputSchema).toEqual({ type: "object" });
  });

  it("first registration of an id wins, across executables and passthroughs", async () => {
    const a = ratel().adaptTo(referenceAdapter());
    await a.tools.register({ dup: exec("first description") });
    await a.tools.register({ dup: exec("second description") });
    expect(a.tools.catalog.get("dup")?.description).toBe("first description");

    // A passthrough claims its id too: a later executable must not shadow it.
    const provider: FakeTool = { description: "provider-run", inputSchema: {} };
    await a.tools.register({ claimed: provider });
    await a.tools.register({ claimed: exec("late executable") });
    expect(a.tools.catalog.has("claimed")).toBe(false);
    expect(a.modelTools().claimed).toBe(provider);
  });

  it("recall returns the adapter's message pair with a private-counter call id", async () => {
    const a = ratel().adaptTo(referenceAdapter());
    await a.tools.register({ deploy_app: exec("Deploy the app to production servers.") });

    const first = await a.recall("deploy to production");
    expect(first).toHaveLength(2);
    expect(first[0].callId).toBe("recall_0");
    expect(first[0].body).toEqual({ query: "deploy to production" }); // ref carries the query
    expect(first[1].callId).toBe("recall_0");
    const recall = first[1].body as SearchCapabilitiesResult;
    expect(recall.tools.groups.length).toBeGreaterThan(0);

    const second = await a.recall("deploy to production");
    expect(second[0].callId).toBe("recall_1"); // monotonic private counter
  });

  it("recall returns [] and spends no call id when nothing matches", async () => {
    const a = ratel().adaptTo(referenceAdapter());
    expect(await a.recall("anything at all")).toEqual([]); // empty catalog, no skills

    await a.tools.register({ deploy_app: exec("Deploy the app to production.") });
    // A query that shares no terms with the only tool → no hits → still [].
    expect(await a.recall("zzzqqq totally unrelated")).toEqual([]);
    // The counter was never spent, so the next real hit is still recall_0.
    expect(((await a.recall("deploy production")) as FakeMessage[])[0].callId).toBe("recall_0");
  });

  it("shares one core's catalog and recall counter across adapter views", async () => {
    const core = ratel();
    const a = core.adaptTo(referenceAdapter());
    const b = core.adaptTo(referenceAdapter());
    await a.tools.register({ shared_tool: exec("Shared search grep tool.") });
    expect(b.tools.has("shared_tool")).toBe(true); // one catalog
    expect(core.tools.has("shared_tool")).toBe(true); // …the core's own

    expect(((await a.recall("shared grep")) as FakeMessage[])[0].callId).toBe("recall_0");
    expect(((await b.recall("shared grep")) as FakeMessage[])[0].callId).toBe("recall_1"); // shared counter
  });

  it("recall carries a skills-only match: the pair is built even with zero tool hits", async () => {
    const a = ratel().adaptTo(referenceAdapter());
    await a.skills.register({
      id: "vercel-deploy",
      name: "vercel-deploy",
      description: "Deploy to Vercel: env vars, preview vs production, rollbacks.",
      tags: ["vercel"],
      body: "# Vercel",
    });
    const msgs = await a.recall("deploy to vercel");
    expect(msgs).toHaveLength(2);
    const recall = msgs[1].body as SearchCapabilitiesResult;
    expect(recall.tools.groups).toEqual([]);
    expect(recall.skills.map((s) => s.skillId)).toContain("vercel-deploy");
  });

  it("tolerates an extend returning a non-object: the view is still the usable base", async () => {
    const bad = {
      ...referenceAdapter(),
      extend: () => 42,
    } as unknown as RatelAdapter<FakeTool, FakeMessage>;
    const a = ratel().adaptTo(bad);
    await a.tools.register({ read_file: exec("Read a file from local disk.") });
    expect(Object.keys(a.modelTools()).sort()).toEqual([...CAPABILITY_IDS].sort());
  });

  it("validates the adapter shape, naming it via adapter.name", () => {
    expect(() => ratel().adaptTo({} as unknown as RatelAdapter)).toThrow(/RatelAdapter/);
    expect(() => ratel().adaptTo({ name: "broken" } as unknown as RatelAdapter)).toThrow(/broken/);
  });
});

describe("unadaptedError", () => {
  it("names the exact adapter package when a known framework is detected", () => {
    const aiErr = unadaptedError((pkg) => pkg === "ai");
    expect(aiErr.message).toContain("@ratel-ai/vercel-ai-sdk");
    expect(aiErr.message).toContain("aiSdk");

    const mastraErr = unadaptedError((pkg) => pkg === "@mastra/core");
    expect(mastraErr.message).toContain("@ratel-ai/mastra");
    expect(mastraErr.message).toContain("mastra");
  });

  it("falls back to a generic adapt-first message when no framework is present", () => {
    const err = unadaptedError(() => false);
    expect(err.message).toContain("adaptTo");
    expect(err.message).not.toContain("Detected");
  });
});
