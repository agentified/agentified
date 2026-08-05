import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { describe, expect, it, vi } from "vitest";
import {
  EmbedderError,
  IntentGraph,
  SkillCatalog,
  SkillRegistry,
  ToolCatalog,
  ToolRegistry,
  type TraceSinkConfig,
} from "./index.js";

/**
 * A catalog where lexical retrieval is confidently wrong: "why is the build
 * broken" hits `docker_build` on the token *build*, while the tool people
 * actually reach for is `gh_run_list`.
 */
async function buildCatalog(trace?: TraceSinkConfig): Promise<ToolCatalog> {
  const catalog = new ToolCatalog(trace ? { trace } : {});
  await catalog.register([
    {
      id: "docker_build",
      name: "docker_build",
      description: "Build a Docker image from a Dockerfile",
      inputSchema: {},
      outputSchema: {},
      execute: async () => "built",
    },
    {
      id: "gh_run_list",
      name: "gh_run_list",
      description: "List CI workflow runs and whether the build passed",
      inputSchema: {},
      outputSchema: {},
      execute: async () => "listed",
    },
    {
      id: "read_file",
      name: "read_file",
      description: "Read a file from disk",
      inputSchema: {},
      outputSchema: {},
      execute: async () => "read",
    },
  ]);
  return catalog;
}

const ids = (hits: readonly { toolId: string }[]) => hits.map((h) => h.toolId);

/** One confirmed observation: search, then invoke what you actually wanted. */
async function useIt(catalog: ToolCatalog, query: string, chosen: string): Promise<void> {
  catalog.search(query, 5);
  await catalog.invoke(chosen, {});
}

describe("adaptive usage ranking", () => {
  it("leaves ranking untouched until it is enabled", async () => {
    const catalog = await buildCatalog();
    const hits = catalog.search("why is the build broken", 5);
    expect(hits[0]?.toolId).toBe("docker_build");
  });

  it("learns from use and then ranks better", async () => {
    const catalog = await buildCatalog();
    const graph = new IntentGraph();
    catalog.experimentalEnableAdaptiveRanking(graph);

    expect(graph.clusterCount).toBe(0);

    await useIt(catalog, "why is the build broken", "gh_run_list");
    await useIt(catalog, "is the build broken again", "gh_run_list");
    await useIt(catalog, "the build broken on main", "gh_run_list");

    expect(graph.clusterCount).toBe(1);

    const order = ids(catalog.search("why is the build broken", 5));
    expect(order.indexOf("gh_run_list")).toBeLessThan(order.indexOf("docker_build"));
  });

  it("does not disturb a query it has no evidence about", async () => {
    const baseline = ids((await buildCatalog()).search("read a file from disk", 5));

    const catalog = await buildCatalog();
    catalog.experimentalEnableAdaptiveRanking(new IntentGraph());
    await useIt(catalog, "why is the build broken", "gh_run_list");
    await useIt(catalog, "is the build broken again", "gh_run_list");
    await useIt(catalog, "the build broken on main", "gh_run_list");

    expect(ids(catalog.search("read a file from disk", 5))).toEqual(baseline);
  });

  it("carries what it learned across processes via the wire form", async () => {
    // The point of `toJson`/`fromJson`: the graph is in memory, so this is how a
    // restart keeps what previous runs discovered.
    const first = await buildCatalog();
    const graph = new IntentGraph();
    first.experimentalEnableAdaptiveRanking(graph);
    await useIt(first, "why is the build broken", "gh_run_list");
    await useIt(first, "is the build broken again", "gh_run_list");
    await useIt(first, "the build broken on main", "gh_run_list");

    const restored = IntentGraph.fromJson(graph.toJson());
    expect(restored.clusterCount).toBe(1);

    const second = await buildCatalog();
    second.experimentalEnableAdaptiveRanking(restored);
    const order = ids(second.search("why is the build broken", 5));
    expect(order.indexOf("gh_run_list")).toBeLessThan(order.indexOf("docker_build"));
  });

  it("tracks writes with a monotonic rev that survives the wire form", async () => {
    // `rev` is the primitive for the caller's storage layer: save only when it
    // changed, and detect a writer that moved past your base.
    const catalog = await buildCatalog();
    const graph = new IntentGraph();
    catalog.experimentalEnableAdaptiveRanking(graph);
    expect(graph.rev).toBe(0);

    await useIt(catalog, "why is the build broken", "gh_run_list");
    const afterOne = graph.rev;
    expect(afterOne).toBeGreaterThan(0);

    await useIt(catalog, "is the build broken again", "gh_run_list");
    expect(graph.rev).toBeGreaterThan(afterOne);

    // The counter is carried across a save/restore, so it stays monotonic.
    expect(IntentGraph.fromJson(graph.toJson()).rev).toBe(graph.rev);
  });

  it("rejects a graph from a schema version it does not read", () => {
    const future = JSON.stringify({
      v: 2,

      built_from_ts: 1,
      intents: [],
    });
    expect(() => IntentGraph.fromJson(future)).toThrow(/version/i);
  });

  it("stops learning and ranking when disabled, keeping what it knows", async () => {
    const catalog = await buildCatalog();
    const graph = new IntentGraph();
    catalog.experimentalEnableAdaptiveRanking(graph);
    await useIt(catalog, "why is the build broken", "gh_run_list");
    await useIt(catalog, "is the build broken again", "gh_run_list");
    await useIt(catalog, "the build broken on main", "gh_run_list");

    catalog.experimentalDisableAdaptiveRanking();
    const order = ids(catalog.search("why is the build broken", 5));
    expect(order[0]).toBe("docker_build");

    await useIt(catalog, "the build broken elsewhere", "gh_run_list");
    expect(graph.clusterCount).toBe(1);
    expect(graph.toJson()).toContain("gh_run_list");
  });

  it("shares one graph between a tool catalog and a skill catalog", async () => {
    // One cluster, two edge maps — giving each catalog its own graph would
    // duplicate the cluster and split the evidence.
    const graph = new IntentGraph();
    const tools = await buildCatalog();
    const skills = new SkillCatalog();
    await skills.register([
      {
        id: "ci-triage",
        name: "ci-triage",
        description: "Diagnose why the build failed in CI",
        tags: [],
        tools: [],
        metadata: {},
        body: "# steps",
      },
    ]);
    tools.experimentalEnableAdaptiveRanking(graph);
    skills.experimentalEnableAdaptiveRanking(graph);

    await useIt(tools, "why is the build broken", "gh_run_list");
    skills.search("why is the build broken", 5);
    skills.invoke("ci-triage");

    expect(graph.clusterCount).toBe(1);
    const wire = JSON.parse(graph.toJson());
    expect(Object.keys(wire.intents[0].tools)).toContain("gh_run_list");
    expect(Object.keys(wire.intents[0].skills)).toContain("ci-triage");
  });

  it("counts support once when one capability search is answered by a tool and a skill", async () => {
    // search_capabilities fans one query to both catalogs (both searches before
    // any invoke). A tool AND a skill used for that question must bump the shared
    // cluster's support by 1, not 1 per catalog (#3) — the two per-catalog
    // learners dedup through the credit slot on the shared graph.
    const graph = new IntentGraph();
    const tools = await buildCatalog();
    const skills = new SkillCatalog();
    await skills.register([
      {
        id: "ci-triage",
        name: "ci-triage",
        description: "Diagnose why the build failed in CI",
        tags: [],
        tools: [],
        metadata: {},
        body: "# steps",
      },
    ]);
    tools.experimentalEnableAdaptiveRanking(graph);
    skills.experimentalEnableAdaptiveRanking(graph);

    // Fan-out ordering: both searches (same query) before any invoke.
    tools.search("why is the build broken", 5);
    skills.search("why is the build broken", 5);
    await tools.invoke("gh_run_list", {});
    skills.invoke("ci-triage");

    const wire = JSON.parse(graph.toJson());
    expect(graph.clusterCount).toBe(1);
    expect(wire.intents[0].support).toBe(1);
    expect(Object.keys(wire.intents[0].tools)).toContain("gh_run_list");
    expect(Object.keys(wire.intents[0].skills)).toContain("ci-triage");
  });

  it("exposes rank and fused so callers avoid the scale-shifting score", async () => {
    const catalog = await buildCatalog();
    const graph = new IntentGraph();
    catalog.experimentalEnableAdaptiveRanking(graph);

    // No evidence yet: raw scores, unfused.
    const cold = catalog.search("why is the build broken", 5);
    expect(cold.map((h) => h.rank)).toEqual(cold.map((_, i) => i));
    expect(cold.every((h) => h.fused === false)).toBe(true);

    await useIt(catalog, "why is the build broken", "gh_run_list");
    await useIt(catalog, "is the build broken again", "gh_run_list");
    await useIt(catalog, "the build broken on main", "gh_run_list");

    // Matched now: fused RRF scores, rank still contiguous from 0.
    const warm = catalog.search("why is the build broken", 5);
    expect(warm[0]?.rank).toBe(0);
    expect(warm.every((h) => h.fused === true)).toBe(true);

    // An unrelated query on the same catalog stays unfused — the between-calls
    // switch `fused` exists to expose.
    expect(catalog.search("read a file from disk", 5).every((h) => !h.fused)).toBe(true);
  });

  // Regression for the base-sink bug: enable/disable rebuilt the learner's inner
  // sink from the memory-sink handle only, dropping a configured jsonl sink to
  // noop — so the trace file silently stopped growing. Registration writes a
  // churn event *before* the toggle, so assert the file grows *after* it.
  async function jsonlCatalog(): Promise<{ catalog: ToolCatalog; size: () => number }> {
    const path = join(mkdtempSync(join(tmpdir(), "ratel-trace-")), "trace.jsonl");
    const catalog = new ToolCatalog({ trace: { kind: "jsonl", sessionId: "s", path } });
    await catalog.register([
      {
        id: "t",
        name: "t",
        description: "a tool",
        inputSchema: {},
        outputSchema: {},
        execute: async () => "ok",
      },
    ]);
    return { catalog, size: () => readFileSync(path, "utf8").length };
  }

  it("keeps writing to a configured jsonl sink after enabling adaptive ranking", async () => {
    const { catalog, size } = await jsonlCatalog();
    catalog.experimentalEnableAdaptiveRanking(new IntentGraph());

    const before = size();
    catalog.search("anything", 5);
    expect(size()).toBeGreaterThan(before);
  });

  it("keeps writing to a configured jsonl sink after disabling adaptive ranking", async () => {
    const { catalog, size } = await jsonlCatalog();
    catalog.experimentalDisableAdaptiveRanking(); // clobbered the sink even when never enabled

    const before = size();
    catalog.search("anything", 5);
    expect(size()).toBeGreaterThan(before);
  });
});

// ---- opt-in auto-rebuild on model change ------------------------------------

/** The pyo3/napi native can't be monkeypatched, so swap the whole native for a
 * fake to exercise the trigger without a real embedding model. `status` is
 * scripted; a dense search returns nothing. */
function fakeNative(state: { status: string }) {
  return {
    enableAdaptiveRanking: () => {},
    adaptiveRankingStatus: () => ({
      status: state.status,
      built: "old-model",
      active: "new-model",
      dimMismatch: false,
    }),
    searchWithMethodAsync: async () => [],
  };
}

describe("rebuildOnModelChange", () => {
  it("recovers a paused graph on the next dense search when enabled", async () => {
    const reg = new ToolRegistry();
    const state = { status: "paused: model mismatch" };
    (reg as unknown as { native: unknown }).native = fakeNative(state);
    reg.experimentalEnableAdaptiveRanking(new IntentGraph(), {
      rebuildOnModelChange: true,
      warnOnModelMismatch: false,
    });
    const rebuild = vi.spyOn(reg, "experimentalRebuildIntentGraph").mockResolvedValue();

    await reg.searchWithMethodAsync("anything", 5, "direct", "semantic");
    expect(rebuild).toHaveBeenCalledOnce();
  });

  it("is off by default", async () => {
    const reg = new ToolRegistry();
    (reg as unknown as { native: unknown }).native = fakeNative({
      status: "paused: model mismatch",
    });
    reg.experimentalEnableAdaptiveRanking(new IntentGraph(), { warnOnModelMismatch: false });
    const rebuild = vi.spyOn(reg, "experimentalRebuildIntentGraph").mockResolvedValue();

    await reg.searchWithMethodAsync("anything", 5, "direct", "semantic");
    expect(rebuild).not.toHaveBeenCalled();
  });

  it("does nothing when the arm is active", async () => {
    const reg = new ToolRegistry();
    (reg as unknown as { native: unknown }).native = fakeNative({ status: "active" });
    reg.experimentalEnableAdaptiveRanking(new IntentGraph(), { rebuildOnModelChange: true });
    const rebuild = vi.spyOn(reg, "experimentalRebuildIntentGraph").mockResolvedValue();

    await reg.searchWithMethodAsync("anything", 5, "direct", "semantic");
    expect(rebuild).not.toHaveBeenCalled();
  });

  it("stops rebuilding once the graph is recovered", async () => {
    const reg = new ToolRegistry();
    const state = { status: "paused: model mismatch" };
    (reg as unknown as { native: unknown }).native = fakeNative(state);
    reg.experimentalEnableAdaptiveRanking(new IntentGraph(), {
      rebuildOnModelChange: true,
      warnOnModelMismatch: false,
    });
    // A successful rebuild flips status to active, so the second search skips it.
    const rebuild = vi.spyOn(reg, "experimentalRebuildIntentGraph").mockImplementation(async () => {
      state.status = "active";
    });

    await reg.searchWithMethodAsync("anything", 5, "direct", "semantic");
    await reg.searchWithMethodAsync("anything", 5, "direct", "semantic");
    expect(rebuild).toHaveBeenCalledOnce();
  });

  it("also drives the skill registry twin", async () => {
    const reg = new SkillRegistry();
    (reg as unknown as { native: unknown }).native = fakeNative({
      status: "paused: model mismatch",
    });
    reg.experimentalEnableAdaptiveRanking(new IntentGraph(), {
      rebuildOnModelChange: true,
      warnOnModelMismatch: false,
    });
    const rebuild = vi.spyOn(reg, "experimentalRebuildIntentGraph").mockResolvedValue();

    await reg.searchWithMethodAsync("anything", 5, "direct", "semantic");
    expect(rebuild).toHaveBeenCalledOnce();
  });
});

describe("experimentalRebuildIntentGraph error mapping (#6)", () => {
  it("surfaces an embedding failure as a typed EmbedderError", async () => {
    const catalog = new ToolCatalog({
      method: "semantic",
      embedding: { local: "/definitely/missing/ratel-embedding-model" },
    });
    const graph = IntentGraph.fromJson(
      JSON.stringify({
        v: 1,
        built_from_ts: 1,
        intents: [
          {
            id: "i0",
            label: "l",
            terms: [],
            members: ["read a file"],
            centroid: [1, 0, 0],
            support: 9,
            tools: {},
            skills: {},
          },
        ],
      }),
    );
    catalog.experimentalEnableAdaptiveRanking(graph, { warnOnModelMismatch: false });

    // Rebuild re-embeds the member under the missing model → load failure. The
    // sibling paths already map this; experimentalRebuildIntentGraph must too.
    const error = await catalog.experimentalRebuildIntentGraph().catch((e: unknown) => e);
    expect(error).toBeInstanceOf(EmbedderError);
  });
});

// ---- embedding-model change detection ---------------------------------------

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

const hub =
  process.env.HF_HUB_CACHE ??
  join(process.env.HF_HOME ?? join(homedir(), ".cache", "huggingface"), "hub");
const bgeSmall = join(
  hub,
  "models--BAAI--bge-small-en-v1.5",
  "snapshots",
  "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
);
const hasModel =
  existsSync(join(bgeSmall, "config.json")) && existsSync(join(bgeSmall, "tokenizer.json"));

/** A graph carrying a 384-dim centroid stamped with a DIFFERENT model than the
 * catalog will use — the persisted-graph-after-model-swap scenario. */
function staleModelGraph(): IntentGraph {
  const centroid = Array.from({ length: 384 }, (_, i) => (i === 0 ? 1 : 0));
  return IntentGraph.fromJson(
    JSON.stringify({
      v: 1,
      built_from_ts: 1,
      model: "some-other-model",
      intents: [
        {
          id: "intent_0",
          label: "l",
          terms: [],
          members: ["why is the build broken"],
          centroid,
          support: 9,
          tools: { gh_run_list: 1.0 },
          skills: {},
        },
      ],
    }),
  );
}

describe.skipIf(!hasModel)("adaptive ranking model-change detection", () => {
  async function semanticCatalog(): Promise<ToolCatalog> {
    const catalog = new ToolCatalog({ method: "semantic" });
    await catalog.register([
      {
        id: "gh_run_list",
        name: "gh_run_list",
        description: "list CI runs",
        inputSchema: {},
        outputSchema: {},
        execute: async () => "ok",
      },
      {
        id: "docker_build",
        name: "docker_build",
        description: "build an image",
        inputSchema: {},
        outputSchema: {},
        execute: async () => "ok",
      },
    ]);
    return catalog;
  }

  it("pauses and warns on a model mismatch, and rebuild restores it", async () => {
    const catalog = await semanticCatalog();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      catalog.experimentalEnableAdaptiveRanking(staleModelGraph());
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("paused: model mismatch");
      expect(warn).toHaveBeenCalledOnce();
      expect(warn.mock.calls[0]?.[0]).toContain("experimentalRebuildIntentGraph()");

      await catalog.experimentalRebuildIntentGraph();
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("active");
    } finally {
      warn.mockRestore();
    }
  });

  it("stays silent when warnOnModelMismatch is false", async () => {
    const catalog = await semanticCatalog();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      catalog.experimentalEnableAdaptiveRanking(staleModelGraph(), { warnOnModelMismatch: false });
      expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("paused: model mismatch");
      expect(warn).not.toHaveBeenCalled();
    } finally {
      warn.mockRestore();
    }
  });

  it("auto-recovers on the next dense search with rebuildOnModelChange", async () => {
    // End to end with a real model: a stale-model graph pauses, but the first
    // dense search rebuilds it under the active model — no manual rebuild call.
    const catalog = await semanticCatalog();
    catalog.experimentalEnableAdaptiveRanking(staleModelGraph(), {
      warnOnModelMismatch: false,
      rebuildOnModelChange: true,
    });
    expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("paused: model mismatch");

    await catalog.searchAsync("why is the build broken", 5, "direct", "semantic");
    expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("active");
  });
});

describe("baseline seeding", () => {
  /** Capture three build/CI turns to a JSONL log with Ratel serving nothing. */
  async function captureBaseline(): Promise<{ catalog: ToolCatalog; logPath: string }> {
    const logPath = `${mkdtempSync(`${tmpdir()}/ratel-seed-`)}/telemetry.jsonl`;
    const catalog = new ToolCatalog({
      trace: { kind: "jsonl", sessionId: "session-1", path: logPath },
    });
    await catalog.register([
      {
        id: "docker_build",
        name: "docker_build",
        description: "Build a Docker image from a Dockerfile",
        inputSchema: {},
        outputSchema: {},
        execute: async () => "built",
      },
      {
        id: "gh_run_list",
        name: "gh_run_list",
        description: "List CI workflow runs and whether the build passed",
        inputSchema: {},
        outputSchema: {},
        execute: async () => "listed",
      },
    ]);

    for (const turn of [
      "why is the build broken",
      "is the build broken again",
      "the build broken on main",
    ]) {
      catalog.experimentalBaselineTurn(turn).invoked("gh_run_list").record();
    }
    return { catalog, logPath };
  }

  it("writes nothing until the turn is recorded", async () => {
    const catalog = await buildCatalog({ kind: "memory", sessionId: "s" });
    catalog.drainTraceEvents(); // discard the registration churn
    const turn = catalog.experimentalBaselineTurn("why is the build broken");
    turn.invoked("gh_run_list");
    expect(catalog.drainTraceEvents()).toHaveLength(0);

    turn.record();
    // The quality gate is "call record, or don't" — a dropped turn leaves no trace.
    const types = catalog.drainTraceEvents().map((e) => (e as { type: string }).type);
    expect(types).toEqual(["search", "invoke_start"]);
  });

  it("attributes several invocations to one turn", async () => {
    const catalog = await buildCatalog({ kind: "memory", sessionId: "s" });
    catalog.drainTraceEvents(); // discard the registration churn
    catalog
      .experimentalBaselineTurn("why is the build broken")
      .invoked("gh_run_list")
      .invokedSkill("triage")
      .record();

    const events = catalog.drainTraceEvents() as { type: string }[];
    expect(events.map((e) => e.type)).toEqual(["search", "invoke_start", "skill_invoke"]);
  });

  it("refuses to record the same turn twice", async () => {
    const catalog = await buildCatalog({ kind: "memory", sessionId: "s" });
    const turn = catalog.experimentalBaselineTurn("why is the build broken");
    turn.record();
    expect(() => turn.record()).toThrow(/already recorded/);
    expect(() => turn.invoked("gh_run_list")).toThrow(/already recorded/);
  });

  it("builds a graph from a log the agent produced without Ratel ranking", async () => {
    const { catalog, logPath } = await captureBaseline();
    // Nothing was attached during capture, so ranking never changed.
    expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("inactive");

    const graph = await catalog.experimentalBuildIntentGraph(readFileSync(logPath, "utf8"), {
      origins: "baseline",
      provenance: "seeded",
    });

    expect(graph.clusterCount).toBe(1);
    const parsed = JSON.parse(graph.toJson());
    expect(parsed.intents[0].support).toBe(3);
    expect(parsed.intents[0].seeded_support).toBe(3);
    expect(parsed.intents[0].tools.gh_run_list).toBe(3);
  });

  it("returns a detached graph so enabling stays an explicit act", async () => {
    const { catalog, logPath } = await captureBaseline();
    const graph = await catalog.experimentalBuildIntentGraph(readFileSync(logPath, "utf8"), {
      origins: "baseline",
    });
    expect(graph.clusterCount).toBe(1);
    expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("inactive");

    catalog.experimentalEnableAdaptiveRanking(graph);
    expect(catalog.experimentalAdaptiveRankingStatus.status).toBe("active");
  });

  it("ignores searches the policy does not accept", async () => {
    const { catalog, logPath } = await captureBaseline();
    // Default policy accepts every origin; a baseline-only one still sees these.
    const seeded = await catalog.experimentalBuildIntentGraph(readFileSync(logPath, "utf8"), {
      origins: "agent",
    });
    expect(seeded.clusterCount).toBe(0);
  });

  it("rejects an unknown policy value instead of silently defaulting", async () => {
    const { logPath } = await captureBaseline();
    const catalog = new ToolCatalog();
    await expect(
      catalog.experimentalBuildIntentGraph(readFileSync(logPath, "utf8"), {
        provenance: "seedd",
      }),
    ).rejects.toThrow(/unknown provenance/);
  });

  it("names the line number of a malformed log entry", async () => {
    // A truncated tail is the realistic case (a crash mid-write). Failing loudly
    // beats silently dropping it: the log is the only record of a capture, and a
    // thinner graph with no explanation is worse than an error you can act on.
    const good =
      '{"v":1,"ts":1,"session_id":"s","type":"search","query":"q",' +
      '"origin":"baseline","top_k":0,"hits":[],"stages":[],"took_ms":0}';
    const catalog = new ToolCatalog();
    await expect(
      catalog.experimentalBuildIntentGraph(`${good}\n{"v":1,"ts":2,"sess`),
    ).rejects.toThrow(/line 2/);
  });

  it("skips blank lines rather than failing on them", async () => {
    const good =
      '{"v":1,"ts":1,"session_id":"s","type":"search","query":"q",' +
      '"origin":"baseline","top_k":0,"hits":[],"stages":[],"took_ms":0}';
    const catalog = new ToolCatalog();
    const graph = await catalog.experimentalBuildIntentGraph(`\n${good}\n\n`);
    // The search was never acted on, so it teaches nothing — but parsing worked.
    expect(graph.clusterCount).toBe(0);
  });
});

describe("policy on the live path", () => {
  it("can be restricted by origin", async () => {
    const catalog = await buildCatalog();
    const graph = new IntentGraph();
    catalog.experimentalEnableAdaptiveRanking(graph, { origins: "baseline" });

    catalog.search("why is the build broken", 5); // origin "direct"
    catalog.recordEvent({ type: "invoke_start", tool_id: "gh_run_list", args_size_bytes: 0 });
    expect(graph.clusterCount).toBe(0);

    catalog.experimentalBaselineTurn("why is the build broken").invoked("gh_run_list").record();
    expect(graph.clusterCount).toBe(1);
  });

  it("rejects an unknown value rather than defaulting", async () => {
    const catalog = await buildCatalog();
    expect(() =>
      // @ts-expect-error the runtime guard still has to hold for untyped callers
      catalog.experimentalEnableAdaptiveRanking(new IntentGraph(), { origins: "nope" }),
    ).toThrow(/unknown origins/);
  });

  it("keeps the policy when the trace sink changes", async () => {
    // Changing the sink rebuilds the learner that decorates it. Rebuilding at
    // the default would silently drop a configured policy.
    const registry = new ToolRegistry();
    registry.register({
      id: "gh_run_list",
      name: "gh_run_list",
      description: "List CI workflow runs and whether the build passed",
      inputSchema: {},
      outputSchema: {},
    });
    const graph = new IntentGraph();
    registry.experimentalEnableAdaptiveRanking(graph, { origins: "baseline" });

    registry.setTraceSink({ kind: "memory", sessionId: "after" });

    registry.search("why is the build broken", 5); // "direct" — filtered out
    registry.recordEvent({ type: "invoke_start", tool_id: "gh_run_list", args_size_bytes: 0 });
    expect(graph.clusterCount).toBe(0);
  });
});
