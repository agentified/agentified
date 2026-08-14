import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { IntentGraph, SkillRegistry, ToolRegistry } from "../native/index.cjs";
import { startDelayedEmbeddingServer } from "./test-support/delayed-embedding-server.js";

describe("native runtime event bridge", () => {
  it("allows unsubscribe while a flush is draining JavaScript callbacks", () => {
    const moduleUrl = new URL("../native/index.cjs", import.meta.url).href;
    const script = `
      import { ToolRegistry } from ${JSON.stringify(moduleUrl)};
      const registry = new ToolRegistry();
      const subscription = registry.subscribeTraceEvents(() => {
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 5);
      }, {
        sessionId: "session-flush-unsubscribe",
        queueCapacity: 512,
        batchSize: 2,
      });
      for (let index = 0; index < 500; index += 1) {
        registry.search(\`query-\${index}\`, 1);
      }
      const flushing = subscription.flush();
      await new Promise((resolve) => setTimeout(resolve, 10));
      subscription.unsubscribe();
      await flushing;
    `;

    const result = spawnSync(process.execPath, ["--input-type=module", "-e", script], {
      encoding: "utf8",
      timeout: 4_000,
    });

    expect(result.error).toBeUndefined();
    expect(result.signal).toBeNull();
    expect(result.status).toBe(0);
  });

  it("does not keep the Node process alive while subscribed", () => {
    const moduleUrl = new URL("../native/index.cjs", import.meta.url).href;
    const script = `
      import { ToolRegistry } from ${JSON.stringify(moduleUrl)};
      const registry = new ToolRegistry();
      globalThis.subscription = registry.subscribeTraceEvents(() => {}, {
        sessionId: "session-exit",
      });
    `;

    const result = spawnSync(process.execPath, ["--input-type=module", "-e", script], {
      encoding: "utf8",
      timeout: 2_000,
    });

    expect(result.error).toBeUndefined();
    expect(result.signal).toBeNull();
    expect(result.status).toBe(0);
  });

  it("pushes worker-thread search events to JavaScript in batches", async () => {
    const registry = new ToolRegistry();
    registry.register({
      id: "read_file",
      name: "read_file",
      description: "Read a file",
      inputSchema: {},
      outputSchema: {},
    });
    const batches: Array<Array<Record<string, unknown>>> = [];
    const subscription = registry.subscribeTraceEvents(
      (batch: Array<Record<string, unknown>>) => batches.push(batch),
      {
        sessionId: "session-ts",
        sourceId: "source-ts",
        queueCapacity: 16,
        batchSize: 8,
      },
    );

    const workerSearch = registry.searchWithMethodAsync("read", 1, "direct", "bm25");
    for (let index = 0; index < 8; index += 1) {
      registry.search(`burst-${index}`, 1);
    }
    await workerSearch;
    await subscription.flush();

    expect(batches.some((batch) => batch.length > 1)).toBe(true);
    const event = batches
      .flat()
      .find((candidate) => candidate.type === "search" && candidate.query === "read");
    expect(event).toMatchObject({
      v: 2,
      session_id: "session-ts",
      source_id: "source-ts",
      type: "search",
      query: "read",
    });
  });

  it("drops oldest events and reports loss while JavaScript is stalled", async () => {
    const registry = new ToolRegistry();
    const events: Array<Record<string, unknown>> = [];
    let firstBatch = true;
    const subscription = registry.subscribeTraceEvents(
      (batch: Array<Record<string, unknown>>) => {
        events.push(...batch);
        if (firstBatch) {
          firstBatch = false;
          Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 100);
        }
      },
      {
        sessionId: "session-drop",
        queueCapacity: 2,
        batchSize: 1,
      },
    );

    await Promise.all(
      Array.from({ length: 64 }, (_, index) =>
        registry.searchWithMethodAsync(`query-${index}`, 1, "direct", "bm25"),
      ),
    );
    await subscription.flush();

    expect(subscription.droppedCount).toBeGreaterThan(0);
    expect(events).toContainEqual(
      expect.objectContaining({
        type: "events_dropped",
        reason: "queue_overflow",
      }),
    );
  });

  it("pushes skill-registry events through the same native seam", async () => {
    const registry = new SkillRegistry();
    registry.register({
      id: "api-design",
      name: "api-design",
      description: "Design an API",
      tags: ["backend"],
      body: "Use resource nouns.",
    });
    const events: Array<Record<string, unknown>> = [];
    const subscription = registry.subscribeTraceEvents(
      (batch: Array<Record<string, unknown>>) => events.push(...batch),
      { sessionId: "session-skill" },
    );

    await registry.searchWithMethodAsync("api", 1, "direct", "bm25");
    await subscription.flush();

    expect(events).toContainEqual(
      expect.objectContaining({ type: "skill_search", session_id: "session-skill" }),
    );
  });

  it("keeps callbacks and usage learning active across a base-sink re-wrap", async () => {
    const registry = new ToolRegistry();
    const graph = new IntentGraph();
    registry.enableAdaptiveRanking(graph);
    const events: Array<Record<string, unknown>> = [];
    const subscription = registry.subscribeTraceEvents(
      (batch: Array<Record<string, unknown>>) => events.push(...batch),
      { sessionId: "session-learner" },
    );
    registry.setTraceSink({ kind: "memory", sessionId: "session-learner" });

    registry.search("read file", 1);
    registry.recordEvent({ type: "invoke_start", tool_id: "read_file", args_size_bytes: 0 });
    await subscription.flush();

    expect(graph.clusterCount).toBe(1);
    expect(events.map((event) => event.type)).toEqual(
      expect.arrayContaining(["search", "invoke_start"]),
    );
    expect(registry.drainTraceEvents().map((event) => event.type)).toEqual(
      expect.arrayContaining(["search", "invoke_start"]),
    );
  });

  it("keeps the base sink synchronous and independently stamped after subscribing", () => {
    const registry = new ToolRegistry();
    registry.setTraceSink({ kind: "memory", sessionId: "base-session" });
    const subscription = registry.subscribeTraceEvents(() => {}, {
      sessionId: "stream-session",
      queueCapacity: 1,
      batchSize: 1,
    });

    for (let index = 0; index < 5_000; index += 1) {
      registry.search(`persist-${index}`, 1);
    }

    const persisted = registry
      .drainTraceEvents()
      .filter((event) => event.type === "search") as Array<Record<string, unknown>>;
    expect(persisted).toHaveLength(5_000);
    expect(persisted.every((event) => event.session_id === "base-session")).toBe(true);
    subscription.unsubscribe();
  });

  it.each([
    {
      kind: "tool",
      create: (url: string) => {
        const registry = new ToolRegistry({ url, model: "test-model" });
        registry.register({
          id: "read_file",
          name: "read_file",
          description: "Read a file",
          inputSchema: {},
          outputSchema: {},
        });
        return registry;
      },
    },
    {
      kind: "skill",
      create: (url: string) => {
        const registry = new SkillRegistry({ url, model: "test-model" });
        registry.register({
          id: "api-design",
          name: "API design",
          description: "Design an API",
          tags: [],
          body: "Use nouns",
        });
        return registry;
      },
    },
  ])("keeps the $kind base sink when a busy registry rejects replacement", async ({ create }) => {
    const server = await startDelayedEmbeddingServer();
    try {
      const registry = create(server.url);
      registry.setTraceSink({ kind: "memory", sessionId: "base-session" });
      const subscription = registry.subscribeTraceEvents(() => {}, {
        sessionId: "stream-session",
      });
      await server.armBatchHold(1);
      const building = registry.buildEmbeddings();
      await server.waitForHeld();

      expect(() =>
        registry.setTraceSink({ kind: "memory", sessionId: "rejected-session" }),
      ).toThrow(/registry busy; await/);
      server.releaseHeld();
      await building;

      registry.search("after-busy", 1);
      await subscription.flush();
      expect(registry.drainTraceEvents()).toContainEqual(
        expect.objectContaining({ type: expect.stringMatching(/search$/), query: "after-busy" }),
      );
    } finally {
      await server.close();
    }
  });
});
