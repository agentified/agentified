import { describe, expect, it } from "vitest";
import { IntentGraph, SkillRegistry, ToolRegistry } from "../native/index.cjs";

describe("native runtime event bridge", () => {
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

    await registry.searchWithMethodAsync("read", 1, "direct", "bm25");
    await subscription.flush();

    expect(batches.some((batch) => batch.length > 0)).toBe(true);
    const event = batches.flat().find((candidate) => candidate.type === "search");
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
  });
});
