import { describe, expect, it } from "vitest";
import { type RuntimeEvent, ratel } from "./index.js";

describe("public runtime events", () => {
  it("merges tool and skill streams under one session stamp", async () => {
    const runtime = ratel({
      events: {
        sessionId: "session-public",
        sourceId: "source-public",
        queueCapacity: 16,
        batchSize: 8,
      },
    });
    const received: RuntimeEvent[] = [];
    const subscription = runtime.events.subscribe((batch) => received.push(...batch));

    await runtime.tools.register({
      id: "read_file",
      name: "read_file",
      description: "Read a file",
      inputSchema: { type: "object", properties: { path: { type: "string" } } },
      outputSchema: { type: "object" },
      execute: () => ({ ok: true }),
    });
    await runtime.skills.register({
      id: "api-design",
      name: "API design",
      description: "Design an API",
      tags: ["backend"],
      metadata: { audience: ["developers"] },
      body: "Private dispatch instructions",
    });
    runtime.tools.search("read", 1);
    runtime.skills.search("api", 1);

    await subscription.flush();

    expect(received.map((event) => event.type)).toEqual(
      expect.arrayContaining(["index_churn", "skill_churn", "search", "skill_search"]),
    );
    expect(received.every((event) => event.session_id === "session-public")).toBe(true);
    expect(received.every((event) => event.source_id === "source-public")).toBe(true);
    expect(received.every((event) => /^[0-9A-HJKMNP-TV-Z]{26}$/.test(event.event_id))).toBe(true);

    subscription.unsubscribe();
  });

  it("returns the complete serializable catalog without executable content", async () => {
    const runtime = ratel({ events: { sourceId: "service-a" } });
    const execute = () => ({ secret: "must never escape" });
    await runtime.tools.register({
      id: "z_tool",
      name: "Z tool",
      description: "Last by id",
      inputSchema: { type: "object", properties: { value: { type: "string" } } },
      outputSchema: { type: "string" },
      execute,
    });
    await runtime.tools.register({
      id: "a_tool",
      name: "A tool",
      description: "First by id",
      inputSchema: { type: "object" },
      outputSchema: { type: "object" },
      execute,
    });
    await runtime.skills.register({
      id: "skill-a",
      name: "Skill A",
      description: "Public skill metadata",
      tags: ["public"],
      tools: ["a_tool"],
      metadata: { stacks: ["typescript"] },
      body: "May contain private instructions",
    });

    const snapshot = runtime.catalog.snapshot();

    expect(snapshot).toEqual({
      sourceId: "service-a",
      tools: [
        {
          id: "a_tool",
          name: "A tool",
          description: "First by id",
          inputSchema: { type: "object" },
          outputSchema: { type: "object" },
        },
        {
          id: "z_tool",
          name: "Z tool",
          description: "Last by id",
          inputSchema: { type: "object", properties: { value: { type: "string" } } },
          outputSchema: { type: "string" },
        },
      ],
      skills: [
        {
          id: "skill-a",
          name: "Skill A",
          description: "Public skill metadata",
          tags: ["public"],
          tools: ["a_tool"],
          metadata: { stacks: ["typescript"] },
        },
      ],
    });
    expect(JSON.stringify(snapshot)).not.toContain("execute");
    expect(JSON.stringify(snapshot)).not.toContain("private instructions");
  });
});
