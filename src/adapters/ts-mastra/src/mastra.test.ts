import type { MastraDBMessage } from "@mastra/core/agent";
import { RequestContext } from "@mastra/core/request-context";
import { createTool, Tool } from "@mastra/core/tools";
import {
  GET_SKILL_CONTENT_ID,
  INVOKE_TOOL_ID,
  ratel,
  SEARCH_CAPABILITIES_ID,
  type SearchCapabilitiesResult,
} from "@ratel-ai/sdk";
import { describe, expect, it } from "vitest";
import { z } from "zod";
// A genuine zod 3 schema (the dev dep pins zod 4, which ships the v3 API under
// its `zod/v3` export). The peer range allows zod 3, and Mastra routes v3 through
// a different converter than v4, so the v3 ingest path needs its own coverage.
import { z as z3 } from "zod/v3";
import { mastra } from "./mastra.js";

// A JSON Schema whose `properties` we read positionally in assertions.
type JsonObjectSchema = { type?: string; properties?: Record<string, { type?: string }> };

// A user MastraDBMessage carrying one text part (Mastra's format-2 content).
function userMsg(...texts: string[]): MastraDBMessage {
  return {
    id: `u_${texts.join("_")}`,
    role: "user",
    createdAt: new Date(),
    content: { format: 2, parts: texts.map((text) => ({ type: "text", text })) },
  };
}

// Invoke an exposed Mastra tool with a valid minimal context plus any live
// fields the test supplies.
function callExposed(
  tool: Tool,
  args: Record<string, unknown>,
  context: Record<string, unknown> = {},
): Promise<unknown> {
  const execute = tool.execute as (input: unknown, context: unknown) => Promise<unknown>;
  return execute(args, {
    observe: {
      async span<T>(_name: string, fn: () => T | Promise<T>): Promise<T> {
        return fn();
      },
      log(): void {},
    },
    ...context,
  });
}

function countHits(result: SearchCapabilitiesResult): number {
  return result.tools.groups.reduce((n, g) => n + g.hits.length, 0);
}

// A view over a fresh core with one BM25-discoverable executable Mastra tool.
function viewWithDeployTool() {
  const view = ratel().adaptTo(mastra());
  view.tools.register({
    deploy_app: createTool({
      id: "deploy_app",
      description: "Deploy the app to production servers.",
      inputSchema: z.object({}),
      execute: async () => ({ deployed: true }),
    }),
  });
  return view;
}

describe("mastra() identity", () => {
  it('names itself "mastra" — matches the core KNOWN_FRAMEWORKS table', () => {
    expect(mastra().name).toBe("mastra");
  });
});

describe("ingest codec", () => {
  it("forwards an app-supplied searchableDescription", () => {
    const native = Object.assign(
      createTool({
        id: "tool",
        description: "composedonlyterm",
        inputSchema: z.object({}),
        execute: async () => ({ ok: true }),
      }),
      { searchableDescription: "overrideonlyterm" },
    );

    const registration = mastra().ingest("tool", native);
    if (registration === "passthrough") throw new Error("expected executable registration");
    expect(registration.searchableDescription).toBe("overrideonlyterm");
  });

  it("passes through a Mastra tool with no execute (client/provider-executed)", () => {
    const providerTool = createTool({ id: "provider_search", description: "provider-run search" });
    expect(mastra().ingest("provider_search", providerTool)).toBe("passthrough");
  });

  it("ingests an executable: description, extracted input schema, omitted output schema", () => {
    const weather = createTool({
      id: "weather",
      description: "Get the weather in a location",
      inputSchema: z.object({ location: z.string() }),
      execute: async ({ location }) => ({ location, tempF: 70 }),
    });
    const reg = mastra().ingest("weather", weather);
    if (reg === "passthrough") throw new Error("expected an executable registration");
    expect(reg.description).toBe("Get the weather in a location");
    const input = reg.inputSchema as JsonObjectSchema;
    expect(input.type).toBe("object");
    expect(input.properties?.location.type).toBe("string");
    // The `$schema` dialect marker is stripped; absent output schema stays absent.
    expect((reg.inputSchema as Record<string, unknown>).$schema).toBeUndefined();
    expect(reg.outputSchema).toBeUndefined();
  });

  it("ingests a tool built with a zod 3 schema (the peer's other converter)", () => {
    const weather = createTool({
      id: "weather",
      description: "Get the weather in a location",
      inputSchema: z3.object({ location: z3.string() }),
      execute: async ({ location }) => ({ location, tempF: 70 }),
    });
    const reg = mastra().ingest("weather", weather);
    if (reg === "passthrough") throw new Error("expected an executable registration");
    const input = reg.inputSchema as JsonObjectSchema;
    expect(input.type).toBe("object");
    expect(input.properties?.location.type).toBe("string");
  });

  it("extracts JSON schema from a tool built with a raw JSON Schema", () => {
    const t = createTool({
      id: "raw",
      description: "raw json schema",
      // Mastra accepts a raw JSON Schema; ingest reads it back off the normalized schema.
      inputSchema: {
        type: "object",
        properties: { q: { type: "string" } },
        required: ["q"],
      } as never,
      execute: async () => ({}),
    });
    const reg = mastra().ingest("raw", t);
    if (reg === "passthrough") throw new Error("expected an executable registration");
    const input = reg.inputSchema as JsonObjectSchema;
    expect(input.properties?.q.type).toBe("string");
  });

  it("defaults a schema-less tool to an object schema", () => {
    const t = createTool({
      id: "bare",
      description: "no schema",
      execute: async () => ({ ok: true }),
    });
    const reg = mastra().ingest("bare", t);
    if (reg === "passthrough") throw new Error("expected an executable registration");
    expect(reg.inputSchema).toEqual({ type: "object" });
  });

  it("converts the output schema when the tool declares one", () => {
    const t = createTool({
      id: "structured",
      description: "structured",
      inputSchema: z.object({}),
      outputSchema: z.object({ ok: z.boolean() }),
      execute: async () => ({ ok: true }),
    });
    const reg = mastra().ingest("structured", t);
    if (reg === "passthrough") throw new Error("expected an executable registration");
    const output = reg.outputSchema as JsonObjectSchema;
    expect(output.properties?.ok.type).toBe("boolean");
  });

  it("surfaces a Mastra ValidationError as a thrown error, not a silent result", async () => {
    const weather = createTool({
      id: "weather",
      description: "Get the weather in a location",
      inputSchema: z.object({ location: z.string() }),
      execute: async ({ location }) => ({ location }),
    });
    const reg = mastra().ingest("weather", weather);
    if (reg === "passthrough") throw new Error("expected an executable registration");
    // Good args pass through the createTool validation wrapper.
    expect(await reg.execute({ location: "Paris" })).toEqual({ location: "Paris" });
    // Missing required arg: Mastra returns a ValidationError object (does not throw);
    // the adapter re-throws so the capability funnel frames it as a failed call.
    await expect(reg.execute({})).rejects.toThrow();
  });

  it("does not mistake an ordinary error-shaped tool result for a ValidationError", async () => {
    const domainFailure = { error: true, message: "card declined" };
    const charge = createTool({
      id: "charge",
      description: "Charge a card",
      execute: async () => domainFailure,
    });
    const reg = mastra().ingest("charge", charge);
    if (reg === "passthrough") throw new Error("expected an executable registration");

    await expect(reg.execute({})).resolves.toEqual(domainFailure);
  });

  it("fabricates a minimal Mastra context when the catalog runs the tool", async () => {
    let seenContext: Record<string, unknown> | undefined;
    const view = ratel().adaptTo(mastra());
    await view.tools.register({
      spy: createTool({
        id: "spy",
        description: "spy",
        execute: async (_input, context) => {
          seenContext = context as Record<string, unknown>;
          return { ran: true };
        },
      }),
    });
    const out = await view.tools.catalog.invoke("spy", {});
    expect(out).toEqual({ ran: true });
    // The direct-call fallback carries the no-op `observe` and a fresh requestContext;
    // a tool reading `agent` / `mastra` / `workflow` sees undefined, not a crash.
    expect(seenContext?.observe).toBeDefined();
    expect(seenContext?.requestContext).toBeDefined();
    expect(seenContext?.agent).toBeUndefined();
    expect(seenContext?.mastra).toBeUndefined();
  });

  it("preserves the caller's Mastra execution context through invoke_tool", async () => {
    const requestContext = new RequestContext([["tenantId", "tenant-42"]]);
    const workspace = { marker: "workspace" };
    const agent = {
      toolCallId: "call-1",
      messages: [],
      suspend: async () => {},
      threadId: "thread-9",
      resourceId: "resource-3",
    };
    const mastraInstance = { marker: "mastra" };
    const abortSignal = new AbortController().signal;
    const core = ratel();
    const registrationView = core.adaptTo(mastra());
    const invocationView = core.adaptTo(mastra());
    await registrationView.tools.register({
      tenant_probe: createTool({
        id: "tenant_probe",
        description: "Read the current tenant",
        execute: async (_input, context) => ({
          sameContext: context.requestContext === requestContext,
          tenantId: context.requestContext?.get("tenantId"),
          sameWorkspace: context.workspace === workspace,
          threadId: context.agent?.threadId,
          resourceId: context.agent?.resourceId,
          sameMastra: context.mastra === mastraInstance,
          sameAbortSignal: context.abortSignal === abortSignal,
        }),
      }),
    });

    const out = await callExposed(
      invocationView.modelTools()[INVOKE_TOOL_ID] as Tool,
      {
        toolId: "tenant_probe",
        args: {},
      },
      {
        requestContext,
        workspace,
        agent,
        mastra: mastraInstance,
        abortSignal,
      },
    );

    expect(out).toEqual({
      sameContext: true,
      tenantId: "tenant-42",
      sameWorkspace: true,
      threadId: "thread-9",
      resourceId: "resource-3",
      sameMastra: true,
      sameAbortSignal: true,
    });
  });

  it("satisfies a registered tool's request context schema through invoke_tool", async () => {
    const requestContext = new RequestContext([["actorId", "actor-7"]]);
    const view = ratel().adaptTo(mastra());
    await view.tools.register({
      actor_probe: createTool({
        id: "actor_probe",
        description: "Read the authorized actor",
        requestContextSchema: z.object({ actorId: z.string() }),
        execute: async (_input, context) => ({
          actorId: context.requestContext?.get("actorId"),
        }),
      }),
    });

    const out = await callExposed(
      view.modelTools()[INVOKE_TOOL_ID] as Tool,
      {
        toolId: "actor_probe",
        args: {},
      },
      { requestContext },
    );

    expect(out).toEqual({ actorId: "actor-7" });
  });

  it("isolates request contexts across concurrent invoke_tool calls", async () => {
    let arrivals = 0;
    let releaseBoth!: () => void;
    const bothArrived = new Promise<void>((resolve) => {
      releaseBoth = resolve;
    });
    const view = ratel().adaptTo(mastra());
    await view.tools.register({
      tenant_probe: createTool({
        id: "tenant_probe",
        description: "Read the current tenant after calls overlap",
        execute: async (_input, context) => {
          arrivals += 1;
          if (arrivals === 2) releaseBoth();
          await bothArrived;
          return { tenantId: context.requestContext?.get("tenantId") };
        },
      }),
    });
    const invokeTool = view.modelTools()[INVOKE_TOOL_ID] as Tool;

    const invokeFor = (tenantId: string) =>
      callExposed(
        invokeTool,
        { toolId: "tenant_probe", args: {} },
        {
          requestContext: new RequestContext([["tenantId", tenantId]]),
        },
      );

    expect(await Promise.all([invokeFor("tenant-a"), invokeFor("tenant-b")])).toEqual([
      { tenantId: "tenant-a" },
      { tenantId: "tenant-b" },
    ]);
  });

  it("does not mistake a lookalike foreign context for Mastra context", async () => {
    const core = ratel();
    const view = core.adaptTo(mastra());
    await view.tools.register({
      tenant_probe: createTool({
        id: "tenant_probe",
        description: "Read the current tenant",
        execute: async (_input, context) => ({
          tenantId: context.requestContext?.get("tenantId"),
        }),
      }),
    });

    const result = await core.modelTools()[INVOKE_TOOL_ID].execute(
      { toolId: "tenant_probe", args: {} },
      {
        adapter: "mastra",
        value: { requestContext: new RequestContext([["tenantId", "foreign-tenant"]]) },
      },
    );

    expect(result).toEqual({ tenantId: undefined });
  });
});

describe("expose codec", () => {
  it("wraps a Ratel capability tool as a genuine Mastra tool that delegates execute", async () => {
    const capability = {
      id: SEARCH_CAPABILITIES_ID,
      name: SEARCH_CAPABILITIES_ID,
      description: "search the catalog",
      inputSchema: { type: "object", properties: { query: { type: "string" } } },
      outputSchema: { type: "object" },
      execute: async (args: unknown) => ({ echoed: args }),
    };
    const exposed = mastra().expose(capability);
    expect(exposed).toBeInstanceOf(Tool);
    expect(exposed.description).toBe("search the catalog");
    expect(typeof exposed.execute).toBe("function");
    // Execution delegates to the capability's own executor (args validated by the
    // hand-written zod schema, then forwarded).
    expect(await callExposed(exposed, { query: "hi" })).toEqual({ echoed: { query: "hi" } });
  });

  it("preserves the capability tools' canonical parameter descriptions for the model", () => {
    const view = ratel().adaptTo(mastra());
    const exposed = view.modelTools();
    const jsonOf = (tool: Tool) =>
      (
        tool.inputSchema as {
          "~standard": { jsonSchema: { input(o: { target: string }): Record<string, unknown> } };
        }
      )["~standard"].jsonSchema.input({ target: "draft-07" }) as {
        properties: Record<string, { description?: string }>;
      };
    const search = jsonOf(exposed[SEARCH_CAPABILITIES_ID] as Tool);
    expect(search.properties.query.description).toBe("describe what you want to do");
    expect(search.properties.topKTools.description).toContain("default 5");
    const skill = jsonOf(exposed[GET_SKILL_CONTENT_ID] as Tool);
    expect(skill.properties.skillId.description).toContain("id of the skill to load");
  });

  it("is permissive on topK so out-of-range values reach the core's clamp", async () => {
    const view = ratel({ recallTopK: 5 }).adaptTo(mastra());
    view.tools.register(
      Object.fromEntries(
        Array.from({ length: 60 }, (_, i) => [
          `grep_${i}`,
          createTool({
            id: `grep_${i}`,
            description: `Search files variant ${i}: grep ripgrep.`,
            execute: async () => ({ ok: true }),
          }),
        ]),
      ),
    );
    const search = view.modelTools()[SEARCH_CAPABILITIES_ID];
    // A value far above the default is honoured, then clamped to 50 by the core —
    // it is NOT rejected by the exposed tool's schema.
    const high = (await callExposed(search, {
      query: "grep files",
      topKTools: 999,
    })) as SearchCapabilitiesResult;
    expect(countHits(high)).toBeGreaterThan(5);
    expect(countHits(high)).toBeLessThanOrEqual(50);
    // A negative value is accepted too and falls back to the default inside the core.
    const low = (await callExposed(search, {
      query: "grep files",
      topKTools: -1,
    })) as SearchCapabilitiesResult;
    expect(countHits(low)).toBeLessThanOrEqual(5);
  });

  it("does not strip a nested args object for invoke_tool", async () => {
    const view = ratel().adaptTo(mastra());
    view.tools.register({
      echo_tool: createTool({
        id: "echo_tool",
        description: "echo the args back",
        execute: async (input) => ({ got: input }),
      }),
    });
    const invoke = view.modelTools()[INVOKE_TOOL_ID];
    const result = await callExposed(invoke, {
      toolId: "echo_tool",
      args: { path: "/tmp/x", n: 3 },
    });
    expect(result).toEqual({ got: { path: "/tmp/x", n: 3 } });
  });
});

describe("recallMessages codec", () => {
  it("renders the recall as one assistant message with a resolved tool-invocation", () => {
    const recall = {
      tools: { groups: [{ server: { name: "fs" }, hits: [{ toolId: "read_file", score: 1 }] }] },
      skills: [],
    } as unknown as SearchCapabilitiesResult;
    const messages = mastra().recallMessages({ callId: "recall_0", query: "read a file" }, recall);
    expect(messages).toHaveLength(1);
    const [message] = messages;
    expect(message.role).toBe("assistant");
    expect(message.content.format).toBe(2);
    const part = message.content.parts[0] as {
      type: string;
      toolInvocation: Record<string, unknown>;
    };
    expect(part.type).toBe("tool-invocation");
    expect(part.toolInvocation.state).toBe("result");
    expect(part.toolInvocation.toolCallId).toBe("recall_0");
    expect(part.toolInvocation.toolName).toBe(SEARCH_CAPABILITIES_ID);
    expect(part.toolInvocation.args).toEqual({ query: "read a file" });
    expect(part.toolInvocation.result).toEqual(recall);
  });
});

describe("recallProcessor (extend)", () => {
  // The processor's processInput, called directly with a partial args object (it
  // reads only `messages`). Cast keeps the tests off the wide ProcessInputArgs type.
  function inject(view: ReturnType<typeof viewWithDeployTool>, messages: MastraDBMessage[]) {
    const processor = view.recallProcessor() as {
      processInput: (args: { messages: MastraDBMessage[] }) => Promise<MastraDBMessage[]>;
    };
    return processor.processInput({ messages });
  }
  function toolCallId(message: MastraDBMessage): unknown {
    return (message.content.parts[0] as { toolInvocation: { toolCallId: unknown } }).toolInvocation
      .toolCallId;
  }

  it("appends the recall message after a user turn with hits (spends recall_0)", async () => {
    const view = viewWithDeployTool();
    const out = await inject(view, [userMsg("deploy to production")]);
    expect(out).toHaveLength(2);
    expect(out[1].role).toBe("assistant");
    expect(toolCallId(out[1])).toBe("recall_0");
  });

  it("no-ops (no id spent) when the last message is not a user turn", async () => {
    const view = viewWithDeployTool();
    const assistant: MastraDBMessage = {
      id: "a1",
      role: "assistant",
      createdAt: new Date(),
      content: { format: 2, parts: [{ type: "text", text: "on it" }] },
    };
    const out = await inject(view, [userMsg("deploy to production"), assistant]);
    expect(out).toHaveLength(2);
    // No id spent, so the next real recall on the same view is still recall_0.
    const real = await inject(view, [userMsg("deploy to production")]);
    expect(toolCallId(real[1])).toBe("recall_0");
  });

  it("no-ops on empty user text and on a zero-hit query, preserving id economy", async () => {
    const view = viewWithDeployTool();
    expect(await inject(view, [userMsg("")])).toHaveLength(1);
    expect(await inject(view, [userMsg("zzzqqq utterly unrelated")])).toHaveLength(1);
    const real = await inject(view, [userMsg("deploy to production")]);
    expect(toolCallId(real[1])).toBe("recall_0");
  });

  it("joins multi-part user text with newlines for the recall query", async () => {
    const view = viewWithDeployTool();
    const out = await inject(view, [userMsg("deploy", "to production")]);
    const args = (out[1].content.parts[0] as { toolInvocation: { args: unknown } }).toolInvocation
      .args;
    expect(args).toEqual({ query: "deploy\nto production" });
  });
});
