import {
  INVOKE_TOOL_ID,
  ratel,
  SEARCH_CAPABILITIES_ID,
  type SearchCapabilitiesResult,
} from "@ratel-ai/sdk";
import { jsonSchema, type ModelMessage, type Tool, tool } from "ai";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { aiSdk } from "./aisdk.js";

// A view over a fresh core with one BM25-discoverable executable tool.
async function viewWithDeployTool() {
  const view = ratel().adaptTo(aiSdk());
  await view.tools.register({
    deploy_app: tool({
      description: "Deploy the app to production servers.",
      inputSchema: z.object({}),
      execute: async () => ({ deployed: true }),
    }),
  });
  return view;
}

// The tool-call part of a synthetic recall's assistant message.
function callPartOf(message: ModelMessage): Record<string, unknown> {
  return (message.content as Array<Record<string, unknown>>)[0];
}

// A JSON Schema whose `properties` we read positionally in assertions.
type JsonObjectSchema = { type?: string; properties?: Record<string, { type?: string }> };

describe("aiSdk() identity", () => {
  it('names itself "ai-sdk" — matches the core KNOWN_FRAMEWORKS table', () => {
    expect(aiSdk().name).toBe("ai-sdk");
  });
});

describe("ingest codec", () => {
  it("passes through a function tool with no execute (provider/client-executed)", () => {
    const providerTool = tool({
      description: "provider-run search",
      inputSchema: z.object({ q: z.string() }),
    });
    expect(aiSdk().ingest("provider_search", providerTool)).toBe("passthrough");
  });

  it("passes through an ai@7 provider tool even when it carries a client execute", () => {
    // Provider-DEFINED tools (anthropic.tools.bash_*/computer_*/textEditor_*) are
    // type:'provider', isProviderExecuted:false, yet run a client-side execute. The
    // catalog can't carry their type / <provider>.<tool> id / args (and they have no
    // rankable description), so they must stay eagerly exposed in native shape.
    const providerTool = {
      type: "provider",
      id: "acme.shell",
      args: {},
      isProviderExecuted: false,
      inputSchema: { type: "object" },
      execute: async () => ({ ran: true }),
    } as unknown as Tool;
    expect(aiSdk().ingest("shell", providerTool)).toBe("passthrough");
  });

  it("passes through an ai@5 provider-defined tool even when it carries a client execute", () => {
    const providerTool = {
      type: "provider-defined",
      id: "acme.shell",
      args: {},
      inputSchema: { type: "object" },
      execute: async () => ({ ran: true }),
    } as unknown as Tool;
    expect(aiSdk().ingest("shell", providerTool)).toBe("passthrough");
  });

  it.each([
    ["approval", { needsApproval: true }],
    ["context", { contextSchema: z.object({ tenantId: z.string() }) }],
    ["input lifecycle", { onInputStart: async () => {} }],
    [
      "model output",
      { toModelOutput: async () => ({ type: "text" as const, value: "formatted" }) },
    ],
    ["provider options", { providerOptions: { acme: { mode: "strict" } } }],
  ])("passes through a function tool with native %s semantics", (_name, extension) => {
    const native = {
      ...tool({
        description: "native lifecycle",
        inputSchema: z.object({}),
        execute: async () => ({ ok: true }),
      }),
      ...extension,
    } as unknown as Tool;

    expect(aiSdk().ingest("native", native)).toBe("passthrough");
  });

  it("ingests an executable: resolves description, converts input schema, omits output schema", () => {
    const weather = tool({
      description: "Get the weather in a location",
      inputSchema: z.object({ location: z.string() }),
      execute: async ({ location }) => ({ location, tempF: 70 }),
    });
    const reg = aiSdk().ingest("weather", weather);
    if (reg === "passthrough") throw new Error("expected an executable registration");
    expect(reg.description).toBe("Get the weather in a location");
    const input = reg.inputSchema as JsonObjectSchema;
    expect(input.type).toBe("object");
    expect(input.properties?.location.type).toBe("string");
    // Absent output schema stays absent — the core defaults it, the adapter never does.
    expect(reg.outputSchema).toBeUndefined();
  });

  it("converts the output schema when the framework tool declares one", () => {
    const t = tool({
      description: "structured",
      inputSchema: z.object({}),
      outputSchema: z.object({ ok: z.boolean() }),
      execute: async () => ({ ok: true }),
    });
    const reg = aiSdk().ingest("structured", t);
    if (reg === "passthrough") throw new Error("expected an executable registration");
    const output = reg.outputSchema as JsonObjectSchema;
    expect(output.type).toBe("object");
    expect(output.properties?.ok.type).toBe("boolean");
  });

  it("resolves a dynamic description once at ingest time, with a null tool context", () => {
    let calls = 0;
    let seenContext: unknown = "unset";
    const t = tool({
      // ai lets a tool description be a function of the live tool context; recall
      // ranks at ingest time, so resolve it now with a fabricated null context.
      description: ((opts: { context: unknown }) => {
        calls++;
        seenContext = opts.context;
        return "dynamic description";
      }) as unknown as string,
      inputSchema: z.object({}),
      execute: async () => ({}),
    });
    const reg = aiSdk().ingest("dyn", t);
    if (reg === "passthrough") throw new Error("expected an executable registration");
    expect(reg.description).toBe("dynamic description");
    expect(calls).toBe(1);
    expect(seenContext).toBeUndefined();
  });

  it("fabricates AI SDK 5–7 execution options when the catalog runs the tool", async () => {
    let seenOptions: Record<string, unknown> | undefined;
    const t = tool({
      description: "spy",
      inputSchema: z.object({}),
      execute: ((_input: unknown, options: Record<string, unknown>) => {
        seenOptions = options;
        return { ran: true };
      }) as unknown as Tool["execute"],
    });
    const reg = aiSdk().ingest("spy", t);
    if (reg === "passthrough") throw new Error("expected an executable registration");
    const out = await reg.execute({});
    expect(out).toEqual({ ran: true });
    // ai@5/6 read experimental_context; ai@7 reads context. Both exist so a
    // tool sees an explicit fake rather than crashing on a missing field.
    expect(seenOptions && "context" in seenOptions).toBe(true);
    expect(seenOptions && "experimental_context" in seenOptions).toBe(true);
    expect(seenOptions?.context).toBeUndefined();
    expect(seenOptions?.experimental_context).toBeUndefined();
    expect(seenOptions?.toolCallId).toBe("ratel_spy");
    expect(seenOptions?.messages).toEqual([]);
  });

  it("rejects an asynchronous input schema synchronously without partially registering a batch", () => {
    const view = ratel().adaptTo(aiSdk());
    const valid = tool({
      description: "valid",
      inputSchema: z.object({}),
      execute: async () => ({ ok: true }),
    });
    const asynchronous = tool({
      description: "async schema",
      inputSchema: jsonSchema(Promise.resolve({ type: "object" })),
      execute: async () => ({ ok: true }),
    });

    expect(() => view.tools.register({ valid, asynchronous })).toThrow(
      'ratel: AI SDK tool "asynchronous" has an asynchronous inputSchema; @ratel-ai/vercel-ai-sdk requires schemas to resolve synchronously',
    );
    expect(view.tools.has("valid")).toBe(false);
    expect(view.tools.has("asynchronous")).toBe(false);
  });

  it("identifies an asynchronous output schema by tool and field", () => {
    const view = ratel().adaptTo(aiSdk());
    const asynchronous = tool({
      description: "async output schema",
      inputSchema: z.object({}),
      outputSchema: jsonSchema(Promise.resolve({ type: "object" })),
      execute: async () => ({ ok: true }),
    });

    expect(() => view.tools.register({ asynchronous })).toThrow(
      'ratel: AI SDK tool "asynchronous" has an asynchronous outputSchema; @ratel-ai/vercel-ai-sdk requires schemas to resolve synchronously',
    );
    expect(view.tools.has("asynchronous")).toBe(false);
  });
});

describe("expose codec", () => {
  it("wraps a Ratel capability tool as a framework tool, framework-shaped (no id/outputSchema)", async () => {
    const capability = {
      id: "cap",
      name: "cap",
      description: "a capability tool",
      inputSchema: { type: "object", properties: { q: { type: "string" } } },
      outputSchema: { type: "object" },
      execute: async (args: unknown) => ({ echoed: args }),
    };
    const exposed = aiSdk().expose(capability) as Tool & { id?: unknown; outputSchema?: unknown };
    expect(exposed.description).toBe("a capability tool");
    expect("id" in exposed).toBe(false);
    expect("outputSchema" in exposed).toBe(false);
    // Execution delegates to the capability's own executor.
    const run = exposed.execute as (args: unknown, opts: unknown) => Promise<unknown>;
    expect(await run({ q: "hi" }, {})).toEqual({ echoed: { q: "hi" } });
  });
});

describe("live execution context", () => {
  // The private, package-stable tag the adapter wraps the framework's live
  // execution context under (ADR-0013: opaque to the core, unwrappable only here).
  const AI_SDK_CONTEXT_KEY = Symbol.for("@ratel-ai/vercel-ai-sdk.execution-context");

  it("expose passes the live options to the capability executor under the private tag", async () => {
    let seenContext: unknown;
    const capability = {
      id: "cap",
      name: "cap",
      description: "a capability tool",
      inputSchema: { type: "object" },
      outputSchema: { type: "object" },
      execute: async (_args: unknown, context: unknown) => {
        seenContext = context;
        return { ok: true };
      },
    };
    const exposed = aiSdk().expose(capability);
    const options = { toolCallId: "x", messages: [] };
    await (exposed.execute as (a: unknown, o: unknown) => Promise<unknown>)({}, options);
    // The whole options object rides under the private symbol — opaque to the core.
    expect((seenContext as Record<symbol, unknown>)[AI_SDK_CONTEXT_KEY]).toBe(options);
  });

  it("threads the caller's live options through invoke_tool to the ingested tool", async () => {
    let seen: Record<string, unknown> | undefined;
    const view = ratel().adaptTo(aiSdk());
    view.tools.register({
      probe: tool({
        description: "reads its live execution context",
        inputSchema: z.object({}),
        execute: ((_input: unknown, options: Record<string, unknown>) => {
          seen = options;
          return { ok: true };
        }) as unknown as Tool["execute"],
      }),
    });

    const liveOptions = {
      toolCallId: "call-live-1",
      messages: [{ role: "assistant", content: "prior turn" }],
      experimental_context: { tenantId: "tenant-42" },
      context: { tenantId: "tenant-42" },
      abortSignal: new AbortController().signal,
    };
    const invokeTool = view.modelTools()[INVOKE_TOOL_ID] as Tool;
    const run = invokeTool.execute as (args: unknown, options: unknown) => Promise<unknown>;
    const out = await run({ toolId: "probe", args: {} }, liveOptions);

    expect(out).toEqual({ ok: true });
    // The ingested tool receives the caller's real options object by identity.
    expect(seen).toBe(liveOptions);
  });

  it("falls back to fabricated options for a direct catalog invocation", async () => {
    let seen: Record<string, unknown> | undefined;
    const view = ratel().adaptTo(aiSdk());
    view.tools.register({
      probe: tool({
        description: "reads its live execution context",
        inputSchema: z.object({}),
        execute: ((_input: unknown, options: Record<string, unknown>) => {
          seen = options;
          return { ok: true };
        }) as unknown as Tool["execute"],
      }),
    });

    // The driver-level escape hatch has no AI SDK invocation to thread.
    await view.tools.catalog.invoke("probe", {});

    expect(seen?.toolCallId).toBe("ratel_probe");
    expect(seen?.messages).toEqual([]);
    expect(seen && "context" in seen).toBe(true);
    expect(seen && "experimental_context" in seen).toBe(true);
  });

  it("does not unwrap a foreign adapter's tagged context (fabricates instead)", async () => {
    let seen: Record<string, unknown> | undefined;
    const view = ratel().adaptTo(aiSdk());
    view.tools.register({
      probe: tool({
        description: "reads its live execution context",
        inputSchema: z.object({}),
        execute: ((_input: unknown, options: Record<string, unknown>) => {
          seen = options;
          return { ok: true };
        }) as unknown as Tool["execute"],
      }),
    });

    // A sibling view over the same catalog tags with a different private symbol;
    // its context must never be mistaken for live AI SDK options.
    const foreign = {
      [Symbol.for("@ratel-ai/mastra.execution-context")]: { requestContext: "x" },
    };
    await view.tools.catalog.invoke("probe", {}, foreign);

    // Fabricated fallback fired, and — the ADR-0013 cross-view property — the
    // sibling view's context did not leak in: no foreign symbol tag, no payload.
    expect(seen?.toolCallId).toBe("ratel_probe");
    expect(Object.getOwnPropertySymbols(seen ?? {})).toHaveLength(0);
    expect(seen && "requestContext" in seen).toBe(false);
  });

  it("unwraps its own tag on a direct catalog invocation", async () => {
    let seen: Record<string, unknown> | undefined;
    const view = ratel().adaptTo(aiSdk());
    view.tools.register({
      probe: tool({
        description: "reads its live execution context",
        inputSchema: z.object({}),
        execute: ((_input: unknown, options: Record<string, unknown>) => {
          seen = options;
          return { ok: true };
        }) as unknown as Tool["execute"],
      }),
    });

    // The positive counterpart to the foreign-tag test: this adapter's own tag
    // unwraps to the live options by identity, even on the direct catalog path.
    const liveOptions = { toolCallId: "call-direct", messages: [] };
    await view.tools.catalog.invoke("probe", {}, { [AI_SDK_CONTEXT_KEY]: liveOptions });

    expect(seen).toBe(liveOptions);
  });
});

describe("passthrough end-to-end", () => {
  it("exposes a provider tool by identity and never catalogs it", () => {
    const view = ratel().adaptTo(aiSdk());
    const providerTool = {
      type: "provider",
      id: "acme.shell",
      args: {},
      isProviderExecuted: false,
      inputSchema: { type: "object" },
      execute: async () => ({ ran: true }),
    } as unknown as Tool;
    view.tools.register({ shell: providerTool });
    // Exposed untouched, in native provider shape...
    expect(view.modelTools().shell).toBe(providerTool);
    // ...and never funneled into the catalog.
    expect(view.tools.catalog.has("shell")).toBe(false);
  });
});

describe("recallMessages codec", () => {
  it("renders the synthetic search_capabilities call/result pair in ModelMessage shape", () => {
    const recall = {
      tools: { groups: [{ server: { name: "fs" }, hits: [{ toolId: "read_file", score: 1 }] }] },
      skills: [],
    } as unknown as SearchCapabilitiesResult;
    const [call, result] = aiSdk().recallMessages(
      { callId: "recall_0", query: "read a file" },
      recall,
    );

    expect(call.role).toBe("assistant");
    const callPart = (call.content as Array<Record<string, unknown>>)[0];
    expect(callPart.type).toBe("tool-call");
    expect(callPart.toolCallId).toBe("recall_0");
    expect(callPart.toolName).toBe(SEARCH_CAPABILITIES_ID);
    expect(callPart.input).toEqual({ query: "read a file" });

    expect(result.role).toBe("tool");
    const resultPart = (result.content as Array<Record<string, unknown>>)[0];
    expect(resultPart.type).toBe("tool-result");
    expect(resultPart.toolCallId).toBe("recall_0");
    expect(resultPart.toolName).toBe(SEARCH_CAPABILITIES_ID);
    // The result is carried as a JSON text part the model reads back.
    expect(resultPart.output).toEqual({ type: "text", value: JSON.stringify(recall) });
  });
});

describe("appendRecall (extend)", () => {
  it("appends the recall pair at the suffix and returns the same array reference", async () => {
    const view = await viewWithDeployTool();
    const messages: ModelMessage[] = [{ role: "user", content: "deploy to production" }];
    const returned = await view.appendRecall(messages);
    // Mutate-and-return: a suffix append extends the cached prefix.
    expect(returned).toBe(messages);
    expect(messages).toHaveLength(3);
    expect(messages[1].role).toBe("assistant");
    expect(messages[2].role).toBe("tool");
    // The first recall spends recall_0 (ids come from the core's counter).
    expect(callPartOf(messages[1]).toolCallId).toBe("recall_0");
  });

  it("no-ops (array untouched, no id spent) when the last message is not a user turn", async () => {
    const view = await viewWithDeployTool();
    const messages: ModelMessage[] = [
      { role: "user", content: "deploy to production" },
      { role: "assistant", content: "on it" },
    ];
    expect(await view.appendRecall(messages)).toHaveLength(2);
    // No id spent, so the next real recall is still recall_0.
    const fresh: ModelMessage[] = [{ role: "user", content: "deploy to production" }];
    await view.appendRecall(fresh);
    expect(callPartOf(fresh[1]).toolCallId).toBe("recall_0");
  });

  it("no-ops on empty user text and on a zero-hit query, preserving id economy", async () => {
    const view = await viewWithDeployTool();
    expect(await view.appendRecall([{ role: "user", content: "" }])).toHaveLength(1);
    expect(
      await view.appendRecall([{ role: "user", content: "zzzqqq utterly unrelated" }]),
    ).toHaveLength(1);
    const real: ModelMessage[] = [{ role: "user", content: "deploy to production" }];
    await view.appendRecall(real);
    expect(callPartOf(real[1]).toolCallId).toBe("recall_0");
  });

  it("joins multi-part user text with newlines for the recall query", async () => {
    const view = await viewWithDeployTool();
    const messages: ModelMessage[] = [
      {
        role: "user",
        content: [
          { type: "text", text: "deploy" },
          { type: "text", text: "to production" },
        ],
      },
    ];
    await view.appendRecall(messages);
    expect(messages).toHaveLength(3);
    expect(callPartOf(messages[1]).input).toEqual({ query: "deploy\nto production" });
  });
});

describe("prepareStep (extend)", () => {
  it("injects a fresh messages array on step 0, leaving the caller's array untouched", async () => {
    const view = await viewWithDeployTool();
    const messages: ModelMessage[] = [{ role: "user", content: "deploy to production" }];
    const result = await view.prepareStep({ stepNumber: 0, messages });
    const injected = (result as { messages: ModelMessage[] }).messages;
    expect(injected).toHaveLength(3);
    // Never mutates: ai's messages override carries forward across steps.
    expect(injected).not.toBe(messages);
    expect(messages).toHaveLength(1);
    // The recall pair is appended as a suffix — the user turn stays first (by
    // identity), the assistant call and tool result follow, in order.
    expect(injected[0]).toBe(messages[0]);
    expect(injected[1].role).toBe("assistant");
    expect(injected[2].role).toBe("tool");
    expect(callPartOf(injected[1]).toolCallId).toBe("recall_0");
  });

  it("returns undefined on later steps, a non-user last message, and zero hits (id economy)", async () => {
    const view = await viewWithDeployTool();
    const messages: ModelMessage[] = [{ role: "user", content: "deploy to production" }];
    expect(await view.prepareStep({ stepNumber: 1, messages })).toBeUndefined();
    expect(
      await view.prepareStep({ stepNumber: 0, messages: [{ role: "assistant", content: "hi" }] }),
    ).toBeUndefined();
    expect(
      await view.prepareStep({
        stepNumber: 0,
        messages: [{ role: "user", content: "zzzqqq utterly unrelated" }],
      }),
    ).toBeUndefined();
    // None of those spent a call id: a real step-0 injection is still recall_0.
    const injected = await view.prepareStep({ stepNumber: 0, messages });
    expect(callPartOf((injected as { messages: ModelMessage[] }).messages[1]).toolCallId).toBe(
      "recall_0",
    );
  });

  it("reinserts the step-0 recall before accumulated responses when a later prompt drops it", async () => {
    const view = await viewWithDeployTool();
    const steps: unknown[] = [];
    const initial: ModelMessage[] = [{ role: "user", content: "deploy to production" }];
    const first = await view.prepareStep({ stepNumber: 0, messages: initial, steps });
    const firstMessages = (first as { messages: ModelMessage[] }).messages;
    const accumulated: ModelMessage[] = [
      ...initial,
      { role: "assistant", content: "working on it" },
    ];

    const later = await view.prepareStep({ stepNumber: 1, messages: accumulated, steps });
    const laterMessages = (later as { messages: ModelMessage[] }).messages;

    expect(laterMessages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
      "tool",
      "assistant",
    ]);
    expect(callPartOf(laterMessages[1]).toolCallId).toBe(callPartOf(firstMessages[1]).toolCallId);
    expect(accumulated).toHaveLength(2);
  });

  it("does not duplicate a cloned recall pair that a later prompt already carries", async () => {
    const view = await viewWithDeployTool();
    const steps: unknown[] = [];
    const initial: ModelMessage[] = [{ role: "user", content: "deploy to production" }];
    const first = await view.prepareStep({ stepNumber: 0, messages: initial, steps });
    const carried = structuredClone((first as { messages: ModelMessage[] }).messages);
    carried.push({ role: "assistant", content: "working on it" });

    expect(await view.prepareStep({ stepNumber: 1, messages: carried, steps })).toBeUndefined();
  });

  it("keeps interleaved runs isolated and performs recall only once per run", async () => {
    const view = await viewWithDeployTool();
    const stepsA: unknown[] = [];
    const stepsB: unknown[] = [];
    const initialA: ModelMessage[] = [{ role: "user", content: "deploy app A" }];
    const initialB: ModelMessage[] = [{ role: "user", content: "deploy app B" }];
    await view.prepareStep({ stepNumber: 0, messages: initialA, steps: stepsA });
    await view.prepareStep({ stepNumber: 0, messages: initialB, steps: stepsB });

    const laterA = await view.prepareStep({
      stepNumber: 1,
      messages: [...initialA, { role: "assistant", content: "A" }],
      steps: stepsA,
    });
    const laterB = await view.prepareStep({
      stepNumber: 1,
      messages: [...initialB, { role: "assistant", content: "B" }],
      steps: stepsB,
    });
    const messagesA = (laterA as { messages: ModelMessage[] }).messages;
    const messagesB = (laterB as { messages: ModelMessage[] }).messages;
    expect(callPartOf(messagesA[1]).toolCallId).toBe("recall_0");
    expect(callPartOf(messagesB[1]).toolCallId).toBe("recall_1");

    const next: ModelMessage[] = [{ role: "user", content: "deploy app C" }];
    await view.appendRecall(next);
    expect(callPartOf(next[1]).toolCallId).toBe("recall_2");
  });

  it("stores no run state and spends no id when step 0 has no hits", async () => {
    const view = await viewWithDeployTool();
    const steps: unknown[] = [];
    const misses: ModelMessage[] = [{ role: "user", content: "zzzqqq utterly unrelated" }];
    expect(await view.prepareStep({ stepNumber: 0, messages: misses, steps })).toBeUndefined();
    expect(await view.prepareStep({ stepNumber: 1, messages: misses, steps })).toBeUndefined();

    const real: ModelMessage[] = [{ role: "user", content: "deploy to production" }];
    await view.appendRecall(real);
    expect(callPartOf(real[1]).toolCallId).toBe("recall_0");
  });
});
