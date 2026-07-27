import { ratel } from "@ratel-ai/sdk";
import {
  generateText,
  jsonSchema,
  type LanguageModel,
  type ModelMessage,
  stepCountIs,
  streamText,
  tool,
} from "ai";
import { describe, expect, it } from "vitest";
import { z } from "zod";
import { aiSdk } from "./aisdk.js";
import { aiSdkMajor, MockLanguageModelV2, usage } from "./test-support/mock-model.js";

describe("AI SDK loop integration", () => {
  it.skipIf(aiSdkMajor < 6)(
    "keeps approval-gated tools native so the host blocks execution",
    async () => {
      const view = ratel().adaptTo(aiSdk());
      let executions = 0;
      await view.tools.register({
        delete_account: tool({
          description: "Permanently delete an account.",
          inputSchema: z.object({ accountId: z.string() }),
          needsApproval: true,
          execute: async () => {
            executions++;
            return { deleted: true };
          },
        }),
      });
      const model = new MockLanguageModelV2([
        {
          content: [
            {
              type: "tool-call",
              toolCallId: "delete-0",
              toolName: "delete_account",
              input: JSON.stringify({ accountId: "acct-1" }),
            },
          ],
          finishReason: "tool-calls",
          usage,
          warnings: [],
        },
      ]);

      const result = await generateText({
        model: model as unknown as LanguageModel,
        tools: view.modelTools(),
        prompt: "delete acct-1",
      });

      expect(executions).toBe(0);
      expect(result.content.map((part) => part.type)).toContain("tool-approval-request");
    },
  );

  it.skipIf(aiSdkMajor < 7)("keeps target context routing and validation native", async () => {
    const view = ratel().adaptTo(aiSdk());
    let receivedContext: unknown;
    const tenantProbe = tool({
      description: "Read the active tenant.",
      inputSchema: z.object({}),
      contextSchema: z.object({
        tenantId: z
          .string()
          .trim()
          .transform((value) => value.toUpperCase()),
      }),
      execute: async (_input, options) => {
        receivedContext = options.context;
        return { ok: true };
      },
    });
    await view.tools.register({ tenant_probe: tenantProbe });
    const tools = view.modelTools();
    const model = new MockLanguageModelV2([
      {
        content: [
          {
            type: "tool-call",
            toolCallId: "tenant-0",
            toolName: "tenant_probe",
            input: JSON.stringify({}),
          },
        ],
        finishReason: "tool-calls",
        usage,
        warnings: [],
      },
    ]);

    await generateText({
      model: model as unknown as LanguageModel,
      tools,
      toolsContext: { tenant_probe: { tenantId: "  acme  " } },
      prompt: "read the tenant",
    });

    expect(tools.tenant_probe).toBe(tenantProbe);
    expect(receivedContext).toEqual({ tenantId: "ACME" });
  });

  it.skipIf(aiSdkMajor < 7)("keeps target model-output conversion native", async () => {
    const view = ratel().adaptTo(aiSdk());
    const rendered = tool({
      description: "Render a status.",
      inputSchema: z.object({}),
      execute: async () => ({ status: "ready" }),
      toModelOutput: async ({ output }) => ({
        type: "text",
        value: `STATUS:${output.status}`,
      }),
    });
    await view.tools.register({ render_status: rendered });
    const tools = view.modelTools();
    const model = new MockLanguageModelV2([
      {
        content: [
          {
            type: "tool-call",
            toolCallId: "render-0",
            toolName: "render_status",
            input: JSON.stringify({}),
          },
        ],
        finishReason: "tool-calls",
        usage,
        warnings: [],
      },
    ]);

    const result = await generateText({
      model: model as unknown as LanguageModel,
      tools,
      prompt: "render status",
    });
    const responseMessages = (
      result as unknown as {
        responseMessages: Array<{ role: string; content: Array<{ output?: unknown }> }>;
      }
    ).responseMessages;
    const toolMessage = responseMessages.find((message) => message.role === "tool");

    expect(tools.render_status).toBe(rendered);
    expect(toolMessage?.content[0]?.output).toEqual({
      type: "text",
      value: "STATUS:ready",
    });
  });

  it("validates and transforms nested tool input before execution", async () => {
    const view = ratel().adaptTo(aiSdk());
    let received: unknown;
    await view.tools.register({
      format_name: tool({
        description: "Normalize a display name.",
        inputSchema: normalizedNameSchema(true),
        execute: async (input) => {
          received = input;
          return input;
        },
      }),
    });
    const model = new MockLanguageModelV2([
      {
        content: [
          {
            type: "tool-call",
            toolCallId: "format-0",
            toolName: "invoke_tool",
            input: JSON.stringify({
              toolId: "format_name",
              args: { name: "  Ada  " },
            }),
          },
        ],
        finishReason: "tool-calls",
        usage,
        warnings: [],
      },
    ]);

    const result = await generateText({
      model: model as unknown as LanguageModel,
      tools: view.modelTools(),
      prompt: "format Ada",
    });

    expect(result.content.filter((part) => part.type === "tool-error")).toEqual([]);
    expect(received).toEqual({ name: "ADA", suffix: "!" });
  });

  it("validates the capability's tolerated flattened input before execution", async () => {
    const view = ratel().adaptTo(aiSdk());
    let received: unknown;
    await view.tools.register({
      format_name: tool({
        description: "Normalize a display name.",
        inputSchema: normalizedNameSchema(false),
        execute: async (input) => {
          received = input;
          return input;
        },
      }),
    });
    const model = new MockLanguageModelV2([
      {
        content: [
          {
            type: "tool-call",
            toolCallId: "format-flat-0",
            toolName: "invoke_tool",
            input: JSON.stringify({ toolId: "format_name", name: "  Ada  " }),
          },
        ],
        finishReason: "tool-calls",
        usage,
        warnings: [],
      },
    ]);

    await generateText({
      model: model as unknown as LanguageModel,
      tools: view.modelTools(),
      prompt: "format Ada",
    });

    expect(received).toEqual({ name: "ADA" });
  });

  it("validates through the registration-owning adapter across AI SDK views", async () => {
    const core = ratel();
    const registrationView = core.adaptTo(aiSdk());
    const modelView = core.adaptTo(aiSdk());
    let received: unknown;
    await registrationView.tools.register({
      format_name: tool({
        description: "Normalize a display name.",
        inputSchema: normalizedNameSchema(true),
        execute: async (input) => {
          received = input;
          return input;
        },
      }),
    });
    const model = new MockLanguageModelV2([
      {
        content: [
          {
            type: "tool-call",
            toolCallId: "format-cross-view-0",
            toolName: "invoke_tool",
            input: JSON.stringify({
              toolId: "format_name",
              args: { name: "  Ada  " },
            }),
          },
        ],
        finishReason: "tool-calls",
        usage,
        warnings: [],
      },
    ]);

    await generateText({
      model: model as unknown as LanguageModel,
      tools: modelView.modelTools(),
      prompt: "format Ada",
    });

    expect(received).toEqual({ name: "ADA", suffix: "!" });
  });

  it("uses the live validator after a native catalog hot-swap", async () => {
    const core = ratel();
    const view = core.adaptTo(aiSdk());
    await view.tools.register({
      replaceable: tool({
        description: "Old string tool.",
        inputSchema: normalizedNameSchema(false),
        execute: async () => ({ old: true }),
      }),
    });
    let received: unknown;
    await core.tools.register({
      id: "replaceable",
      name: "replaceable",
      description: "New numeric tool.",
      inputSchema: {
        type: "object",
        properties: { count: { type: "number" } },
        required: ["count"],
      },
      outputSchema: { type: "object" },
      execute: async (input) => {
        received = input;
        return input;
      },
    });
    const model = new MockLanguageModelV2([
      {
        content: [
          {
            type: "tool-call",
            toolCallId: "replace-0",
            toolName: "invoke_tool",
            input: JSON.stringify({ toolId: "replaceable", args: { count: 2 } }),
          },
        ],
        finishReason: "tool-calls",
        usage,
        warnings: [],
      },
    ]);

    await generateText({
      model: model as unknown as LanguageModel,
      tools: view.modelTools(),
      prompt: "use the replacement",
    });

    expect(received).toEqual({ count: 2 });
  });

  it("allows a target schema to transform its object input to another root type", async () => {
    const view = ratel().adaptTo(aiSdk());
    let received: unknown;
    await view.tools.register({
      name_key: tool({
        description: "Create a normalized name key.",
        inputSchema: jsonSchema<string>(
          {
            type: "object",
            properties: { name: { type: "string" } },
            required: ["name"],
          },
          {
            validate: (value) => {
              const name = (value as { name?: unknown })?.name;
              return typeof name === "string"
                ? { success: true, value: name.trim().toUpperCase() }
                : { success: false, error: new TypeError("name must be a string") };
            },
          },
        ),
        execute: async (input) => {
          received = input;
          return { key: input };
        },
      }),
    });
    const model = new MockLanguageModelV2([
      {
        content: [
          {
            type: "tool-call",
            toolCallId: "key-0",
            toolName: "invoke_tool",
            input: JSON.stringify({ toolId: "name_key", args: { name: "  Ada  " } }),
          },
        ],
        finishReason: "tool-calls",
        usage,
        warnings: [],
      },
    ]);

    await generateText({
      model: model as unknown as LanguageModel,
      tools: view.modelTools(),
      prompt: "create the key",
    });

    expect(received).toBe("ADA");
  });

  it.skipIf(aiSdkMajor < 6)(
    "preserves preliminary and final results from a cataloged streaming tool",
    async () => {
      const view = ratel().adaptTo(aiSdk());
      await view.tools.register({
        run_job: tool({
          description: "Run a background job.",
          inputSchema: z.object({}),
          execute: async function* () {
            yield { stage: "working" };
            yield { stage: "done" };
          },
        }),
      });
      const model = new MockLanguageModelV2(
        [],
        [
          streamResult(
            {
              type: "tool-call",
              toolCallId: "job-0",
              toolName: "invoke_tool",
              input: JSON.stringify({ toolId: "run_job", args: {} }),
            },
            { type: "finish", finishReason: "tool-calls", usage },
          ),
        ],
      );
      const result = streamText({
        model: model as unknown as LanguageModel,
        tools: view.modelTools(),
        prompt: "run the job",
      });
      const outputs: Array<{ output: unknown; preliminary?: boolean }> = [];

      for await (const part of result.fullStream) {
        if (part.type === "tool-result") {
          outputs.push({ output: part.output, preliminary: part.preliminary });
        }
      }

      expect(outputs).toEqual([
        { output: { stage: "working" }, preliminary: true },
        { output: { stage: "done" }, preliminary: true },
        { output: { stage: "done" }, preliminary: undefined },
      ]);
    },
  );

  it("surfaces a cataloged target exception as a tool error", async () => {
    const view = ratel().adaptTo(aiSdk());
    await view.tools.register({
      explode: tool({
        description: "Fail with a diagnostic error.",
        inputSchema: z.object({}),
        execute: async () => {
          throw new Error("target exploded");
        },
      }),
    });
    const model = new MockLanguageModelV2([
      {
        content: [
          {
            type: "tool-call",
            toolCallId: "explode-0",
            toolName: "invoke_tool",
            input: JSON.stringify({ toolId: "explode", args: {} }),
          },
        ],
        finishReason: "tool-calls",
        usage,
        warnings: [],
      },
    ]);

    const result = await generateText({
      model: model as unknown as LanguageModel,
      tools: view.modelTools(),
      prompt: "run the failing tool",
    });
    const errors = result.content.filter((part) => part.type === "tool-error");

    expect(errors).toHaveLength(1);
    expect(String(errors[0]?.error)).toContain("target exploded");
  });

  it("keeps one recall pair in both prompts of a two-step generateText loop", async () => {
    const view = ratel().adaptTo(aiSdk());
    let executions = 0;
    await view.tools.register({
      deploy_app: tool({
        description: "Deploy the app to production servers.",
        inputSchema: z.object({}),
        execute: async () => {
          executions++;
          return { deployed: true };
        },
      }),
    });
    const model = new MockLanguageModelV2([
      {
        content: [
          {
            type: "tool-call",
            toolCallId: "invoke-0",
            toolName: "invoke_tool",
            input: JSON.stringify({ toolId: "deploy_app", args: {} }),
          },
        ],
        finishReason: "tool-calls",
        usage,
        warnings: [],
      },
      {
        content: [{ type: "text", text: "deployed." }],
        finishReason: "stop",
        usage,
        warnings: [],
      },
    ]);
    const messages: ModelMessage[] = [{ role: "user", content: "deploy to production" }];

    const result = await generateText({
      model: model as unknown as LanguageModel,
      tools: view.modelTools(),
      messages,
      prepareStep: view.prepareStep,
      stopWhen: stepCountIs(2),
    });

    expect(result.text).toBe("deployed.");
    expect(executions).toBe(1);
    expect(model.doGenerateCalls).toHaveLength(2);
    expect(recallIdOccurrences(model.doGenerateCalls[0].prompt, "recall_0")).toBe(2);
    expect(recallIdOccurrences(model.doGenerateCalls[1].prompt, "recall_0")).toBe(2);
    expect(messages).toEqual([{ role: "user", content: "deploy to production" }]);

    const next: ModelMessage[] = [{ role: "user", content: "deploy again" }];
    await view.appendRecall(next);
    expect(JSON.stringify(next)).toContain("recall_1");
  });

  it("keeps one recall pair in both prompts of a two-step streamText loop", async () => {
    const view = ratel().adaptTo(aiSdk());
    let executions = 0;
    await view.tools.register({
      deploy_app: tool({
        description: "Deploy the app to production servers.",
        inputSchema: z.object({}),
        execute: async () => {
          executions++;
          return { deployed: true };
        },
      }),
    });
    const model = new MockLanguageModelV2(
      [],
      [
        streamResult(
          {
            type: "tool-call",
            toolCallId: "invoke-0",
            toolName: "invoke_tool",
            input: JSON.stringify({ toolId: "deploy_app", args: {} }),
          },
          { type: "finish", finishReason: "tool-calls", usage },
        ),
        streamResult(
          { type: "text-start", id: "text-0" },
          { type: "text-delta", id: "text-0", delta: "deployed." },
          { type: "text-end", id: "text-0" },
          { type: "finish", finishReason: "stop", usage },
        ),
      ],
    );
    const messages: ModelMessage[] = [{ role: "user", content: "deploy to production" }];

    const result = streamText({
      model: model as unknown as LanguageModel,
      tools: view.modelTools(),
      messages,
      prepareStep: view.prepareStep,
      stopWhen: stepCountIs(2),
    });

    // ai@5.0.0 (the v5 floor) resolves `result.text` only after the stream is
    // consumed; later releases auto-consume on await. Consume explicitly so the
    // same test runs at every supported release.
    await result.consumeStream();
    expect(await result.text).toBe("deployed.");
    expect(executions).toBe(1);
    expect(model.doStreamCalls).toHaveLength(2);
    expect(recallIdOccurrences(model.doStreamCalls[0].prompt, "recall_0")).toBe(2);
    expect(recallIdOccurrences(model.doStreamCalls[1].prompt, "recall_0")).toBe(2);
    expect(messages).toEqual([{ role: "user", content: "deploy to production" }]);

    const next: ModelMessage[] = [{ role: "user", content: "deploy again" }];
    await view.appendRecall(next);
    expect(JSON.stringify(next)).toContain("recall_1");
  });
});

function recallIdOccurrences(prompt: unknown[], callId: string): number {
  return JSON.stringify(prompt).split(callId).length - 1;
}

function streamResult(...chunks: unknown[]): { stream: ReadableStream<unknown> } {
  return {
    stream: new ReadableStream({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk);
        controller.close();
      },
    }),
  };
}

function normalizedNameSchema(withSuffix: boolean) {
  return jsonSchema<{ name: string; suffix?: string }>(
    {
      type: "object",
      properties: {
        name: { type: "string" },
        ...(withSuffix ? { suffix: { type: "string" } } : {}),
      },
      required: ["name"],
      additionalProperties: false,
    },
    {
      validate: (value) => {
        if (
          value === null ||
          typeof value !== "object" ||
          typeof (value as { name?: unknown }).name !== "string"
        ) {
          return { success: false, error: new TypeError("name must be a string") };
        }
        const input = value as { name: string; suffix?: unknown };
        if (input.suffix !== undefined && typeof input.suffix !== "string") {
          return { success: false, error: new TypeError("suffix must be a string") };
        }
        return {
          success: true,
          value: {
            name: input.name.trim().toUpperCase(),
            ...(withSuffix ? { suffix: input.suffix ?? "!" } : {}),
          },
        };
      },
    },
  );
}
