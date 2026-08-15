import { deepStrictEqual, ok, strictEqual } from "node:assert/strict";
import type { JSONSchema7 } from "../index.js";
import { type RatelAdapter, ToolCatalog } from "../index.js";
import type { AdapterConformanceOptions, ConformanceToolSpec, RecallExpectation } from "./cases.js";

/**
 * The tool shape of the {@link referenceAdapter | reference adapter} — a
 * miniature stand-in for a real framework's tool type. Its `execute` is
 * optional: a tool without one is provider-executed and
 * {@link RatelAdapter.ingest | ingests} as a passthrough, exactly as a
 * client-run framework tool would.
 */
export interface FakeTool {
  /** What retrieval ranks on and the exposed codec preserves. */
  description: string;
  /** Optional description-only retrieval override. */
  searchableDescription?: string;
  /** The catalog's native JSON-Schema spelling — no cast needed. */
  inputSchema: JSONSchema7;
  /** Runs the tool; absent marks it provider-executed (a passthrough). */
  execute?: (input: unknown) => unknown;
}

/**
 * The message shape of the {@link referenceAdapter | reference adapter}: a
 * fully observable synthetic recall pair. `role` distinguishes the call from
 * its result, `callId` is the core's minted id, and `body` carries the query
 * (on the call) or the recall result (on the result) — so a conformance
 * validator can read every part back without a real framework.
 */
export interface FakeMessage {
  /** `"call"` for the synthetic tool call, `"result"` for its response. */
  role: "call" | "result";
  /** The recall call id both messages share. */
  callId: string;
  /** `{ query }` on the call message; the recall result on the result message. */
  body: unknown;
}

/**
 * The framework-idiomatic helpers the reference adapter merges onto the adapted
 * view via {@link RatelAdapter.extend} — one field, proving the `TExt` generic
 * flows through.
 */
export interface FakeExt {
  /** Observable marker that `extend` ran against the real base surface. */
  label: string;
}

/**
 * A tiny in-repo {@link RatelAdapter} standing in for a real framework package.
 * Its tool type ({@link FakeTool}) carries its own shape and its message type
 * ({@link FakeMessage}) is fully observable, so tests — and the conformance
 * battery — can assert the core drove `ingest`/`expose`/`recallMessages`
 * correctly without pulling in any framework. Ships as the living example the
 * conformance meta-tests keep honest.
 *
 * `FakeTool` has no framework execution context, so this smallest reference
 * keeps one-argument executors. A context-aware adapter additionally wraps its
 * live context under a private stable tag in `expose` and unwraps only that tag
 * in the `CatalogRegistration.execute` returned by `ingest`.
 *
 * @returns A reference adapter over the {@link FakeTool}/{@link FakeMessage} shapes.
 */
export function referenceAdapter(): RatelAdapter<FakeTool, FakeMessage, FakeExt> {
  return {
    name: "reference",
    ingest(_id, tool) {
      if (!tool.execute) return "passthrough";
      return {
        description: tool.description,
        searchableDescription: tool.searchableDescription,
        inputSchema: tool.inputSchema,
        execute: tool.execute,
      };
    },
    expose(tool) {
      return {
        description: tool.description,
        inputSchema: tool.inputSchema as JSONSchema7,
        execute: (input) => tool.execute?.(input),
      };
    },
    recallMessages(ref, recall) {
      return [
        { role: "call", callId: ref.callId, body: { query: ref.query } },
        { role: "result", callId: ref.callId, body: recall },
      ];
    },
    extend(base) {
      return { label: `adapted:${base.tools.catalog instanceof ToolCatalog}` };
    },
  };
}

/**
 * The {@link AdapterConformanceOptions} for the {@link referenceAdapter}: the
 * hooks that teach the battery how to build {@link FakeTool}s, call exposed
 * ones, and validate the recall pair. Both the SDK's own conformance test and
 * this package's meta-tests run the battery through it, so it is the worked
 * example a real adapter's options copy.
 *
 * @returns Conformance options wired to the reference adapter.
 */
export function referenceConformanceOptions(): AdapterConformanceOptions<FakeTool, FakeMessage> {
  return {
    adapter: referenceAdapter,
    makeExecutableTool,
    makePassthroughTool,
    callExposedTool: (tool, args) => tool.execute?.(args),
    validateRecallPair,
    validateExposedTool,
  };
}

/**
 * Build a reference {@link FakeTool} that {@link RatelAdapter.ingest | ingests}
 * as an executable — its `execute` returns `spec.result` (default
 * `{ ok: true }`), so a conformance case can prove the catalog ran the
 * framework executor by observing that value come back.
 *
 * @param spec - The description to rank on and the result to observe.
 * @returns A framework tool with an executor.
 */
export function makeExecutableTool(spec: ConformanceToolSpec): FakeTool {
  const result = spec.result ?? { ok: true };
  return {
    description: spec.description,
    inputSchema: { type: "object" },
    execute: () => result,
  };
}

/**
 * Build a reference {@link FakeTool} that {@link RatelAdapter.ingest | ingests}
 * as a passthrough — no `execute`, so the adapter must keep it eagerly exposed
 * and out of the catalog.
 *
 * @param spec - The description to rank on (the result is unused for a passthrough).
 * @returns A framework tool without an executor.
 */
export function makePassthroughTool(spec: ConformanceToolSpec): FakeTool {
  return {
    description: spec.description,
    inputSchema: { type: "object" },
  };
}

/** Assert the reference recall pair encodes the testkit's {@link RecallExpectation}. */
function validateRecallPair(messages: FakeMessage[], expected: RecallExpectation): void {
  strictEqual(messages.length, 2, "recall pair must have exactly two messages");
  const [call, result] = messages;
  strictEqual(call.role, "call", "first message is the call");
  strictEqual(result.role, "result", "second message is the result");
  strictEqual(call.callId, expected.callId, "call message carries the expected call id");
  strictEqual(result.callId, expected.callId, "result message shares the call id");
  deepStrictEqual(call.body, { query: expected.query }, "call message carries the query");
  deepStrictEqual(result.body, expected.recall, "result message carries the canonical recall");
}

/**
 * Assert an exposed reference tool went through the `expose` codec: a codec
 * output has description/inputSchema/execute but — unlike a raw
 * `ExecutableTool` — no `id`/`outputSchema`.
 */
function validateExposedTool(tool: FakeTool): void {
  ok(typeof tool.description === "string", "exposed tool keeps a description");
  ok(typeof tool.execute === "function", "exposed tool is callable");
  ok(!("id" in tool), "exposed tool is framework-shaped, not a raw ExecutableTool (has id)");
  ok(!("outputSchema" in tool), "exposed tool is framework-shaped (has outputSchema)");
}
