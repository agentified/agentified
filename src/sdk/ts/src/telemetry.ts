/**
 * OpenTelemetry emission for the SDK's `ratel.*` / `gen_ai.*` funnel (ADR-0007).
 * The catalog, capability-tool, skill, and MCP paths call these helpers to
 * open a span around each operation and, when selected by content capture, emit a
 * structured Logs EventRecord. Names and attribute keys come from the OTel-free
 * `@ratel-ai/telemetry` vocabulary.
 *
 * Emission is **transparent**: records go to whichever OpenTelemetry tracer and logger
 * providers are registered globally. The SDK never registers one — that is the host's
 * job (`new NodeSDK({ spanProcessors })`). Until the host wires providers, every span is
 * a no-op `NonRecordingSpan`, so instrumentation is effectively free and the local trace
 * stream (`recordEvent`) is untouched. This mirrors how the Vercel AI SDK instruments:
 * the library emits; the app decides where it goes.
 *
 * Message/tool content (`ratel.search.query`, tool args/result) follows the ecosystem
 * capture gate's span-attribute and Logs EventRecord channels (default off), per ADR-0007.
 */

import {
  context,
  type Context as OtelContext,
  type Span,
  SpanKind,
  SpanStatusCode,
  trace,
} from "@opentelemetry/api";
import { type AnyValue, logs } from "@opentelemetry/api-logs";
import {
  AuthOutcome,
  ContentCapture,
  clearContentCapture,
  contentCaptureMode,
  EXECUTE_TOOL,
  GEN_AI_OPERATION_NAME,
  GEN_AI_TOOL_CALL_ARGUMENTS,
  GEN_AI_TOOL_CALL_RESULT,
  GEN_AI_TOOL_NAME,
  RATEL_AUTH_FLOW,
  RATEL_AUTH_OUTCOME,
  RATEL_ORIGIN,
  RATEL_SEARCH,
  RATEL_SEARCH_HIT_COUNT,
  RATEL_SEARCH_QUERY,
  RATEL_SEARCH_RESULTS,
  RATEL_SEARCH_TARGET,
  RATEL_SEARCH_TOP_K,
  RATEL_SKILL_ID,
  RATEL_SKILL_LOAD,
  RATEL_TOOL_ARGS_SIZE_BYTES,
  RATEL_TOOL_EXECUTION_DETAILS,
  RATEL_UPSTREAM_REGISTER,
  RATEL_UPSTREAM_SERVER,
  RATEL_UPSTREAM_TOOL_COUNT,
  RATEL_UPSTREAM_TRANSPORT,
  type SearchTarget,
  setContentCapture,
} from "@ratel-ai/telemetry";
import { isAsyncIterable, isPromiseLike } from "./async.js";
import type { SearchOrigin } from "./catalog.js";

const TRACER_NAME = "@ratel-ai/sdk";
const LOGGER_NAME = "@ratel-ai/sdk";

function getTracer() {
  return trace.getTracer(TRACER_NAME);
}

function getLogger() {
  return logs.getLogger(LOGGER_NAME);
}

/** Content rides span attributes only when the capture gate selects a span mode. */
function captureContentOnSpan(): boolean {
  const mode = contentCaptureMode();
  return mode === ContentCapture.SpanOnly || mode === ContentCapture.SpanAndEvent;
}

/** Content rides Logs EventRecords when the capture gate selects an event mode. */
function captureContentOnEvent(): boolean {
  const mode = contentCaptureMode();
  return mode === ContentCapture.EventOnly || mode === ContentCapture.SpanAndEvent;
}

/**
 * Emit the Opt-In tool execution EventRecord with structured arguments and,
 * on success, a structured result.
 */
function addToolContentEvent(
  toolId: string,
  args: unknown,
  eventContext: OtelContext,
  result?: { value: unknown },
): void {
  const attributes = {
    [GEN_AI_OPERATION_NAME]: EXECUTE_TOOL,
    [GEN_AI_TOOL_NAME]: toolId,
    [GEN_AI_TOOL_CALL_ARGUMENTS]: toLogValue(args),
    ...(result ? { [GEN_AI_TOOL_CALL_RESULT]: toLogValue(result.value) } : {}),
  };
  getLogger().emit({
    eventName: RATEL_TOOL_EXECUTION_DETAILS,
    attributes,
    context: eventContext,
  });
}

/**
 * Emit the Opt-In `ratel.search.results` EventRecord carrying the search text.
 * Hit ids/scores/BM25 timing are local-stream only.
 */
function addSearchResultsEvent(query: string, eventContext: OtelContext): void {
  getLogger().emit({
    eventName: RATEL_SEARCH_RESULTS,
    attributes: { [RATEL_SEARCH_QUERY]: query },
    context: eventContext,
  });
}

/** UTF-8 byte size of the JSON-encoded args (0 if not encodable). */
export function argsSizeBytes(args: unknown): number {
  try {
    return new TextEncoder().encode(JSON.stringify(args) ?? "").length;
  } catch {
    return 0;
  }
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value) ?? "";
  } catch {
    return "";
  }
}

function toLogValue(value: unknown): AnyValue {
  const encoded = safeJson(value);
  if (encoded === "") return null;
  return JSON.parse(encoded) as AnyValue;
}

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

/**
 * The upstream MCP server backing a tool, derived from the `<server>__<tool>`
 * id convention. `undefined` for a plain (non-proxied) tool id.
 */
export function upstreamFromToolId(toolId: string): string | undefined {
  const idx = toolId.indexOf("__");
  if (idx <= 0) return undefined;
  return toolId.slice(0, idx);
}

/** Close a span in the failure path: record the exception + ERROR status. */
function fail(span: Span, err: unknown): void {
  if (err instanceof Error) span.recordException(err);
  span.setStatus({ code: SpanStatusCode.ERROR, message: errorMessage(err) });
}

/**
 * Wrap a tool invocation in a standard `execute_tool` span (`gen_ai.operation.name
 * = execute_tool`, enriched with `ratel.*`). Deliberately the OTel gen_ai tool
 * operation, not a bespoke span, so a generic backend understands it
 * (ADR-0007). Preserves the executor's immediate return shape and keeps the
 * span open through `AsyncIterable` completion, cancellation, or failure.
 */
export function traceExecuteTool<T>(toolId: string, args: unknown, run: () => T): T {
  return getTracer().startActiveSpan(
    `${EXECUTE_TOOL} ${toolId}`,
    { kind: SpanKind.INTERNAL },
    (span) => {
      const activeContext = trace.setSpan(context.active(), span);
      span.setAttribute(GEN_AI_OPERATION_NAME, EXECUTE_TOOL);
      span.setAttribute(GEN_AI_TOOL_NAME, toolId);
      const upstream = upstreamFromToolId(toolId);
      if (upstream) span.setAttribute(RATEL_UPSTREAM_SERVER, upstream);
      span.setAttribute(RATEL_TOOL_ARGS_SIZE_BYTES, argsSizeBytes(args));
      if (captureContentOnSpan()) span.setAttribute(GEN_AI_TOOL_CALL_ARGUMENTS, safeJson(args));

      const succeed = (result: unknown): void => {
        if (captureContentOnSpan()) span.setAttribute(GEN_AI_TOOL_CALL_RESULT, safeJson(result));
        if (captureContentOnEvent()) {
          addToolContentEvent(toolId, args, activeContext, { value: result });
        }
        span.setStatus({ code: SpanStatusCode.OK });
        span.end();
      };
      const reject = (err: unknown): void => {
        if (captureContentOnEvent()) addToolContentEvent(toolId, args, activeContext);
        fail(span, err);
        span.end();
      };

      try {
        return observeExecutionResult(run(), succeed, reject, activeContext) as T;
      } catch (err) {
        reject(err);
        throw err;
      }
    },
  );
}

function observeExecutionResult(
  result: unknown,
  onSuccess: (result: unknown) => void,
  onError: (error: unknown) => void,
  activeContext: OtelContext,
): unknown {
  if (isAsyncIterable(result)) {
    return observeAsyncIterable(result, onSuccess, onError, activeContext);
  }
  if (isPromiseLike(result)) {
    return Promise.resolve(result).then(
      (value) => {
        onSuccess(value);
        return value;
      },
      (error) => {
        onError(error);
        throw error;
      },
    );
  }
  onSuccess(result);
  return result;
}

async function* observeAsyncIterable(
  iterable: AsyncIterable<unknown>,
  onSuccess: (result: unknown) => void,
  onError: (error: unknown) => void,
  activeContext: OtelContext,
): AsyncGenerator<unknown> {
  const iterator = iterable[Symbol.asyncIterator]();
  let completed = false;
  let failed = false;
  let lastValue: unknown;
  try {
    while (true) {
      const next = await context.with(activeContext, () => iterator.next());
      if (next.done) {
        completed = true;
        break;
      }
      lastValue = next.value;
      yield next.value;
    }
  } catch (error) {
    failed = true;
    onError(error);
    throw error;
  } finally {
    if (!completed && !failed && iterator.return) {
      await closeAsyncIterator(iterator, activeContext, (error) => {
        failed = true;
        onError(error);
      });
    }
    if (!failed) onSuccess(lastValue);
  }
}

async function closeAsyncIterator(
  iterator: AsyncIterator<unknown>,
  activeContext: OtelContext,
  onError: (error: unknown) => void,
): Promise<void> {
  try {
    await context.with(activeContext, () => iterator.return?.());
  } catch (error) {
    onError(error);
    throw error;
  }
}

/**
 * Wrap a capability search (tool or skill) in a `ratel.search` span. Synchronous:
 * the native BM25 search returns inline. `run` returns the hit array; its length
 * becomes `ratel.search.hit_count`.
 */
export function traceSearch<T extends { length: number }>(
  target: SearchTarget,
  query: string,
  topK: number,
  origin: SearchOrigin,
  run: () => T,
): T {
  return getTracer().startActiveSpan(RATEL_SEARCH, { kind: SpanKind.INTERNAL }, (span) => {
    const eventContext = trace.setSpan(context.active(), span);
    span.setAttribute(RATEL_SEARCH_TARGET, target);
    span.setAttribute(RATEL_SEARCH_TOP_K, topK);
    span.setAttribute(RATEL_ORIGIN, origin);
    if (captureContentOnSpan()) span.setAttribute(RATEL_SEARCH_QUERY, query);
    try {
      const hits = run();
      span.setAttribute(RATEL_SEARCH_HIT_COUNT, hits.length);
      if (captureContentOnEvent()) addSearchResultsEvent(query, eventContext);
      span.setStatus({ code: SpanStatusCode.OK });
      return hits;
    } catch (err) {
      fail(span, err);
      throw err;
    } finally {
      span.end();
    }
  });
}

/** Wrap an asynchronous capability search in a `ratel.search` span. */
export function traceSearchAsync<T extends { length: number }>(
  target: SearchTarget,
  query: string,
  topK: number,
  origin: SearchOrigin,
  run: () => Promise<T>,
): Promise<T> {
  return getTracer().startActiveSpan(RATEL_SEARCH, { kind: SpanKind.INTERNAL }, async (span) => {
    const eventContext = trace.setSpan(context.active(), span);
    span.setAttribute(RATEL_SEARCH_TARGET, target);
    span.setAttribute(RATEL_SEARCH_TOP_K, topK);
    span.setAttribute(RATEL_ORIGIN, origin);
    if (captureContentOnSpan()) span.setAttribute(RATEL_SEARCH_QUERY, query);
    try {
      const hits = await run();
      span.setAttribute(RATEL_SEARCH_HIT_COUNT, hits.length);
      if (captureContentOnEvent()) addSearchResultsEvent(query, eventContext);
      span.setStatus({ code: SpanStatusCode.OK });
      return hits;
    } catch (err) {
      fail(span, err);
      throw err;
    } finally {
      span.end();
    }
  });
}

/** Wrap a skill-content load in a `ratel.skill.load` span. */
export function traceSkillLoad<T>(skillId: string, run: () => T): T {
  return getTracer().startActiveSpan(RATEL_SKILL_LOAD, { kind: SpanKind.INTERNAL }, (span) => {
    span.setAttribute(RATEL_SKILL_ID, skillId);
    try {
      const body = run();
      span.setStatus({ code: SpanStatusCode.OK });
      return body;
    } catch (err) {
      fail(span, err);
      throw err;
    } finally {
      span.end();
    }
  });
}

/**
 * Wrap an upstream-MCP registration in a `ratel.upstream.register` span. `run`
 * receives a `reportToolCount` callback to set `ratel.upstream.tool_count` once
 * the tool list is known.
 */
export function traceUpstreamRegister<T>(
  server: string,
  transport: string,
  run: (reportToolCount: (n: number) => void) => Promise<T>,
): Promise<T> {
  return getTracer().startActiveSpan(
    RATEL_UPSTREAM_REGISTER,
    { kind: SpanKind.INTERNAL },
    async (span) => {
      span.setAttribute(RATEL_UPSTREAM_SERVER, server);
      span.setAttribute(RATEL_UPSTREAM_TRANSPORT, transport);
      try {
        const result = await run((n) => span.setAttribute(RATEL_UPSTREAM_TOOL_COUNT, n));
        span.setStatus({ code: SpanStatusCode.OK });
        return result;
      } catch (err) {
        fail(span, err);
        throw err;
      } finally {
        span.end();
      }
    },
  );
}

/**
 * Mark an upstream tool call that failed with a 401 / needs-reauthorization: a
 * short `ratel.auth.flow` span carrying `ratel.auth.outcome = needs_auth`.
 */
export function recordAuthNeeded(server?: string): void {
  const span = getTracer().startSpan(RATEL_AUTH_FLOW, { kind: SpanKind.INTERNAL });
  if (server) span.setAttribute(RATEL_UPSTREAM_SERVER, server);
  span.setAttribute(RATEL_AUTH_OUTCOME, AuthOutcome.NeedsAuth);
  span.end();
}

// Re-exported so hosts configuring capture don't need a second import from
// @ratel-ai/telemetry.
export { ContentCapture, clearContentCapture, setContentCapture };
