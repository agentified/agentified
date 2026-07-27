/**
 * `examples/telemetry-ts` — emit Ratel's `ratel.*` telemetry through the standard
 * OpenTelemetry JS SDK.
 *
 * Runnable offline: its trace-only demo wires a `ConsoleSpanExporter` so spans print
 * to stdout (no collector, no API key). The Ratel-specific part is the vocabulary from
 * `@ratel-ai/telemetry` — the constants and value enums you set as span attributes.
 * Ratel ships no bootstrap (ADR-0007): in production you own the providers and swap the
 * console exporter for the OTLP trace + Logs exporters at `RATEL_OTLP_ENDPOINT` (shown at
 * the end); everything else stays identical.
 */

import { context, type Tracer, trace } from "@opentelemetry/api";
import type { Logger } from "@opentelemetry/api-logs";
import { OTLPLogExporter } from "@opentelemetry/exporter-logs-otlp-proto";
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-proto";
import { resourceFromAttributes } from "@opentelemetry/resources";
import { BatchLogRecordProcessor, LoggerProvider } from "@opentelemetry/sdk-logs";
import {
  BatchSpanProcessor,
  ConsoleSpanExporter,
  SimpleSpanProcessor,
} from "@opentelemetry/sdk-trace-base";
import { NodeTracerProvider } from "@opentelemetry/sdk-trace-node";
import { ATTR_SERVICE_NAME } from "@opentelemetry/semantic-conventions";

import {
  contentCaptureMode,
  EXECUTE_TOOL,
  GEN_AI_OPERATION_NAME,
  GEN_AI_TOOL_NAME,
  Origin,
  RATEL_ORIGIN,
  RATEL_SEARCH,
  RATEL_SEARCH_HIT_COUNT,
  RATEL_SEARCH_QUERY,
  RATEL_SEARCH_RESULTS,
  RATEL_SEARCH_TARGET,
  RATEL_SEARCH_TOP_K,
  RATEL_TOOL_ARGS_SIZE_BYTES,
  RATEL_UPSTREAM_SERVER,
  RATEL_UPSTREAM_TRANSPORT,
  SearchTarget,
  SEMCONV_VERSION,
} from "@ratel-ai/telemetry";

/**
 * Endpoint and auth are the host's to resolve: `@ratel-ai/telemetry` is vocabulary
 * only and carries no exporter config. The names and the two helpers below are all
 * of it, and they are what you swap for your own configuration source (a secrets
 * manager, the standard `OTEL_EXPORTER_OTLP_*` vars, a config file, ...).
 */
const ENDPOINT_ENV = "RATEL_OTLP_ENDPOINT";
const API_KEY_ENV = "RATEL_API_KEY";
const SERVICE_NAME = "ratel-telemetry-example";

/** The Logs endpoint is the traces URL's sibling: `/v1/traces` -> `/v1/logs`. */
function deriveLogsUrl(tracesUrl: string): string {
  return tracesUrl.replace(/\/v1\/traces(?=\/?(?:[?#]|$))/, "/v1/logs");
}

/** Bearer auth when a key is configured; no header at all when it is not. */
function authHeaders(apiKey: string | undefined): Record<string, string> {
  return apiKey ? { Authorization: `Bearer ${apiKey}` } : {};
}

/**
 * Emit one realistic Ratel trace: a `ratel.search` (capability search) span
 * followed by an `execute_tool` span enriched with the `ratel.*` overlay, both
 * hanging under a root span so they share ONE trace. This is the pattern you copy
 * into your own agent — only the constants come from Ratel; the tracer is the
 * stock OTel SDK. Pass a `logger` to also emit the content-bearing EventRecord half.
 */
function emitRatelTrace(tracer: Tracer, logger?: Logger): void {
  // A root span standing in for the caller's own span — the agent turn (in a real
  // app, the LLM `chat` gen_ai span or the inbound request). Ratel's search + tool
  // spans hang under it, so the `ratel.*` overlay and the `gen_ai.*` call land in
  // ONE trace, told apart by namespace and joined on trace/span id (CONVENTIONS.md).
  const root = tracer.startSpan("agent turn");
  // Parent the two spans on `root` by threading its context explicitly, rather than
  // relying on the active context (which needs a registered ContextManager the
  // offline demo skips).
  const parentCtx = trace.setSpan(context.active(), root);

  // 1. Capability search — the agent asks Ratel which tools fit the prompt.
  const search = tracer.startSpan(
    RATEL_SEARCH,
    {
      attributes: {
        [RATEL_ORIGIN]: Origin.Agent, // synthesized inside the agent loop
        [RATEL_SEARCH_TARGET]: SearchTarget.Tool,
        [RATEL_SEARCH_TOP_K]: 5,
        [RATEL_SEARCH_HIT_COUNT]: 2,
      },
    },
    parentCtx,
  );
  // The second stream: the query text stays off the span and rides a
  // `ratel.search.results` EventRecord instead, parented on the search span so both
  // streams join on trace/span id. A real emitter gates this on `contentCaptureMode()`.
  logger?.emit({
    eventName: RATEL_SEARCH_RESULTS,
    attributes: { [RATEL_SEARCH_QUERY]: "email the team the release notes" },
    context: trace.setSpan(parentCtx, search),
  });
  search.end();

  // 2. Tool invocation — a standard gen_ai `execute_tool` span (so any OTel
  //    backend understands it) enriched with `ratel.*` attributes.
  const invoke = tracer.startSpan(
    EXECUTE_TOOL,
    {
      attributes: {
        [GEN_AI_OPERATION_NAME]: EXECUTE_TOOL,
        [GEN_AI_TOOL_NAME]: "send_email",
        [RATEL_ORIGIN]: Origin.Agent,
        [RATEL_TOOL_ARGS_SIZE_BYTES]: 128,
        [RATEL_UPSTREAM_SERVER]: "gmail",
        [RATEL_UPSTREAM_TRANSPORT]: "stdio",
      },
    },
    parentCtx,
  );
  invoke.end();

  root.end();
}

async function main(): Promise<void> {
  console.log(`@ratel-ai/telemetry — semconv pin ${SEMCONV_VERSION}`);
  console.log(`content capture: ${contentCaptureMode()} (gated by OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT)\n`);

  // --- The runnable demo: emit spans to the console (no network) ---
  const provider = new NodeTracerProvider({
    resource: resourceFromAttributes({ [ATTR_SERVICE_NAME]: SERVICE_NAME }),
    spanProcessors: [new SimpleSpanProcessor(new ConsoleSpanExporter())],
  });
  const tracer = provider.getTracer("@ratel-ai/example-telemetry");

  console.log("--- emitting a ratel.search + execute_tool trace ---");
  emitRatelTrace(tracer);
  await provider.forceFlush();
  await provider.shutdown();

  // --- Production wiring: traces + EventRecords exported to Ratel over OTLP ---
  // Endpoint and auth are plain host configuration; shown here with demo values so the
  // derivation is visible without sending anything:
  const demoUrl = "https://cloud.ratel.sh/v1/traces";
  const demoHeaders = authHeaders("sk-demo");
  console.log("\n--- exporter config the host resolves (illustrative demo values) ---");
  console.log(`  url:         ${demoUrl}`);
  console.log(`  logsUrl:     ${deriveLogsUrl(demoUrl)}`);
  console.log(`  serviceName: ${SERVICE_NAME}`);
  console.log(`  headers:     ${Object.keys(demoHeaders).join(", ") || "(none)"}`);

  // The host owns the providers. Same construction as the offline demo, with OTLP batch
  // processors in place of the console one, plus a LoggerProvider for the EventRecord
  // stream. Both are threaded into `emitRatelTrace` rather than registered globally.
  const endpoint = process.env[ENDPOINT_ENV];
  if (endpoint) {
    const logsUrl = deriveLogsUrl(endpoint);
    const headers = authHeaders(process.env[API_KEY_ENV]);
    const resource = resourceFromAttributes({ [ATTR_SERVICE_NAME]: SERVICE_NAME });
    const tracerProvider = new NodeTracerProvider({
      resource,
      spanProcessors: [new BatchSpanProcessor(new OTLPTraceExporter({ url: endpoint, headers }))],
    });
    const loggerProvider = new LoggerProvider({
      resource,
      processors: [
        new BatchLogRecordProcessor({ exporter: new OTLPLogExporter({ url: logsUrl, headers }) }),
      ],
    });

    console.log(`\n--- ${ENDPOINT_ENV} set — exporting a real trace to ${endpoint} ---`);
    console.log(`--- and a ratel.search.results EventRecord to ${logsUrl} ---`);
    emitRatelTrace(
      tracerProvider.getTracer("@ratel-ai/example-telemetry"),
      loggerProvider.getLogger("@ratel-ai/example-telemetry"),
    );
    // Shutting the providers down flushes both batch processors. An unreachable collector
    // makes that reject, so report it rather than failing the whole example.
    for (const owned of [tracerProvider, loggerProvider]) {
      await owned.shutdown().catch((err) => console.error(`  export failed: ${err}`));
    }
  } else {
    console.log(
      `\n(set ${ENDPOINT_ENV} — and optionally ${API_KEY_ENV} — to export a real trace over OTLP)`,
    );
  }

  console.log("\nOK");
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
