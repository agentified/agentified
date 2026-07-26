"""examples/telemetry-python — emit Ratel's `ratel.*` telemetry through the standard
OpenTelemetry Python SDK. The Python mirror of `examples/telemetry-ts/src/index.ts`.

Runnable offline: its trace-only demo wires a ConsoleSpanExporter so spans print to
stdout (no collector, no API key). The Ratel-specific part is the vocabulary from
`ratel_ai_telemetry` — the constants and value enums you set as span attributes. In
production you swap the console exporter for `init()` (shown at the end), which wires
OTLP trace and Logs exporters to RATEL_OTLP_ENDPOINT; everything else stays identical.

    uv run main.py                          # print the spans (offline)
    RATEL_OTLP_ENDPOINT=... uv run main.py  # export a real trace via init()
"""

from __future__ import annotations

import os

from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Tracer, set_span_in_context
from ratel_ai_telemetry import (
    CAPTURE_CONTENT_ENV,
    EXECUTE_TOOL,
    GEN_AI_OPERATION_NAME,
    GEN_AI_TOOL_NAME,
    RATEL_ORIGIN,
    RATEL_SEARCH,
    RATEL_SEARCH_HIT_COUNT,
    RATEL_SEARCH_TARGET,
    RATEL_SEARCH_TOP_K,
    RATEL_TOOL_ARGS_SIZE_BYTES,
    RATEL_UPSTREAM_SERVER,
    RATEL_UPSTREAM_TRANSPORT,
    SEMCONV_VERSION,
    Origin,
    SearchTarget,
)

# init() + the OTLP config live in the .otlp submodule (the [otlp] extra) so the
# vocabulary above stays OTel-free.
from ratel_ai_telemetry.otlp import (
    DEFAULT_SERVICE_NAME,
    OTLP_ENDPOINT_ENV,
    content_capture_mode,
    init,
    resolve_otlp_config,
)


def emit_ratel_trace(tracer: Tracer) -> None:
    """Emit one realistic Ratel trace: a ratel.search (capability search) span
    followed by an execute_tool span enriched with the ratel.* overlay, both hanging
    under a root span so they share ONE trace. This is the pattern you copy into your
    own agent — only the constants come from Ratel; the tracer is the stock OTel SDK.
    """
    # A root span standing in for the caller's own span — the agent turn (in a real
    # app, the LLM `chat` gen_ai span or the inbound request). Ratel's search + tool
    # spans hang under it, so the ratel.* overlay and the gen_ai.* call land in ONE
    # trace, told apart by namespace and joined on trace/span id (CONVENTIONS.md).
    root = tracer.start_span("agent turn")
    # Parent the two spans on root by threading its context explicitly.
    parent_ctx = set_span_in_context(root)

    # 1. Capability search — the agent asks Ratel which tools fit the prompt.
    #    Origin / SearchTarget are str-enums, so .value is the exact wire string.
    search = tracer.start_span(
        RATEL_SEARCH,
        context=parent_ctx,
        attributes={
            RATEL_ORIGIN: Origin.AGENT.value,  # synthesized inside the agent loop
            RATEL_SEARCH_TARGET: SearchTarget.TOOL.value,
            RATEL_SEARCH_TOP_K: 5,
            RATEL_SEARCH_HIT_COUNT: 2,
        },
    )
    search.end()

    # 2. Tool invocation — a standard gen_ai execute_tool span (so any OTel backend
    #    understands it) enriched with ratel.* attributes.
    invoke = tracer.start_span(
        EXECUTE_TOOL,
        context=parent_ctx,
        attributes={
            GEN_AI_OPERATION_NAME: EXECUTE_TOOL,
            GEN_AI_TOOL_NAME: "send_email",
            RATEL_ORIGIN: Origin.AGENT.value,
            RATEL_TOOL_ARGS_SIZE_BYTES: 128,
            RATEL_UPSTREAM_SERVER: "gmail",
            RATEL_UPSTREAM_TRANSPORT: "stdio",
        },
    )
    invoke.end()

    root.end()


def main() -> None:
    print(f"ratel-ai-telemetry — semconv pin {SEMCONV_VERSION}")
    print(f"content capture: {content_capture_mode().value} (gated by {CAPTURE_CONTENT_ENV})\n")

    # --- The runnable demo: emit spans to the console (no network) ---
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "ratel-telemetry-example"}))
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    tracer = provider.get_tracer("ratel-example-telemetry")

    print("--- emitting a ratel.search + execute_tool trace ---")
    emit_ratel_trace(tracer)
    provider.force_flush()
    provider.shutdown()

    # --- Production wiring: traces + EventRecords exported to Ratel via init() ---
    # resolve_otlp_config is pure (no network), so we can show how endpoint + auth
    # resolve without sending anything:
    cfg = resolve_otlp_config(api_key="sk-demo", endpoint="https://cloud.ratel.sh/v1/traces")
    print("\n--- how init() resolves options -> exporter config (illustrative demo values) ---")
    print(f"  url:          {cfg.url}")
    print(f"  logs_url:     {cfg.logs_url}")
    print(f"  service_name: {cfg.service_name} (default {DEFAULT_SERVICE_NAME})")
    print(f"  headers:      {', '.join(cfg.headers) or '(none)'}")

    # One startup call for both on/off paths. When disabled, init() needs no endpoint or
    # OTel setup; when enabled it also reads RATEL_API_KEY from the environment. init() returns
    # a shutdown handle — emit through the global OTel API, not off the handle.
    endpoint = os.environ.get(OTLP_ENDPOINT_ENV)
    telemetry = init(enabled=bool(endpoint))
    if endpoint:
        print(f"\n--- {OTLP_ENDPOINT_ENV} set — exporting a real trace to {endpoint} ---")
        from opentelemetry import trace

        emit_ratel_trace(trace.get_tracer("ratel-example-telemetry"))
    else:
        print(
            f"\n(set {OTLP_ENDPOINT_ENV} — and optionally RATEL_API_KEY — "
            "to export a real trace via init())"
        )
    telemetry.shutdown()

    print("\nOK")


if __name__ == "__main__":
    main()
