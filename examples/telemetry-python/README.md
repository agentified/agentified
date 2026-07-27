# `examples/telemetry-python` — emit `ratel.*` telemetry with OpenTelemetry (Python)

The Python mirror of [`examples/telemetry-ts`](../telemetry-ts/README.md): how to emit Ratel's telemetry vocabulary through the standard [OpenTelemetry Python SDK](https://opentelemetry.io/docs/languages/python/) using [`ratel-ai-telemetry`](../../src/telemetry/python/README.md). The package supplies the `ratel.*` vocabulary plus optional standard OTLP trace and Logs wiring.

The mirror is not exact, by design: in TS the host builds and owns the OTel providers, while Python keeps `init()` wiring both exporters.

The trace-only offline demo emits one realistic trace — a `ratel.search` span followed by an `execute_tool` span under a root agent-turn span — and prints it with a `ConsoleSpanExporter`. Production `init()` additionally exports content-bearing Logs EventRecords.

## Setup

```bash
uv run main.py
```

`uv run` resolves `ratel-ai-telemetry` from this monorepo (see `[tool.uv.sources]` in `pyproject.toml`) and the OTel SDK from PyPI, then runs `main.py`. It prints the trace (the agent-turn root plus the two Ratel spans, with their `ratel.*` / `gen_ai.*` attributes) and shows how `init()` resolves its endpoint + auth.

To export a real trace instead of printing, set the endpoint and run again:

```bash
export RATEL_OTLP_ENDPOINT=https://cloud.ratel.sh/v1/traces
export RATEL_API_KEY=sk-...          # optional; sent as Authorization: Bearer
uv run main.py
```

## What it illustrates

- **The vocabulary is just constants.** `RATEL_SEARCH`, `EXECUTE_TOOL`, `RATEL_ORIGIN`, `GEN_AI_TOOL_NAME`, … are imported from `ratel_ai_telemetry` and set as attributes on stock OTel spans. `Origin` / `SearchTarget` are `str`-enums, so each member equals its exact wire string (`Origin.AGENT.value == "agent"`, and `Origin.AGENT` is itself usable as an attribute value).
- **Tool calls are standard `gen_ai` spans.** The invocation is an `execute_tool` span (so any OTel backend understands it), enriched with `ratel.*` attributes — not a bespoke Ratel span.
- **`init()` is optional sugar.** `resolve_otlp_config()` resolves trace and Logs URLs plus auth; `init()` wires both OTLP exporters and returns one shutdown handle. The example calls it once with `enabled=bool(os.environ.get(OTLP_ENDPOINT_ENV))`, so the disabled path needs no env gate or error handling. The endpoint comes from `RATEL_OTLP_ENDPOINT`; the superseded `RATEL_URL` still resolves and warns. A host already running OTel adds both `ratel_span_processor()` and `ratel_log_record_processor()`.
- **Content capture is gated.** `content_capture_mode()` reads `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` (default `NO_CONTENT`).

## Layout

```
main.py    emit_ratel_trace() — builds the trace (root + the two Ratel spans) from the constants; main() wires the exporter and prints
```
