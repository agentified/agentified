# `ratel-ai-telemetry` (Python)

The `ratel.*` telemetry vocabulary for Python: the constants that codify the Tier 2 overlay
of [`CONVENTIONS.md`](https://github.com/ratel-ai/ratel/blob/main/src/telemetry/CONVENTIONS.md) (attribute keys, span/EventRecord names, the
`Origin`/`SearchTarget`/`AuthOutcome` value enums, the pinned semconv version). **Importing
the constants pulls no OpenTelemetry SDK** — the vocabulary stays weight-free for the SDK
(emit side), the cloud (read side), and edge/serverless emitters
([ADR-0007](https://github.com/ratel-ai/ratel/blob/main/docs/adr/0007-telemetry-two-streams.md)). `init()` — turnkey OTLP
exporter sugar over the standard OTel Python SDK — lives in the `ratel_ai_telemetry.otlp`
submodule behind the optional `[otlp]` extra.

## Usage

```python
from opentelemetry import trace
from ratel_ai_telemetry import EXECUTE_TOOL, GEN_AI_OPERATION_NAME, GEN_AI_TOOL_NAME, RATEL_ORIGIN, Origin

# Emit a standard gen_ai `execute_tool` span enriched with the ratel.* overlay,
# on your own OTel provider — the constants alone, no extra needed.
span = trace.get_tracer("my-agent").start_span(
    EXECUTE_TOOL,
    attributes={
        GEN_AI_OPERATION_NAME: EXECUTE_TOOL,
        GEN_AI_TOOL_NAME: "send_email",
        RATEL_ORIGIN: Origin.AGENT.value,
    },
)
span.end()
```

Want turnkey OTLP export to Ratel? Install `ratel-ai-telemetry[otlp]` and call `init()`:

```python
from ratel_ai_telemetry.otlp import init  # also importable as `from ratel_ai_telemetry import init`

handle = init()  # reads RATEL_OTLP_ENDPOINT + RATEL_API_KEY (or pass endpoint=, api_key=, headers=)
# ... emit spans and EventRecords through the global OTel APIs ...
handle.shutdown()  # flush the exporter on exit
```

`init()` returns a shutdown handle (`handle.shutdown()` / `handle.force_flush()`), not a provider —
emit through the global OTel API. Explicit arguments beat the environment: an explicit `api_key=`
sets the Bearer header, and the `RATEL_API_KEY` fallback never overrides an `Authorization` header
you pass yourself. The endpoint resolves from `RATEL_OTLP_ENDPOINT`; the superseded `RATEL_URL` is
still read as a fallback and warns, since it also names the SDK's catalog source (ADR-0003). `endpoint` is the full traces URL; `logs_endpoint` overrides the Logs URL that
otherwise derives from sibling `/v1/logs`. On first setup, pass `enabled=False` to get an OTel-free
no-op shutdown handle without endpoint configuration or the `[otlp]` extra, `span_filter=` to
narrow spans, or `log_filter=` to narrow EventRecords (both default to accepting everything).
Repeated `init()` calls return the exact
handle from the first successful Ratel-owned initialization—even if a later caller is disabled—so
hot reload and multiple callers do not fight over the global provider; the first call's
configuration remains authoritative, and shutting that shared handle down stops export for every
caller. Shutdown is terminal: OTel's global provider is set once per process, so after
`handle.shutdown()` a later `init()` raises rather than return a dead handle. A foreign provider
still produces the actionable processor-composition error, including when it wins a registration
race.

A complete, offline-runnable version (console exporter + a `ratel.search` → `execute_tool` trace)
is in [`examples/telemetry-python`](https://github.com/ratel-ai/ratel/tree/main/examples/telemetry-python).

### Coexisting with other providers (Langfuse, the Vercel AI SDK, ...)

OpenTelemetry allows one global provider per signal, with many processors on each. When a partner
already owns the providers, add the Ratel processors instead of calling `init()`. Their defaults
forward only named `gen_ai.*` / `ratel.*` signal spans and EventRecords:

```python
from opentelemetry import _logs
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk.trace import TracerProvider
from ratel_ai_telemetry.otlp import ratel_log_record_processor, ratel_span_processor

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(existing_langfuse_processor)  # keeps every span
tracer_provider.add_span_processor(ratel_span_processor())  # Ratel takes gen_ai.*/ratel.* only

logger_provider = LoggerProvider()
logger_provider.add_log_record_processor(ratel_log_record_processor())
_logs.set_logger_provider(logger_provider)
```

Pass `span_filter=lambda _s: True` or `log_filter=lambda _r: True` (or your own predicates)
to override the defaults. `ratel_span_exporter()` and `ratel_log_exporter()` are the bare OTLP
exporters if you want to wire your own processors. Note that per-span filtering can orphan the
AI SDK's `ai.*` wrapper from its `gen_ai.*` child; send everything (or tail-sample) when you
need full-trace fidelity rather than just the gen_ai/ratel metrics. `enabled=False` returns an
OTel-free no-op processor without resolving configuration.

## Package shape

- Distribution name: `ratel-ai-telemetry`; import name: `ratel_ai_telemetry`
- Pure Python (hatchling build, no Rust extension); OTel-free constants, `init()` behind the
  `[otlp]` extra. That extra installs the complete exporter/SDK stack; callers do not install
  individual OpenTelemetry packages.
- Targets Python >=3.9 (the `[otlp]` OTel deps are pinned below 1.42, the last line supporting 3.9)
- Released under the `telemetry-py-v*` tag prefix ([ADR-0008](https://github.com/ratel-ai/ratel/blob/main/docs/adr/0008-release-engineering.md))
- MIT ([ADR-0009](https://github.com/ratel-ai/ratel/blob/main/docs/adr/0009-licensing.md))

## Build & test

From this directory (needs [uv](https://docs.astral.sh/uv/)):

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv -e '.[dev]'
.venv/bin/ruff check . && .venv/bin/mypy ratel_ai_telemetry && .venv/bin/pytest
```

Unlike the Python SDK there is no `maturin develop` step — the package is pure Python,
installed editable (`[dev]` pulls the `[otlp]` extra so the tests exercise the real SDK).
The tests cover the vocabulary (each constant asserted against the pin), disabled/filtered/
idempotent/foreign-provider `init()` behavior for both signal providers, endpoint/auth resolution
and the content-capture gate, the default predicates and processor no-op/filtering behavior, a
purity guard that importing the package pulls no OTel, and the shared
contract-against-the-pin conformance in
[`conformance/`](https://github.com/ratel-ai/ratel/tree/main/src/telemetry/conformance) (spans and EventRecords built from these constants through the
real SDK must emit the exact pinned keys).
