<div align="center">
  <h1>ratel-ai</h1>
  <p>Context engineering for Python agents.</p>

  <p>
    <a href="https://docs.ratel.sh">Docs</a> •
    <a href="https://github.com/ratel-ai/ratel">GitHub</a> •
    <a href="https://discord.gg/75vAPdjYqT">Discord</a>
  </p>

  <p>
    <a href="https://pypi.org/project/ratel-ai/"><img src="https://img.shields.io/pypi/v/ratel-ai?label=pypi&color=3775a9" alt="PyPI" /></a>
    <a href="https://github.com/ratel-ai/ratel/stargazers"><img src="https://img.shields.io/github/stars/ratel-ai/ratel?style=social" alt="GitHub stars" /></a>
    <a href="https://github.com/ratel-ai/ratel/blob/main/LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license" /></a>
  </p>
</div>

`ratel-ai` retrieves the tools and skills relevant to each agent turn instead of sending the full catalog to the model. It bundles Ratel's Rust engine in-process: BM25 by default, with configurable semantic and hybrid retrieval available when needed. The default and local-model paths require no API key, vector database, or service. Installing a published package on a supported prebuilt target also requires no Rust toolchain.

Use `ToolCatalog` for ranked tools with sync or async handlers and `SkillCatalog` for ranked Markdown playbooks loaded on demand. Expose `search_capabilities_tool`, `invoke_tool_tool`, and `get_skill_content_tool` so an agent can discover tools and skills, invoke tools, and load full skill instructions. Tools from existing MCP servers can be ingested into the tool catalog with the `mcp` extra. **Experimental — facts:** the opt-in `ratel_ai.experimental` namespace adds `FactCatalog` for constant grounding content (a shop's address, a brand's voice). See [Facts](#facts-experimental) below. This API may change or be removed without a major version bump.

Every tool, skill, and fact accepts an optional experimental `experimental_searchable_description`. It replaces only the description component used by BM25 and embeddings; the model-facing `description` is unchanged, names plus skill/fact tags remain indexed, and opted-in tool schemas are not indexed. When omitted, stable behavior is unchanged: tools rank their description and schema tokens, while skills and facts rank their description. This API may change without a major-version bump.

Semantic and hybrid retrieval use a configurable embedding model ([ADR 0012](../../../docs/adr/0012-configurable-embedding-models.md)), set per catalog via the `embedding` argument: the built-in default, a HuggingFace repo or local directory (in-process), or an OpenAI-compatible endpoint (OpenAI, Ollama, TEI, vLLM).

Hybrid fuses the two arms on normalised scores ([ADR 0024](../../../docs/adr/0024-hybrid-fuses-on-scores.md)). `experimental_dense_weight` (default `0.7`) sets how much of that score the semantic arm carries, with BM25 taking the remainder — `0` is pure lexical, `1` pure dense, and anything outside `[0, 1]` raises rather than being clamped. The default was measured on catalogs of natural-language descriptions; a catalog keyed on exact identifiers, error codes, or internal jargon gives BM25 purchase those corpora do not have and will want a lower value. It is read by `"hybrid"` only and does not scale the adaptive-ranking arm.

For semantic or hybrid retrieval, `register()` folds embedding in: it accepts one tool or a whole batch and embeds on a worker thread, so model loading, HTTP, and inference never block the asyncio loop or hold the GIL — and embedding errors surface right at `register()`:

```python
async def retrieve(tools):
    catalog = ToolCatalog(method="semantic", embedding={"ollama": "nomic-embed-text"})
    await catalog.register(tools)                              # embeds the batch here
    return await catalog.search_async("deploy the service", 5)
```

`register()` is async for every method (BM25 too); `search()` stays synchronous for BM25 only, and `search_async()` covers all three. To change the endpoint's model or vector dimension, construct a new catalog and re-register.

A `SkillCatalog` also takes a whole reloaded catalog at once with `replace_all()`, for a source that fetches the full set rather than individual changes ([ADR 0015](../../../docs/adr/0015-whole-catalog-skill-reload.md)). The batch *is* the catalog: ids missing from it are removed, including ones registered in-process, so a host that mixes local and remote skills composes the batch itself. It mutates in place, so every holder of the catalog sees the reload without being rebuilt.

```python
outcome = await catalog.replace_all([*local_skills, *await fetch_remote_skills()])
print(f"reload: +{outcome.added} -{outcome.removed} ~{outcome.updated}")
```

The corpus swap is the synchronous half of that call, so the counts are already final when it returns — read them without awaiting and a reload whose embedding pass fails still reports what it changed:

```python
reload = catalog.replace_all(batch)  # corpus is live; counts are final
try:
    await reload  # drives the embedding pass
except EmbedderError:
    log.warning("applied +%d -%d, embeddings pending", reload.added, reload.removed)
```

Only new and re-worded skills are embedded — reloading an unchanged catalog costs no embedding calls — and a reload that races an in-flight operation — dense work, but also an ordinary BM25 `search_async` holding the read lock — raises rather than applying half of itself.

Build-time embedding artifacts ([ADR 0018](../../../docs/adr/0018-build-time-embedding-artifacts.md), experimental) avoid corpus/document embedding inference for covered entries on cold start: `experimental_build_embedding_artifact` writes a mixed Tool+Skill RAT1 (halves merged internally; no public merge API), and catalogs accept `experimental_embedding_artifact` (`path` or `bytes`; default `on_miss="error"`) to warm the dense cache on `register` / `replace_all` — each call re-resolves the artifact source and re-warms the whole current corpus. `ToolRegistry` / `SkillRegistry` also expose `experimental_build_embedding_artifact` and `experimental_warm_embeddings_from_artifact`. With default `on_miss="error"`, every id in each non-empty registering corpus must be covered; a tool-only artifact is valid while Skill stays empty (and vice versa); when both sides register, use a mixed artifact or `on_miss="embed"`. Semantic/hybrid search still requires query embedding through the configured backend; Local/HF paths may still initialize/load the model, and endpoint performs its normal remote query embedding. `ArtifactWarmError` covers warm failures (`.code`, `.missing`); `ArtifactError` covers non-embedder artifact construction failures (`EmbedderError` remains the embedding/backend failure); `IncompatibleMergeError` may surface from the high-level mixed builder's internal Tool+Skill composition; writing the output file may raise `OSError`.

## Install

```bash
pip install ratel-ai
# MCP ingestion: pip install 'ratel-ai[mcp]'
```

## Quickstart

Save as `quickstart.py`, then run `python quickstart.py`:

```python
import asyncio
from ratel_ai import ExecutableTool, ToolCatalog

async def main():
    catalog = ToolCatalog()
    await catalog.register(
        ExecutableTool(
            id="get_weather",
            name="get_weather",
            description="Get the current weather for a city.",
            input_schema={"properties": {"city": {"type": "string"}}},
            output_schema={"type": "object"},
            execute=lambda args: {"forecast": f"Sunny in {args['city']}"},
        )
    )

    hit = catalog.search("What is the weather in Rome?", 1)[0]
    print(await catalog.invoke(hit.tool_id, {"city": "Rome"}))


asyncio.run(main())
```

Continue with the [Python guide](https://docs.ratel.sh/docs/sdks/python), [capability tools](https://docs.ratel.sh/docs/capability-tools), [API reference](https://docs.ratel.sh/docs/api/sdk-python), or the [Pydantic AI example](https://github.com/ratel-ai/ratel/tree/main/examples/pydantic-ai).

## Runtime events and catalog snapshots

`RuntimeEvents` merges tool and skill facts into one bounded push stream. Give the paired
`RuntimeCatalog` the stream's `source_id` so envelopes and full snapshots identify the same
deployment source:

```python
from ratel_ai import RuntimeCatalog, RuntimeEvents, SkillCatalog, ToolCatalog

tools = ToolCatalog()
skills = SkillCatalog()
events = RuntimeEvents(
    [tools, skills],
    session_id="agent-session",
    source_id="checkout-agent",
    experimental_catalog_definitions=True,
)
catalog = RuntimeCatalog(tools, skills, source_id=events.source_id)

async def publish(batch):
    await send_runtime_facts(batch)

subscription = events.subscribe(publish)  # call from the target asyncio event loop
# Register, search, and invoke through tools / skills as usual.
await subscription.flush()
snapshot = catalog.snapshot()
subscription.unsubscribe()
```

Async handlers are marshaled onto the subscribing event loop; synchronous handlers run on the
native callback thread. Both are observational and fail open. `flush()` waits for work already
accepted by the bounded native queues and for async handlers to settle. Subscribing a remote
publisher must set `experimental_catalog_definitions=True` to consent to public
`catalog_definition` fields regardless of the OTel message-content capture setting. Definition events are lossy and change-sensitive; snapshots are
the authoritative full replacement for removals and recovery. They contain sorted public
definitions only—never tool executors or skill bodies. Python exposes no Cloud transport;
applications may publish these events and snapshots through their own adapter.

## Facts (experimental)

Tools and skills are **pulled** — a query ranks them and only the winners reach the model. Facts are the opposite: constant content the agent should always work from (a shop's address, hours, a brand's voice), **pushed** into the context and deduplicated so it is injected once rather than every turn.

Facts live in the opt-in `ratel_ai.experimental` namespace and may change without a major version bump. Registering one is like a skill, plus a `pin` tier:

```python
from ratel_ai.experimental import Fact, FactCatalog, Pin

facts = FactCatalog()
await facts.register([
    Fact(
        id="shop-address",
        name="shop address & hours",
        description="where the shop is and when it's open",
        body="Fade & Blade — 12 Baker Street, London. Open Mon–Sat 9am–7pm.",
        pin=Pin.ALWAYS,       # every turn, regardless of the query
    ),
    Fact(
        id="cancellation",
        name="cancellation policy",
        description="cancelling or rescheduling a booking, and refunds",
        body="Cancel at least 24h ahead for a full refund; same-day is a 50% fee.",
        pin=Pin.RETRIEVED,    # only when the turn's query ranks it in (default)
    ),
])
```

Then pick **one** of two injection modes per turn.

**`ground()` — persist into your stored history.** Returns only the facts not already present; render each `body` verbatim and keep it in the messages you save. It takes a **list of per-message strings** — flatten multi-part content yourself, and note that a bare `str` is rejected (it is itself a `Sequence[str]`, so it would be iterated character by character):

```python
def text_of(message: dict) -> str:
    content = message["content"]
    if isinstance(content, str):
        return content
    return "\n".join(part.get("text", "") for part in content)  # multi-part content

result = await facts.ground(user_text, [text_of(m) for m in messages])
for item in result.inject:
    messages.append({"role": "system", "content": item.body})  # verbatim — presence is the dedupe
```

Turn 1 injects the address; turn 2 sees it in the transcript and injects nothing. It re-injects only when the body is gone (compaction) or was edited — `item.reason` is `"never"` / `"evicted"` / `"mutated"`.

**`ground_snapshot()` — per call, nothing stored.** Returns the full applicable set every time; put it in the request you're about to send and discard it:

```python
snapshot = await facts.ground_snapshot(user_text)
payload = [{"role": "system", "content": f.body} for f in snapshot] + messages
```

Use `ground()` for a long-lived agent whose messages you persist; `ground_snapshot()` for one-shot or stateless calls, or to keep injected content out of your stored history.

Facts are **host-driven**: the model-facing `search_capabilities` tool is unchanged and never returns facts — you decide what is true and inject it, rather than letting the model discover it. Every decision is traced (`fact_inject` with its reason, `fact_inject_skip`, `fact_snapshot`), so the skip rate — the tokens you saved — is measurable. See [ADR-0017](../../../docs/adr/0017-facts-and-injection-freshness.md).

Telemetry export is optional. With the `otlp` extra installed, `configure_telemetry()` reads `RATEL_OTLP_ENDPOINT` (falling back to the superseded `RATEL_URL`, which warns) and `RATEL_API_KEY`, wires trace and Logs exporters, and returns a shutdown handle. It exports only `gen_ai.*`/`ratel.*` signal spans and EventRecords by default — `export_all_spans=True` widens spans only. Message and tool content stays off by default; opt in with `capture_content`/`include_span_and_events` (see the [telemetry guide](https://docs.ratel.sh/docs/telemetry) for the capture modes and their privacy implications). Experimental catalog-definition export additionally requires `RATEL_EXPERIMENTAL_CATALOG_DEFINITIONS=true`. Changed definitions then emit one `ratel.catalog.definition` EventRecord per registry-local content hash. Hosts that already own OpenTelemetry providers add both `ratel_span_processor` and `ratel_log_record_processor` instead.

Package layout: `ratel_ai/` is the Python surface (including `embedding_artifact.py` for build/warm helpers), `native/` contains the PyO3 binding, and `tests/` exercises both. For local development, create `.venv` with `uv`, install `maturin`, `pytest`, `pytest-asyncio`, `ruff`, and `mypy`, then run `.venv/bin/maturin develop` and `.venv/bin/pytest`.
