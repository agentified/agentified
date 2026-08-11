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

Use `ToolCatalog` for ranked tools with sync or async handlers and `SkillCatalog` for ranked Markdown playbooks loaded on demand. Expose `search_capabilities_tool`, `invoke_tool_tool`, and `get_skill_content_tool` so an agent can discover tools and skills, invoke tools, and load full skill instructions. Tools from existing MCP servers can be ingested into the tool catalog with the `mcp` extra.

Semantic and hybrid retrieval use a configurable embedding model ([ADR 0012](../../../docs/adr/0012-configurable-embedding-models.md)), set per catalog via the `embedding` argument: the built-in default, a HuggingFace repo or local directory (in-process), or an OpenAI-compatible endpoint (OpenAI, Ollama, TEI, vLLM).

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

Build-time embedding artifacts ([ADR 0017](../../../docs/adr/0017-build-time-embedding-artifacts.md)) skip re-embedding an unchanged corpus on cold start: `experimental_build_embedding_artifact` writes a mixed Tool+Skill RAT1 file, and catalogs accept `experimental_embedding_artifact` (`path` or `bytes`; default `on_miss="error"`) to warm the dense cache on `register` / `replace_all` for any search method. Artifact warm failures are typed `ArtifactWarmError`.

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

Telemetry export is optional. With the `otlp` extra installed, `configure_telemetry()` reads `RATEL_OTLP_ENDPOINT` (falling back to the superseded `RATEL_URL`, which warns) and `RATEL_API_KEY`, wires trace and Logs exporters, and returns a shutdown handle. It exports only `gen_ai.*`/`ratel.*` signal spans and EventRecords by default — `export_all_spans=True` widens spans only. Message/tool content stays off by default; opt in with `capture_content`/`include_span_and_events` (see the [telemetry guide](https://docs.ratel.sh/docs/telemetry) for the capture modes and their privacy implications). Hosts that already own OpenTelemetry providers add both `ratel_span_processor` and `ratel_log_record_processor` instead.

Package layout: `ratel_ai/` is the Python surface (including `embedding_artifact.py` for build/warm helpers), `native/` contains the PyO3 binding, and `tests/` exercises both. For local development, create `.venv` with `uv`, install `maturin`, `pytest`, `pytest-asyncio`, `ruff`, and `mypy`, then run `.venv/bin/maturin develop` and `.venv/bin/pytest`.
