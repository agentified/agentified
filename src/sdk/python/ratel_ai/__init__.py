"""Python SDK for Ratel — context engineering for AI agents.

Mirrors the public surface of the TypeScript SDK (`@ratel-ai/sdk`):
- `ToolRegistry` / `SearchHit`, `SkillRegistry` / `SkillHit` — metadata
  registries with synchronous BM25 and explicit async dense retrieval, one per
  corpus.
- `ToolCatalog` / `ExecutableTool` — registry plus executable handlers.
- `SkillCatalog` / `Skill` — the on-demand skill analogue of `ToolCatalog`.
- `search_capabilities_tool` / `invoke_tool_tool` / `get_skill_content_tool` —
  framework-neutral capability tools.
- `register_mcp_server` — ingest an upstream MCP server's tools (extra: mcp).

The facts/grounding surface (`FactCatalog` / `Fact`, the push-path grounding
analogue: constant content injected into the context, gated by the pure
`grounding` freshness planner) is **experimental** and lives in the opt-in
`ratel_ai.experimental` namespace, not here — it may change without a major
version bump.
"""

from ._native import SearchHit, SkillHit
from .capabilities import (
    INVOKE_TOOL_ID,
    SEARCH_CAPABILITIES_ID,
    OnUnauthorized,
    UpstreamServerInfo,
    format_upstream_line,
    invoke_tool_tool,
    search_capabilities_tool,
)
from .catalog import (
    EmbeddingModelConfig,
    EmbeddingSpec,
    EndpointEmbeddingConfig,
    ExecutableTool,
    Executor,
    HuggingFaceEmbeddingConfig,
    LocalEmbeddingConfig,
    OllamaEmbeddingConfig,
    SearchMethod,
    SearchOrigin,
    Tool,
    ToolCatalog,
    ToolRegistry,
    TraceSinkConfig,
)

# Deprecated pre-0.2.0 surface (see compat.py) — kept so `ratel-ai==0.1.x`
# callers keep working after upgrading to 0.2.0. Slated for removal (RAT-250).
from .compat import SEARCH_TOOLS_ID, search_tools_tool
from .exceptions import DimensionMismatchError, EmbedderError
from .mcp import McpServerHandle, register_mcp_server
from .skill_catalog import Skill, SkillCatalog, SkillRegistry
from .skill_tools import GET_SKILL_CONTENT_ID, get_skill_content_tool

# OpenTelemetry export of the ratel.*/gen_ai.* funnel (ADR-0007). The SDK always
# emits spans to the active OTel provider (a no-op until one is wired);
# configure_telemetry is optional sugar that installs a Ratel-owned OTLP exporter
# (needs the [otlp] extra).
from .telemetry import configure_telemetry

__all__ = [
    "GET_SKILL_CONTENT_ID",
    "INVOKE_TOOL_ID",
    "SEARCH_CAPABILITIES_ID",
    "SEARCH_TOOLS_ID",
    "DimensionMismatchError",
    "EmbedderError",
    "EmbeddingModelConfig",
    "EmbeddingSpec",
    "EndpointEmbeddingConfig",
    "ExecutableTool",
    "Executor",
    "HuggingFaceEmbeddingConfig",
    "LocalEmbeddingConfig",
    "McpServerHandle",
    "OnUnauthorized",
    "OllamaEmbeddingConfig",
    "SearchHit",
    "SearchMethod",
    "SearchOrigin",
    "Skill",
    "SkillCatalog",
    "SkillHit",
    "SkillRegistry",
    "Tool",
    "ToolCatalog",
    "ToolRegistry",
    "TraceSinkConfig",
    "UpstreamServerInfo",
    "configure_telemetry",
    "format_upstream_line",
    "get_skill_content_tool",
    "invoke_tool_tool",
    "register_mcp_server",
    "search_capabilities_tool",
    "search_tools_tool",
]
