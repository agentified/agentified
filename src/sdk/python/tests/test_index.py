"""Public-surface smoke test — mirrors `src/sdk/ts/src/index.test.ts`."""

import ratel_ai


def test_public_exports_present() -> None:
    expected = {
        "ToolRegistry",
        "SearchHit",
        "ToolCatalog",
        "ExecutableTool",
        "Tool",
        "TraceSinkConfig",
        "EmbeddingModelConfig",
        "EmbeddingSpec",
        "EndpointEmbeddingConfig",
        "HuggingFaceEmbeddingConfig",
        "LocalEmbeddingConfig",
        "OllamaEmbeddingConfig",
        "SEARCH_CAPABILITIES_ID",
        "INVOKE_TOOL_ID",
        "search_capabilities_tool",
        "invoke_tool_tool",
        "format_upstream_line",
        "UpstreamServerInfo",
        "register_mcp_server",
        "McpServerHandle",
        "McpToolsListError",
        # skills surface
        "SkillRegistry",
        "SkillHit",
        "SkillCatalog",
        "Skill",
        "GET_SKILL_CONTENT_ID",
        "get_skill_content_tool",
    }
    assert expected.issubset(set(ratel_ai.__all__))
    for name in expected:
        assert hasattr(ratel_ai, name), name


async def test_end_to_end_register_search_invoke_is_wired() -> None:
    # The README quickstart path, condensed.
    catalog = ratel_ai.ToolCatalog()
    await catalog.register(
        ratel_ai.ExecutableTool(
            id="read_file",
            name="read_file",
            description="Read a file from local disk.",
            input_schema={"properties": {"path": {"type": "string"}}},
            output_schema={"properties": {"contents": {"type": "string"}}},
            execute=lambda args: {"contents": "data"},
        )
    )
    hits = catalog.search("read a file", 3)
    assert hits[0].tool_id == "read_file"
