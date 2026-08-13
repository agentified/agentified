"""Public runtime-events and catalog-snapshot contract (ADR-0019)."""

from __future__ import annotations

import json

import pytest

from ratel_ai import (
    ExecutableTool,
    RuntimeCatalog,
    Skill,
    SkillCatalog,
    ToolCatalog,
)


@pytest.mark.asyncio
async def test_returns_complete_serializable_catalog_without_executable_content() -> None:
    tools = ToolCatalog()
    skills = SkillCatalog()
    await tools.register(
        [
            ExecutableTool(
                id="z_tool",
                name="Z tool",
                description="Last by id",
                input_schema={"type": "object", "properties": {"value": {"type": "string"}}},
                output_schema={"type": "string"},
                execute=lambda _args: {"secret": "must never escape"},
            ),
            ExecutableTool(
                id="a_tool",
                name="A tool",
                description="First by id",
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                execute=lambda _args: {},
            ),
        ]
    )
    await skills.register(
        Skill(
            id="skill-a",
            name="Skill A",
            description="Public skill metadata",
            tags=["public"],
            tools=["a_tool"],
            metadata={"stacks": ["python"]},
            body="May contain private instructions",
        )
    )

    snapshot = RuntimeCatalog(tools, skills, source_id="service-a").snapshot()

    assert snapshot == {
        "source_id": "service-a",
        "tools": [
            {
                "id": "a_tool",
                "name": "A tool",
                "description": "First by id",
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
            },
            {
                "id": "z_tool",
                "name": "Z tool",
                "description": "Last by id",
                "input_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                },
                "output_schema": {"type": "string"},
            },
        ],
        "skills": [
            {
                "id": "skill-a",
                "name": "Skill A",
                "description": "Public skill metadata",
                "tags": ["public"],
                "tools": ["a_tool"],
                "metadata": {"stacks": ["python"]},
            }
        ],
    }
    encoded = json.dumps(snapshot)
    assert "execute" not in encoded
    assert "private instructions" not in encoded
