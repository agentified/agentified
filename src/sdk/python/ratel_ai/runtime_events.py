"""Public runtime-event stream and catalog snapshot seams (ADR-0019)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .catalog import ToolCatalog
    from .skill_catalog import SkillCatalog


class RuntimeCatalog:
    """Complete serializable tool and skill state for snapshot publication."""

    def __init__(
        self,
        tools: ToolCatalog,
        skills: SkillCatalog,
        *,
        source_id: str,
    ) -> None:
        self._tools = tools
        self._skills = skills
        self._source_id = source_id

    def snapshot(self) -> dict[str, Any]:
        """Return the current full replacement snapshot."""
        return {
            "source_id": self._source_id,
            "tools": self._tools.snapshot(),
            "skills": self._skills.snapshot(),
        }
