"""Build-time embedding artifacts (ADR-0018) — Python experimental surface.

Hosts own filesystem I/O; core remains bytes-only. Catalogs warm a configured
artifact before eager document embedding when
``experimental_embedding_artifact`` is set.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict, Union, cast

from ._native import merge_embedding_artifacts

if TYPE_CHECKING:
    from .catalog import EmbeddingSpec, Tool
    from .skill_catalog import Skill

OnArtifactMiss = Literal["error", "embed"]


class _ExperimentalEmbeddingArtifactOptions(TypedDict, total=False):
    on_miss: OnArtifactMiss


class ExperimentalEmbeddingArtifactPath(_ExperimentalEmbeddingArtifactOptions):
    """Filesystem path to a RAT1 embedding artifact."""

    path: str


class ExperimentalEmbeddingArtifactBytes(_ExperimentalEmbeddingArtifactOptions):
    """In-memory RAT1 embedding artifact bytes."""

    bytes: bytes


ExperimentalEmbeddingArtifact = Union[
    ExperimentalEmbeddingArtifactPath,
    ExperimentalEmbeddingArtifactBytes,
]


async def experimental_build_embedding_artifact(
    *,
    output: str | os.PathLike[str],
    embedding: EmbeddingSpec | None = None,
    tools: Iterable[Tool] | None = None,
    skills: Iterable[Skill] | None = None,
) -> None:
    """Build one mixed Tool+Skill RAT1 and write it to ``output``.

    Registers metadata on BM25 registries so each document is embedded at most
    once (not via catalog dense build). Empty/omitted corpora are valid; both
    empty yields a valid empty RAT1. Always builds both registry sides (empty →
    empty RAT1) so merge always receives two parts. Parent directories must
    already exist; no atomic write.

    Raises:
        EmbedderError: Embedding/model/inference failure from either registry build.
        ArtifactError: Non-embedder artifact construction failure from either registry build.
        IncompatibleMergeError: Valid Tool/Skill artifact parts cannot be internally merged.
        OSError: Filesystem failure writing ``output`` (not wrapped as artifact errors).
    """
    from .catalog import ToolRegistry
    from .skill_catalog import SkillRegistry

    tool_items = list(tools or ())
    skill_items = list(skills or ())

    tool_registry = ToolRegistry(embedding, method="bm25")
    if tool_items:
        await tool_registry.register(tool_items)
    tool_bytes = await tool_registry.experimental_build_embedding_artifact()

    skill_registry = SkillRegistry(embedding, method="bm25")
    if skill_items:
        await skill_registry.register(skill_items)
    skill_bytes = await skill_registry.experimental_build_embedding_artifact()

    merged = merge_embedding_artifacts([tool_bytes, skill_bytes])
    await asyncio.to_thread(Path(output).write_bytes, merged)


def _validate_embedding_artifact(
    config: ExperimentalEmbeddingArtifact,
) -> tuple[str | None, bytes | None, OnArtifactMiss]:
    if not isinstance(config, Mapping):
        raise TypeError("experimental_embedding_artifact must be a mapping")
    unknown = set(config) - {"path", "bytes", "on_miss"}
    if unknown:
        raise TypeError(
            f"experimental_embedding_artifact has unknown keys: {sorted(unknown)}"
        )
    path_raw = config.get("path")
    bytes_raw = config.get("bytes")
    has_path = path_raw is not None
    has_bytes = bytes_raw is not None
    if has_path == has_bytes:
        raise TypeError(
            "experimental_embedding_artifact requires exactly one of 'path' or 'bytes'"
        )
    on_miss_raw = config.get("on_miss")
    if on_miss_raw is None:
        on_miss_raw = "error"
    if on_miss_raw not in ("error", "embed"):
        raise ValueError(
            f'unknown on-artifact-miss policy {on_miss_raw!r} (expected "error" or "embed")'
        )
    assert on_miss_raw == "error" or on_miss_raw == "embed"
    on_miss: OnArtifactMiss = on_miss_raw
    if has_path:
        path = cast(ExperimentalEmbeddingArtifactPath, config)["path"]
        if not isinstance(path, str):
            raise TypeError("experimental_embedding_artifact path must be a str")
        return path, None, on_miss
    raw = cast(ExperimentalEmbeddingArtifactBytes, config)["bytes"]
    if not isinstance(raw, bytes):
        raise TypeError("experimental_embedding_artifact bytes must be bytes")
    return None, raw, on_miss


async def resolve_embedding_artifact(
    config: ExperimentalEmbeddingArtifact,
) -> tuple[bytes, OnArtifactMiss]:
    """Load artifact bytes and resolve ``on_miss`` (default ``"error"``)."""
    path, raw, on_miss = _validate_embedding_artifact(config)
    if raw is not None:
        return raw, on_miss
    assert path is not None
    return await asyncio.to_thread(Path(path).read_bytes), on_miss
