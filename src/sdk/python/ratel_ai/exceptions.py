"""Typed model errors, surfaced from the native binding.

``EmbedderError`` subclasses ``RuntimeError`` so existing ``except RuntimeError``
handlers keep working; ``DimensionMismatchError`` subclasses it specifically for
vector-width mismatches. A model-identity mismatch remains an ``EmbedderError``.
Invalid embedding *config* (a bad source combination) is raised as a plain
``ValueError`` at construction.

``CompressorError`` is the prompt-compression analogue (ADR-0016), also a
``RuntimeError`` subclass. It is deliberately a separate type: every
``EmbedderError`` message says "embedding model", and compression is stateless
per call, so it has no dimension- or model-mismatch failure modes.
"""

from __future__ import annotations

from ._native import CompressorError, DimensionMismatchError, EmbedderError

__all__ = ["CompressorError", "DimensionMismatchError", "EmbedderError"]
