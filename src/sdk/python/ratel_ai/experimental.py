"""Experimental facts/grounding API — opt in explicitly.

This module re-exports the entire facts and grounding surface: `FactCatalog` /
`Fact` (the push-path grounding analogue — constant content injected into the
context), the pure `plan_injection` freshness gate, and their supporting types.

It is an **experimental** API. It may change or be removed without a major
version bump, so it lives outside the stable root package instead of `ratel_ai`.
Import from here so the dependency is explicit:

    from ratel_ai.experimental import FactCatalog, Fact, plan_injection

Constructing a `FactCatalog` emits a one-time `ExperimentalWarning`; set
`RATEL_EXPERIMENTAL_SILENCE=1` to silence it.
"""

from __future__ import annotations

from .fact_catalog import (
    ExperimentalWarning,
    Fact,
    FactCatalog,
    FactHit,
    FactRegistry,
    Pin,
)
from .grounding import (
    FACT_ID_PATTERN,
    FactCandidate,
    GroundingItem,
    GroundingResult,
    GroundingSnapshotItem,
    InjectionDecision,
    InjectionDecisionReason,
    InjectionReason,
    PinTier,
    plan_injection,
)

__all__ = [
    "FACT_ID_PATTERN",
    "ExperimentalWarning",
    "Fact",
    "FactCandidate",
    "FactCatalog",
    "FactHit",
    "FactRegistry",
    "GroundingItem",
    "GroundingResult",
    "GroundingSnapshotItem",
    "InjectionDecision",
    "InjectionDecisionReason",
    "InjectionReason",
    "Pin",
    "PinTier",
    "plan_injection",
]
