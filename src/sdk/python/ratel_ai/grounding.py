"""The re-injection freshness gate for facts — the Python port of `grounding.ts`.

This is the pure decision layer behind "inject a fact only if it isn't already
in the context." Once a fact's body is injected as a transcript message it
*stays* in the history on every later turn, so re-appending it each turn would
duplicate tokens and confuse the model. This module decides, per fact, whether
to (re-)inject — and the signal is **the fact's own body text**: a fact is
"present" when its body appears verbatim anywhere in the transcript. No markers,
no tags, no extra tokens — the injected content is its own record. Compaction
dropping the text naturally re-arms injection, and an edited body (no longer
found verbatim) naturally re-injects the new version.

The one contract this puts on hosts: render ``body`` **verbatim** in the message
you append (decorate around it, don't rewrite it) — otherwise the gate can't see
it next turn and will re-inject.

Everything here is a pure function of its inputs (framework-agnostic): the
caller extracts per-message text and renders the chosen injections; this module
never touches a message shape.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "FACT_ID_PATTERN",
    "FactCandidate",
    "assert_message_sequence",
    "GroundingItem",
    "GroundingResult",
    "GroundingSnapshotItem",
    "InjectionDecision",
    "InjectionDecisionReason",
    "InjectionReason",
    "PinTier",
    "plan_injection",
]

InjectionReason = Literal["never", "evicted", "mutated"]
"""Why a fact was chosen for (re-)injection. Mirrors the core `FactInjectReason`.

- ``never`` — not present in the transcript and never injected this session.
- ``evicted`` — injected earlier but its body is gone now (trimmed / compacted
  out of the window).
- ``mutated`` — the registered body changed since it was injected: either the
  current body is absent, or the *stale* body it replaced is still in the window
  (a retraction, where the new body is a substring of the old).
"""

InjectionDecisionReason = Literal["never", "evicted", "mutated", "fresh"]
"""An `InjectionReason`, plus ``fresh`` — a fact left alone because its body is
still in context."""

PinTier = Literal["always", "retrieved"]
"""The two tiers a fact's `pin` splits into — always-on vs retrieval-gated."""


def assert_message_sequence(transcript: Sequence[str]) -> None:
    """Reject a bare ``str`` passed where a sequence of messages is expected.

    ``str`` is itself a ``Sequence[str]``, so a caller who passes one satisfies
    the annotation, mypy stays green, and the gate then iterates *characters* —
    no single "message" contains a multi-character body, so every fact reads as
    absent and re-injects every turn. TypeScript's ``readonly string[]`` rejects
    this at compile time; Python needs the runtime guard.

    Args:
        transcript: the value to check.

    Raises:
        TypeError: if `transcript` is a ``str``.
    """
    if isinstance(transcript, str):
        raise TypeError(
            "ratel: transcript must be a sequence of message strings, not a single str "
            "(a str is itself a Sequence[str], so it would be iterated character by "
            "character and silently defeat the freshness gate) — pass [transcript] "
            "for a one-message history"
        )


@dataclass(frozen=True)
class FactCandidate:
    """A fact under consideration for injection this turn."""

    # The fact id — keys the session's injected-body memory and the trace events.
    id: str
    # The fact's current body — the text whose presence in the transcript is checked.
    body: str


@dataclass(frozen=True)
class InjectionDecision:
    """One fact's verdict from `plan_injection`."""

    # The fact id this verdict is for.
    id: str
    # Whether to (re-)inject the fact this turn.
    inject: bool
    # Why — an `InjectionReason` when injecting, ``fresh`` when skipping.
    reason: InjectionDecisionReason


def plan_injection(
    candidates: Sequence[FactCandidate],
    transcript: Sequence[str],
    previously_injected: Mapping[str, str] | None = None,
) -> list[InjectionDecision]:
    """Decide, per candidate, whether to inject its body this turn.

    The heart of the freshness gate. Pure and deterministic: the verdicts come
    back in candidate order and depend only on the inputs, so repeated calls in
    one turn never disagree.

    Presence is a literal substring check of the candidate's body against the
    transcript text — no regex, no parsing, no markers; the fastest and most
    robust form of the test, and semantically honest: it answers "is this
    information in the window?" regardless of who put it there. A candidate with
    an empty body is trivially present (there is nothing to inject) and is
    skipped as ``fresh``.

    Presence alone is not sufficient for an **edited** body, though. A retraction
    shrinks the body, which leaves the new text a substring of the stale text
    still sitting in the transcript — so a plain presence check reads ``fresh``
    and the retracted claim stays asserted to the model forever. A known-mutated
    fact whose *stale* body is still in the window therefore re-injects as
    ``mutated``, regardless of presence. (The stale copy can't be removed from an
    append-only transcript; injecting the correction at least puts the current
    version last.)

    Args:
        candidates: the facts to consider this turn (pinned always-on facts plus
            retrieved hits).
        transcript: per-message text of the current history, oldest first. A bare
            ``str`` is rejected: it is itself a ``Sequence[str]``, so it would
            iterate characters and silently defeat the gate.
        previously_injected: the bodies this session already injected, keyed by
            fact id — the caller's bookkeeping (e.g. `FactCatalog`'s). Refines
            the absent case: absent + previously-injected-same-body ⇒
            ``evicted``; absent + previously-injected-different-body ⇒
            ``mutated``; absent + unseen ⇒ ``never``. Omit it and every absent
            fact reads as ``never``.

    Returns:
        One `InjectionDecision` per candidate, in the same order.

    Raises:
        TypeError: if `transcript` is a bare ``str`` rather than a sequence of
            messages.
    """
    assert_message_sequence(transcript)

    # Per-message scan: a body counts as present only when contained within a
    # single message — matching how injections render (one message carries the
    # whole body). Scanning a joined haystack instead would false-positive when
    # two unrelated messages happen to end/start with the two halves of a
    # multi-line body.
    def present(text: str) -> bool:
        return any(text in message for message in transcript)

    decisions: list[InjectionDecision] = []
    for candidate in candidates:
        # Nothing to inject: trivially fresh, and checked first so a body edited
        # down to "" can never be injected by the retraction rule below.
        if candidate.body == "":
            decisions.append(InjectionDecision(candidate.id, False, "fresh"))
            continue
        last = previously_injected.get(candidate.id) if previously_injected is not None else None
        # Retraction: the body changed and the version it replaced is still in
        # the window. Presence of the new body proves nothing here — shrinking a
        # body leaves it a substring of the stale text.
        if last is not None and last != candidate.body and present(last):
            decisions.append(InjectionDecision(candidate.id, True, "mutated"))
        elif present(candidate.body):
            decisions.append(InjectionDecision(candidate.id, False, "fresh"))
        elif last is None:
            decisions.append(InjectionDecision(candidate.id, True, "never"))
        elif last == candidate.body:
            decisions.append(InjectionDecision(candidate.id, True, "evicted"))
        else:
            decisions.append(InjectionDecision(candidate.id, True, "mutated"))
    return decisions


@dataclass(frozen=True)
class GroundingItem:
    """One fact the grounding pass decided to (re-)inject this turn."""

    # The fact id.
    id: str
    # The fact body — render it **verbatim** as (part of) the message content;
    # its presence in the transcript is what dedupes the next turn.
    body: str
    # Why it was injected.
    reason: InjectionReason
    # Which tier it came from.
    pin: PinTier


@dataclass(frozen=True)
class GroundingResult:
    """The outcome of a grounding pass — what to inject and what was left fresh."""

    # Facts to render into the transcript, always-on tier first.
    inject: list[GroundingItem] = field(default_factory=list)
    # Ids left alone because their body is still in the context (observability).
    skipped: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GroundingSnapshotItem:
    """One fact riding along in a per-call grounding snapshot.

    Produced by `FactCatalog.ground_snapshot` — nothing persisted.
    """

    # The fact id.
    id: str
    # The fact body.
    body: str
    # Which tier it came from.
    pin: PinTier


FACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]+\Z")
"""The set of fact ids a catalog accepts.

Ids ride in trace events and in structured injection payloads (the adapter
tool-pair shape), so they stay conservative: letters, digits, and ``. _ : -``
only.
"""
