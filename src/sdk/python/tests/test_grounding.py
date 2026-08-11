"""Tests for the grounding freshness planner — mirrors `src/sdk/ts/src/grounding.test.ts`."""

import pytest

from ratel_ai.experimental import FactCandidate, InjectionDecision, plan_injection


def cand(fact_id: str, body: str) -> FactCandidate:
    return FactCandidate(fact_id, body)


ADDRESS = "12 Baker Street, London. Open Mon–Sat 9–7."


# --- plan_injection (content-presence gate) --------------------------------


def test_injects_absent_fact_as_never_with_no_session_history() -> None:
    (decision,) = plan_injection([cand("a", ADDRESS)], [])
    assert decision == InjectionDecision("a", True, "never")


def test_skips_fact_whose_body_is_already_in_transcript() -> None:
    # The token-saving case.
    (decision,) = plan_injection(
        [cand("a", ADDRESS)],
        ["user asked something", f"Here you go: {ADDRESS}"],
    )
    assert decision == InjectionDecision("a", False, "fresh")


def test_presence_is_who_put_it_there_agnostic() -> None:
    # The assistant (or even the user) said the fact verbatim — the info is in
    # the window, so injecting again would duplicate it.
    (decision,) = plan_injection(
        [cand("a", ADDRESS)],
        [f"assistant: We're at {ADDRESS} — see you soon!"],
    )
    assert decision.inject is False


def test_classifies_absent_plus_previously_injected_same_body_as_evicted() -> None:
    (decision,) = plan_injection(
        [cand("a", ADDRESS)],
        ["a summary that dropped the fact"],
        previously_injected={"a": ADDRESS},
    )
    assert decision == InjectionDecision("a", True, "evicted")


def test_classifies_absent_plus_previously_injected_different_body_as_mutated() -> None:
    (decision,) = plan_injection(
        [cand("a", "New location: 40 Oxford Street.")],
        [f"old turn still contains: {ADDRESS}"],
        previously_injected={"a": ADDRESS},
    )
    assert decision == InjectionDecision("a", True, "mutated")


def test_reinjects_a_retracted_body_as_mutated_even_though_it_reads_present() -> None:
    # The retraction direction: shrinking a body leaves the new text a substring
    # of the stale text still in the window, so a plain presence check would call
    # it ``fresh`` and the withdrawn clause would stay asserted to the model
    # forever. The stale body's presence is what arms this.
    old_body = "Cancel 24h ahead for a full refund; same-day is a 50% fee."
    new_body = "Cancel 24h ahead for a full refund;"
    (decision,) = plan_injection(
        [cand("cx", new_body)],
        [old_body],
        previously_injected={"cx": old_body},
    )
    assert decision == InjectionDecision("cx", True, "mutated")
    assert "50% fee" in old_body  # the retracted clause is what's at stake


def test_retraction_converges_once_the_new_body_is_injected() -> None:
    old_body = "Cancel 24h ahead for a full refund; same-day is a 50% fee."
    new_body = "Cancel 24h ahead for a full refund;"
    # Turn N+1: the stale body is still in history, but so is the correction, and
    # the session state now records the new body — so no re-injection loop.
    (decision,) = plan_injection(
        [cand("cx", new_body)],
        [old_body, new_body],
        previously_injected={"cx": new_body},
    )
    assert decision == InjectionDecision("cx", False, "fresh")


def test_does_not_reinject_when_stale_body_is_gone_and_new_one_is_present() -> None:
    # Compaction dropped the old text and the new body is already rendered:
    # nothing is misstated, so the retraction rule must stay quiet.
    (decision,) = plan_injection(
        [cand("a", "New location: 40 Oxford Street.")],
        ["New location: 40 Oxford Street."],
        previously_injected={"a": ADDRESS},
    )
    assert decision == InjectionDecision("a", False, "fresh")


def test_never_injects_an_empty_body_even_when_the_stale_one_is_present() -> None:
    # A fact edited down to "" has nothing to render; the retraction rule must
    # not resurrect it as an empty injection.
    (decision,) = plan_injection([cand("a", "")], [ADDRESS], previously_injected={"a": ADDRESS})
    assert decision == InjectionDecision("a", False, "fresh")


def test_rejects_a_bare_str_transcript() -> None:
    # `str` is itself a `Sequence[str]`, so this satisfies the annotation and
    # mypy — then iterates characters, and every fact reads as absent forever.
    # TypeScript's `readonly string[]` rejects it at compile time; Python needs
    # the runtime guard.
    with pytest.raises(TypeError, match="sequence of message strings"):
        plan_injection([cand("a", "AAA")], "a transcript containing AAA verbatim")  # type: ignore[arg-type]


def test_edited_body_that_is_already_present_is_simply_fresh() -> None:
    new_body = "New location: 40 Oxford Street."
    (decision,) = plan_injection(
        [cand("a", new_body)],
        [f"someone already mentioned: {new_body}"],
        previously_injected={"a": ADDRESS},
    )
    assert decision.inject is False


def test_treats_empty_body_as_trivially_present() -> None:
    # Nothing to inject.
    (decision,) = plan_injection([cand("a", "")], [])
    assert decision == InjectionDecision("a", False, "fresh")


def test_body_split_across_two_messages_is_not_falsely_present() -> None:
    # Two unrelated messages ending/starting with the halves of a multi-line
    # body must NOT reconstruct it across the message boundary.
    body = "12 Baker Street\nLondon NW1, open daily."
    (decision,) = plan_injection(
        [cand("shop-address", body)],
        ["it's on 12 Baker Street", "London NW1, open daily. Come say hi!"],
    )
    assert decision.inject is True
    assert decision.reason == "never"


def test_matches_bodies_that_span_lines_within_one_message() -> None:
    multiline = "Line one of the policy.\nLine two of the policy."
    (decision,) = plan_injection([cand("a", multiline)], [f"intro\n{multiline}\noutro"])
    assert decision.inject is False


def test_is_order_preserving_and_deterministic() -> None:
    candidates = [cand("a", "alpha body"), cand("b", "beta body"), cand("c", "gamma body")]
    transcript = ["contains beta body here"]
    first = plan_injection(candidates, transcript)
    second = plan_injection(candidates, transcript)
    assert [d.id for d in first] == ["a", "b", "c"]
    assert [d.inject for d in first] == [True, False, True]
    assert first == second


def test_returns_empty_plan_for_no_candidates() -> None:
    assert plan_injection([], ["anything"]) == []
