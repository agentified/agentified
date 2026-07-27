"""Tests for the grounding freshness planner — mirrors `src/sdk/ts/src/grounding.test.ts`."""

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
