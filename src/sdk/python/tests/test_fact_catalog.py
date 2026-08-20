"""Tests for FactCatalog — mirrors `src/sdk/ts/src/fact-catalog.test.ts`.

`FactCatalog.ground` is the Python home of the freshness gate (the TS SDK also
exposes it as `ratel().ground`, which delegates to the catalog). The pure
planner is covered in `test_grounding.py`; here we exercise the stateful catalog
method (the facts surface is host-driven only — the model-facing
`search_capabilities` tool stays fact-free, in verbatim parity with the TS SDK).
"""

import warnings
from dataclasses import replace

import pytest

import ratel_ai.fact_catalog
from ratel_ai.catalog import TraceSinkConfig
from ratel_ai.experimental import (
    ExperimentalWarning,
    Fact,
    FactCatalog,
    FactRegistry,
    Pin,
)

address = Fact(
    id="shop-address",
    name="shop-address",
    description="Where the barbershop is located and its opening hours.",
    tags=["location"],
    body="12 Baker Street, London. Open Mon–Sat, 9am–7pm.",
    pin="always",
)

cancellation = Fact(
    id="cancellation",
    name="cancellation-policy",
    description="How to cancel or reschedule a booking and get a refund.",
    tags=["booking"],
    body="Cancel at least 24h ahead for a full refund.",
    pin="retrieved",
)


# --- FactCatalog ----------------------------------------------------------


def test_returns_no_hits_from_empty_catalog() -> None:
    assert FactCatalog().search("anything", 5) == []


async def test_registers_facts_and_ranks_relevant_first() -> None:
    catalog = FactCatalog()
    await catalog.register([address, cancellation])
    hits = catalog.search("how do I cancel and get a refund", 5)
    assert len(hits) > 0
    assert hits[0].fact_id == "cancellation"


async def test_experimental_searchable_description_replaces_fact_description_for_ranking() -> None:
    catalog = FactCatalog()
    await catalog.register(
        Fact(
            id="fact",
            name="fact",
            description="composedonlyterm",
            experimental_searchable_description="overrideonlyterm",
        )
    )

    assert [hit.fact_id for hit in catalog.search("overrideonlyterm", 5)] == ["fact"]
    assert catalog.search("composedonlyterm", 5) == []


async def test_pinned_returns_only_always_in_registration_order() -> None:
    catalog = FactCatalog()
    await catalog.register(
        [
            Fact(id="a", name="a", description="always one", pin="always"),
            Fact(id="r", name="r", description="retrieved one", pin="retrieved"),
            Fact(id="b", name="b", description="always two", pin="always"),
        ]
    )
    assert [f.id for f in catalog.pinned()] == ["a", "b"]


async def test_treats_fact_with_no_pin_as_retrieved() -> None:
    catalog = FactCatalog()
    await catalog.register(Fact(id="x", name="x", description="no pin given"))
    assert catalog.pinned() == []
    assert catalog.size() == 1


async def test_re_registers_an_id_in_place() -> None:
    catalog = FactCatalog()
    await catalog.register(address)
    await catalog.register(replace(address, pin="retrieved"))
    assert catalog.size() == 1
    assert catalog.pinned() == []  # adopted the new pin


async def test_rejects_an_id_with_a_trailing_newline() -> None:
    # Python's `$` matches before a trailing newline; the pattern anchors with
    # `\Z` so "shop-address\n" is rejected like any other out-of-set character.
    catalog = FactCatalog()
    with pytest.raises(ValueError, match="must match"):
        await catalog.register(Fact(id="shop-address\n", name="n", description="d"))


async def test_rejects_an_id_with_characters_outside_the_allowed_set() -> None:
    catalog = FactCatalog()
    with pytest.raises(ValueError, match="must match"):
        await catalog.register(Fact(id="bad id", name="n", description="d"))


async def test_rejects_an_unknown_pin_value() -> None:
    catalog = FactCatalog()
    with pytest.raises(ValueError, match="invalid pin"):
        await catalog.register(Fact(id="x", name="n", description="d", pin="pinned"))


async def test_get_returns_registered_fact_including_body() -> None:
    catalog = FactCatalog()
    await catalog.register(cancellation)
    assert catalog.has("cancellation")
    fact = catalog.get("cancellation")
    assert fact is not None
    assert "full refund" in fact.body
    assert catalog.get("nope") is None


async def test_register_accepts_a_flat_tuple() -> None:
    catalog = FactCatalog()
    registry = catalog._registry
    await registry.register("hours", "hours", "opening hours", ["time"], {}, "9-5", "always")
    assert registry.search("opening hours", 5)[0].fact_id == "hours"


# --- Pin enum -------------------------------------------------------------


def test_pin_enum_is_interchangeable_with_wire_strings() -> None:
    assert Pin.ALWAYS == "always"
    assert Pin.RETRIEVED == "retrieved"


async def test_register_accepts_the_pin_enum() -> None:
    catalog = FactCatalog()
    await catalog.register(Fact(id="a", name="a", description="d", body="x", pin=Pin.ALWAYS))
    assert [f.id for f in catalog.pinned()] == ["a"]


# --- experimental warning -------------------------------------------------


def test_factcatalog_emits_experimental_warning_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RATEL_EXPERIMENTAL_SILENCE", raising=False)
    # The warning is once-per-process; reset the latch so this test sees it.
    # Via monkeypatch so the latch is restored on teardown — a bare assignment
    # would leak `False` into later tests and re-fire the warning.
    monkeypatch.setattr(ratel_ai.fact_catalog, "_warned", False)
    with pytest.warns(ExperimentalWarning, match="experimental"):
        FactCatalog()
    # Latched now: a second construction in the same process stays quiet.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FactCatalog()
    assert not any(isinstance(w.message, ExperimentalWarning) for w in caught)


def test_experimental_silence_env_suppresses_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATEL_EXPERIMENTAL_SILENCE", "1")
    # The silenced path returns before latching `_warned`, so a bare assignment
    # would leave it `False` for the rest of the process; monkeypatch restores it.
    monkeypatch.setattr(ratel_ai.fact_catalog, "_warned", False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FactCatalog()
    assert not any(isinstance(w.message, ExperimentalWarning) for w in caught)


# --- FactCatalog.ground (the freshness gate) ------------------------------


async def test_ground_injects_pinned_then_skips_when_fresh() -> None:
    catalog = FactCatalog()
    await catalog.register([address, cancellation])

    first = await catalog.ground("hi", [])
    ids = [i.id for i in first.inject]
    assert "shop-address" in ids
    shop = next(i for i in first.inject if i.id == "shop-address")
    assert shop.reason == "never"
    assert shop.pin == "always"
    assert "Baker Street" in shop.body

    transcript = [i.body for i in first.inject]
    second = await catalog.ground("hi again", transcript)
    assert second.inject == []
    assert "shop-address" in second.skipped


async def test_ground_reinjects_mutated_body() -> None:
    catalog = FactCatalog()
    await catalog.register(address)
    first = await catalog.ground("hi", [])
    transcript = [i.body for i in first.inject]

    await catalog.register(replace(address, body="New location: 40 Oxford Street."))
    second = await catalog.ground("hi", transcript)
    shop = next(i for i in second.inject if i.id == "shop-address")
    assert shop.reason == "mutated"
    assert "Oxford Street" in shop.body


async def test_ground_reinjects_evicted_via_session_state() -> None:
    catalog = FactCatalog()
    await catalog.register(address)
    await catalog.ground("hi", [])
    second = await catalog.ground("hi", ["a summary that dropped the fact"])
    shop = next(i for i in second.inject if i.id == "shop-address")
    assert shop.reason == "evicted"


async def test_ground_injects_retrieval_gated_fact_when_query_ranks_it() -> None:
    catalog = FactCatalog()
    await catalog.register([address, cancellation])
    result = await catalog.ground("I need to cancel my appointment", [])
    assert "cancellation" in [i.id for i in result.inject]


async def test_ground_emits_inject_and_skip_trace_events() -> None:
    catalog = FactCatalog(trace=TraceSinkConfig(kind="memory", session_id="t"))
    await catalog.register(address)
    first = await catalog.ground("hi", [])
    transcript = [i.body for i in first.inject]
    await catalog.ground("hi", transcript)
    types = [e["type"] for e in catalog.drain_trace_events()]
    assert "fact_inject" in types
    assert "fact_inject_skip" in types


async def test_ground_decorated_body_is_still_detected() -> None:
    # Rendering the body verbatim is the whole dedupe contract: a host may
    # decorate around the body — presence still detects it.
    catalog = FactCatalog()
    await catalog.register(address)
    first = await catalog.ground("hi", [])
    shop = next(i for i in first.inject if i.id == "shop-address")
    second = await catalog.ground("hi", [f"Note for the agent: {shop.body}"])
    assert "shop-address" in second.skipped


# --- FactCatalog.ground_snapshot (the stateless per-call mode) ------------


async def test_ground_snapshot_returns_pinned_plus_matched() -> None:
    catalog = FactCatalog()
    await catalog.register([address, cancellation])
    items = await catalog.ground_snapshot("I need to cancel my appointment")
    ids = [i.id for i in items]
    assert "shop-address" in ids  # pinned rides along regardless of match
    assert "cancellation" in ids  # matched by the query
    assert ids.index("shop-address") < ids.index("cancellation")  # pinned first
    for item in items:
        assert len(item.body) > 0


async def test_ground_snapshot_is_stateless_across_calls() -> None:
    catalog = FactCatalog()
    await catalog.register([address, cancellation])
    first = await catalog.ground_snapshot("hi")
    second = await catalog.ground_snapshot("hi")
    assert second == first
    assert "shop-address" in [i.id for i in second]


async def test_ground_snapshot_does_not_disturb_ground_freshness_state() -> None:
    catalog = FactCatalog()
    await catalog.register(address)
    await catalog.ground_snapshot("hi")  # snapshots never mark anything injected
    result = await catalog.ground("hi", [])
    shop = next(i for i in result.inject if i.id == "shop-address")
    assert shop.reason == "never"


async def test_ground_snapshot_emits_fact_snapshot_trace_events() -> None:
    catalog = FactCatalog(trace=TraceSinkConfig(kind="memory", session_id="t"))
    await catalog.register([address, cancellation])
    await catalog.ground_snapshot("cancel my booking")
    snaps = [e["fact_id"] for e in catalog.drain_trace_events() if e["type"] == "fact_snapshot"]
    assert "shop-address" in snaps
    assert "cancellation" in snaps


# --- topK budgets the retrieved tier only ---------------------------------

# Every fact shares the term "shop", so both tiers are ranked by QUERY below and
# genuinely compete for slots; the pinned three match it more strongly and take
# the top three.
_PINNED_TRIO = [
    Fact(id="p1", name="shop address", description="the shop address", body="ADDR", pin="always"),
    Fact(id="p2", name="shop hours", description="the shop hours", body="HOURS", pin="always"),
    Fact(id="p3", name="shop voice", description="the shop voice", body="VOICE", pin="always"),
]
_RETRIEVED_TRIO = [
    Fact(id="r1", name="shop cancellation", description="shop booking cancellation", body="CANCEL"),
    Fact(id="r2", name="shop pricing", description="shop prices", body="PRICE"),
    Fact(id="r3", name="shop barbers", description="shop barbers", body="BARBER"),
]
_QUERY = "shop address shop hours shop voice"


async def _stocked() -> FactCatalog:
    catalog = FactCatalog()
    await catalog.register([*_PINNED_TRIO, *_RETRIEVED_TRIO])
    return catalog


async def test_pinned_facts_do_not_consume_the_retrieved_budget() -> None:
    catalog = await _stocked()
    # The query favours the PINNED tier, so with a plain-`k` search all three top
    # slots go to pinned facts that are then filtered back out, leaving the
    # retrieved tier permanently empty. Pin the top-3 assumption first, so this
    # test fails loudly rather than silently passing if the ranking shifts.
    ranked = [hit.fact_id for hit in await catalog.search_async(_QUERY, 3)]
    assert ranked == ["p1", "p2", "p3"], "fixture assumption: pinned take the top 3 slots"

    items = await catalog.ground_snapshot(_QUERY, top_k=3)
    assert len([i for i in items if i.pin == "retrieved"]) == 3
    assert len([i for i in items if i.pin == "always"]) == 3


async def test_returns_at_most_top_k_retrieved_facts() -> None:
    catalog = await _stocked()
    items = await catalog.ground_snapshot(_QUERY, top_k=1)
    assert len([i for i in items if i.pin == "retrieved"]) == 1
    assert len([i for i in items if i.pin == "always"]) == 3  # pinned are never capped


async def test_honours_the_catalog_level_facts_top_k_default() -> None:
    catalog = FactCatalog(facts_top_k=2)
    await catalog.register(_RETRIEVED_TRIO)
    assert len(await catalog.ground_snapshot("shop cancellation pricing barbers")) == 2


async def test_a_pinned_fact_that_also_ranks_appears_once_as_pinned() -> None:
    catalog = await _stocked()
    hits = [i for i in await catalog.ground_snapshot("shop address", top_k=5) if i.id == "p1"]
    assert len(hits) == 1
    assert hits[0].pin == "always"


# --- FactRegistry (the lower-level public surface) -------------------------


@pytest.mark.parametrize("bad_id", ["has space", "id\nwith\nnewlines", "", "emoji🙂"])
async def test_registry_rejects_an_invalid_id(bad_id: str) -> None:
    # The registry is exported from `experimental` too, so it can't be a hole in
    # the id contract the catalog enforces — the ids ride the same trace events
    # and structured injection payloads either way.
    with pytest.raises(ValueError, match="fact id"):
        await FactRegistry().register([Fact(id=bad_id, name="n", description="d", body="b")])


async def test_registry_rejects_an_unknown_pin_value() -> None:
    with pytest.raises(ValueError, match="pin"):
        await FactRegistry().register(
            [Fact(id="ok", name="n", description="d", body="b", pin="sometimes")]
        )


async def test_registry_leaves_the_corpus_untouched_when_any_fact_is_invalid() -> None:
    registry = FactRegistry()
    with pytest.raises(ValueError, match="fact id"):
        await registry.register(
            [
                Fact(id="good", name="good", description="a valid fact", body="b"),
                Fact(id="bad id", name="bad", description="an invalid fact", body="b"),
            ]
        )
    assert registry.search("valid fact", 5) == []


async def test_registry_accepts_a_well_formed_id() -> None:
    registry = FactRegistry()
    await registry.register(
        [Fact(id="shop.address:v1-2_3", name="n", description="where the shop is", body="b")]
    )
    assert registry.search("where the shop is", 5)[0].fact_id == "shop.address:v1-2_3"


# --- ground: reason fidelity, multi-turn state, transcript typing ----------


async def test_ground_carries_the_decisions_reason_not_a_constant() -> None:
    # Hardcoding ``reason: "never"`` in the catalog used to pass every test here;
    # the whole point of the gate is that the reason distinguishes the cases.
    catalog = FactCatalog(trace=TraceSinkConfig(kind="memory", session_id="t"))
    await catalog.register(address)
    await catalog.ground("hi", [])
    injects = [e for e in catalog.drain_trace_events() if e["type"] == "fact_inject"]
    assert injects[0]["reason"] == "never"

    await catalog.ground("hi", ["a summary that dropped everything"])
    injects = [e for e in catalog.drain_trace_events() if e["type"] == "fact_inject"]
    assert injects[0]["reason"] == "evicted"


async def test_ground_refreshes_its_injected_body_memory_across_three_turns() -> None:
    # Two-turn tests can't see this: after a ``mutated`` re-inject the session
    # state must hold the NEW body, or turn 3 re-reports ``mutated`` forever.
    catalog = FactCatalog()
    await catalog.register(address)
    turn1 = await catalog.ground("hi", [])
    assert next(i for i in turn1.inject if i.id == "shop-address").reason == "never"

    await catalog.register(replace(address, body="New location: 40 Oxford Street."))
    turn2 = await catalog.ground("hi", [i.body for i in turn1.inject])
    assert next(i for i in turn2.inject if i.id == "shop-address").reason == "mutated"

    turn3 = await catalog.ground(
        "hi", [i.body for i in turn1.inject] + [i.body for i in turn2.inject]
    )
    assert [i.id for i in turn3.inject] == []
    assert "shop-address" in turn3.skipped


async def test_ground_rejects_a_bare_str_transcript() -> None:
    catalog = FactCatalog()
    await catalog.register(address)
    with pytest.raises(TypeError, match="sequence of message strings"):
        await catalog.ground("hi", address.body)  # type: ignore[arg-type]
