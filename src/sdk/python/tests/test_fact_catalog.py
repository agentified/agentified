"""Tests for FactCatalog — mirrors `src/sdk/ts/src/fact-catalog.test.ts`.

`FactCatalog.ground` is the Python home of the freshness gate (the TS SDK also
exposes it as `ratel().ground`, which delegates to the catalog). The pure
planner is covered in `test_grounding.py`; here we exercise the stateful catalog
method and the facts bucket via `search_capabilities_tool(..., fact_catalog=...)`.
"""

import warnings
from dataclasses import replace

import pytest

import ratel_ai.fact_catalog
from ratel_ai import ToolCatalog, search_capabilities_tool
from ratel_ai.catalog import TraceSinkConfig
from ratel_ai.experimental import ExperimentalWarning, Fact, FactCatalog, Pin

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


# --- facts bucket in search_capabilities (the recall analogue) ------------


async def test_search_capabilities_surfaces_facts_with_body_inline() -> None:
    tools = ToolCatalog()
    facts = FactCatalog()
    await facts.register(cancellation)
    search = search_capabilities_tool(tools, fact_catalog=facts)
    result = await search.execute({"query": "how do I cancel and get a refund"})
    assert any(f["factId"] == "cancellation" for f in result["facts"])
    hit = next(f for f in result["facts"] if f["factId"] == "cancellation")
    assert "full refund" in hit["body"]
    assert set(hit) == {"factId", "score", "description", "body"}


async def test_search_capabilities_empty_facts_bucket_when_no_fact_catalog() -> None:
    tools = ToolCatalog()
    search = search_capabilities_tool(tools)
    result = await search.execute({"query": "totally unrelated query about rockets"})
    assert result["facts"] == []


async def test_search_capabilities_clamps_top_k_facts() -> None:
    tools = ToolCatalog()
    facts = FactCatalog()
    await facts.register(
        [
            Fact(id="f1", name="f1", description="cancel a booking and get a refund"),
            Fact(id="f2", name="f2", description="reschedule a booking appointment"),
        ]
    )
    search = search_capabilities_tool(tools, fact_catalog=facts)
    result = await search.execute({"query": "cancel or reschedule a booking", "topKFacts": 1})
    assert len(result["facts"]) == 1


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
    ratel_ai.fact_catalog._warned = False
    with pytest.warns(ExperimentalWarning, match="experimental"):
        FactCatalog()
    # Latched now: a second construction in the same process stays quiet.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        FactCatalog()
    assert not any(isinstance(w.message, ExperimentalWarning) for w in caught)


def test_experimental_silence_env_suppresses_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATEL_EXPERIMENTAL_SILENCE", "1")
    ratel_ai.fact_catalog._warned = False
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
