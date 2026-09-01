"""Tests for SkillCatalog — mirrors `src/sdk/ts/src/skill-catalog.test.ts`."""

import asyncio
import threading

import pytest

from ratel_ai import EmbedderError, ReplaceOutcome, Skill, SkillCatalog, SkillRegistry


async def _catalog(*skills: Skill) -> SkillCatalog:
    c = SkillCatalog()
    for s in skills:
        await c.register(s)
    return c


async def test_skill_removed_methods_are_gone() -> None:
    # register_many / build_embeddings / rebuild_embeddings were folded into
    # the variadic, self-embedding `register` (RAT-379/async-register).
    for obj in (SkillCatalog(), SkillRegistry()):
        assert not hasattr(obj, "register_many")
        assert not hasattr(obj, "build_embeddings")
        assert not hasattr(obj, "rebuild_embeddings")


async def test_ranks_by_relevance_and_round_trips_metadata() -> None:
    catalog = await _catalog(
        Skill(
            id="supabase-auth",
            name="supabase-auth",
            description="Supabase auth: sessions, RLS, SSR client.",
            tags=["login", "sign in"],
            metadata={"stacks": ["supabase"]},
        ),
        Skill(id="vercel-deploy", name="vercel-deploy", description="Deploy to Vercel."),
    )
    hits = catalog.search("set up login", 5, "agent")
    assert hits[0].skill_id == "supabase-auth"
    assert catalog.has("supabase-auth")
    assert catalog.get("supabase-auth").metadata == {"stacks": ["supabase"]}
    assert catalog.size() == 2


async def test_invoke_returns_body_and_records_event() -> None:
    catalog = SkillCatalog()
    await catalog.register(Skill(id="s", name="s", description="d", body="# Body\n\nsteps"))
    assert "steps" in catalog.invoke("s")


def test_invoke_unknown_id_raises() -> None:
    catalog = SkillCatalog()
    with pytest.raises(ValueError, match="unknown skillId"):
        catalog.invoke("nope")


async def test_minimal_skill_without_tags_or_body() -> None:
    # A minimal skill (no tags/body) is valid in both SDKs; body defaults to "".
    catalog = SkillCatalog()
    await catalog.register(
        Skill(id="min", name="min", description="a minimal skill, no tags or body")
    )
    assert catalog.has("min")
    assert catalog.invoke("min") == ""
    assert catalog.search("minimal", 5)[0].skill_id == "min"


async def test_experimental_searchable_description_replaces_skill_description_for_ranking() -> None:
    catalog = SkillCatalog()
    await catalog.register(
        Skill(
            id="skill",
            name="skill",
            description="composedonlyterm",
            experimental_searchable_description="overrideonlyterm",
        )
    )

    assert [hit.skill_id for hit in catalog.search("overrideonlyterm", 5)] == ["skill"]
    assert catalog.search("composedonlyterm", 5) == []


async def test_re_register_replaces_in_place() -> None:
    # Re-registering an id replaces it in the native corpus, not appends a
    # duplicate: the id ranks once and the latest metadata wins (RAT-378).
    catalog = SkillCatalog()
    await catalog.register(Skill(id="s", name="s", description="REST API design", tags=["api"]))
    await catalog.register(
        Skill(
            id="s",
            name="s",
            description="Build animation-rich HTML presentations from scratch.",
            tags=["frontend"],
            body="# Slides\n\nUpdated body.",
        )
    )
    assert catalog.size() == 1
    hits = catalog.search("animation-rich HTML presentations", 10)
    assert [h.skill_id for h in hits].count("s") == 1
    assert catalog.get("s").body == "# Slides\n\nUpdated body."


async def test_semantic_skill_registration_embeds_and_search_finds_hit(
    delayed_embedding_endpoint: str,
) -> None:
    # Registration on a semantic/hybrid catalog now embeds eagerly (inside
    # `register`), so "metadata only" no longer holds — prove the end-to-end
    # effect instead: search_async finds the hit right after registration.
    catalog = SkillCatalog(
        method="semantic",
        embedding={"url": delayed_embedding_endpoint, "model": "test-model"},
    )

    await catalog.register(Skill(id="s", name="s", description="Skill"))

    assert catalog.has("s")
    hits = await catalog.search_async("skill", 5)
    assert hits[0].skill_id == "s"


async def test_unawaited_skill_register_commits_bm25_metadata_synchronously() -> None:
    # Metadata is indexed the instant register() is called; a forgotten `await`
    # never drops the corpus. BM25 search finds the skill without awaiting.
    catalog = SkillCatalog()
    pending = catalog.register(Skill(id="deploy", name="deploy", description="Deploy an app"))
    assert catalog.has("deploy")
    assert catalog.search("deploy an app", 5)[0].skill_id == "deploy"
    await pending  # drain the no-op awaitable


async def test_unawaited_semantic_skill_register_raises_on_dense_search(
    delayed_embedding_endpoint: str,
) -> None:
    # A forgotten `await` on a semantic skill catalog is caught at the next dense
    # search with an await-specific message; awaiting the build recovers.
    catalog = SkillCatalog(
        method="semantic",
        embedding={"url": delayed_embedding_endpoint, "model": "test-model"},
    )
    pending = catalog.register(Skill(id="s", name="s", description="Skill"))

    assert catalog.has("s")
    with pytest.raises(RuntimeError, match="was not awaited"):
        await catalog.search_async("skill", 5)

    await pending
    hits = await catalog.search_async("skill", 5)
    assert hits[0].skill_id == "s"


async def test_skill_catalog_register_accepts_an_iterable() -> None:
    catalog = SkillCatalog()
    await catalog.register(
        [
            Skill(id="auth", name="auth", description="Set up login"),
            Skill(id="deploy", name="deploy", description="Deploy an app"),
        ]
    )

    assert catalog.size() == 2


async def test_skill_register_iterable_validation_failure_commits_nothing() -> None:
    catalog = SkillCatalog()
    invalid = Skill(
        id="invalid",
        name="invalid",
        description="Invalid metadata",
        metadata={"bad": [object()]},  # type: ignore[list-item]
    )

    with pytest.raises((TypeError, ValueError)):
        await catalog.register(
            [Skill(id="auth", name="auth", description="Set up authentication"), invalid]
        )

    assert catalog.size() == 0
    assert catalog.search("authentication", 5) == []


def test_synchronous_dense_skill_search_points_to_search_async() -> None:
    # search() rejects a resolved semantic/hybrid method before ever touching
    # the registry, so this needs no registration (and no working model).
    catalog = SkillCatalog(method="semantic", embedding={"local": "/missing/ratel-model"})

    with pytest.raises(RuntimeError, match=r"await .*search_async"):
        catalog.search("skill", 5)


async def test_skill_catalog_register_empty_batch_on_semantic_is_a_noop() -> None:
    # Empty corpus short-circuits before any embedder load, even for the eager
    # per-register build a semantic/hybrid catalog now runs.
    catalog = SkillCatalog(method="semantic", embedding={"local": "/missing/ratel-model"})
    await catalog.register([])  # no skills → no model load, must not raise
    assert catalog.size() == 0


async def test_nonempty_skill_async_search_and_busy_registration(
    controlled_embedding_endpoint: tuple[str, threading.Event, threading.Event],
) -> None:
    endpoint, request_started, send_response = controlled_embedding_endpoint
    catalog = SkillCatalog(method="semantic", embedding={"url": endpoint, "model": "test-model"})
    register = asyncio.create_task(
        catalog.register(Skill(id="auth", name="auth", description="Set up authentication"))
    )
    for _ in range(200):
        if request_started.is_set():
            break
        await asyncio.sleep(0.01)
    assert request_started.is_set()
    try:
        with pytest.raises(RuntimeError, match=r"^registry busy; await the active operation$"):
            await catalog.register(Skill(id="deploy", name="deploy", description="Deploy an app"))
    finally:
        send_response.set()
    await register

    hits = await catalog.search_async("authentication", 5)
    assert hits[0].skill_id == "auth"


async def test_replace_all_drops_ids_absent_from_the_batch() -> None:
    catalog = await _catalog(
        Skill(id="slides", name="slides", description="Build animation-rich presentations."),
        Skill(id="api-design", name="api-design", description="REST API design patterns."),
    )

    outcome = await catalog.replace_all(
        [
            Skill(id="api-design", name="api-design", description="REST API design patterns."),
            Skill(
                id="migrations",
                name="migrations",
                description="Reversible database migrations.",
            ),
        ]
    )

    assert outcome == ReplaceOutcome(added=1, removed=1, updated=0, unchanged=1)
    assert catalog.size() == 2
    assert not catalog.has("slides")
    assert catalog.get("slides") is None
    assert catalog.search("animation-rich presentations", 5) == []
    assert catalog.search("reversible database migrations", 5)[0].skill_id == "migrations"


async def test_replace_all_makes_a_dropped_skill_unloadable() -> None:
    catalog = await _catalog(Skill(id="slides", name="slides", description="Build decks."))

    await catalog.replace_all([])

    assert catalog.size() == 0
    # The capability-tool boundary turns this into a structured error for the agent.
    with pytest.raises(ValueError, match=r"unknown skillId: slides"):
        catalog.invoke("slides")


async def test_replace_all_serves_the_reloaded_content_for_a_kept_id() -> None:
    catalog = await _catalog(
        Skill(id="slides", name="slides", description="Build decks.", body="# old")
    )

    outcome = await catalog.replace_all(
        [Skill(id="slides", name="slides", description="Build static HTML decks.", body="# new")]
    )

    assert outcome == ReplaceOutcome(added=0, removed=0, updated=1, unchanged=0)
    assert catalog.get("slides").description == "Build static HTML decks."
    assert catalog.invoke("slides") == "# new"
    assert catalog.search("static HTML decks", 5)[0].skill_id == "slides"


async def test_replace_all_keeps_the_last_of_duplicate_ids() -> None:
    catalog = SkillCatalog()

    await catalog.replace_all(
        [
            Skill(id="slides", name="slides", description="First."),
            Skill(id="slides", name="slides", description="Second wins."),
        ]
    )

    assert catalog.size() == 1
    assert catalog.get("slides").description == "Second wins."


async def test_replace_all_rejects_non_skill_items_before_touching_the_corpus() -> None:
    catalog = await _catalog(Skill(id="slides", name="slides", description="Build decks."))

    with pytest.raises(TypeError, match=r"replace_all requires Skill items"):
        await catalog.replace_all(["not-a-skill"])  # type: ignore[list-item]

    assert catalog.has("slides"), "a rejected batch must leave the corpus untouched"


async def test_replace_all_on_semantic_embeds_the_new_skills(
    controlled_embedding_endpoint: tuple[str, threading.Event, threading.Event],
) -> None:
    endpoint, _request_started, send_response = controlled_embedding_endpoint
    send_response.set()
    catalog = SkillCatalog(method="semantic", embedding={"url": endpoint, "model": "test-model"})
    await catalog.register(Skill(id="auth", name="auth", description="Set up authentication"))

    await catalog.replace_all(
        [Skill(id="deploy", name="deploy", description="Deploy an app to production")]
    )

    hits = await catalog.search_async("deploy", 5, method="semantic")
    assert [hit.skill_id for hit in hits] == ["deploy"]


async def test_replace_all_refuses_a_second_reload_while_the_first_embeds(
    controlled_embedding_endpoint: tuple[str, threading.Event, threading.Event],
) -> None:
    # The TS twin can fire both reloads in one tick, because napi takes the
    # dense permit synchronously. Here `_dense_pending` only rises once the
    # embedding request is actually in flight, so the endpoint holds the first
    # reload open to create the overlap.
    endpoint, request_started, send_response = controlled_embedding_endpoint
    catalog = SkillCatalog(method="semantic", embedding={"url": endpoint, "model": "test-model"})

    first = asyncio.ensure_future(
        catalog.replace_all([Skill(id="slides", name="slides", description="Build decks.")])
    )
    await asyncio.to_thread(request_started.wait)
    try:
        # Refused outright — not queued behind the first, not merged into it.
        with pytest.raises(RuntimeError, match=r"^registry busy; await the active operation$"):
            catalog.replace_all(
                [Skill(id="api-design", name="api-design", description="REST API patterns.")]
            )
    finally:
        send_response.set()
    await first

    # The corpus is exactly the first batch, never a blend of the two.
    assert catalog.has("slides")
    assert not catalog.has("api-design")
    assert catalog.size() == 1


async def test_replace_all_reports_counts_when_the_embedding_pass_fails() -> None:
    catalog = SkillCatalog(method="semantic", embedding={"local": "/missing/ratel-model"})

    reload = catalog.replace_all(
        [
            Skill(id="slides", name="slides", description="Build decks."),
            Skill(id="api-design", name="api-design", description="REST API design patterns."),
        ]
    )

    # The swap commits before the embedding pass is driven, so the counts are
    # final at call time. A reload that fails to embed still reports what it
    # changed — the corpus is live and BM25 ranks it, which is exactly the
    # state ADR-0015 tells a periodic source to expect and report on.
    assert (reload.added, reload.removed) == (2, 0)

    with pytest.raises(EmbedderError):
        await reload
    assert catalog.has("slides")


def test_dense_weight_is_wired_on_the_skill_catalog_too() -> None:
    """Separate wiring from ToolCatalog's, so it needs its own proof."""
    for bad in (-0.1, 1.1, 70.0, float("nan")):
        with pytest.raises(ValueError, match="experimental_dense_weight"):
            SkillCatalog(method="hybrid", experimental_dense_weight=bad)
    for good in (0.0, 0.3, 0.7, 1.0):
        SkillCatalog(method="hybrid", experimental_dense_weight=good)
