"""Adaptive usage ranking through the Python SDK (ADR-0014).

The catalog learns from what people invoke after a search, then ranks with it.
The load-bearing negative is that a query with no evidence behind it is
completely unaffected.
"""

import json
import os
import warnings
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ratel_ai import ExecutableTool, IntentGraph, SkillCatalog, ToolCatalog, TraceSinkConfig
from ratel_ai.skill_catalog import Skill


async def build_catalog() -> ToolCatalog:
    """A catalog where lexical retrieval is confidently wrong.

    "why is the build broken" hits `docker_build` on the token *build*, while
    the tool people actually reach for is `gh_run_list`.
    """
    catalog = ToolCatalog()
    await catalog.register(
        [
            ExecutableTool(
                id="docker_build",
                name="docker_build",
                description="Build a Docker image from a Dockerfile",
                execute=lambda _args: "built",
            ),
            ExecutableTool(
                id="gh_run_list",
                name="gh_run_list",
                description="List CI workflow runs and whether the build passed",
                execute=lambda _args: "listed",
            ),
            ExecutableTool(
                id="read_file",
                name="read_file",
                description="Read a file from disk",
                execute=lambda _args: "read",
            ),
        ]
    )
    return catalog


def ids(hits) -> list[str]:
    return [hit.tool_id for hit in hits]


async def use_it(catalog: ToolCatalog, query: str, chosen: str) -> None:
    """One confirmed observation: search, then invoke what you actually wanted."""
    catalog.search(query, 5)
    await catalog.invoke(chosen, {})


@pytest.mark.asyncio
async def test_ranking_is_untouched_until_enabled() -> None:
    catalog = await build_catalog()
    assert ids(catalog.search("why is the build broken", 5))[0] == "docker_build"


@pytest.mark.asyncio
async def test_learns_from_use_and_then_ranks_better() -> None:
    catalog = await build_catalog()
    graph = IntentGraph()
    catalog.experimental_enable_adaptive_ranking(graph)
    assert graph.cluster_count == 0

    await use_it(catalog, "why is the build broken", "gh_run_list")
    await use_it(catalog, "is the build broken again", "gh_run_list")
    await use_it(catalog, "the build broken on main", "gh_run_list")

    assert graph.cluster_count == 1
    order = ids(catalog.search("why is the build broken", 5))
    assert order.index("gh_run_list") < order.index("docker_build")


@pytest.mark.asyncio
async def test_a_query_with_no_evidence_is_unaffected() -> None:
    baseline = ids((await build_catalog()).search("read a file from disk", 5))

    catalog = await build_catalog()
    catalog.experimental_enable_adaptive_ranking(IntentGraph())
    await use_it(catalog, "why is the build broken", "gh_run_list")
    await use_it(catalog, "is the build broken again", "gh_run_list")
    await use_it(catalog, "the build broken on main", "gh_run_list")

    assert ids(catalog.search("read a file from disk", 5)) == baseline


@pytest.mark.asyncio
async def test_learning_survives_a_restart_via_the_wire_form() -> None:
    """The graph is in memory, so this is how a restart keeps what was learned."""
    first = await build_catalog()
    graph = IntentGraph()
    first.experimental_enable_adaptive_ranking(graph)
    await use_it(first, "why is the build broken", "gh_run_list")
    await use_it(first, "is the build broken again", "gh_run_list")
    await use_it(first, "the build broken on main", "gh_run_list")

    restored = IntentGraph.from_json(graph.to_json())
    assert restored.cluster_count == 1

    second = await build_catalog()
    second.experimental_enable_adaptive_ranking(restored)
    order = ids(second.search("why is the build broken", 5))
    assert order.index("gh_run_list") < order.index("docker_build")


@pytest.mark.asyncio
async def test_rev_tracks_writes_and_survives_the_wire_form() -> None:
    """`rev` lets a caller save only when changed and detect a stale base."""
    catalog = await build_catalog()
    graph = IntentGraph()
    catalog.experimental_enable_adaptive_ranking(graph)
    assert graph.rev == 0

    await use_it(catalog, "why is the build broken", "gh_run_list")
    after_one = graph.rev
    assert after_one > 0

    await use_it(catalog, "is the build broken again", "gh_run_list")
    assert graph.rev > after_one

    # Carried across a save/restore, so the counter stays monotonic.
    assert IntentGraph.from_json(graph.to_json()).rev == graph.rev


def test_a_future_schema_version_is_rejected() -> None:
    future = json.dumps({"v": 2, "built_from_ts": 1, "intents": []})
    with pytest.raises(ValueError, match="version"):
        IntentGraph.from_json(future)


@pytest.mark.asyncio
async def test_disabling_stops_ranking_but_keeps_what_was_learned() -> None:
    catalog = await build_catalog()
    graph = IntentGraph()
    catalog.experimental_enable_adaptive_ranking(graph)
    await use_it(catalog, "why is the build broken", "gh_run_list")
    await use_it(catalog, "is the build broken again", "gh_run_list")
    await use_it(catalog, "the build broken on main", "gh_run_list")

    catalog.experimental_disable_adaptive_ranking()
    assert ids(catalog.search("why is the build broken", 5))[0] == "docker_build"
    assert graph.cluster_count == 1
    assert "gh_run_list" in graph.to_json()


async def _jsonl_catalog(path: Path) -> ToolCatalog:
    catalog = ToolCatalog(trace=TraceSinkConfig(kind="jsonl", session_id="s", path=str(path)))
    await catalog.register(
        ExecutableTool(id="t", name="t", description="a tool", execute=lambda _a: "ok")
    )
    return catalog


@pytest.mark.asyncio
async def test_jsonl_sink_survives_enabling_adaptive_ranking(tmp_path: Path) -> None:
    # Regression: enable/disable rebuilt the learner's inner sink from the
    # memory-sink handle only, dropping a configured jsonl sink to noop — so the
    # trace file silently stopped growing. Registration writes a churn event
    # before the toggle, so assert the file grows *after* it.
    path = tmp_path / "trace.jsonl"
    catalog = await _jsonl_catalog(path)
    catalog.experimental_enable_adaptive_ranking(IntentGraph())

    before = path.stat().st_size
    catalog.search("anything", 5)
    assert path.stat().st_size > before


@pytest.mark.asyncio
async def test_jsonl_sink_survives_disabling_adaptive_ranking(tmp_path: Path) -> None:
    # The disable path clobbered the sink the same way, even when adaptive
    # ranking was never enabled.
    path = tmp_path / "trace.jsonl"
    catalog = await _jsonl_catalog(path)
    catalog.experimental_disable_adaptive_ranking()

    before = path.stat().st_size
    catalog.search("anything", 5)
    assert path.stat().st_size > before


class _FakeNative:
    """Stands in for the pyo3 native registry (which can't be monkeypatched) so
    the auto-rebuild trigger can be tested without a real embedding model. A
    dense search returns nothing; status is scripted."""

    def __init__(self, status: str) -> None:
        self._status = status

    def adaptive_ranking_status(self) -> tuple[str, str, str, bool]:
        return (self._status, "old-model", "new-model", False)

    def _search_with_method(self, query: str, top_k: int, origin: str, method: str) -> list:
        return []


async def _semantic_with_fake(status: str, *, flag: bool):
    """A semantic tool catalog whose native layer is faked and whose rebuild is
    spied. `flag` sets rebuild_on_model_change without touching the native."""
    catalog = ToolCatalog(method="semantic", embedding={"local": "/missing/ratel-model"})
    reg = catalog._registry
    reg._native = _FakeNative(status)
    reg._rebuild_on_model_change = flag
    reg.experimental_rebuild_intent_graph = AsyncMock()  # type: ignore[method-assign]
    return catalog, reg


@pytest.mark.asyncio
async def test_auto_rebuild_recovers_a_paused_graph_on_the_next_dense_search() -> None:
    catalog, reg = await _semantic_with_fake("paused: model mismatch", flag=True)
    await catalog.search_async("anything", 5, method="semantic")
    reg.experimental_rebuild_intent_graph.assert_awaited_once()


@pytest.mark.asyncio
async def test_auto_rebuild_is_off_by_default() -> None:
    catalog, reg = await _semantic_with_fake("paused: model mismatch", flag=False)
    await catalog.search_async("anything", 5, method="semantic")
    reg.experimental_rebuild_intent_graph.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_rebuild_does_nothing_when_the_arm_is_active() -> None:
    catalog, reg = await _semantic_with_fake("active", flag=True)
    await catalog.search_async("anything", 5, method="semantic")
    reg.experimental_rebuild_intent_graph.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_rebuild_stops_once_the_graph_is_recovered() -> None:
    # A successful rebuild flips status to active, so a second dense search does
    # not rebuild again — self-healing, not rebuild-every-search.
    catalog, reg = await _semantic_with_fake("paused: model mismatch", flag=True)

    async def _recover() -> None:
        reg._native._status = "active"

    reg.experimental_rebuild_intent_graph = AsyncMock(side_effect=_recover)  # type: ignore[method-assign]
    await catalog.search_async("anything", 5, method="semantic")
    await catalog.search_async("anything", 5, method="semantic")
    reg.experimental_rebuild_intent_graph.assert_awaited_once()


@pytest.mark.asyncio
async def test_skill_catalog_auto_rebuild_recovers_a_paused_graph() -> None:
    # The skill twin runs the same trigger code.
    catalog = SkillCatalog(method="semantic", embedding={"local": "/missing/ratel-model"})
    reg = catalog._registry
    reg._native = _FakeNative("paused: model mismatch")
    reg._rebuild_on_model_change = True
    reg.experimental_rebuild_intent_graph = AsyncMock()  # type: ignore[method-assign]

    await catalog.search_async("anything", 5, method="semantic")
    reg.experimental_rebuild_intent_graph.assert_awaited_once()


@pytest.mark.asyncio
async def test_enable_adaptive_ranking_raises_the_typed_busy_error_mid_build() -> None:
    # enable/disable take &mut self natively; called during an in-flight dense
    # build they must surface the SDK's typed busy error, not a raw pyo3
    # "Already borrowed". _dense_pending > 0 stands in for the in-flight build.
    catalog = await build_catalog()
    catalog._registry._dense_pending = 1
    with pytest.raises(RuntimeError, match="registry busy"):
        catalog.experimental_enable_adaptive_ranking(IntentGraph())


@pytest.mark.asyncio
async def test_disable_adaptive_ranking_raises_the_typed_busy_error_mid_build() -> None:
    catalog = await build_catalog()
    catalog._registry._dense_pending = 1
    with pytest.raises(RuntimeError, match="registry busy"):
        catalog.experimental_disable_adaptive_ranking()


@pytest.mark.asyncio
async def test_skill_enable_adaptive_ranking_raises_the_typed_busy_error_mid_build() -> None:
    # The skill twin runs the same guard.
    catalog = SkillCatalog()
    await catalog.register(
        Skill(id="s", name="s", description="a skill", tags=[], tools=[], metadata={}, body="# s")
    )
    catalog._registry._dense_pending = 1
    with pytest.raises(RuntimeError, match="registry busy"):
        catalog.experimental_enable_adaptive_ranking(IntentGraph())


@pytest.mark.asyncio
async def test_adaptive_ranking_status_exposes_the_model_detail() -> None:
    # Parity with TS: the status is still a str (== / startswith work) but also
    # carries which models a pause involves — otherwise only reachable via stderr.
    catalog = await build_catalog()
    catalog._registry._native = _FakeNative("paused: model mismatch")
    s = catalog.experimental_adaptive_ranking_status
    assert s == "paused: model mismatch"
    assert s.startswith("paused")
    assert s.built == "old-model"
    assert s.active == "new-model"
    assert s.dim_mismatch is False


@pytest.mark.asyncio
async def test_skill_adaptive_ranking_status_exposes_the_model_detail() -> None:
    catalog = SkillCatalog()
    await catalog.register(
        Skill(id="s", name="s", description="a skill", tags=[], tools=[], metadata={}, body="# s")
    )
    catalog._registry._native = _FakeNative("paused: model mismatch")
    s = catalog.experimental_adaptive_ranking_status
    assert s == "paused: model mismatch"
    assert s.built == "old-model"
    assert s.active == "new-model"


@pytest.mark.asyncio
async def test_one_graph_is_shared_between_tool_and_skill_catalogs() -> None:
    """One cluster, two edge maps.

    Giving each catalog its own graph would duplicate the cluster and split the
    evidence behind it.
    """
    graph = IntentGraph()
    tools = await build_catalog()
    skills = SkillCatalog()
    await skills.register(
        [
            Skill(
                id="ci-triage",
                name="ci-triage",
                description="Diagnose why the build failed in CI",
                tags=[],
                tools=[],
                metadata={},
                body="# steps",
            )
        ]
    )
    tools.experimental_enable_adaptive_ranking(graph)
    skills.experimental_enable_adaptive_ranking(graph)

    await use_it(tools, "why is the build broken", "gh_run_list")
    skills.search("why is the build broken", 5)
    skills.invoke("ci-triage")

    assert graph.cluster_count == 1
    wire = json.loads(graph.to_json())
    assert "gh_run_list" in wire["intents"][0]["tools"]
    assert "ci-triage" in wire["intents"][0]["skills"]


@pytest.mark.asyncio
async def test_a_capability_search_answered_by_both_counts_support_once() -> None:
    # search_capabilities fans ONE query to both catalogs (both searches happen
    # before any invoke). A tool AND a skill used for that one question must bump
    # the shared cluster's support by 1, not 1 per catalog (#3). Each catalog has
    # its own learner; they dedup through the credit slot on the shared graph.
    graph = IntentGraph()
    tools = await build_catalog()
    skills = SkillCatalog()
    await skills.register(
        Skill(
            id="ci-triage",
            name="ci-triage",
            description="Diagnose why the build failed in CI",
            tags=[],
            tools=[],
            metadata={},
            body="# steps",
        )
    )
    tools.experimental_enable_adaptive_ranking(graph)
    skills.experimental_enable_adaptive_ranking(graph)

    # Fan-out ordering: both searches (same query) before any invoke.
    tools.search("why is the build broken", 5)
    skills.search("why is the build broken", 5)
    await tools.invoke("gh_run_list", {})
    skills.invoke("ci-triage")

    wire = json.loads(graph.to_json())
    assert graph.cluster_count == 1
    assert wire["intents"][0]["support"] == 1, "one question, not one per catalog"
    assert "gh_run_list" in wire["intents"][0]["tools"]
    assert "ci-triage" in wire["intents"][0]["skills"]


@pytest.mark.asyncio
async def test_rank_and_fused_expose_the_scale_switch() -> None:
    catalog = await build_catalog()
    graph = IntentGraph()
    catalog.experimental_enable_adaptive_ranking(graph)

    cold = catalog.search("why is the build broken", 5)
    assert [h.rank for h in cold] == list(range(len(cold)))
    assert all(h.fused is False for h in cold)

    await use_it(catalog, "why is the build broken", "gh_run_list")
    await use_it(catalog, "is the build broken again", "gh_run_list")
    await use_it(catalog, "the build broken on main", "gh_run_list")

    warm = catalog.search("why is the build broken", 5)
    assert warm[0].rank == 0
    assert all(h.fused is True for h in warm)

    # Unrelated query on the same catalog stays unfused.
    assert all(not h.fused for h in catalog.search("read a file from disk", 5))


# ---- embedding-model change detection ---------------------------------------

_hub = Path(
    os.environ.get(
        "HF_HUB_CACHE",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
    )
)
_bge = (
    _hub
    / "models--BAAI--bge-small-en-v1.5"
    / "snapshots"
    / "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
)
_has_model = (_bge / "config.json").exists() and (_bge / "tokenizer.json").exists()


def _stale_graph() -> IntentGraph:
    centroid = [1.0 if i == 0 else 0.0 for i in range(384)]
    return IntentGraph.from_json(
        json.dumps(
            {
                "v": 1,
                "built_from_ts": 1,
                "model": "some-other-model",
                "intents": [
                    {
                        "id": "intent_0",
                        "label": "l",
                        "terms": [],
                        "members": ["why is the build broken"],
                        "centroid": centroid,
                        "support": 9,
                        "tools": {"gh_run_list": 1.0},
                        "skills": {},
                    }
                ],
            }
        )
    )


async def _semantic_catalog() -> ToolCatalog:
    catalog = ToolCatalog(method="semantic")
    await catalog.register(
        [
            ExecutableTool(
                id="gh_run_list",
                name="gh_run_list",
                description="list CI runs",
                execute=lambda _a: "ok",
            ),
            ExecutableTool(
                id="docker_build",
                name="docker_build",
                description="build an image",
                execute=lambda _a: "ok",
            ),
        ]
    )
    return catalog


@pytest.mark.skipif(not _has_model, reason="bge-small not cached")
@pytest.mark.asyncio
async def test_model_mismatch_pauses_warns_and_rebuild_restores() -> None:
    catalog = await _semantic_catalog()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        catalog.experimental_enable_adaptive_ranking(_stale_graph())
        assert catalog.experimental_adaptive_ranking_status == "paused: model mismatch"
        assert any("experimental_rebuild_intent_graph()" in str(w.message) for w in caught)

    await catalog.experimental_rebuild_intent_graph()
    assert catalog.experimental_adaptive_ranking_status == "active"


@pytest.mark.skipif(not _has_model, reason="bge-small not cached")
@pytest.mark.asyncio
async def test_warn_can_be_suppressed() -> None:
    catalog = await _semantic_catalog()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        catalog.experimental_enable_adaptive_ranking(_stale_graph(), warn_on_model_mismatch=False)
        assert catalog.experimental_adaptive_ranking_status == "paused: model mismatch"
        assert not caught


@pytest.mark.skipif(not _has_model, reason="bge-small not cached")
@pytest.mark.asyncio
async def test_rebuild_on_model_change_recovers_without_a_manual_rebuild() -> None:
    # End to end with a real model: a stale-model graph pauses, but the first
    # dense search rebuilds it under the active model and comes back active — no
    # explicit experimental_rebuild_intent_graph() call.
    catalog = await _semantic_catalog()
    catalog.experimental_enable_adaptive_ranking(
        _stale_graph(), warn_on_model_mismatch=False, rebuild_on_model_change=True
    )
    assert catalog.experimental_adaptive_ranking_status == "paused: model mismatch"

    await catalog.search_async("why is the build broken", 5, method="semantic")
    assert catalog.experimental_adaptive_ranking_status == "active"
