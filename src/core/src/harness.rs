//! Measurement harness for the intent graph's dense clustering tier.
//!
//! Not a test — a tool. Every entry point here is `#[ignore]`d and invoked
//! explicitly; CI runs plain `cargo test --workspace` with no `--ignored` step.
//!
//! ```text
//! cargo test -p ratel-ai-core --lib harness_report -- --ignored --nocapture
//! ```
//!
//! **It reports the rule's own numbers rather than recomputing them.**
//! [`crate::usage::dense_verdict`] already decides admission and carries
//! `admitted`, `covered`, `score` and `centroid_cos`; [`Intent::coverage`]
//! already returns `qualifying`, `required` and `fraction`. The harness records
//! those verdicts turn by turn and rolls them up, so it cannot drift from what
//! the rule actually does — a parallel implementation of "coverage" would.
//!
//! Everything runs off checked-in vectors through [`FixtureEmbedder`], so a run
//! needs no model and is byte-reproducible. The model is touched only by
//! `regenerate_harness_fixtures`.

use std::collections::HashMap;
use std::sync::Arc;

use crate::DenseWeight;
use crate::embedding::{Embedded, Embedder, EmbedderError};
use crate::fusion::SCORE_FUSION_DENSE_WEIGHT;
use crate::indexing::searchable_text;
use crate::tool::Tool;
use crate::trace::NoopSink;
use crate::usage::{Capability, ClusterPolicy, Intent, IntentGraph, dense_verdict};

const TURNS: &str = include_str!("../tests/fixtures/turns.json");
const CATALOG: &str = include_str!("../tests/fixtures/catalog.json");

/// Every turn stamps the same timestamp, so `recency_factor` is 1.0 and nothing
/// is evicted — the graph is a pure function of the fixture, not of wall clock.
const T0: u64 = 1_753_000_000_000;

/// How deep the served tables and the offline simulation cut. The arms still
/// retrieve to `RETRIEVE_DEPTH` before fusion; this is only the final slice, and
/// it is shared so the simulation and the live path cannot drift apart.
const SERVED_K: usize = 5;

// ---- fixtures ---------------------------------------------------------------

pub(crate) struct Turn {
    pub intent: String,
    pub invoked: String,
    pub query: String,
    pub vector: Vec<f32>,
}

pub(crate) struct CatalogEntry {
    pub id: String,
    pub description: String,
    /// The tool's argument schema. Carried because the projection folds it in,
    /// so a catalog without one cannot measure anything about how much of a
    /// tool's ranking comes from its parameters rather than its description.
    pub input_schema: serde_json::Value,
    /// The exact projection string this entry's vector was embedded from.
    ///
    /// Recorded because the lookup key and the lookup are both computed by the
    /// *current* `searchable_text`, so they always agree — a change to the
    /// projection leaves the stored vector describing text that no longer
    /// exists, and nothing in the key would notice. This is the same guard RAT1
    /// carries per entry as `projection_hash`.
    pub projection: String,
    pub vector: Vec<f32>,
}

fn json(src: &str) -> serde_json::Value {
    serde_json::from_str(src).expect("fixture is valid json")
}

fn floats(v: &serde_json::Value) -> Vec<f32> {
    v.as_array()
        .expect("vector array")
        .iter()
        .map(|x| x.as_f64().expect("float") as f32)
        .collect()
}

pub(crate) fn turns() -> Vec<Turn> {
    json(TURNS)["turns"]
        .as_array()
        .expect("turns array")
        .iter()
        .map(|t| Turn {
            intent: t["intent"].as_str().expect("intent").into(),
            invoked: t["invoked"].as_str().expect("invoked").into(),
            query: t["query"].as_str().expect("query").into(),
            vector: floats(&t["vector"]),
        })
        .collect()
}

pub(crate) fn catalog() -> Vec<CatalogEntry> {
    json(CATALOG)["tools"]
        .as_array()
        .expect("tools array")
        .iter()
        .map(|t| CatalogEntry {
            id: t["id"].as_str().expect("id").into(),
            description: t["description"].as_str().expect("description").into(),
            input_schema: t["input_schema"].clone(),
            projection: t["projection"].as_str().unwrap_or_default().into(),
            vector: floats(&t["vector"]),
        })
        .collect()
}

/// The `Tool` the registry sees. Built the same way in the harness and the
/// regenerator, so the text a vector was computed for is the text looked up.
pub(crate) fn tool_of(entry: &CatalogEntry) -> Tool {
    Tool {
        id: entry.id.clone(),
        name: entry.id.clone(),
        description: entry.description.clone(),
        // The fixture exercises the stable projection, not the override.
        experimental_searchable_description: None,
        input_schema: entry.input_schema.clone(),
        output_schema: serde_json::json!({}),
    }
}

// ---- a model-free embedder --------------------------------------------------

/// Answers from the checked-in vectors: queries on the query side, tool
/// projections on the document side. Panics on unknown text rather than
/// returning something plausible — a silent zero vector would look like a
/// legitimate non-match and quietly corrupt every number downstream.
pub(crate) struct FixtureEmbedder {
    docs: HashMap<String, Vec<f32>>,
    queries: HashMap<String, Vec<f32>>,
}

impl FixtureEmbedder {
    pub(crate) fn new() -> Self {
        let docs = catalog()
            .iter()
            .map(|e| (searchable_text(&tool_of(e)), e.vector.clone()))
            .collect();
        let queries = turns().into_iter().map(|t| (t.query, t.vector)).collect();
        Self { docs, queries }
    }

    fn look_up(map: &HashMap<String, Vec<f32>>, text: &str, side: &str) -> Vec<f32> {
        map.get(text).cloned().unwrap_or_else(|| {
            panic!("no {side}-side vector for {text:?} — regenerate the harness fixtures")
        })
    }
}

impl Embedder for FixtureEmbedder {
    fn embed_doc(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        Ok(Self::look_up(&self.docs, text, "doc"))
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedderError> {
        Ok(Self::look_up(&self.queries, text, "query"))
    }

    fn embed_query_batch_with_identity(
        &self,
        texts: &[String],
    ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
        Ok(Embedded {
            value: texts
                .iter()
                .map(|t| Self::look_up(&self.queries, t, "query"))
                .collect(),
            fingerprint: self.fingerprint(),
        })
    }

    fn fingerprint(&self) -> String {
        "fixture:bge-small-en-v1.5".into()
    }
}

// ---- building the graph -----------------------------------------------------

/// Replay `turns` through the real learning path, in order, and return the graph
/// plus the verdicts each turn saw.
pub(crate) fn replay(turns: &[Turn]) -> (IntentGraph, Vec<TurnRecord>) {
    replay_under(turns, ClusterPolicy::default())
}

/// Replay under an explicit policy — the sweep's driver.
pub(crate) fn replay_under(
    turns: &[Turn],
    policy: ClusterPolicy,
) -> (IntentGraph, Vec<TurnRecord>) {
    let mut g = IntentGraph::empty();
    g.set_cluster_policy(policy);
    let mut records = Vec::with_capacity(turns.len());
    for turn in turns {
        // Every candidate cluster's verdict, captured before the graph moves.
        let verdicts: Vec<ClusterVerdict> = g
            .intents
            .iter()
            .map(|it| ClusterVerdict::of(it, &turn.vector, policy))
            .collect();
        let before: Vec<String> = g.intents.iter().map(|i| i.id.clone()).collect();

        // Under the embedder's *own* fingerprint, not a nickname. `usage_arm`
        // compares the graph's recorded model against the registry's built
        // fingerprint and pauses the arm on any difference — so a graph stamped
        // "fixture" against an embedder reporting "fixture:bge-small-en-v1.5"
        // serves with no usage arm at all, silently.
        g.note_query_vector(
            &turn.query,
            &turn.vector,
            &FixtureEmbedder::new().fingerprint(),
        );
        g.observe_live(&turn.query, Capability::Tool, &turn.invoked, T0, true);

        let joined = g
            .intents
            .iter()
            .find(|it| it.members.contains(&turn.query))
            .map(|it| it.id.clone());
        let seeded = joined.as_ref().is_some_and(|id| !before.contains(id));
        records.push(TurnRecord {
            query: turn.query.clone(),
            intent: turn.intent.clone(),
            joined,
            seeded,
            verdicts,
        });
    }
    (g, records)
}

pub(crate) struct TurnRecord {
    pub query: String,
    pub intent: String,
    pub joined: Option<String>,
    pub seeded: bool,
    pub verdicts: Vec<ClusterVerdict>,
}

/// One cluster's verdict for one query, flattened out of the rule's own types.
pub(crate) struct ClusterVerdict {
    pub id: String,
    pub members: usize,
    pub admitted: bool,
    pub covered: bool,
    pub centroid_cos: f32,
    /// `None` when the cluster carries no comparable member vector.
    pub coverage: Option<(u32, u32, f32)>,
}

impl ClusterVerdict {
    fn of(it: &Intent, query: &[f32], policy: ClusterPolicy) -> Self {
        let verdict = dense_verdict(it, query, policy);
        Self {
            id: it.id.clone(),
            members: it.members.len(),
            admitted: verdict.is_some_and(|v| v.admitted),
            covered: verdict.is_some_and(|v| v.covered),
            centroid_cos: verdict.map_or(f32::NAN, |v| v.centroid_cos),
            coverage: it
                .coverage(query, policy)
                .map(|c| (c.qualifying, c.required, c.fraction)),
        }
    }
}

// ---- regeneration -----------------------------------------------------------

/// Re-embed both fixtures against the real model. Ignored: a tool, not a test,
/// and it needs bge-small on disk.
///
/// `cargo test -p ratel-ai-core --lib regenerate_harness_fixtures -- --ignored`
#[test]
#[ignore]
fn regenerate_harness_fixtures() {
    use crate::embedding::embedder_with_telemetry;
    use crate::embedding_config::EmbeddingModel;

    let embedder = embedder_with_telemetry(&EmbeddingModel::Default, &NoopSink).expect("model");
    let fmt = |v: &[f32]| {
        v.iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    };

    let mut doc = json(TURNS);
    for turn in doc["turns"].as_array_mut().expect("turns").iter_mut() {
        let text = turn["query"].as_str().expect("query").to_string();
        let v = embedder.embed_query(&text).expect("embed query");
        turn["vector"] = serde_json::json!(v);
        let _ = fmt(&v);
    }
    write_fixture("turns.json", &doc, "turns", &["intent", "invoked", "query"]);

    let mut doc = json(CATALOG);
    for entry in doc["tools"].as_array_mut().expect("tools").iter_mut() {
        // Through `tool_of`, not a second `Tool` literal: the harness looks a
        // vector up by the projection string, so the two must build the tool
        // identically or every lookup misses.
        let tool = tool_of(&CatalogEntry {
            id: entry["id"].as_str().expect("id").into(),
            description: entry["description"].as_str().expect("description").into(),
            input_schema: entry["input_schema"].clone(),
            projection: String::new(),
            vector: Vec::new(),
        });
        let projection = searchable_text(&tool);
        let v = embedder.embed_doc(&projection).expect("embed doc");
        entry["projection"] = serde_json::json!(projection);
        entry["vector"] = serde_json::json!(v);
    }
    write_fixture(
        "catalog.json",
        &doc,
        "tools",
        &["id", "description", "input_schema", "projection"],
    );
}

/// Write a fixture with one entry per line, vectors at 6 significant figures —
/// readable diffs, and small enough to keep in the repo.
fn write_fixture(name: &str, doc: &serde_json::Value, key: &str, fields: &[&str]) {
    let rows: Vec<String> = doc[key]
        .as_array()
        .expect("entries")
        .iter()
        .map(|e| {
            let head: Vec<String> = fields
                .iter()
                .map(|f| format!("\"{f}\": {}", e[f]))
                .collect();
            let nums: Vec<String> = e["vector"]
                .as_array()
                .expect("vector")
                .iter()
                .map(|x| format!("{:.6}", x.as_f64().expect("float")))
                .collect();
            format!(
                "    {{ {}, \"vector\": [{}] }}",
                head.join(", "),
                nums.join(",")
            )
        })
        .collect();
    let body = format!(
        "{{\n  \"note\": {},\n  \"model\": {},\n  \"revision\": {},\n  \"{key}\": [\n{}\n  ]\n}}\n",
        doc["note"],
        doc["model"],
        doc["revision"],
        rows.join(",\n")
    );
    std::fs::write(
        format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR")),
        body,
    )
    .expect("write fixture");
}

// ---- metrics ----------------------------------------------------------------

/// Cluster shape. `largest_share` and `singleton_rate` are always reported
/// together: neither means anything alone, and that pairing is what stops a
/// too-strict rule from looking like a win when it has merely shattered the
/// graph.
pub(crate) struct Shape {
    pub clusters: usize,
    pub sizes: Vec<usize>,
    pub largest_share: f32,
    pub singleton_rate: f32,
}

pub(crate) fn shape(g: &IntentGraph, turns: usize) -> Shape {
    let mut sizes: Vec<usize> = g.intents.iter().map(|i| i.members.len()).collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    Shape {
        largest_share: sizes.first().copied().unwrap_or(0) as f32 / turns as f32,
        singleton_rate: sizes.iter().filter(|n| **n == 1).count() as f32
            / sizes.len().max(1) as f32,
        clusters: sizes.len(),
        sizes,
    }
}

/// Share of turns sitting in a cluster whose majority label is their own.
///
/// The direct read on **homogeneity**, and a better one than F1 for that
/// question: F1 folds precision and recall together, so a graph that merged
/// everything scores well on recall for the same reason it is useless. Purity
/// asks only whether a cluster is about one thing. Read it beside the cluster
/// count — a graph of singletons is perfectly pure and has learned nothing.
pub(crate) fn purity(g: &IntentGraph, turns: &[Turn]) -> f32 {
    let label_of = |query: &str| {
        turns
            .iter()
            .find(|t| t.query == query)
            .map(|t| t.intent.clone())
    };
    let mut total = 0usize;
    let mut majority = 0usize;
    for it in &g.intents {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for member in &it.members {
            if let Some(label) = label_of(member) {
                *counts.entry(label).or_default() += 1;
            }
        }
        total += it.members.len();
        majority += counts.values().copied().max().unwrap_or(0);
    }
    if total == 0 {
        return 0.0;
    }
    majority as f32 / total as f32
}

/// Precision, recall and F1 over the "these two queries belong together"
/// relation, scored against the fixture's intent labels.
///
/// A bare cluster count cannot tell *split correctly into seven* from
/// *shattered into seven*; this can. It inherits whatever the labels claim,
/// which is why the fixture says in its own header that they are a judgement
/// call.
pub(crate) struct Merge {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

pub(crate) fn merge_quality(g: &IntentGraph, turns: &[Turn]) -> Merge {
    let cluster_of = |query: &str| {
        g.intents
            .iter()
            .position(|it| it.members.iter().any(|m| m == query))
    };
    let (mut tp, mut fp, mut fne) = (0u32, 0u32, 0u32);
    for (i, a) in turns.iter().enumerate() {
        for b in turns.iter().skip(i + 1) {
            if a.query == b.query {
                continue; // the same query is trivially together with itself
            }
            let same_label = a.intent == b.intent;
            let same_cluster = match (cluster_of(&a.query), cluster_of(&b.query)) {
                (Some(x), Some(y)) => x == y,
                _ => false,
            };
            match (same_cluster, same_label) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fne += 1,
                (false, false) => {}
            }
        }
    }
    let precision = tp as f32 / (tp + fp).max(1) as f32;
    let recall = tp as f32 / (tp + fne).max(1) as f32;
    Merge {
        precision,
        recall,
        f1: if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        },
    }
}

/// Served top-[`SERVED_K`] through the real fusion — BM25, dense and the usage arm.
pub(crate) fn served_topk(g: &IntentGraph, turns: &[Turn]) -> Vec<(String, Vec<String>)> {
    use crate::method::SearchMethod;
    use crate::trace::Origin;
    use std::sync::RwLock;

    let mut reg = crate::ToolRegistry::with_embedder_for_test(Arc::new(FixtureEmbedder::new()));
    for entry in catalog() {
        reg.register(tool_of(&entry));
    }
    reg.build_embeddings().expect("fixture vectors");
    reg.set_intent_graph(Some(Arc::new(RwLock::new(g.clone()))));

    let mut seen = Vec::new();
    let mut out = Vec::new();
    for turn in turns {
        if seen.iter().any(|q| q == &turn.query) {
            continue;
        }
        seen.push(turn.query.clone());
        let hits = reg
            .search_with_method(&turn.query, SERVED_K, Origin::Direct, SearchMethod::Hybrid)
            .expect("hybrid search");
        // `fused` cannot check this: hybrid sets it unconditionally, so the
        // assertion it replaces passed happily while the arm was paused by a
        // model-fingerprint mismatch and the served numbers measured two arms.
        assert!(
            matches!(
                reg.adaptive_ranking_status(),
                crate::AdaptiveRankingStatus::Active
            ),
            "the usage arm is not serving ({:?}), so these numbers measure only \
             BM25 and dense and every comparison drawn from them is vacuous",
            reg.adaptive_ranking_status()
        );
        out.push((
            turn.query.clone(),
            hits.into_iter().map(|h| h.tool_id).collect(),
        ));
    }
    out
}

/// Replay the turns as production sees them: search first, then invoke, so each
/// turn's impressions are the list the ranker actually served **at that point in
/// the graph's life**, not the list it would serve once fully grown.
///
/// Deliberately separate from [`replay`] rather than folded into it. Every other
/// section reports numbers from that function, and impressions must not be able
/// to move them — a search per turn also costs more than the sections that do not
/// need one.
pub(crate) fn replay_with_impressions(turns: &[Turn]) -> IntentGraph {
    use crate::method::SearchMethod;
    use crate::trace::Origin;
    use std::sync::RwLock;

    let shared = Arc::new(RwLock::new(IntentGraph::empty()));
    let mut reg = crate::ToolRegistry::with_embedder_for_test(Arc::new(FixtureEmbedder::new()));
    for entry in catalog() {
        reg.register(tool_of(&entry));
    }
    reg.build_embeddings().expect("fixture vectors");
    reg.set_intent_graph(Some(shared.clone()));

    let fingerprint = FixtureEmbedder::new().fingerprint();
    for turn in turns {
        let served: Vec<String> = reg
            .search_with_method(&turn.query, SERVED_K, Origin::Direct, SearchMethod::Hybrid)
            .expect("hybrid search")
            .into_iter()
            .map(|h| h.tool_id)
            .collect();
        let mut g = shared.write().expect("graph lock");
        g.note_query_vector(&turn.query, &turn.vector, &fingerprint);
        g.observe_surfacing(
            &turn.query,
            Capability::Tool,
            &turn.invoked,
            T0,
            true,
            &served,
        );
    }
    shared.read().expect("graph lock").clone()
}

/// Every arm's own ranking for one query, carrying the scores the fusion throws
/// away at `tool_registry.rs`'s `.map(|(id, _)| id)`.
///
/// Collected with **no graph attached**. With one attached, the `Bm25` and
/// `Semantic` paths fuse the usage arm in and return RRF scores, so neither
/// arm's own numbers would survive to be read here. The usage arm is taken
/// straight off the graph instead, which is what lets an offline simulation
/// reproduce the served order exactly rather than approximately.
pub(crate) struct ArmRankings {
    pub query: String,
    pub intent: String,
    /// `(id, raw BM25 score)`, best first. Unbounded and corpus-dependent —
    /// which is exactly why no fixed threshold can be written against it.
    pub bm25: Vec<(String, f32)>,
    /// `(id, cosine)`, best first. Bounded by the geometry of the model, so this
    /// is the arm a threshold can mean something on.
    pub dense: Vec<(String, f32)>,
    /// The usage arm's promoted ids and its fusion weight; absent when no
    /// cluster matched the query.
    pub usage: Option<(Vec<String>, f32)>,
    /// `Σ idf(query terms)` — the ceiling `score_fuse` divides the BM25 arm by.
    pub ceiling: f32,
}

impl ArmRankings {
    /// The usage arm in the shape `score_fuse` takes it.
    pub fn usage_pair(&self) -> Option<(&[String], f32)> {
        self.usage.as_ref().map(|(ids, w)| (ids.as_slice(), *w))
    }

    /// Each arm's own top score.
    pub fn bm25_top(&self) -> Option<(&str, f32)> {
        self.bm25.first().map(|(id, s)| (id.as_str(), *s))
    }
    pub fn dense_top(&self) -> Option<(&str, f32)> {
        self.dense.first().map(|(id, s)| (id.as_str(), *s))
    }
}

/// Each arm's full ranking, per distinct query, in fixture order.
pub(crate) fn served_arms(g: &IntentGraph, turns: &[Turn]) -> Vec<ArmRankings> {
    use crate::method::SearchMethod;
    use crate::trace::Origin;

    let entries = catalog();
    let mut reg = crate::ToolRegistry::with_embedder_for_test(Arc::new(FixtureEmbedder::new()));
    for entry in &entries {
        reg.register(tool_of(entry));
    }
    reg.build_embeddings().expect("fixture vectors");
    let index = crate::search::Bm25Index::build(
        entries
            .iter()
            .map(|e| (e.id.clone(), searchable_text(&tool_of(e))))
            .collect::<Vec<_>>(),
    );
    // The whole corpus, so nothing is truncated before the simulation sees it.
    let depth = entries.len();

    let mut seen: Vec<String> = Vec::new();
    let mut out = Vec::new();
    for turn in turns {
        if seen.iter().any(|q| q == &turn.query) {
            continue;
        }
        seen.push(turn.query.clone());
        let scored = |m: SearchMethod| -> Vec<(String, f32)> {
            reg.search_with_method(&turn.query, depth, Origin::Direct, m)
                .expect("arm search")
                .into_iter()
                .map(|h| (h.tool_id, h.score))
                .collect()
        };
        out.push(ArmRankings {
            query: turn.query.clone(),
            intent: turn.intent.clone(),
            bm25: scored(SearchMethod::Bm25),
            dense: scored(SearchMethod::Semantic),
            usage: g
                .arm(&turn.query, Some(&turn.vector), Capability::Tool, &|id| {
                    entries.iter().any(|e| e.id == id)
                })
                .into_arm()
                .map(|a| (a.ids.clone(), a.weight())),
            ceiling: index.query_ceiling(&turn.query),
        });
    }
    out
}

/// Median of an unsorted slice; `None` when empty. Used on cosine samples, where
/// a mean would be dragged by the long left tail of unrelated tools.
fn median(xs: &[f32]) -> Option<f32> {
    if xs.is_empty() {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(v[v.len() / 2])
}

/// One weighting under test: how it is labelled in the report, and the
/// `(bm25, dense)` weights it gives a query from that query's own arm scores.
pub(crate) struct Weighting {
    pub label: String,
    pub weights: ArmWeights,
    /// The dense weight this row sweeps, so the reference row is found by value.
    /// It used to be detected by a substring of `label`, which silently changed
    /// the table the moment the label was reworded.
    pub dense_weight: Option<f32>,
}

/// The `(bm25, dense)` weights one weighting gives a query.
type ArmWeights = Box<dyn Fn(&ArmRankings) -> (f32, f32)>;

impl Weighting {
    /// One dense weight; BM25 takes the remainder. The tuple's first slot is
    /// unused and kept only so `Weighting` stays one shape.
    fn dense(w: f32, label: &str) -> Self {
        Self {
            label: label.to_string(),
            dense_weight: Some(w),
            weights: Box::new(move |_| (1.0 - w, w)),
        }
    }
}

/// Recompute the fusion offline at one weighting and return each query's top-[`SERVED_K`].
///
/// Calls the engine's own [`rrf_fuse_weighted`] over the arms `served_arms`
/// collected, so the only thing varying across a sweep is the weights.
/// `the_offline_simulation_reproduces_the_real_served_order` pins it against the
/// live search path: a simulation that drifts from the fusion it predicts is
/// worse than no simulation, because it is still convincing.
pub(crate) fn simulate(arms: &[ArmRankings], w: &Weighting) -> Vec<(String, Vec<String>)> {
    arms.iter()
        .map(|a| {
            let (_, w_dense) = (w.weights)(a);
            let mut fused = score_fuse_at(&a.bm25, a.ceiling, &a.dense, a.usage_pair(), w_dense);
            fused.truncate(SERVED_K);
            (
                a.query.clone(),
                fused.into_iter().map(|(id, _)| id).collect(),
            )
        })
        .collect()
}

/// `score_fuse` at an arbitrary dense weight, so the sweep can vary what the
/// shipped call takes from the catalog's [`DenseWeight`].
///
/// This used to be a hand-copied reimplementation, which is how a sweep quietly
/// stops measuring the engine it claims to. Now that the weight is a parameter
/// it delegates, so the sweep cannot drift from the shipped rule.
fn score_fuse_at(
    bm25: &[(String, f32)],
    ceiling: f32,
    dense: &[(String, f32)],
    usage: Option<(&[String], f32)>,
    w_dense: f32,
) -> Vec<(String, f32)> {
    let weight = DenseWeight::new(w_dense).expect("sweep weights are in [0, 1]");
    crate::fusion::score_fuse(bm25, ceiling, dense, usage, weight)
}

/// Read-phrased queries whose top-1 is a write op — the reported failure,
/// counted the same way the served table counts it.
fn read_served_write(arms: &[ArmRankings], sim: &[(String, Vec<String>)]) -> usize {
    arms.iter()
        .zip(sim)
        .filter(|(a, (_, hits))| {
            is_read_intent(&a.intent) && hits.first().is_some_and(|h| is_write(h))
        })
        .count()
}

/// Served top-1 against the tool the turn actually invoked, and the reported
/// failure count beside it. The label is the only ground truth the fixture has;
/// the write-op count is a proxy that predates it.
pub(crate) fn served_quality(served: &[(String, Vec<String>)], turns: &[Turn]) -> (usize, usize) {
    let mut correct = 0;
    let mut write_on_read = 0;
    for (query, hits) in served {
        let Some(turn) = turns.iter().find(|t| &t.query == query) else {
            continue;
        };
        if hits.first() == Some(&turn.invoked) {
            correct += 1;
        }
        if is_read_intent(&turn.intent) && hits.first().is_some_and(|h| is_write(h)) {
            write_on_read += 1;
        }
    }
    (correct, write_on_read)
}

/// Ops that write. The reported failure was a read-phrased query being served
/// one of these.
fn is_write(id: &str) -> bool {
    id.starts_with("create_")
        || id.starts_with("update_")
        || id.starts_with("post_")
        || id.starts_with("add_")
        || id.starts_with("link_")
        || id.starts_with("complete_")
}

fn is_read_intent(intent: &str) -> bool {
    matches!(
        intent,
        "find" | "exists" | "count" | "filter" | "doc_read" | "doc_search"
    )
}

// ---- the report -------------------------------------------------------------

/// Run the 30 turns and write `tests/fixtures/harness-results.md`.
///
/// `cargo test -p ratel-ai-core --lib harness_report -- --ignored --nocapture`
#[test]
#[ignore]
fn harness_report() {
    let turns = turns();
    let (graph, records) = replay(&turns);
    let out = render(&turns, &graph, &records);
    print!("{out}");
    std::fs::write(
        format!(
            "{}/tests/fixtures/harness-results.md",
            env!("CARGO_MANIFEST_DIR")
        ),
        out,
    )
    .expect("write results");
}

fn render(turns: &[Turn], graph: &IntentGraph, records: &[TurnRecord]) -> String {
    use std::fmt::Write;
    let mut o = String::new();

    let _ = writeln!(o, "# Intent-graph harness — after R1\n");
    let _ = writeln!(
        o,
        "Generated by `cargo test -p ratel-ai-core --lib harness_report -- --ignored`.\n\
         Fixtures: `turns.json` ({} turns, {} distinct queries), `catalog.json` ({} ops).\n\n\
         **Absolute numbers mean little — only deltas between runs on this fixture do.** One\n\
         embedding model, two domains, labels assigned by one person, and the same turns both\n\
         build and evaluate the graph, so this measures whether a change altered behaviour, not\n\
         whether it generalises.\n",
        turns.len(),
        {
            let mut q: Vec<&str> = turns.iter().map(|t| t.query.as_str()).collect();
            q.sort_unstable();
            q.dedup();
            q.len()
        },
        catalog().len()
    );

    // -- shape ---------------------------------------------------------------
    let s = shape(graph, turns.len());
    let m = merge_quality(graph, turns);
    let _ = writeln!(
        o,
        "Compare against `harness-baseline.md`, the frozen pre-R1 capture, by diffing the two\n\
         files. Nothing here restates those numbers: a hardcoded summary goes stale the moment\n\
         the fixtures change, and says so silently.\n"
    );
    let _ = writeln!(o, "## Cluster shape\n");
    let _ = writeln!(o, "| metric | value |\n|---|---|");
    let _ = writeln!(o, "| clusters | {} |", s.clusters);
    let _ = writeln!(o, "| sizes | {:?} |", s.sizes);
    let _ = writeln!(o, "| largest share | {:.3} |", s.largest_share);
    let _ = writeln!(o, "| singleton rate | {:.3} |", s.singleton_rate);
    let _ = writeln!(o, "| purity | {:.3} |", purity(graph, turns));
    let _ = writeln!(o, "| merge precision | {:.3} |", m.precision);
    let _ = writeln!(o, "| merge recall | {:.3} |", m.recall);
    let _ = writeln!(o, "| merge F1 | {:.3} |", m.f1);

    // -- clusters ------------------------------------------------------------
    let _ = writeln!(o, "\n## Clusters\n");
    for it in &graph.intents {
        let _ = writeln!(
            o,
            "- **{}** · {} members · support {} · cohesion {:.3}",
            it.id,
            it.members.len(),
            it.support,
            it.cohesion
        );
        for member in &it.members {
            let label = turns
                .iter()
                .find(|t| &t.query == member)
                .map_or("?", |t| t.intent.as_str());
            let _ = writeln!(o, "  - `{label}` {member}");
        }
        let mut edges: Vec<(&String, &f32)> = it.tools.iter().collect();
        edges.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(std::cmp::Ordering::Equal));
        let _ = writeln!(
            o,
            "  - edges: {}",
            edges
                .iter()
                .map(|(id, w)| format!("{id}={w}"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // -- decisions -----------------------------------------------------------
    let _ = writeln!(o, "\n## Admission decisions\n");
    let _ = writeln!(
        o,
        "Per turn: the winning cluster's verdict, straight out of `dense_verdict` and\n\
         `Intent::coverage`. `cov` is `qualifying/required` and the fraction.\n"
    );
    let _ = writeln!(
        o,
        "| # | intent | query | outcome | best candidate | centroid cos | cov |\n|---|---|---|---|---|---|---|"
    );
    for (i, r) in records.iter().enumerate() {
        let best = r.verdicts.iter().filter(|v| v.admitted).max_by(|a, b| {
            a.centroid_cos
                .partial_cmp(&b.centroid_cos)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let (best_id, cos, cov) = match best {
            Some(v) => (
                format!(
                    "{} ({} members{})",
                    v.id,
                    v.members,
                    if v.covered { "" } else { ", centroid only" }
                ),
                format!("{:.3}", v.centroid_cos),
                v.coverage
                    .map_or("—".into(), |(q, req, f)| format!("{q}/{req} ({f:.2})")),
            ),
            None => ("none admitted".into(), "—".into(), "—".into()),
        };
        let outcome = match (&r.joined, r.seeded) {
            (Some(id), true) => format!("seeded {id}"),
            (Some(id), false) => format!("joined {id}"),
            (None, _) => "unclustered".into(),
        };
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {} | {} |",
            i + 1,
            r.intent,
            r.query,
            outcome,
            best_id,
            cos,
            cov
        );
    }

    // -- rejections that mattered -------------------------------------------
    let near: Vec<&TurnRecord> = records
        .iter()
        .filter(|r| {
            r.seeded
                && r.verdicts
                    .iter()
                    .any(|v| !v.admitted && v.centroid_cos >= 0.70)
        })
        .collect();
    let _ = writeln!(
        o,
        "\n**{} turns seeded a new cluster despite a cluster whose centroid alone would have \
         admitted them.** That gap is the bug R1 closed.\n",
        near.len()
    );

    // -- the calibration curve ----------------------------------------------
    let _ = writeln!(o, "\n## Policy sweep\n");
    let _ = writeln!(
        o,
        "The same 30 turns under a grid of policies. Cluster count alone cannot tell *split\n\
         correctly* from *shattered*, so read it against merge F1 and the singleton rate — a\n\
         setting that raises precision by shredding the graph shows up here as F1 falling.\n"
    );
    let _ = writeln!(
        o,
        "| similarity | coverage | clusters | largest | singletons | purity | precision | recall | F1 |\n\
         |---|---|---|---|---|---|---|---|---|"
    );
    for similarity in [0.60f32, 0.70, 0.80, 0.90] {
        for coverage in [0.3f32, 0.5, 0.7] {
            let policy = ClusterPolicy::default()
                .with_similarity(similarity)
                .with_coverage(coverage);
            let (swept, _) = replay_under(turns, policy);
            let sh = shape(&swept, turns.len());
            let mq = merge_quality(&swept, turns);
            let mark = if policy == ClusterPolicy::default() {
                " **(default)**"
            } else {
                ""
            };
            let _ = writeln!(
                o,
                "| {similarity:.2}{mark} | {coverage:.2} | {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
                sh.clusters,
                sh.largest_share,
                sh.singleton_rate,
                purity(&swept, turns),
                mq.precision,
                mq.recall,
                mq.f1
            );
        }
    }

    // The same turns replayed search-then-invoke, so every cluster carries what
    // its searches surfaced. Clustering is identical — same turns, same order,
    // same policy — so a cluster id from `graph` names the same cluster here.
    let with_impressions = replay_with_impressions(turns);
    // The real thing: `Intent::ranked` already applies the impression penalty, so
    // arming the impressions graph *is* the damped order. Recomputing the formula
    // here would measure a copy of it.
    let damped_order = |query: &str, vector: &[f32]| -> Vec<String> {
        with_impressions
            .arm(query, Some(vector), Capability::Tool, &|_| true)
            .into_arm()
            .map(|a| a.ids.into_iter().take(3).collect())
            .unwrap_or_default()
    };

    // -- what the arm contributed -------------------------------------------
    let _ = writeln!(o, "\n## Arm contribution\n");
    let _ = writeln!(
        o,
        "The cluster each query arms and the ids it promotes. Read this beside the served\n\
         table below: the arm can change completely without the served ranking moving, because\n\
         it enters the fusion at half weight against two full-weight arms.\n\
         \n\
         `raw order` is what invocation counts alone would promote, so the cluster-frequency\n\
         weight's effect is visible rather than only asserted. `damped` is the same arm over a\n\
         graph that recorded impressions, so the `passed_over` penalty applies — the left\n\
         column's graph recorded none, which is why the two can differ at all. `= (same)`\n\
         means unchanged from `promoted`.\n"
    );
    let _ = writeln!(
        o,
        "| intent | query | cluster | similarity | promoted | raw order | damped |\n\
         |---|---|---|---|---|---|---|"
    );
    let mut listed: Vec<&str> = Vec::new();
    let mut reordered = 0usize;
    let mut ratio_reordered = 0usize;
    let mut ratio_new_leader = 0usize;
    for turn in turns {
        if listed.iter().any(|q| *q == turn.query) {
            continue;
        }
        listed.push(&turn.query);
        let row = match graph
            .arm(&turn.query, Some(&turn.vector), Capability::Tool, &|_| true)
            .into_arm()
        {
            Some(a) => {
                // What these same edges would promote ordered by raw invocation
                // count alone, so the cluster-frequency weight's effect is
                // visible in the report rather than only asserted in a test.
                let raw: Vec<&str> = graph
                    .intents
                    .iter()
                    .find(|it| it.id == a.intent_id)
                    .map(|it| {
                        let mut e: Vec<(&String, &f32)> = it.tools.iter().collect();
                        e.sort_by(|x, y| {
                            y.1.partial_cmp(x.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                                .then_with(|| x.0.cmp(y.0))
                        });
                        e.into_iter().take(3).map(|(id, _)| id.as_str()).collect()
                    })
                    .unwrap_or_default();
                let promoted: Vec<&str> = a.ids.iter().take(3).map(String::as_str).collect();
                if raw != promoted {
                    reordered += 1;
                }
                let by_ratio = damped_order(&turn.query, &turn.vector);
                let by_ratio: Vec<&str> = by_ratio.iter().map(String::as_str).collect();
                if by_ratio != promoted {
                    ratio_reordered += 1;
                    // The one that can reach the fusion: the arm's rank-0 id is
                    // what a low-ranked capability is promoted past.
                    if by_ratio.first() != promoted.first() {
                        ratio_new_leader += 1;
                    }
                }
                format!(
                    "| {} | {} | {} | {:.3} | {} | {} | {} |",
                    turn.intent,
                    turn.query,
                    a.intent_id,
                    a.similarity,
                    promoted.join(", "),
                    if raw == promoted {
                        "= (same)".to_string()
                    } else {
                        raw.join(", ")
                    },
                    if by_ratio == promoted {
                        "= (same)".to_string()
                    } else {
                        by_ratio.join(", ")
                    }
                )
            }
            None => format!("| {} | {} | — | — | — | — | — |", turn.intent, turn.query),
        };
        let _ = writeln!(o, "{row}");
    }

    let _ = writeln!(
        o,
        "\n**{reordered} of {} arms are reordered by the cluster-frequency weight.** Watch this as\n\
         fixtures grow: at this size it behaves as a principled tie-break rather than a ranking\n\
         signal, and whether that changes is the question a bigger graph answers.\n\
         \n\
         **The impression penalty reorders {ratio_reordered} of {}, and changes which id leads\n\
         in {ratio_new_leader}.** Only the leader count can reach a served result: the arm enters\n\
         at half weight or less, so a swap further down its list is absorbed by two full-weight\n\
         arms. Whether those leaders survive the fusion is the served table below.\n",
        listed.len(),
        listed.len()
    );

    // -- what each arm knew and the fusion ignored ---------------------------
    let arms = served_arms(graph, turns);
    let served = served_topk(graph, turns);
    let served_top1 = |q: &str| -> Option<&str> {
        served
            .iter()
            .find(|(query, _)| query == q)
            .and_then(|(_, hits)| hits.first().map(String::as_str))
    };

    let _ = writeln!(o, "\n## Arm scores\n");
    let _ = writeln!(
        o,
        "Each arm's own top-1 and the score it was chosen on — the numbers `hybrid_search_traced`\n\
         computes and then discards at `.map(|(id, _)| id)`. Rank position is all that survives\n\
         into the fusion, so a cosine of 0.92 and one of 0.31 vote at identical strength.\n\
         \n\
         The roll-up below is the load-bearing measurement: if a low top cosine does **not**\n\
         separate the queries that are served a write op from the ones that are not, then\n\
         weighting the dense arm by that cosine cannot help, and no amount of tuning will\n\
         change it.\n"
    );
    let _ = writeln!(
        o,
        "| intent | query | bm25 top-1 | score | dense top-1 | cos | usage w | served top-1 |\n\
         |---|---|---|---|---|---|---|---|"
    );
    let mut bm25_bad = 0usize;
    let mut dense_bad = 0usize;
    let mut reads = 0usize;
    let mut cos_when_bad: Vec<f32> = Vec::new();
    let mut cos_when_ok: Vec<f32> = Vec::new();
    for a in &arms {
        let read = is_read_intent(&a.intent);
        if read {
            reads += 1;
            if a.bm25_top().is_some_and(|(id, _)| is_write(id)) {
                bm25_bad += 1;
            }
            if a.dense_top().is_some_and(|(id, _)| is_write(id)) {
                dense_bad += 1;
            }
            if let Some((_, cos)) = a.dense_top() {
                if served_top1(&a.query).is_some_and(is_write) {
                    cos_when_bad.push(cos);
                } else {
                    cos_when_ok.push(cos);
                }
            }
        }
        let (bid, bs) = a
            .bm25_top()
            .map_or(("—".into(), String::from("—")), |(id, s)| {
                (id.to_string(), format!("{s:.3}"))
            });
        let (did, ds) = a
            .dense_top()
            .map_or(("—".into(), String::from("—")), |(id, s)| {
                (id.to_string(), format!("{s:.3}"))
            });
        let _ = writeln!(
            o,
            "| {} | {} | {bid} | {bs} | {did} | {ds} | {} | {} |",
            a.intent,
            a.query,
            a.usage
                .as_ref()
                .map_or("—".to_string(), |(_, w)| format!("{w:.3}")),
            served_top1(&a.query).unwrap_or("—")
        );
    }
    let fmt = |m: Option<f32>| m.map_or("—".to_string(), |v| format!("{v:.3}"));
    let _ = writeln!(
        o,
        "\n**On the {reads} read-phrased queries, BM25's own top-1 was a write op {bm25_bad} times \
         and the dense arm's {dense_bad} times.**\n\
         \n\
         **Median dense top-1 cosine when the served top-1 was a write op: {} ({} queries); \
         when it was not: {} ({} queries).** A gap here is the evidence that the dense arm knows \
         when it is guessing; no gap means it does not, and R10 is dead on this fixture whatever \
         the ramp is set to.\n",
        fmt(median(&cos_when_bad)),
        cos_when_bad.len(),
        fmt(median(&cos_when_ok)),
        cos_when_ok.len(),
    );

    // -- how should the two content arms be split? ----------------------------
    let _ = writeln!(o, "\n## Fusion weight sweep\n");
    let _ = writeln!(
        o,
        "Hybrid fuses the two content arms on their **relevance scores**, so the split between\n\
         them is one number — `DenseWeight`, a per-catalog setting defaulting to 0.7. This\n\
         sweeps it, refusing the same arms offline through the same arithmetic the engine uses.\n\
         \n\
         `read→write` is the reported failure at top-1, lower is better; `correct` counts queries\n\
         whose top-1 is the tool the turn actually invoked — the fixture's only ground truth;\n\
         `moved` counts top-1s that differ from the default weight.\n"
    );
    let _ = writeln!(
        o,
        "| dense / bm25 | correct | read→write | moved |\n|---|---|---|---|"
    );
    let variants: Vec<Weighting> = vec![
        Weighting::dense(0.5, "0.5 dense / 0.5 bm25"),
        Weighting::dense(0.6, "0.6 / 0.4"),
        Weighting::dense(0.7, "0.7 / 0.3 **(default)**"),
        Weighting::dense(0.8, "0.8 / 0.2"),
        Weighting::dense(0.9, "0.9 / 0.1"),
        Weighting::dense(1.0, "1.0 dense only"),
    ];
    let reference = simulate(
        &arms,
        &Weighting::dense(SCORE_FUSION_DENSE_WEIGHT, "shipped"),
    );
    for v in &variants {
        let sim = simulate(&arms, v);
        let moved = sim
            .iter()
            .zip(&reference)
            .filter(|((_, a), (_, b))| a.first() != b.first())
            .count();
        let correct = sim
            .iter()
            .filter(|(q, hits)| {
                turns
                    .iter()
                    .find(|t| &t.query == q)
                    .is_some_and(|t| hits.first() == Some(&t.invoked))
            })
            .count();
        let _ = writeln!(
            o,
            "| {} | {} of {} | {} | {} |",
            v.label,
            correct,
            sim.len(),
            read_served_write(&arms, &sim),
            if v.dense_weight == Some(SCORE_FUSION_DENSE_WEIGHT) {
                "\u{2014}".to_string()
            } else {
                format!("{moved} of {}", arms.len())
            },
        );
    }
    let _ = writeln!(
        o,
        "\n**`correct` is the column that decides this**, and `read→write` is a proxy that predates\n\
         it. A weight that moves many top-1s without raising `correct` has changed the ranking,\n\
         not improved it.\n"
    );

    // -- what the length penalty costs -----------------------------------------
    let _ = writeln!(o, "\n## BM25 length penalty (`b`)\n");
    let _ = writeln!(
        o,
        "The lexical arm alone, over the same 47 queries, at the old `b = 0.4` and the standard\n\
         `b = 0.75` it was raised to. `b` scales how hard a long document is\n\
         penalised: at 0 length is ignored, at 1 it is fully normalised.\n\
         \n\
         `top-1 correct` counts the queries whose BM25 top-1 is the tool the turn actually\n\
         invoked. `write on read` counts read-phrased queries whose BM25 top-1 is a write op.\n"
    );
    let _ = writeln!(
        o,
        "| b | top-1 correct | write on read | top-1 changed |\n|---|---|---|---|"
    );
    {
        let docs: Vec<(String, String)> = catalog()
            .iter()
            .map(|e| (e.id.clone(), searchable_text(&tool_of(e))))
            .collect();
        let mut queries: Vec<&Turn> = Vec::new();
        for t in turns {
            if !queries.iter().any(|q| q.query == t.query) {
                queries.push(t);
            }
        }
        let mut reference: Vec<String> = Vec::new();
        for b in [0.4f32, 0.75] {
            let index =
                crate::search::Bm25Index::build_with(docs.clone(), crate::search::BM25_K1, b);
            let tops: Vec<String> = queries
                .iter()
                .map(|t| {
                    index
                        .search(&t.query, 1)
                        .first()
                        .map(|(id, _)| id.clone())
                        .unwrap_or_default()
                })
                .collect();
            let correct = queries
                .iter()
                .zip(&tops)
                .filter(|(t, top)| &t.invoked == *top)
                .count();
            let bad = queries
                .iter()
                .zip(&tops)
                .filter(|(t, top)| is_read_intent(&t.intent) && is_write(top))
                .count();
            let changed = if reference.is_empty() {
                "—".to_string()
            } else {
                format!(
                    "{} of {}",
                    tops.iter().zip(&reference).filter(|(a, b)| a != b).count(),
                    tops.len()
                )
            };
            let mark = if (b - crate::search::BM25_B).abs() < f32::EPSILON {
                " **(current)**"
            } else {
                ""
            };
            let _ = writeln!(
                o,
                "| {b:.2}{mark} | {correct} of {} | {bad} of {} | {changed} |",
                queries.len(),
                queries.iter().filter(|t| is_read_intent(&t.intent)).count()
            );
            if reference.is_empty() {
                reference = tops;
            }
        }
    }
    let _ = writeln!(
        o,
        "\nThe lexical arm alone is weak either way — it is one of three, and the served numbers\n\
         below are what fusion makes of it.\n"
    );

    // -- the relevance score, per method -------------------------------------
    let _ = writeln!(o, "\n## Relevance scores\n");
    let _ = writeln!(
        o,
        "`SearchHit::relevance` for the top {SERVED_K} of each method, on a query where the arms\n\
         disagree. Semantic is `(cos + 1) / 2`; BM25 is `score / Σ idf(query terms)`, clamped;\n\
         hybrid is the fused score itself, already absolute in `[0, 1]` (ADR-0024), so the\n\
         raw and relevance columns are equal.\n\
         \n\
         All three are absolute — they compare across queries and do not move when\n\
         `top_k` does. Read the BM25 column's ceiling: no tool exceeds 0.52 because\n\
         `authent` carries the query's largest IDF and appears in no document, so half the\n\
         query's discriminating mass is unanswerable by this catalog. That is the number\n\
         saying so.\n\
         \n\
         The hybrid column no longer pins 1.00 at the top. Under the rank fusion this\n\
         replaced it did, whatever the query matched, which is what made the number\n\
         undisplayable; here the best hit scores what it actually earned.\n"
    );
    let _ = writeln!(
        o,
        "| method | # | tool | raw | relevance |\n|---|---|---|---|---|"
    );
    {
        use crate::method::SearchMethod;
        use crate::trace::Origin;

        // No graph attached, deliberately. With one, the usage arm fuses into
        // every method, `score` becomes RRF, and all three rows normalize
        // min-max — the affine cosine rule would never appear. See the note
        // below: that is what an integrator with adaptive ranking on gets.
        let mut reg = crate::ToolRegistry::with_embedder_for_test(Arc::new(FixtureEmbedder::new()));
        for entry in catalog() {
            reg.register(tool_of(&entry));
        }
        reg.build_embeddings().expect("fixture vectors");

        let q = "find tasks related to authentication";
        for (label, method) in [
            ("bm25", SearchMethod::Bm25),
            ("semantic", SearchMethod::Semantic),
            ("hybrid", SearchMethod::Hybrid),
        ] {
            for (i, h) in reg
                .search_with_method(q, SERVED_K, Origin::Direct, method)
                .expect("search")
                .iter()
                .enumerate()
            {
                let _ = writeln!(
                    o,
                    "| {label} | {} | {} | {:.4} | {:.3} |",
                    i + 1,
                    h.tool_id,
                    h.score,
                    h.relevance
                );
            }
        }
        let _ = writeln!(
            o,
            "\nQuery: `{q}` \u{b7} invoked: `{}`\n\n\
             **The affine rule reaches almost nobody.** These rows have no intent graph \
             attached. With adaptive ranking on, the usage arm fuses into the Bm25 and \
             Semantic paths too, `score` becomes RRF, and all three methods fall back to \
             min-max. The one rule that preserves an absolute level applies only when \
             adaptive ranking is off.\n",
            turns
                .iter()
                .find(|t| t.query == q)
                .map_or("?", |t| t.invoked.as_str())
        );
    }

    // -- what was shown, against what was used --------------------------------
    let _ = writeln!(o, "\n## Impressions\n");
    let _ = writeln!(
        o,
        "`Intent::surfaced` against `Intent::tools`, per cluster: how many of a cluster's\n\
         searches put each tool in front of the caller, beside how many invoked it. Recorded\n\
         only; nothing in ranking reads it.\n\
         \n\
         `ratio` is `(invoked + 1) / (surfaced + 2)` — Laplace-smoothed, so a tool shown once\n\
         and invoked once does not outrank one shown fifty times and invoked forty. Read it\n\
         against the `invoked` column, which is the order the arm actually serves today: where\n\
         the two disagree is where a tool is riding on volume rather than on answering the\n\
         question.\n"
    );
    let _ = writeln!(
        o,
        "| cluster | tool | surfaced | invoked | ratio |\n|---|---|---|---|---|"
    );
    let mut shown_never_used = 0usize;
    let mut disagreements = 0usize;
    for it in &with_impressions.intents {
        // Every id either arm knows about, so a tool shown and never invoked is
        // visible rather than silently absent — that is the whole point.
        let mut ids: Vec<&String> = it.surfaced_tools.keys().chain(it.tools.keys()).collect();
        ids.sort_unstable();
        ids.dedup();
        let ratio = |id: &str| {
            let s = it.surfaced_tools.get(id).copied().unwrap_or(0) as f32;
            let i = it.tools.get(id).copied().unwrap_or(0.0);
            (i + 1.0) / (s + 2.0)
        };
        let mut rows: Vec<&String> = ids.clone();
        rows.sort_by(|a, b| {
            ratio(b)
                .partial_cmp(&ratio(a))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        });
        let by_invoked = {
            let mut v: Vec<&String> = ids.clone();
            v.sort_by(|a, b| {
                it.tools
                    .get(*b)
                    .unwrap_or(&0.0)
                    .partial_cmp(it.tools.get(*a).unwrap_or(&0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.cmp(b))
            });
            v
        };
        if rows.first() != by_invoked.first() {
            disagreements += 1;
        }
        for id in &rows {
            let s = it.surfaced_tools.get(*id).copied().unwrap_or(0);
            let i = it.tools.get(*id).copied().unwrap_or(0.0);
            if i == 0.0 {
                shown_never_used += 1;
            }
            let _ = writeln!(o, "| {} | {id} | {s} | {i:.0} | {:.3} |", it.id, ratio(id));
        }
    }
    let _ = writeln!(
        o,
        "\n**{shown_never_used} (cluster, tool) pairs were surfaced and never invoked** — evidence \
         that exists nowhere in `tools`, because only invocations write edges.\n\
         \n\
         **The ratio would lead a different tool in {disagreements} of {} clusters.** That count \
         is the whole question: at zero, ranking on impressions changes nothing and is not worth \
         the risk; the larger it is, the more the current order is volume rather than fit — and \
         the more a bad ratio could do damage. Neither reading is available from invocation \
         counts alone.\n",
        with_impressions.intents.len()
    );

    // -- does the penalty reach a served result? ------------------------------
    let _ = writeln!(o, "\n## Served, with and without the impression penalty\n");
    let _ = writeln!(
        o,
        "The same 47 queries served end to end through all three arms, over two graphs grown\n\
         from the same turns: one that recorded no impressions, so `passed_over` is `1.0`\n\
         everywhere, and one that did.\n\
         \n\
         `top-1 correct` is against the tool the turn actually invoked — the fixture's only\n\
         ground truth. `write on read` is the reported failure, and a proxy. Both must be read\n\
         together: a change that fixes the proxy while lowering accuracy has moved the ranking,\n\
         not improved it.\n"
    );
    let _ = writeln!(
        o,
        "| graph | top-1 correct | write on read | top-1 changed |\n|---|---|---|---|"
    );
    let plain = served_topk(graph, turns);
    let damped = served_topk(&with_impressions, turns);
    let reads = turns
        .iter()
        .filter(|t| is_read_intent(&t.intent))
        .map(|t| t.query.clone())
        .collect::<std::collections::HashSet<_>>()
        .len();
    let moved = plain
        .iter()
        .zip(&damped)
        .filter(|((_, a), (_, b))| a.first() != b.first())
        .count();
    for (label, served, changed) in [
        ("no impressions", &plain, "—".to_string()),
        (
            "with the penalty",
            &damped,
            format!("{moved} of {}", plain.len()),
        ),
    ] {
        let (correct, bad) = served_quality(served, turns);
        let _ = writeln!(
            o,
            "| {label} | {correct} of {} | {bad} of {reads} | {changed} |",
            served.len()
        );
    }

    // -- served top-k --------------------------------------------------------
    let mut write_to_read = 0usize;
    let _ = writeln!(
        o,
        "## Served top-{SERVED_K} (hybrid: BM25 + dense + usage arm)\n"
    );
    let _ = writeln!(o, "| intent | query | top-{SERVED_K} |\n|---|---|---|");
    for (query, hits) in &served {
        let intent = turns
            .iter()
            .find(|t| &t.query == query)
            .map_or("?", |t| t.intent.as_str());
        if is_read_intent(intent) && hits.first().is_some_and(|h| is_write(h)) {
            write_to_read += 1;
        }
        let _ = writeln!(o, "| {intent} | {query} | {} |", hits.join(", "));
    }
    let _ = writeln!(
        o,
        "\n**Read-phrased queries served a write op at top-1: {write_to_read} of {}.** This is the \
         reported failure, measured end to end.\n",
        served
            .iter()
            .filter(|(q, _)| turns
                .iter()
                .find(|t| &t.query == q)
                .is_some_and(|t| is_read_intent(&t.intent)))
            .count()
    );

    o
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cross-check that the roll-up reads the rule rather than
    /// reimplementing it: restricted to the original twelve incident queries,
    /// the harness must reproduce what we measured directly — seven clusters,
    /// largest four. If this drifts, nothing the harness reports is trustworthy.
    #[test]
    fn the_harness_reproduces_the_known_twelve_query_result() {
        let twelve: Vec<Turn> = turns().into_iter().take(12).collect();
        let (g, _) = replay(&twelve);
        let s = shape(&g, twelve.len());
        assert_eq!(s.clusters, 7, "sizes {:?}", s.sizes);
        assert_eq!(s.sizes.first(), Some(&4), "sizes {:?}", s.sizes);
    }

    /// Every vector the harness will ask for must be in a fixture — a missing
    /// one panics mid-run, which is a worse failure than a fast assertion.
    #[test]
    fn every_doc_vector_was_embedded_from_the_current_projection() {
        // The keys cannot catch this: both the map and the lookup are built by
        // the same `searchable_text`, so they agree by construction even when the
        // stored vector describes text that no longer exists. Changing the
        // projection without regenerating would otherwise measure the old one.
        for entry in catalog() {
            let now = searchable_text(&tool_of(&entry));
            assert_eq!(
                entry.projection, now,
                "{} was embedded from a different projection — regenerate the \
                 harness fixtures",
                entry.id
            );
        }
    }

    /// The sweep's whole authority rests on this. It recomputes fusion from arms
    /// collected out-of-band, so nothing but this test stops it from quietly
    /// predicting a ranking the engine does not produce — and a sweep that is
    /// wrong but plausible is worse than none, because it still gets acted on.
    #[test]
    fn the_offline_simulation_reproduces_the_real_served_order() {
        let turns = turns();
        let (graph, _) = replay(&turns);
        let arms = served_arms(&graph, &turns);
        let shipped = Weighting::dense(SCORE_FUSION_DENSE_WEIGHT, "shipped");

        assert_eq!(
            simulate(&arms, &shipped),
            served_topk(&graph, &turns),
            "the simulation at the shipped weight must be the live hybrid path, query \
             for query and id for id"
        );
    }

    #[test]
    fn the_fixtures_cover_every_lookup_the_harness_makes() {
        let e = FixtureEmbedder::new();
        for turn in turns() {
            e.embed_query(&turn.query).expect("query vector");
        }
        for entry in catalog() {
            e.embed_doc(&searchable_text(&tool_of(&entry)))
                .expect("doc vector");
        }
    }
}
