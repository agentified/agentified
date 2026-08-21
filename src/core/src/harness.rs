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

use crate::embedding::{Embedded, Embedder, EmbedderError};
use crate::indexing::searchable_text;
use crate::tool::Tool;
use crate::trace::NoopSink;
use crate::usage::{Capability, ClusterPolicy, Intent, IntentGraph, dense_verdict};

const TURNS: &str = include_str!("../tests/fixtures/turns.json");
const CATALOG: &str = include_str!("../tests/fixtures/catalog.json");

/// Every turn stamps the same timestamp, so `recency_factor` is 1.0 and nothing
/// is evicted — the graph is a pure function of the fixture, not of wall clock.
const T0: u64 = 1_753_000_000_000;

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

        g.note_query_vector(&turn.query, &turn.vector, "fixture");
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
            vector: Vec::new(),
        });
        let v = embedder
            .embed_doc(&searchable_text(&tool))
            .expect("embed doc");
        entry["vector"] = serde_json::json!(v);
    }
    write_fixture(
        "catalog.json",
        &doc,
        "tools",
        &["id", "description", "input_schema"],
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

/// Served top-k through the real fusion — BM25, dense and the usage arm.
pub(crate) fn served_top3(g: &IntentGraph, turns: &[Turn]) -> Vec<(String, Vec<String>)> {
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
            .search_with_method(&turn.query, 3, Origin::Direct, SearchMethod::Hybrid)
            .expect("hybrid search");
        assert!(
            hits.first().is_none_or(|h| h.fused),
            "the usage arm must contribute, or the served numbers below measure \
             only BM25 and dense and the comparison is vacuous"
        );
        out.push((
            turn.query.clone(),
            hits.into_iter().map(|h| h.tool_id).collect(),
        ));
    }
    out
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

    // -- what the arm contributed -------------------------------------------
    let _ = writeln!(o, "\n## Arm contribution\n");
    let _ = writeln!(
        o,
        "The cluster each query arms and the ids it promotes. Read this beside the served\n\
         table below: the arm can change completely without the served ranking moving, because\n\
         it enters the fusion at half weight against two full-weight arms.\n"
    );
    let _ = writeln!(
        o,
        "| intent | query | cluster | similarity | promoted | raw order |\n|---|---|---|---|---|---|"
    );
    let mut listed: Vec<&str> = Vec::new();
    let mut reordered = 0usize;
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
                format!(
                    "| {} | {} | {} | {:.3} | {} | {} |",
                    turn.intent,
                    turn.query,
                    a.intent_id,
                    a.similarity,
                    promoted.join(", "),
                    if raw == promoted {
                        "= (same)".to_string()
                    } else {
                        raw.join(", ")
                    }
                )
            }
            None => format!("| {} | {} | — | — | — | — |", turn.intent, turn.query),
        };
        let _ = writeln!(o, "{row}");
    }

    let _ = writeln!(
        o,
        "\n**{reordered} of {} arms are reordered by the cluster-frequency weight.** Watch this as\n\
         fixtures grow: at this size it behaves as a principled tie-break rather than a ranking\n\
         signal, and whether that changes is the question a bigger graph answers.\n",
        listed.len()
    );

    // -- served top-k --------------------------------------------------------
    let served = served_top3(graph, turns);
    let mut write_to_read = 0usize;
    let _ = writeln!(o, "## Served top-3 (hybrid: BM25 + dense + usage arm)\n");
    let _ = writeln!(o, "| intent | query | top-3 |\n|---|---|---|");
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
