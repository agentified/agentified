use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{Fact, Skill, Tool};

/// Catalog entry type carried by [`TraceEvent::CatalogDefinition`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogKind {
    /// An executable tool definition.
    Tool,
    /// An on-demand skill definition.
    Skill,
    /// A grounding fact definition.
    Fact,
}

#[derive(Serialize)]
struct CatalogDefinitionContent<'a> {
    kind: CatalogKind,
    id: &'a str,
    name: &'a str,
    description: &'a str,
    tags: &'a [String],
    input_schema: Option<&'a serde_json::Value>,
    output_schema: Option<&'a serde_json::Value>,
    searchable_description: &'a str,
    searchable_description_overridden: bool,
}

/// Where a search came from. Trace consumers separate the paths: rerankers
/// train on agent calls, the inspector shows all of them, and offline graph
/// construction reads only the baseline ones.
///
/// **Non-exhaustive**, for the same reason [`TraceEvent`] is: the set of
/// origins grows as Ratel learns to sit in more places, and a downstream
/// `match` acquiring a `_ =>` arm once is cheaper than a breaking release per
/// variant. Constructing existing variants is unaffected; only exhaustive
/// matches need the arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Origin {
    /// A direct API call — SDK helpers, library callers, benchmarks. Wire
    /// value `direct`.
    Direct,
    /// A call the agent synthesized inside its loop, via the capability
    /// tools. Wire value `agent`.
    Agent,
    /// A query recorded while Ratel was **observing but not serving**: the host
    /// captured the turn's text so the invocations that follow can be
    /// attributed to it, while the agent chose from its own full tool list.
    /// Wire value `baseline`.
    ///
    /// Ratel's own search path never produces this — it is written by a host
    /// running a baseline capture, and it is what marks an observation as
    /// unbiased evidence rather than something the ranker influenced.
    Baseline,
}

/// How a registry corpus changed — carried by [`TraceEvent::IndexChurn`]
/// (tools), [`TraceEvent::SkillChurn`] (skills), and [`TraceEvent::FactChurn`]
/// (facts).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChurnKind {
    /// An item was registered — including a replace-in-place re-register of
    /// an existing id. Wire value `add`.
    Add,
    /// An item was removed from the corpus. Wire value `remove`.
    Remove,
}

/// Outcome of the one-time embedding-model load. `Slow` flags a machine that may
/// be underpowered for the model; `Failed` a load that errored (network, cache,
/// corrupt weights).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbedderLoadStatus {
    /// The model loaded within the expected budget. Wire value `ok`.
    Ok,
    /// The model loaded, but slowly — the machine may be underpowered for it.
    /// Wire value `slow`.
    Slow,
    /// The load errored (network, cache, corrupt weights); the accompanying
    /// `reason` carries the error. Wire value `failed`.
    Failed,
}

/// One ranked tool hit inside a [`TraceEvent::Search`] event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchHitTrace {
    /// Id of the matching tool.
    pub tool_id: String,
    /// The engine score, widened to `f64` — same per-method semantics as
    /// [`crate::SearchHit::score`].
    pub score: f64,
}

/// Timing and top score of one engine stage of a search. BM25 searches emit
/// one `bm25` stage, semantic searches one `dense` stage; hybrid emits
/// `bm25`, `dense`, and `rrf`, in that order. Semantic and hybrid searches
/// that short-circuit on an empty corpus or `top_k == 0` emit no stages.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchStage {
    /// Stage name: `"bm25"`, `"dense"`, or `"rrf"`.
    pub name: String,
    /// Stage wall time, in milliseconds.
    pub took_ms: u64,
    /// Best score the stage produced (that stage's scale); `None` when it
    /// returned no hits.
    pub top_score: Option<f64>,
}

/// One ranked skill hit inside a [`TraceEvent::SkillSearch`] event — the
/// skill-side twin of [`SearchHitTrace`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillHitTrace {
    /// Id of the matching skill.
    pub skill_id: String,
    /// The engine score, widened to `f64` — same per-method semantics as
    /// [`crate::SkillHit::score`].
    pub score: f64,
}

/// One ranked fact hit inside a [`TraceEvent::FactSearch`] event — the
/// fact-side twin of [`SkillHitTrace`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FactHitTrace {
    /// Id of the matching fact.
    pub fact_id: String,
    /// The engine score, widened to `f64` — same per-method semantics as
    /// [`crate::FactHit::score`].
    pub score: f64,
}

/// Why a fact's body was (re-)injected into the context, carried by
/// [`TraceEvent::FactInject`]. The grounding layer decides this by scanning
/// the transcript for the fact's own body text (content presence); it is the
/// observable half of the re-injection freshness gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FactInjectReason {
    /// Not present in the transcript and never injected this session — a
    /// first injection. Wire value `never`.
    Never,
    /// Injected earlier but its body is gone from the window now (trimmed /
    /// compacted out), so it is re-injected. Wire value `evicted`.
    Evicted,
    /// The registered body changed since it was injected (the current body is
    /// absent and differs from the one last injected), so the new version is
    /// injected. Wire value `mutated`.
    Mutated,
}

/// Every event produced by any layer of Ratel. New variants are additive;
/// renames or removals are breaking — see ADR-0007.
///
/// On the wire each event is a JSON object whose `type` tag is the variant
/// name in snake_case (`IndexChurn` → `index_churn`), with the variant's
/// fields flattened beside it; sinks wrap it in a [`TraceEnvelope`]. All
/// `took_ms` fields are wall time in milliseconds.
///
/// `#[non_exhaustive]` is what makes "new variants are additive" *true* rather
/// than aspirational: it requires downstream `match`es to carry a `_ =>` arm, so
/// a future event variant lands there instead of breaking their compile. Two
/// axes, only the first mechanical:
///
/// - **New variant** → non-breaking, enforced here.
/// - **New field on an existing variant** → non-breaking only if consumers
///   destructure with a trailing `..` (as this crate always does); variant-level
///   non-exhaustiveness is intentionally *not* used, since it would also block
///   downstream from constructing events by literal.
///
/// Renames and removals are breaking on both axes.
///
/// ```
/// use ratel_ai_core::TraceEvent;
/// // A downstream matcher must include `_ =>`, and is then future-proof:
/// fn kind(e: &TraceEvent) -> &str {
///     match e {
///         TraceEvent::Search { .. } => "search",
///         _ => "other",
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum TraceEvent {
    /// A catalog definition was registered or materially changed. Content is
    /// present on the local stream; SDKs project it to the opt-in
    /// `ratel.catalog.definition` Logs EventRecord.
    CatalogDefinition {
        /// Catalog entry type.
        kind: CatalogKind,
        /// Stable catalog entry id.
        id: String,
        /// Model-facing display/callable name.
        name: String,
        /// Model-facing description.
        description: String,
        /// Search tags; empty for tools.
        tags: Vec<String>,
        /// Tool input JSON Schema; absent for skills and facts.
        input_schema: Option<serde_json::Value>,
        /// Tool output JSON Schema; absent for skills and facts.
        output_schema: Option<serde_json::Value>,
        /// Effective searchable description after applying the optional override.
        searchable_description: String,
        /// Whether the effective searchable description came from an override.
        searchable_description_overridden: bool,
        /// Lowercase SHA-256 of the definition fields above.
        content_hash: String,
    },
    /// A [`crate::ToolRegistry`] search completed (any [`crate::SearchMethod`]).
    /// Carries the query, the requested `top_k`, the ranked `hits` with
    /// scores, the per-engine `stages` timings, and the total wall time.
    Search {
        /// The search text.
        query: String,
        /// Direct library call vs agent-synthesized.
        origin: Origin,
        /// Requested result count.
        top_k: u32,
        /// The ranked results, best-first.
        hits: Vec<SearchHitTrace>,
        /// Per-engine stage timings (`bm25` / `dense` / `rrf`).
        stages: Vec<SearchStage>,
        /// Total search wall time, in milliseconds.
        took_ms: u64,
    },
    /// The tool corpus changed: [`crate::ToolRegistry::register`] emits this
    /// with [`ChurnKind::Add`] for both a fresh registration and a
    /// replace-in-place re-register.
    IndexChurn {
        /// Whether the id was added or removed.
        kind: ChurnKind,
        /// Id of the affected tool.
        tool_id: String,
    },
    /// A [`crate::SkillRegistry`] search completed — the skill-side twin of
    /// [`TraceEvent::Search`], with the same shape.
    SkillSearch {
        /// The search text.
        query: String,
        /// Direct library call vs agent-synthesized.
        origin: Origin,
        /// Requested result count.
        top_k: u32,
        /// The ranked results, best-first.
        hits: Vec<SkillHitTrace>,
        /// Per-engine stage timings (`bm25` / `dense` / `rrf`).
        stages: Vec<SearchStage>,
        /// Total search wall time, in milliseconds.
        took_ms: u64,
    },
    /// The skill corpus changed — the skill-side twin of
    /// [`TraceEvent::IndexChurn`]. [`crate::SkillRegistry::register`] emits
    /// [`ChurnKind::Add`] only; [`crate::SkillRegistry::replace_all`] emits
    /// either kind, and is the only source of [`ChurnKind::Remove`] for skills.
    SkillChurn {
        /// Whether the id was added or removed.
        kind: ChurnKind,
        /// Id of the affected skill.
        skill_id: String,
    },
    /// A skill's body was loaded for dispatch (the `get_skill_content` path).
    /// Emitted by the SDK skill catalogs via
    /// [`crate::SkillRegistry::record_event`].
    SkillInvoke {
        /// Id of the loaded skill.
        skill_id: String,
        /// Load wall time, in milliseconds.
        took_ms: u64,
    },
    /// A [`crate::FactRegistry`] search completed — the fact-side twin of
    /// [`TraceEvent::SkillSearch`], with the same shape.
    FactSearch {
        /// The search text.
        query: String,
        /// Direct library call vs agent-synthesized.
        origin: Origin,
        /// Requested result count.
        top_k: u32,
        /// The ranked results, best-first.
        hits: Vec<FactHitTrace>,
        /// Per-engine stage timings (`bm25` / `dense` / `rrf`).
        stages: Vec<SearchStage>,
        /// Total search wall time, in milliseconds.
        took_ms: u64,
    },
    /// The fact corpus changed — the fact-side twin of
    /// [`TraceEvent::SkillChurn`], emitted by [`crate::FactRegistry::register`].
    FactChurn {
        /// Whether the id was added or removed.
        kind: ChurnKind,
        /// Id of the affected fact.
        fact_id: String,
    },
    /// A fact's body was injected into the context by the grounding layer.
    /// Emitted by the SDK via [`crate::FactRegistry::record_event`]; `reason`
    /// records why the re-injection freshness gate let it through.
    FactInject {
        /// Id of the injected fact.
        fact_id: String,
        /// Why it was (re-)injected this turn.
        reason: FactInjectReason,
    },
    /// A fact was *not* re-injected because it is still fresh in the context —
    /// the token-saving half of the freshness gate, surfaced so the saving is
    /// observable. Emitted by the SDK via
    /// [`crate::FactRegistry::record_event`].
    FactInjectSkip {
        /// Id of the fact that was already present and left alone.
        fact_id: String,
    },
    /// A fact rode along in a per-call grounding snapshot — the stateless
    /// `groundSnapshot` path: recomputed each call, nothing persisted, no
    /// freshness gate. The per-call twin of [`TraceEvent::FactInject`], emitted
    /// by the SDK via [`crate::FactRegistry::record_event`] once per fact per
    /// snapshot.
    FactSnapshot {
        /// Id of the fact included in the snapshot.
        fact_id: String,
    },
    /// A tool invocation began. Emitted by the SDK catalogs just before the
    /// tool's executor runs; paired with [`TraceEvent::InvokeEnd`] or
    /// [`TraceEvent::InvokeError`].
    InvokeStart {
        /// Id of the invoked tool.
        tool_id: String,
        /// Size of the serialized argument payload, in bytes.
        args_size_bytes: u64,
    },
    /// A tool invocation completed successfully.
    InvokeEnd {
        /// Id of the invoked tool.
        tool_id: String,
        /// Invocation wall time, in milliseconds.
        took_ms: u64,
    },
    /// A tool invocation failed; `error` carries the executor's message.
    InvokeError {
        /// Id of the invoked tool.
        tool_id: String,
        /// Wall time until the failure, in milliseconds.
        took_ms: u64,
        /// The failure message.
        error: String,
    },
    /// The agent searched the catalog through the capability tools
    /// (`search_capabilities`, or the deprecated `search_tools`). Carries only
    /// the hit *count*; the ranked list with scores is on the underlying
    /// [`TraceEvent::Search`] / [`TraceEvent::SkillSearch`] the registries
    /// emit for the same call. The `gateway_*` wire prefix is frozen
    /// (ADR-0007: renames are breaking).
    GatewaySearch {
        /// The search text.
        query: String,
        /// Direct library call vs agent-synthesized.
        origin: Origin,
        /// Requested result count.
        top_k: u32,
        /// Number of results returned.
        hits: u32,
        /// Total search wall time, in milliseconds.
        took_ms: u64,
    },
    /// The agent invoked a tool through the `invoke_tool` capability tool and
    /// it succeeded.
    GatewayInvoke {
        /// Id of the invoked tool.
        tool_id: String,
        /// Invocation wall time, in milliseconds.
        took_ms: u64,
    },
    /// A capability-tool call failed: an unknown tool/skill id, an executor
    /// error, or an upstream that needs auth.
    GatewayError {
        /// Id of the tool (or skill) the call named.
        tool_id: String,
        /// The failure message (e.g. `needs_auth`).
        error: String,
    },
    /// An upstream MCP server's tools were ingested into the catalog
    /// (the SDK's `register_mcp_server`).
    UpstreamRegister {
        /// Upstream server name.
        server: String,
        /// Transport used to reach it (e.g. `stdio` / `http` / `sse`).
        transport: String,
        /// Number of tools ingested.
        tool_count: u32,
    },
    /// A proxied call to a tool backed by an upstream MCP server completed.
    UpstreamInvoke {
        /// Upstream server name.
        server: String,
        /// Id of the invoked tool.
        tool_id: String,
        /// Invocation wall time, in milliseconds.
        took_ms: u64,
    },
    /// A proxied upstream call failed; `error` carries the upstream's message.
    UpstreamError {
        /// Upstream server name.
        server: String,
        /// Id of the invoked tool.
        tool_id: String,
        /// The failure message.
        error: String,
    },
    /// A credential refresh for an upstream MCP server was attempted.
    AuthRefresh {
        /// Upstream server name.
        upstream: String,
        /// Whether the refresh produced valid credentials.
        ok: bool,
    },
    /// An upstream MCP server challenged for auth (e.g. a 401): user
    /// interaction is required before its tools work.
    AuthNeeds {
        /// Upstream server name.
        upstream: String,
    },
    /// An interactive auth flow (e.g. OAuth) started for an upstream MCP
    /// server; paired with [`TraceEvent::AuthFlowEnd`].
    AuthFlowStart {
        /// Upstream server name.
        upstream: String,
    },
    /// The interactive auth flow ended.
    AuthFlowEnd {
        /// Upstream server name.
        upstream: String,
        /// Whether the flow produced valid credentials.
        ok: bool,
    },
    /// One fan-out subscriber lost events because its bounded queue overflowed.
    EventsDropped {
        /// Number of events dropped during this observation window.
        dropped_count: u64,
        /// Stable machine-readable loss reason; currently `queue_overflow`.
        reason: String,
        /// Timestamp of the first drop in this report, in Unix milliseconds.
        window_start_ts: u64,
        /// Timestamp of the last drop in this report, in Unix milliseconds.
        window_end_ts: u64,
    },
    /// Emitted once, on the first (cold) load of the embedding model. `status`
    /// flags a slow load (possibly underpowered machine) or a failed one;
    /// `reason` carries the hint / error. See `embedding.rs` and ADR-0011.
    EmbedderLoad {
        /// Resolved model display name: repo id, local path, or endpoint model
        /// and URL.
        model: String,
        /// Load outcome: ok, slow, or failed.
        status: EmbedderLoadStatus,
        /// Load wall time, in milliseconds (`0` when the load failed before
        /// timing).
        took_ms: u64,
        /// The slow-load hint or the load error; `None` on a normal load.
        reason: Option<String>,
    },
    /// Emitted once when a configured embedding model is actually downloaded to
    /// the HuggingFace cache (a cold fetch), carrying the real byte size — so a
    /// multi-second first-run download is never a silent surprise. See ADR-0012.
    EmbedderDownload {
        /// The model that was downloaded.
        model: String,
        /// Real download size, in bytes.
        bytes: u64,
    },
    /// Emitted when a semantic/hybrid search runs against an embedding set built
    /// with a *different* model than the one now configured. Retrieval fails
    /// rather than mixing vector spaces; the caller must rebuild the complete
    /// embedding cache. See `dense_cache.rs` and ADR-0012.
    EmbedderModelMismatch {
        /// The model the existing embeddings were built with.
        built: String,
        /// The model now configured.
        active: String,
    },
    /// Emitted once when a semantic/hybrid search finds the attached intent
    /// graph's centroids were built with a *different* embedding model than the
    /// active one, so cosine across the two spaces would be meaningless. Unlike
    /// [`Self::EmbedderModelMismatch`] (corpus, fatal), the usage arm merely
    /// **pauses** — base ranking is unaffected — until the graph is rebuilt. See
    /// `usage.rs` and ADR-0014.
    UsageModelMismatch {
        /// The graph's model — its fingerprint, or its centroid width when the
        /// mismatch is dimensional.
        built: String,
        /// The active model, in the same units as `built`.
        active: String,
        /// `true` when the models differ in output dimension, `false` when only
        /// the model identity differs at the same width (a same-dim swap a length
        /// check cannot catch).
        dim_mismatch: bool,
    },
    /// Emitted on every search of a registry that has an intent graph attached,
    /// recording whether usage history contributed to the ranking (ADR-0014).
    /// A registry with no graph emits nothing, so this event's presence is
    /// itself the signal that adaptive ranking is switched on.
    ///
    /// `intent: None` is the **miss** case: the query matched no cluster and
    /// ranked exactly as it would have with no graph at all. A rising share of
    /// misses means the graph no longer covers what is being asked — the cue to
    /// re-derive it.
    ///
    /// `intent: Some(_)` with `promoted: 0` and `dropped > 0` is a different
    /// failure wearing similar clothes: the cluster matched, but every
    /// capability it remembers has left the catalog, so it contributed nothing.
    /// That is catalog drift, not a coverage gap, and re-deriving the graph
    /// fixes it. Reading the two apart is what [`Self::UsageBoost::dropped`] is
    /// for.
    UsageBoost {
        /// Id of the matched cluster; `None` when nothing cleared the match
        /// threshold.
        intent: Option<String>,
        /// How well the query matched the cluster — cosine on the dense tier,
        /// token-overlap share on the lexical one. `0.0` on a miss. Scales
        /// differ between tiers, so compare within one. Reported so near-misses
        /// are visible and the threshold can be judged against real traffic.
        similarity: f64,
        /// The matched cluster's observation count, which scales the arm's
        /// weight. `0` on a miss.
        support: u32,
        /// How many capability ids the arm contributed to the fusion. `0` on a
        /// miss.
        promoted: u32,
        /// How many ids the matched cluster remembers that the registry no
        /// longer defines, so they were dropped from the arm rather than
        /// ranked. `0` on a miss — nothing matched, so nothing was dropped.
        ///
        /// Non-zero means the graph and the catalog have drifted apart. It says
        /// nothing about ranking quality on its own: an id the agent cannot
        /// invoke must never be returned, so dropping is correct. What it
        /// buys is that the drop is *visible* instead of being folded into the
        /// miss rate.
        ///
        /// `#[serde(default)]` so a log written before this field existed still
        /// replays: an older `usage_boost` line reads as `0`, which is what it
        /// meant.
        #[serde(default)]
        dropped: u32,
    },
    /// Emitted once when an in-process model's pooling could not be detected
    /// (no `1_Pooling/config.json`) and no override was given, so a mode was
    /// assumed. A non-silent guess: set `pooling` to correct it. See ADR-0012.
    EmbedderPoolingAssumed {
        /// The model whose pooling could not be detected.
        model: String,
        /// The pooling mode that was assumed (`"cls"` or `"mean"`).
        pooling: String,
    },
}

impl TraceEvent {
    pub(crate) fn catalog_definition_for_tool(tool: &Tool) -> Self {
        Self::catalog_definition(
            CatalogKind::Tool,
            &tool.id,
            &tool.name,
            &tool.description,
            &[],
            Some(tool.input_schema.clone()),
            Some(tool.output_schema.clone()),
            tool.searchable_description.as_deref(),
        )
    }

    pub(crate) fn catalog_definition_for_skill(skill: &Skill) -> Self {
        Self::catalog_definition(
            CatalogKind::Skill,
            &skill.id,
            &skill.name,
            &skill.description,
            &skill.tags,
            None,
            None,
            skill.searchable_description.as_deref(),
        )
    }

    pub(crate) fn catalog_definition_for_fact(fact: &Fact) -> Self {
        Self::catalog_definition(
            CatalogKind::Fact,
            &fact.id,
            &fact.name,
            &fact.description,
            &fact.tags,
            None,
            None,
            fact.searchable_description.as_deref(),
        )
    }

    pub(crate) fn catalog_definition_hash(&self) -> Option<&str> {
        match self {
            Self::CatalogDefinition { content_hash, .. } => Some(content_hash),
            _ => None,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn catalog_definition(
        kind: CatalogKind,
        id: &str,
        name: &str,
        description: &str,
        tags: &[String],
        input_schema: Option<serde_json::Value>,
        output_schema: Option<serde_json::Value>,
        override_description: Option<&str>,
    ) -> Self {
        let searchable_description = override_description.unwrap_or(description);
        let searchable_description_overridden = override_description.is_some();
        let content = CatalogDefinitionContent {
            kind,
            id,
            name,
            description,
            tags,
            input_schema: input_schema.as_ref(),
            output_schema: output_schema.as_ref(),
            searchable_description,
            searchable_description_overridden,
        };
        let content_hash = catalog_definition_hash(&content);
        Self::CatalogDefinition {
            kind,
            id: id.into(),
            name: name.into(),
            description: description.into(),
            tags: tags.to_vec(),
            input_schema,
            output_schema,
            searchable_description: searchable_description.into(),
            searchable_description_overridden,
            content_hash,
        }
    }
}

fn catalog_definition_hash(content: &CatalogDefinitionContent<'_>) -> String {
    let mut value = serde_json::to_value(content).expect("catalog definition is JSON serializable");
    canonicalize_json(&mut value);
    format!("{:x}", Sha256::digest(value.to_string().as_bytes()))
}

fn canonicalize_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => values.iter_mut().for_each(canonicalize_json),
        serde_json::Value::Object(map) => {
            map.values_mut().for_each(canonicalize_json);
            map.sort_keys();
        }
        _ => {}
    }
}

/// Per-event correlation fields supplied by the emitting integration.
///
/// Sinks fill stable envelope fields such as `event_id`, `session_id`, and
/// `source_id`; callers use this context only for facts known at the emission
/// site. Missing fields are omitted from the flattened JSON envelope.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TraceEventContext {
    /// Client-generated id for this event. Sinks mint one when absent.
    pub event_id: Option<String>,
    /// Id shared by every event in one invocation lifecycle.
    pub invocation_id: Option<String>,
    /// Catalog revision known when the event was emitted.
    pub catalog_version: Option<String>,
    /// Deployment environment supplied by the application.
    pub environment: Option<String>,
    /// Application-provided subject id.
    pub end_user_id: Option<String>,
    /// Active OpenTelemetry trace id, when available.
    pub trace_id: Option<String>,
    /// Active OpenTelemetry span id, when available.
    pub span_id: Option<String>,
}

impl TraceEventContext {
    /// Create context for one invocation lifecycle with a fresh opaque id.
    /// Clone and pass it with the start and terminal event so concurrent calls
    /// to the same tool remain paired even when they finish out of order.
    pub fn new_invocation() -> Self {
        Self {
            invocation_id: Some(ulid::Ulid::new().to_string()),
            ..Self::default()
        }
    }
}

/// The versioned wrapper a sink writes around each [`TraceEvent`]: schema
/// version, stable identity, timestamp, and correlation fields. On the wire the event is flattened
/// (`#[serde(flatten)]`), so its `type` tag and fields sit beside `v` / `ts` /
/// `session_id` in one JSON object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceEnvelope {
    /// Envelope schema version; currently `2`.
    pub v: u32,
    /// Client-generated ULID identifying exactly this event.
    #[serde(default)]
    pub event_id: String,
    /// Event time, in milliseconds since the Unix epoch.
    pub ts: u64,
    /// The session the event belongs to, as given to the sink — correlates
    /// all events from one agent session.
    pub session_id: String,
    /// Stable source identity shared by events and catalog snapshots.
    #[serde(default)]
    pub source_id: String,
    /// Id shared by every event in one invocation lifecycle.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub invocation_id: Option<String>,
    /// Catalog revision known when the event was emitted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub catalog_version: Option<String>,
    /// Deployment environment supplied by the application.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub environment: Option<String>,
    /// Application-provided subject id.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_user_id: Option<String>,
    /// Active OpenTelemetry trace id, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<String>,
    /// Active OpenTelemetry span id, when available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span_id: Option<String>,
    /// The event itself, flattened into the envelope on the wire.
    #[serde(flatten)]
    pub event: TraceEvent,
}
