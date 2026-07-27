use serde::{Deserialize, Serialize};

/// Distinguishes a direct API call (pre-fetch helpers, library callers,
/// benchmarks) from one the agent synthesized inside its loop (capability tool).
/// Used to separate the two paths in trace consumers (rerankers train on agent
/// calls, inspector shows both).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    /// A direct API call — SDK helpers, library callers, benchmarks. Wire
    /// value `direct`.
    Direct,
    /// A call the agent synthesized inside its loop, via the capability
    /// tools. Wire value `agent`.
    Agent,
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TraceEvent {
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
    /// [`TraceEvent::IndexChurn`], emitted by [`crate::SkillRegistry::register`].
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

/// The versioned wrapper a sink writes around each [`TraceEvent`]: schema
/// version, timestamp, and session id. On the wire the event is flattened
/// (`#[serde(flatten)]`), so its `type` tag and fields sit beside `v` / `ts` /
/// `session_id` in one JSON object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TraceEnvelope {
    /// Envelope schema version; currently `1`.
    pub v: u32,
    /// Event time, in milliseconds since the Unix epoch.
    pub ts: u64,
    /// The session the event belongs to, as given to the sink — correlates
    /// all events from one agent session.
    pub session_id: String,
    /// The event itself, flattened into the envelope on the wire.
    #[serde(flatten)]
    pub event: TraceEvent,
}
