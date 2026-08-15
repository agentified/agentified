/// A tool registered for retrieval — one entry in a [`crate::ToolRegistry`]
/// corpus.
///
/// `name` and the effective searchable description drive ranking. The name is
/// indexed whole and identifier-split; the description component defaults to
/// [`Self::description`] and can be replaced by [`Self::searchable_description`].
/// Schemas remain model-facing and are not indexed. `id` is the stable key hits
/// carry back and is not itself indexed.
pub struct Tool {
    /// Stable identifier, returned in [`crate::SearchHit::tool_id`].
    /// Registering the same id again replaces the entry in place. Not indexed
    /// for ranking.
    pub id: String,
    /// Model-facing tool name (e.g. `read_file`). Indexed both verbatim and
    /// space-split, so snake_case/camelCase constituent words match.
    pub name: String,
    /// What the tool does — the primary ranking text.
    pub description: String,
    /// Optional replacement for the description component used by BM25 and
    /// dense retrieval. The tool name remains indexed. `None` uses
    /// [`Self::description`].
    pub searchable_description: Option<String>,
    /// JSON Schema of the tool's arguments. Model-facing; not indexed.
    pub input_schema: serde_json::Value,
    /// JSON Schema of the tool's result. Model-facing; not indexed.
    pub output_schema: serde_json::Value,
}
