/// A tool registered for retrieval — one entry in a [`crate::ToolRegistry`]
/// corpus.
///
/// `name` and the effective searchable description drive ranking. The name is
/// indexed whole and identifier-split; the description component defaults to
/// [`Self::description`] and can be replaced by [`Self::experimental_searchable_description`].
/// Schemas stay indexed on the stable path and are excluded when the experimental override is
/// present. `id` is the stable key hits
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
    /// An experimental replacement for the description component used by BM25 and dense
    /// retrieval. Providing it opts out of schema indexing. The tool name remains indexed;
    /// `None` preserves the stable description-plus-schema projection.
    pub experimental_searchable_description: Option<String>,
    /// JSON Schema of the tool's arguments. Indexed unless the experimental override is present.
    pub input_schema: serde_json::Value,
    /// JSON Schema of the tool's result. Indexed unless the experimental override is present.
    pub output_schema: serde_json::Value,
}
