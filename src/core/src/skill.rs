use std::collections::HashMap;

/// A skill registered for retrieval — the on-demand analog of a [`crate::Tool`].
///
/// `name`, the effective searchable description, and `tags` drive ranking.
/// The description component defaults to [`Self::description`] and can be
/// replaced by [`Self::experimental_searchable_description`]. `tags` are author-declared
/// labels and task phrases ("frontend", "login form") folded into the BM25
/// text so a terse intent prompt matches the skill. `tools` are the ids of tools the body's
/// instructions call — an explicit dependency edge, **not** indexed;
/// `search_capabilities` pulls them into its tools bucket so the agent gets a
/// skill and the tools it needs in one turn instead of a second search.
/// `metadata` is free-form, non-indexed context for higher layers — e.g.
/// `{"stacks": ["react"]}` for the push-path ranker to boost/filter by project
/// context, deliberately *not* matched as query terms. `body` is the dispatch
/// payload and is also not indexed, so a long body never skews relevance.
#[derive(PartialEq, Eq)]
pub struct Skill {
    /// Stable identifier, returned in [`crate::SkillHit::skill_id`].
    /// Registering the same id again replaces the entry in place. Not indexed
    /// for ranking.
    pub id: String,
    /// Skill name. Indexed both verbatim and space-split, so
    /// snake_case/camelCase/kebab constituent words match.
    pub name: String,
    /// What the skill is for — the primary ranking text.
    pub description: String,
    /// Experimental replacement for the description component used by BM25 and
    /// dense retrieval. The skill name and tags remain indexed. `None` uses
    /// [`Self::description`].
    pub experimental_searchable_description: Option<String>,
    /// Author-declared labels and task phrases, indexed alongside the
    /// description.
    pub tags: Vec<String>,
    /// Ids of tools the body's instructions call — a dependency edge for
    /// higher layers, not indexed.
    pub tools: Vec<String>,
    /// Free-form, non-indexed context for higher layers (push-path
    /// boosting/filtering); never matched as query terms.
    pub metadata: HashMap<String, Vec<String>>,
    /// The skill's full instructions — the dispatch payload, not indexed.
    pub body: String,
}
