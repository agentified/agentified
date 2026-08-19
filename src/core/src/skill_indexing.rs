use crate::indexing::push_identifier;
use crate::skill::Skill;

/// Flatten a skill into the text the BM25 index scores against.
///
/// Mirrors [`crate::indexing::searchable_text`] for tools: the name is pushed
/// both whole and identifier-split, then the effective searchable description
/// and each tag. Tags are author-declared labels and task phrases ("frontend",
/// "login form") so a terse intent prompt matches the skill; like all indexed text they are tokenized at
/// index *and* query time (lowercased, stemmed, stop-words removed), not matched
/// verbatim. The `body` is intentionally excluded — it is the dispatch payload,
/// not a ranking signal (a 15 KB body would otherwise drown the description's
/// term weights). `tools` and `metadata` are likewise excluded: `tools` are a
/// dependency edge surfaced via `search_capabilities` and `metadata` (e.g. `stacks`) biases
/// the push ranker — neither is a query term.
pub(crate) fn searchable_text(skill: &Skill) -> String {
    let mut tokens: Vec<String> = Vec::new();
    if !skill.name.is_empty() {
        push_identifier(&skill.name, &mut tokens);
    }
    let description = skill
        .experimental_searchable_description
        .as_deref()
        .unwrap_or(&skill.description);
    if !description.is_empty() {
        tokens.push(description.to_string());
    }
    for tag in &skill.tags {
        if !tag.is_empty() {
            push_identifier(tag, &mut tokens);
        }
    }
    tokens.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn slides_skill() -> Skill {
        Skill {
            id: "frontend-slides".into(),
            name: "frontend-slides".into(),
            description: "Build animation-rich HTML presentations".into(),
            experimental_searchable_description: None,
            tags: vec![
                "frontend".into(),
                "presentations".into(),
                "slide deck".into(),
            ],
            tools: vec!["fs__write_file".into()],
            metadata: HashMap::from([("stacks".into(), vec!["react".into()])]),
            body: "# Frontend Slides\n\nLong body that must not affect ranking…".into(),
        }
    }

    #[test]
    fn searchable_text_is_deterministic() {
        let skill = slides_skill();
        assert_eq!(searchable_text(&skill), searchable_text(&skill));
    }

    #[test]
    fn searchable_text_splits_hyphenated_name_for_word_matching() {
        let skill = slides_skill();
        let text = searchable_text(&skill);
        // Hyphens aren't identifier separators, so the whole name survives; the
        // description and tags supply the natural-language terms.
        assert!(
            text.contains("frontend-slides"),
            "whole name missing: {text}"
        );
        assert!(text.contains("presentations"), "tag missing: {text}");
    }

    #[test]
    fn searchable_text_excludes_body() {
        let skill = slides_skill();
        let text = searchable_text(&skill);
        assert!(
            !text.contains("must not affect ranking"),
            "body leaked into index: {text}"
        );
    }

    #[test]
    fn searchable_text_splits_snake_case_identifiers() {
        let skill = Skill {
            id: "code_review".into(),
            name: "code_review".into(),
            description: String::new(),
            experimental_searchable_description: None,
            tags: vec![],
            tools: vec![],
            metadata: HashMap::new(),
            body: String::new(),
        };
        let text = searchable_text(&skill);
        assert!(text.contains("code review"), "snake_case not split: {text}");
    }

    #[test]
    fn searchable_text_includes_tag_phrases() {
        // Tags carry author task phrases ("slide deck"); they are indexed so a
        // terse intent prompt matches the skill.
        let skill = slides_skill();
        let text = searchable_text(&skill);
        assert!(text.contains("slide deck"), "tag phrase missing: {text}");
    }

    #[test]
    fn searchable_text_excludes_metadata() {
        // Metadata (e.g. stacks) biases the push ranker by project context; its
        // values are not query terms.
        let skill = slides_skill();
        let text = searchable_text(&skill);
        assert!(
            !text.contains("react"),
            "metadata leaked into index: {text}"
        );
    }

    #[test]
    fn searchable_text_excludes_tools() {
        // Declared tool deps are surfaced via `search_capabilities`, not matched as terms.
        let skill = slides_skill();
        let text = searchable_text(&skill);
        assert!(
            !text.contains("write_file"),
            "tool dep leaked into index: {text}"
        );
    }
}
