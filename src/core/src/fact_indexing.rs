use crate::fact::Fact;
use crate::indexing::push_identifier;

/// Flatten a fact into the text the BM25 index scores against.
///
/// Mirrors [`crate::skill_indexing::searchable_text`] for skills: the name is
/// pushed both whole and identifier-split, then the description and each tag.
/// Like all indexed text these are tokenized at index *and* query time
/// (lowercased, stemmed, stop-words removed), not matched verbatim. The `body`
/// is intentionally excluded — it is the injected content, not a ranking signal
/// (a 2 KB body would otherwise drown the description's term weights). `pin` and
/// `metadata` are likewise excluded: `pin` splits the injection tiers and
/// `metadata` biases the push ranker — neither is a query term.
pub(crate) fn searchable_text(fact: &Fact) -> String {
    let mut tokens: Vec<String> = Vec::new();
    if !fact.name.is_empty() {
        push_identifier(&fact.name, &mut tokens);
    }
    if !fact.description.is_empty() {
        tokens.push(fact.description.clone());
    }
    for tag in &fact.tags {
        if !tag.is_empty() {
            push_identifier(tag, &mut tokens);
        }
    }
    tokens.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fact::PinMode;
    use std::collections::HashMap;

    fn shop_fact() -> Fact {
        Fact {
            id: "shop-address".into(),
            name: "shop-address".into(),
            description: "Where the barbershop is located and its opening hours".into(),
            tags: vec!["location".into(), "opening hours".into()],
            metadata: HashMap::from([("stacks".into(), vec!["react".into()])]),
            body: "12 Baker Street, London — must not affect ranking".into(),
            pin: PinMode::Always,
        }
    }

    #[test]
    fn searchable_text_is_deterministic() {
        let fact = shop_fact();
        assert_eq!(searchable_text(&fact), searchable_text(&fact));
    }

    #[test]
    fn searchable_text_splits_hyphenated_name_for_word_matching() {
        let text = searchable_text(&shop_fact());
        // Hyphens aren't identifier separators, so the whole name survives; the
        // description and tags supply the natural-language terms.
        assert!(text.contains("shop-address"), "whole name missing: {text}");
        assert!(text.contains("opening hours"), "tag missing: {text}");
    }

    #[test]
    fn searchable_text_excludes_body() {
        let text = searchable_text(&shop_fact());
        assert!(
            !text.contains("must not affect ranking"),
            "body leaked into index: {text}"
        );
    }

    #[test]
    fn searchable_text_splits_snake_case_identifiers() {
        let fact = Fact {
            id: "cancellation_policy".into(),
            name: "cancellation_policy".into(),
            description: String::new(),
            tags: vec![],
            metadata: HashMap::new(),
            body: String::new(),
            pin: PinMode::Retrieved,
        };
        let text = searchable_text(&fact);
        assert!(
            text.contains("cancellation policy"),
            "snake_case not split: {text}"
        );
    }

    #[test]
    fn searchable_text_excludes_metadata() {
        // Metadata (e.g. stacks) biases the push ranker by project context; its
        // values are not query terms.
        let text = searchable_text(&shop_fact());
        assert!(
            !text.contains("react"),
            "metadata leaked into index: {text}"
        );
    }
}
