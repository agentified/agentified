use crate::tool::Tool;

pub(crate) fn searchable_text(tool: &Tool) -> String {
    let mut tokens: Vec<String> = Vec::new();
    if !tool.name.is_empty() {
        push_identifier(&tool.name, &mut tokens);
    }
    if let Some(description) = &tool.experimental_searchable_description {
        if !description.is_empty() {
            tokens.push(description.clone());
        }
    } else {
        if !tool.description.is_empty() {
            tokens.push(tool.description.clone());
        }
        // Input property NAMES only. A parameter called `branch` genuinely helps
        // match "tasks for a branch"; the prose describing it does not, and the
        // output schema describes what comes back rather than what was asked
        // for. See ADR-0023.
        flatten(&tool.input_schema, &mut tokens);
    }
    tokens.join(" ")
}

/// Collect property **names**, recursing through nested objects and array items.
///
/// Names only. A property's `description` is prose written to help a model fill
/// the argument in, and it is routinely longer than the tool's own description —
/// so a tool with fifteen parameters arrives at the index as mostly a list of
/// what its arguments mean rather than what it does. `enum` values are data.
/// Both used to be folded in here, and measurably inflated parameter-heavy write
/// ops past read ops that answered the query (ADR-0023).
fn flatten(value: &serde_json::Value, tokens: &mut Vec<String>) {
    if let Some(properties) = value.get("properties").and_then(|value| value.as_object()) {
        for (key, value) in properties {
            push_identifier(key, tokens);
            flatten(value, tokens);
        }
    }
    if let Some(items) = value.get("items") {
        flatten(items, tokens);
    }
}

// Push the original identifier and, if it differs, a space-split form so that
// the bm25 crate's UAX #29 tokenizer (which keeps `snake_case` and `camelCase`
// whole) still surfaces the constituent words.
pub(crate) fn push_identifier(s: &str, tokens: &mut Vec<String>) {
    tokens.push(s.to_string());
    let split = split_identifier(s);
    if split != s {
        tokens.push(split);
    }
}

pub(crate) fn split_identifier(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 4);
    let mut prev: Option<char> = None;
    for c in s.chars() {
        if c == '_' {
            out.push(' ');
        } else if c.is_uppercase() && matches!(prev, Some(p) if p.is_lowercase()) {
            out.push(' ');
            out.push(c);
        } else {
            out.push(c);
        }
        prev = Some(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn read_file_tool() -> Tool {
        Tool {
            id: "read_file".into(),
            name: "read_file".into(),
            description: "Read a file from disk".into(),
            experimental_searchable_description: None,
            input_schema: json!({
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "absolute path"
                    },
                    "encoding": {
                        "type": "string",
                        "enum": ["utf8", "binary"],
                        "description": "file encoding"
                    }
                }
            }),
            // A real output schema, so the negative assertions below test
            // something: it describes what comes back, not what was asked for.
            output_schema: json!({
                "properties": {
                    "checksum": {
                        "type": "string",
                        "description": "sha256 digest of the returned bytes"
                    }
                }
            }),
        }
    }

    #[test]
    fn searchable_text_is_deterministic() {
        let tool = read_file_tool();
        let first = searchable_text(&tool);
        let second = searchable_text(&tool);
        assert_eq!(first, second);
    }

    /// The stable projection keeps the INPUT schema's property names and drops
    /// the output schema entirely (ADR-0023). It kept both until that ADR: an
    /// output schema describes what comes back, never what was asked for, so it
    /// only ever inflated a tool's document.
    #[test]
    fn stable_searchable_text_keeps_input_names_and_drops_the_output_schema() {
        let tool = read_file_tool();
        let text = searchable_text(&tool);
        assert!(text.contains("path"), "input property name missing: {text}");
        assert!(!text.contains("checksum"), "output schema leaked: {text}");
    }

    #[test]
    fn searchable_text_preserves_schema_defined_property_order() {
        let tool = read_file_tool();
        let text = searchable_text(&tool);
        let path_idx = text.find("path").expect("path token missing");
        let encoding_idx = text.find("encoding").expect("encoding token missing");
        assert!(
            path_idx < encoding_idx,
            "expected schema-defined order (path before encoding) in: {text}"
        );
    }

    #[test]
    fn searchable_text_omits_json_structure_keywords() {
        let tool = read_file_tool();
        let text = searchable_text(&tool);
        // Tokens we explicitly skip: type names, structural keys, JSON syntax.
        assert!(
            !text.contains("\"type\""),
            "raw type quoting leaked: {text}"
        );
        assert!(
            !text.contains("\"properties\""),
            "properties leaked: {text}"
        );
        assert!(!text.contains('{'), "JSON braces leaked: {text}");
        // And the parts of a schema that are prose or data rather than a name:
        // a property's description says how to fill the argument in, an enum
        // value is data. Neither says what the tool is for (ADR-0023).
        assert!(
            !text.contains("absolute path"),
            "property description leaked: {text}"
        );
        assert!(!text.contains("utf8"), "enum value leaked: {text}");
        assert!(!text.contains("sha256"), "output schema leaked: {text}");
        // The names themselves are the point of keeping the schema at all.
        assert!(text.contains("path"), "property name missing: {text}");
        assert!(text.contains("encoding"), "property name missing: {text}");
    }
}
