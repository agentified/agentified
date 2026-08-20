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
        flatten(&tool.input_schema, &mut tokens);
        flatten(&tool.output_schema, &mut tokens);
    }
    tokens.join(" ")
}

fn flatten(value: &serde_json::Value, tokens: &mut Vec<String>) {
    if let Some(properties) = value.get("properties").and_then(|value| value.as_object()) {
        for (key, value) in properties {
            push_identifier(key, tokens);
            push_field_tokens(value, tokens);
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

fn push_field_tokens(value: &serde_json::Value, tokens: &mut Vec<String>) {
    if let Some(description) = value.get("description").and_then(|value| value.as_str()) {
        tokens.push(description.to_string());
    }
    if let Some(values) = value.get("enum").and_then(|value| value.as_array()) {
        for value in values {
            if let Some(value) = value.as_str() {
                tokens.push(value.to_string());
            }
        }
    }
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

    #[test]
    fn stable_searchable_text_keeps_schemas() {
        let tool = read_file_tool();
        let text = searchable_text(&tool);
        assert!(text.contains("path"), "input schema missing: {text}");
        assert!(text.contains("checksum"), "output schema missing: {text}");
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
    }
}
