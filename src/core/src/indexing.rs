use crate::tool::Tool;

pub(crate) fn searchable_text(tool: &Tool) -> String {
    let mut tokens: Vec<String> = Vec::new();
    if !tool.name.is_empty() {
        push_identifier(&tool.name, &mut tokens);
    }
    let description = tool
        .searchable_description
        .as_deref()
        .unwrap_or(&tool.description);
    if !description.is_empty() {
        tokens.push(description.to_string());
    }
    tokens.join(" ")
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
            searchable_description: None,
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
            output_schema: json!({}),
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
    fn searchable_text_excludes_schemas() {
        let tool = read_file_tool();
        let text = searchable_text(&tool);
        assert!(!text.contains("path"), "input schema leaked: {text}");
        assert!(!text.contains("encoding"), "output schema leaked: {text}");
    }
}
