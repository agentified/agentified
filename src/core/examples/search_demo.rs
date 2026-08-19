//! Tiny demo of `ratel-ai-core` tool search.
//!
//! Run from the repo root:
//!
//! ```bash
//! cargo run -p ratel-ai-core --example search_demo
//! cargo run -p ratel-ai-core --example search_demo -- "your query here"
//! ```

use ratel_ai_core::{Tool, ToolRegistry};
use serde_json::json;

fn main() {
    let mut registry = ToolRegistry::new();

    registry.register(Tool {
        id: "read_file".into(),
        name: "read_file".into(),
        description: "Read a file from local disk and return its textual contents.".into(),
        experimental_searchable_description: None,
        input_schema: json!({
            "properties": {
                "path": { "type": "string", "description": "absolute path to the file" },
                "encoding": {
                    "type": "string",
                    "enum": ["utf8", "binary"],
                    "description": "how to decode the bytes"
                }
            }
        }),
        output_schema: json!({
            "properties": {
                "contents": { "type": "string", "description": "decoded file contents" }
            }
        }),
    });

    registry.register(Tool {
        id: "write_file".into(),
        name: "write_file".into(),
        description: "Write textual contents to a file on local disk.".into(),
        experimental_searchable_description: None,
        input_schema: json!({
            "properties": {
                "path": { "type": "string", "description": "absolute path to the file" },
                "contents": { "type": "string", "description": "bytes to write" }
            }
        }),
        output_schema: json!({}),
    });

    registry.register(Tool {
        id: "search_files".into(),
        name: "search_files".into(),
        description: "Grep across files in a directory using a regular expression.".into(),
        experimental_searchable_description: None,
        input_schema: json!({
            "properties": {
                "root": { "type": "string", "description": "directory to scan recursively" },
                "pattern": { "type": "string", "description": "regular expression to match" }
            }
        }),
        output_schema: json!({
            "properties": {
                "matches": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "matching file path" },
                            "line": { "type": "number", "description": "line number of the match" }
                        }
                    }
                }
            }
        }),
    });

    registry.register(Tool {
        id: "run_command".into(),
        name: "run_command".into(),
        description: "Execute a shell command and capture stdout, stderr, exit code.".into(),
        experimental_searchable_description: None,
        input_schema: json!({
            "properties": {
                "command": { "type": "string", "description": "command line to run" },
                "shell": {
                    "type": "string",
                    "enum": ["bash", "zsh", "sh"],
                    "description": "which shell to use"
                }
            }
        }),
        output_schema: json!({
            "properties": {
                "stdout": { "type": "string" },
                "stderr": { "type": "string" },
                "exit_code": { "type": "number" }
            }
        }),
    });

    let queries: Vec<String> = match std::env::args().nth(1) {
        Some(q) => vec![q],
        None => vec![
            "read a text file".into(),
            "find a regex in a directory".into(),
            "execute a shell command".into(),
            "binary encoding".into(),
        ],
    };

    for query in queries {
        println!("\nquery: {query:?}");
        let hits = registry.search(&query, 5);
        if hits.is_empty() {
            println!("  (no matches)");
            continue;
        }
        for (rank, hit) in hits.iter().enumerate() {
            println!("  {}. {:<14} score={:.4}", rank + 1, hit.tool_id, hit.score);
        }
    }
}
