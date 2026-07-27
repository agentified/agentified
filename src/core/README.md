<div align="center">
  <h1>ratel-ai-core</h1>
  <p>In-process tool, skill, and fact retrieval for AI agents.</p>

  <p>
    <a href="https://docs.ratel.sh">Docs</a> •
    <a href="https://github.com/ratel-ai/ratel">GitHub</a> •
    <a href="https://discord.gg/75vAPdjYqT">Discord</a>
  </p>

  <p>
    <a href="https://crates.io/crates/ratel-ai-core"><img src="https://img.shields.io/crates/v/ratel-ai-core?label=crates.io&color=e57300" alt="crates.io" /></a>
    <a href="https://docs.rs/ratel-ai-core"><img src="https://img.shields.io/docsrs/ratel-ai-core?label=docs.rs" alt="docs.rs" /></a>
    <a href="https://github.com/ratel-ai/ratel/stargazers"><img src="https://img.shields.io/github/stars/ratel-ai/ratel?style=social" alt="GitHub stars" /></a>
    <a href="https://github.com/ratel-ai/ratel/blob/main/LICENSE-APACHE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 license" /></a>
  </p>
</div>

`ratel-ai-core` is Ratel's Rust retrieval engine. Register tool, skill, or fact metadata once, then rank the catalog for each agent turn. Tools and skills are *pulled* on relevance; facts are constant grounding content the higher layers *push* into the context, always-on or retrieval-gated. BM25 is the model-free default; semantic and hybrid retrieval use either an in-process model or a configured OpenAI-compatible embedding endpoint. The retrieval engine and cache stay in-process, with no vector database or Ratel service to deploy.

This crate owns retrieval and its local trace stream. Tool execution, MCP connections, and authentication integrations live in the SDK and local distribution.

## Install

```bash
cargo add ratel-ai-core
```

## Quickstart

Add this to `src/main.rs`, then run `cargo run`:

```rust
use ratel_ai_core::{Tool, ToolRegistry};

fn main() {
    let mut registry = ToolRegistry::new();
    for (id, description) in [
        ("read_file", "Read text from a local file"),
        ("send_email", "Send an email to a recipient"),
    ] {
        registry.register(Tool {
            id: id.into(),
            name: id.into(),
            description: description.into(),
            input_schema: Default::default(),
            output_schema: Default::default(),
        });
    }

    let hits = registry.search("read a local file", 1);
    assert_eq!(hits[0].tool_id, "read_file");
    println!("best tool: {}", hits[0].tool_id);
}
```

Package layout: `src/` holds the retrieval and trace engine, `examples/` contains runnable demos, and `tests/` covers integration behavior. From a repository checkout, run `cargo test -p ratel-ai-core` or `cargo run -p ratel-ai-core --example search_demo`.

Continue with [tool retrieval](https://docs.ratel.sh/docs/tool-retrieval), the [Rust API reference](https://docs.rs/ratel-ai-core/latest/ratel_ai_core/), or the [source repository](https://github.com/ratel-ai/ratel). Benchmark results are maintained separately in [ratel-ai/ratel-bench](https://github.com/ratel-ai/ratel-bench).
