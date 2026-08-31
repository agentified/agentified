use ratel_ai_core::{Tool, ToolRegistry};
use serde_json::json;

fn empty_schema() -> serde_json::Value {
    json!({})
}

#[test]
fn empty_registry_returns_no_results() {
    let registry = ToolRegistry::new();
    let hits = registry.search("anything", 5);
    assert!(hits.is_empty());
}

#[test]
fn snake_case_name_is_split_for_natural_language_queries() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "search_files".into(),
        name: "search_files".into(),
        description: String::new(),
        experimental_searchable_description: None,
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    });
    registry.register(Tool {
        id: "decoy".into(),
        name: "decoy".into(),
        description: "unrelated background tool".into(),
        experimental_searchable_description: None,
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    });

    let hits = registry.search("search", 5);

    assert!(
        !hits.is_empty(),
        "expected snake_case name to match its parts"
    );
    assert_eq!(hits[0].tool_id, "search_files");
}

#[test]
fn camel_case_name_is_split_for_natural_language_queries() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "computeHash".into(),
        name: "computeHash".into(),
        description: String::new(),
        experimental_searchable_description: None,
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    });

    let hits = registry.search("compute", 5);

    assert!(
        !hits.is_empty(),
        "expected camelCase name to match its parts"
    );
    assert_eq!(hits[0].tool_id, "computeHash");
}

#[test]
fn stable_projection_indexes_schema_property_names() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "tool".into(),
        name: "tool".into(),
        description: String::new(),
        experimental_searchable_description: None,
        input_schema: json!({
            "properties": {
                "user-id": {}
            }
        }),
        output_schema: empty_schema(),
    });

    let hits = registry.search("user", 5);

    assert_eq!(hits[0].tool_id, "tool");
}

#[test]
fn re_registering_same_id_replaces_entry() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "shared".into(),
        name: "shared".into(),
        description: "yodel mountain".into(),
        experimental_searchable_description: None,
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    });
    registry.register(Tool {
        id: "shared".into(),
        name: "shared".into(),
        description: "kitchen pancake".into(),
        experimental_searchable_description: None,
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    });

    let stale_hits = registry.search("yodel mountain", 5);
    let fresh_hits = registry.search("kitchen pancake", 5);

    assert!(
        stale_hits.is_empty(),
        "old description should not match anymore"
    );
    assert_eq!(fresh_hits.len(), 1);
    assert_eq!(fresh_hits[0].tool_id, "shared");
    // Replace-in-place: the corpus holds exactly one entry for the id, not two.
    assert_eq!(registry.len(), 1);
}

#[test]
fn re_register_keeps_corpus_size_stable() {
    // Repeatedly re-registering the same id must not grow the corpus — the
    // RAT-378 regression (a duplicate would drift BM25 avgdl and leak memory).
    let mut registry = ToolRegistry::new();
    for i in 0..50 {
        registry.register(Tool {
            id: "hot".into(),
            name: "hot".into(),
            description: format!("revision {i} of a hot-reloaded tool"),
            experimental_searchable_description: None,
            input_schema: empty_schema(),
            output_schema: empty_schema(),
        });
    }
    assert_eq!(registry.len(), 1, "50 re-registers, one entry");
    // The single surviving entry ranks once — never a duplicate hit.
    let hits = registry.search("hot-reloaded tool", 5);
    assert_eq!(hits.first().map(|h| h.tool_id.as_str()), Some("hot"));
    assert_eq!(hits.len(), 1);
}

#[test]
fn mutation_after_a_warmed_search_is_visible_in_the_next_search() {
    // Searches before a register must not leave any stale ranking state
    // behind: whatever the engine reuses across searches, a mutation is
    // visible in the very next call.
    let tool = |id: &str, desc: &str| Tool {
        id: id.into(),
        name: id.into(),
        description: desc.into(),
        experimental_searchable_description: None,
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    };

    let mut registry = ToolRegistry::new();
    registry.register(tool("read", "read a file from disk"));
    registry.register(tool("write", "write bytes to a socket"));
    // Warm: several searches before the mutation.
    for _ in 0..3 {
        let _ = registry.search("read a file", 5);
    }

    registry.register(tool("archive", "compress a directory into an archive"));
    assert_eq!(
        registry.search("compress an archive", 5)[0].tool_id,
        "archive",
        "a tool registered after searches must rank immediately"
    );

    registry.register(tool("read", "stream sensor telemetry"));
    // Query on the OLD description only ("read" would still match the name).
    assert!(
        registry.search("file disk", 5).is_empty(),
        "replaced content must stop matching immediately"
    );
    assert_eq!(
        registry.search("stream sensor telemetry", 5)[0].tool_id,
        "read"
    );
}

#[test]
fn warmed_registry_hits_are_byte_identical_to_a_fresh_registry() {
    // ADR-0011 pins BM25 behavior byte-for-byte: a registry that has already
    // searched and mutated must score exactly like a fresh one holding the
    // same final corpus — ids, order, AND f32 scores.
    let tool = |id: &str, desc: &str| Tool {
        id: id.into(),
        name: id.into(),
        description: desc.into(),
        experimental_searchable_description: None,
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    };
    let corpus = [
        ("read", "read a file from disk"),
        ("write", "write bytes to a socket"),
        ("archive", "compress a directory into an archive"),
    ];

    let mut warmed = ToolRegistry::new();
    warmed.register(tool("read", "an obsolete description"));
    warmed.register(tool("write", corpus[1].1));
    let _ = warmed.search("file", 5); // warm before the corpus settles
    warmed.register(tool(corpus[0].0, corpus[0].1)); // replace in place
    warmed.register(tool(corpus[2].0, corpus[2].1));
    let _ = warmed.search("socket", 5); // warm again on the final corpus

    let mut fresh = ToolRegistry::new();
    for (id, desc) in corpus {
        fresh.register(tool(id, desc));
    }

    for query in ["read a file", "compress the directory", "write bytes"] {
        let warmed_hits: Vec<(String, f32)> = warmed
            .search(query, 5)
            .into_iter()
            .map(|h| (h.tool_id, h.score))
            .collect();
        let fresh_hits: Vec<(String, f32)> = fresh
            .search(query, 5)
            .into_iter()
            .map(|h| (h.tool_id, h.score))
            .collect();
        assert_eq!(warmed_hits, fresh_hits, "query={query}");
    }
}

#[test]
fn search_ranks_stronger_match_above_weaker() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "strong".into(),
        name: "compress".into(),
        description: "compress directories into compress archives quickly".into(),
        experimental_searchable_description: None,
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    });
    // The weak tool's only signal is a property NAME. It used to be an enum
    // value, which is no longer indexed at all (ADR-0023) — the point of the
    // test is the ranking, so it needs a signal that still exists.
    registry.register(Tool {
        id: "weak".into(),
        name: "convert".into(),
        description: String::new(),
        experimental_searchable_description: None,
        input_schema: json!({
            "properties": {
                "compress": { "type": "boolean" }
            }
        }),
        output_schema: empty_schema(),
    });

    let hits = registry.search("compress", 5);

    assert!(
        hits.len() >= 2,
        "expected both tools to match, got {}",
        hits.len()
    );
    assert_eq!(hits[0].tool_id, "strong");
    assert_eq!(hits[1].tool_id, "weak");
    assert!(hits[0].score > hits[1].score);
}

#[test]
fn experimental_projection_ignores_a_schema_only_match() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "strong".into(),
        name: "compress".into(),
        description: "compress directories into compress archives quickly".into(),
        experimental_searchable_description: Some(
            "compress directories into compress archives quickly".into(),
        ),
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    });
    registry.register(Tool {
        id: "weak".into(),
        name: "convert".into(),
        description: String::new(),
        experimental_searchable_description: Some(String::new()),
        input_schema: json!({
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["compress", "expand"]
                }
            }
        }),
        output_schema: empty_schema(),
    });

    let hits = registry.search("compress", 5);

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].tool_id, "strong");
}

#[test]
fn search_respects_top_k_bound() {
    let mut registry = ToolRegistry::new();
    for i in 0..5 {
        registry.register(Tool {
            id: format!("tool_{i}"),
            name: format!("tool_{i}"),
            description: "shared keyword shrubbery".into(),
            experimental_searchable_description: None,
            input_schema: empty_schema(),
            output_schema: empty_schema(),
        });
    }

    let hits = registry.search("shrubbery", 3);

    assert!(
        hits.len() <= 3,
        "expected at most 3 hits, got {}",
        hits.len()
    );
}

/// The output schema describes what comes BACK, not what the caller asked for,
/// so none of it reaches the index (ADR-0023).
#[test]
fn an_output_schema_description_is_not_indexed() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "weather".into(),
        name: "weather".into(),
        description: String::new(),
        experimental_searchable_description: None,
        input_schema: empty_schema(),
        output_schema: json!({
            "properties": {
                "temperature_celsius": {
                    "type": "number",
                    "description": "ambient temperature reading at the station"
                }
            }
        }),
    });

    let hits = registry.search("ambient temperature reading", 5);

    assert!(
        hits.is_empty(),
        "output schema text must not be searchable, got {:?}",
        hits.iter().map(|h| h.tool_id.as_str()).collect::<Vec<_>>()
    );
}

/// Nested property NAMES still reach the index; the prose describing them does
/// not, at any depth (ADR-0023).
#[test]
fn a_nested_property_description_is_not_indexed() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "deploy".into(),
        name: "deploy".into(),
        description: String::new(),
        experimental_searchable_description: None,
        input_schema: json!({
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "infra": {
                            "type": "object",
                            "properties": {
                                "region": {
                                    "type": "string",
                                    "description": "datacenter location identifier"
                                }
                            }
                        }
                    }
                }
            }
        }),
        output_schema: empty_schema(),
    });

    let hits = registry.search("datacenter location identifier", 5);

    assert!(
        hits.is_empty(),
        "nested property descriptions must not be searchable, got {:?}",
        hits.iter().map(|h| h.tool_id.as_str()).collect::<Vec<_>>()
    );
}

/// The same through `items`: the property names inside an array's element shape
/// are indexed, their descriptions are not (ADR-0023).
#[test]
fn an_array_item_description_is_not_indexed() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "batch".into(),
        name: "batch".into(),
        description: String::new(),
        experimental_searchable_description: None,
        input_schema: json!({
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {
                                "type": "string",
                                "description": "unique product identifier"
                            }
                        }
                    }
                }
            }
        }),
        output_schema: empty_schema(),
    });

    let hits = registry.search("unique product identifier", 5);

    assert!(
        hits.is_empty(),
        "array item descriptions must not be searchable, got {:?}",
        hits.iter().map(|h| h.tool_id.as_str()).collect::<Vec<_>>()
    );
}

/// Enum values are data, not a description of what a tool is for — `"toml"` says
/// nothing about `convert`'s purpose (ADR-0023).
#[test]
fn an_enum_value_is_not_indexed() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "convert".into(),
        name: "convert".into(),
        description: String::new(),
        experimental_searchable_description: None,
        input_schema: json!({
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["yaml", "toml", "json"]
                }
            }
        }),
        output_schema: empty_schema(),
    });

    let hits = registry.search("toml", 5);

    assert!(
        hits.is_empty(),
        "enum values must not be searchable, got {:?}",
        hits.iter().map(|h| h.tool_id.as_str()).collect::<Vec<_>>()
    );
}

/// A parameter's description is written to help a model fill the argument in,
/// and is routinely longer than the tool's own description — it inflated
/// parameter-heavy tools past ones that answered the query (ADR-0023).
#[test]
fn an_input_param_description_is_not_indexed() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "fetch".into(),
        name: "fetch".into(),
        description: String::new(),
        experimental_searchable_description: None,
        input_schema: json!({
            "properties": {
                "url": {
                    "type": "string",
                    "description": "remote http target to retrieve"
                }
            }
        }),
        output_schema: empty_schema(),
    });

    let hits = registry.search("remote http target", 5);

    assert!(
        hits.is_empty(),
        "parameter descriptions must not be searchable, got {:?}",
        hits.iter().map(|h| h.tool_id.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn experimental_projection_does_not_match_input_param_description() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "fetch".into(),
        name: "fetch".into(),
        description: String::new(),
        experimental_searchable_description: Some(String::new()),
        input_schema: json!({
            "properties": {
                "url": {
                    "type": "string",
                    "description": "remote http target to retrieve"
                }
            }
        }),
        output_schema: empty_schema(),
    });

    let hits = registry.search("remote http target", 5);

    assert!(hits.is_empty(), "schemas are model-facing, not indexed");
}

#[test]
fn experimental_projection_does_not_match_input_param_name() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "fetch".into(),
        name: "fetch".into(),
        description: String::new(),
        experimental_searchable_description: Some(String::new()),
        input_schema: json!({
            "properties": {
                "endpoint": {}
            }
        }),
        output_schema: empty_schema(),
    });

    let hits = registry.search("endpoint", 5);

    assert!(hits.is_empty(), "schemas are model-facing, not indexed");
}

#[test]
fn search_matches_tool_description() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "diff".into(),
        name: "diff".into(),
        description: "compute the unified textual difference between two files".into(),
        experimental_searchable_description: None,
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    });

    let hits = registry.search("unified textual difference", 5);

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].tool_id, "diff");
    assert!(hits[0].score > 0.0);
}

#[test]
fn experimental_searchable_description_replaces_tool_description_but_keeps_name() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "billing".into(),
        name: "billing_helper".into(),
        description: "orchestrate zeppelin manifests".into(),
        experimental_searchable_description: Some("reconcile overdue invoices".into()),
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    });

    assert_eq!(registry.search("overdue invoices", 5)[0].tool_id, "billing");
    assert!(registry.search("zeppelin manifests", 5).is_empty());
    assert_eq!(registry.search("billing", 5)[0].tool_id, "billing");
}

#[test]
fn search_matches_tool_name() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "read_file".into(),
        name: "read_file".into(),
        description: String::new(),
        experimental_searchable_description: None,
        input_schema: empty_schema(),
        output_schema: empty_schema(),
    });

    let hits = registry.search("read_file", 5);

    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].tool_id, "read_file");
    assert!(hits[0].score > 0.0);
}

#[test]
fn tied_scores_are_ordered_by_tool_id() {
    // Two tools with identical searchable text score identically for any
    // matching query. The bm25 crate collects candidates through a HashSet,
    // so on equal scores the order falls back to hash-seed iteration order
    // and flips between processes. The registry must break ties stably.
    let mut registry = ToolRegistry::new();
    for id in ["zeta_tool", "alpha_tool"] {
        registry.register(Tool {
            id: id.into(),
            name: id.into(),
            description: "send a notification message to a channel".into(),
            experimental_searchable_description: None,
            input_schema: empty_schema(),
            output_schema: empty_schema(),
        });
    }

    let hits = registry.search("notification message", 5);

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].score, hits[1].score, "fixture must produce a tie");
    assert_eq!(hits[0].tool_id, "alpha_tool");
    assert_eq!(hits[1].tool_id, "zeta_tool");
}

#[test]
fn tied_scores_keep_top_k_membership_stable() {
    // Regression for the flicker observed while reproducing issue #56:
    // with a tie at the top_k boundary, which tool made the cut depended
    // on hash-seed iteration order, so top-K membership changed across
    // process runs. With a stable tie-break the cut is always the same.
    let mut registry = ToolRegistry::new();
    for id in ["zeta_tool", "mid_tool", "alpha_tool"] {
        registry.register(Tool {
            id: id.into(),
            name: id.into(),
            description: "send a notification message to a channel".into(),
            experimental_searchable_description: None,
            input_schema: empty_schema(),
            output_schema: empty_schema(),
        });
    }

    let hits = registry.search("notification message", 2);

    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].tool_id, "alpha_tool");
    assert_eq!(hits[1].tool_id, "mid_tool");
}

#[test]
fn experimental_projection_does_not_match_output_schema_description() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "weather".into(),
        name: "weather".into(),
        description: String::new(),
        experimental_searchable_description: Some(String::new()),
        input_schema: empty_schema(),
        output_schema: json!({
            "properties": {
                "temperature_celsius": {
                    "type": "number",
                    "description": "ambient temperature reading at the station"
                }
            }
        }),
    });

    let hits = registry.search("ambient temperature reading", 5);

    assert!(hits.is_empty(), "schemas are model-facing, not indexed");
}

#[test]
fn experimental_projection_does_not_match_nested_object_description() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "deploy".into(),
        name: "deploy".into(),
        description: String::new(),
        experimental_searchable_description: Some(String::new()),
        input_schema: json!({
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {
                        "infra": {
                            "type": "object",
                            "properties": {
                                "region": {
                                    "type": "string",
                                    "description": "datacenter location identifier"
                                }
                            }
                        }
                    }
                }
            }
        }),
        output_schema: empty_schema(),
    });

    let hits = registry.search("datacenter location identifier", 5);

    assert!(hits.is_empty(), "schemas are model-facing, not indexed");
}

#[test]
fn experimental_projection_does_not_match_array_items_description() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "batch".into(),
        name: "batch".into(),
        description: String::new(),
        experimental_searchable_description: Some(String::new()),
        input_schema: json!({
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sku": {
                                "type": "string",
                                "description": "unique product identifier"
                            }
                        }
                    }
                }
            }
        }),
        output_schema: empty_schema(),
    });

    let hits = registry.search("unique product identifier", 5);

    assert!(hits.is_empty(), "schemas are model-facing, not indexed");
}

#[test]
fn experimental_projection_does_not_match_enum_value() {
    let mut registry = ToolRegistry::new();
    registry.register(Tool {
        id: "convert".into(),
        name: "convert".into(),
        description: String::new(),
        experimental_searchable_description: Some(String::new()),
        input_schema: json!({
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["yaml", "toml", "json"]
                }
            }
        }),
        output_schema: empty_schema(),
    });

    let hits = registry.search("toml", 5);

    assert!(hits.is_empty(), "schemas are model-facing, not indexed");
}
