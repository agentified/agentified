# conformance

The shared **contract-against-the-pin** fixtures for the telemetry helpers.

`fixtures.json` is one canonical set of
`(scenario -> expected span + structured OpenTelemetry EventRecords)`.
Each **consuming** language helper's conformance test loads it, builds each span **from its own
`ratel.*` constants through the real OpenTelemetry SDK**, emits Events through the Logs API,
and asserts both signals exactly. These Events are not SpanEvents.

## Fixture shape

| Field | Meaning |
|---|---|
| `span` | logical span id, mapped by the test to a span-name constant (e.g. `execute_tool` → `EXECUTE_TOOL`, `ratel_search` → `RATEL_SEARCH`) |
| `set` | attributes to set, by logical id mapped to an attribute-key constant (e.g. `ratel_origin` → `RATEL_ORIGIN`) |
| `emit_events` | optional Logs EventRecords: logical event id plus structured attributes keyed by logical attribute id |
| `expect_name` | the exact wire span name the pin requires |
| `expect_attributes` | the exact wire attribute keys + values the pin requires |
| `expect_events` | optional exact wire EventRecord names and structured attributes (empty when omitted) |

The logical ids in `span` / `set` / `emit_events` are the constant names lowercased; each helper
keeps a small map from logical id to its own constant, and those maps are the unit under test.

## Consumers

- **TS** — `../ts/src/conformance.test.ts`
- **Python** — `../python/tests/test_conformance.py`
