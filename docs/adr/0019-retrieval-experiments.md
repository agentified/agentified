# 19. Retrieval experiments: deterministic assignment and shadow evaluation

Date: 2026-07-30

## Status

Accepted

Builds on [ADR-0007](0007-telemetry-two-streams.md) (the OpenTelemetry `ratel.*`
overlay), [ADR-0011](0011-selectable-retrieval-methods.md) (rank-based fusion across
incomparable score scales), [ADR-0013](0013-framework-adapter-spi.md) (pure TypeScript
framework-neutral orchestration), and
[ADR-0014](0014-adaptive-usage-ranking.md) (experimental-surface naming).

## Context

Hosts need to compare retrieval strategies without changing the result served to a user.
Today they build their own assignment, detached-shadow, comparison, and telemetry harnesses.
Kestral's reference implementation reached roughly 2,400 lines and 204 tests, demonstrating
both the demand and how many edge cases a one-off harness has to rediscover.

The initial `defineExperiment` proposal left several load-bearing details ambiguous:
deterministic split assignment had no type, readiness had no enforceable deadline field,
detached work had no drain handle, and outcome reports had no repeat or correlation
semantics. Building the reference implementation also exposed concrete correctness issues:

- `ratel.search` is already the capability-search span and cannot also name an experiment
  arm.
- `context.bind(ctx, anAlreadyCreatedPromise)` does not bind detached work.
- baggage propagation disappears without a registered OpenTelemetry `ContextManager`, even
  when a trace provider exists.
- result-level facts cannot be represented on ranked items when a ranking is empty.
- score values cannot be compared across BM25, cosine, and RRF scales.

This record is the normative contract for the first SDK implementation and its telemetry
vocabulary.

## Decision

### Boundary, terminology, and release posture

Experiments are a **pure TypeScript SDK facility** over host-supplied asynchronous functions.
They do not route through the Rust retrieval core, and this decision adds no Python experiment
surface. The cross-language telemetry packages mirror only the wire vocabulary.

An **experiment arm** is one named host-supplied selector. It is unrelated to
ADR-0014's internal ranking *arm*, which is one signal fused inside a catalog. A **shadow**
is an experiment-arm dispatch whose result is not intended to serve at dispatch time; it is
unrelated to tool-id shadowing. The **assigned arm** is selected explicitly or by the split.
The **effective arm** is the arm whose transformed result reaches the caller after fallback.
The dispatch **role** is immutable and is exactly `serving | shadow`; there is no `served`
spelling. A shadow run reused as fallback remains role `shadow`, while `effectiveArm` records
that its result served.

The first public behavior entry point is
**`experimentalDefineExperiment`**, exported from `@ratel-ai/sdk`'s root. There is no stable
`defineExperiment` alias yet. Passive types keep ordinary names (`Experiment`,
`ExperimentConfig`, `ExperimentSplit`, and so on), and methods on the returned experimental
object keep ordinary verbs. Promotion later drops the factory prefix. Run-scoped
`Ratel.experiment()` sugar is deferred: it cannot be vendored without augmenting the already
published `Ratel` interface and creating a cutover collision.

### Public contract

The following sketch fixes public field names and behavior. Generic details may be refined
without changing the semantics below.

```ts
export type ExperimentArmRole = "serving" | "shadow";
export type ExperimentArmOutcome = "ok" | "empty" | "timeout" | "error";

export interface ExperimentRankedItem {
  id: string;
  score?: number;
  attrs?: Record<string, string | number | boolean | null>;
}

export type ExperimentSplit<Arm extends string = string> = readonly Readonly<{
  arm: Arm;
  weight: number;
}>[];

export type ExperimentReportedOutcome =
  | { label: string; score?: number }
  | { label?: string; score: number };

export type ExperimentEvaluationReference =
  | "peer-selection"
  | {
      kind: "invocation";
      window: { turns?: number; maxAgeMs?: number };
      attribution?: "last-selection" | "all-in-window";
    };

export interface ExperimentSelection<Result, Arm extends string = string> {
  result: Result;
  selectionId: string;
  assignedArm: Arm;
  effectiveArm: Arm;
  durationMs: number;
}

export interface Experiment<Params, Result, Arm extends string = string> {
  select(
    params: Params,
    options: {
      unitId: string;
      arm?: Arm;
      shadow?: boolean;
      k?: number;
      transform?: (result: Result) => Result;
      attributes?: Record<string, string | number | boolean>;
    },
  ): Promise<ExperimentSelection<Result, Arm>>;

  reportInvocation(args: { unitId: string; toolId: string; turn?: number }): void;
  reportOutcome(args: { selectionId: string } & ExperimentReportedOutcome): void;
  warm(): Promise<void>;
  drain(): Promise<void>;
}

export interface ExperimentConfig<Params, Result, Arm extends string = string> {
  id: string;
  arms: Record<
    Arm,
    {
      select: (
        params: Params,
        context: { role: ExperimentArmRole },
      ) => Promise<Result>;
      warmup?: () => Promise<void>;
    }
  >;
  split?: ExperimentSplit<Arm>;
  ranking: (result: Result) => ExperimentRankedItem[];
  evaluation: {
    k?: number;
    references: readonly ExperimentEvaluationReference[];
    attributes?: (
      result: Result,
    ) => Record<string, string | number | boolean | null>;
    outcome?: boolean;
  };
  fallbackArm?: Arm;
  shadowPolicy?: { concurrency?: number };
}

export declare function experimentalDefineExperiment<
  Params,
  Result,
  Arm extends string,
>(
  config: ExperimentConfig<Params, Result, Arm>,
): Experiment<Params, Result, Arm>;
```

`shadowPolicy` deliberately names the define-time capacity policy; `select({ shadow })` is the
per-call boolean toggle. Using `shadow` for both is rejected.

Factory validation is synchronous. A malformed split, empty arm set, unknown fallback,
invalid define-time `k`, invalid concurrency, or invalid evaluation window throws before an
`Experiment` is returned. Call-level relational errors (unknown explicit arm, neither explicit
arm nor split, or shadow requested with one arm) throw synchronously before any arm starts.
Caller attributes that collide with a fixed experiment or standard error key also throw
synchronously; controlled telemetry can never be overwritten.

### Deterministic assignment

`ExperimentSplit` is ordered integer weight allocation:

- It names every declared arm exactly once. A weight is a non-negative safe integer; the total is
  positive and also no greater than `Number.MAX_SAFE_INTEGER`. A zero-weight arm remains
  available for explicit selection, fallback, and shadow evaluation.
- `select({ arm })` is an explicit override and wins over `split`. If `arm` is absent, a valid
  split is required. A split is never accepted and ignored.
- Assignment hashes the UTF-8 bytes of `JSON.stringify([experimentId, unitId])` with SHA-256,
  interprets the full digest as an unsigned big-endian integer, and takes it modulo the total
  weight. The first allocation whose cumulative weight exceeds that bucket wins. Array order
  is therefore part of the published configuration.
- The same experiment id, unit id, allocations, and order produce the same arm across processes.
  Changing the experiment id, weights, or order is allowed to move cohorts.

The SDK passes the exact `params` object to every arm; assignment never rewrites arm parameters.
The raw `unitId` remains in process. Telemetry and in-memory joins use the first 16 lowercase
hexadecimal characters of `SHA-256(unitId)`, which is pseudonymous rather than anonymous.

Every selection also receives an opaque UUID `selectionId`. It is returned to the caller and
stamped on all arm telemetry. Unlike the unit window used for invocation attribution, it gives
outcome reports and descendant framework spans an exact join key.

### Serving, fallback, and transforms

The assigned arm starts first with role `serving`. A fallback runs only when that arm rejects;
an empty successful ranking does not trigger fallback. `fallbackArm` must name a declared arm.
If it equals the failed assigned arm, the original error is rethrown rather than retrying the
same work.

When the fallback arm was admitted as a shadow by the same selection, that run is reused whether
its arm promise is still pending or has already settled; the SDK never invokes it twice. It keeps
role `shadow`. A capacity-skipped shadow was never dispatched and therefore starts fresh as
fallback with role `serving`, outside shadow capacity. If fallback also rejects, its error reaches
the caller. `assignedArm` remains the original assignment and `effectiveArm` names the result
actually returned. `durationMs` is that effective arm's measured duration, not total `select()`
wall time.

`transform` is synchronous and applies identically to every successful arm result **before**
ranking, result attributes, outcome classification, comparison, telemetry, and the value
returned from `select()`. A serving or fallback transform failure propagates: returning the
untransformed value could expose data the transform was meant to hide. A detached-shadow
transform failure is contained and, when peer comparison is configured, recorded as a dropped
comparison.

`ranking` and `evaluation.attributes` are observational projections. Their failures are
diagnosed and omit the affected evaluation channel; they never replace or reject a successfully
transformed serving result. A failed ranking projection classifies that arm as `error` and drops
its peer comparison when configured. It records `ratel.experiment.ranking_error`, omits
`hit_count` and the results EventRecord, marks the arm span `ERROR`, and still returns the
transformed value. A failed result-attribute projection records
`ratel.experiment.result_attributes_error` and omits only result-attribute agreement.
The SDK calls `evaluation.attributes` only when `"peer-selection"` is configured. It projects
eagerly once for every arm whose ranking succeeds, while that arm span is open, even if the arm
is later consumed as fallback or otherwise produces a drop instead of a comparison.

### Warmup, cold state, readiness, and outcomes

The SDK deduplicates concurrent warmups per arm. `warm()` starts every unresolved declared
warmup concurrently and never rejects. A successfully resolved warmup remains memoized for that
experiment instance, so later `warm()` calls do not rerun it; a failed one is retried.
`select()` lazily starts warmup for each arm it actually dispatches: the assigned arm, every
capacity-admitted shadow, and a fresh fallback. It does not start warmup for capacity-skipped
shadows. No dispatch waits for warmup, regardless of role.

An arm is `cold` for a dispatch exactly when it declares `warmup` and that warmup had not
resolved when `select()` began. The arm's **warmup callback** must reject on failure. The SDK
catches that rejection, keeps the arm cold, clears the memoized promise, and permits the next
`warm()`/`select()` to retry. A warmup callback that catches its own failure and resolves violates
the contract by falsely marking the arm warm.

The SDK does not impose a generic readiness timer. It owns warmup deduplication and cold tracking;
the arm owns its readiness/execution deadline and may choose a different deadline from
`context.role` (for example, short for `serving`, long for `shadow`). Wrapping an arbitrary
promise in an SDK timer would not cancel the work. SDK-owned deadlines require a future
`AbortSignal` contract and a new decision.

Arm outcomes are:

- `ok`: selection resolved and its transformed ranking is non-empty.
- `empty`: selection resolved and its transformed ranking is empty.
- `timeout`: selection rejected with a value whose `name` is exactly `TimeoutError`.
- `error`: any other rejection, a transform failure, or a ranking projection failure.

The name-based timeout convention interoperates with host error classes; no Ratel-specific
timeout class is required.

### Shadow scheduling and lifecycle

`select({ shadow: true })` attempts every arm other than the assigned arm in the
own-enumerable `Object.keys(config.arms)` order defined by ECMAScript. `false` or omission starts
none. The serving arm starts before any shadow.

`shadowPolicy.concurrency` is a positive integer, defaults to one, and applies across concurrent
calls on one experiment instance. Capacity is **skip, never queue**: an arm attempted with no
slot is not invoked and emits a skip record. Separate experiment instances have separate
capacity. A slot is acquired immediately before calling a shadow arm's `select` callback and is
released as soon as that callback's returned promise settles (or the callback throws).
Transform, ranking, result-attribute projection, arm telemetry, and any continuation waiting for
the serving result run after release and do not hold shadow capacity.

`select()` returns after the effective transformed result is ready; it does not await detached
shadows. Shadow rejection and comparison failure are contained and recorded. Telemetry failures
are contained and dropped because the failing telemetry path cannot reliably diagnose itself.
None can escape as an unhandled rejection. Capacity and spans are released in `finally` paths.

`drain()` snapshots admitted shadow-arm promises plus their transform, comparison, and telemetry
continuations started by that experiment before the call, waits for them with all-settled
semantics, and always resolves. It does not directly snapshot warmup or serving-arm promises, but
a snapshotted comparison continuation can await that selection's serving/fallback decision and
therefore wait indirectly on serving work. A shadow later reused as fallback remains in any
earlier snapshot. `drain()` is repeatable and does not close the experiment or wait for work
started after the snapshot. A host applies its own shutdown budget. With no cancellation
contract, `drain()` can wait forever for an arm that never settles; the arm-owned deadline is the
protection against that.

### Comparison and attribution

`select({ k })` overrides `evaluation.k`; the default is 10. A define-time value must be an
integer at least one. An invalid request value is observational input, so it is ignored in favor
of the valid define-time value or default rather than failing a live selection. `k` only sets the
Jaccard window; it never truncates the result returned to the caller.

Peer comparison is rank-based:

- top-1 agreement compares the first ids; two empty rankings agree;
- exact-order agreement compares the complete ordered id lists;
- overlap count uses the complete de-duplicated id sets;
- Jaccard@K first slices each ordered list to K, then de-duplicates that slice into a set; it is
  one when both sets are empty.

Scores are emitted as measurements but never compared: BM25, cosine, and RRF scales are not
commensurate (ADR-0011 and ADR-0014).

`ExperimentRankedItem.attrs` remains a separate item-level channel. The rank-zero items are
compared over keys present on both sides using strict scalar equality; an empty side makes no
item-attribute claim.
`evaluation.attributes(result)` adds result-level facts after transform. Those maps are compared
over the **union** of keys: a key present on one side only disagrees, `null` is distinct from
missing, and present primitive values use strict equality. This preserves facts such as
`isSequence: false` and `appliedDomain: null` even when a ranking is empty. Item- and
result-level agreement are emitted separately.

When `"peer-selection"` is present in `evaluation.references`, each completed shadow is compared
once against the **effective served result**. Without that reference, shadows still run and emit
their own arm/results telemetry but produce neither comparison nor drop records. There is never a
shadow-vs-shadow comparison. For each peer-eligible admitted shadow, exactly one comparison or one
drop is emitted, with this reason precedence:

1. `arm-failed` when that shadow's select, transform, or ranking failed;
2. `fallback-consumed` when that shadow became the effective fallback;
3. `selection-failed` when no arm produced an effective result;
4. `served-ranking-failed` when an effective transformed result served but its ranking failed;
5. `comparison-failed` when comparison itself failed.

Otherwise the shadow compares to the effective served ranking. A shadow consumed as fallback has
no self-comparison; other completed shadows compare against it. This terminal record lets every
continuation and `drain()` finish even when serving fails.

An invocation reference has this shape:

```ts
type ExperimentEvaluationReference =
  | "peer-selection"
  | {
      kind: "invocation";
      window: { turns?: number; maxAgeMs?: number };
      attribution?: "last-selection" | "all-in-window";
    };
```

`window.turns` means the last N completed selections recorded for that unit, positionally; it is
not a caller turn ordinal or a process-global counter. If both bounds exist, a selection must be
inside both the positional and age windows. Stateless or distributed hosts rely on `maxAgeMs`.
`reportInvocation({ turn })` may emit the supplied turn as telemetry but never uses it for
windowing. The in-process buffer is bounded and best-effort; a missing match emits an unattributed
invocation rather than throwing.

Only selections whose ranking projection succeeded enter this buffer or count toward
`window.turns`. A ranking-failed selection has no truthful tool rank and is therefore invisible to
invocation attribution; a later report may be unattributed rather than encoding an unknown rank
as `-1`. The SDK timestamps a buffered selection immediately before its successful `select()`
promise fulfills. `age_ms` and `maxAgeMs` use elapsed time from that timestamp to
`reportInvocation`; a negative clock delta is clamped to zero.

`evaluation.references` is non-empty and contains at most one `"peer-selection"` entry and at
most one invocation entry; duplicates are configuration errors. An invocation window requires at
least one bound.
`turns` is a positive safe integer, `maxAgeMs` is a positive finite number, and attribution
defaults to `"last-selection"`. With no invocation reference, `reportInvocation` is a no-op. With
`"last-selection"` it emits one record for the newest match; with `"all-in-window"` it emits one
record per match; either mode emits one unattributed record when there is no match.

When `evaluation.outcome` is true, `reportOutcome` accepts the exact `selectionId` returned by
`select`. `label` is a free, non-empty string, `score` is a finite number, and at least one is
required; both may be present. Invalid arguments fail synchronously. Valid reporting is
fire-and-forget and telemetry failures do not reach application code. Every call is an
append-only observation. Repeated reports are distinct records, never last-wins; future
idempotence would require an explicit outcome id. Calling `reportOutcome` when
`evaluation.outcome` is false or omitted is a synchronous configuration error. Runtime validation
checks only that `selectionId` is a non-empty string; the SDK does not retain an unbounded issued-id
set or reject ids from an earlier process. Passing an id returned by this experiment is the
caller's obligation, and the telemetry join reveals an unknown id as unattributed.

### OpenTelemetry contract

Every arm run opens one span named **`ratel.experiment.arm`** from tracer scope
**`@ratel-ai/sdk`**. The logger scope is the same. `ratel.search` remains exclusively the
capability-search span.

The following string keys are set directly on every arm span and placed in OpenTelemetry baggage
for descendant correlation:

| Key | Value |
|---|---|
| `ratel.experiment.id` | configured experiment id |
| `ratel.experiment.selection_id` | opaque selection UUID |
| `ratel.experiment.arm` | this run's declared arm |
| `ratel.experiment.role` | `serving \| shadow` |
| `ratel.experiment.unit` | 16-hex unit hash |

The attribute constants and baggage-key constants intentionally resolve to the same strings.
Only these controlled string values enter baggage; caller attributes do not.

The span kind is `INTERNAL`. In addition to the five required stamp strings, its fixed schema is:

| Attribute | Type and presence |
|---|---|
| `ratel.experiment.cold` | required boolean at dispatch |
| `ratel.experiment.outcome` | required `ok \| empty \| timeout \| error` at completion |
| `ratel.experiment.duration_ms` | required non-negative number measuring only the arm callback |
| `ratel.experiment.hit_count` | non-negative integer when ranking succeeds; absent on ranking failure |
| `ratel.experiment.ranking_error` | error type on ranking failure; otherwise absent |
| `ratel.experiment.result_attributes_error` | error type when the result-level projector fails; otherwise absent |
| `ratel.experiment.result_attrs_encoding_error` | error type when gated item attrs cannot be encoded; otherwise absent |

`ok` and `empty` set span status `OK`. `timeout` and `error` set span status `ERROR`, even for a
best-effort shadow or a failure absorbed by fallback: the span describes that arm operation, not
the eventual caller impact. Recovery is expressed by the fallback/comparison records.
For an arm-callback rejection, a non-empty string `name` property drives timeout classification.
For every failure's `error.type` and the `ranking_error`, `result_attributes_error`, and
`result_attrs_encoding_error` diagnostic values, that property also wins; otherwise the type is
its `Error.name` or, for a non-`Error` value, `typeof value`. A plain object named `TimeoutError`
is therefore a timeout only when the arm callback rejects with it; a transform or projection
throwing the same object remains an `error`. Error instances are also recorded as exceptions,
and the status message is their message; non-Error values use `String(value)`.

`ratel.experiment.effective_arm` is selection-level and never placed on an arm span after the
fact. Doing so would require holding a failed or already-settled span open through fallback and
would make span duration cease to mean arm latency.

Experiment events are OpenTelemetry Logs **EventRecords**, never SpanEvents:

- `ratel.experiment.results`: one for every arm whose ranking projection succeeds, including an
  empty ranking; full five-key arm stamp; explicit context of that arm span.
- `ratel.experiment.comparison`: one per eligible completed shadow when peer comparison is
  configured; full shadow-arm stamp; explicit shadow-arm context.
- `ratel.experiment.skip`: one per capacity-skipped arm; stamp of the assigned serving arm plus
  the skipped arm field; explicit assigned-arm context.
- `ratel.experiment.fallback`: one per successful fallback; stamp of the failed assigned arm plus
  fallback fields; explicit failed-arm context, even if its span has ended.
- `ratel.experiment.drop`: one per peer-eligible admitted shadow that produces no comparison;
  full shadow-arm stamp; explicit shadow-arm context.
- `ratel.experiment.invocation`: zero when invocation evaluation is disabled; otherwise one per
  attributed match, or one unattributed record. It carries experiment id + unit hash, plus
  selection id/effective arm only when attributed, and uses the active context at report time.
- `ratel.experiment.outcome`: one per valid call; experiment id + supplied selection id; active
  context at report time.

Fixed event attributes are:

- results: `ratel.experiment.result_ids`, `ratel.experiment.result_scores`,
  `ratel.experiment.result_attrs`;
- comparison: `ratel.experiment.served.arm`, `ratel.experiment.served.outcome`,
  `ratel.experiment.served.duration_ms`, `ratel.experiment.served.hit_count`,
  `ratel.experiment.shadow.arm`, `ratel.experiment.shadow.outcome`,
  `ratel.experiment.shadow.duration_ms`, `ratel.experiment.shadow.hit_count`,
  `ratel.experiment.agreement.top1`, `ratel.experiment.agreement.exact_order`,
  `ratel.experiment.agreement.overlap_count`, `ratel.experiment.agreement.jaccard_at_k`,
  `ratel.experiment.agreement.k`, `ratel.experiment.agreement.item_attrs`, and
  `ratel.experiment.agreement.result_attrs`;
- lifecycle: `ratel.experiment.skip.arm`, `ratel.experiment.skip.concurrency`,
  `ratel.experiment.skip.reason`, `ratel.experiment.fallback.effective_arm`,
  `ratel.experiment.fallback.reused_shadow`, and `ratel.experiment.drop.reason`;
- invocation: `ratel.experiment.invocation.attributed`,
  `ratel.experiment.invocation.rank`, `ratel.experiment.invocation.age_ms`,
  `ratel.experiment.turn`, `ratel.experiment.effective_arm`, plus standard
  `gen_ai.tool.name`;
- reported outcome: `ratel.experiment.outcome.label` and
  `ratel.experiment.outcome.score`.

Result ids are a required ordered `string[]`, preserving duplicates and ranking order. Scores are
a `number[]` only when every ranked item supplies a finite score; otherwise the entire field is
absent. Result attrs preserve positional alignment with ids. Comparison arm/outcome fields are
strings, durations are non-negative numbers, hit/overlap counts and K are non-negative integers,
agreement flags are booleans, and Jaccard is in `[0, 1]`. Item/result attribute agreement fields
are structured maps from projected key to boolean, avoiding dynamic OTel attribute names.
`agreement.item_attrs` is present when both rankings contain a rank-zero item and is `{}` when
those items have no shared attribute keys; it is absent when either ranking is empty.
`agreement.result_attrs` is present when the projector is configured and succeeds for both
results, including `{}` when both projected maps are empty; it is absent when unconfigured or
either projection fails.

Skip reason is exactly `capacity`. Drop reason is exactly
`arm-failed | fallback-consumed | selection-failed | served-ranking-failed | comparison-failed`,
with the precedence defined above. Fallback reuse is a boolean.

An attributed invocation record sets `ratel.experiment.invocation.attributed = true` and requires
`ratel.experiment.selection_id`, `ratel.experiment.effective_arm`,
`ratel.experiment.invocation.rank`, and `ratel.experiment.invocation.age_ms`. Rank is `-1` when the
tool is absent from a successfully projected ranking and otherwise the zero-based index of its
first occurrence; age is non-negative milliseconds. An unattributed record sets
`attributed = false` and omits all four fields. Both forms carry caller turn when supplied, as an
integer. Outcome label and score obey the public validation above.

Phase 2 mirrors every fixed experiment name in the Rust, TypeScript, and Python vocabulary
packages, plus the closed role, outcome, skip-reason, and drop-reason wire values. Borrowed
`gen_ai.tool.name` retains its existing constant. Standard `error.type` keeps its OTel semantics
and is emitted directly rather than duplicated into the Ratel vocabulary.

The SDK constructs baggage from the active parent context, starts the arm span with that explicit
context, sets the span into the derived arm context, and starts arm work inside:

```ts
context.with(armContext, () => startDetachedWork())
```

It preserves all unrelated parent baggage and overwrites the five controlled experiment keys for
the inner arm context. A nested experiment therefore sees its own stamp inside its subtree; leaving
`context.with` restores the outer context. Arm-correlated EventRecords are emitted with their
explicit stored arm context, even if the span has already ended, so logging cannot inflate arm
duration. Invocation and outcome reports use the active context at report time and their explicit
join attributes; they never pretend a stale arm is active.

It never creates a promise first and passes it to `context.bind`; binding an already-created
promise is a no-op. Hosts must register a `ContextManager` unconditionally, independent of
whether an exporter is configured. Without one, direct arm attributes still survive when a
trace provider records the span, but baggage and active-parent propagation to descendants do
not. With no registered providers the entire telemetry path is a clean no-op.

The SDK never registers providers or exporters. Hosts that want EventRecords must configure a
log-record processor as well as span processors.

### Content and callback boundaries

The experiment layer is structurally query-blind: it does not inspect or emit `params`, the raw
query, or the raw result. Ranked ids and scores are declared measurement identifiers and are
emitted even when content capture is off. Callers must therefore use stable non-content ids.

Arbitrary per-item `attrs` are content. They follow
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`: `SPAN_ONLY` and `SPAN_AND_EVENT` place
`ratel.experiment.result_attrs` on the arm span as a JSON string; `EVENT_ONLY` and
`SPAN_AND_EVENT` place the same id-aligned array as structured AnyValue on
`ratel.experiment.results`; `NO_CONTENT` emits neither. Missing item attrs occupy `null` positions.
The existing Python heterogeneous-array encoding (`ratel.type = array` plus indexed
`ratel.items`) applies when its OTel version cannot carry the structured array directly. Encoding
failure records `ratel.experiment.result_attrs_encoding_error` on the arm span and omits only
`result_attrs`; ids, scores, serving, and comparison continue. Result-level projection values are
never emitted; only agreement booleans are.

`select({ attributes })` is copied as non-content scalar telemetry to every arm span and
arm-correlated record emitted during that selection and is not capture-gated. Exact fixed
experiment keys, `gen_ai.tool.name`, and `error.type` are reserved; a collision throws before
dispatch. The caller is responsible for excluding content and high-cardinality personal data.

## Consequences

- One additive SDK primitive replaces host-specific assignment and shadow orchestration while
  preserving the byte-for-byte behavior of the retrieval methods it wraps.
- Assignment is reproducible and testable. Allocation order and experiment id are cohort inputs,
  so changing them is an explicit rollout change.
- Fallback, N-arm comparison, empty rankings, result-level parity, and delayed outcomes have one
  shared vocabulary across OTel backends and Ratel Cloud.
- The raw unit id never leaves the process, but its unsalted short hash is pseudonymous and
  cross-experiment joinable. Hosts should pass opaque stable ids rather than email addresses or
  other direct identifiers.
- Detached work cannot fail a served request and can be drained, but the SDK cannot cancel a host
  promise. Arms must enforce their own deadlines.
- Correct descendant correlation requires a host `ContextManager`; exporting comparison,
  invocation, and outcome EventRecords also requires a Logs provider/processor. Documentation
  must state both requirements.
- The telemetry vocabulary publishes before the SDK, and the SDK before framework adapters, per
  the existing release-order rule.

## Rejected

- **Reuse `ratel.search` for arm spans:** it collides with capability search and cannot truthfully
  supply `ratel.search.target`.
- **A stable `defineExperiment` or immediate `Ratel.experiment()` method:** violates the
  additive-experimental convention and makes vendored-stub cutover collide with module
  augmentation.
- **Rust-core orchestration:** arms are host functions and the orchestration is TypeScript state;
  crossing FFI adds no shared-engine value.
- **`context.bind` on detached promises or baggage-only stamping:** the former does nothing after
  promise creation; the latter loses the arm identity without a `ContextManager`.
- **SDK-owned readiness timers without cancellation:** they report a timeout while leaving the
  underlying work running. Arm-owned role-specific deadlines are honest.
- **Queued shadows:** they turn best-effort evaluation into delayed background load and make
  shutdown unbounded. Capacity skips instead.
- **Score-delta metrics:** the rankers' score scales are not comparable.
- **Content-gating ranked ids and scores:** those values are the experiment measurement and do not
  contain query or result bodies.
- **SpanEvents for experiment events:** ADR-0007's structured event channel is the Logs API.
- **Last-wins outcome reports:** already-exported observations cannot be overwritten honestly;
  repeats are append-only.
