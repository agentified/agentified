# 16. Experimental prompt compression: token-importance-aware LLMLingua-2

Date: 2026-07-30

## Status

Accepted

Inherits the model-loading posture of [ADR-0011](0011-selectable-retrieval-methods.md) (pure-Rust
Candle, no ONNX) and the model-configuration and download policy of
[ADR-0012](0012-configurable-embedding-models.md). Follows the experimental-prefix convention
established by [ADR-0014](0014-adaptive-usage-ranking.md) and [AGENTS.md](../../AGENTS.md).
Ratifies the prototype at `prompt-compression` (TypeScript, `@atjsh/llmlingua-2` over ONNX
Runtime) with the corrections below.

## Context

Every capability in this crate is **selection**: keep the catalog outside the context window and
retrieve only what the task needs. `lib.rs` states it as the crate's purpose. Nothing in the
engine rewrites text — retrieval picks *which* entries enter the prompt and stops there.

That leaves the larger half of the context window untouched. Agents carry prose we never see and
cannot select away: meeting transcripts, retrieved passages, pasted documents, prior-turn
summaries. Selection has no lever there, and the tokens are usually the ones that push a request
past its budget.

LLMLingua-2 is the obvious instrument: a BERT token classifier trained to label each token
INCLUDE or EXCLUDE, so `P(INCLUDE)` is a per-token importance score and a threshold on it is a
compressor. The prototype confirmed the win — a 566-token transcript to 234 tokens (2.42×) at
`rate=0.4` with BERT-base multilingual, retaining **6/6** hand-checked critical facts, where the
TinyBERT variant needed `rate=0.7` (barely any compression) to match.

It also confirmed that **rate is the wrong knob on its own**, and that the failures are not
tuning problems:

- `"translate this to french"` compresses to `"translate to"` at *every* rate from 0.4 to 0.9.
  Short prompts have no redundancy to spend; raising the rate does not recover them.
- Negations vanish. `"doesn't support"` → `"support"` inverts the claim, and the classifier has
  no reason to protect it because `doesn` and `t` are separately-scored subwords.
- Numbers fragment. `8,400` and `2.14` are multi-token, and a threshold applied per word
  cheerfully keeps half of each.
- Output is mangled independently of what was dropped: `pg _ upgrade`, `doesn ' t`, `4 , 200` —
  artifacts of reconstructing text by detokenizing rather than by slicing the original.

So the decision is not "run LLMLingua-2". It is what policy sits between the importance scores
and the output text.

## Decision

**Ship a standalone `PromptCompressor` in core (`ExperimentalCompression` in the SDKs): the
LLMLingua-2 importance signal plus a policy layer that makes the loss bounded, visible, and
steerable.** Six choices carry it.

### 1. Atoms, not words

Group words into **atoms** — maximal runs of words with zero bytes between them in the original
text — and keep or drop each atom whole. `doesn't` is one atom, not `["doesn", "'", "t"]`.
`8,400`, `2.14`, and `db.r6g.4xlarge` are each one atom.

This single rule fixes three of the four failure modes above at once, because all three are the
same bug: the model scores subwords, the policy decides on words, and the boundary between them
is where meaning leaks. Deciding at the atom makes the unit of the decision match the unit of
meaning.

One exception: atoms do not glue across CJK characters. WordPiece's `handle_chinese_chars` makes
each ideograph its own token with no separating bytes, so the general rule would collapse a
Chinese sentence into one unsplittable atom and compress nothing.

### 2. Protection, charged to the budget first

Some content must survive at any rate. Protected atoms take `importance = 1.0` and are always
kept: caller-supplied literal or regex spans, a negation list (`protect_negations`, **on** by
default), and optionally any atom containing a digit (`protect_numbers`, **off** by default).

That split is measured, not assumed, and it corrected this ADR's first draft. On the reference
transcript at rate 0.25, blanket digit protection marks 23 units — `Q3`, `12 terabytes`,
`November 2024`, `four hours` — and the budget they consume starves prose that matters more: it
loses **two** critical facts where leaving it off loses one, and it is never better at any rate
below 0.4. Negation protection marks three units and, at rate 0.15, is exactly the difference
between keeping and losing `doesn't support`.

The lesson generalizes: **protection is a budget tax, so it must be selective.** A rule that
fires on a whole syntactic class spends more than it saves. The number-splitting failure this
was originally meant to fix (`8,400` kept as `8` + `400`) is fixed by atoms, not by protection —
the two were conflated when this decision was drafted. A caller who needs specific figures names
them with `protect`, which is both cheaper and more precise.

The budget interaction is the load-bearing part:

- The budget is `B = round(rate · total_tokens)`. Protected cost is charged to `B` **first**.
- The threshold is computed over the **unprotected** atoms against the remainder. Protected
  atoms are excluded from the sample — leaving the injected `1.0`s in would inflate the upper
  tail and silently loosen compression for everything else.
- If protection alone exceeds `B`, keep exactly the protected atoms and set `budget_exceeded`.
  Protection is a hard promise; `rate` is documented as approximate.

The alternative — protection additive *on top* of the rate — was rejected because it makes
output size unbounded in the number of matches. A caller who protects `\d+` on a financial
document would get no compression and no signal that anything was wrong.

### 3. A token-weighted threshold, in the model's own tokens

The threshold is a quantile over importances weighted by the tokens each atom costs, which is
what makes `rate` approximate a *token* keep-ratio rather than a word one. The weights are the
compression model's own WordPiece counts, which the single encode already produced.

**No tiktoken dependency.** The quantile needs only relative weights; a BPE table would mean
either vendoring ~1.7 MB of ranks (against ADR-0011's never-bundle-weights posture) or a second
unpinned download, to serve a weighting heuristic. And o200k is the wrong target regardless:
Ratel's users prompt Claude, Gemini, and local models, so precision about OpenAI's tokenizer is
precision about the wrong thing. `rate` is documented as approximate, and `CompressionStats`
reports exact in/out counts so a caller can calibrate against their own tokenizer empirically.

This also removes the prototype's dominant cost. Its 2.7 s was roughly 80% accidental —
TensorFlow.js softmax on the JS CPU backend, plus a per-word tiktoken encode *inside* the
threshold loop. Its own leaner path, same model and input, ran in 574 ms.

### 4. Reconstruct from the original bytes

Output is a concatenation of slices of the input string, located by the tokenizer's byte
offsets. Never detokenize.

Tokenization is lossy — text cleaning, CJK space insertion, `[UNK]`, accent handling — and a
WordPiece decoder reinserts spaces by rule. Slicing the original makes "compression never
mangles text it kept" a structural property rather than a heuristic that mostly holds. The
separator between two kept atoms is the original span verbatim when nothing was dropped between
them, and otherwise collapses to a blank line, a newline, a space, or nothing, matching what the
skipped span contained.

### 5. Encode once; chunks are index ranges

BERT's 512-position limit forces chunking. Rather than slice text and re-encode, encode the whole
prompt once with truncation disabled and make chunk boundaries index ranges into that single
token stream; each row is `[CLS] ids[range] [SEP]` built from ids already in hand.

Offsets are then global by construction, no substring can tokenize differently in isolation than
in context, and the prototype's off-by-one chunk truncation — `slice(st, n - 1)`, which turned
`french` into `fren` and was the direct cause of its worst output — cannot recur, because nothing
is ever truncated. A property test asserts the chunks exactly partition the stream.

### 6. A minimum-length gate that returns the input untouched

Below `min_words` (pre-tokenizer, so no model is loaded) or `min_tokens` (exact), return the
input verbatim with `stats.gate` naming the reason. `rate >= 1.0` likewise.

The prototype established that short prompts are **out of domain, not under-tuned**. Returning
them unchanged is the only correct behavior, and surfacing *that it happened* is what stops a
caller from believing compression occurred when it did not.

### Model, and how it loads

The default is `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`, pinned to a
commit for reproducibility. It ships `model.safetensors` (709 MB) and a fast `tokenizer.json`,
and declares `"model_type": "bert"` — which matters, because `candle-transformers` provides only
`BertModel` (encoder-only) and its loader resolves the checkpoint's `bert.*` tensor prefix from
exactly that field. The `*ForTokenClassification` head it does not provide is a
`candle_nn::Linear` over `classifier.weight` / `classifier.bias`; that is the whole delta, and it
keeps the feature inside ADR-0014's encoder-only envelope. The checkpoint omits `num_labels` and
`id2label`, so the head width is read from the classifier tensor's shape and anything but two
labels is refused rather than scored against the wrong class.

Download policy is ADR-0012's, unchanged: the built-in default auto-downloads into the shared
HuggingFace cache; an explicit HuggingFace repo is cache-only unless `download: true`; a local
path never touches the network. The one judgement call is that this default is ~5× the default
embedding model. Auto-download is still right, because unlike embeddings **nothing in Ratel
invokes compression implicitly** — a caller who reaches `compress()` has asked for precisely
this — but it warrants a louder pre-fetch notice and a `preload()` so a server can pay the cost
deliberately rather than inside the first request.

### Where it sits

`PromptCompressor` is standalone and orthogonal to catalogs, registries, and search. Nothing
loads, reads a file, or touches the network unless a caller constructs one and calls it; a test
pins that tool search is byte-identical with the module linked.

The SDK entry points carry the `experimental` prefix (`ExperimentalCompression`), per ADR-0014's
convention: core ships plain names, the marker sits at the SDK boundary, and it is removed on
promotion or the surface is removed. Compression is lossy by nature and we have no eval data yet
beyond one prototype fixture, so the marker is honest rather than ceremonial.

## Consequences

- The engine gains its first **transformation** of text. Every prior capability chose among
  existing strings; this one produces a string that did not exist. Selection is verifiable by
  construction (an id either is or is not in the catalog); compression is not, which is why the
  explainability channel and the gate are part of the decision rather than conveniences.
- **`compress()` can make an agent worse, and nothing here prevents that.** The gate, the
  protections, and the per-atom `kept`/`dropped` scores make the loss visible and steerable; they
  do not make it safe. This is the first surface in the crate where correct use requires the
  caller to evaluate the output on their own data.
- **A second model class enters the crate**, roughly 5× the default embedder. Peak memory and
  cold-start for a process that uses both are now dominated by compression.
- Compression is stateless per call, so it needs none of the embedding path's machinery — no
  dimension checks, no model-mismatch pausing, no rebuild. It also must not inherit the
  registry's dense gating: it touches no registry state, so serializing it against rebuilds would
  buy nothing and cost latency.
- Quality outside the training domain is **unmeasured**. The classifier is multilingual but was
  tuned on English meeting transcripts, and the built-in negation list is English. Code,
  structured payloads, and non-English prose are documented as unknown rather than implied.
- The defaults (`rate = 0.40`, `min_tokens = 50`, `min_words = 40`, `max_chunks = 16`) come from
  one prototype fixture and judgement. They are **unswept**, in the same sense ADR-0014 says of
  its own constants. `protect_numbers = false` is the exception: it was measured.

### Measured results

On the reference transcript (`src/core/tests/fixtures/transcript.txt`), default model, release
build, 8-core M-series laptop (4 performance cores):

| rate | model tokens | ratio | critical facts |
|---|---|---|---|
| 0.20 | 623 → 125 | 4.98× | 3 / 6 |
| 0.33 | 623 → 206 | 3.02× | 6 / 6 |
| **0.40** | **623 → 249** | **2.50×** | **6 / 6** |
| 0.50 | 623 → 312 | 2.00× | 6 / 6 |
| 0.70 | 623 → 436 | 1.43× | 6 / 6 |

The prototype's ONNX path measured 566 → 234 o200k tokens (2.42×) at the same rate; the counts
differ because these are mBERT WordPiece tokens, the ratio agrees. Output is free of the
prototype's detokenizer damage: `pg_upgrade`, `db.r6g.4xlarge`, `doesn't`, `8,400`, and `2.14`
all survive intact.

**Latency is the weak result and is not hidden.** Median **~2.0–2.6 s** for a 623-token document
(min ~1.6 s, 2 chunks), against **574 ms** for the prototype's ONNX Runtime path on the same
model and input — roughly **3.5–4.5× slower**. Grouping equal-length chunks so a short tail is
not padded to a full 512 positions recovered ~13%; the remainder is Candle's CPU GEMM against
ONNX Runtime's, and closing it would mean reversing this ADR's first rejected option. Threading
matters: performance is flat from 4 threads up, and `RAYON_NUM_THREADS=1` roughly doubles the
time, so a request-per-thread server should pin it rather than let every request oversubscribe.

This is above the bar set when the work was planned (~1.5 s), which is why:

- the module documents the number rather than implying compression is cheap,
- `preload()` exists and is recommended for any server,
- and **promotion out of `experimental` should not happen until either a distilled checkpoint or
  a faster CPU path closes the gap.** `CompressionModel::{HuggingFace, Local}` is the escape
  hatch that lets a user do the former today without a core change.
- Two shared HuggingFace-cache helpers move out of `embedding.rs` into a common module so the
  cross-process lock-retry logic exists once. This is the only change to an existing code path,
  and `embedding.rs`'s tests passing unchanged is the evidence it was behavior-preserving.

## Rejected

- **ONNX Runtime (`ort`), as the prototype used.** It would be faster — the prototype's lean ONNX
  path ran 574 ms where Candle is expected to be 1.5–3× that — but ADR-0011 excluded C++ native
  deps so the wheels and node addons stay cleanly cross-compilable, and the SDKs ship prebuilt
  binaries for five targets. A speed win on one feature is not worth reversing the crate's
  distribution posture. If the measured latency proves unacceptable, the escape hatch is a
  distilled checkpoint through the existing model config, not a second inference runtime.

- **`search(query, { compress: true })`.** `SearchHit` carries `tool_id` and `score` and no text
  at all, so there is nothing at that layer to compress; serving the flag would mean moving
  description text into the hit purely to feed it. Independently: search is the hot path (BM25 is
  sub-millisecond), and a boolean that silently attaches a multi-second model forward pass to it
  is a trap — this crate already separates expensive model work by name and by await. And it
  would recompress unchanged text on every call.

- **Compression inside `ToolCatalog`.** Not because descriptions are always short — some upstream
  MCP servers ship 500+ token descriptions that genuinely warrant it — but because `ToolCatalog`
  is stateful (registry, embedding cache, dense gating, busy semantics) and compression is
  stateless. Folding a 709 MB model handle into that lifecycle entangles two for no gain. The
  right form of that feature is compression **at registration time**, computed once and cached,
  which this decision leaves open and does not foreclose.

- **Compressing skill bodies at `get_skill_content` time.** The best-fitting long prose in the
  repo, and deliberately deferred: skill bodies are imperative instructions, where a dropped
  negation changes what the agent *does* rather than what it knows. Revisit when the protection
  layer has eval data behind it.

- **Per-segment budget allocation** (allocate the keep-budget per sentence, with a floor, so a
  global threshold cannot erase a whole sentence). A real improvement and a likely v2. Held back
  because it adds largest-remainder apportionment plus floor-deficit reclamation — meaningful
  logic — for a failure mode the fixture does not exhibit, and the four choices above are the
  ones the prototype's evidence actually demands.

- **An OpenAI-compatible endpoint variant**, mirroring `embedding_config.rs`. Nothing standard
  returns per-token classification logits; `/embeddings` returns vectors. There is no protocol to
  speak, so the variant would be a shape with no implementation behind it.

- **Quantization or fp16 in v1.** `candle-transformers` ships no quantized BERT
  token-classification path, and `DType::F16` on CPU is *slower* in Candle (gemm upcasts) while
  halving memory — a memory knob presented as a speed knob. Revisit against a measurement.

- **The XLM-RoBERTa-large LLMLingua-2 checkpoint.** Larger, not smaller, and would need a
  `RobertaModel` that `candle-transformers` does not provide. The multilingual BERT-base
  checkpoint is the one the prototype measured.
