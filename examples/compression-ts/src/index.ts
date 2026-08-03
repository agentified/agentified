/**
 * Compression on ten coding-task queries: before, after, and latency.
 *
 *   pnpm --filter @ratel-ai/example-compression start
 *
 * Downloads the ~700 MB compression model on first run.
 *
 * The min-length gate is disabled here so every query actually compresses — by
 * default half of these are short enough that Ratel returns them verbatim.
 */
import { type CompressionOptions, ExperimentalCompression } from "@ratel-ai/sdk";

const QUERIES: [string, string][] = [
  ["terse-imperative", "fix the failing test"],
  ["short-howto", "How do I reset a git branch to match origin/main exactly?"],
  ["short-why", "Why is my React component re-rendering on every keystroke?"],
  [
    "negated-constraint",
    "Refactor this handler to use async/await instead of promise chains, but do not " +
      "change the public function signature and do not add any new dependencies.",
  ],
  [
    "versioned-upgrade",
    "We're on TypeScript 5.2 and want to move to 5.7. Our build uses ts-node with " +
      "CommonJS output and we have a few files using experimental decorators. What " +
      "breaks, and in what order should we do the upgrade?",
  ],
  [
    "stack-trace-debug",
    "I'm getting `TypeError: Cannot read properties of undefined (reading 'map')` in " +
      "src/components/UserList.tsx line 42, but only in production builds, never in dev. " +
      "The component fetches from /api/users which returns 200 with a valid JSON array " +
      "when I curl it. What should I check first?",
  ],
  [
    "perf-with-numbers",
    "Our Postgres query that joins orders to line_items takes 4.2 seconds on a table " +
      "with 12 million rows. There's a btree index on orders.created_at but the planner " +
      "is doing a sequential scan. EXPLAIN ANALYZE shows the filter removing 11.8 " +
      "million rows. How do I get it under 200ms?",
  ],
  [
    "multi-constraint-refactor",
    "Split this 800-line module into smaller files. Keep the public exports identical " +
      "so no caller has to change, don't introduce a barrel file, put each React hook in " +
      "its own file under hooks/, and leave the existing tests passing without editing them.",
  ],
  [
    "build-failure",
    "CI fails with `error: linking with cc failed: exit status: 1` and `undefined " +
      "reference to pthread_create` when cross-compiling to aarch64-unknown-linux-gnu. " +
      "It builds fine on x86_64 and on macOS locally. The crate depends on openssl-sys " +
      "0.9. How do I fix the cross build?",
  ],
  [
    "review-request",
    "Review this pull request for correctness and security. It adds a new endpoint " +
      "POST /api/admin/users that creates users, reads the role from the request body, " +
      "and does not currently check whether the caller is an admin. Tell me what's wrong " +
      "and how you'd fix it.",
  ],
];

/**
 * Words too common to carry topic. Dropping these is compression working, so
 * the content-only column excludes them.
 */
const STOPWORDS = new Set([
  "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "at", "for", "with", "as",
  "is", "are", "was", "were", "be", "it", "its", "this", "that", "i", "we", "you", "my", "our",
  "so", "do", "does", "did", "has", "have", "how", "what", "when", "should", "would", "can",
  "from", "by", "into", "there", "me",
]);

/** Normalize a unit to a comparable word; `null` for punctuation-only. */
function toWord(raw: string): string | null {
  const w = [...raw.toLowerCase()]
    .filter((c) => /[\p{L}\p{N}_.\-/]/u.test(c))
    .join("")
    .replace(/^[._\-/]+|[._\-/]+$/g, "");
  return w === "" ? null : w;
}

/**
 * Jaccard similarity between the input's and the output's word sets.
 *
 * Output is a subsequence of the input, so `set(out)` is always a subset of
 * `set(in)` — the intersection *is* the output set and the union *is* the input
 * set. Jaccard therefore reduces to `|out| / |in|`, i.e. word recall.
 *
 * Sets are distinct words, so dropping a repeated word costs nothing — which is
 * the point: repetition is the redundancy compression is meant to remove.
 */
function jaccard(kept: string[], dropped: string[], contentOnly: boolean): number {
  const keep = (w: string) => !contentOnly || !STOPWORDS.has(w);
  const all = new Set([...kept, ...dropped].filter(keep));
  const surviving = new Set(kept.filter(keep));
  return all.size === 0 ? 1 : surviving.size / all.size;
}

const compressor = new ExperimentalCompression();

console.log("loading model…");
const cold = Date.now();
await compressor.preload();
console.log(`ready in ${Date.now() - cold} ms\n`);

// minWords/minTokens at 0 turns the gate off so every query compresses.
const options: CompressionOptions = { minImportance: 0.5, minWords: 0, minTokens: 0 };

// One warm-up so the first query isn't charged for lazily-faulted pages.
await compressor.compress(QUERIES[0][1], options);

const rows: [string, number, number, number, number, number][] = [];
for (const [name, query] of QUERIES) {
  // Median of 5 so a single scheduling hiccup doesn't set the number.
  const timings: number[] = [];
  let result = await compressor.compress(query, options);
  for (let i = 0; i < 5; i++) {
    const started = performance.now();
    result = await compressor.compress(query, options);
    timings.push(performance.now() - started);
  }
  timings.sort((a, b) => a - b);
  const { stats } = result;

  console.log(`── ${name} ${"─".repeat(Math.max(0, 45 - name.length))}`);
  console.log(`before (${stats.modelTokensIn} tokens):\n  ${query}`);
  console.log(`after  (${stats.modelTokensOut} tokens):\n  ${result.text}`);
  // score() applies no policy, so these are the raw model opinions. A
  // high-importance unit marked "no" was lost to the budget, not to judgement.
  const scored = await compressor.score(query);
  const keptStarts = new Set(result.kept.map((u) => u.start));
  console.log(`\n${"unit".padEnd(30)} ${"tokens".padStart(7)} ${"imp".padStart(7)}  kept`);
  for (const u of scored) {
    console.log(
      `${u.text.slice(0, 30).padEnd(30)} ${String(u.tokens).padStart(7)} ` +
        `${u.importance.toFixed(3).padStart(7)}  ${keptStarts.has(u.start) ? "yes" : "no"}`,
    );
  }
  console.log();

  const kept = result.kept.map((u) => toWord(u.text)).filter((w): w is string => w !== null);
  const dropped = result.dropped.map((u) => toWord(u.text)).filter((w): w is string => w !== null);
  const jAll = jaccard(kept, dropped, false);
  const jContent = jaccard(kept, dropped, true);

  console.log(
    `ratio ${(stats.modelTokensIn / Math.max(stats.modelTokensOut, 1)).toFixed(2)}x   ` +
      `jaccard ${jAll.toFixed(2)} (content ${jContent.toFixed(2)})   ` +
      `latency ${timings[2].toFixed(0)} ms ` +
      `(min ${timings[0].toFixed(0)}, max ${timings[timings.length - 1].toFixed(0)})\n`,
  );
  rows.push([name, stats.modelTokensIn, stats.modelTokensOut, timings[2], jAll, jContent]);
}

console.log(
  `${"query".padEnd(28)} ${"before".padStart(7)} ${"after".padStart(7)} ` +
    `${"ratio".padStart(8)} ${"jaccard".padStart(9)} ${"content".padStart(9)} ` +
    `${"median ms".padStart(10)}`,
);
console.log("─".repeat(84));
let totalIn = 0;
let totalOut = 0;
let totalMs = 0;
let sumJ = 0;
let sumJc = 0;
for (const [name, before, after, ms, jAll, jContent] of rows) {
  console.log(
    `${name.padEnd(28)} ${String(before).padStart(7)} ${String(after).padStart(7)} ` +
      `${(before / Math.max(after, 1)).toFixed(2).padStart(7)}x ` +
      `${jAll.toFixed(2).padStart(9)} ${jContent.toFixed(2).padStart(9)} ` +
      `${ms.toFixed(0).padStart(10)}`,
  );
  totalIn += before;
  totalOut += after;
  totalMs += ms;
  sumJ += jAll;
  sumJc += jContent;
}
console.log("─".repeat(84));
console.log(
  `${"TOTAL / mean".padEnd(28)} ${String(totalIn).padStart(7)} ${String(totalOut).padStart(7)} ` +
    `${(totalIn / Math.max(totalOut, 1)).toFixed(2).padStart(7)}x ` +
    `${(sumJ / rows.length).toFixed(2).padStart(9)} ` +
    `${(sumJc / rows.length).toFixed(2).padStart(9)} ` +
    `${totalMs.toFixed(0).padStart(10)}`,
);
