/**
 * Experimental prompt compression: compress the context, keep the scaffolding.
 *
 *   pnpm --filter @ratel-ai/example-compression start
 *
 * Downloads the ~700 MB compression model on first run.
 */
import { ExperimentalCompression } from "@ratel-ai/sdk";

/** Notes an agent is carrying — the part worth compressing. */
const NOTES = `
Speaker 1: Let's get the Q3 infrastructure review started. The first item is the
database migration we've been putting off since May. We're running Postgres 12 in
production, which reached end of life in November 2024, so we are unsupported and
security patches are no longer issued.

Speaker 2: We tried this migration once already, in April, and it failed. The
pg_upgrade step took about four hours on the primary and our maintenance window
was only two hours, so we rolled back halfway through. That rollback took another
hour and we ended up with roughly three hours of degraded read performance.

Speaker 1: What's different this time?

Speaker 2: We're using logical replication instead of pg_upgrade, so we can build
the Postgres 16 replica alongside the existing primary and let it catch up over
several days with no downtime. The cutover is a DNS switch plus a connection pool
drain, somewhere between 90 seconds and 4 minutes. We've tested this exact
procedure twice in staging with a full production data snapshot.

Speaker 3: What does running both clusters cost?

Speaker 2: About 8,400 dollars for the two week overlap. TimescaleDB is the risky
one: the version we're on, 2.8, doesn't support Postgres 16 at all, so we need to
bump Timescale to 2.14 first in a separate window.
`.trim();

// The instruction and the question are never compressed — losing a word in
// either changes the task, not the background.
const INSTRUCTION =
  "You are given notes from an engineering meeting. Answer using only these notes.";
const QUESTION = "What is the plan for the Postgres migration, and what is the main risk?";

const compressor = new ExperimentalCompression();

// Pay the ~700 MB load deliberately rather than inside the first request.
console.log("loading model…");
const loadStart = Date.now();
await compressor.preload();
console.log(`ready in ${Date.now() - loadStart} ms\n`);

const result = await compressor.compress(NOTES, { rate: 0.4 });

console.log("=== compressed notes ===");
console.log(result.text);
console.log("\n=== what it cost ===");
console.log(
  `${result.stats.modelTokensIn} -> ${result.stats.modelTokensOut} model tokens ` +
    `(${(result.stats.modelTokensIn / result.stats.modelTokensOut).toFixed(2)}x), ` +
    `${result.stats.wordsIn} -> ${result.stats.wordsOut} units, ` +
    `${result.stats.chunks} encoder pass(es), ${result.stats.tookMs} ms`,
);

// The explainability channel: why each unit went, not just that it did.
const worst = [...result.dropped].sort((a, b) => b.importance - a.importance).slice(0, 6);
console.log(
  "closest calls (highest-scoring dropped):",
  worst.map((w) => `${w.text}(${w.importance.toFixed(2)})`).join(", "),
);

// Assemble the real prompt: only the middle was rewritten.
const prompt = `${INSTRUCTION}\n\n<notes>\n${result.text}\n</notes>\n\nQuestion: ${QUESTION}`;
console.log(`\nprompt assembled: ${prompt.length} chars (notes were ${NOTES.length})`);

// Short prompts are out of domain — they come back untouched, and say so.
const short = await compressor.compress("translate this to french");
console.log(`\nshort input -> gate=${short.stats.gate}, text unchanged: ${short.text === "translate this to french"}`);

// Protect what must survive at any rate. Naming the figures is cheaper and more
// precise than protecting every digit.
const aggressive = await compressor.compress(NOTES, {
  rate: 0.15,
  protect: [/8,400/, /2\.14/],
});
console.log(
  `\nat rate 0.15 with protection: ${aggressive.stats.modelTokensOut} tokens, ` +
    `8,400 kept=${aggressive.text.includes("8,400")}, 2.14 kept=${aggressive.text.includes("2.14")}, ` +
    `budgetExceeded=${aggressive.stats.budgetExceeded}`,
);
