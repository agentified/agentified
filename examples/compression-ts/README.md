# Prompt compression (TypeScript)

Compresses meeting notes to ~40% of their tokens while keeping the facts an
answer depends on, then assembles a prompt where **only the context was
rewritten**.

```bash
pnpm --filter @ratel-ai/example-compression start
```

First run downloads the ~700 MB LLMLingua-2 checkpoint into the shared
HuggingFace cache.

What it demonstrates:

- `preload()` — pay the cold load at startup, not inside a request.
- `compress()` — the token budget, and `stats` reporting exactly what it cost.
- `dropped` — the closest calls, so you can see *why* something went.
- The gate — a short prompt comes back untouched, with `stats.gate` saying so.
- `protect` — naming the figures that must survive an aggressive rate.

Compression is experimental and lossy. Compress the context; leave your
instructions and the user's question verbatim.
