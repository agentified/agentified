---
name: ops-publish-to-ametyst
description: "Publish local .claude skills/loops to Ametyst — verbatim, flattened, no brain rewrite; optional retire-local (MOVE originals to deprecated-skills/, never delete) makes Ametyst the source of truth."
---

<!-- ametyst-managed: sync-skills -->

This is a pointer to the Ametyst compound `ops-publish-to-ametyst` — the real compound lives in the Ametyst workspace, not in this file.

To run it, call the `runCompound` MCP tool with compound="ops-publish-to-ametyst" (present the match and get confirmation first, per the getCompound/getLoop stop-gate convention).

Fallback: if run-by-slug returns 404 (slug→id resolution is not landed yet), call `getCompound` with the slug "ops-publish-to-ametyst" to fetch the body, then execute it inline.
