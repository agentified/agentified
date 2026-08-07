# End-to-end product checks (`e2e/`)

These runners exercise the **real, installed distributables** (the wheel, the npm
tarball + native binary) through each SDK's **public API** — not the
dev-mode source build. They are driven by the `ci` workflow's `verify` matrix
(`.github/workflows/ci.yml`) after artifacts are built, and can be run locally.

## What it covers

A single, language-neutral fixture + scenario is exercised identically by every SDK,
so a behavior divergence between SDKs (they all wrap the same Rust BM25 core) makes
exactly one runner fail:

- `fixtures/catalog.json` — the shared tool catalog (the input).
- `fixtures/skills.json` — the shared skill catalog (the on-demand analogue).
- `scenario.json` — the assertions (the source of truth): per-query top-1 ranking for
  tools and skills, direct invoke, the unified `search_capabilities` (tools + skills),
  `invoke_tool`, `get_skill_content`, the deprecated tools-only `search_tools` compatibility
  shim, and skill→tool cross-pollination (a matched skill's declared `tools` ride into the
  tools bucket at score 0).

| Runner | Package under test | Surface exercised |
|--------|--------------------|-------------------|
| `python/run_e2e.py` | `ratel-ai` wheel | `ToolCatalog` + `SkillCatalog` search/invoke; current `search_capabilities_tool` / `invoke_tool_tool` / `get_skill_content_tool`; deprecated `search_tools_tool` compatibility shim |
| `ts/run_e2e.mjs` | `@ratel-ai/sdk` (+ native binary) | `ToolCatalog` + `SkillCatalog` search/invoke; current `searchCapabilitiesTool` / `invokeToolTool` / `getSkillContentTool`; deprecated `searchToolsTool` compatibility shim |

## Run locally

**Python** — build & install the wheel into a throwaway venv, then run:
```bash
cd src/sdk/python
python -m venv /tmp/ratel-e2e-py && /tmp/ratel-e2e-py/bin/pip install -U pip maturin
/tmp/ratel-e2e-py/bin/maturin build --release --out /tmp/ratel-wheels
/tmp/ratel-e2e-py/bin/pip install /tmp/ratel-wheels/*.whl
/tmp/ratel-e2e-py/bin/python ../../../e2e/python/run_e2e.py
```

**TS** — this mirrors the `ci.yml` `verify` job's "Node SDK" step exactly, and you must follow it:
the loader package alone is **not** runnable (its `optionalDependencies` are injected
only at publish time and it ships no `.node`, so installing it alone fails with
`Cannot find native binding`). You build the host's native binary, copy it into the
matching `npm/<triple>/` subpackage, then pack **and install both** the loader and the
subpackage. The runner is copied next to the install (so the bare `@ratel-ai/sdk` import
resolves against the artifact, not the workspace source) and `RATEL_E2E_DIR` points it at
the repo's fixtures:
```bash
repo="$PWD"
pnpm --filter @ratel-ai/sdk build            # builds the host native binary + tsc
node_file="$(ls src/sdk/ts/native/ratel-sdk.*.node | head -1)"
triple="$(basename "$node_file" .node | sed 's/^ratel-sdk\.//')"   # e.g. darwin-arm64
cp "$node_file" "src/sdk/ts/npm/$triple/"
rm -rf /tmp/ratel-tgz && mkdir -p /tmp/ratel-tgz
pnpm --filter @ratel-ai/sdk pack --pack-destination /tmp/ratel-tgz   # loader
npm pack "./src/sdk/ts/npm/$triple" --pack-destination /tmp/ratel-tgz # native subpackage
mkdir -p /tmp/ratel-e2e-ts && cd /tmp/ratel-e2e-ts && npm init -y >/dev/null
npm install /tmp/ratel-tgz/*.tgz             # install loader + subpackage together
cp "$repo/e2e/ts/run_e2e.mjs" ./run_e2e.mjs
RATEL_E2E_DIR="$repo/e2e" node run_e2e.mjs
```

## Extending

When you add product surface (new tools, new skills, a new SDK method):

1. Add the tool(s) to `fixtures/catalog.json`, or the skill(s) to `fixtures/skills.json`
   (give each a distinctive description; most skills set only `id`/`name`/`description`/`body`,
   while `deploy-web-service` also declares a `tools` dependency for the cross-pollination case.
   The TS and Python `Skill` types are in parity, so `tags`/`tools`/`metadata` work identically
   on both).
2. Add a query with an unambiguous expected top-1 to `scenario.json` — `searches` for tools,
   `skillSearches` for skills (write the description so the query's terms clearly match it;
   assertions check top-1 / membership, not full ordering, to stay robust to score ties).
   For a `search_capabilities` case, pick a query whose terms match both the expected tool
   and the expected skill (the two corpora are ranked independently).
3. If you exercise a new method/command, extend the relevant runner(s) — and keep the
   Python and TS runners in lockstep so parity stays enforced.
