#!/usr/bin/env node
// Reference verifier for the Ratel catalog-source contract, v1.
//
// It is the executable oracle for `vectors.json`: it implements the frozen
// ETag / canonicalization algorithm, the scope-overlay resolution, and the
// `If-None-Match` matcher exactly as `protocol/v1/README.md` specifies, then
// checks every vector against its committed expectation.
//
//   node verify.mjs            assert committed expectations (CI mode; exits 1 on mismatch)
//   node verify.mjs --update   recompute and write expectations back into vectors.json
//
// The text spec in ../README.md is normative; this file is a conformant
// reference implementation of it. A third-party source or loader passes the
// contract by reproducing the committed ETags in vectors.json — it does not
// need to run this script.

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const VECTORS_PATH = join(HERE, 'vectors.json');

// The frozen v1 content projection: exactly these fields, in this order.
export const SKILL_FIELDS = ['id', 'name', 'description', 'tags', 'tools', 'metadata', 'body'];

// UTF-8 bytewise comparison (== Unicode code-point order for the ASCII ids/keys
// the contract uses). Used for the set sort-by-id and the metadata key sort.
function byteCompare(a, b) {
  return Buffer.compare(Buffer.from(a, 'utf8'), Buffer.from(b, 'utf8'));
}

// Canonical JSON for one projected skill: fixed field order, metadata keys
// byte-sorted, arrays in authored order, minimal JSON escaping, raw UTF-8,
// no insignificant whitespace. JSON.stringify on a string / string[] already
// emits minimal escapes + raw UTF-8 + compact form, which is what the spec
// requires; we drive field and metadata-key order by hand so object-key
// iteration order can never influence the bytes.
export function canonicalSkill(skill) {
  const s = JSON.stringify;
  const metaKeys = Object.keys(skill.metadata ?? {}).sort(byteCompare);
  const meta = '{' + metaKeys.map((k) => s(k) + ':' + s(skill.metadata[k])).join(',') + '}';
  return (
    '{' +
    '"id":' + s(skill.id) + ',' +
    '"name":' + s(skill.name) + ',' +
    '"description":' + s(skill.description) + ',' +
    '"tags":' + s(skill.tags) + ',' +
    '"tools":' + s(skill.tools) + ',' +
    '"metadata":' + meta + ',' +
    '"body":' + s(skill.body) +
    '}'
  );
}

// Canonical bytes for a resolved set: skills sorted by id, each canonicalized,
// joined as a compact JSON array.
export function canonicalSet(skills) {
  const sorted = [...skills].sort((a, b) => byteCompare(a.id, b.id));
  return '[' + sorted.map(canonicalSkill).join(',') + ']';
}

// ETag over a resolved set: lowercase hex SHA-256 of the canonical bytes,
// wrapped as a strong entity-tag.
export function etagOf(skills) {
  const hex = createHash('sha256').update(canonicalSet(skills), 'utf8').digest('hex');
  return { hex, etag: '"' + hex + '"' };
}

// Resolve the published set for a scope: absent scope => global layer; a named
// subject => its layer overlaid on the global layer, subject winning on `name`
// collision; an unknown subject => the global layer (empty overlay).
export function resolve(catalog, scope) {
  const global = catalog.global ?? [];
  if (scope == null) return sortById(global);
  const layer = (catalog.subjects ?? {})[scope] ?? [];
  const byName = new Map();
  for (const sk of global) byName.set(sk.name, sk);
  for (const sk of layer) byName.set(sk.name, sk); // subject wins on name collision
  return sortById([...byName.values()]);
}

function sortById(skills) {
  return [...skills].sort((a, b) => byteCompare(a.id, b.id));
}

// Strip a `W/` weak prefix and surrounding quotes to the opaque tag value.
function opaque(tag) {
  let t = tag.trim();
  if (t.startsWith('W/')) t = t.slice(2).trim();
  if (t.length >= 2 && t.startsWith('"') && t.endsWith('"')) t = t.slice(1, -1);
  return t;
}

// If-None-Match matcher (weak comparison, per RFC 7232 §3.2). Returns true when
// the request is a cache hit (=> 304). `*` matches any current representation.
// Comma-list tokens are compared by opaque value; the v1 ETags are comma-free
// hex, so a plain split is sufficient.
export function ifNoneMatchMatches(headerValue, currentEtag) {
  if (headerValue == null) return false;
  const v = headerValue.trim();
  if (v === '*') return true;
  const current = opaque(currentEtag);
  return v.split(',').some((tok) => {
    const o = opaque(tok);
    return o.length > 0 && o === current;
  });
}

// ---- intent graph (ADR-0013) ----------------------------------------------

// Structural validation of an intent graph against schema/intent-graph.schema.json.
// Hand-rolled rather than run through a JSON Schema library so this file keeps its
// node-builtins-only property; the schema remains normative and this mirrors it.
// Returns a list of human-readable violations (empty === valid).
export function validateGraph(doc) {
  const errs = [];
  const isObj = (x) => typeof x === 'object' && x !== null && !Array.isArray(x);
  const isInt = (x) => typeof x === 'number' && Number.isInteger(x);

  if (!isObj(doc)) return ['graph must be an object'];
  if (doc.v !== 1) errs.push(`v must be 1, got ${JSON.stringify(doc.v)}`);
  if (!(isInt(doc.built_from_ts) && doc.built_from_ts >= 0)) {
    errs.push(`built_from_ts must be a non-negative integer, got ${JSON.stringify(doc.built_from_ts)}`);
  }
  if (doc.model !== undefined && !(typeof doc.model === 'string' && doc.model.length > 0)) {
    errs.push(`model, when present, must be a non-empty string, got ${JSON.stringify(doc.model)}`);
  }
  if (doc.rev !== undefined && !(isInt(doc.rev) && doc.rev >= 0)) {
    errs.push(`rev, when present, must be a non-negative integer, got ${JSON.stringify(doc.rev)}`);
  }
  if (!Array.isArray(doc.intents)) return errs.concat('intents must be an array');

  const seen = new Set();
  for (const [i, it] of doc.intents.entries()) {
    const at = `intents[${i}]`;
    if (!isObj(it)) { errs.push(`${at} must be an object`); continue; }

    if (!(typeof it.id === 'string' && it.id.length > 0)) errs.push(`${at}.id must be a non-empty string`);
    else if (seen.has(it.id)) errs.push(`${at}.id "${it.id}" is duplicated`);
    else seen.add(it.id);

    if (typeof it.label !== 'string') errs.push(`${at}.label must be a string`);
    if (!Array.isArray(it.terms) || !it.terms.every((t) => typeof t === 'string')) {
      errs.push(`${at}.terms must be an array of strings`);
    }
    // `members` is the match key: a row without it can never be matched.
    if (!Array.isArray(it.members) || it.members.length < 1 || !it.members.every((m) => typeof m === 'string')) {
      errs.push(`${at}.members must be a non-empty array of strings`);
    }
    // Optional — absent when the producer clustered lexically.
    if (it.centroid !== undefined) {
      if (!Array.isArray(it.centroid) || it.centroid.length < 1 || !it.centroid.every((n) => typeof n === 'number')) {
        errs.push(`${at}.centroid, when present, must be a non-empty array of numbers`);
      }
    }
    if (!(isInt(it.support) && it.support >= 1)) {
      errs.push(`${at}.support must be an integer >= 1, got ${JSON.stringify(it.support)}`);
    }
    for (const key of ['tools', 'skills']) {
      const edges = it[key];
      if (!isObj(edges)) { errs.push(`${at}.${key} must be an object`); continue; }
      for (const [id, w] of Object.entries(edges)) {
        if (!(typeof w === 'number' && w > 0)) {
          errs.push(`${at}.${key}["${id}"] must be a number > 0, got ${JSON.stringify(w)}`);
        }
      }
    }
  }
  return errs;
}

// ---- vector runner ---------------------------------------------------------

function buildInm(kind, current, other) {
  switch (kind) {
    case 'self': return current.etag;
    case 'weakSelf': return 'W/' + current.etag;
    case 'star': return '*';
    case 'listWithSelf': return '"deadbeef", ' + current.etag;
    case 'listMiss': return '"deadbeef", "c0ffeec0ffeec0ffee"';
    case 'absent': return null;
    case 'other': return other.etag;
    default: throw new Error(`unknown If-None-Match kind: ${kind}`);
  }
}

function arraysEqual(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => x === b[i]);
}

function main() {
  const update = process.argv.includes('--update');
  const doc = JSON.parse(readFileSync(VECTORS_PATH, 'utf8'));
  const byName = {}; // vector name -> {hex, etag}
  const failures = [];

  // ETag vectors
  for (const v of doc.etag) {
    const catalog = doc.catalogs[v.catalog];
    if (!catalog) { failures.push(`etag/${v.name}: unknown catalog "${v.catalog}"`); continue; }
    const resolved = resolve(catalog, v.scope ?? null);
    const ids = resolved.map((s) => s.id);
    const { hex, etag } = etagOf(resolved);
    byName[v.name] = { hex, etag };
    if (update) {
      v.expect = { resolvedIds: ids, etag };
    } else {
      if (etag !== v.expect?.etag) failures.push(`etag/${v.name}: got ${etag}, expected ${v.expect?.etag}`);
      if (!arraysEqual(ids, v.expect?.resolvedIds)) {
        failures.push(`etag/${v.name}: resolvedIds ${JSON.stringify(ids)} != ${JSON.stringify(v.expect?.resolvedIds)}`);
      }
    }
  }

  // equal / distinct groups
  for (const group of doc.equalEtags ?? []) {
    const tags = group.map((n) => byName[n]?.etag);
    if (new Set(tags).size !== 1) failures.push(`equalEtags: [${group}] are not all equal -> ${JSON.stringify(tags)}`);
  }
  for (const group of doc.distinctEtags ?? []) {
    const tags = group.map((n) => byName[n]?.etag);
    if (new Set(tags).size !== group.length) failures.push(`distinctEtags: [${group}] are not pairwise distinct -> ${JSON.stringify(tags)}`);
  }

  // If-None-Match vectors
  for (const v of doc.inm ?? []) {
    const current = byName[v.current];
    const other = v.of ? byName[v.of] : undefined;
    if (!current) { failures.push(`inm/${v.name}: unknown current vector "${v.current}"`); continue; }
    const header = buildInm(v.ifNoneMatch.kind, current, other);
    const matched = ifNoneMatchMatches(header, current.etag);
    const status = matched ? 304 : 200;
    if (status !== v.expect) failures.push(`inm/${v.name}: header=${JSON.stringify(header)} -> ${status}, expected ${v.expect}`);
  }

  // Intent-graph structural vectors (ADR-0013). No expectations to regenerate:
  // validity is derived from the schema, not from a committed hash.
  const graph = doc.graph ?? { valid: [], invalid: [] };
  for (const v of graph.valid ?? []) {
    const errs = validateGraph(v.doc);
    if (errs.length) failures.push(`graph/valid/${v.name}: expected valid, got [${errs.join('; ')}]`);
  }
  for (const v of graph.invalid ?? []) {
    const errs = validateGraph(v.doc);
    if (!errs.length) failures.push(`graph/invalid/${v.name}: expected rejection (${v.because}), but it validated`);
  }

  if (update) {
    writeFileSync(VECTORS_PATH, JSON.stringify(doc, null, 2) + '\n', 'utf8');
    console.log(`updated ${doc.etag.length} etag expectations in vectors.json`);
    for (const v of doc.etag) console.log(`  ${v.name.padEnd(18)} ${v.expect.etag}`);
    return;
  }

  if (failures.length) {
    console.error(`FAIL — ${failures.length} conformance mismatch(es):`);
    for (const f of failures) console.error('  - ' + f);
    process.exit(1);
  }
  const g = (graph.valid ?? []).length + (graph.invalid ?? []).length;
  const n = (doc.etag?.length ?? 0) + (doc.inm?.length ?? 0) + (doc.equalEtags?.length ?? 0) + (doc.distinctEtags?.length ?? 0) + g;
  console.log(`OK — ${n} conformance checks passed (${doc.etag.length} etag, ${doc.inm.length} if-none-match, ${(doc.equalEtags ?? []).length} equal-groups, ${(doc.distinctEtags ?? []).length} distinct-groups, ${g} intent-graph).`);
}

// Only run the suite when invoked as a script. The helpers above (`etagOf`,
// `resolve`, `ifNoneMatchMatches`, `validateGraph`) are exported for reuse, and
// importing them must not print a report or exit the host process.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main();
}
