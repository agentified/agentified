import { describe, expect, it, vi } from "vitest";
import { type Fact, FactCatalog, FactRegistry, Pin } from "./experimental.js";
import { resetExperimentalWarningForTest } from "./experimental-warning.js";
import { ratel } from "./index.js";

const address: Fact = {
  id: "shop-address",
  name: "shop-address",
  description: "Where the barbershop is located and its opening hours.",
  tags: ["location"],
  body: "12 Baker Street, London. Open Mon–Sat, 9am–7pm.",
  pin: "always",
};

const cancellation: Fact = {
  id: "cancellation",
  name: "cancellation-policy",
  description: "How to cancel or reschedule a booking and get a refund.",
  tags: ["booking"],
  body: "Cancel at least 24h ahead for a full refund.",
  pin: "retrieved",
};

describe("experimental warning", () => {
  it("warns once that the facts API is experimental", () => {
    resetExperimentalWarningForTest();
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    new FactCatalog();
    new FactCatalog();
    expect(spy).toHaveBeenCalledTimes(1);
    expect(String(spy.mock.calls[0]?.[0])).toMatch(/experimental/i);
    spy.mockRestore();
  });
});

describe("FactCatalog", () => {
  it("returns no hits from an empty catalog", () => {
    expect(new FactCatalog().search("anything", 5)).toEqual([]);
  });

  it("registers facts and ranks the relevant one first", async () => {
    const catalog = new FactCatalog();
    await catalog.register([address, cancellation]);
    const hits = catalog.search("how do I cancel and get a refund", 5);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].factId).toBe("cancellation");
  });

  it("ranks a fact by searchableDescription instead of its description", async () => {
    const catalog = new FactCatalog();
    await catalog.register({
      id: "fact",
      name: "fact",
      description: "composedonlyterm",
      searchableDescription: "overrideonlyterm",
    });

    expect(catalog.search("overrideonlyterm", 5).map((hit) => hit.factId)).toEqual(["fact"]);
    expect(catalog.search("composedonlyterm", 5)).toEqual([]);
  });

  it("pinned() returns only always facts in registration order", async () => {
    const catalog = new FactCatalog();
    await catalog.register([
      { id: "a", name: "a", description: "always one", pin: "always" },
      { id: "r", name: "r", description: "retrieved one", pin: "retrieved" },
      { id: "b", name: "b", description: "always two", pin: "always" },
    ]);
    expect(catalog.pinned().map((f) => f.id)).toEqual(["a", "b"]);
  });

  it("treats a fact with no pin as retrieved", async () => {
    const catalog = new FactCatalog();
    await catalog.register({ id: "x", name: "x", description: "no pin given" });
    expect(catalog.pinned()).toEqual([]);
    expect(catalog.size()).toBe(1);
  });

  it("re-registers an id in place", async () => {
    const catalog = new FactCatalog();
    await catalog.register(address);
    await catalog.register({ ...address, pin: "retrieved" });
    expect(catalog.size()).toBe(1);
    expect(catalog.pinned()).toEqual([]); // adopted the new pin
  });

  it("rejects an id with characters outside the allowed set", async () => {
    const catalog = new FactCatalog();
    await expect(catalog.register({ id: "bad id", name: "n", description: "d" })).rejects.toThrow(
      /must match/,
    );
  });

  it("rejects an unknown pin value", async () => {
    const catalog = new FactCatalog();
    await expect(
      // biome-ignore lint/suspicious/noExplicitAny: exercising a bad runtime value
      catalog.register({ id: "x", name: "n", description: "d", pin: "pinned" as any }),
    ).rejects.toThrow(/invalid pin/);
  });

  it("accepts the Pin enum in place of the raw string", async () => {
    expect(Pin.Always).toBe("always");
    expect(Pin.Retrieved).toBe("retrieved");
    const catalog = new FactCatalog();
    await catalog.register({ id: "a", name: "a", description: "d", body: "x", pin: Pin.Always });
    expect(catalog.pinned().map((f) => f.id)).toEqual(["a"]);
  });
});

describe("FactRegistry (the lower-level public surface)", () => {
  // The registry is exported from `experimental` too, so it can't be a hole in
  // the id contract the catalog enforces — the ids ride the same trace events
  // and structured injection payloads either way.
  it.each([
    ["has space"],
    ["id\nwith\nnewlines"],
    [""],
    ["emoji🙂"],
  ])("rejects the invalid id %j", async (id) => {
    await expect(
      new FactRegistry().register([{ id, name: "n", description: "d", body: "b" }]),
    ).rejects.toThrow(/fact id/);
  });

  it("rejects an unknown pin value", async () => {
    await expect(
      new FactRegistry().register([
        { id: "ok", name: "n", description: "d", body: "b", pin: "sometimes" as never },
      ]),
    ).rejects.toThrow(/pin/);
  });

  it("leaves the corpus untouched when any fact in the batch is invalid", async () => {
    const registry = new FactRegistry();
    await expect(
      registry.register([
        { id: "good", name: "good", description: "a valid fact", body: "b" },
        { id: "bad id", name: "bad", description: "an invalid fact", body: "b" },
      ]),
    ).rejects.toThrow(/fact id/);
    expect(registry.search("valid fact", 5)).toEqual([]);
  });

  it("accepts a well-formed id", async () => {
    const registry = new FactRegistry();
    await registry.register([
      { id: "shop.address:v1-2_3", name: "n", description: "where the shop is", body: "b" },
    ]);
    expect(registry.search("where the shop is", 5)[0]?.factId).toBe("shop.address:v1-2_3");
  });
});

describe("FactCatalog.ground (the freshness gate, on the catalog)", () => {
  const setup = async () => {
    const catalog = new FactCatalog({ trace: { kind: "memory", sessionId: "t" } });
    await catalog.register([address, cancellation]);
    return catalog;
  };

  it("injects a pinned fact on an empty transcript, then skips it once fresh", async () => {
    const catalog = await setup();
    const first = await catalog.ground("hi", []);
    expect(first.inject.map((i) => i.id)).toContain("shop-address");
    expect(first.inject.find((i) => i.id === "shop-address")?.pin).toBe(Pin.Always);

    const transcript = first.inject.map((i) => i.body);
    const second = await catalog.ground("hi again", transcript);
    expect(second.inject).toEqual([]);
    expect(second.skipped).toContain("shop-address");
  });

  it("tracks evicted vs never via its own session state", async () => {
    const catalog = await setup();
    await catalog.ground("hi", []);
    const again = await catalog.ground("hi", ["a summary that dropped the fact"]);
    expect(again.inject.find((i) => i.id === "shop-address")?.reason).toBe("evicted");
  });

  it("rendering the body verbatim is the whole dedupe contract", async () => {
    const catalog = await setup();
    const { inject } = await catalog.ground("hi", []);
    const shop = inject.find((i) => i.id === "shop-address");
    // A host may decorate around the body — presence still detects it.
    const second = await catalog.ground("hi", [`Note for the agent: ${shop?.body}`]);
    expect(second.skipped).toContain("shop-address");
  });

  it("r.ground delegates to the same catalog state as r.facts.ground", async () => {
    const r = ratel();
    await r.facts.register(address);
    const viaCore = await r.ground("hi", []);
    expect(viaCore.inject.map((i) => i.id)).toContain("shop-address");
    // Same session state: grounding again (body present) skips it.
    const transcript = viaCore.inject.map((i) => i.body);
    const viaCatalog = await r.facts.ground("hi", transcript);
    expect(viaCatalog.skipped).toContain("shop-address");
  });
});

describe("FactCatalog.groundSnapshot (the stateless per-call mode)", () => {
  const setup = async () => {
    const catalog = new FactCatalog({ trace: { kind: "memory", sessionId: "t" } });
    await catalog.register([address, cancellation]);
    return catalog;
  };

  it("returns pinned plus query-matched facts", async () => {
    const catalog = await setup();
    const items = await catalog.groundSnapshot("I need to cancel my appointment");
    const ids = items.map((i) => i.id);
    expect(ids).toContain("shop-address"); // pinned rides along regardless of match
    expect(ids).toContain("cancellation"); // matched by the query
    expect(ids.indexOf("shop-address")).toBeLessThan(ids.indexOf("cancellation")); // pinned first
    for (const item of items) {
      expect(item.body.length).toBeGreaterThan(0);
    }
  });

  it("is stateless: identical calls return the full set every time (no dedup, no memory)", async () => {
    const catalog = await setup();
    const first = await catalog.groundSnapshot("hi");
    const second = await catalog.groundSnapshot("hi");
    expect(second).toEqual(first);
    expect(second.map((i) => i.id)).toContain("shop-address");
  });

  it("does not disturb ground()'s freshness state", async () => {
    const catalog = await setup();
    await catalog.groundSnapshot("hi"); // snapshots never mark anything as injected
    const { inject } = await catalog.ground("hi", []);
    expect(inject.find((i) => i.id === "shop-address")?.reason).toBe("never");
  });

  it("emits a fact_snapshot trace event per fact", async () => {
    const catalog = await setup();
    await catalog.groundSnapshot("cancel my booking");
    const events = catalog.drainTraceEvents() as Array<{ type: string; fact_id: string }>;
    const snaps = events.filter((e) => e.type === "fact_snapshot").map((e) => e.fact_id);
    expect(snaps).toContain("shop-address");
    expect(snaps).toContain("cancellation");
  });

  it("r.groundSnapshot delegates to the catalog", async () => {
    const r = ratel();
    await r.facts.register(address);
    const items = await r.groundSnapshot("anything");
    expect(items.map((i) => i.id)).toContain("shop-address");
  });
});

describe("topK budgets the retrieved tier only", () => {
  // Three always-on facts whose text also matches the query, so they compete
  // for ranked slots with the retrieved tier — the ADR's own motivating shape
  // (address, hours, brand voice), which starved the retrieved tier to zero
  // when the search ran with a plain `k`.
  // Every fact shares the term "shop", so both tiers are ranked by the query
  // below and genuinely compete for slots; the pinned three match it more
  // strongly and take the top three.
  const pinnedTrio: Fact[] = [
    {
      id: "p1",
      name: "shop address",
      description: "the shop address",
      body: "ADDR",
      pin: "always",
    },
    { id: "p2", name: "shop hours", description: "the shop hours", body: "HOURS", pin: "always" },
    { id: "p3", name: "shop voice", description: "the shop voice", body: "VOICE", pin: "always" },
  ];
  const retrievedTrio: Fact[] = [
    {
      id: "r1",
      name: "shop cancellation",
      description: "shop booking cancellation",
      body: "CANCEL",
    },
    { id: "r2", name: "shop pricing", description: "shop prices", body: "PRICE" },
    { id: "r3", name: "shop barbers", description: "shop barbers", body: "BARBER" },
  ];

  const stocked = async (): Promise<FactCatalog> => {
    const catalog = new FactCatalog();
    await catalog.register([...pinnedTrio, ...retrievedTrio]);
    return catalog;
  };

  const QUERY = "shop address shop hours shop voice";

  it("pinned facts do not consume the retrieved budget", async () => {
    const catalog = await stocked();
    // The query favours the PINNED tier, so with a plain-`k` search all three
    // top slots go to pinned facts that are then filtered back out, leaving the
    // retrieved tier permanently empty. Pin the top-3 assumption first, so this
    // test fails loudly rather than silently passing if the ranking shifts.
    const ranked = (await catalog.searchAsync(QUERY, 3)).map((h) => h.factId);
    expect(ranked, "fixture assumption: pinned facts take the top 3 slots").toEqual([
      "p1",
      "p2",
      "p3",
    ]);

    const items = await catalog.groundSnapshot(QUERY, { topK: 3 });
    const retrieved = items.filter((i) => i.pin === Pin.Retrieved);
    expect(retrieved.length, "the retrieved tier must not be starved by pinned facts").toBe(3);
    expect(items.filter((i) => i.pin === Pin.Always)).toHaveLength(3);
  });

  it("returns at most topK retrieved facts", async () => {
    const catalog = await stocked();
    const items = await catalog.groundSnapshot(QUERY, { topK: 1 });
    expect(items.filter((i) => i.pin === Pin.Retrieved)).toHaveLength(1);
    expect(items.filter((i) => i.pin === Pin.Always)).toHaveLength(3); // pinned are never capped
  });

  it("honours the catalog-level factsTopK default", async () => {
    const catalog = new FactCatalog({ factsTopK: 2 });
    await catalog.register(retrievedTrio);
    const items = await catalog.groundSnapshot("shop cancellation pricing barbers");
    expect(items).toHaveLength(2);
  });

  it("honours RatelConfig.factsTopK through r.groundSnapshot", async () => {
    const r = ratel({ factsTopK: 1 });
    await r.facts.register(retrievedTrio);
    const items = await r.groundSnapshot("shop cancellation pricing barbers");
    expect(items).toHaveLength(1);
  });

  it("a pinned fact that also ranks appears once, as pinned", async () => {
    const catalog = await stocked();
    const items = await catalog.groundSnapshot("shop address", { topK: 5 });
    const addressHits = items.filter((i) => i.id === "p1");
    expect(addressHits).toHaveLength(1);
    expect(addressHits[0]?.pin).toBe(Pin.Always);
  });
});

describe("ratel().ground — the freshness gate", () => {
  const setup = async () => {
    const r = ratel({ trace: { kind: "memory", sessionId: "t" } });
    await r.facts.register([address, cancellation]);
    return r;
  };

  it("injects a pinned fact on an empty transcript as `never`", async () => {
    const r = await setup();
    const { inject, skipped } = await r.ground("hi", []);
    expect(inject.map((i) => i.id)).toContain("shop-address");
    const shop = inject.find((i) => i.id === "shop-address");
    expect(shop?.reason).toBe("never");
    expect(shop?.pin).toBe("always");
    expect(shop?.body).toContain("Baker Street");
    expect(skipped).toEqual([]);
  });

  it("skips a fact already fresh in the transcript", async () => {
    const r = await setup();
    const first = await r.ground("hi", []);
    // Simulate the adapter having rendered the injected fact into the history.
    const transcript = ["earlier user turn", ...first.inject.map((i) => i.body)];
    const second = await r.ground("hi again", transcript);
    expect(second.inject).toEqual([]);
    expect(second.skipped).toContain("shop-address");
  });

  it("re-injects as `mutated` after the fact body is edited", async () => {
    const r = await setup();
    const first = await r.ground("hi", []);
    const transcript = first.inject.map((i) => i.body);
    await r.facts.register({ ...address, body: "New location: 40 Oxford Street." });
    const second = await r.ground("hi", transcript);
    const shop = second.inject.find((i) => i.id === "shop-address");
    expect(shop?.reason).toBe("mutated");
    expect(shop?.body).toContain("Oxford Street");
  });

  it("re-injects as `evicted` when compaction drops the fact", async () => {
    const r = await setup();
    await r.ground("hi", []);
    const second = await r.ground("hi", ["a summary that dropped the fact"]);
    const shop = second.inject.find((i) => i.id === "shop-address");
    expect(shop?.reason).toBe("evicted");
  });

  it("injects a retrieval-gated fact only when the query ranks it in", async () => {
    const r = await setup();
    const { inject } = await r.ground("I need to cancel my appointment", []);
    expect(inject.map((i) => i.id)).toContain("cancellation");
  });

  it("emits fact_inject then fact_inject_skip trace events", async () => {
    const r = await setup();
    const first = await r.ground("hi", []);
    const transcript = first.inject.map((i) => i.body);
    await r.ground("hi", transcript);
    const events = r.facts.drainTraceEvents() as Array<{ type: string; fact_id: string }>;
    const types = events.map((e) => e.type);
    expect(types).toContain("fact_inject");
    expect(types).toContain("fact_inject_skip");
  });

  it("carries the decision's reason on the fact_inject event, not a constant", async () => {
    // Hardcoding `reason: "never"` in the catalog used to pass every test here;
    // the whole point of the gate is that the reason distinguishes the cases.
    const r = await setup();
    await r.ground("hi", []); // never
    const firstEvents = r.facts.drainTraceEvents() as Array<{ type: string; reason?: string }>;
    expect(firstEvents.find((e) => e.type === "fact_inject")?.reason).toBe("never");

    await r.ground("hi", ["a summary that dropped everything"]); // evicted
    const secondEvents = r.facts.drainTraceEvents() as Array<{ type: string; reason?: string }>;
    expect(secondEvents.find((e) => e.type === "fact_inject")?.reason).toBe("evicted");
  });

  it("refreshes its injected-body memory across three turns", async () => {
    // Two-turn tests can't see this: after a `mutated` re-inject the session
    // state must hold the NEW body, or turn 3 re-reports `mutated` forever.
    const r = await setup();
    const turn1 = await r.ground("hi", []);
    expect(turn1.inject.find((i) => i.id === "shop-address")?.reason).toBe("never");

    await r.facts.register({ ...address, body: "New location: 40 Oxford Street." });
    const turn2 = await r.ground(
      "hi",
      turn1.inject.map((i) => i.body),
    );
    expect(turn2.inject.find((i) => i.id === "shop-address")?.reason).toBe("mutated");

    // Turn 3: the edited body is now rendered in the history and nothing changed.
    const turn3 = await r.ground("hi", [
      ...turn1.inject.map((i) => i.body),
      ...turn2.inject.map((i) => i.body),
    ]);
    expect(turn3.inject.map((i) => i.id)).not.toContain("shop-address");
    expect(turn3.skipped).toContain("shop-address");
  });
});

describe("experimental quarantine on the stable path", () => {
  it("does not warn when a host only uses tools/skills (facts are lazy)", async () => {
    resetExperimentalWarningForTest();
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const r = ratel();
    await r.tools.register({
      id: "t1",
      name: "t1",
      description: "read a file from disk",
      inputSchema: { type: "object" },
      outputSchema: { type: "object" },
      execute: () => ({}),
    });
    await r.skills.register({ id: "s1", name: "s1", description: "a playbook" });
    await r.recall("read a file");
    r.modelTools();
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });

  it("constructs the fact catalog (and warns) only on first facts use", async () => {
    resetExperimentalWarningForTest();
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const r = ratel();
    expect(spy).not.toHaveBeenCalled();
    void r.facts; // first touch builds it
    expect(spy).toHaveBeenCalledTimes(1);
    spy.mockRestore();
  });

  it("returns the same lazily-built catalog on every access", () => {
    resetExperimentalWarningForTest();
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const r = ratel();
    expect(r.facts).toBe(r.facts);
    spy.mockRestore();
  });

  it("adaptTo does not construct the catalog — the spread must not read `facts`", async () => {
    // Regression: `{ ...base, ...ext }` performs [[Get]] on every own ENUMERABLE
    // property, so an enumerable `facts` getter made every adapted host build
    // the experimental catalog (and see its warning) without touching facts.
    // The accessor is non-enumerable for exactly this reason.
    resetExperimentalWarningForTest();
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const view = ratel().adaptTo({
      name: "test-adapter",
      ingest: () => "passthrough" as const,
      expose: (tool: unknown) => tool,
      recallMessages: () => [],
    });
    await view.recall("anything");
    view.modelTools();
    expect(spy, "adapting must not build the experimental fact catalog").not.toHaveBeenCalled();
    // Still reachable and still lazy — the quarantine, not a removal.
    expect(view.facts).toBeInstanceOf(FactCatalog);
    expect(spy).toHaveBeenCalledTimes(1);
    spy.mockRestore();
  });

  it("spreading or serializing a handle does not construct the catalog", async () => {
    // Same [[Get]] hazard from the host's side: `{ ...r }`, `JSON.stringify(r)`,
    // and any logger walking own enumerable keys.
    resetExperimentalWarningForTest();
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const r = ratel();
    void { ...r };
    void JSON.stringify(r);
    void Object.keys(r);
    expect(spy).not.toHaveBeenCalled();
    spy.mockRestore();
  });

  it("RATEL_EXPERIMENTAL_SILENCE suppresses the warning", () => {
    resetExperimentalWarningForTest();
    const previous = process.env.RATEL_EXPERIMENTAL_SILENCE;
    process.env.RATEL_EXPERIMENTAL_SILENCE = "1";
    const spy = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      new FactCatalog();
      expect(spy).not.toHaveBeenCalled();
    } finally {
      spy.mockRestore();
      if (previous === undefined) delete process.env.RATEL_EXPERIMENTAL_SILENCE;
      else process.env.RATEL_EXPERIMENTAL_SILENCE = previous;
      resetExperimentalWarningForTest();
    }
  });

  it("model-facing search_capabilities returns no facts key (schema parity)", async () => {
    const r = ratel();
    await r.tools.register({
      id: "t1",
      name: "t1",
      description: "read a file from disk",
      inputSchema: { type: "object" },
      outputSchema: { type: "object" },
      execute: () => ({}),
    });
    const search = r.modelTools().search_capabilities;
    const result = (await search.execute({ query: "read a file" })) as Record<string, unknown>;
    expect(Object.keys(result).sort()).toEqual(["skills", "tools"]);
  });
});

describe("recall never touches facts (facts are a separate, host-driven path)", () => {
  it("returns only tools and skills — no facts key — even with facts registered", async () => {
    const r = ratel();
    await r.facts.register([address, cancellation]);
    await r.tools.register({
      id: "t1",
      name: "t1",
      description: "cancel a subscription",
      inputSchema: { type: "object" },
      outputSchema: { type: "object" },
      execute: () => ({}),
    });
    const result = await r.recall("how do I cancel and get a refund");
    expect(result).not.toBeNull();
    expect(Object.keys(result as object).sort()).toEqual(["skills", "tools"]);
  });

  it("never searches the fact catalog (no fact_search event)", async () => {
    const r = ratel({ trace: { kind: "memory", sessionId: "t" } });
    await r.facts.register(cancellation);
    r.facts.drainTraceEvents(); // clear registration churn
    await r.recall("how do I cancel and get a refund");
    const events = r.facts.drainTraceEvents() as Array<{ type: string }>;
    expect(events.filter((e) => e.type === "fact_search")).toEqual([]);
  });

  it("adapted views share the core's fact catalog and grounding state", async () => {
    const r = ratel();
    const view = r.adaptTo({
      name: "test-adapter",
      ingest: () => "passthrough" as const,
      expose: (tool: unknown) => tool,
      recallMessages: () => [],
    });
    expect(view.facts).toBe(r.facts);
    await r.facts.register(address);
    const first = await view.ground("hi", []);
    expect(first.inject.map((i) => i.id)).toContain("shop-address");
    // Grounding again through the CORE with the rendered transcript skips —
    // one shared injected-body state across both views.
    const second = await r.ground(
      "hi",
      first.inject.map((i) => i.body),
    );
    expect(second.skipped).toContain("shop-address");
  });
});
