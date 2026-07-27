import { describe, expect, it, vi } from "vitest";
import { type Fact, FactCatalog, Pin } from "./experimental.js";
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
});

describe("ratel().recall — facts bucket", () => {
  it("surfaces relevant facts in the recall result with the body inline", async () => {
    const r = ratel();
    await r.facts.register(cancellation);
    const result = await r.recall("how do I cancel and get a refund");
    expect(result).not.toBeNull();
    expect(result?.facts.map((f) => f.factId)).toContain("cancellation");
    const hit = result?.facts.find((f) => f.factId === "cancellation");
    expect(hit?.body).toContain("full refund");
  });

  it("returns an empty facts bucket when no fact catalog content matches", async () => {
    const r = ratel();
    const result = await r.recall("totally unrelated query about rockets");
    expect(result).toBeNull();
  });
});
