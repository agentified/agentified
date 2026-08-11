import { describe, expect, it } from "vitest";
import { type FactCandidate, planInjection } from "./experimental.js";

const cand = (id: string, body: string): FactCandidate => ({ id, body });

describe("planInjection (content-presence gate)", () => {
  const address = "12 Baker Street, London. Open Mon–Sat 9–7.";

  it("injects an absent fact as `never` with no session history", () => {
    const [d] = planInjection({ candidates: [cand("a", address)], transcript: [] });
    expect(d).toEqual({ id: "a", inject: true, reason: "never" });
  });

  it("skips a fact whose body is already in the transcript (the token-saving case)", () => {
    const [d] = planInjection({
      candidates: [cand("a", address)],
      transcript: ["user asked something", `Here you go: ${address}`],
    });
    expect(d).toEqual({ id: "a", inject: false, reason: "fresh" });
  });

  it("presence is who-put-it-there agnostic: a verbatim echo by anyone counts", () => {
    // The assistant (or even the user) said the fact verbatim — the info is in
    // the window, so injecting again would duplicate it.
    const [d] = planInjection({
      candidates: [cand("a", address)],
      transcript: [`assistant: We're at ${address} — see you soon!`],
    });
    expect(d.inject).toBe(false);
  });

  it("classifies absent + previously-injected-same-body as `evicted`", () => {
    const [d] = planInjection({
      candidates: [cand("a", address)],
      transcript: ["a summary that dropped the fact"],
      previouslyInjected: new Map([["a", address]]),
    });
    expect(d).toEqual({ id: "a", inject: true, reason: "evicted" });
  });

  it("classifies absent + previously-injected-different-body as `mutated`", () => {
    const [d] = planInjection({
      candidates: [cand("a", "New location: 40 Oxford Street.")],
      transcript: [`old turn still contains: ${address}`],
      previouslyInjected: new Map([["a", address]]),
    });
    expect(d).toEqual({ id: "a", inject: true, reason: "mutated" });
  });

  it("re-injects a RETRACTED body as `mutated` even though it reads as present", () => {
    // The retraction direction: shrinking a body leaves the new text a
    // substring of the stale text still in the window, so a plain presence
    // check would call it `fresh` and the withdrawn clause would stay asserted
    // to the model forever. The stale body's presence is what arms this.
    const oldBody = "Cancel 24h ahead for a full refund; same-day is a 50% fee.";
    const newBody = "Cancel 24h ahead for a full refund;";
    const [d] = planInjection({
      candidates: [cand("cx", newBody)],
      transcript: [oldBody],
      previouslyInjected: new Map([["cx", oldBody]]),
    });
    expect(d).toEqual({ id: "cx", inject: true, reason: "mutated" });
    expect(oldBody.includes("50% fee"), "the retracted clause is what's at stake").toBe(true);
  });

  it("converges: once the new body is injected, the next turn is fresh again", () => {
    const oldBody = "Cancel 24h ahead for a full refund; same-day is a 50% fee.";
    const newBody = "Cancel 24h ahead for a full refund;";
    // Turn N+1: the stale body is still in history, but so is the correction,
    // and the session state now records the new body — so no re-injection loop.
    const [d] = planInjection({
      candidates: [cand("cx", newBody)],
      transcript: [oldBody, newBody],
      previouslyInjected: new Map([["cx", newBody]]),
    });
    expect(d).toEqual({ id: "cx", inject: false, reason: "fresh" });
  });

  it("does not re-inject when the stale body is gone and the new one is present", () => {
    // Compaction dropped the old text and the new body is already rendered:
    // nothing is misstated, so the retraction rule must stay quiet.
    const [d] = planInjection({
      candidates: [cand("a", "New location: 40 Oxford Street.")],
      transcript: ["New location: 40 Oxford Street."],
      previouslyInjected: new Map([["a", address]]),
    });
    expect(d).toEqual({ id: "a", inject: false, reason: "fresh" });
  });

  it("never injects an empty body, even when the stale one is still present", () => {
    // A fact edited down to "" has nothing to render; the retraction rule must
    // not resurrect it as an empty injection.
    const [d] = planInjection({
      candidates: [cand("a", "")],
      transcript: [address],
      previouslyInjected: new Map([["a", address]]),
    });
    expect(d).toEqual({ id: "a", inject: false, reason: "fresh" });
  });

  it("an edited body that is somehow already present is simply fresh", () => {
    const newBody = "New location: 40 Oxford Street.";
    const [d] = planInjection({
      candidates: [cand("a", newBody)],
      transcript: [`someone already mentioned: ${newBody}`],
      previouslyInjected: new Map([["a", address]]),
    });
    expect(d.inject).toBe(false);
  });

  it("treats an empty body as trivially present (nothing to inject)", () => {
    const [d] = planInjection({ candidates: [cand("a", "")], transcript: [] });
    expect(d).toEqual({ id: "a", inject: false, reason: "fresh" });
  });

  it("does not falsely match a body split across two messages", () => {
    // Two unrelated messages ending/starting with the halves of a multi-line
    // body must NOT reconstruct it across the message boundary.
    const body = "12 Baker Street\nLondon NW1, open daily.";
    const [d] = planInjection({
      candidates: [cand("shop-address", body)],
      transcript: ["it's on 12 Baker Street", "London NW1, open daily. Come say hi!"],
    });
    expect(d).toEqual({ id: "shop-address", inject: true, reason: "never" });
  });

  it("matches bodies that span lines within one message", () => {
    const multiline = "Line one of the policy.\nLine two of the policy.";
    const [d] = planInjection({
      candidates: [cand("a", multiline)],
      transcript: [`intro\n${multiline}\noutro`],
    });
    expect(d.inject).toBe(false);
  });

  it("is order-preserving and deterministic across repeated calls", () => {
    const input = {
      candidates: [cand("a", "alpha body"), cand("b", "beta body"), cand("c", "gamma body")],
      transcript: ["contains beta body here"],
    };
    const first = planInjection(input);
    const second = planInjection(input);
    expect(first.map((d) => d.id)).toEqual(["a", "b", "c"]);
    expect(first.map((d) => d.inject)).toEqual([true, false, true]);
    expect(first).toEqual(second);
  });

  it("returns an empty plan for no candidates", () => {
    expect(planInjection({ candidates: [], transcript: ["anything"] })).toEqual([]);
  });
});
