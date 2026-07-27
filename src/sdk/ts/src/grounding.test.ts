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
