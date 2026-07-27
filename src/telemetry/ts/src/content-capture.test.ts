import { afterEach, describe, expect, it } from "vitest";
import {
  ContentCapture,
  clearContentCapture,
  contentCaptureMode,
  setContentCapture,
} from "./index.js";

describe("contentCaptureMode", () => {
  it("defaults to NO_CONTENT when unset or empty", () => {
    expect(contentCaptureMode({})).toBe(ContentCapture.NoContent);
    expect(contentCaptureMode({ OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: "" })).toBe(
      ContentCapture.NoContent,
    );
  });

  it("parses each enum value, case-insensitively", () => {
    const env = (v: string) => ({
      OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: v,
    });
    expect(contentCaptureMode(env("NO_CONTENT"))).toBe(ContentCapture.NoContent);
    expect(contentCaptureMode(env("span_only"))).toBe(ContentCapture.SpanOnly);
    expect(contentCaptureMode(env("Event_Only"))).toBe(ContentCapture.EventOnly);
    expect(contentCaptureMode(env("SPAN_AND_EVENT"))).toBe(ContentCapture.SpanAndEvent);
  });

  it("maps the legacy boolean form (true -> full capture, false -> none)", () => {
    const env = (v: string) => ({
      OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: v,
    });
    expect(contentCaptureMode(env("true"))).toBe(ContentCapture.SpanAndEvent);
    expect(contentCaptureMode(env("false"))).toBe(ContentCapture.NoContent);
  });
});

describe("setContentCapture (programmatic override)", () => {
  const env = (v: string) => ({
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: v,
  });

  afterEach(() => {
    setContentCapture(null); // module-level state; never leak an override across tests
  });

  it("wins over an explicitly set env, in either direction", () => {
    setContentCapture(ContentCapture.NoContent);
    expect(contentCaptureMode(env("SPAN_ONLY"))).toBe(ContentCapture.NoContent);

    setContentCapture(ContentCapture.SpanAndEvent);
    expect(contentCaptureMode(env("NO_CONTENT"))).toBe(ContentCapture.SpanAndEvent);
  });

  it("applies when the env is unset", () => {
    setContentCapture(ContentCapture.EventOnly);
    expect(contentCaptureMode({})).toBe(ContentCapture.EventOnly);
  });

  it("clearing (null or undefined) restores env parsing", () => {
    setContentCapture(ContentCapture.NoContent);
    setContentCapture(null);
    expect(contentCaptureMode(env("SPAN_ONLY"))).toBe(ContentCapture.SpanOnly);

    setContentCapture(ContentCapture.NoContent);
    setContentCapture(undefined);
    expect(contentCaptureMode(env("SPAN_AND_EVENT"))).toBe(ContentCapture.SpanAndEvent);
    expect(contentCaptureMode({})).toBe(ContentCapture.NoContent);
  });

  it("never set: env parsing is untouched (clearing with nothing set is a no-op)", () => {
    setContentCapture(null);
    expect(contentCaptureMode(env("event_only"))).toBe(ContentCapture.EventOnly);
    expect(contentCaptureMode({})).toBe(ContentCapture.NoContent);
  });

  it("normalizes like the env var: case-insensitive, trimmed, legacy boolean forms", () => {
    setContentCapture("span_only" as ContentCapture);
    expect(contentCaptureMode({})).toBe(ContentCapture.SpanOnly);

    setContentCapture(" SPAN_AND_EVENT " as ContentCapture);
    expect(contentCaptureMode({})).toBe(ContentCapture.SpanAndEvent);

    setContentCapture("true" as ContentCapture);
    expect(contentCaptureMode({})).toBe(ContentCapture.SpanAndEvent);

    setContentCapture("0" as ContentCapture);
    expect(contentCaptureMode(env("SPAN_AND_EVENT"))).toBe(ContentCapture.NoContent);
  });

  it("throws a TypeError naming the valid values on an unrecognized mode", () => {
    expect(() => setContentCapture("garbage" as ContentCapture)).toThrow(TypeError);
    expect(() => setContentCapture("garbage" as ContentCapture)).toThrow(
      /NO_CONTENT.*SPAN_ONLY.*EVENT_ONLY.*SPAN_AND_EVENT/,
    );
  });

  it("stores nothing on a failed set: env parsing keeps ruling after the throw", () => {
    expect(() => setContentCapture("SPAN_ONLY_TYPO" as ContentCapture)).toThrow(TypeError);
    expect(contentCaptureMode(env("SPAN_ONLY"))).toBe(ContentCapture.SpanOnly);
    expect(contentCaptureMode({})).toBe(ContentCapture.NoContent);
  });
});

describe("clearContentCapture (generation-scoped clear)", () => {
  const env = (v: string) => ({
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT: v,
  });

  afterEach(() => {
    setContentCapture(null);
  });

  it("only the most recent setter can clear: a stale generation no-ops", () => {
    const g1 = setContentCapture(ContentCapture.NoContent);
    const g2 = setContentCapture(ContentCapture.EventOnly);
    expect(g2).toBeGreaterThan(g1);

    clearContentCapture(g1); // stale — must not clobber g2's override
    expect(contentCaptureMode(env("SPAN_AND_EVENT"))).toBe(ContentCapture.EventOnly);

    clearContentCapture(g2); // current owner — clears, env rules again
    expect(contentCaptureMode(env("SPAN_AND_EVENT"))).toBe(ContentCapture.SpanAndEvent);
    expect(contentCaptureMode({})).toBe(ContentCapture.NoContent);
  });

  it("is idempotent for the current generation", () => {
    const g = setContentCapture(ContentCapture.SpanOnly);
    clearContentCapture(g);
    clearContentCapture(g);
    expect(contentCaptureMode({})).toBe(ContentCapture.NoContent);
  });

  it("an unconditional setContentCapture(null) invalidates outstanding generations", () => {
    const g1 = setContentCapture(ContentCapture.SpanOnly);
    setContentCapture(null); // the direct user's clear is the newest config action
    const g3 = setContentCapture(ContentCapture.EventOnly);

    clearContentCapture(g1); // stale — the slot moved on twice since
    expect(contentCaptureMode({})).toBe(ContentCapture.EventOnly);
    clearContentCapture(g3);
    expect(contentCaptureMode({})).toBe(ContentCapture.NoContent);
  });
});
