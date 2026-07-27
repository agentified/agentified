/**
 * The re-injection freshness gate for facts — the pure decision layer behind
 * "inject a fact only if it isn't already in the context."
 *
 * Once a fact's body is injected as a transcript message it *stays* in the
 * history on every later turn, so re-appending it each turn would duplicate
 * tokens and confuse the model. This module decides, per fact, whether to
 * (re-)inject — and the signal is **the fact's own body text**: a fact is
 * "present" when its body appears verbatim anywhere in the transcript. No
 * markers, no tags, no extra tokens — the injected content is its own record.
 * Compaction dropping the text naturally re-arms injection, and an edited body
 * (no longer found verbatim) naturally re-injects the new version.
 *
 * The one contract this puts on hosts: render `body` **verbatim** in the
 * message you append (decorate around it, don't rewrite it) — otherwise the
 * gate can't see it next turn and will re-inject.
 *
 * Everything here is a pure function of its inputs (framework-agnostic): the
 * caller extracts per-message text and renders the chosen injections; this
 * module never touches a message shape.
 */

/**
 * Why a fact was chosen for (re-)injection. Mirrors the core
 * `FactInjectReason` trace enum.
 *
 * - `never` — not present in the transcript and never injected this session.
 * - `evicted` — injected earlier but its body is gone now (trimmed / compacted
 *   out of the window).
 * - `mutated` — the registered body changed since it was injected (the current
 *   body is absent and differs from the one last injected).
 */
export type InjectionReason = "never" | "evicted" | "mutated";

/** The verdict `fresh` marks a fact left alone because its body is still in context. */
export type InjectionDecisionReason = InjectionReason | "fresh";

/**
 * The two tiers a fact's `pin` splits into. A const object rather than a TS
 * `enum` (matching `SearchTarget`/`Origin` in `@ratel-ai/telemetry`): reference
 * it symbolically as `Pin.Always`, and the plain wire strings `"always"` /
 * `"retrieved"` stay assignable to it.
 */
export const Pin = {
  /** Always-on: injected every applicable turn, never dropped. */
  Always: "always",
  /** Retrieval-gated (the default): surfaced only when a query ranks it in. */
  Retrieved: "retrieved",
} as const;
export type Pin = (typeof Pin)[keyof typeof Pin];

/** A fact under consideration for injection this turn. */
export interface FactCandidate {
  /** The fact id — keys the session's injected-body memory and the trace events. */
  id: string;
  /** The fact's current body — the text whose presence in the transcript is checked. */
  body: string;
}

/** Input to {@link planInjection}. */
export interface PlanInjectionInput {
  /** The facts to consider this turn (pinned always-on facts plus retrieved hits). */
  candidates: readonly FactCandidate[];
  /** Per-message text of the current history, oldest first. */
  transcript: readonly string[];
  /**
   * The bodies this session already injected, keyed by fact id — the caller's
   * bookkeeping (e.g. {@link FactCatalog}'s). Refines the absent case: absent +
   * previously-injected-same-body ⇒ `evicted`; absent + previously-injected-
   * different-body ⇒ `mutated`; absent + unseen ⇒ `never`. Omit it and every
   * absent fact reads as `never`.
   */
  previouslyInjected?: ReadonlyMap<string, string>;
}

/** One fact's verdict from {@link planInjection}. */
export interface InjectionDecision {
  /** The fact id this verdict is for. */
  id: string;
  /** Whether to (re-)inject the fact this turn. */
  inject: boolean;
  /** Why — an {@link InjectionReason} when injecting, `fresh` when skipping. */
  reason: InjectionDecisionReason;
}

/**
 * Decide, per candidate, whether to inject its body this turn — the heart of
 * the freshness gate. Pure and deterministic: the verdicts come back in
 * candidate order and depend only on the inputs, so repeated calls in one turn
 * never disagree.
 *
 * Presence is a literal substring check of the candidate's body against the
 * transcript text — no regex, no parsing, no markers; the fastest and most
 * robust form of the test, and semantically honest: it answers "is this
 * information in the window?" regardless of who put it there. A candidate with
 * an empty body is trivially present (there is nothing to inject) and is
 * skipped as `fresh`.
 *
 * @param input - Candidates, the transcript, and the session's injected-body memory.
 * @returns One {@link InjectionDecision} per candidate, in the same order.
 */
export function planInjection(input: PlanInjectionInput): InjectionDecision[] {
  // One haystack, one substring scan per candidate. Bodies are injected as
  // (part of) a single message, so a per-message join can't split them.
  const haystack = input.transcript.join("\n");
  const previous = input.previouslyInjected;

  return input.candidates.map((candidate): InjectionDecision => {
    if (candidate.body === "" || haystack.includes(candidate.body)) {
      return { id: candidate.id, inject: false, reason: "fresh" };
    }
    const lastInjected = previous?.get(candidate.id);
    if (lastInjected === undefined) {
      return { id: candidate.id, inject: true, reason: "never" };
    }
    return {
      id: candidate.id,
      inject: true,
      reason: lastInjected === candidate.body ? "evicted" : "mutated",
    };
  });
}

/** One fact the grounding pass decided to (re-)inject this turn. */
export interface GroundingItem {
  /** The fact id. */
  id: string;
  /**
   * The fact body — render it **verbatim** as (part of) the message content;
   * its presence in the transcript is what dedupes the next turn.
   */
  body: string;
  /** Why it was injected. */
  reason: InjectionReason;
  /** Which tier it came from. */
  pin: Pin;
}

/** The outcome of a grounding pass — what to inject and what was left fresh. */
export interface GroundingResult {
  /** Facts to render into the transcript, always-on tier first. */
  inject: GroundingItem[];
  /** Ids left alone because their body is still in the context (observability). */
  skipped: string[];
}

/**
 * One fact riding along in a per-call grounding snapshot
 * ({@link FactCatalog.groundSnapshot}) — nothing persisted.
 */
export interface GroundingSnapshotItem {
  /** The fact id. */
  id: string;
  /** The fact body. */
  body: string;
  /** Which tier it came from. */
  pin: Pin;
}

/** Per-call options for {@link FactCatalog.ground} and {@link FactCatalog.groundSnapshot}. */
export interface GroundOptions {
  /** Max retrieval-gated facts to consider (capped at 50, default 3). */
  topK?: number;
}

/**
 * The set of fact ids a catalog accepts. Ids ride in trace events and in
 * structured injection payloads (the adapter tool-pair shape), so they stay
 * conservative: letters, digits, and `. _ : -` only.
 */
export const FACT_ID_PATTERN = /^[A-Za-z0-9._:-]+$/;
