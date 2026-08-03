//! What survives: protection, then selection under a token budget.
//!
//! Entirely pure — importance scores come in as plain numbers on [`Atom`]s, so
//! every rule here is unit-tested with no model on disk.
//!
//! ## The budget
//!
//! `B = round(rate · total_tokens)`, measured in the **compression model's own**
//! tokens. Protected atoms are charged to `B` first and are always kept; the
//! remainder is spent on the rest, highest importance first. Two consequences
//! worth stating out loud, because both are deliberate:
//!
//! - Protected atoms are **excluded from the selection sample**. Leaving their
//!   injected `1.0`s in would raise the bar's own reference distribution and
//!   silently loosen compression for everything else.
//! - When protection alone exceeds `B`, protection wins and `budget_exceeded` is
//!   reported. Protection is a hard promise; `rate` is documented as
//!   approximate. The alternative — protection *added on top* of the rate —
//!   makes output size unbounded in the number of matches, with no signal.

use regex::Regex;

use super::error::{CompressorError, cfg};
use super::tokens::Atom;

/// A span of the input that must survive compression at any rate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProtectPattern {
    /// Match this exact text wherever it occurs.
    Literal(String),
    /// Match this regular expression (Rust `regex` syntax — no lookaround or
    /// backreferences).
    Regex(String),
}

/// Words whose loss inverts a claim rather than shortening it.
///
/// Compared against the atom's text with apostrophes and whitespace removed, so
/// `doesn't` matches `doesnt` — which works only because a contraction is a
/// single atom. Discourse markers (`but`, `however`) are deliberately absent:
/// they are common enough that protecting them would spend real budget without
/// protecting a claim.
const NEGATIONS: [&str; 24] = [
    "not", "no", "never", "none", "nor", "neither", "without", "cannot", "cant", "dont", "doesnt",
    "didnt", "isnt", "arent", "wasnt", "werent", "wont", "shouldnt", "couldnt", "wouldnt", "hasnt",
    "havent", "hadnt", "unless",
];

/// Compile caller-supplied protect patterns. A literal is escaped, so a pattern
/// containing regex metacharacters means what it says.
///
/// # Errors
///
/// [`CompressorError::Config`] for an unparseable regex — failing fast beats
/// silently protecting nothing.
pub(crate) fn compile_protectors(
    patterns: &[ProtectPattern],
) -> Result<Vec<Regex>, CompressorError> {
    patterns
        .iter()
        .map(|p| {
            let source = match p {
                ProtectPattern::Literal(s) => regex::escape(s),
                ProtectPattern::Regex(s) => s.clone(),
            };
            Regex::new(&source).map_err(|e| cfg(format!("invalid protect pattern: {e}")))
        })
        .collect()
}

/// Options the protection pass reads. A borrowed view so `policy` needs no
/// dependency on the public options struct.
pub(crate) struct Protection<'a> {
    pub patterns: &'a [Regex],
    pub numbers: bool,
    pub negations: bool,
    pub negation_terms: Option<&'a [String]>,
}

/// Mark atoms that must survive, setting `protected` and pinning `importance` to
/// `1.0`. Returns how many were marked.
///
/// Pattern matches protect any atom they **overlap**, not only ones they
/// contain, so a pattern straddling a word boundary protects both sides rather
/// than neither.
pub(crate) fn apply_protection(text: &str, atoms: &mut [Atom], opts: &Protection<'_>) -> u32 {
    for pattern in opts.patterns {
        for m in pattern.find_iter(text) {
            let (ms, me) = (m.start(), m.end());
            if ms == me {
                continue; // an empty match protects nothing
            }
            // Atoms are sorted and disjoint; take the overlapping window.
            let first = atoms.partition_point(|a| a.end <= ms);
            for atom in atoms[first..].iter_mut() {
                if atom.start >= me {
                    break;
                }
                atom.protected = true;
            }
        }
    }

    if opts.numbers || opts.negations {
        let custom = opts.negation_terms;
        let mut negations = Vec::new();
        for (i, atom) in atoms.iter_mut().enumerate() {
            if atom.protected {
                continue;
            }
            let raw = &text[atom.start..atom.end];
            if opts.numbers && raw.bytes().any(|b| b.is_ascii_digit()) {
                atom.protected = true;
                continue;
            }
            if opts.negations && is_negation(raw, custom) {
                atom.protected = true;
                negations.push(i);
            }
        }
        // Protecting a negation without what it negates is worse than protecting
        // neither: the surviving negator bonds to the next word that happens to
        // live, forging a claim the input never made. `without any downtime.
        // Then the actual cutover` compressed to `without cutover` — the exact
        // inversion of the plan. Extend each negation's protection forward to
        // the content word it governs.
        for i in negations {
            for atom in atoms.iter_mut().skip(i + 1).take(NEGATION_SCOPE_ATOMS) {
                let raw = &text[atom.start..atom.end];
                if is_closed_class(raw) {
                    continue; // `any`, `the`, `a` — keep looking for the head
                }
                atom.protected = true;
                break;
            }
        }
    }

    let mut count = 0;
    for atom in atoms.iter_mut() {
        if atom.protected {
            atom.importance = 1.0;
            count += 1;
        }
    }
    count
}

/// Atoms to scan forward from a protected negation, looking for the content
/// word it governs. Three covers `without` → `any` → `downtime`; beyond that the
/// negation's scope is too far away to be confident about.
const NEGATION_SCOPE_ATOMS: usize = 3;

/// Dropped tokens between a kept function word and the next survivor, above
/// which the function word is assumed to have lost what it attaches to.
///
/// Small gaps usually mean a modifier went (`the quick brown fox` → `the fox`,
/// which still reads correctly). A large gap means the head noun or verb is gone
/// and the function word is now bonding to something unrelated.
const STRANDED_GAP_TOKENS: u32 = 3;

/// Words that carry no meaning alone — they exist to attach to something else.
/// Kept deliberately free of negations (`not`, `no`, `without`, …): those are
/// load-bearing on their own and are handled by [`apply_protection`], which
/// extends their protection to what they govern rather than dropping them.
const CLOSED_CLASS: [&str; 46] = [
    "a", "an", "the", "of", "in", "on", "at", "to", "for", "with", "from", "by", "about", "into",
    "onto", "over", "under", "between", "through", "during", "and", "or", "but", "if", "so", "as",
    "than", "then", "that", "this", "these", "those", "is", "are", "was", "were", "be", "been",
    "being", "am", "it", "its", "up", "out", "any", "some",
];

/// Whether an atom is a closed-class word, ignoring case and punctuation.
fn is_closed_class(raw: &str) -> bool {
    let normalized: String = raw
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    !normalized.is_empty() && CLOSED_CLASS.contains(&normalized.as_str())
}

/// Drop kept function words whose attachment target did not survive.
///
/// A preposition, article, or conjunction means nothing on its own — it points
/// at a neighbour. When selection keeps one but drops what it points at, the
/// renderer joins it to the next survivor and manufactures a phrase the input
/// never contained. Removing the orphan degrades the text; leaving it in
/// *falsifies* the text, which is strictly worse.
///
/// Protected atoms are never dropped here — protection is a hard promise, and a
/// protected negator has already had its scope extended by [`apply_protection`].
///
/// The freed budget is deliberately **not** re-spent: pulling in another atom
/// could strand another function word, and iterating to a fixed point buys less
/// than it costs in explicability.
pub(crate) fn attach_function_words(text: &str, atoms: &[Atom], selection: &mut Selection) {
    for i in 0..atoms.len() {
        if !selection.keep[i] || atoms[i].protected {
            continue;
        }
        if !is_closed_class(&text[atoms[i].start..atoms[i].end]) {
            continue;
        }
        // Tokens dropped between this atom and the next survivor. No survivor at
        // all means it trails the output with nothing to attach to.
        let mut gap = 0u32;
        let mut stranded = true;
        for (j, atom) in atoms.iter().enumerate().skip(i + 1) {
            if selection.keep[j] {
                stranded = gap >= STRANDED_GAP_TOKENS;
                break;
            }
            gap += atom.token_count();
        }
        if stranded {
            selection.keep[i] = false;
            selection.tokens_out -= atoms[i].token_count();
        }
    }
}

/// Whether an atom reads as a negation, ignoring case, apostrophes, and
/// whitespace so `doesn't` / `doesn t` / `Doesn't` all match.
fn is_negation(raw: &str, custom: Option<&[String]>) -> bool {
    let normalized: String = raw
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect();
    if normalized.is_empty() {
        return false;
    }
    match custom {
        Some(terms) => terms.iter().any(|t| t.eq_ignore_ascii_case(&normalized)),
        None => NEGATIONS.contains(&normalized.as_str()),
    }
}

/// The outcome of spending the budget.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Selection {
    /// Per-atom keep flags, in atom order.
    pub keep: Vec<bool>,
    /// Model tokens the kept atoms cost.
    pub tokens_out: u32,
}

/// Keep every unit the model rates at or above `min_importance`, plus anything
/// protected.
///
/// **There is no token budget.** An earlier version spent a
/// `rate × total_tokens` budget in importance order, which meant the compression
/// ratio was an input and the surviving content was whatever happened to fit.
/// That inverted the relationship: on `How do I reset a git branch to match
/// origin/main exactly?` the model rated five units 0.88–0.996 and rated
/// everything else below 0.11 — but a 6-token budget fitted only four of them,
/// so `origin/main` was destroyed to satisfy an arithmetic constraint the text
/// never agreed to.
///
/// The classifier already answers the only question that matters: which tokens
/// carry the meaning. Keeping exactly those makes the **ratio an output** — text
/// full of filler compresses hard, dense text barely compresses, and nothing
/// rated important is ever traded away to hit a number.
///
/// The practical consequence is that a caller with a hard context limit can no
/// longer demand a size. That is the honest failure mode: if the result still
/// doesn't fit, this text cannot be compressed that far without losing material
/// the model considers load-bearing, and the caller should know that rather than
/// receive a silently gutted version.
pub(crate) fn select(atoms: &[Atom], min_importance: f32) -> Selection {
    let mut keep = vec![false; atoms.len()];
    let mut tokens_out = 0u32;
    for (i, atom) in atoms.iter().enumerate() {
        // Protection is a hard promise and rides the same scale — `apply_protection`
        // pins those to 1.0, so they clear any threshold.
        if atom.protected || atom.importance >= min_importance {
            keep[i] = true;
            tokens_out += atom.token_count();
        }
    }
    Selection { keep, tokens_out }
}

#[cfg(test)]
mod tests {
    use super::super::tokens::tests::spans;
    use super::super::tokens::{apply_probs, atoms_from_tokens};
    use super::*;

    fn build(text: &str, pieces: &[(&str, u32)], probs: &[f32]) -> Vec<Atom> {
        let toks = spans(text, pieces);
        let mut atoms = atoms_from_tokens(text, &toks);
        apply_probs(&mut atoms, probs);
        atoms
    }

    fn kept<'a>(text: &'a str, atoms: &[Atom], sel: &Selection) -> Vec<&'a str> {
        atoms
            .iter()
            .zip(&sel.keep)
            .filter(|(_, k)| **k)
            .map(|(a, _)| &text[a.start..a.end])
            .collect()
    }

    fn none() -> Protection<'static> {
        Protection {
            patterns: &[],
            numbers: false,
            negations: false,
            negation_terms: None,
        }
    }

    // ---- selection ------------------------------------------------------

    #[test]
    fn selection_keeps_everything_at_or_above_the_bar() {
        let text = "alpha beta gamma delta";
        let atoms = build(
            text,
            &[("alpha", 0), ("beta", 1), ("gamma", 2), ("delta", 3)],
            &[0.1, 0.9, 0.2, 0.8],
        );
        let sel = select(&atoms, 0.5);
        assert_eq!(kept(text, &atoms, &sel), ["beta", "delta"]);
        assert_eq!(sel.tokens_out, 2);
    }

    /// The reason the budget was removed. The model rated five units 0.88–0.996
    /// and everything else below 0.11 — a clean split. A `rate`-driven budget of
    /// 6 tokens fitted only four of them and destroyed `origin/main` (0.88, 3
    /// tokens) to satisfy an arithmetic constraint. A bar keeps all five.
    #[test]
    fn an_expensive_unit_is_kept_when_it_clears_the_bar() {
        let text = "reset git branch match origin/main";
        let atoms = build(
            text,
            &[
                ("reset", 0),
                ("git", 1),
                ("branch", 2),
                ("match", 3),
                ("origin", 4),
                ("/", 5),
                ("main", 6),
            ],
            &[0.996, 0.990, 0.995, 0.993, 0.88, 0.88, 0.88],
        );
        assert_eq!(
            atoms[4].token_count(),
            3,
            "the identifier is one 3-token atom"
        );
        let sel = select(&atoms, 0.5);
        assert_eq!(
            kept(text, &atoms, &sel),
            ["reset", "git", "branch", "match", "origin/main"],
            "cost must not decide what survives"
        );
    }

    #[test]
    fn the_ratio_is_an_output_not_an_input() {
        // Same bar, two texts: redundant prose compresses hard, dense text
        // barely compresses. Neither trades away something above the bar.
        let filler = "the migration is really quite very much delayed";
        let redundant = build(
            filler,
            &[
                ("the", 0),
                ("migration", 1),
                ("is", 2),
                ("really", 3),
                ("quite", 4),
                ("very", 5),
                ("much", 6),
                ("delayed", 7),
            ],
            &[0.02, 0.99, 0.03, 0.05, 0.04, 0.02, 0.03, 0.97],
        );
        let dense_text = "bump pg_upgrade to 2.14 now";
        let dense = build(
            dense_text,
            &[
                ("bump", 0),
                ("pg_upgrade", 1),
                ("to", 2),
                ("2.14", 3),
                ("now", 4),
            ],
            &[0.95, 0.99, 0.06, 0.98, 0.91],
        );
        let a = select(&redundant, 0.5);
        let b = select(&dense, 0.5);
        assert_eq!(a.keep.iter().filter(|k| **k).count(), 2, "8 units -> 2");
        assert_eq!(b.keep.iter().filter(|k| **k).count(), 4, "5 units -> 4");
    }

    #[test]
    fn selection_is_deterministic() {
        let text = "a b c d";
        let atoms = build(text, &[("a", 0), ("b", 1), ("c", 2), ("d", 3)], &[0.5; 4]);
        let first = select(&atoms, 0.5);
        assert_eq!(first, select(&atoms, 0.5));
        assert_eq!(
            kept(text, &atoms, &first),
            ["a", "b", "c", "d"],
            "0.5 >= 0.5"
        );
    }

    #[test]
    fn a_zero_bar_keeps_everything() {
        let text = "a b c";
        let atoms = build(text, &[("a", 0), ("b", 1), ("c", 2)], &[0.1, 0.2, 0.3]);
        let sel = select(&atoms, 0.0);
        assert!(sel.keep.iter().all(|k| *k));
    }

    #[test]
    fn an_empty_atom_list_selects_nothing() {
        let sel = select(&[], 0.5);
        assert!(sel.keep.is_empty());
        assert_eq!(sel.tokens_out, 0);
    }

    // ---- protection ------------------------------------------------------

    #[test]
    fn protected_atoms_are_kept_regardless_of_score() {
        let text = "cost was 8,400 dollars total";
        let mut atoms = build(
            text,
            &[
                ("cost", 0),
                ("was", 1),
                ("8", 2),
                (",", 3),
                ("400", 4),
                ("dollars", 5),
                ("total", 6),
            ],
            &[0.9, 0.2, 0.1, 0.1, 0.1, 0.3, 0.25],
        );
        let n = apply_protection(
            text,
            &mut atoms,
            &Protection {
                numbers: true,
                ..none()
            },
        );
        assert_eq!(n, 1);
        // `8,400` scores 0.1 — far below the bar — and survives anyway.
        let sel = select(&atoms, 0.5);
        assert_eq!(kept(text, &atoms, &sel), ["cost", "8,400"]);
        assert_eq!(sel.tokens_out, 4);
    }

    #[test]
    fn protection_does_not_change_what_the_bar_does_to_anything_else() {
        // Protection pins its own atoms to 1.0 and leaves every other score
        // untouched, so it can never loosen or tighten the bar for the rest.
        let text = "1 2 3 alpha beta";
        let mut protectedd = build(
            text,
            &[("1", 0), ("2", 1), ("3", 2), ("alpha", 3), ("beta", 4)],
            &[0.0, 0.0, 0.0, 0.6, 0.4],
        );
        apply_protection(
            text,
            &mut protectedd,
            &Protection {
                numbers: true,
                ..none()
            },
        );
        let bare = build(
            text,
            &[("1", 0), ("2", 1), ("3", 2), ("alpha", 3), ("beta", 4)],
            &[0.0, 0.0, 0.0, 0.6, 0.4],
        );
        let with = select(&protectedd, 0.5);
        let without = select(&bare, 0.5);
        assert_eq!(kept(text, &protectedd, &with), ["1", "2", "3", "alpha"]);
        assert_eq!(kept(text, &bare, &without), ["alpha"]);
    }

    // ---- function-word attachment ---------------------------------------

    /// The bug this pass exists for. `without any downtime. Then the actual
    /// cutover` compressed to `without cutover` — a claim the input never made,
    /// and the exact inverse of the plan being described.
    #[test]
    fn a_protected_negation_also_protects_what_it_negates() {
        let text = "catch up without any downtime. Then the cutover is quick";
        let mut atoms = build(
            text,
            &[
                ("catch", 0),
                ("up", 1),
                ("without", 2),
                ("any", 3),
                ("downtime", 4),
                (".", 5),
                ("Then", 6),
                ("the", 7),
                ("cutover", 8),
                ("is", 9),
                ("quick", 10),
            ],
            &[0.5, 0.4, 0.2, 0.1, 0.15, 0.1, 0.3, 0.1, 0.9, 0.1, 0.6],
        );
        apply_protection(
            text,
            &mut atoms,
            &Protection {
                negations: true,
                ..none()
            },
        );
        // `without` is protected as a negation; `any` is closed-class so the
        // scope walks past it to `downtime`, the word actually being negated.
        let protected: Vec<&str> = atoms
            .iter()
            .filter(|a| a.protected)
            .map(|a| &text[a.start..a.end])
            .collect();
        // `downtime.` is one atom — the period is byte-adjacent to it.
        assert_eq!(protected, ["without", "downtime."]);

        // With both protected, the negation can no longer bond to `cutover`.
        let mut sel = select(&atoms, 0.5);
        attach_function_words(text, &atoms, &mut sel);
        let out = kept(text, &atoms, &sel);
        assert!(out.contains(&"without"), "got {out:?}");
        assert!(
            out.contains(&"downtime."),
            "the negated word must survive: {out:?}"
        );
    }

    #[test]
    fn a_stranded_function_word_is_dropped_rather_than_rebonded() {
        // `of` keeps nothing it can attach to — four tokens vanish after it.
        let text = "the cost of running both clusters in parallel today";
        let atoms = build(
            text,
            &[
                ("the", 0),
                ("cost", 1),
                ("of", 2),
                ("running", 3),
                ("both", 4),
                ("clusters", 5),
                ("in", 6),
                ("parallel", 7),
                ("today", 8),
            ],
            &[0.2, 0.9, 0.85, 0.1, 0.1, 0.1, 0.1, 0.2, 0.8],
        );
        let mut sel = select(&atoms, 0.5);
        assert!(
            kept(text, &atoms, &sel).contains(&"of"),
            "precondition: `of` clears the bar on its own"
        );
        attach_function_words(text, &atoms, &mut sel);
        let out = kept(text, &atoms, &sel);
        assert!(
            !out.contains(&"of"),
            "orphaned preposition must go: {out:?}"
        );
        assert_eq!(out, ["cost", "today"]);
        assert_eq!(sel.tokens_out, 2, "the freed token is reported");
    }

    #[test]
    fn a_function_word_whose_target_survives_is_kept() {
        // Only a modifier was dropped, so `the` still attaches correctly —
        // `the quick fox` reads fine and must not be mangled into `quick fox`.
        let text = "the quick fox ran";
        let atoms = build(
            text,
            &[("the", 0), ("quick", 1), ("fox", 2), ("ran", 3)],
            &[0.8, 0.1, 0.9, 0.85],
        );
        let mut sel = select(&atoms, 0.75); // drops only `quick`
        attach_function_words(text, &atoms, &mut sel);
        assert_eq!(kept(text, &atoms, &sel), ["the", "fox", "ran"]);
    }

    #[test]
    fn a_trailing_function_word_is_dropped() {
        let text = "sign-off from Priya last Thursday afternoon";
        let atoms = build(
            text,
            &[
                ("sign-off", 0),
                ("from", 1),
                ("Priya", 2),
                ("last", 3),
                ("Thursday", 4),
                ("afternoon", 5),
            ],
            &[0.9, 0.85, 0.9, 0.1, 0.1, 0.1],
        );
        let mut sel = select(&atoms, 0.55);
        attach_function_words(text, &atoms, &mut sel);
        // `from` still points at `Priya`, so it stays; nothing trails it.
        assert_eq!(kept(text, &atoms, &sel), ["sign-off", "from", "Priya"]);
    }

    #[test]
    fn protection_outranks_the_attachment_pass() {
        // Protection is a hard promise: a user-protected function word survives
        // even with nothing to attach to.
        let text = "measured in furlongs per fortnight overall";
        let mut atoms = build(
            text,
            &[
                ("measured", 0),
                ("in", 1),
                ("furlongs", 2),
                ("per", 3),
                ("fortnight", 4),
                ("overall", 5),
            ],
            &[0.9, 0.1, 0.1, 0.1, 0.1, 0.8],
        );
        let res = compile_protectors(&[ProtectPattern::Literal("in".into())]).unwrap();
        apply_protection(
            text,
            &mut atoms,
            &Protection {
                patterns: &res,
                ..none()
            },
        );
        let mut sel = select(&atoms, 0.5);
        attach_function_words(text, &atoms, &mut sel);
        assert!(
            kept(text, &atoms, &sel).contains(&"in"),
            "a protected atom must survive the attachment pass"
        );
    }

    #[test]
    fn negations_are_not_in_the_closed_class_list() {
        // They carry meaning alone, so they are protected and scope-extended
        // rather than dropped as orphans.
        for word in ["not", "no", "never", "without", "cannot", "neither"] {
            assert!(!is_closed_class(word), "{word} must not be closed-class");
        }
        for word in ["the", "of", "and", "with", "is"] {
            assert!(is_closed_class(word), "{word} must be closed-class");
        }
    }

    #[test]
    fn a_custom_negation_list_replaces_the_builtin_one() {
        let text = "kein problem";
        let mut atoms = build(text, &[("kein", 0), ("problem", 1)], &[0.1, 0.9]);
        let terms = vec!["kein".to_string()];
        apply_protection(
            text,
            &mut atoms,
            &Protection {
                negations: true,
                negation_terms: Some(&terms),
                ..none()
            },
        );
        assert!(atoms[0].protected, "the custom term is recognized");
        assert!(
            atoms[1].protected,
            "and its scope is extended to what it negates"
        );
    }

    #[test]
    fn digits_anywhere_in_an_atom_protect_it() {
        let text = "v2 plain 3rd";
        let mut atoms = build(
            text,
            &[("v2", 0), ("plain", 1), ("3rd", 2)],
            &[0.1, 0.1, 0.1],
        );
        apply_protection(
            text,
            &mut atoms,
            &Protection {
                numbers: true,
                ..none()
            },
        );
        assert_eq!(
            atoms.iter().map(|a| a.protected).collect::<Vec<_>>(),
            [true, false, true]
        );
    }

    #[test]
    fn a_pattern_protects_every_atom_it_overlaps() {
        // The match straddles the boundary between `beta` and `gamma`; both must
        // be protected, not neither.
        let text = "alpha beta gamma delta";
        let mut atoms = build(
            text,
            &[("alpha", 0), ("beta", 1), ("gamma", 2), ("delta", 3)],
            &[0.1; 4],
        );
        let res = compile_protectors(&[ProtectPattern::Regex("ta gam".into())]).unwrap();
        let n = apply_protection(
            text,
            &mut atoms,
            &Protection {
                patterns: &res,
                ..none()
            },
        );
        assert_eq!(n, 2);
        assert_eq!(
            atoms.iter().map(|a| a.protected).collect::<Vec<_>>(),
            [false, true, true, false]
        );
    }

    #[test]
    fn a_literal_pattern_is_escaped_so_metacharacters_mean_themselves() {
        let text = "use a.b or axb";
        let mut atoms = build(
            text,
            &[("use", 0), ("a.b", 1), ("or", 2), ("axb", 3)],
            &[0.1; 4],
        );
        let res = compile_protectors(&[ProtectPattern::Literal("a.b".into())]).unwrap();
        apply_protection(
            text,
            &mut atoms,
            &Protection {
                patterns: &res,
                ..none()
            },
        );
        assert!(atoms[1].protected, "the literal must match");
        assert!(!atoms[3].protected, "`.` must not have matched `x`");
    }

    #[test]
    fn an_invalid_regex_is_a_config_error_not_a_silent_no_op() {
        let e = compile_protectors(&[ProtectPattern::Regex("a(".into())]).unwrap_err();
        assert!(matches!(e, CompressorError::Config { .. }));
        assert!(e.to_string().contains("invalid protect pattern"));
    }

    #[test]
    fn an_empty_match_protects_nothing() {
        let text = "alpha beta";
        let mut atoms = build(text, &[("alpha", 0), ("beta", 1)], &[0.1, 0.1]);
        let res = compile_protectors(&[ProtectPattern::Regex("x*".into())]).unwrap();
        assert_eq!(
            apply_protection(
                text,
                &mut atoms,
                &Protection {
                    patterns: &res,
                    ..none()
                }
            ),
            0
        );
    }
}
