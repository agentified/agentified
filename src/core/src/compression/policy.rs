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
        for atom in atoms.iter_mut() {
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
    /// Protection alone exceeded the budget, so `rate` was overrun.
    pub budget_exceeded: bool,
}

/// Keep protected atoms, then spend what remains of the budget on the rest,
/// highest importance first.
///
/// This is the **exact** form of a token-budget-weighted threshold: charging each
/// atom its token cost and taking in importance order selects precisely the atoms
/// a token-weighted quantile would, without percentile interpolation or a
/// tie-comparison that over- or under-shoots. Ties break by atom index, so output
/// is byte-stable across runs.
///
/// One refinement over a pure threshold: an atom that does not fit is **skipped,
/// not stopped at**, so a single long atom cannot starve every shorter atom
/// behind it.
pub(crate) fn select(atoms: &[Atom], rate: f32) -> Selection {
    let mut keep = vec![false; atoms.len()];
    let total_tokens: u32 = atoms.iter().map(Atom::token_count).sum();
    if total_tokens == 0 {
        return Selection {
            keep,
            tokens_out: 0,
            budget_exceeded: false,
        };
    }

    let budget = (rate.clamp(0.0, 1.0) * total_tokens as f32).round() as u32;

    let mut protected_cost = 0u32;
    for (i, atom) in atoms.iter().enumerate() {
        if atom.protected {
            keep[i] = true;
            protected_cost += atom.token_count();
        }
    }

    if protected_cost >= budget {
        // Protection is a hard promise; `rate` is approximate. Say so rather than
        // dropping something the caller marked as load-bearing.
        return Selection {
            keep,
            tokens_out: protected_cost,
            budget_exceeded: protected_cost > budget,
        };
    }

    let mut free_budget = budget - protected_cost;
    // Descending importance, ties by atom index — deterministic across runs.
    let mut candidates: Vec<usize> = (0..atoms.len()).filter(|&i| !atoms[i].protected).collect();
    candidates.sort_by(|&a, &b| {
        atoms[b]
            .importance
            .total_cmp(&atoms[a].importance)
            .then(a.cmp(&b))
    });

    let mut tokens_out = protected_cost;
    for i in candidates {
        let cost = atoms[i].token_count();
        if cost <= free_budget {
            keep[i] = true;
            free_budget -= cost;
            tokens_out += cost;
        }
        // else: skip this atom, keep spending on shorter ones behind it.
    }

    Selection {
        keep,
        tokens_out,
        budget_exceeded: false,
    }
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
    fn selection_keeps_the_most_important_atoms_within_the_budget() {
        let text = "alpha beta gamma delta";
        let atoms = build(
            text,
            &[("alpha", 0), ("beta", 1), ("gamma", 2), ("delta", 3)],
            &[0.1, 0.9, 0.2, 0.8],
        );
        let sel = select(&atoms, 0.5); // 4 tokens -> budget 2
        assert_eq!(kept(text, &atoms, &sel), ["beta", "delta"]);
        assert_eq!(sel.tokens_out, 2);
        assert!(!sel.budget_exceeded);
    }

    #[test]
    fn the_budget_is_measured_in_tokens_not_words() {
        // `internationalization` costs 4 tokens; three 1-token words cost 3.
        // A word-counting budget would keep the long word and one other; a token
        // budget keeps all three short ones.
        let text = "internationalization a b c";
        let atoms = build(
            text,
            &[
                ("intern", 0),
                ("ation", 0),
                ("aliz", 0),
                ("ation", 0),
                ("a", 1),
                ("b", 2),
                ("c", 3),
            ],
            &[0.9, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5],
        );
        assert_eq!(atoms[0].token_count(), 4);
        let sel = select(&atoms, 0.45); // 7 tokens -> budget 3
        assert_eq!(kept(text, &atoms, &sel), ["a", "b", "c"]);
        assert_eq!(sel.tokens_out, 3);
    }

    #[test]
    fn an_oversized_atom_is_skipped_rather_than_stopping_selection() {
        let text = "enormous a b";
        let atoms = build(
            text,
            &[("en", 0), ("orm", 0), ("ous", 0), ("a", 1), ("b", 2)],
            &[0.99, 0.99, 0.99, 0.5, 0.4],
        );
        let sel = select(&atoms, 0.4); // 5 tokens -> budget 2; the 3-token atom can't fit
        assert_eq!(kept(text, &atoms, &sel), ["a", "b"]);
    }

    #[test]
    fn ties_break_by_position_so_output_is_stable() {
        let text = "a b c d";
        let atoms = build(text, &[("a", 0), ("b", 1), ("c", 2), ("d", 3)], &[0.5; 4]);
        let first = select(&atoms, 0.5);
        assert_eq!(kept(text, &atoms, &first), ["a", "b"]);
        assert_eq!(
            first,
            select(&atoms, 0.5),
            "selection must be deterministic"
        );
    }

    #[test]
    fn rate_one_keeps_everything() {
        let text = "a b c";
        let atoms = build(text, &[("a", 0), ("b", 1), ("c", 2)], &[0.1, 0.2, 0.3]);
        let sel = select(&atoms, 1.0);
        assert!(sel.keep.iter().all(|k| *k));
        assert!(!sel.budget_exceeded);
    }

    #[test]
    fn an_empty_atom_list_selects_nothing_without_dividing_by_zero() {
        let sel = select(&[], 0.4);
        assert!(sel.keep.is_empty());
        assert_eq!(sel.tokens_out, 0);
    }

    // ---- protection x budget --------------------------------------------

    #[test]
    fn protected_atoms_are_kept_and_charged_to_the_budget_first() {
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
        // 7 tokens -> budget 4. `8,400` costs 3 of them despite scoring lowest.
        let sel = select(&atoms, 0.55);
        assert_eq!(kept(text, &atoms, &sel), ["cost", "8,400"]);
        assert_eq!(sel.tokens_out, 4);
    }

    #[test]
    fn protection_is_excluded_from_the_selection_sample() {
        // The injected 1.0s must not become the reference the rest is judged
        // against — otherwise protecting content would loosen compression
        // everywhere else.
        let text = "1 2 3 alpha beta";
        let mut atoms = build(
            text,
            &[("1", 0), ("2", 1), ("3", 2), ("alpha", 3), ("beta", 4)],
            &[0.0, 0.0, 0.0, 0.6, 0.4],
        );
        apply_protection(
            text,
            &mut atoms,
            &Protection {
                numbers: true,
                ..none()
            },
        );
        // 5 tokens -> budget 4; 3 go to the digits, 1 remains for the best of the rest.
        let sel = select(&atoms, 0.8);
        assert_eq!(kept(text, &atoms, &sel), ["1", "2", "3", "alpha"]);
    }

    #[test]
    fn protection_over_budget_keeps_only_protected_and_flags_it() {
        let text = "1 2 3 4 alpha";
        let mut atoms = build(
            text,
            &[("1", 0), ("2", 1), ("3", 2), ("4", 3), ("alpha", 4)],
            &[0.0, 0.0, 0.0, 0.0, 0.9],
        );
        apply_protection(
            text,
            &mut atoms,
            &Protection {
                numbers: true,
                ..none()
            },
        );
        let sel = select(&atoms, 0.2); // budget 1, protection costs 4
        assert_eq!(kept(text, &atoms, &sel), ["1", "2", "3", "4"]);
        assert!(
            sel.budget_exceeded,
            "the overrun must be reported, not hidden"
        );
        assert_eq!(sel.tokens_out, 4);
    }

    #[test]
    fn protection_exactly_at_budget_is_not_an_overrun() {
        let text = "1 2 alpha beta";
        let mut atoms = build(
            text,
            &[("1", 0), ("2", 1), ("alpha", 2), ("beta", 3)],
            &[0.0, 0.0, 0.9, 0.8],
        );
        apply_protection(
            text,
            &mut atoms,
            &Protection {
                numbers: true,
                ..none()
            },
        );
        let sel = select(&atoms, 0.5); // budget 2 == protected cost
        assert!(!sel.budget_exceeded);
        assert_eq!(kept(text, &atoms, &sel), ["1", "2"]);
    }

    // ---- what gets protected --------------------------------------------

    #[test]
    fn negations_are_protected_by_default_and_can_be_disabled() {
        let text = "it doesn't support that";
        let pieces: &[(&str, u32)] = &[
            ("it", 0),
            ("doesn", 1),
            ("'", 2),
            ("t", 3),
            ("support", 4),
            ("that", 5),
        ];
        let probs = &[0.9, 0.1, 0.1, 0.1, 0.8, 0.2];

        let mut on = build(text, pieces, probs);
        apply_protection(
            text,
            &mut on,
            &Protection {
                negations: true,
                ..none()
            },
        );
        assert!(on[1].protected, "`doesn't` must be protected");
        // Budget 3 of 6 tokens, and the contraction costs all 3 — protection buys
        // the negation at the price of everything else, which is the trade.
        assert_eq!(kept(text, &on, &select(&on, 0.5)), ["doesn't"]);

        // Without it, the negation scores lowest and is dropped outright,
        // inverting the claim — the prototype's exact failure.
        let off = build(text, pieces, probs);
        assert_eq!(
            kept(text, &off, &select(&off, 0.5)),
            ["it", "support", "that"]
        );
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
        assert!(atoms[0].protected);
        assert!(!atoms[1].protected);
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
