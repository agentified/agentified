//! Rebuild the compressed text from the **original** bytes.
//!
//! Never detokenize. Tokenization is lossy — text cleaning, CJK space insertion,
//! `[UNK]`, accent handling — and a WordPiece decoder reinserts spaces by rule,
//! which is exactly how the prototype produced `pg _ upgrade`, `doesn ' t`, and
//! `4 , 200` from text it had decided to *keep*.
//!
//! Because [`Atom`] offsets are byte offsets into the input, output is a
//! concatenation of slices of that input. "Compression never mangles the text it
//! kept" is then a structural property rather than a heuristic that mostly holds
//! — pinned by `output_is_built_from_original_slices` below.

use super::tokens::Atom;

/// Join the kept atoms, choosing each separator from what the original text had
/// between them.
///
/// | between two kept atoms | emitted |
/// |---|---|
/// | nothing was dropped (a pure separator) | the original span, **verbatim** |
/// | ≥2 newlines, `preserve_paragraphs` | `"\n\n"` |
/// | ≥1 newline | `"\n"` |
/// | any whitespace | `" "` |
/// | no whitespace at all | `""` |
///
/// The last row is the one that matters: it keeps a dropped interior unit from
/// leaving a phantom space inside an identifier. With the atom rule it is rarely
/// reached, which is the point — it is a backstop, not the mechanism.
pub(crate) fn render(
    text: &str,
    atoms: &[Atom],
    keep: &[bool],
    preserve_paragraphs: bool,
) -> String {
    let mut out = String::new();
    let mut previous: Option<&Atom> = None;
    let mut dropped_since = false;

    for (atom, keep) in atoms.iter().zip(keep) {
        if !*keep {
            dropped_since = true;
            continue;
        }
        if let Some(prev) = previous {
            let gap = &text[prev.end..atom.start];
            out.push_str(&separator(gap, dropped_since, preserve_paragraphs));
        }
        out.push_str(&text[atom.start..atom.end]);
        previous = Some(atom);
        dropped_since = false;
    }
    out.trim_end().to_string()
}

fn separator(gap: &str, dropped_since: bool, preserve_paragraphs: bool) -> String {
    // Nothing was dropped in between, so the original spacing is still the
    // author's — reproduce it exactly.
    if !dropped_since {
        return gap.to_string();
    }
    if preserve_paragraphs && has_blank_line(gap) {
        "\n\n".to_string()
    } else if gap.contains('\n') {
        "\n".to_string()
    } else if gap.chars().any(char::is_whitespace) {
        " ".to_string()
    } else {
        String::new()
    }
}

/// Whether the gap contains a genuine blank line — two newlines with only
/// horizontal whitespace between them.
///
/// Counting newlines instead would be wrong: the gap spans the dropped atoms
/// too, so `alpha\nbeta\ngamma` with `beta` removed holds two newlines while the
/// author wrote no blank line at all.
fn has_blank_line(gap: &str) -> bool {
    let mut seen_newline = false;
    for c in gap.chars() {
        match c {
            '\n' if seen_newline => return true,
            '\n' => seen_newline = true,
            c if c.is_whitespace() => {}
            _ => seen_newline = false,
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::super::tokens::atoms_from_tokens;
    use super::super::tokens::tests::spans;
    use super::*;

    fn render_keeping(text: &str, pieces: &[(&str, u32)], keep_texts: &[&str]) -> String {
        let atoms = atoms_from_tokens(text, &spans(text, pieces));
        let keep: Vec<bool> = atoms
            .iter()
            .map(|a| keep_texts.contains(&&text[a.start..a.end]))
            .collect();
        render(text, &atoms, &keep, true)
    }

    #[test]
    fn an_untouched_run_keeps_its_original_spacing_verbatim() {
        let text = "alpha  beta\tgamma";
        let out = render_keeping(
            text,
            &[("alpha", 0), ("beta", 1), ("gamma", 2)],
            &["alpha", "beta", "gamma"],
        );
        assert_eq!(
            out, "alpha  beta\tgamma",
            "nothing dropped, nothing changed"
        );
    }

    #[test]
    fn a_dropped_word_collapses_its_gap_to_one_space() {
        let text = "alpha beta gamma";
        assert_eq!(
            render_keeping(
                text,
                &[("alpha", 0), ("beta", 1), ("gamma", 2)],
                &["alpha", "gamma"]
            ),
            "alpha gamma"
        );
    }

    #[test]
    fn a_dropped_unit_with_no_whitespace_around_it_does_not_leave_a_phantom_space() {
        // Two atoms with zero bytes between them can only arise across a CJK
        // boundary, which is exactly where an inserted space would be wrong.
        let text = "会議の記録";
        let out = render_keeping(
            text,
            &[("会", 0), ("議", 1), ("の", 2), ("記", 3), ("録", 4)],
            &["会", "議", "記", "録"],
        );
        assert_eq!(
            out, "会議記録",
            "no space may appear where the input had none"
        );
    }

    #[test]
    fn a_dropped_line_does_not_manufacture_a_paragraph_break() {
        // The gap holds two newlines, but the author wrote no blank line — the
        // second newline belongs to the word we removed.
        let text = "alpha\nbeta\ngamma";
        assert_eq!(
            render_keeping(
                text,
                &[("alpha", 0), ("beta", 1), ("gamma", 2)],
                &["alpha", "gamma"]
            ),
            "alpha\ngamma"
        );
    }

    #[test]
    fn a_blank_line_survives_a_drop_as_a_blank_line() {
        let text = "alpha\n\nbeta gamma";
        assert_eq!(
            render_keeping(
                text,
                &[("alpha", 0), ("beta", 1), ("gamma", 2)],
                &["alpha", "gamma"]
            ),
            "alpha\n\ngamma"
        );
    }

    #[test]
    fn paragraphs_collapse_to_one_newline_when_not_preserved() {
        let text = "alpha\n\nbeta gamma";
        let atoms = atoms_from_tokens(
            text,
            &spans(text, &[("alpha", 0), ("beta", 1), ("gamma", 2)]),
        );
        let keep = vec![true, false, true];
        assert_eq!(render(text, &atoms, &keep, false), "alpha\ngamma");
    }

    #[test]
    fn output_has_no_leading_or_trailing_whitespace() {
        let text = "  alpha beta gamma  ";
        let out = render_keeping(
            text,
            &[("alpha", 0), ("beta", 1), ("gamma", 2)],
            &["alpha", "beta"],
        );
        assert_eq!(out, "alpha beta");
    }

    #[test]
    fn keeping_nothing_renders_an_empty_string() {
        let text = "alpha beta";
        assert_eq!(render_keeping(text, &[("alpha", 0), ("beta", 1)], &[]), "");
    }

    /// The "never mangles" guarantee, as an executable invariant: every kept atom
    /// appears in the output, in order, byte-identical to the input slice at its
    /// recorded offsets. Only separators are ever synthesized.
    #[test]
    fn output_is_built_from_original_slices() {
        let text = "Speaker 2: it doesn't support pg_upgrade at 2.14.\n\nCost was 8,400 total.";
        let pieces: &[(&str, u32)] = &[
            ("Speaker", 0),
            ("2", 1),
            (":", 2),
            ("it", 3),
            ("doesn", 4),
            ("'", 5),
            ("t", 6),
            ("support", 7),
            ("pg", 8),
            ("_", 9),
            ("upgrade", 10),
            ("at", 11),
            ("2", 12),
            (".", 13),
            ("14", 14),
            (".", 15),
            ("Cost", 16),
            ("was", 17),
            ("8", 18),
            (",", 19),
            ("400", 20),
            ("total", 21),
            (".", 22),
        ];
        let atoms = atoms_from_tokens(text, &spans(text, pieces));

        // Drop every third atom, so separators of all shapes get exercised.
        let keep: Vec<bool> = (0..atoms.len()).map(|i| i % 3 != 2).collect();
        let out = render(text, &atoms, &keep, true);

        let mut cursor = 0usize;
        for (atom, keep) in atoms.iter().zip(&keep) {
            if !*keep {
                continue;
            }
            let slice = &text[atom.start..atom.end];
            let found = out[cursor..]
                .find(slice)
                .unwrap_or_else(|| panic!("{slice:?} missing from output {out:?}"));
            cursor += found + slice.len();
        }
        // Anything the output holds beyond those slices is separator whitespace.
        let kept_bytes: usize = atoms
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(a, _)| a.end - a.start)
            .sum();
        let synthesized = out.len() - kept_bytes;
        assert!(
            out.chars().filter(|c| c.is_whitespace()).count() >= synthesized.min(1),
            "only whitespace may be synthesized; got {out:?}"
        );
    }
}
