//! From model tokens to the units compression actually decides on.
//!
//! The whole prompt is encoded **once**; everything here operates on that single
//! token stream, described by [`TokenSpan`]s. Nothing in this module runs a
//! model, so the structural rules — what an atom is, where sentences break, how
//! chunks partition the stream — are unit-tested against hand-built token
//! streams with no weights on disk.
//!
//! ## Why atoms
//!
//! The classifier scores **subwords**; meaning lives in **words**. Deciding at
//! the subword is what shredded the prototype's output: `doesn't` scored as
//! `doesn` + `'` + `t` loses its negation, `8,400` keeps half a number, and
//! `pg_upgrade` comes back as `pg _ upgrade`.
//!
//! An [`Atom`] is a maximal run of tokens with **no bytes between them in the
//! original text**, kept or dropped as one unit. That makes the unit of the
//! decision match the unit of meaning, and it is the single rule that fixes all
//! three failures.

use std::ops::Range;

/// One model token, located in the **original** text.
///
/// `start`/`end` are byte offsets produced by the tokenizer (`Encoding` offsets
/// are byte offsets into the input, so no arithmetic is needed to map back).
/// `word` is the pre-tokenizer split index (`Encoding::get_word_ids`) — used in
/// preference to a `##` prefix test because it is correct for CJK, for
/// punctuation, and for any WordPiece variant that marks continuations
/// differently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TokenSpan {
    /// Vocabulary id, carried so chunk rows are built without re-encoding.
    pub id: u32,
    /// Pre-tokenizer word index this token belongs to.
    pub word: u32,
    /// Byte offset of the token's first byte in the original text.
    pub start: usize,
    /// Byte offset one past the token's last byte.
    pub end: usize,
}

/// A keep-or-drop unit: a run of tokens with no bytes between them.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Atom {
    /// Byte range in the original text.
    pub start: usize,
    /// Byte offset one past the atom's last byte.
    pub end: usize,
    /// Indices into the token stream this atom covers.
    pub tokens: Range<usize>,
    /// Mean `P(INCLUDE)` over the atom's tokens; `1.0` once protected.
    pub importance: f32,
    /// Set by the policy layer when the atom must survive at any rate.
    pub protected: bool,
    /// Index of the sentence this atom belongs to.
    pub sentence: u32,
}

impl Atom {
    /// Model tokens this atom costs — the weight in the budget and the quantile.
    pub(crate) fn token_count(&self) -> u32 {
        self.tokens.len() as u32
    }
}

/// Group a token stream into atoms and assign sentence indices.
///
/// Two adjacent tokens join when they belong to the same pre-tokenizer word, or
/// when they are byte-adjacent in the original and neither side is CJK. The CJK
/// exception exists because WordPiece's `handle_chinese_chars` splits every
/// ideograph into its own word with no separating bytes — without it a Chinese
/// sentence would collapse into one unsplittable atom and compress to nothing.
pub(crate) fn atoms_from_tokens(text: &str, tokens: &[TokenSpan]) -> Vec<Atom> {
    let mut atoms: Vec<Atom> = Vec::new();
    for (i, tok) in tokens.iter().enumerate() {
        let joins = match (atoms.last(), tokens.get(i.wrapping_sub(1))) {
            (Some(last), Some(prev)) => {
                prev.word == tok.word || (last.end == tok.start && glues(text, last.end))
            }
            _ => false,
        };
        if joins {
            let last = atoms.last_mut().expect("joins implies a previous atom");
            last.end = tok.end;
            last.tokens.end = i + 1;
        } else {
            atoms.push(Atom {
                start: tok.start,
                end: tok.end,
                tokens: i..i + 1,
                importance: 0.0,
                protected: false,
                sentence: 0,
            });
        }
    }
    assign_sentences(text, &mut atoms);
    atoms
}

/// Whether two byte-adjacent words glue into one atom, judged at their shared
/// boundary `at`. False across a CJK boundary (see [`atoms_from_tokens`]).
fn glues(text: &str, at: usize) -> bool {
    let before = text[..at].chars().next_back();
    let after = text[at..].chars().next();
    !matches!(before, Some(c) if is_cjk(c)) && !matches!(after, Some(c) if is_cjk(c))
}

/// Scripts whose pre-tokenizer splits every character, so byte-adjacency carries
/// no information about word boundaries.
fn is_cjk(c: char) -> bool {
    matches!(c as u32,
        0x3040..=0x30FF     // Hiragana, Katakana
        | 0x3400..=0x4DBF   // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF   // CJK Unified Ideographs
        | 0xAC00..=0xD7AF   // Hangul syllables
        | 0xF900..=0xFAFF   // CJK Compatibility Ideographs
        | 0x20000..=0x2FA1F // Extensions B..F + Compatibility Supplement
    )
}

/// Sentence terminators, including their full-width CJK forms.
const TERMINATORS: [char; 7] = ['.', '?', '!', '…', '。', '！', '？'];

/// Number each atom with its sentence.
///
/// A break happens after an atom that ends in terminal punctuation and is
/// followed by whitespace, or across a blank line. Because a decimal like `2.14`
/// is a *single atom*, its interior period is never a candidate — the atom rule
/// pays for itself twice. An abbreviation (`e.g.`) does break a sentence early;
/// that only shifts a chunk boundary, which is why it is tolerated rather than
/// special-cased with a word list.
fn assign_sentences(text: &str, atoms: &mut [Atom]) {
    let mut sentence = 0u32;
    for i in 0..atoms.len() {
        atoms[i].sentence = sentence;
        let Some(next) = atoms.get(i + 1) else { break };
        let gap = &text[atoms[i].end..next.start];
        let ends_sentence = text[..atoms[i].end]
            .chars()
            .next_back()
            .is_some_and(|c| TERMINATORS.contains(&c));
        let blank_line = gap.matches('\n').count() >= 2;
        if blank_line || (ends_sentence && gap.chars().any(char::is_whitespace)) {
            sentence += 1;
        }
    }
}

/// Partition the token stream into chunks that each fit the model's positional
/// limit.
///
/// Greedy forward fill, preferring a sentence boundary, then an atom boundary,
/// and hard-cutting only inside an atom longer than a whole chunk (a base64 blob
/// or similar). Splitting an atom across chunks is harmless — the atom is still
/// selected as one unit, its importance is just averaged over probabilities from
/// two forward passes.
///
/// The returned ranges **exactly partition** `0..token_count`: contiguous, no
/// gap, no overlap. Nothing is ever truncated, which is why the prototype's
/// off-by-one chunk bug (which silently ate the last token of every chunk) has
/// no analogue here.
pub(crate) fn plan_chunks(
    token_count: usize,
    atoms: &[Atom],
    max_body: usize,
) -> Vec<Range<usize>> {
    assert!(max_body > 0, "max_body must be positive");
    if token_count == 0 {
        return Vec::new();
    }

    // Token indices at which a chunk may begin, and the subset that also starts a
    // sentence. Both are ascending by construction.
    let mut atom_starts: Vec<usize> = Vec::with_capacity(atoms.len());
    let mut sentence_starts: Vec<usize> = Vec::new();
    for (i, atom) in atoms.iter().enumerate() {
        atom_starts.push(atom.tokens.start);
        if i == 0 || atoms[i - 1].sentence != atom.sentence {
            sentence_starts.push(atom.tokens.start);
        }
    }

    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < token_count {
        let limit = start + max_body;
        if limit >= token_count {
            chunks.push(start..token_count);
            break;
        }
        // Furthest boundary that still fits, preferring sentence over atom.
        let cut = last_boundary_within(&sentence_starts, start, limit)
            .or_else(|| last_boundary_within(&atom_starts, start, limit))
            .unwrap_or(limit); // an atom longer than a chunk: hard-cut inside it
        chunks.push(start..cut);
        start = cut;
    }
    chunks
}

/// The largest boundary in `starts` that is `> after` and `<= limit`.
fn last_boundary_within(starts: &[usize], after: usize, limit: usize) -> Option<usize> {
    starts.iter().copied().rfind(|&s| s > after && s <= limit)
}

/// Fill each atom's importance with the mean `P(INCLUDE)` over its tokens.
///
/// A plain mean over the atom's subwords: every token carries one unit of the
/// budget, so weighting by token count within an atom would be weighting by one.
pub(crate) fn apply_probs(atoms: &mut [Atom], probs: &[f32]) {
    for atom in atoms.iter_mut() {
        let slice = &probs[atom.tokens.clone()];
        atom.importance = if slice.is_empty() {
            0.0
        } else {
            slice.iter().sum::<f32>() / slice.len() as f32
        };
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Build a token stream from `(text_slice, word_index)` pairs by finding each
    /// slice in `text`, so tests read as text rather than as offset arithmetic.
    /// Ids are positional — nothing here depends on real vocabulary ids.
    pub(crate) fn spans(text: &str, pieces: &[(&str, u32)]) -> Vec<TokenSpan> {
        let mut cursor = 0usize;
        pieces
            .iter()
            .enumerate()
            .map(|(i, (piece, word))| {
                let start = cursor
                    + text[cursor..]
                        .find(piece)
                        .unwrap_or_else(|| panic!("{piece:?} not found after byte {cursor}"));
                cursor = start + piece.len();
                TokenSpan {
                    id: i as u32,
                    word: *word,
                    start,
                    end: cursor,
                }
            })
            .collect()
    }

    fn atom_texts(text: &str, atoms: &[Atom]) -> Vec<String> {
        atoms
            .iter()
            .map(|a| text[a.start..a.end].to_string())
            .collect()
    }

    #[test]
    fn a_contraction_is_one_atom() {
        // The prototype dropped `doesn't` -> `support`, inverting the claim.
        let text = "it doesn't support that";
        let toks = spans(
            text,
            &[
                ("it", 0),
                ("doesn", 1),
                ("'", 2),
                ("t", 3),
                ("support", 4),
                ("that", 5),
            ],
        );
        assert_eq!(
            atom_texts(text, &atoms_from_tokens(text, &toks)),
            ["it", "doesn't", "support", "that"]
        );
    }

    #[test]
    fn numbers_and_dotted_identifiers_are_one_atom_each() {
        let text = "cost 8,400 on db.r6g.4xlarge at 2.14";
        let toks = spans(
            text,
            &[
                ("cost", 0),
                ("8", 1),
                (",", 2),
                ("400", 3),
                ("on", 4),
                ("db", 5),
                (".", 6),
                ("r6", 7),
                ("g", 7),
                (".", 8),
                ("4", 9),
                ("xl", 9),
                ("arge", 9),
                ("at", 10),
                ("2", 11),
                (".", 12),
                ("14", 13),
            ],
        );
        assert_eq!(
            atom_texts(text, &atoms_from_tokens(text, &toks)),
            ["cost", "8,400", "on", "db.r6g.4xlarge", "at", "2.14"]
        );
    }

    #[test]
    fn subwords_of_one_word_join_even_without_byte_adjacency_ambiguity() {
        let text = "pg_upgrade failed";
        let toks = spans(text, &[("pg", 0), ("_", 1), ("upgrade", 2), ("failed", 3)]);
        assert_eq!(
            atom_texts(text, &atoms_from_tokens(text, &toks)),
            ["pg_upgrade", "failed"]
        );
    }

    #[test]
    fn atoms_do_not_glue_across_cjk() {
        // Every ideograph is its own word with no separating bytes; gluing on
        // byte-adjacency alone would make the whole run one unsplittable atom.
        let text = "会議の記録";
        let toks = spans(
            text,
            &[("会", 0), ("議", 1), ("の", 2), ("記", 3), ("録", 4)],
        );
        assert_eq!(
            atom_texts(text, &atoms_from_tokens(text, &toks)),
            ["会", "議", "の", "記", "録"]
        );
    }

    #[test]
    fn multi_token_cjk_words_still_join_within_a_word() {
        // Same word index means one word, CJK or not.
        let text = "東京都";
        let toks = spans(text, &[("東", 0), ("京", 0), ("都", 0)]);
        assert_eq!(
            atom_texts(text, &atoms_from_tokens(text, &toks)),
            ["東京都"]
        );
    }

    #[test]
    fn sentences_break_on_terminal_punctuation_and_blank_lines() {
        let text = "one two. three\n\nfour";
        let toks = spans(
            text,
            &[("one", 0), ("two", 1), (".", 2), ("three", 3), ("four", 4)],
        );
        let atoms = atoms_from_tokens(text, &toks);
        assert_eq!(
            atoms.iter().map(|a| a.sentence).collect::<Vec<_>>(),
            [0, 0, 1, 2]
        );
    }

    #[test]
    fn a_decimal_never_breaks_a_sentence() {
        // `2.14` is one atom, so its period is interior and invisible to the
        // sentence rule — no abbreviation list needed for this case.
        let text = "upgrade to 2.14 first";
        let toks = spans(
            text,
            &[
                ("upgrade", 0),
                ("to", 1),
                ("2", 2),
                (".", 3),
                ("14", 4),
                ("first", 5),
            ],
        );
        let atoms = atoms_from_tokens(text, &toks);
        assert!(
            atoms.iter().all(|a| a.sentence == 0),
            "got {:?}",
            atoms.iter().map(|a| a.sentence).collect::<Vec<_>>()
        );
    }

    #[test]
    fn a_terminator_with_no_following_whitespace_is_not_a_break() {
        let text = "see http://a.b/c?x=1 now";
        let toks = spans(
            text,
            &[("see", 0), ("http", 1), ("://a.b/c?x=1", 2), ("now", 3)],
        );
        let atoms = atoms_from_tokens(text, &toks);
        assert_eq!(
            atoms.iter().map(|a| a.sentence).collect::<Vec<_>>(),
            [0, 0, 0]
        );
    }

    #[test]
    fn importance_is_the_mean_over_an_atoms_subwords() {
        let text = "doesn't work";
        let toks = spans(text, &[("doesn", 0), ("'", 1), ("t", 2), ("work", 3)]);
        let mut atoms = atoms_from_tokens(text, &toks);
        apply_probs(&mut atoms, &[0.9, 0.3, 0.6, 0.5]);
        assert!((atoms[0].importance - 0.6).abs() < 1e-6, "{:?}", atoms[0]);
        assert!((atoms[1].importance - 0.5).abs() < 1e-6);
        assert_eq!(atoms[0].token_count(), 3);
    }

    // ---- chunk planning -------------------------------------------------

    fn linear_atoms(n_atoms: usize, tokens_per_atom: usize, sentence_every: usize) -> Vec<Atom> {
        (0..n_atoms)
            .map(|i| Atom {
                start: i * 10,
                end: i * 10 + 5,
                tokens: (i * tokens_per_atom)..((i + 1) * tokens_per_atom),
                importance: 0.0,
                protected: false,
                sentence: (i / sentence_every) as u32,
            })
            .collect()
    }

    /// The invariant that makes the prototype's off-by-one truncation bug
    /// impossible: chunks exactly partition the stream.
    fn assert_partitions(chunks: &[Range<usize>], total: usize, max_body: usize) {
        if total == 0 {
            assert!(chunks.is_empty());
            return;
        }
        assert_eq!(chunks.first().unwrap().start, 0, "must start at 0");
        assert_eq!(chunks.last().unwrap().end, total, "must reach the end");
        for w in chunks.windows(2) {
            assert_eq!(w[0].end, w[1].start, "contiguous, no gap or overlap");
        }
        for c in chunks {
            assert!(!c.is_empty(), "no empty chunk");
            assert!(c.len() <= max_body, "chunk {c:?} exceeds {max_body}");
        }
        assert_eq!(chunks.iter().map(|c| c.len()).sum::<usize>(), total);
    }

    #[test]
    fn chunks_partition_the_token_stream_exactly() {
        for tokens_per_atom in [1, 3, 7] {
            for n_atoms in [1, 2, 5, 17, 64, 200] {
                for max_body in [1, 4, 10, 64, 510] {
                    let atoms = linear_atoms(n_atoms, tokens_per_atom, 4);
                    let total = n_atoms * tokens_per_atom;
                    let chunks = plan_chunks(total, &atoms, max_body);
                    assert_partitions(&chunks, total, max_body);
                }
            }
        }
    }

    #[test]
    fn an_empty_stream_plans_no_chunks() {
        assert!(plan_chunks(0, &[], 510).is_empty());
    }

    #[test]
    fn a_stream_within_the_limit_is_one_chunk() {
        let atoms = linear_atoms(10, 3, 4);
        assert_eq!(plan_chunks(30, &atoms, 510), vec![0..30]);
    }

    #[test]
    fn chunks_prefer_sentence_boundaries() {
        // 12 atoms x 2 tokens = 24; sentences every 3 atoms (every 6 tokens).
        let atoms = linear_atoms(12, 2, 3);
        let chunks = plan_chunks(24, &atoms, 10);
        assert_partitions(&chunks, 24, 10);
        // The first cut lands on a sentence start (6 or 12), never mid-sentence.
        assert!(
            chunks[0].end.is_multiple_of(6),
            "expected a sentence boundary, got {:?}",
            chunks[0]
        );
    }

    #[test]
    fn chunks_fall_back_to_atom_boundaries_inside_a_long_sentence() {
        // One sentence spanning everything: no sentence boundary is available.
        let atoms = linear_atoms(20, 2, 1_000);
        let chunks = plan_chunks(40, &atoms, 9);
        assert_partitions(&chunks, 40, 9);
        // Every cut is an atom boundary (even index), never mid-atom.
        assert!(
            chunks.iter().all(|c| c.start.is_multiple_of(2)),
            "cut inside an atom: {chunks:?}"
        );
    }

    #[test]
    fn an_atom_longer_than_a_chunk_is_split_rather_than_dropped() {
        // A base64 blob with no whitespace. Splitting it across forward passes is
        // harmless — it is still selected as one unit — whereas erroring or
        // truncating would lose the caller's text.
        let atoms = vec![Atom {
            start: 0,
            end: 4_000,
            tokens: 0..1_000,
            importance: 0.0,
            protected: false,
            sentence: 0,
        }];
        let chunks = plan_chunks(1_000, &atoms, 510);
        assert_partitions(&chunks, 1_000, 510);
        assert_eq!(chunks, vec![0..510, 510..1_000]);
    }
}
