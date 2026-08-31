//! Build-time embedding artifact: corpus vectors serialized for runtime load
//! without re-inferring.
//!
//! Registries build single-kind bytes; [`merge_embedding_artifacts`] combines
//! them into one mixed Tool+Skill RAT1. Warming ignores other known kinds.
//! Model identity in the header is the artifact-scoped identity from
//! [`Embedder::embed_batch_with_artifact_identity`] (Endpoint keeps batch-resolved
//! response identity; Local uses a portable content digest).

use sha2::{Digest, Sha256};

use crate::dense_cache::Embeddable;
use crate::embedding::{Embedded, Embedder, EmbedderError};

const MAGIC: &[u8; 4] = b"RAT1";
const SUPPORTED_FORMAT_VERSION: u32 = 1;
const VERSION_PREFIX_LEN: usize = 8;
const FILE_PREFIX_LEN: usize = 4 + 4 + 8 + 32;

/// Catalog origin of an artifact entry (tool vs skill registry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ArtifactEntryKind {
    /// Entry built from a tool catalog item.
    Tool,
    /// Entry built from a skill catalog item.
    Skill,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ArtifactHeader {
    pub format_version: u32,
    /// Hash-derived version of searchable-text projection logic at build time
    pub projection_version: u32,
    /// Resolved embedder identity stamped on every vector in the artifact
    pub model_fingerprint: String,
    /// Common width of every vector in the artifact
    pub dim: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ArtifactEntry {
    pub kind: ArtifactEntryKind,
    /// Stable catalog id ([`Embeddable::embed_id`])
    pub id: String,
    /// SHA-256 of the projection text at build time, the text itself is not stored
    pub projection_hash: [u8; 32],
    /// L2-normalized embedding of the item's projection text
    pub vector: Vec<f32>,
}

/// Failure building or loading a binary embedding artifact.
#[derive(Debug, Clone)]
pub enum ArtifactError {
    /// File shorter than the format prefix or declared payload length.
    TooShort {
        /// Minimum bytes required at this check.
        needed: usize,
        /// Bytes actually available.
        got: usize,
    },
    /// Magic bytes were not `RAT1`.
    InvalidMagic {
        /// Four bytes read where the magic should be.
        got: [u8; 4],
    },
    /// `format_version` is newer than this binary supports.
    UnsupportedFormatVersion {
        /// Version stamped in the file.
        found: u32,
        /// Highest version this binary can read.
        supported: u32,
    },
    /// Payload SHA-256 did not match the checksum in the file prefix.
    ChecksumMismatch,
    /// Payload bytes failed structural decode after a valid checksum.
    CorruptPayload {
        /// Byte offset into the payload where decode failed.
        at: usize,
        /// Human-readable reason for the failure.
        detail: String,
    },
    /// Embedder returned vectors of mixed widths in one build batch.
    InconsistentVectorWidth {
        /// Width of the first vector in the batch.
        expected: usize,
        /// Width of the diverging vector.
        got: usize,
    },
    /// Embedder returned a vector that is not L2-normalized.
    VectorNotNormalized {
        /// Catalog id of the item whose vector failed the unit-norm check.
        id: String,
    },
    /// Non-empty artifact declared `dim == 0` (invalid embedding space).
    NonEmptyZeroDim,
    /// A vector failed semantic checks (non-finite or not unit-normalized).
    /// Raised for embedder output during [`build_artifact`] and for vectors
    /// decoded from checksum-valid RAT1. Distinct from
    /// [`Self::VectorNotNormalized`], which is the build-time failure for
    /// finite non-unit embedder vectors.
    InvalidVector {
        /// Catalog id of the failing entry.
        id: String,
        /// Short deterministic reason (`non-finite component`, `not unit-normalized`).
        detail: String,
    },
    /// Valid artifacts that cannot be combined (header mismatch or duplicate
    /// `(kind, id)`). Not [`Self::CorruptPayload`]: each input may decode alone.
    IncompatibleMerge {
        /// Why the parts cannot be combined.
        detail: String,
    },
    /// Underlying embedder failure during [`build_artifact`].
    Embedder(EmbedderError),
}

impl ArtifactError {
    fn hint(&self) -> &'static str {
        match self {
            ArtifactError::TooShort { .. } => {
                "the artifact file is truncated or payload_len exceeds the file size"
            }
            ArtifactError::InvalidMagic { .. } => {
                "verify the file is a Ratel embedding artifact (magic RAT1)"
            }
            ArtifactError::UnsupportedFormatVersion { .. } => {
                "rebuild the artifact with a compatible Ratel version"
            }
            ArtifactError::ChecksumMismatch => {
                "the artifact is corrupt or was modified; rebuild from source"
            }
            ArtifactError::CorruptPayload { .. } => {
                "the artifact payload is malformed; rebuild from source"
            }
            ArtifactError::InconsistentVectorWidth { .. } => {
                "the embedder returned vectors of mixed widths; fix the model or corpus"
            }
            ArtifactError::VectorNotNormalized { .. } => {
                "the embedder must return L2-normalized vectors before building an artifact"
            }
            ArtifactError::NonEmptyZeroDim => {
                "a non-empty embedding artifact must declare a positive vector dimension"
            }
            ArtifactError::InvalidVector { .. } => {
                "vectors must be finite and unit-normalized; fix the embedder output or rebuild the artifact"
            }
            ArtifactError::IncompatibleMerge { .. } => {
                "rebuild each part with the same model and projection, or drop the conflicting entry"
            }
            ArtifactError::Embedder(_) => {
                "fix the embedding model or corpus before building the artifact"
            }
        }
    }
}

impl std::fmt::Display for ArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let hint = self.hint();
        match self {
            ArtifactError::TooShort { needed, got } => write!(
                f,
                "embedding artifact too short: need at least {needed} bytes, got {got} (hint: {hint})"
            ),
            ArtifactError::InvalidMagic { got } => write!(
                f,
                "embedding artifact invalid magic: expected RAT1, got {} (hint: {hint})",
                std::str::from_utf8(got).unwrap_or("<non-utf8>")
            ),
            ArtifactError::UnsupportedFormatVersion { found, supported } => write!(
                f,
                "embedding artifact format version {found} is unsupported (supported: {supported}) (hint: {hint})"
            ),
            ArtifactError::ChecksumMismatch => {
                write!(f, "embedding artifact checksum mismatch (hint: {hint})")
            }
            ArtifactError::CorruptPayload { at, detail } => write!(
                f,
                "embedding artifact payload corrupt at offset {at}: {detail} (hint: {hint})"
            ),
            ArtifactError::InconsistentVectorWidth { expected, got } => write!(
                f,
                "embedding artifact build: vector width {got} != expected {expected} (hint: {hint})"
            ),
            ArtifactError::VectorNotNormalized { id } => write!(
                f,
                "embedding artifact build: vector for id {id:?} is not L2-normalized (hint: {hint})"
            ),
            ArtifactError::NonEmptyZeroDim => write!(
                f,
                "embedding artifact has entries but dim is 0 (hint: {hint})"
            ),
            ArtifactError::InvalidVector { id, detail } => write!(
                f,
                "embedding artifact vector for id {id:?} is invalid: {detail} (hint: {hint})"
            ),
            ArtifactError::IncompatibleMerge { detail } => write!(
                f,
                "embedding artifact merge incompatible: {detail} (hint: {hint})"
            ),
            ArtifactError::Embedder(e) => write!(f, "embedding artifact build failed: {e}"),
        }
    }
}

impl std::error::Error for ArtifactError {}

impl From<EmbedderError> for ArtifactError {
    fn from(value: EmbedderError) -> Self {
        Self::Embedder(value)
    }
}

/// Hash of every catalog projection source — bumps when projection logic
/// changes without a manual version constant.
pub(crate) fn projection_version() -> u32 {
    let mut h = Sha256::new();
    h.update(include_str!("indexing.rs"));
    h.update(include_str!("skill_indexing.rs"));
    h.update(include_str!("fact_indexing.rs"));
    u32::from_le_bytes(h.finalize()[..4].try_into().expect("4 bytes"))
}

/// A valid RAT1 artifact with zero entries (no embedder required)
pub(crate) fn build_empty_artifact() -> Result<Vec<u8>, ArtifactError> {
    let header = ArtifactHeader {
        format_version: SUPPORTED_FORMAT_VERSION,
        projection_version: projection_version(),
        model_fingerprint: String::new(),
        dim: 0,
    };
    let payload = encode_payload(&header, &[])?;
    Ok(assemble_file(&payload))
}

pub(crate) fn build_artifact<'a, T: Embeddable + 'a>(
    kind: ArtifactEntryKind,
    items: impl IntoIterator<Item = &'a T>,
    embedder: &dyn Embedder,
) -> Result<Vec<u8>, ArtifactError> {
    let rows: Vec<(String, String)> = items
        .into_iter()
        .map(|item| (item.embed_id().to_string(), item.embed_text()))
        .collect();

    if rows.is_empty() {
        return build_empty_artifact();
    }

    let texts: Vec<String> = rows.iter().map(|(_, text)| text.clone()).collect();
    let Embedded {
        value: vectors,
        fingerprint: model_fingerprint,
    } = embedder.embed_batch_with_artifact_identity(&texts)?;

    if vectors.len() != rows.len() {
        return Err(ArtifactError::Embedder(EmbedderError::Inference {
            source: format!(
                "embedder returned {} embeddings for {} inputs",
                vectors.len(),
                rows.len()
            ),
        }));
    }

    let dim = vectors.first().map(Vec::len).unwrap_or(0);
    require_positive_dim_when_nonempty(rows.len(), dim)?;
    for ((id, _), vector) in rows.iter().zip(&vectors) {
        if vector.len() != dim {
            return Err(ArtifactError::InconsistentVectorWidth {
                expected: dim,
                got: vector.len(),
            });
        }
        match classify_vector_semantics(vector) {
            Ok(()) => {}
            Err(VectorSemanticIssue::NonFinite) => {
                return Err(ArtifactError::InvalidVector {
                    id: id.clone(),
                    detail: "non-finite component".into(),
                });
            }
            Err(VectorSemanticIssue::NotUnitNormalized) => {
                return Err(ArtifactError::VectorNotNormalized { id: id.clone() });
            }
        }
    }

    let entries: Vec<ArtifactEntry> = rows
        .into_iter()
        .zip(vectors)
        .map(|((id, text), vector)| ArtifactEntry {
            kind,
            id,
            projection_hash: hash_projection_text(&text),
            vector,
        })
        .collect();

    let header = ArtifactHeader {
        format_version: SUPPORTED_FORMAT_VERSION,
        projection_version: projection_version(),
        model_fingerprint,
        dim,
    };

    let payload = encode_payload(&header, &entries)?;
    Ok(assemble_file(&payload))
}

pub(crate) fn load_and_validate(
    bytes: &[u8],
) -> Result<(ArtifactHeader, Vec<ArtifactEntry>), ArtifactError> {
    if bytes.len() < VERSION_PREFIX_LEN {
        return Err(ArtifactError::TooShort {
            needed: VERSION_PREFIX_LEN,
            got: bytes.len(),
        });
    }

    let mut magic = [0u8; 4];
    magic.copy_from_slice(&bytes[..4]);
    if &magic != MAGIC {
        return Err(ArtifactError::InvalidMagic { got: magic });
    }

    let format_version = u32::from_le_bytes(bytes[4..8].try_into().expect("4 bytes"));
    if format_version != SUPPORTED_FORMAT_VERSION {
        return Err(ArtifactError::UnsupportedFormatVersion {
            found: format_version,
            supported: SUPPORTED_FORMAT_VERSION,
        });
    }

    if bytes.len() < FILE_PREFIX_LEN {
        return Err(ArtifactError::TooShort {
            needed: FILE_PREFIX_LEN,
            got: bytes.len(),
        });
    }

    let payload_len = u64::from_le_bytes(bytes[8..16].try_into().expect("8 bytes")) as usize;
    let declared_checksum: [u8; 32] = bytes[16..48].try_into().expect("32 bytes");

    let needed = FILE_PREFIX_LEN
        .checked_add(payload_len)
        .ok_or(ArtifactError::CorruptPayload {
            at: 8,
            detail: "payload_len overflow".into(),
        })?;
    if bytes.len() < needed {
        return Err(ArtifactError::TooShort {
            needed,
            got: bytes.len(),
        });
    }
    if bytes.len() > needed {
        return Err(ArtifactError::CorruptPayload {
            at: payload_len,
            detail: format!("{} trailing bytes after artifact", bytes.len() - needed),
        });
    }

    let payload = &bytes[FILE_PREFIX_LEN..needed];
    let computed = Sha256::digest(payload);
    if computed.as_slice() != declared_checksum {
        return Err(ArtifactError::ChecksumMismatch);
    }

    decode_payload(format_version, payload)
}

/// Merge valid RAT1 parts into one artifact. Empty parts are skipped; nonempty
/// parts must share format/projection version, fingerprint, and dim. Duplicate
/// `(kind, id)` → [`ArtifactError::IncompatibleMerge`].
pub fn merge_embedding_artifacts(parts: &[&[u8]]) -> Result<Vec<u8>, ArtifactError> {
    let mut base_header: Option<ArtifactHeader> = None;
    let mut merged: Vec<ArtifactEntry> = Vec::new();
    let mut seen: std::collections::HashSet<(ArtifactEntryKind, String)> =
        std::collections::HashSet::new();

    for part in parts {
        let (header, entries) = load_and_validate(part)?;
        if entries.is_empty() {
            continue;
        }
        match &base_header {
            None => base_header = Some(header.clone()),
            Some(base) => {
                if header.format_version != base.format_version {
                    return Err(ArtifactError::IncompatibleMerge {
                        detail: format!(
                            "format_version {} != {}",
                            header.format_version, base.format_version
                        ),
                    });
                }
                if header.projection_version != base.projection_version {
                    return Err(ArtifactError::IncompatibleMerge {
                        detail: format!(
                            "projection_version {} != {}",
                            header.projection_version, base.projection_version
                        ),
                    });
                }
                if header.model_fingerprint != base.model_fingerprint {
                    return Err(ArtifactError::IncompatibleMerge {
                        detail: format!(
                            "model_fingerprint {:?} != {:?}",
                            header.model_fingerprint, base.model_fingerprint
                        ),
                    });
                }
                if header.dim != base.dim {
                    return Err(ArtifactError::IncompatibleMerge {
                        detail: format!("dim {} != {}", header.dim, base.dim),
                    });
                }
            }
        }
        for entry in entries {
            let key = (entry.kind, entry.id.clone());
            if !seen.insert(key) {
                return Err(ArtifactError::IncompatibleMerge {
                    detail: format!("duplicate entry kind={:?} id={:?}", entry.kind, entry.id),
                });
            }
            merged.push(entry);
        }
    }

    let Some(header) = base_header else {
        return build_empty_artifact();
    };
    let payload = encode_payload(&header, &merged)?;
    Ok(assemble_file(&payload))
}

pub(crate) fn hash_projection_text(text: &str) -> [u8; 32] {
    Sha256::digest(text.as_bytes()).into()
}

/// Shared squared-L2 unit tolerance used by build and decode.
const UNIT_NORM_SQ_TOLERANCE: f32 = 1e-4;

fn unit_norm_sq_ok(norm_sq: f32) -> bool {
    (norm_sq - 1.0).abs() <= UNIT_NORM_SQ_TOLERANCE
}

fn is_unit_normalized(vector: &[f32]) -> bool {
    let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
    unit_norm_sq_ok(norm_sq)
}

fn require_positive_dim_when_nonempty(entry_count: usize, dim: usize) -> Result<(), ArtifactError> {
    if entry_count > 0 && dim == 0 {
        return Err(ArtifactError::NonEmptyZeroDim);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorSemanticIssue {
    NonFinite,
    NotUnitNormalized,
}

fn classify_vector_semantics(vector: &[f32]) -> Result<(), VectorSemanticIssue> {
    if vector.iter().any(|x| !x.is_finite()) {
        return Err(VectorSemanticIssue::NonFinite);
    }
    if !is_unit_normalized(vector) {
        return Err(VectorSemanticIssue::NotUnitNormalized);
    }
    Ok(())
}

fn assemble_file(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(FILE_PREFIX_LEN + payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&SUPPORTED_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&Sha256::digest(payload));
    out.extend_from_slice(payload);
    out
}

/// Test-only: assemble a checksum-valid RAT1 from arbitrary entries (may be
/// semantically invalid). Bypasses build-time semantic checks so load/warm/merge
/// regressions can target the decode boundary.
#[cfg(test)]
pub(crate) fn test_hand_artifact(
    projection_version: u32,
    dim: usize,
    fingerprint: &str,
    entries: &[ArtifactEntry],
) -> Vec<u8> {
    let header = ArtifactHeader {
        format_version: SUPPORTED_FORMAT_VERSION,
        projection_version,
        model_fingerprint: fingerprint.into(),
        dim,
    };
    let payload = encode_payload(&header, entries).expect("test hand artifact encode");
    assemble_file(&payload)
}

fn encode_payload(
    header: &ArtifactHeader,
    entries: &[ArtifactEntry],
) -> Result<Vec<u8>, ArtifactError> {
    let mut out = Vec::new();
    write_u32(&mut out, header.projection_version);
    write_u32(
        &mut out,
        header
            .dim
            .try_into()
            .map_err(|_| ArtifactError::CorruptPayload {
                at: 0,
                detail: "dim does not fit u32".into(),
            })?,
    );
    let model_fp_at = out.len();
    write_utf8(&mut out, &header.model_fingerprint, model_fp_at)?;
    let entry_count_at = out.len();
    write_u32(
        &mut out,
        entries
            .len()
            .try_into()
            .map_err(|_| ArtifactError::CorruptPayload {
                at: entry_count_at,
                detail: "entry_count does not fit u32".into(),
            })?,
    );

    for entry in entries {
        write_kind(&mut out, entry.kind);
        let id_at = out.len();
        write_utf8(&mut out, &entry.id, id_at)?;
        out.extend_from_slice(&entry.projection_hash);
        if entry.vector.len() != header.dim {
            return Err(ArtifactError::CorruptPayload {
                at: out.len(),
                detail: format!(
                    "entry {} vector width {} != header dim {}",
                    entry.id,
                    entry.vector.len(),
                    header.dim
                ),
            });
        }
        for &value in &entry.vector {
            out.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(out)
}

fn decode_payload(
    format_version: u32,
    payload: &[u8],
) -> Result<(ArtifactHeader, Vec<ArtifactEntry>), ArtifactError> {
    let mut cursor = 0usize;
    let projection_version = read_u32(payload, &mut cursor)?;
    let dim = read_u32(payload, &mut cursor)? as usize;
    let model_fingerprint = read_utf8(payload, &mut cursor)?;
    let entry_count = read_u32(payload, &mut cursor)? as usize;

    // Reject absurd dim/entry_count before Vec::with_capacity: each entry needs at
    // least kind(1) + id_len(4) + hash(32) + dim*f32 (empty id). Checked math only —
    // no arbitrary global size caps.
    let remaining = payload.len().saturating_sub(cursor);
    let vector_bytes = dim.checked_mul(4).ok_or(ArtifactError::CorruptPayload {
        at: cursor,
        detail: "dim*4 overflow".into(),
    })?;
    let min_entry =
        (1usize + 4 + 32)
            .checked_add(vector_bytes)
            .ok_or(ArtifactError::CorruptPayload {
                at: cursor,
                detail: "min entry size overflow".into(),
            })?;
    let min_total = entry_count
        .checked_mul(min_entry)
        .ok_or(ArtifactError::CorruptPayload {
            at: cursor,
            detail: "entry_count*min_entry overflow".into(),
        })?;
    if min_total > remaining {
        return Err(ArtifactError::CorruptPayload {
            at: cursor,
            detail: format!("entries need at least {min_total} bytes but only {remaining} remain"),
        });
    }

    require_positive_dim_when_nonempty(entry_count, dim)?;

    let mut entries = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let kind = read_kind(payload, &mut cursor)?;
        let id = read_utf8(payload, &mut cursor)?;
        let projection_hash = read_fixed::<32>(payload, &mut cursor)?;
        let mut vector = Vec::with_capacity(dim);
        let mut norm_sq = 0.0f32;
        for _ in 0..dim {
            let value = read_f32(payload, &mut cursor)?;
            if !value.is_finite() {
                return Err(ArtifactError::InvalidVector {
                    id,
                    detail: "non-finite component".into(),
                });
            }
            norm_sq += value * value;
            vector.push(value);
        }
        if !unit_norm_sq_ok(norm_sq) {
            return Err(ArtifactError::InvalidVector {
                id,
                detail: "not unit-normalized".into(),
            });
        }
        entries.push(ArtifactEntry {
            kind,
            id,
            projection_hash,
            vector,
        });
    }

    if cursor != payload.len() {
        return Err(ArtifactError::CorruptPayload {
            at: cursor,
            detail: format!("{} trailing bytes after entries", payload.len() - cursor),
        });
    }

    Ok((
        ArtifactHeader {
            format_version,
            projection_version,
            model_fingerprint,
            dim,
        },
        entries,
    ))
}

fn write_kind(out: &mut Vec<u8>, kind: ArtifactEntryKind) {
    out.push(match kind {
        ArtifactEntryKind::Tool => 0,
        ArtifactEntryKind::Skill => 1,
    });
}

fn read_kind(payload: &[u8], cursor: &mut usize) -> Result<ArtifactEntryKind, ArtifactError> {
    let byte = read_byte(payload, cursor)?;
    match byte {
        0 => Ok(ArtifactEntryKind::Tool),
        1 => Ok(ArtifactEntryKind::Skill),
        other => Err(ArtifactError::CorruptPayload {
            at: cursor.saturating_sub(1),
            detail: format!("unknown entry kind {other}"),
        }),
    }
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn write_utf8(out: &mut Vec<u8>, s: &str, at: usize) -> Result<(), ArtifactError> {
    let bytes = s.as_bytes();
    let len = u32::try_from(bytes.len()).map_err(|_| ArtifactError::CorruptPayload {
        at,
        detail: format!("utf8 field length {} does not fit u32", bytes.len()),
    })?;
    write_u32(out, len);
    out.extend_from_slice(bytes);
    Ok(())
}

fn read_u32(payload: &[u8], cursor: &mut usize) -> Result<u32, ArtifactError> {
    let at = *cursor;
    let end = at.checked_add(4).ok_or(ArtifactError::CorruptPayload {
        at,
        detail: "u32 read overflow".into(),
    })?;
    if payload.len() < end {
        return Err(ArtifactError::CorruptPayload {
            at,
            detail: "unexpected end of payload reading u32".into(),
        });
    }
    let value = u32::from_le_bytes(payload[at..end].try_into().expect("4 bytes"));
    *cursor = end;
    Ok(value)
}

fn read_f32(payload: &[u8], cursor: &mut usize) -> Result<f32, ArtifactError> {
    let at = *cursor;
    let end = at.checked_add(4).ok_or(ArtifactError::CorruptPayload {
        at,
        detail: "f32 read overflow".into(),
    })?;
    if payload.len() < end {
        return Err(ArtifactError::CorruptPayload {
            at,
            detail: "unexpected end of payload reading f32".into(),
        });
    }
    let value = f32::from_le_bytes(payload[at..end].try_into().expect("4 bytes"));
    *cursor = end;
    Ok(value)
}

fn read_byte(payload: &[u8], cursor: &mut usize) -> Result<u8, ArtifactError> {
    let at = *cursor;
    if at >= payload.len() {
        return Err(ArtifactError::CorruptPayload {
            at,
            detail: "unexpected end of payload reading byte".into(),
        });
    }
    *cursor = at + 1;
    Ok(payload[at])
}

fn read_fixed<const N: usize>(
    payload: &[u8],
    cursor: &mut usize,
) -> Result<[u8; N], ArtifactError> {
    let at = *cursor;
    let end = at.checked_add(N).ok_or(ArtifactError::CorruptPayload {
        at,
        detail: format!("fixed-{N} read overflow"),
    })?;
    if payload.len() < end {
        return Err(ArtifactError::CorruptPayload {
            at,
            detail: format!("unexpected end of payload reading fixed-{N}"),
        });
    }
    let mut out = [0u8; N];
    out.copy_from_slice(&payload[at..end]);
    *cursor = end;
    Ok(out)
}

fn read_utf8(payload: &[u8], cursor: &mut usize) -> Result<String, ArtifactError> {
    let len = read_u32(payload, cursor)? as usize;
    let start = *cursor;
    let end = start
        .checked_add(len)
        .ok_or(ArtifactError::CorruptPayload {
            at: start,
            detail: "utf8 length overflow".into(),
        })?;
    if payload.len() < end {
        return Err(ArtifactError::CorruptPayload {
            at: start,
            detail: format!("utf8 field claims {len} bytes but payload ends early"),
        });
    }
    let s =
        std::str::from_utf8(&payload[start..end]).map_err(|e| ArtifactError::CorruptPayload {
            at: start,
            detail: format!("invalid utf8: {e}"),
        })?;
    *cursor = end;
    Ok(s.to_string())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::embedding::Embedded;

    struct StubItem {
        id: String,
        text: String,
    }

    impl Embeddable for StubItem {
        fn embed_id(&self) -> &str {
            &self.id
        }
        fn embed_text(&self) -> String {
            self.text.clone()
        }
    }

    struct StubEmbedder {
        fingerprint: String,
        vectors: Vec<Vec<f32>>,
        /// When true, return `vectors` as-is even if len ≠ texts.len().
        passthrough_batch: bool,
    }

    impl Embedder for StubEmbedder {
        fn embed_doc(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            unreachable!("artifact tests use batch path")
        }

        fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            unreachable!("artifact tests use batch path")
        }

        fn embed_batch_with_identity(
            &self,
            texts: &[String],
        ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
            if texts.is_empty() {
                return Ok(Embedded {
                    value: Vec::new(),
                    fingerprint: self.fingerprint.clone(),
                });
            }
            let value = if self.passthrough_batch || self.vectors.len() == texts.len() {
                self.vectors.clone()
            } else {
                let template = &self.vectors[0];
                texts.iter().map(|_| template.clone()).collect()
            };
            Ok(Embedded {
                value,
                fingerprint: self.fingerprint.clone(),
            })
        }

        fn fingerprint(&self) -> String {
            self.fingerprint.clone()
        }
    }

    /// Endpoint-like stub: static `fingerprint()` differs from the batch-resolved
    /// identity returned by `embed_batch_with_identity`.
    struct BatchResolvedStub {
        static_fingerprint: String,
        batch_fingerprint: String,
        vectors: Vec<Vec<f32>>,
    }

    impl Embedder for BatchResolvedStub {
        fn embed_doc(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            unreachable!("artifact tests use batch path")
        }
        fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            unreachable!("artifact tests use batch path")
        }
        fn embed_batch_with_identity(
            &self,
            texts: &[String],
        ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
            assert_eq!(texts.len(), self.vectors.len());
            Ok(Embedded {
                value: self.vectors.clone(),
                fingerprint: self.batch_fingerprint.clone(),
            })
        }
        fn fingerprint(&self) -> String {
            self.static_fingerprint.clone()
        }
    }

    /// Local-like stub: runtime / `embed_batch_with_identity` differ from
    /// `embed_batch_with_artifact_identity` — proves build uses the artifact path.
    struct ArtifactAwareBatchStub {
        runtime: String,
        artifact: String,
        vectors: Vec<Vec<f32>>,
    }

    impl Embedder for ArtifactAwareBatchStub {
        fn embed_doc(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            unreachable!("artifact tests use batch path")
        }
        fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedderError> {
            unreachable!("artifact tests use batch path")
        }
        fn embed_batch_with_identity(
            &self,
            texts: &[String],
        ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
            assert_eq!(texts.len(), self.vectors.len());
            Ok(Embedded {
                value: self.vectors.clone(),
                fingerprint: self.runtime.clone(),
            })
        }
        fn embed_batch_with_artifact_identity(
            &self,
            texts: &[String],
        ) -> Result<Embedded<Vec<Vec<f32>>>, EmbedderError> {
            assert_eq!(texts.len(), self.vectors.len());
            Ok(Embedded {
                value: self.vectors.clone(),
                fingerprint: self.artifact.clone(),
            })
        }
        fn fingerprint(&self) -> String {
            self.runtime.clone()
        }
        fn artifact_identity(&self) -> Result<String, EmbedderError> {
            Ok(self.artifact.clone())
        }
    }

    fn stub_item(id: &str, text: &str) -> StubItem {
        StubItem {
            id: id.into(),
            text: text.into(),
        }
    }

    fn unit(v: [f32; 2]) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    fn unit3(v: [f32; 3]) -> Vec<f32> {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / norm).collect()
    }

    fn sample_embedder_for(items: &[StubItem]) -> Arc<StubEmbedder> {
        Arc::new(StubEmbedder {
            fingerprint: "hf|repo=1:r|revision=4:main|pool=3:cls".into(),
            vectors: items
                .iter()
                .enumerate()
                .map(|(i, _)| {
                    if i % 2 == 0 {
                        unit([1.0, 0.0])
                    } else {
                        unit([0.0, 1.0])
                    }
                })
                .collect(),
            passthrough_batch: false,
        })
    }

    fn hand_artifact(
        projection_version: u32,
        dim: usize,
        fingerprint: &str,
        entries: &[ArtifactEntry],
    ) -> Vec<u8> {
        test_hand_artifact(projection_version, dim, fingerprint, entries)
    }

    #[test]
    fn build_followed_by_load_round_trips_header_and_entries() {
        let items = [
            stub_item("read_file", "read file from disk"),
            stub_item("write_file", "write file to disk"),
        ];
        let bytes = build_artifact(
            ArtifactEntryKind::Tool,
            &items,
            sample_embedder_for(&items).as_ref(),
        )
        .unwrap();
        let (header, entries) = load_and_validate(&bytes).unwrap();

        assert_eq!(header.format_version, SUPPORTED_FORMAT_VERSION);
        assert_eq!(header.projection_version, projection_version());
        assert_eq!(
            header.model_fingerprint,
            "hf|repo=1:r|revision=4:main|pool=3:cls"
        );
        assert_eq!(header.dim, 2);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].kind, ArtifactEntryKind::Tool);
        assert_eq!(entries[0].id, "read_file");
        assert_eq!(
            entries[0].projection_hash,
            hash_projection_text("read file from disk")
        );
        assert_eq!(entries[1].id, "write_file");
    }

    #[test]
    fn build_uses_batch_identity_for_endpoint_semantics() {
        let items = [stub_item("a", "alpha")];
        let embedder = BatchResolvedStub {
            static_fingerprint: "endpoint|url=1:u|model=9:configured".into(),
            batch_fingerprint: "endpoint|url=1:u|model=8:resolved".into(),
            vectors: vec![unit([1.0, 0.0])],
        };
        let bytes = build_artifact(ArtifactEntryKind::Tool, &items, &embedder).unwrap();
        let (header, _) = load_and_validate(&bytes).unwrap();
        assert_eq!(
            header.model_fingerprint, "endpoint|url=1:u|model=8:resolved",
            "RAT1 header must use the batch-resolved identity, not static fingerprint()"
        );
    }

    #[test]
    fn build_artifact_uses_artifact_aware_batch_identity() {
        let items = [stub_item("a", "alpha")];
        let embedder = ArtifactAwareBatchStub {
            runtime: "local|path=11:/models/foo|pool=4:mean".into(),
            artifact: "local|content=64:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb|pool=4:mean"
                .into(),
            vectors: vec![unit([1.0, 0.0])],
        };
        let bytes = build_artifact(ArtifactEntryKind::Tool, &items, &embedder).unwrap();
        let (header, _) = load_and_validate(&bytes).unwrap();
        assert_eq!(
            header.model_fingerprint, embedder.artifact,
            "build_artifact must call embed_batch_with_artifact_identity, not only embed_batch_with_identity"
        );
        assert_ne!(
            header.model_fingerprint, embedder.runtime,
            "header must not fall back to the runtime identity"
        );
    }

    #[test]
    fn flipped_checksum_byte_fails_validation() {
        let items = [stub_item("a", "alpha")];
        let mut bytes = build_artifact(
            ArtifactEntryKind::Tool,
            &items,
            sample_embedder_for(&items).as_ref(),
        )
        .unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        assert!(matches!(
            load_and_validate(&bytes),
            Err(ArtifactError::ChecksumMismatch)
        ));
    }

    #[test]
    fn unknown_format_version_rejects_without_reading_payload() {
        let items = [stub_item("a", "alpha")];
        let mut bytes = build_artifact(
            ArtifactEntryKind::Tool,
            &items,
            sample_embedder_for(&items).as_ref(),
        )
        .unwrap();
        bytes[4..8].copy_from_slice(&999u32.to_le_bytes());
        if bytes.len() > FILE_PREFIX_LEN {
            bytes[FILE_PREFIX_LEN] ^= 0xff;
        }
        assert!(matches!(
            load_and_validate(&bytes),
            Err(ArtifactError::UnsupportedFormatVersion {
                found: 999,
                supported: SUPPORTED_FORMAT_VERSION
            })
        ));
    }

    #[test]
    fn truncated_file_fails_with_too_short() {
        assert!(matches!(
            load_and_validate(b"RAT"),
            Err(ArtifactError::TooShort { needed: 8, got: 3 })
        ));

        let items = [stub_item("a", "alpha")];
        let bytes = build_artifact(
            ArtifactEntryKind::Tool,
            &items,
            sample_embedder_for(&items).as_ref(),
        )
        .unwrap();
        let mut truncated = bytes.clone();
        truncated[8..16].copy_from_slice(&((bytes.len() + 100) as u64).to_le_bytes());
        assert!(matches!(
            load_and_validate(&truncated),
            Err(ArtifactError::TooShort { .. })
        ));
    }

    #[test]
    fn structurally_corrupt_payload_with_valid_checksum_is_rejected() {
        let mut payload = Vec::new();
        write_u32(&mut payload, projection_version());
        write_u32(&mut payload, 2);
        write_utf8(&mut payload, "fp", 0).unwrap();
        write_u32(&mut payload, 1);
        payload.push(0);
        let id_at = payload.len();
        write_utf8(&mut payload, "x", id_at).unwrap();
        payload.extend_from_slice(&[0u8; 32]);
        payload.extend_from_slice(&1.0f32.to_le_bytes());

        let file = assemble_file(&payload);
        assert!(matches!(
            load_and_validate(&file),
            Err(ArtifactError::CorruptPayload { .. })
        ));
    }

    #[test]
    fn absurd_entry_count_with_valid_checksum_is_rejected_before_allocation() {
        let mut payload = Vec::new();
        write_u32(&mut payload, projection_version());
        write_u32(&mut payload, 2);
        write_utf8(&mut payload, "fp", 0).unwrap();
        write_u32(&mut payload, u32::MAX);

        let file = assemble_file(&payload);
        assert!(matches!(
            load_and_validate(&file),
            Err(ArtifactError::CorruptPayload { .. })
        ));
    }

    #[test]
    fn absurd_dim_with_valid_checksum_is_rejected_before_allocation() {
        let mut payload = Vec::new();
        write_u32(&mut payload, projection_version());
        write_u32(&mut payload, u32::MAX);
        write_utf8(&mut payload, "fp", 0).unwrap();
        write_u32(&mut payload, 1);
        // Minimal entry prefix so decode reaches the dim capacity path conceptually;
        // the pre-check must reject before allocating dim floats.
        payload.push(0);
        let id_at = payload.len();
        write_utf8(&mut payload, "x", id_at).unwrap();
        payload.extend_from_slice(&[0u8; 32]);

        let file = assemble_file(&payload);
        assert!(matches!(
            load_and_validate(&file),
            Err(ArtifactError::CorruptPayload { .. })
        ));
    }

    #[test]
    fn trailing_bytes_after_artifact_are_rejected() {
        let items = [stub_item("a", "alpha")];
        let bytes = build_artifact(
            ArtifactEntryKind::Tool,
            &items,
            sample_embedder_for(&items).as_ref(),
        )
        .unwrap();
        let mut with_garbage = bytes.clone();
        with_garbage.push(0xFF);
        assert!(matches!(
            load_and_validate(&with_garbage),
            Err(ArtifactError::CorruptPayload { at, .. })
                if at == bytes.len() - FILE_PREFIX_LEN
        ));
    }

    #[test]
    fn serialized_bytes_contain_no_sensitive_plaintext() {
        let secret_description = "SECRET_DESCRIPTION_DO_NOT_SERIALIZE";
        let secret_body = "SECRET_BODY_WITH_EXECUTOR_AND_CREDENTIALS";
        let secret_api_key = "sk-live-abc123supersecret";
        let projection = "minimal public projection";

        let sensitive = SensitiveItem {
            id: "tool_id_only".into(),
            description: secret_description.into(),
            body: secret_body.into(),
            api_key: secret_api_key.into(),
            projection: projection.into(),
        };

        let embedder = Arc::new(StubEmbedder {
            fingerprint: "test|model=1:m".into(),
            vectors: vec![unit([1.0, 0.0])],
            passthrough_batch: false,
        });

        let bytes =
            build_artifact(ArtifactEntryKind::Tool, [&sensitive], embedder.as_ref()).unwrap();

        let blob = String::from_utf8_lossy(&bytes);
        for forbidden in [
            secret_description,
            secret_body,
            secret_api_key,
            "executor",
            projection,
        ] {
            assert!(
                !blob.contains(forbidden),
                "forbidden plaintext {forbidden:?} found in artifact bytes"
            );
        }
        assert!(
            blob.contains("tool_id_only"),
            "id is part of the wire format"
        );
    }

    // description, body and api_key are intentionally never read
    // the test proves they never reach the artifact bytes
    #[allow(dead_code)]
    struct SensitiveItem {
        id: String,
        description: String,
        body: String,
        api_key: String,
        projection: String,
    }

    impl Embeddable for SensitiveItem {
        fn embed_id(&self) -> &str {
            &self.id
        }
        fn embed_text(&self) -> String {
            self.projection.clone()
        }
    }

    #[test]
    fn mixed_width_vectors_reject_with_inconsistent_width() {
        let items = [stub_item("first", "alpha"), stub_item("second", "beta")];
        let embedder = Arc::new(StubEmbedder {
            fingerprint: "test|model=1:m".into(),
            vectors: vec![unit([1.0, 0.0]), vec![0.0, 1.0, 0.0]],
            passthrough_batch: false,
        });
        assert!(matches!(
            build_artifact(ArtifactEntryKind::Tool, &items, embedder.as_ref()),
            Err(ArtifactError::InconsistentVectorWidth {
                expected: 2,
                got: 3,
            })
        ));
    }

    #[test]
    fn non_normalized_vector_rejects_with_vector_not_normalized() {
        let items = [stub_item("bad_item", "alpha")];
        let embedder = Arc::new(StubEmbedder {
            fingerprint: "test|model=1:m".into(),
            vectors: vec![vec![2.0, 0.0]],
            passthrough_batch: false,
        });
        assert!(matches!(
            build_artifact(ArtifactEntryKind::Tool, &items, embedder.as_ref()),
            Err(ArtifactError::VectorNotNormalized { id }) if id == "bad_item"
        ));
    }

    #[test]
    fn build_rejects_non_finite_vector_with_invalid_vector() {
        let items = [stub_item("nan_item", "alpha")];
        let embedder = Arc::new(StubEmbedder {
            fingerprint: "test|model=1:m".into(),
            vectors: vec![vec![f32::NAN, 0.0]],
            passthrough_batch: false,
        });
        assert!(matches!(
            build_artifact(ArtifactEntryKind::Tool, &items, embedder.as_ref()),
            Err(ArtifactError::InvalidVector { id, detail })
                if id == "nan_item" && detail == "non-finite component"
        ));
    }

    #[test]
    fn built_vectors_are_l2_normalized() {
        let items = [stub_item("a", "alpha"), stub_item("b", "beta")];
        let bytes = build_artifact(
            ArtifactEntryKind::Skill,
            &items,
            sample_embedder_for(&items).as_ref(),
        )
        .unwrap();
        let (_, entries) = load_and_validate(&bytes).unwrap();
        for entry in entries {
            assert!(
                is_unit_normalized(&entry.vector),
                "vector for {} must be unit-normalized",
                entry.id
            );
        }
    }

    fn hand_entry(id: &str, vector: Vec<f32>) -> ArtifactEntry {
        ArtifactEntry {
            kind: ArtifactEntryKind::Tool,
            id: id.into(),
            projection_hash: [0u8; 32],
            vector,
        }
    }

    #[test]
    fn load_rejects_nonempty_zero_dim_with_valid_checksum() {
        let bytes = hand_artifact(
            projection_version(),
            0,
            "fp-zero-dim",
            &[hand_entry("a", vec![])],
        );
        assert!(matches!(
            load_and_validate(&bytes),
            Err(ArtifactError::NonEmptyZeroDim)
        ));
    }

    #[test]
    fn build_rejects_nonempty_zero_dim_vectors() {
        let items = [stub_item("empty_vec", "alpha")];
        let embedder = Arc::new(StubEmbedder {
            fingerprint: "test|model=1:m".into(),
            vectors: vec![vec![]],
            passthrough_batch: false,
        });
        assert!(matches!(
            build_artifact(ArtifactEntryKind::Tool, &items, embedder.as_ref()),
            Err(ArtifactError::NonEmptyZeroDim)
        ));
    }

    #[test]
    fn canonical_empty_artifact_still_loads() {
        let bytes = build_empty_artifact().unwrap();
        let (header, entries) = load_and_validate(&bytes).unwrap();
        assert!(entries.is_empty());
        assert_eq!(header.dim, 0);
    }

    #[test]
    fn empty_artifact_with_nonzero_dim_still_loads() {
        let bytes = hand_artifact(projection_version(), 8, "fp-empty-nonzero-dim", &[]);
        let (header, entries) = load_and_validate(&bytes).unwrap();
        assert!(entries.is_empty());
        assert_eq!(header.dim, 8);
    }

    #[test]
    fn load_rejects_nan_vector_with_valid_checksum() {
        let bytes = hand_artifact(
            projection_version(),
            2,
            "fp-nan",
            &[hand_entry("bad", vec![f32::NAN, 0.0])],
        );
        assert!(matches!(
            load_and_validate(&bytes),
            Err(ArtifactError::InvalidVector { id, detail })
                if id == "bad" && detail == "non-finite component"
        ));
    }

    #[test]
    fn load_rejects_pos_infinity_vector_with_valid_checksum() {
        let bytes = hand_artifact(
            projection_version(),
            2,
            "fp-pinf",
            &[hand_entry("bad", vec![f32::INFINITY, 0.0])],
        );
        assert!(matches!(
            load_and_validate(&bytes),
            Err(ArtifactError::InvalidVector { id, detail })
                if id == "bad" && detail == "non-finite component"
        ));
    }

    #[test]
    fn load_rejects_neg_infinity_vector_with_valid_checksum() {
        let bytes = hand_artifact(
            projection_version(),
            2,
            "fp-ninf",
            &[hand_entry("bad", vec![f32::NEG_INFINITY, 0.0])],
        );
        assert!(matches!(
            load_and_validate(&bytes),
            Err(ArtifactError::InvalidVector { id, detail })
                if id == "bad" && detail == "non-finite component"
        ));
    }

    #[test]
    fn load_rejects_non_unit_vector_with_valid_checksum() {
        let bytes = hand_artifact(
            projection_version(),
            2,
            "fp-nonunit",
            &[hand_entry("bad", vec![3.0, 0.0])],
        );
        assert!(matches!(
            load_and_validate(&bytes),
            Err(ArtifactError::InvalidVector { id, detail })
                if id == "bad" && detail == "not unit-normalized"
        ));
    }

    #[test]
    fn load_accepts_unit_normalized_vector() {
        let bytes = hand_artifact(
            projection_version(),
            2,
            "fp-unit",
            &[hand_entry("ok", unit([1.0, 0.0]))],
        );
        let (header, entries) = load_and_validate(&bytes).unwrap();
        assert_eq!(header.dim, 2);
        assert_eq!(entries.len(), 1);
        assert!(is_unit_normalized(&entries[0].vector));
    }

    #[test]
    fn unit_norm_tolerance_matches_is_unit_normalized() {
        let tol = super::UNIT_NORM_SQ_TOLERANCE;
        // Shared squared-norm predicate with values comfortably inside / outside
        // the band (avoid 1.0 ± tol, which f32 addition may push past the bound).
        assert!(super::unit_norm_sq_ok(1.0));
        assert!(super::unit_norm_sq_ok(1.0 + tol * 0.5));
        assert!(!super::unit_norm_sq_ok(1.0 + tol * 2.0));

        let inside = vec![(1.0 + tol * 0.5).sqrt(), 0.0];
        assert!(is_unit_normalized(&inside));
        let bytes_ok = hand_artifact(
            projection_version(),
            2,
            "fp-tol-in",
            &[hand_entry("ok", inside)],
        );
        assert!(load_and_validate(&bytes_ok).is_ok());

        let outside = vec![(1.0 + tol * 2.0).sqrt(), 0.0];
        assert!(!is_unit_normalized(&outside));
        let bytes_bad = hand_artifact(
            projection_version(),
            2,
            "fp-tol-out",
            &[hand_entry("bad", outside)],
        );
        assert!(matches!(
            load_and_validate(&bytes_bad),
            Err(ArtifactError::InvalidVector { detail, .. }) if detail == "not unit-normalized"
        ));
    }

    #[test]
    fn merge_rejects_checksum_valid_invalid_vector() {
        let bad = hand_artifact(
            projection_version(),
            2,
            "fp-merge-nan",
            &[hand_entry("bad", vec![f32::NAN, 0.0])],
        );
        assert!(matches!(
            merge_embedding_artifacts(&[&bad]),
            Err(ArtifactError::InvalidVector { id, detail })
                if id == "bad" && detail == "non-finite component"
        ));
    }

    #[test]
    fn merge_tool_and_skill_artifacts_round_trips() {
        let tools = [stub_item("t", "tool text")];
        let skills = [stub_item("s", "skill text")];
        let embedder =
            sample_embedder_for(&[stub_item("t", "tool text"), stub_item("s", "skill text")]);
        let tool_bytes =
            build_artifact(ArtifactEntryKind::Tool, &tools, embedder.as_ref()).unwrap();
        let skill_bytes =
            build_artifact(ArtifactEntryKind::Skill, &skills, embedder.as_ref()).unwrap();
        let merged = merge_embedding_artifacts(&[&tool_bytes, &skill_bytes]).unwrap();
        let (header, entries) = load_and_validate(&merged).unwrap();
        assert_eq!(header.format_version, SUPPORTED_FORMAT_VERSION);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].kind, ArtifactEntryKind::Tool);
        assert_eq!(entries[0].id, "t");
        assert_eq!(entries[1].kind, ArtifactEntryKind::Skill);
        assert_eq!(entries[1].id, "s");
    }

    #[test]
    fn merge_empty_parts_yields_empty_artifact() {
        let empty = build_empty_artifact().unwrap();
        let merged = merge_embedding_artifacts(&[&empty, &empty]).unwrap();
        let (header, entries) = load_and_validate(&merged).unwrap();
        assert!(entries.is_empty());
        assert_eq!(header.dim, 0);
        assert!(header.model_fingerprint.is_empty());
    }

    #[test]
    fn merge_empty_with_nonempty_is_identity() {
        let items = [stub_item("a", "alpha")];
        let nonempty = build_artifact(
            ArtifactEntryKind::Tool,
            &items,
            sample_embedder_for(&items).as_ref(),
        )
        .unwrap();
        let empty = build_empty_artifact().unwrap();
        let merged = merge_embedding_artifacts(&[&empty, &nonempty]).unwrap();
        let (_, entries) = load_and_validate(&merged).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "a");
    }

    #[test]
    fn merge_rejects_model_fingerprint_mismatch() {
        let items = [stub_item("a", "alpha")];
        let a = build_artifact(
            ArtifactEntryKind::Tool,
            &items,
            Arc::new(StubEmbedder {
                fingerprint: "fp-a".into(),
                vectors: vec![unit([1.0, 0.0])],
                passthrough_batch: false,
            })
            .as_ref(),
        )
        .unwrap();
        let b = build_artifact(
            ArtifactEntryKind::Skill,
            &items,
            Arc::new(StubEmbedder {
                fingerprint: "fp-b".into(),
                vectors: vec![unit([0.0, 1.0])],
                passthrough_batch: false,
            })
            .as_ref(),
        )
        .unwrap();
        assert!(matches!(
            merge_embedding_artifacts(&[&a, &b]),
            Err(ArtifactError::IncompatibleMerge { .. })
        ));
    }

    #[test]
    fn merge_rejects_duplicate_kind_id() {
        let items = [stub_item("dup", "alpha")];
        let a = build_artifact(
            ArtifactEntryKind::Tool,
            &items,
            sample_embedder_for(&items).as_ref(),
        )
        .unwrap();
        let b = build_artifact(
            ArtifactEntryKind::Tool,
            &items,
            sample_embedder_for(&items).as_ref(),
        )
        .unwrap();
        assert!(matches!(
            merge_embedding_artifacts(&[&a, &b]),
            Err(ArtifactError::IncompatibleMerge { detail }) if detail.contains("duplicate")
        ));
    }

    #[test]
    fn merge_allows_same_id_across_kinds() {
        let items = [stub_item("search", "text")];
        let tool = build_artifact(
            ArtifactEntryKind::Tool,
            &items,
            sample_embedder_for(&items).as_ref(),
        )
        .unwrap();
        let skill = build_artifact(
            ArtifactEntryKind::Skill,
            &items,
            sample_embedder_for(&items).as_ref(),
        )
        .unwrap();
        let merged = merge_embedding_artifacts(&[&tool, &skill]).unwrap();
        let (_, entries) = load_and_validate(&merged).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].kind, ArtifactEntryKind::Tool);
        assert_eq!(entries[1].kind, ArtifactEntryKind::Skill);
        assert_eq!(entries[0].id, "search");
        assert_eq!(entries[1].id, "search");
    }

    #[test]
    fn merge_malformed_input_is_corrupt_not_incompatible() {
        assert!(matches!(
            merge_embedding_artifacts(&[b"not-a-rat1-file"]),
            Err(ArtifactError::InvalidMagic { .. } | ArtifactError::TooShort { .. })
        ));
    }

    #[test]
    fn build_rejects_fewer_vectors_than_inputs() {
        let items = [stub_item("a", "alpha"), stub_item("b", "beta")];
        let err = build_artifact(
            ArtifactEntryKind::Tool,
            &items,
            Arc::new(StubEmbedder {
                fingerprint: "fp".into(),
                vectors: vec![unit([1.0, 0.0])],
                passthrough_batch: true,
            })
            .as_ref(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ArtifactError::Embedder(EmbedderError::Inference { source })
                if source.contains("1 embeddings for 2 inputs")
        ));
    }

    #[test]
    fn build_rejects_more_vectors_than_inputs() {
        let items = [stub_item("a", "alpha")];
        let err = build_artifact(
            ArtifactEntryKind::Tool,
            &items,
            Arc::new(StubEmbedder {
                fingerprint: "fp".into(),
                vectors: vec![unit([1.0, 0.0]), unit([0.0, 1.0])],
                passthrough_batch: true,
            })
            .as_ref(),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ArtifactError::Embedder(EmbedderError::Inference { source })
                if source.contains("2 embeddings for 1 inputs")
        ));
    }

    #[test]
    fn merge_rejects_dim_mismatch_as_incompatible() {
        let a = hand_artifact(
            1,
            2,
            "fp",
            &[ArtifactEntry {
                kind: ArtifactEntryKind::Tool,
                id: "a".into(),
                projection_hash: [0; 32],
                vector: unit([1.0, 0.0]),
            }],
        );
        let b = hand_artifact(
            1,
            3,
            "fp",
            &[ArtifactEntry {
                kind: ArtifactEntryKind::Skill,
                id: "b".into(),
                projection_hash: [1; 32],
                vector: unit3([1.0, 0.0, 0.0]),
            }],
        );
        assert!(matches!(
            merge_embedding_artifacts(&[&a, &b]),
            Err(ArtifactError::IncompatibleMerge { detail }) if detail.contains("dim")
        ));
    }

    #[test]
    fn merge_rejects_projection_version_mismatch_as_incompatible() {
        let a = hand_artifact(
            1,
            2,
            "fp",
            &[ArtifactEntry {
                kind: ArtifactEntryKind::Tool,
                id: "a".into(),
                projection_hash: [0; 32],
                vector: unit([1.0, 0.0]),
            }],
        );
        let b = hand_artifact(
            2,
            2,
            "fp",
            &[ArtifactEntry {
                kind: ArtifactEntryKind::Skill,
                id: "b".into(),
                projection_hash: [1; 32],
                vector: unit([0.0, 1.0]),
            }],
        );
        assert!(matches!(
            merge_embedding_artifacts(&[&a, &b]),
            Err(ArtifactError::IncompatibleMerge { detail })
                if detail.contains("projection_version")
        ));
    }
}
