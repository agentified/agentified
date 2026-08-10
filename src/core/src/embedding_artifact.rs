//! Build-time embedding artifact: corpus vectors serialized for runtime load
//! without re-inferring. Model identity is only `model_fingerprint` from the
//! embed batch (pooling/prefixes live inside it via `fingerprint_suffix`).

#![allow(dead_code)]

use sha2::{Digest, Sha256};

use crate::dense_cache::Embeddable;
use crate::embedding::{Embedder, EmbedderError, Embedded};

const MAGIC: &[u8; 4] = b"RAT1";
const SUPPORTED_FORMAT_VERSION: u32 = 1;
const VERSION_PREFIX_LEN: usize = 8;
const FILE_PREFIX_LEN: usize = 4 + 4 + 8 + 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArtifactEntryKind {
    Tool,
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

#[derive(Debug, Clone)]
pub(crate) enum ArtifactError {
    TooShort { needed: usize, got: usize },
    InvalidMagic { got: [u8; 4] },
    UnsupportedFormatVersion { found: u32, supported: u32 },
    ChecksumMismatch,
    CorruptPayload { at: usize, detail: String },
    InconsistentVectorWidth { expected: usize, got: usize },
    VectorNotNormalized { id: String },
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

/// Hash of [`crate::indexing`] + [`crate::skill_indexing`] sources — bumps when
/// projection logic changes without a manual version constant.
pub(crate) fn projection_version() -> u32 {
    let mut h = Sha256::new();
    h.update(include_str!("indexing.rs"));
    h.update(include_str!("skill_indexing.rs"));
    u32::from_le_bytes(h.finalize()[..4].try_into().expect("4 bytes"))
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

    let texts: Vec<String> = rows.iter().map(|(_, text)| text.clone()).collect();
    let Embedded {
        value: vectors,
        fingerprint: model_fingerprint,
    } = embedder.embed_batch_with_identity(&texts)?;

    let dim = vectors.first().map(Vec::len).unwrap_or(0);
    for ((id, _), vector) in rows.iter().zip(&vectors) {
        if vector.len() != dim {
            return Err(ArtifactError::InconsistentVectorWidth {
                expected: dim,
                got: vector.len(),
            });
        }
        if dim != 0 && !is_unit_normalized(vector) {
            return Err(ArtifactError::VectorNotNormalized { id: id.clone() });
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
    assemble_file(&payload)
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

    let payload = &bytes[FILE_PREFIX_LEN..needed];
    let computed = Sha256::digest(payload);
    if computed.as_slice() != declared_checksum {
        return Err(ArtifactError::ChecksumMismatch);
    }

    decode_payload(format_version, payload)
}

fn hash_projection_text(text: &str) -> [u8; 32] {
    Sha256::digest(text.as_bytes()).into()
}

fn is_unit_normalized(vector: &[f32]) -> bool {
    let norm_sq: f32 = vector.iter().map(|x| x * x).sum();
    (norm_sq - 1.0).abs() <= 1e-4
}

fn assemble_file(payload: &[u8]) -> Result<Vec<u8>, ArtifactError> {
    let mut out = Vec::with_capacity(FILE_PREFIX_LEN + payload.len());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&SUPPORTED_FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    out.extend_from_slice(&Sha256::digest(payload));
    out.extend_from_slice(payload);
    Ok(out)
}

fn encode_payload(header: &ArtifactHeader, entries: &[ArtifactEntry]) -> Result<Vec<u8>, ArtifactError> {
    let mut out = Vec::new();
    write_u32(&mut out, header.projection_version);
    write_u32(&mut out, header.dim.try_into().map_err(|_| ArtifactError::CorruptPayload {
        at: 0,
        detail: "dim does not fit u32".into(),
    })?);
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
        write_kind(&mut out, entry.kind)?;
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

    let mut entries = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        let kind = read_kind(payload, &mut cursor)?;
        let id = read_utf8(payload, &mut cursor)?;
        let projection_hash = read_fixed::<32>(payload, &mut cursor)?;
        let mut vector = Vec::with_capacity(dim);
        for _ in 0..dim {
            vector.push(read_f32(payload, &mut cursor)?);
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

fn write_kind(out: &mut Vec<u8>, kind: ArtifactEntryKind) -> Result<(), ArtifactError> {
    out.push(match kind {
        ArtifactEntryKind::Tool => 0,
        ArtifactEntryKind::Skill => 1,
    });
    Ok(())
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
    let end = start.checked_add(len).ok_or(ArtifactError::CorruptPayload {
        at: start,
        detail: "utf8 length overflow".into(),
    })?;
    if payload.len() < end {
        return Err(ArtifactError::CorruptPayload {
            at: start,
            detail: format!("utf8 field claims {len} bytes but payload ends early"),
        });
    }
    let s = std::str::from_utf8(&payload[start..end]).map_err(|e| ArtifactError::CorruptPayload {
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
            let value = if self.vectors.len() == texts.len() {
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
        })
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
        assert_eq!(header.model_fingerprint, "hf|repo=1:r|revision=4:main|pool=3:cls");
        assert_eq!(header.dim, 2);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].kind, ArtifactEntryKind::Tool);
        assert_eq!(entries[0].id, "read_file");
        assert_eq!(entries[0].projection_hash, hash_projection_text("read file from disk"));
        assert_eq!(entries[1].id, "write_file");
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

        let file = assemble_file(&payload).unwrap();
        assert!(matches!(
            load_and_validate(&file),
            Err(ArtifactError::CorruptPayload { .. })
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
        assert!(blob.contains("tool_id_only"), "id is part of the wire format");
    }

    struct SensitiveItem {
        id: String,
        #[allow(dead_code)]
        description: String,
        #[allow(dead_code)]
        body: String,
        #[allow(dead_code)]
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
        let items = [
            stub_item("first", "alpha"),
            stub_item("second", "beta"),
        ];
        let embedder = Arc::new(StubEmbedder {
            fingerprint: "test|model=1:m".into(),
            vectors: vec![unit([1.0, 0.0]), vec![0.0, 1.0, 0.0]],
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
        });
        assert!(matches!(
            build_artifact(ArtifactEntryKind::Tool, &items, embedder.as_ref()),
            Err(ArtifactError::VectorNotNormalized { id }) if id == "bad_item"
        ));
    }

    #[test]
    fn built_vectors_are_l2_normalized() {
        let items = [
            stub_item("a", "alpha"),
            stub_item("b", "beta"),
        ];
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
}
