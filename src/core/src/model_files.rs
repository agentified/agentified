//! Shared plumbing for the crate's in-process models: the keyed load cache and
//! the HuggingFace-cache fetch helpers.
//!
//! Two model families live in this crate — the dense embedders
//! ([`crate::embedding`], ADR-0011/0012) and the prompt compressor
//! ([`crate::compression`], ADR-0016). They resolve different files into
//! different types and raise different error enums, but they share the parts
//! that are genuinely hard: a process-wide cache that loads once per model
//! without caching failures, and a fetch that survives the cold-cache
//! cross-process download race. Those exist **once**, here.
//!
//! Everything is generic over the caller's error type: the mechanics are shared,
//! the diagnostics stay specific ("embedding model X" vs "compression model X"),
//! so neither family inherits the other's vocabulary.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hf_hub::api::sync::ApiRepo;

/// A per-key load slot in a keyed cache: `None` until the value is loaded, then
/// `Some`. Guarded by its own mutex so a `load` serializes callers of *that* key
/// without holding the whole-map lock.
pub(crate) type LoadSlot<T> = Arc<Mutex<Option<Arc<T>>>>;

/// Get-or-load into a keyed cache with **no failure caching**: on `Err` the key's
/// slot stays empty so a later call retries. Returns the value plus `Some(load_ms)`
/// on the call that performed the load, `None` on warm reuse. Generic so the
/// non-poisoning + once contract is unit-tested without touching the network.
///
/// The map lock is held only long enough to get/create the key's slot; `load`
/// runs against that per-key slot mutex, never the map. So **different keys load
/// concurrently** (a cold load of one model no longer blocks another), while
/// **same-key loads stay single-flight** (concurrent cold misses share the slot,
/// so `load` runs once and exactly one caller reports `Some(load_ms)`).
pub(crate) fn get_or_load_keyed<T: ?Sized, E>(
    cache: &Mutex<HashMap<String, LoadSlot<T>>>,
    key: &str,
    load: impl FnOnce() -> Result<Arc<T>, E>,
) -> Result<(Arc<T>, Option<u64>), E> {
    // Hold the map lock only to get/create this key's slot, then release it.
    let slot = {
        let mut guard = cache.lock().expect("model cache mutex poisoned");
        Arc::clone(guard.entry(key.to_string()).or_default())
    };
    // Serialize per key on the slot — the map lock is NOT held across `load`.
    let mut slot = slot.lock().expect("model cache slot mutex poisoned");
    if let Some(existing) = slot.as_ref() {
        return Ok((existing.clone(), None));
    }
    let started = Instant::now();
    let loaded = load()?; // Err leaves the slot `None`, so a later call retries.
    let took_ms = started.elapsed().as_millis() as u64;
    *slot = Some(loaded.clone());
    Ok((loaded, Some(took_ms)))
}

/// Resolve one model file from the HF cache, tolerating the cross-process
/// download race on a cold cache. hf-hub guards each blob with a *non-blocking*
/// `flock` and gives up after ~5s (5 × 1s); a first fetch of the weights takes
/// longer, so when several processes load a model at once on a cold cache —
/// parallel test workers, a web server's worker pool cold-starting,
/// `multiprocessing` — every process but the lock holder gets `LockAcquisition`
/// and would fail. Retry with backoff: the losers wait for the winner's download
/// to land, then `get()` returns the now-cached blob without locking (hf-hub
/// checks the cache before it locks). Any other failure is handed to `classify`
/// and returned immediately. See ADR-0011.
pub(crate) fn fetch_cached_with<E>(
    repo: &ApiRepo,
    file: &str,
    classify: impl Fn(&str) -> E,
) -> Result<PathBuf, E> {
    // ~30 × (up to hf-hub's own ~5s lock wait + 1s backoff) comfortably outlasts
    // a single cold-cache download; the loser normally succeeds within a few.
    const MAX_ATTEMPTS: u32 = 30;
    const BACKOFF: Duration = Duration::from_secs(1);

    let mut attempt = 1;
    loop {
        match repo.get(file) {
            Ok(path) => return Ok(path),
            Err(e) => {
                let msg = e.to_string();
                if attempt < MAX_ATTEMPTS && is_lock_contention(&msg) {
                    attempt += 1;
                    std::thread::sleep(BACKOFF);
                    continue;
                }
                return Err(classify(&msg));
            }
        }
    }
}

/// True only when an hf-hub fetch failed because another process holds the
/// download lock — the one error worth retrying, since the blob appears once the
/// winner finishes. Every other failure is terminal and classified by the caller.
pub(crate) fn is_lock_contention(err: &str) -> bool {
    err.contains("Lock acquisition failed")
}

/// Whether an hf-hub fetch error means the cache itself could not be written
/// (permissions, disk full, read-only mount) rather than a network/model problem.
/// The distinction matters because the two have completely different remedies.
pub(crate) fn is_cache_unwritable(msg: &str) -> bool {
    let lower = msg.to_lowercase();
    lower.contains("permission denied")
        || lower.contains("read-only")
        || lower.contains("no space")
        || lower.contains("os error 13") // EACCES
        || lower.contains("os error 28") // ENOSPC
        || lower.contains("os error 30") // EROFS
}

/// Whether an hf-hub fetch error is a "file not in repo" (so the caller can try a
/// fallback file) rather than a real network/cache failure.
pub(crate) fn is_not_found(source: &str) -> bool {
    let l = source.to_lowercase();
    l.contains("404") || l.contains("not found") || l.contains("entry not found")
}

/// Best-effort optional fetch of a small side file (pooling config, alt weights):
/// `None` on any error, so a missing file never fails the load.
pub(crate) fn fetch_optional(repo: &ApiRepo, file: &str) -> Option<PathBuf> {
    repo.get(file).ok()
}

/// The concrete commit SHA a HuggingFace fetch resolved to, read from the cached
/// snapshot path (`…/snapshots/<sha>/<file>`). `None` if the path isn't in that
/// layout — the caller falls back to the requested revision string.
pub(crate) fn snapshot_sha(weights_path: &Path) -> Option<String> {
    let name = weights_path.parent()?.file_name()?.to_str()?;
    (name.len() == 40 && name.chars().all(|c| c.is_ascii_hexdigit())).then(|| name.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_lock_contention_is_retried() {
        // The cold-cache download race: retry and wait for the winner.
        assert!(is_lock_contention(
            "Lock acquisition failed: /home/u/.cache/huggingface/hub/models--BAAI--bge-small-en-v1.5/blobs/abc.lock"
        ));
        // Everything else is terminal — classify, don't spin.
        assert!(!is_lock_contention("request error: connection refused"));
        assert!(!is_lock_contention("Http(reqwest::Error { status: 404 })"));
        assert!(!is_lock_contention(
            "No such file or directory (os error 2)"
        ));
    }

    #[test]
    fn cache_unwritable_covers_permission_space_and_readonly() {
        assert!(is_cache_unwritable("Permission denied (os error 13)"));
        assert!(is_cache_unwritable("No space left on device (os error 28)"));
        assert!(is_cache_unwritable("Read-only file system (os error 30)"));
        assert!(!is_cache_unwritable(
            "error sending request: connection refused"
        ));
    }

    #[test]
    fn snapshot_sha_reads_the_commit_from_the_cache_layout() {
        let sha = "5f0c82792b7ea14c6484e015b6a072009496b7f2";
        assert_eq!(
            snapshot_sha(Path::new(&format!(
                "/hub/models--a--b/snapshots/{sha}/model.safetensors"
            ))),
            Some(sha.to_string())
        );
        // Not a 40-char hex directory → the caller keeps the requested revision.
        assert_eq!(
            snapshot_sha(Path::new("/models/local/model.safetensors")),
            None
        );
    }

    #[test]
    fn keyed_cache_is_generic_over_the_error_type() {
        // The mechanics are shared; each model family keeps its own error enum.
        let cache: Mutex<HashMap<String, LoadSlot<i32>>> = Mutex::new(HashMap::new());
        assert!(get_or_load_keyed(&cache, "k", || Err::<Arc<i32>, String>("boom".into())).is_err());
        let (v, ms) = get_or_load_keyed(&cache, "k", || Ok::<_, String>(Arc::new(7))).unwrap();
        assert_eq!(*v, 7);
        assert!(ms.is_some());
    }
}
