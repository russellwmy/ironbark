//! `experts.bin` / `experts.json`: where each expert record sits.
//!
//! Record `l·E + e` (layer `l`, expert `e`) sits at offset `(l·E + e) · stride`; `stride` is
//! `record_bytes` rounded up to whole pages. The store treats a record as opaque bytes: the
//! engine that writes `experts.bin` decides what is inside.

use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;

use crate::error::{Error, Result};

/// Turn off the OS page cache for `file` (`F_NOCACHE`): reads and writes go to storage and
/// leave no cached copy. Pages already cached are still served, so a file must be written
/// with this set for later reads to be real storage reads.
///
/// # Errors
///
/// The `fcntl` error, if the descriptor rejects the flag.
pub fn no_cache(file: &File) -> std::io::Result<()> {
    // SAFETY: `file` is borrowed for the whole call, so its descriptor is valid; F_NOCACHE
    // takes an int flag and touches no memory.
    let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
    if rc == -1 {
        Err(std::io::Error::last_os_error())
    } else {
        Ok(())
    }
}

/// Hash of an expert record (as in `manifest.json`): FNV-1a over little-endian u64 words, then
/// the tail bytes. Fast, and enough to catch truncated, partial or corrupted copies (not
/// tampering).
#[must_use]
pub fn hash(bytes: &[u8]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    let (words, tail) = bytes.as_chunks::<8>();
    for w in words {
        h = (h ^ u64::from_le_bytes(*w)).wrapping_mul(PRIME);
    }
    for &b in tail {
        h = (h ^ u64::from(b)).wrapping_mul(PRIME);
    }
    h
}

/// Shape of `experts.bin`, from `experts.json`.
#[derive(Debug, Clone)]
pub struct Layout {
    /// MoE layers.
    pub layers: usize,
    /// Experts per layer.
    pub experts: usize,
    /// Bytes of one record (without padding).
    pub record_bytes: u64,
    /// Bytes between records (page-aligned).
    pub stride: u64,
}

impl Layout {
    /// Read `experts.json` from `dir`.
    ///
    /// # Errors
    ///
    /// Fails if the file is missing, not JSON, or lacks a field.
    pub fn read(dir: &Path) -> Result<Self> {
        let path = dir.join("experts.json");
        let raw = std::fs::read(&path).map_err(Error::io(&path))?;
        let v: serde_json::Value = serde_json::from_slice(&raw).map_err(Error::json(&path))?;
        let u = |k: &str| v[k].as_u64().ok_or_else(|| Error::format(&path, format!("missing {k}")));
        Ok(Layout {
            layers: u("layers")? as usize,
            experts: u("experts")? as usize,
            record_bytes: u("record_bytes")?,
            stride: u("stride")?,
        })
    }
}
