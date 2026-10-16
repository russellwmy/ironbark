//! `manifest.json` of a packed model: the pack format version, every file's size and hash, and
//! one hash per expert record, so a copy (to a phone, say) can be checked with [`verify`].
//!
//! ```text
//!   { "version": 1,
//!     "files": { "config.json": { "bytes": 2710, "hash": "9c1e…" }, …,
//!                "experts.bin": { "bytes": …, "stride": 655360, "record_bytes": 645120,
//!                                 "records": ["5a0f…", …] },
//!                "embed.bin": { "bytes": …, "stride": 65536, "record_bytes": 65536,
//!                               "records": [ one hash per chunk ] } } }
//! ```
//!
//! Hashes are FNV-1a over little-endian u64 words (then the tail bytes): fast, and enough to
//! catch truncated, partial or corrupted copies (not tampering).

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use serde_json::{Map, Value, json};

use crate::{Error, Result};

/// Pack format version this library writes and reads.
pub const VERSION: u64 = 1;

/// Manifest file name.
pub const FILE: &str = "manifest.json";

/// Fail unless `v` (from the file at `path`) is of pack format [`VERSION`].
pub(crate) fn check_version(v: &Value, path: &Path) -> Result<()> {
    match v["version"].as_u64() {
        Some(VERSION) => Ok(()),
        found => Err(Error::Format(format!(
            "{}: pack format {}, this library reads v{VERSION}; repack with `ironbark pack`",
            path.display(),
            found.map_or_else(|| "unversioned".to_owned(), |n| format!("v{n}"))
        ))),
    }
}

/// Hash of `bytes` (FNV-1a over u64 words).
pub use ironbark_experts::layout::hash;

fn hex(h: u64) -> String {
    format!("{h:016x}")
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |source| Error::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Write `dir/manifest.json`: every file in `dir` (except the manifest), with `records` the
/// hashes of the expert records of `experts.bin` (`record_bytes` each, `stride` apart).
///
/// # Errors
///
/// Fails if a file cannot be read or the manifest written.
pub fn write(dir: &Path, records: &[u64], record_bytes: usize, stride: usize) -> Result<()> {
    let mut files = Map::new();
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .map_err(io(dir))?
        .filter_map(|e| e.ok().and_then(|e| e.file_name().into_string().ok()))
        .filter(|n| n != FILE && !n.starts_with('.'))
        .collect();
    names.sort();
    for name in names {
        let path = dir.join(&name);
        let bytes = std::fs::metadata(&path).map_err(io(&path))?.len();
        let entry = if name == "experts.bin" {
            json!({
                "bytes": bytes,
                "stride": stride,
                "record_bytes": record_bytes,
                "records": records.iter().map(|&h| hex(h)).collect::<Vec<_>>(),
            })
        } else if name == crate::embed::FILE {
            // Per chunk, so a sparse copy can check each one it fetches (`ironbark-fetch`).
            let data = std::fs::read(&path).map_err(io(&path))?;
            let c = crate::embed::chunk(data.len());
            json!({
                "bytes": bytes,
                "stride": c,
                "record_bytes": c,
                "records": data.chunks(c).map(|d| hex(hash(d))).collect::<Vec<_>>(),
            })
        } else {
            json!({ "bytes": bytes, "hash": hex(hash(&std::fs::read(&path).map_err(io(&path))?)) })
        };
        files.insert(name, entry);
    }
    let path = dir.join(FILE);
    let text = serde_json::to_string_pretty(&json!({ "version": VERSION, "files": files }))
        .map_err(|e| Error::Format(e.to_string()))?;
    std::fs::write(&path, text).map_err(io(&path))
}

/// What [`verify`] checked and found.
#[derive(Debug, Clone, Default)]
pub struct Report {
    /// Files checked.
    pub files: usize,
    /// Expert records checked.
    pub records: usize,
    /// Bytes read.
    pub bytes: u64,
    /// Each problem found (missing, wrong size, wrong hash), empty if the pack is intact.
    pub problems: Vec<String>,
}

/// Check every file and expert record of the pack in `dir` against its manifest, reading
/// `experts.bin` uncached on `threads` threads.
///
/// # Errors
///
/// Fails if there is no readable manifest, or it is of another format version. Damage to the pack
/// itself is reported in [`Report::problems`].
pub fn verify(dir: &Path, threads: usize) -> Result<Report> {
    let path = dir.join(FILE);
    let raw = std::fs::read(&path).map_err(io(&path))?;
    let m: Value = serde_json::from_slice(&raw).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
    check_version(&m, &path)?;
    let files = m["files"].as_object().ok_or_else(|| Error::Format(format!("{}: no files", path.display())))?;
    // Experts per layer, to name a damaged record by layer and expert.
    let per_layer = ironbark_experts::Layout::read(dir).map_or(1, |l| l.experts.max(1));
    let mut r = Report::default();
    for (name, f) in files {
        r.files += 1;
        let p = dir.join(name);
        let want = f["bytes"].as_u64().unwrap_or(u64::MAX);
        let Ok(md) = std::fs::metadata(&p) else {
            r.problems.push(format!("{name}: missing"));
            continue;
        };
        let have = md.len();
        if have != want {
            r.problems.push(format!("{name}: {have} bytes, {want} expected (incomplete copy?)"));
            continue;
        }
        if let Some(records) = f["records"].as_array() {
            let stride = f["stride"].as_u64().unwrap_or(0) as usize;
            let len = f["record_bytes"].as_u64().unwrap_or(0) as usize;
            let want: Vec<&str> = records.iter().filter_map(Value::as_str).collect();
            let bad = check_records(&p, &want, stride, len, threads.max(1))?;
            r.records += want.len();
            r.bytes += (want.len() * len) as u64;
            r.problems.extend(bad.into_iter().map(|i| {
                if name == "experts.bin" {
                    format!("{name}: record {i} (layer {}, expert {}) differs", i / per_layer, i % per_layer)
                } else {
                    format!("{name}: chunk {i} differs")
                }
            }));
        } else {
            let data = std::fs::read(&p).map_err(io(&p))?;
            r.bytes += data.len() as u64;
            if f["hash"].as_str() != Some(hex(hash(&data)).as_str()) {
                r.problems.push(format!("{name}: contents differ"));
            }
        }
    }
    Ok(r)
}

/// Indices of the records of `path` whose hash differs from `want`.
fn check_records(path: &Path, want: &[&str], stride: usize, len: usize, threads: usize) -> Result<Vec<usize>> {
    let file = File::open(path).map_err(io(path))?;
    ironbark_experts::layout::no_cache(&file).map_err(io(path))?;
    let per = want.len().div_ceil(threads);
    let bad = std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let file = &file;
                sc.spawn(move || -> std::io::Result<Vec<usize>> {
                    let mut buf = vec![0u8; len];
                    let mut bad = Vec::new();
                    for (i, w) in want.iter().enumerate().skip(t * per).take(per) {
                        file.read_exact_at(&mut buf, (i * stride) as u64)?;
                        if hex(hash(&buf)) != *w {
                            bad.push(i);
                        }
                    }
                    Ok(bad)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| Err(std::io::Error::other("checker panicked"))))
            .collect::<std::io::Result<Vec<Vec<usize>>>>()
    })
    .map_err(io(path))?;
    Ok(bad.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verify_catches_size_and_content_changes() {
        let dir = std::env::temp_dir().join(format!("ironbark-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (len, stride) = (100usize, 128usize);
        let recs: Vec<Vec<u8>> = (0..4u8).map(|i| vec![i; len]).collect();
        let mut bin = vec![0u8; 4 * stride];
        for (i, r) in recs.iter().enumerate() {
            bin[i * stride..i * stride + len].copy_from_slice(r);
        }
        std::fs::write(dir.join("experts.bin"), &bin).unwrap();
        std::fs::write(dir.join("config.json"), b"{}").unwrap();
        let hashes: Vec<u64> = recs.iter().map(|r| hash(r)).collect();
        write(&dir, &hashes, len, stride).unwrap();
        assert_eq!(verify(&dir, 2).unwrap().problems, Vec::<String>::new());

        bin[2 * stride + 5] ^= 1;
        std::fs::write(dir.join("experts.bin"), &bin).unwrap();
        let r = verify(&dir, 2).unwrap();
        assert_eq!(r.problems.len(), 1, "{:?}", r.problems);
        assert!(r.problems[0].contains("record 2"));

        std::fs::write(dir.join("config.json"), b"{ }").unwrap();
        let r = verify(&dir, 2).unwrap();
        assert!(r.problems.iter().any(|p| p.contains("config.json")), "{:?}", r.problems);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
