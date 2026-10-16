//! Build a local `experts.bin` (or `embed.bin`) from a remote copy (a server or CDN holding the
//! same file), separate from the engine: a [`Sparse`] file fills in the background on its own,
//! and is also the engine's [`Records`] (experts) or [`Chunks`] (embedding) source, so a record
//! the engine needs first is fetched at once.
//!
//! ```text
//!   app ──▶ Sparse::open(dir, fetch) ──▶ start(threads, order) ──▶ progress() … stop()
//!                 │
//!                 └──▶ (optional) the engine reads through it: Store::open_with(dir, …, sparse)
//!
//!   read record i (engine loader)
//!     present bit set? ──yes──▶ pread local experts.bin
//!         │ no
//!         ▼
//!     Fetch::fetch(i · stride, record_bytes)   (HTTP Range GET, a callback, a file, …)
//!     hash == manifest record hash? ──no──▶ retry, then fail
//!         │ yes
//!         ▼
//!     serve it ─▶ pwrite + fsync local experts.bin ─▶ set bit i in experts.present
//!
//!   fill threads: the same for every record not yet present, in a given order, holding off
//!   while an engine read is fetching. When every bit is set, experts.present is removed: the next
//!   open is a plain local one.
//! ```
//!
//! A crash loses at most records whose bit was not yet written; their data is fetched again.

use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

use ironbark_experts::layout::{Layout, hash, no_cache};
use ironbark_experts::{Chunks, Error, Records, Result};

/// Presence bitmap of a sparse `experts.bin` (1 bit per record, record `i` in byte `i / 8`).
pub const PRESENT: &str = "experts.present";

/// The memory-mapped embedding of a packed model.
pub const EMBED: &str = "embed.bin";

/// Fetch attempts per record before a load fails.
const TRIES: u32 = 4;

/// A remote copy of `experts.bin`.
pub trait Fetch: Send + Sync {
    /// Fill `buf` with the remote file's bytes `offset .. offset + buf.len()`.
    ///
    /// # Errors
    ///
    /// Any transport error; the caller retries.
    fn fetch(&self, offset: u64, buf: &mut [u8]) -> io::Result<()>;
}

/// A complete `experts.bin` elsewhere on disk as the remote copy, with an optional delay per
/// fetch to stand in for a network round trip (for tests and benchmarks).
#[derive(Debug)]
pub struct FileFetch {
    file: File,
    delay: Duration,
}

impl FileFetch {
    /// Serve records from `path`, sleeping `delay` per fetch.
    ///
    /// # Errors
    ///
    /// Fails if `path` cannot be opened.
    pub fn open(path: &Path, delay: Duration) -> Result<Self> {
        let file = File::open(path).map_err(Error::io(path))?;
        no_cache(&file).map_err(Error::io(path))?;
        Ok(Self { file, delay })
    }
}

impl Fetch for FileFetch {
    #[expect(clippy::disallowed_methods, reason = "stands in for a network round trip")]
    fn fetch(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        std::thread::sleep(self.delay);
        self.file.read_exact_at(buf, offset)
    }
}

/// Download progress of a sparse `experts.bin`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Progress {
    /// Records on local disk.
    pub present: usize,
    /// Records in the file.
    pub total: usize,
    /// Bytes fetched (including retries).
    pub fetched_bytes: u64,
    /// Fetches that failed or returned a wrong hash (each was retried).
    pub errors: u64,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What fill threads wait on.
#[derive(Debug, Default)]
struct Gate {
    /// Loader fetches in flight: fill threads hold off while it is non-zero.
    demand: usize,
    /// Records being fetched: another reader of one waits for it instead of fetching again.
    fetching: Vec<bool>,
    stop: bool,
}

/// A local `experts.bin` that fills from a [`Fetch`]; complete once every record is present.
pub struct Sparse {
    file: File,
    path: PathBuf,
    present_path: PathBuf,
    /// `None` when complete at open.
    present_file: Option<File>,
    /// Presence bits; also serializes writes to `present_file`. Lock order: `gate`, then `bits`.
    bits: Mutex<Vec<u8>>,
    present: AtomicUsize,
    /// Expected hash per record (from `manifest.json`).
    hashes: Vec<u64>,
    fetch: Arc<dyn Fetch>,
    rec: usize,
    stride: u64,
    gate: Mutex<Gate>,
    /// Signalled when `gate` changes or a record arrives.
    changed: Condvar,
    fetched_bytes: AtomicU64,
    errors: AtomicU64,
    /// The last error a fill thread hit (for the progress report).
    last_error: Mutex<Option<String>>,
    fillers: Mutex<Vec<JoinHandle<()>>>,
}

impl fmt::Debug for Sparse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Sparse")
            .field("path", &self.path)
            .field("progress", &self.progress())
            .finish_non_exhaustive()
    }
}

/// The `manifest.json` entry of file `name` in `dir`.
fn manifest_entry(dir: &Path, name: &str) -> Result<serde_json::Value> {
    let path = dir.join("manifest.json");
    let raw = std::fs::read(&path).map_err(Error::io(&path))?;
    let mut v: serde_json::Value = serde_json::from_slice(&raw).map_err(Error::json(&path))?;
    Ok(v["files"][name].take())
}

/// Per-record hashes of file `name` from `dir/manifest.json`.
fn record_hashes(dir: &Path, name: &str, records: usize) -> Result<Vec<u64>> {
    let hashes: Option<Vec<u64>> = manifest_entry(dir, name)?["records"]
        .as_array()
        .map(|a| a.iter().filter_map(|h| h.as_str().and_then(|h| u64::from_str_radix(h, 16).ok())).collect());
    match hashes {
        Some(h) if h.len() == records => Ok(h),
        _ => Err(Error::format(
            dir.join("manifest.json"),
            format!("needs {records} record hashes for {name} (re-run pack)"),
        )),
    }
}

impl Records for Sparse {
    fn read(&self, i: usize, buf: &mut [u8]) -> io::Result<()> {
        Sparse::read(self, i, buf)
    }
}

impl Chunks for Sparse {
    fn ensure(&self, i: usize) -> io::Result<()> {
        if self.has(i) {
            return Ok(());
        }
        let mut buf = vec![0u8; self.rec];
        Sparse::read(self, i, &mut buf)
    }

    fn has(&self, i: usize) -> bool {
        Sparse::has(self, i)
    }

    fn complete(&self) -> bool {
        self.present.load(Relaxed) == self.hashes.len()
    }
}

impl Sparse {
    /// Open (or start) the `experts.bin` of `dir`, filling from `fetch`. `dir` needs
    /// `experts.json` and `manifest.json` (the record hashes). A complete `experts.bin` (no
    /// `experts.present` next to it) opens with every record present and never fetches.
    ///
    /// # Errors
    ///
    /// Fails if `experts.json` or `manifest.json` cannot be read, or the local files cannot be
    /// created.
    pub fn open(dir: &Path, fetch: Arc<dyn Fetch>) -> Result<Arc<Self>> {
        let layout = Layout::read(dir)?;
        let records = layout.layers * layout.experts;
        let hashes = record_hashes(dir, "experts.bin", records)?;
        // Read uncached, like a local `experts.bin`: the loaders copy records into the slab.
        Self::open_with(dir, "experts.bin", hashes, layout.record_bytes as usize, layout.stride, true, fetch)
    }

    /// Open (or start) the `embed.bin` of `dir`, filling from `fetch` (a remote copy of
    /// `embed.bin`), in the chunks `manifest.json` lists with their hashes. The engine maps the
    /// file, so it is written through the page cache.
    ///
    /// # Errors
    ///
    /// Fails if `manifest.json` cannot be read, or the local files cannot be created.
    pub fn open_embed(dir: &Path, fetch: Arc<dyn Fetch>) -> Result<Arc<Self>> {
        let entry = manifest_entry(dir, EMBED)?;
        let (chunk, bytes) = (entry["record_bytes"].as_u64().unwrap_or(0), entry["bytes"].as_u64().unwrap_or(0));
        if chunk == 0 || bytes % chunk != 0 {
            return Err(Error::format(dir.join("manifest.json"), format!("no chunks for {EMBED} (re-run pack)")));
        }
        let hashes = record_hashes(dir, EMBED, (bytes / chunk) as usize)?;
        Self::open_with(dir, EMBED, hashes, chunk as usize, chunk, false, fetch)
    }

    /// Open file `name` of `dir`: `hashes.len()` records of `rec` bytes, `stride` apart, with
    /// its presence bitmap `<stem>.present`.
    fn open_with(
        dir: &Path,
        name: &str,
        hashes: Vec<u64>,
        rec: usize,
        stride: u64,
        uncached: bool,
        fetch: Arc<dyn Fetch>,
    ) -> Result<Arc<Self>> {
        let records = hashes.len();
        let path = dir.join(name);
        let present_path = dir.join(format!("{}.present", name.trim_end_matches(".bin")));
        let complete = path.exists() && !present_path.exists();
        // The bitmap is created before the data file, so a data file without one is complete.
        let present_file = if complete {
            None
        } else {
            let f = OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&present_path)
                .map_err(Error::io(&present_path))?;
            Some(f)
        };
        let mut bits = vec![0u8; records.div_ceil(8)];
        if complete {
            (0..records).for_each(|i| bits[i / 8] |= 1 << (i % 8));
        } else {
            let have =
                present_file.as_ref().map_or(Ok(0), |f| f.read_at(&mut bits, 0)).map_err(Error::io(&present_path))?;
            bits[have..].fill(0);
        }
        // Bits past the last record are never set.
        let present = (0..records).filter(|&i| bits[i / 8] & (1 << (i % 8)) != 0).count();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)
            .map_err(Error::io(&path))?;
        // Whole strides, as `pack` writes it. Holes read as zeros and take no space until written.
        let size = records as u64 * stride;
        if file.metadata().map_err(Error::io(&path))?.len() != size {
            file.set_len(size).map_err(Error::io(&path))?;
        }
        if uncached {
            no_cache(&file).map_err(Error::io(&path))?;
        }
        Ok(Arc::new(Sparse {
            file,
            path,
            present_path,
            present_file,
            bits: Mutex::new(bits),
            present: AtomicUsize::new(present),
            hashes,
            fetch,
            rec,
            stride,
            gate: Mutex::new(Gate {
                fetching: vec![false; records],
                ..Gate::default()
            }),
            changed: Condvar::new(),
            fetched_bytes: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            last_error: Mutex::new(None),
            fillers: Mutex::new(Vec::new()),
        }))
    }

    fn has(&self, i: usize) -> bool {
        let bits = lock(&self.bits);
        bits[i / 8] & (1 << (i % 8)) != 0
    }

    /// Records present, bytes fetched and retried fetches.
    #[must_use]
    pub fn progress(&self) -> Progress {
        Progress {
            present: self.present.load(Relaxed),
            total: self.hashes.len(),
            fetched_bytes: self.fetched_bytes.load(Relaxed),
            errors: self.errors.load(Relaxed),
        }
    }

    /// The last error a fill thread hit (it retries that record later).
    #[must_use]
    pub fn last_error(&self) -> Option<String> {
        lock(&self.last_error).clone()
    }

    /// Read record `i` into `buf`: from disk if present, else fetched (and kept on disk).
    fn read(&self, i: usize, buf: &mut [u8]) -> io::Result<()> {
        let mut g = lock(&self.gate);
        loop {
            if self.has(i) {
                drop(g);
                return self.file.read_exact_at(buf, i as u64 * self.stride);
            }
            if !g.fetching[i] {
                break;
            }
            // A fill thread is fetching it: wait for that (or its failure).
            g = self.changed.wait(g).unwrap_or_else(PoisonError::into_inner);
        }
        g.fetching[i] = true;
        g.demand += 1;
        drop(g);
        let r = self.download(i, buf);
        let mut g = lock(&self.gate);
        g.fetching[i] = false;
        g.demand -= 1;
        drop(g);
        self.changed.notify_all();
        r
    }

    /// Fetch record `i` into `buf`, check its hash, and keep it on disk.
    fn download(&self, i: usize, buf: &mut [u8]) -> io::Result<()> {
        let off = i as u64 * self.stride;
        let mut err = None;
        for attempt in 0..TRIES {
            if attempt > 0 {
                self.pause(Duration::from_millis(100 << attempt));
            }
            let got = self.fetch.fetch(off, buf);
            self.fetched_bytes.fetch_add(buf.len() as u64, Relaxed);
            match got {
                Ok(()) if hash(buf) == self.hashes[i] => {
                    // A local write failure (disk full) still serves the record; it is fetched
                    // again next time.
                    let _ = self.keep(i, buf);
                    return Ok(());
                }
                Ok(()) => err = Some(io::Error::new(io::ErrorKind::InvalidData, "hash differs from the manifest")),
                Err(e) => err = Some(e),
            }
            self.errors.fetch_add(1, Relaxed);
        }
        let e = err.unwrap_or_else(|| io::Error::other("no fetch attempted"));
        Err(io::Error::new(e.kind(), format!("fetch record {i}: {e}")))
    }

    /// Write record `i` to disk, then mark it present.
    fn keep(&self, i: usize, buf: &[u8]) -> io::Result<()> {
        self.file.write_all_at(buf, i as u64 * self.stride)?;
        // The data is durable before its bit: a crash can lose a record, never claim a bad one.
        self.file.sync_data()?;
        let mut bits = lock(&self.bits);
        let b = &mut bits[i / 8];
        if *b & (1 << (i % 8)) != 0 {
            return Ok(());
        }
        *b |= 1 << (i % 8);
        if let Some(f) = &self.present_file {
            f.write_all_at(&[*b], (i / 8) as u64)?;
        }
        // Counted under `bits`, after the bitmap is gone, so a full count means complete on disk.
        let n = self.present.load(Relaxed) + 1;
        if n == self.hashes.len() {
            // Complete: the next open reads experts.bin as a plain local file.
            std::fs::remove_file(&self.present_path)?;
        }
        self.present.store(n, Relaxed);
        drop(bits);
        // Through the gate lock, so a `wait_filled` between its check and its wait still wakes.
        drop(lock(&self.gate));
        self.changed.notify_all();
        Ok(())
    }

    /// Start `threads` fill threads that fetch every missing record, in `order` first (record
    /// indices `l * experts + e`, e.g. most used first), then the rest. They hold off while an
    /// engine read is fetching, and end when every record is present or on [`Sparse::stop`].
    /// Threads still running from an earlier start are stopped first.
    pub fn start(self: &Arc<Self>, threads: usize, order: &[usize]) {
        self.stop();
        lock(&self.gate).stop = false;
        let total = self.hashes.len();
        let mut seen = vec![false; total];
        let mut queue: Vec<usize> =
            order.iter().copied().filter(|&i| i < total && !std::mem::replace(&mut seen[i], true)).collect();
        queue.extend((0..total).filter(|&i| !seen[i]));
        let queue = Arc::new(queue);
        let next = Arc::new(AtomicUsize::new(0));
        // A thread that fails to spawn only means fewer fill threads: the engine still fetches
        // the records it needs on demand.
        let handles = (0..threads)
            .filter_map(|i| {
                let (s, queue, next) = (Arc::clone(self), Arc::clone(&queue), Arc::clone(&next));
                std::thread::Builder::new()
                    .name(format!("ironbark-fill-{i}"))
                    .spawn(move || s.fill(&queue, &next))
                    .ok()
            })
            .collect();
        *lock(&self.fillers) = handles;
    }

    fn fill(&self, queue: &[usize], next: &AtomicUsize) {
        let mut buf = vec![0u8; self.rec];
        // Records that failed go round again after a pause (e.g. the network came back).
        let mut failed = Vec::new();
        loop {
            let i = match next.fetch_add(1, Relaxed) {
                n if n < queue.len() => queue[n],
                _ if failed.is_empty() => return,
                _ => {
                    self.pause(Duration::from_secs(5));
                    failed.remove(0)
                }
            };
            // Loaders first: the token being generated waits on them.
            let mut g = self
                .changed
                .wait_while(lock(&self.gate), |g| g.demand > 0 && !g.stop)
                .unwrap_or_else(PoisonError::into_inner);
            if g.stop {
                return;
            }
            if g.fetching[i] || self.has(i) {
                continue;
            }
            g.fetching[i] = true;
            drop(g);
            let r = self.download(i, &mut buf);
            lock(&self.gate).fetching[i] = false;
            self.changed.notify_all();
            if let Err(e) = r {
                *lock(&self.last_error) = Some(e.to_string());
                failed.push(i);
            }
        }
    }

    /// Wait up to `d`, or until stopped.
    fn pause(&self, d: Duration) {
        let _ = self.changed.wait_timeout_while(lock(&self.gate), d, |g| !g.stop);
    }

    /// Wait up to `timeout` for every record to be on disk; whether they are.
    #[must_use]
    pub fn wait_filled(&self, timeout: Duration) -> bool {
        let total = self.hashes.len();
        let g = self
            .changed
            .wait_timeout_while(lock(&self.gate), timeout, |g| !g.stop && self.present.load(Relaxed) < total);
        drop(g);
        self.present.load(Relaxed) == total
    }

    /// Stop the fill threads and wait for them (an engine read still fetches what it needs).
    pub fn stop(&self) {
        lock(&self.gate).stop = true;
        self.changed.notify_all();
        let handles = std::mem::take(&mut *lock(&self.fillers));
        for h in handles {
            let _ = h.join();
        }
    }
}
