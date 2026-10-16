//! Event-driven expert cache over a [`Slab`].
//!
//! ```text
//!   compute thread                          loader threads (N)
//!   request(l,e) ──Demand event──▶ ┐        sleep on `jobs_ready`; take Demand first, else Prefetch
//!   prefetch(l,e) ─Prefetch event─▶ ┼──▶     read the record into the slot (`Records`: pread, or another source)
//!   wait(l,e) ◀── loaded[slot] ─────┘        mark Ready, notify that slot's waiters only
//!     └─▶ ExpertRef (pins the slot; unpins on drop)
//! ```
//!
//! Slot states (changed only under `inner`):
//!
//! ```text
//!   Free ──claim──▶ Loading ──loader done──▶ Ready ──evict (pins == 0)──▶ Loading (new expert)
//!   (`pins` is atomic: pinned under `inner`, released lock-free by `ExpertRef::drop`)
//!                    one writer                any number of readers (pins)
//! ```
//!
//! Each layer owns `per_layer` slots (a cyclic layer-0..L access pattern defeats one global
//! LRU). The budget is hard: when every slot of a layer is pinned or loading, a demand load
//! waits in [`Store::wait`] until one is released; a prefetch is dropped.
//!
//! Lock order: `inner`, then `jobs`. Never hold either during a read.

use std::collections::VecDeque;
use std::fmt;
use std::ops::Deref;
use std::path::Path;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed, Ordering::SeqCst};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Instant;

use crate::error::{Error, Result};

use crate::layout::Layout;
use crate::records::Records;
use crate::slab::Slab;

/// Load job: (layer, expert, slot).
type Job = (usize, usize, usize);

/// Two-lane job queue. Loaders sleep on `ready` and take demand jobs before prefetch jobs;
/// no lock is held while a loader reads.
#[derive(Debug, Default)]
struct Lanes {
    demand: VecDeque<Job>,
    prefetch: VecDeque<Job>,
    /// Set by [`Store::close`]: loaders exit.
    closed: bool,
}

/// Lock without poisoning: a panic elsewhere already reports the bug, and the cache
/// state stays valid between statements, so later users can keep going.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Free,
    Loading { prefetch: bool },
    Ready { fresh: Fresh },
}

/// Whether a `Ready` slot has been used since it was loaded, and by which kind of load.
/// The first use of a fresh demand load is a miss even if the load finished before `wait`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fresh {
    No,
    Demand,
    Prefetch,
}

#[derive(Debug, Clone, Copy)]
struct Meta {
    expert: usize,
    state: State,
    last: u64,
}

struct Inner {
    /// Slot metadata; layer `l` owns `l * per_layer .. (l + 1) * per_layer`.
    meta: Vec<Meta>,
    /// `(l * experts + e)` -> slot holding or loading that expert.
    index: Vec<Option<usize>>,
    tick: u64,
    /// The first failed load: every later wait returns it.
    failed: Option<String>,
}

/// Counters for the benchmark report. Reset by storing 0.
#[derive(Debug, Default)]
pub struct Stats {
    /// Lookups that found the expert ready.
    pub hits: AtomicU64,
    /// Loads started by a router decision.
    pub demand_loads: AtomicU64,
    /// Needed while its prefetch was still loading.
    pub late_prefetch: AtomicU64,
    /// Prefetch loads started.
    pub prefetch_issued: AtomicU64,
    /// Prefetched experts that were later used.
    pub prefetch_used: AtomicU64,
    /// Prefetches skipped because no slot was free.
    pub prefetch_dropped: AtomicU64,
    /// Demand loads that had to wait for a slot to be released.
    pub slot_waits: AtomicU64,
    /// Bytes read from storage.
    pub bytes_read: AtomicU64,
    /// Time compute spent in `wait`.
    pub wait_ns: AtomicU64,
}

/// Expert cache fed by a pool of loader threads.
pub struct Store {
    /// Shape of the expert file.
    pub layout: Layout,
    per_layer: usize,
    records: Arc<dyn Records>,
    slab: Slab,
    inner: Mutex<Inner>,
    /// Per slot: signalled when its load finishes (wakes only the tasks waiting for it).
    loaded: Vec<Condvar>,
    /// Per layer: signalled when one of its slots may have become claimable.
    released: Vec<Condvar>,
    /// Readers per slot. Incremented only under `inner` (in `wait`); decremented without a
    /// lock when an [`ExpertRef`] drops, so releasing an expert is lock-free.
    pins: Vec<AtomicU32>,
    /// Compute tasks blocked because their layer had no claimable slot.
    slot_waiters: AtomicUsize,
    jobs: Mutex<Lanes>,
    jobs_ready: Condvar,
    /// Loader threads, joined by [`Store::close`].
    loaders: Mutex<Vec<JoinHandle<()>>>,
    /// Counters for the benchmark report.
    pub stats: Stats,
}

impl fmt::Debug for Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Store")
            .field("layout", &self.layout)
            .field("per_layer", &self.per_layer)
            .field("slab_bytes", &(self.slab.len() * self.slab.slot_bytes()))
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

/// A pinned expert's record bytes. The slot stays pinned (not evictable) until this is dropped.
pub struct ExpertRef<'a> {
    store: &'a Store,
    slot: usize,
    data: &'a [u8],
}

impl ExpertRef<'_> {
    /// Slab slot holding the expert (stays valid, and unwritten, while this ref lives).
    #[must_use]
    pub fn slot(&self) -> usize {
        self.slot
    }
}

impl Deref for ExpertRef<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.data
    }
}

impl fmt::Debug for ExpertRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExpertRef").field("slot", &self.slot).field("len", &self.data.len()).finish()
    }
}

impl Drop for ExpertRef<'_> {
    fn drop(&mut self) {
        self.store.unpin(self.slot);
    }
}

/// A pin that owns its store handle, so it can outlive the borrow of an [`ExpertRef`] (e.g. be
/// kept until GPU work queued on the slot has finished). Unpins on drop.
#[derive(Debug)]
pub struct Pin {
    store: Arc<Store>,
    slot: usize,
}

impl Pin {
    /// Pin the slot of `r` again, owned. The new pin is independent of `r`.
    #[must_use]
    pub fn new(store: &Arc<Store>, r: &ExpertRef<'_>) -> Self {
        store.pins[r.slot].fetch_add(1, SeqCst);
        Self {
            store: Arc::clone(store),
            slot: r.slot,
        }
    }
}

impl Drop for Pin {
    fn drop(&mut self) {
        self.store.unpin(self.slot);
    }
}

impl Store {
    /// Release one pin of `slot`.
    fn unpin(&self, slot: usize) {
        // SeqCst on both `pins` and `slot_waiters` (here and in `wait`) gives one total order:
        // either we see the waiter's increment and wake it, or its re-check of `claim` sees
        // our decrement. Release semantics also order our reads of the slot before a reuse.
        let last = self.pins[slot].fetch_sub(1, SeqCst) == 1;
        if last && self.slot_waiters.load(SeqCst) > 0 {
            // Taking the lock guarantees the waiter is inside its wait, so the notify is not lost.
            drop(lock(&self.inner));
            self.released[slot / self.per_layer].notify_all();
        }
    }

    /// Open `experts.bin` in `dir` with `per_layer` cache slots per layer and `loaders` threads.
    /// When `per_layer` covers every expert, all experts are preloaded (no streaming).
    ///
    /// # Errors
    ///
    /// Fails if `experts.json` / `experts.bin` cannot be read, or the record stride is not
    /// whole pages.
    pub fn open(dir: &Path, per_layer: usize, loaders: usize) -> Result<Arc<Self>> {
        Self::open_with(dir, per_layer, loaders, crate::records::local(dir)?)
    }

    /// Like [`Store::open`], with records read from `records` instead of `dir/experts.bin`
    /// (`dir` still gives `experts.json`).
    ///
    /// # Errors
    ///
    /// Fails if `experts.json` cannot be read, or the record stride is not whole pages.
    pub fn open_with(dir: &Path, per_layer: usize, loaders: usize, records: Arc<dyn Records>) -> Result<Arc<Self>> {
        let layout = Layout::read(dir)?;
        Self::start(layout, &dir.join("experts.bin"), records, per_layer, loaders)
    }

    fn start(
        layout: Layout,
        path: &Path,
        records: Arc<dyn Records>,
        per_layer: usize,
        loaders: usize,
    ) -> Result<Arc<Self>> {
        let stride = usize::try_from(layout.stride).map_err(|_| Error::format(path, "stride too large"))?;
        if !stride.is_multiple_of(crate::slab::PAGE_BYTES) || (stride as u64) < layout.record_bytes {
            return Err(Error::format(path, "stride must be whole pages and fit a record (re-run pack)"));
        }
        let per_layer = per_layer.min(layout.experts);
        let free = Meta {
            expert: 0,
            state: State::Free,
            last: 0,
        };
        let loaded = (0..layout.layers * per_layer).map(|_| Condvar::new()).collect();
        let released = (0..layout.layers).map(|_| Condvar::new()).collect();
        let store = Arc::new(Store {
            slab: Slab::new(layout.layers * per_layer, stride),
            inner: Mutex::new(Inner {
                meta: vec![free; layout.layers * per_layer],
                index: vec![None; layout.layers * layout.experts],
                tick: 0,
                failed: None,
            }),
            pins: (0..layout.layers * per_layer).map(|_| AtomicU32::new(0)).collect(),
            slot_waiters: AtomicUsize::new(0),
            layout,
            per_layer,
            records,
            loaded,
            released,
            jobs: Mutex::new(Lanes::default()),
            jobs_ready: Condvar::new(),
            loaders: Mutex::new(Vec::new()),
            stats: Stats::default(),
        });
        if per_layer == store.layout.experts {
            store.preload().map_err(Error::io(path))?;
        }
        let handles = (0..loaders)
            .map(|i| {
                let s = store.clone();
                std::thread::Builder::new().name(format!("ironbark-loader-{i}")).spawn(move || s.loader())
            })
            .collect::<std::io::Result<Vec<_>>>()
            .map_err(Error::io(path))?;
        *lock(&store.loaders) = handles;
        Ok(store)
    }

    /// Stop the loader threads, which hold the store alive, and wait for them; the store is
    /// freed when the last other handle drops. Queued loads are dropped, so call it only when
    /// nothing waits on the store.
    pub fn close(&self) {
        lock(&self.jobs).closed = true;
        self.jobs_ready.notify_all();
        let handles = std::mem::take(&mut *lock(&self.loaders));
        for h in handles {
            // A loader that panicked already reported it; the store closes either way.
            let _ = h.join();
        }
    }

    /// Read expert (l, e) into `slot`.
    ///
    /// # Safety
    ///
    /// The caller must own `slot` exclusively (state `Loading` claimed by this job, or
    /// single-threaded preload).
    unsafe fn read_into(&self, l: usize, e: usize, slot: usize) -> std::io::Result<()> {
        let n = self.layout.record_bytes as usize;
        // SAFETY: the caller owns `slot` exclusively.
        let buf = unsafe { self.slab.slot_mut(slot) };
        let i = l * self.layout.experts + e;
        self.records.read(i, &mut buf[..n])?;
        self.stats.bytes_read.fetch_add(n as u64, Relaxed);
        Ok(())
    }

    fn preload(&self) -> std::io::Result<()> {
        let mut g = lock(&self.inner);
        for l in 0..self.layout.layers {
            for e in 0..self.layout.experts {
                let slot = l * self.per_layer + e;
                // SAFETY: no loader threads exist yet and we hold `inner`: exclusive access.
                unsafe { self.read_into(l, e, slot)? };
                g.meta[slot] = Meta {
                    expert: e,
                    state: State::Ready { fresh: Fresh::No },
                    last: 0,
                };
                g.index[l * self.layout.experts + e] = Some(slot);
            }
        }
        self.stats.bytes_read.store(0, Relaxed);
        Ok(())
    }

    fn loader(&self) {
        loop {
            let (l, e, slot) = {
                let mut q = lock(&self.jobs);
                loop {
                    if q.closed {
                        return;
                    }
                    if let Some(j) = q.demand.pop_front().or_else(|| q.prefetch.pop_front()) {
                        break j;
                    }
                    q = self.jobs_ready.wait(q).unwrap_or_else(PoisonError::into_inner);
                }
            };
            // SAFETY: `claim` set this slot to `Loading` for this job; no reader or other
            // writer can touch it until we mark it `Ready` below.
            let read = unsafe { self.read_into(l, e, slot) };
            let mut g = lock(&self.inner);
            if let Err(err) = read {
                // Free the slot and fail every waiter (a waiting task cannot see this thread).
                g.failed.get_or_insert_with(|| format!("expert {e} of layer {l}: {err}"));
                g.index[l * self.layout.experts + e] = None;
                g.meta[slot].state = State::Free;
                drop(g);
                self.loaded[slot].notify_all();
                self.released.iter().for_each(Condvar::notify_all);
                continue;
            }
            g.tick += 1;
            let last = g.tick;
            let m = &mut g.meta[slot];
            let State::Loading { prefetch } = m.state else {
                unreachable!("slot {slot} changed state while loading");
            };
            m.state = State::Ready {
                fresh: if prefetch { Fresh::Prefetch } else { Fresh::Demand },
            };
            m.last = last;
            drop(g);
            self.loaded[slot].notify_all();
            // A finished load is also a claimable slot for a task blocked on a full layer.
            if self.slot_waiters.load(SeqCst) > 0 {
                self.released[l].notify_all();
            }
        }
    }

    /// Claim a slot of layer `l` for expert `e` and mark it `Loading`: a free slot first,
    /// else the least recently used unpinned `Ready` slot. `None` if all are pinned or loading.
    fn claim(&self, g: &mut Inner, l: usize, e: usize, prefetch: bool) -> Option<usize> {
        let range = l * self.per_layer..(l + 1) * self.per_layer;
        let slot = range.clone().find(|&s| g.meta[s].state == State::Free).or_else(|| {
            range
                .filter(|&s| matches!(g.meta[s].state, State::Ready { .. }) && self.pins[s].load(SeqCst) == 0)
                .min_by_key(|&s| g.meta[s].last)
        })?;
        if g.meta[slot].state != State::Free {
            let old = g.meta[slot].expert;
            g.index[l * self.layout.experts + old] = None;
        }
        g.meta[slot] = Meta {
            expert: e,
            state: State::Loading { prefetch },
            last: 0,
        };
        g.index[l * self.layout.experts + e] = Some(slot);
        Some(slot)
    }

    /// Queue a load and wake one loader.
    fn send(&self, job: Job, prefetch: bool) {
        let mut q = lock(&self.jobs);
        if prefetch {
            q.prefetch.push_back(job);
        } else {
            q.demand.push_back(job);
        }
        drop(q);
        self.jobs_ready.notify_one();
    }

    /// Event: the router of layer `l` chose expert `e`. Starts a load if a slot is
    /// available; never blocks. If none is, [`Store::wait`] starts it later.
    pub fn request(&self, l: usize, e: usize) {
        let mut g = lock(&self.inner);
        if g.index[l * self.layout.experts + e].is_some() {
            return;
        }
        let Some(slot) = self.claim(&mut g, l, e, false) else {
            return;
        };
        self.stats.demand_loads.fetch_add(1, Relaxed);
        drop(g);
        self.send((l, e, slot), false);
    }

    /// Event: a predictor expects layer `l` to need expert `e` soon. Dropped if no slot is free.
    pub fn prefetch(&self, l: usize, e: usize) {
        let mut g = lock(&self.inner);
        if g.index[l * self.layout.experts + e].is_some() {
            return;
        }
        let Some(slot) = self.claim(&mut g, l, e, true) else {
            self.stats.prefetch_dropped.fetch_add(1, Relaxed);
            return;
        };
        self.stats.prefetch_issued.fetch_add(1, Relaxed);
        drop(g);
        self.send((l, e, slot), true);
    }

    /// Block until expert (l, e) is resident (call `request` first) and pin it.
    ///
    /// # Errors
    ///
    /// [`Error::Load`] once any expert failed to load (e.g. a read error): the store stops
    /// serving, since the files are no longer trustworthy.
    pub fn wait(&self, l: usize, e: usize) -> Result<ExpertRef<'_>> {
        let t0 = Instant::now();
        let key = l * self.layout.experts + e;
        let mut g = lock(&self.inner);
        let mut first = true;
        loop {
            if let Some(msg) = &g.failed {
                return Err(Error::Load(msg.clone()));
            }
            let Some(slot) = g.index[key] else {
                first = false;
                let mut claimed = self.claim(&mut g, l, e, false);
                if claimed.is_none() {
                    // Every slot of the layer is pinned or loading. Register, then re-check
                    // (see `Store::unpin` for why this cannot miss a release).
                    self.slot_waiters.fetch_add(1, SeqCst);
                    claimed = self.claim(&mut g, l, e, false);
                    if claimed.is_none() {
                        self.stats.slot_waits.fetch_add(1, Relaxed);
                        g = self.released[l].wait(g).unwrap_or_else(PoisonError::into_inner);
                    }
                    self.slot_waiters.fetch_sub(1, SeqCst);
                }
                if let Some(slot) = claimed {
                    self.stats.demand_loads.fetch_add(1, Relaxed);
                    drop(g);
                    self.send((l, e, slot), false);
                    g = lock(&self.inner);
                }
                continue;
            };
            match g.meta[slot].state {
                State::Ready { fresh } => {
                    g.tick += 1;
                    let tick = g.tick;
                    let m = &mut g.meta[slot];
                    m.last = tick;
                    m.state = State::Ready { fresh: Fresh::No };
                    self.pins[slot].fetch_add(1, SeqCst);
                    if fresh == Fresh::Prefetch {
                        self.stats.prefetch_used.fetch_add(1, Relaxed);
                    }
                    if first && fresh != Fresh::Demand {
                        self.stats.hits.fetch_add(1, Relaxed);
                    }
                    drop(g);
                    // SAFETY: the slot is `Ready` and now pinned, so no loader writes it
                    // until the returned `ExpertRef` is dropped.
                    let data = unsafe { self.slab.slot(slot, self.layout.record_bytes as usize) };
                    self.stats.wait_ns.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
                    return Ok(ExpertRef {
                        store: self,
                        slot,
                        data,
                    });
                }
                State::Loading { prefetch } => {
                    if first && prefetch {
                        self.stats.late_prefetch.fetch_add(1, Relaxed);
                    }
                    first = false;
                    g = self.loaded[slot].wait(g).unwrap_or_else(PoisonError::into_inner);
                }
                State::Free => unreachable!("index points at a free slot"),
            }
        }
    }

    /// The slot memory, e.g. to wrap as one GPU buffer: slot `l * per_layer() + i` belongs to
    /// layer `l`. A slot is written only while no [`ExpertRef`] pins it.
    #[must_use]
    pub fn slab(&self) -> &Slab {
        &self.slab
    }

    /// Cache slots per layer.
    #[must_use]
    pub fn per_layer(&self) -> usize {
        self.per_layer
    }

    /// Experts currently held or loading.
    pub fn resident(&self) -> usize {
        lock(&self.inner).meta.iter().filter(|m| m.state != State::Free).count()
    }
}
