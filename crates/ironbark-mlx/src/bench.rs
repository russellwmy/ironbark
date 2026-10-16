//! Measurements shared by the CLI (macOS) and the app library (iOS): greedy generation with
//! timings, expert cache counters, the GPU round trip, and storage reads.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;

use mlx_rs::ops::indexing::argmax_axis;
use mlx_rs::transforms::async_eval;
use mlx_rs::{Array, Dtype};

use ironbark_experts::Layout;

use crate::{Error, Model, Result};

/// A greedy generation with its timings.
#[derive(Debug, Clone)]
pub struct Generation {
    /// Generated tokens.
    pub tokens: Vec<u32>,
    /// Prompt tokens.
    pub prompt: usize,
    /// Prefill seconds (the whole prompt, batched).
    pub prefill_s: f64,
    /// Decode seconds for `tokens.len()` tokens.
    pub decode_s: f64,
    /// MLX active bytes before prefill, after prefill, after decode.
    pub active: [usize; 3],
    /// MLX peak bytes during prefill.
    pub prefill_peak: usize,
    /// Highest process footprint during prefill and decode, sampled (bytes; 0 if unread).
    pub footprint_peak: u64,
    /// Where the decode steps spent time.
    pub timing: crate::StepTiming,
    /// Nanoseconds blocked reading each token back (summed).
    pub read_ns: u64,
}

/// Prefill `ids`, then decode `n` tokens greedily. The expert counters are zeroed after
/// prefill, so [`ExpertStats::of`] reports decode only.
///
/// # Errors
///
/// Fails if `ids` is empty or an MLX op fails.
pub fn generate(m: &mut Model, ids: &[u32], n: usize) -> Result<Generation> {
    if ids.is_empty() {
        return Err(Error::Format("empty prompt".into()));
    }
    m.reset();
    let active = || mlx_rs::memory::active_memory().unwrap_or(0);
    let before = active();
    mlx_rs::memory::reset_peak_memory()?;
    let sampler = crate::device::PeakSampler::start();
    let t = Instant::now();
    let logits = m.prefill(ids, false)?;
    logits.eval()?;
    let prefill_s = t.elapsed().as_secs_f64();
    let (after_prefill, prefill_peak) = (active(), mlx_rs::memory::peak_memory().unwrap_or(0));

    // Pipelined decode: queue step t+1 from the lazy argmax of step t, then read token t
    // while the GPU runs step t+1.
    reset_stats(m);
    let t = Instant::now();
    let pick = |l: &Array| -> Result<Array> { Ok(argmax_axis(l, -1, None)?.reshape(&[1])?.as_dtype(Dtype::Uint32)?) };
    let mut y = pick(&logits)?;
    async_eval([&y])?;
    let mut tokens = Vec::with_capacity(n);
    let mut read = std::time::Duration::ZERO;
    for _ in 0..n {
        let ny = pick(&m.step_lazy(&y)?)?;
        async_eval([&ny])?;
        let t = Instant::now();
        tokens.push(y.item_exact::<u32>());
        read += t.elapsed();
        y = ny;
    }
    let decode_s = t.elapsed().as_secs_f64();
    y.eval()?;
    let footprint_peak = sampler.stop();
    Ok(Generation {
        tokens,
        prompt: ids.len(),
        prefill_s,
        decode_s,
        active: [before, after_prefill, active()],
        prefill_peak,
        footprint_peak,
        timing: m.timing,
        read_ns: read.as_nanos() as u64,
    })
}

/// Zero the step timing and the expert store counters.
pub fn reset_stats(m: &mut Model) {
    m.timing = crate::StepTiming::default();
    if let Some(s) = m.store() {
        let st = &s.stats;
        for a in [
            &st.hits,
            &st.demand_loads,
            &st.late_prefetch,
            &st.prefetch_issued,
            &st.prefetch_used,
            &st.prefetch_dropped,
            &st.bytes_read,
            &st.wait_ns,
            &st.slot_waits,
        ] {
            a.store(0, Relaxed);
        }
    }
}

/// Expert cache counters per decoded token.
#[derive(Debug, Clone, Copy)]
pub struct ExpertStats {
    /// Cache slots per layer.
    pub per_layer: usize,
    /// Experts per layer.
    pub experts: usize,
    /// Cache (slab) size in bytes.
    pub cache_bytes: usize,
    /// Fraction of expert uses found in the cache.
    pub hit: f64,
    /// Demand loads per token.
    pub loads: f64,
    /// Bytes read per token.
    pub read: f64,
    /// Milliseconds waited for loads per token.
    pub wait_ms: f64,
    /// Prefetches issued per token.
    pub prefetch_issued: f64,
    /// Fraction of prefetches used.
    pub prefetch_used: f64,
    /// Prefetches still loading when needed, per token.
    pub late: f64,
    /// Prefetches dropped (no free slot), per token.
    pub dropped: f64,
}

impl ExpertStats {
    /// Counters of `m`'s store over `n` tokens. Resident experts count as a full cache that
    /// always hits; `None` only for a model without experts.
    #[must_use]
    pub fn of(m: &Model, n: usize) -> Option<Self> {
        let Some(s) = m.store() else {
            let resident: Vec<_> = m.layers.iter().filter_map(|l| l.experts.as_ref()).collect();
            let first = resident.first()?;
            let bytes = resident
                .iter()
                .flat_map(|x| [&x.up_gate.w, &x.up_gate.scales, &x.down.w, &x.down.scales])
                .map(Array::nbytes)
                .sum();
            let e = first.up_gate.w.shape()[0] as usize;
            return Some(Self {
                per_layer: e,
                experts: e,
                cache_bytes: bytes,
                hit: 1.0,
                loads: 0.0,
                read: 0.0,
                wait_ms: 0.0,
                prefetch_issued: 0.0,
                prefetch_used: 0.0,
                late: 0.0,
                dropped: 0.0,
            });
        };
        let st = &s.stats;
        let g = |a: &AtomicU64| a.load(Relaxed) as f64;
        let (hits, loads, late) = (g(&st.hits), g(&st.demand_loads), g(&st.late_prefetch));
        let n = n.max(1) as f64;
        Some(Self {
            per_layer: s.per_layer(),
            experts: s.layout.experts,
            cache_bytes: s.slab().len() * s.slab().slot_bytes(),
            // A cache that holds every expert is never asked (decode maps to slots on the GPU).
            hit: if s.per_layer() >= s.layout.experts {
                1.0
            } else {
                hits / (hits + loads + late).max(1.0)
            },
            loads: loads / n,
            read: g(&st.bytes_read) / n,
            wait_ms: g(&st.wait_ns) / 1e6 / n,
            prefetch_issued: g(&st.prefetch_issued) / n,
            prefetch_used: g(&st.prefetch_used) / g(&st.prefetch_issued).max(1.0),
            late: late / n,
            dropped: g(&st.prefetch_dropped) / n,
        })
    }
}

/// Median microseconds of an MLX GPU round trip: one tiny op evaluated and read back, the
/// sync each streamed layer pays once per token.
///
/// # Errors
///
/// Fails if an MLX op fails.
pub fn round_trip_us(iters: usize) -> Result<f64> {
    let mut x = Array::from_slice(&[1u32], &[1]);
    let one = Array::from_slice(&[1u32], &[1]);
    let mut us = Vec::with_capacity(iters);
    for i in 0..iters + 10 {
        let t = Instant::now();
        x = mlx_rs::ops::add(&x, &one)?;
        let _ = x.item_exact::<u32>();
        if i >= 10 {
            us.push(t.elapsed().as_secs_f64() * 1e6);
        }
    }
    us.sort_by(f64::total_cmp);
    Ok(us[us.len() / 2])
}

/// Storage read speed for expert records.
#[derive(Debug, Clone, Copy)]
pub struct ReadProbe {
    /// Bytes per record.
    pub record_bytes: u64,
    /// Median microseconds per record, one reader.
    pub latency_us: f64,
    /// Throughput with one reader, bytes/s.
    pub serial: f64,
    /// Throughput with `threads` readers, bytes/s.
    pub parallel: f64,
    /// Readers in the parallel run.
    pub threads: usize,
}

/// Read random expert records of `dir/experts.bin` uncached: `reads` with one reader, then
/// `reads` per reader with `threads` readers.
///
/// # Errors
///
/// Fails if the packed experts cannot be opened or read.
pub fn read_probe(dir: &Path, reads: usize, threads: usize) -> Result<ReadProbe> {
    let io = |source| Error::Io {
        path: dir.to_path_buf(),
        source,
    };
    let layout = Layout::read(dir).map_err(|e| Error::Format(format!("{}: {e}", dir.display())))?;
    let path = dir.join("experts.bin");
    let file = File::open(&path).map_err(io)?;
    ironbark_experts::layout::no_cache(&file).map_err(io)?;
    let records = (layout.layers * layout.experts) as u64;
    let len = layout.record_bytes;
    // Distinct records per reader (xorshift), so nothing comes from a cache.
    let picks = |seed: u64| {
        let mut s = seed | 1;
        (0..reads)
            .map(move |_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % records) * layout.stride
            })
            .collect::<Vec<u64>>()
    };
    let read_all = |offsets: &[u64]| -> std::io::Result<Vec<f64>> {
        let mut buf = vec![0u8; len as usize];
        let mut us = Vec::with_capacity(offsets.len());
        for &o in offsets {
            let t = Instant::now();
            file.read_exact_at(&mut buf, o)?;
            us.push(t.elapsed().as_secs_f64() * 1e6);
        }
        Ok(us)
    };

    let t = Instant::now();
    let mut us = read_all(&picks(0x9e37_79b9_7f4a_7c15)).map_err(io)?;
    let serial = (reads as u64 * len) as f64 / t.elapsed().as_secs_f64();
    us.sort_by(f64::total_cmp);

    let t = Instant::now();
    std::thread::scope(|sc| {
        let handles: Vec<_> = (0..threads)
            .map(|i| {
                let offsets = picks(0x2545_f491_4f6c_dd1d ^ ((i as u64 + 1) * 0x1000_0001));
                sc.spawn(move || read_all(&offsets))
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|h| h.join().unwrap_or_else(|_| Err(std::io::Error::other("reader panicked"))).map(drop))
    })
    .map_err(io)?;
    let parallel = (threads * reads) as f64 * len as f64 / t.elapsed().as_secs_f64();
    Ok(ReadProbe {
        record_bytes: len,
        latency_us: us[us.len() / 2],
        serial,
        parallel,
        threads,
    })
}
