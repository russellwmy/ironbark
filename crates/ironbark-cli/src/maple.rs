//! Greedy decode and perplexity of Maple on the GPU (MLX).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow};
use mlx_rs::Array;
use mlx_rs::ops::indexing::argmax_axis;

use ironbark_fetch::{FileFetch, Sparse};
use ironbark_mlx::Model;
use ironbark_mlx::bench::{ExpertStats, reset_stats};
use ironbark_mlx::device::{Hardware, footprint};
use ironbark_mlx::plan::{self, Plan};

/// Where experts live: resident, or streamed from a packed dir with a per-layer cache.
#[derive(Debug, Clone)]
pub(crate) struct Experts {
    /// Packed experts dir (`ironbark pack`); `None` = all experts resident.
    pub(crate) dir: Option<PathBuf>,
    /// Cache slots per layer (of 256); `None` = what fits `budget`.
    pub(crate) cache: Option<usize>,
    /// Footprint budget in bytes; `None` = from the device.
    pub(crate) budget: Option<u64>,
    /// Loader threads.
    pub(crate) loaders: usize,
    /// Layers ahead to predict and prefetch.
    pub(crate) lookahead: usize,
    /// Fill `dir`'s experts.bin and embed.bin on demand from a remote copy.
    pub(crate) fetch: Option<Remote>,
}

/// A complete pack standing in for a server.
#[derive(Debug, Clone)]
pub(crate) struct Remote {
    /// Complete packed dir.
    pub(crate) from: PathBuf,
    /// Delay per fetch.
    pub(crate) delay: Duration,
    /// Background fill threads for experts.bin.
    pub(crate) fill: usize,
    /// Background fill threads for embed.bin.
    pub(crate) embed_fill: usize,
}

/// The sparse experts.bin and embed.bin of a fetching run.
struct Fetching {
    experts: Arc<Sparse>,
    embed: Arc<Sparse>,
    opened: Instant,
}

impl Fetching {
    fn report(&self) {
        let (x, e) = (self.experts.progress(), self.embed.progress());
        println!(
            "fetch: experts {}/{} records ({:.0} MB), embed {}/{} chunks ({:.1} MB), {:.1}s after open",
            x.present,
            x.total,
            x.fetched_bytes as f64 / 1e6,
            e.present,
            e.total,
            e.fetched_bytes as f64 / 1e6,
            self.opened.elapsed().as_secs_f64()
        );
    }
}

fn load(dir: &Path, ex: &Experts) -> Result<Model> {
    Ok(load_fetching(dir, ex)?.0)
}

fn load_fetching(dir: &Path, ex: &Experts) -> Result<(Model, Option<Fetching>)> {
    let mut fetching = None;
    let mut m = match &ex.dir {
        Some(e) => {
            let budget = ex.budget.unwrap_or_else(|| plan::budget(&Hardware::detect()));
            let p = Plan::new(e, budget, plan::CONTEXT)?;
            let slots = ex.cache.unwrap_or(p.slots);
            println!(
                "budget {:.2} GB: {slots} experts per layer (fits {}), expected peak {:.2} GB",
                budget as f64 / 1e9,
                p.slots,
                p.peak as f64 / 1e9
            );
            mlx_rs::memory::set_cache_limit(p.buffer_cache as usize)?;
            match &ex.fetch {
                Some(r) => {
                    let opened = Instant::now();
                    let remote = |name: &str| -> Result<Arc<FileFetch>> {
                        Ok(Arc::new(FileFetch::open(&r.from.join(name), r.delay)?))
                    };
                    let experts = Sparse::open(e, remote("experts.bin")?)?;
                    let embed = Sparse::open_embed(e, remote("embed.bin")?)?;
                    let mut m = Model::load_records(dir, slots, ex.loaders, experts.clone())?;
                    m.set_embed_chunks(Some(embed.clone()));
                    if r.embed_fill > 0 {
                        embed.start(r.embed_fill, &[]);
                    }
                    if r.fill > 0 {
                        experts.start(r.fill, &[]);
                    }
                    fetching = Some(Fetching { experts, embed, opened });
                    m
                }
                None => Model::load_streaming(dir, e, slots, ex.loaders)?,
            }
        }
        None => Model::load(dir)?,
    };
    m.lookahead = ex.lookahead;
    Ok((m, fetching))
}

/// Expert cache report for `n` decoded tokens.
fn report_stats(m: &Model, n: usize) {
    let Some(s) = ExpertStats::of(m, n) else { return };
    println!(
        "experts: cache {}/{} per layer = {:.2} GB  hit {:.1}%  loads {:.1}/tok  read {:.1} MB/tok  load wait {:.2} ms/tok",
        s.per_layer,
        s.experts,
        s.cache_bytes as f64 / 1e9,
        100.0 * s.hit,
        s.loads,
        s.read / 1e6,
        s.wait_ms,
    );
    if s.prefetch_issued > 0.0 {
        println!(
            "prefetch: issued {:.1}/tok  used {:.0}%  late {:.1}/tok  dropped {:.1}/tok",
            s.prefetch_issued,
            100.0 * s.prefetch_used,
            s.late,
            s.dropped,
        );
    }
}

/// Greedy next token from logits `[1, 1, vocab]`.
fn next(logits: &Array) -> Result<u32> {
    Ok(argmax_axis(logits, -1, None)?.item_exact::<u32>())
}

/// The `n` largest logits of `[1, 1, vocab]` as (token, logit).
fn top_n(logits: &Array, n: usize) -> Result<Vec<(u32, f32)>> {
    let l = logits.as_dtype(mlx_rs::Dtype::Float32)?;
    let v: Vec<f32> = l.as_slice::<f32>().to_vec();
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| v[*b].total_cmp(&v[*a]));
    Ok(idx.into_iter().take(n).map(|i| (i as u32, v[i])).collect())
}

pub(crate) fn run(dir: &Path, ex: &Experts, prompt: &str, n_gen: usize, top: usize) -> Result<()> {
    let t = Instant::now();
    let (mut m, fetching) = load_fetching(dir, ex)?;
    let load_s = t.elapsed().as_secs_f64();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
    let ids = tok.encode(prompt, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
    anyhow::ensure!(!ids.is_empty(), "empty prompt");

    let t = Instant::now();
    let mut logits = m.prefill(&ids, false)?;
    logits.eval()?;
    let prefill_s = t.elapsed().as_secs_f64();

    let mut out = Vec::with_capacity(n_gen);
    for i in 0..top.min(n_gen) {
        println!("step {i} top5 {:?}", top_n(&logits, 5)?);
        let next = next(&logits)?;
        out.push(next);
        logits = m.step(next)?;
    }
    // Pipelined decode: queue step t+1 from the lazy argmax of step t, then read token t
    // while the GPU runs step t+1.
    reset_stats(&mut m);
    let t = Instant::now();
    let mut y = argmax_axis(&logits, -1, None)?.reshape(&[1])?.as_dtype(mlx_rs::Dtype::Uint32)?;
    mlx_rs::transforms::async_eval([&y])?;
    for _ in out.len()..n_gen {
        let nl = m.step_lazy(&y)?;
        let ny = argmax_axis(&nl, -1, None)?.reshape(&[1])?.as_dtype(mlx_rs::Dtype::Uint32)?;
        mlx_rs::transforms::async_eval([&ny])?;
        out.push(y.item_exact::<u32>());
        y = ny;
    }
    let decoded = n_gen - top.min(n_gen);
    let decode_s = t.elapsed().as_secs_f64();

    println!("{}", tok.decode(&out, true).map_err(|e| anyhow!("{e}"))?);
    println!(
        "load {load_s:.1}s  prefill {} tok in {prefill_s:.2}s ({:.1} tok/s)  decode {decoded} tok: {:.1} tok/s",
        ids.len(),
        ids.len() as f64 / prefill_s,
        decoded as f64 / decode_s
    );
    report_stats(&m, decoded);
    if let Some(f) = &fetching {
        f.report();
    }
    println!(
        "MLX memory: active {:.2} GB, peak {:.2} GB",
        mlx_rs::memory::active_memory()? as f64 / 1e9,
        mlx_rs::memory::peak_memory()? as f64 / 1e9
    );
    if let Some(f) = footprint() {
        println!("footprint: {:.2} GB, peak {:.2} GB", f.now as f64 / 1e9, f.peak as f64 / 1e9);
    }
    Ok(())
}

/// Negative log-likelihood of `next` under logits `v`.
fn nll(v: &[f32], next: u32) -> f64 {
    let mx = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let lse = f64::from(mx) + v.iter().map(|x| f64::from(x - mx).exp()).sum::<f64>().ln();
    lse - f64::from(v[next as usize])
}

/// Perplexity over the first `n` tokens of `text` (token by token, or one batched forward with
/// `batch`; the first half is context, the second half is scored), for comparing with a
/// reference implementation.
pub(crate) fn ppl(dir: &Path, ex: &Experts, text: &Path, n: usize, batch: bool) -> Result<()> {
    let mut m = load(dir, ex)?;
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
    let raw = std::fs::read_to_string(text)?;
    let ids = tok.encode(raw, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
    anyhow::ensure!(ids.len() > n, "text has only {} tokens", ids.len());
    if batch {
        // One batched forward over the first n - 1 tokens; logits of every position.
        let logits = m.prefill(&ids[..n - 1], true)?.as_dtype(mlx_rs::Dtype::Float32)?;
        let vocab = m.cfg.vocab;
        let all = logits.as_slice::<f32>();
        let (mut total, mut count) = (0f64, 0usize);
        for t in n / 2..n - 1 {
            total += nll(&all[t * vocab..(t + 1) * vocab], ids[t + 1]);
            count += 1;
        }
        println!("ppl {:.4} over {count} tokens (batched)", (total / count as f64).exp());
        return Ok(());
    }
    let (mut total, mut count) = (0f64, 0usize);
    for t in 0..n - 1 {
        if t == n / 2 {
            // The first half is context (and warms the cache); report the scored half.
            reset_stats(&mut m);
        }
        let l = m.step(ids[t])?;
        if t >= n / 2 {
            let v = l.as_dtype(mlx_rs::Dtype::Float32)?;
            total += nll(v.as_slice::<f32>(), ids[t + 1]);
            count += 1;
        }
    }
    println!("ppl {:.4} over {count} tokens", (total / count as f64).exp());
    report_stats(&m, n - 1 - n / 2);
    Ok(())
}

pub(crate) fn profile(dir: &Path, ex: &Experts) -> Result<()> {
    let m = load(dir, ex)?;
    let p = m.profile(200)?;
    for (part, us, n) in &p.parts {
        println!("{part:16} {us:8.1} us x {n:2} = {:7.2} ms", us * *n as f64 / 1e3);
    }
    println!("token (GPU only) {:.2} ms = {:.0} tok/s", p.token_us / 1e3, 1e6 / p.token_us);
    Ok(())
}
