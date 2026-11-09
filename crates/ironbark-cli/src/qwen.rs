//! Greedy decode and perplexity of Qwen3.5 / 3.6 MoE on the GPU (MLX): an MLX checkpoint
//! (every weight resident) or a pack (ternary experts cached or streamed from `experts.bin`).

use std::path::Path;
use std::time::Instant;

use anyhow::{Result, anyhow};
use ironbark_mlx::ops::indexing::argmax_axis;

use ironbark_engine::qwen::Model;
use ironbark_engine::qwen::vision::{self, Vision};

use crate::maple::Experts;
use crate::util::{memory, top_n};

/// Room for everything but the expert cache: slim weights plus caches, runtime and prefill.
const OVERHEAD: u64 = 1 << 30;

/// The model in `dir`: a pack (it has `experts.json`) with its experts cached per `ex`
/// (`--experts` names the pack to stream from), or an MLX checkpoint.
fn load(dir: &Path, ex: &Experts) -> Result<Model> {
    let pack = ex.dir.clone().or_else(|| dir.join("experts.json").exists().then(|| dir.to_path_buf()));
    let mut m = match &pack {
        Some(e) => {
            let layout = ironbark_experts::Layout::read(e)?;
            let slots = match (ex.cache, ex.budget) {
                (Some(c), _) => c,
                (None, Some(b)) => {
                    let weights = std::fs::metadata(e.join("model.safetensors")).map_or(0, |m| m.len());
                    (b.saturating_sub(weights + OVERHEAD) / (layout.layers as u64 * layout.stride)) as usize
                }
                (None, None) => layout.experts,
            }
            .clamp(8, layout.experts);
            println!(
                "{slots} experts per layer: cache {:.2} GB",
                (slots as u64 * layout.layers as u64 * layout.stride) as f64 / 1e9
            );
            Model::load_streaming(dir, e, slots, ex.loaders)?
        }
        None => Model::load(dir)?,
    };
    m.lookahead = ex.lookahead;
    m.lookahead_k = ex.lookahead_k;
    m.cfg.dynamic.drop_below = ex.drop_below;
    m.cfg.dynamic.min_k = ex.min_k;
    if ex.route_stats {
        m.log_routes();
    }
    Ok(m)
}

/// Expert cache counters and readback time over `n` decoded tokens.
fn report(m: &Model, n: usize) {
    if let Some(s) = m.store() {
        let st = &s.stats;
        let get =
            |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed) as f64 / n.max(1) as f64;
        println!(
            "experts/tok: hits {:.1}  demand loads {:.1}  late prefetch {:.1}  prefetch issued {:.1}  sync {:.1} ms/tok",
            get(&st.hits),
            get(&st.demand_loads),
            get(&st.late_prefetch),
            get(&st.prefetch_issued),
            m.sync_ns as f64 / 1e6 / n.max(1) as f64
        );
        if m.route_log.is_some() {
            println!(
                "experts/tok: prefetch used {:.1}  dropped {:.1}  read {:.1} MB  load wait {:.1} ms/tok",
                get(&st.prefetch_used),
                get(&st.prefetch_dropped),
                get(&st.bytes_read) / 1e6,
                get(&st.wait_ns) / 1e6
            );
        }
    }
}

/// The chat template's prompt for one user message, reasoning off (`enable_thinking=False`).
fn chat(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// Greedy decode of `n_gen` tokens after `prompt` (in the chat format with `as_chat`), with
/// the top-5 logits of the first `top` steps.
pub(crate) fn run(dir: &Path, ex: &Experts, prompt: &str, n_gen: usize, top: usize, as_chat: bool) -> Result<()> {
    let t = Instant::now();
    let mut m = load(dir, ex)?;
    let load_s = t.elapsed().as_secs_f64();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
    let text = if as_chat { chat(prompt) } else { prompt.to_owned() };
    let ids = tok.encode(text, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
    anyhow::ensure!(!ids.is_empty(), "empty prompt");
    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| tok.token_to_id(t)).collect();

    let t = Instant::now();
    let mut logits = m.prefill(&ids, false)?;
    logits.eval()?;
    let prefill_s = t.elapsed().as_secs_f64();

    let mut out = Vec::with_capacity(n_gen);
    for i in 0..top.min(n_gen) {
        println!("step {i} top5 {:?}", top_n(&logits, 5)?);
        let next = argmax_axis(&logits, -1, None)?.item_exact::<u32>();
        out.push(next);
        logits = m.step(next)?;
    }
    m.sync_ns = 0;
    if m.route_log.is_some()
        && let Some(s) = m.store()
    {
        // Count decode only: the cache filled by prefill is not a decode cost.
        let st = &s.stats;
        for a in [
            &st.hits,
            &st.demand_loads,
            &st.late_prefetch,
            &st.prefetch_issued,
            &st.prefetch_used,
            &st.prefetch_dropped,
            &st.slot_waits,
            &st.bytes_read,
            &st.wait_ns,
        ] {
            a.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let t = Instant::now();
    let mut y = argmax_axis(&logits, -1, None)?.reshape(&[1])?.as_dtype(ironbark_mlx::Dtype::Uint32)?;
    ironbark_mlx::transforms::async_eval([&y])?;
    while out.len() < n_gen {
        let nl = m.step_lazy(&y)?;
        let ny = argmax_axis(&nl, -1, None)?.reshape(&[1])?.as_dtype(ironbark_mlx::Dtype::Uint32)?;
        let mut queued = m.state_arrays();
        queued.push(&ny);
        ironbark_mlx::transforms::async_eval(queued)?;
        let t = y.item_exact::<u32>();
        out.push(t);
        y = ny;
        if as_chat && eos.contains(&t) {
            break;
        }
    }
    let decoded = out.len() - top.min(n_gen);
    let decode_s = t.elapsed().as_secs_f64();
    println!("{}", tok.decode(&out, true).map_err(|e| anyhow!("{e}"))?);
    println!(
        "load {load_s:.1}s  prefill {} tok in {prefill_s:.2}s ({:.1} tok/s)  decode {decoded} tok: {:.1} tok/s",
        ids.len(),
        ids.len() as f64 / prefill_s,
        decoded as f64 / decode_s
    );
    report(&m, decoded);
    if let Some(log) = &m.route_log {
        route_report(log);
    }
    memory()
}

/// Share of `want` found in `have`.
fn recall(want: &[u16], have: &[u16]) -> (usize, usize) {
    (want.iter().filter(|e| have.contains(e)).count(), want.len())
}

/// How well each layer's experts were guessed before its routing was known: by the lookahead
/// (its router on the previous layer's residual, top `n`), by the previous token's experts in
/// the same layer, and by both (the union, with its size: reads if all were prefetched).
fn route_report(log: &ironbark_engine::qwen::RouteLog) {
    const TOPS: [usize; 4] = [8, 12, 16, 32];
    // Per layer: hits for each top, previous token, union of top 8 and previous; union sizes.
    let layers = log.chosen.len();
    let mut rows = Vec::with_capacity(layers);
    let mut total = [0usize; 7];
    let mut union_size = (0usize, 0usize);
    for l in 0..layers {
        let (chosen, guess) = (&log.chosen[l], &log.guess[l]);
        let mut r = [0usize; 7];
        for t in 0..chosen.len() {
            let want = &chosen[t];
            if let Some(g) = guess.get(t) {
                for (j, &n) in TOPS.iter().enumerate() {
                    r[j] += recall(want, &g[..n.min(g.len())]).0;
                }
                r[4] += want.len();
            }
            if t > 0 {
                let prev = &chosen[t - 1];
                r[5] += recall(want, prev).0;
                if let Some(g) = guess.get(t) {
                    let mut u: Vec<u16> = g[..8.min(g.len())].to_vec();
                    u.extend(prev.iter().filter(|e| !g[..8.min(g.len())].contains(e)));
                    r[6] += recall(want, &u).0;
                    union_size.0 += u.len();
                    union_size.1 += 1;
                }
            }
        }
        // Previous-token recall is over tokens after the first; scale by the same denominator.
        let prev_den: usize = chosen.iter().skip(1).map(Vec::len).sum();
        rows.push((l, r, prev_den));
        for j in 0..7 {
            total[j] += r[j];
        }
    }
    let pct = |a: usize, b: usize| 100.0 * a as f64 / b.max(1) as f64;
    let tokens = log.chosen.first().map_or(0, Vec::len);
    println!("routing guesses over {tokens} decode tokens (recall of the experts each layer ran):");
    println!("layer  look@8  look@12  look@16  look@32  prev-token  look@8+prev");
    let prev_total: usize = rows.iter().map(|r| r.2).sum();
    let guessed_prev: usize = rows.iter().filter(|r| r.1[4] > 0).map(|r| r.2).sum();
    let guessed_total: usize = rows.iter().filter(|r| r.1[4] > 0).map(|r| r.1[6]).sum();
    for (l, r, pd) in &rows {
        let look = |j: usize| {
            if r[4] > 0 {
                format!("{:6.1}%", pct(r[j], r[4]))
            } else {
                "     -".into()
            }
        };
        let both = if r[4] > 0 {
            format!("{:6.1}%", pct(r[6], *pd))
        } else {
            "     -".into()
        };
        println!("{l:>5}  {}  {}  {}  {}  {:9.1}%  {}", look(0), look(1), look(2), look(3), pct(r[5], *pd), both);
    }
    println!(
        "all    {:5.1}%  {:6.1}%  {:6.1}%  {:6.1}%  {:9.1}%  {:6.1}%  (union holds {:.1} experts)",
        pct(total[0], total[4]),
        pct(total[1], total[4]),
        pct(total[2], total[4]),
        pct(total[3], total[4]),
        pct(total[5], prev_total),
        pct(guessed_total, guessed_prev),
        union_size.0 as f64 / union_size.1.max(1) as f64
    );
}

/// Greedy decode of up to `n_gen` tokens answering `prompt` about the image at `path` (resized
/// to at most `max_pixels`), in one chat turn.
pub(crate) fn run_image(
    dir: &Path,
    ex: &Experts,
    path: &Path,
    max_pixels: usize,
    prompt: &str,
    n_gen: usize,
) -> Result<()> {
    let t = Instant::now();
    let mut m = load(dir, ex)?;
    let vision = Vision::load(dir)?.ok_or_else(|| anyhow!("{}: no vision weights (re-run pack)", dir.display()))?;
    let load_s = t.elapsed().as_secs_f64();
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| anyhow!("{e}"))?;
    let id = |s: &str| tok.token_to_id(s).ok_or_else(|| anyhow!("tokenizer: no {s}"));
    let pad = id("<|image_pad|>")?;

    let t = Instant::now();
    let rgb = image::open(path).map_err(|e| anyhow!("{}: {e}", path.display()))?.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let px = vision::preprocess(&vision.cfg, rgb.as_raw(), w, h, max_pixels)?;
    let img = vision.encode(&px)?;
    let encode_s = t.elapsed().as_secs_f64();
    drop(vision);
    ironbark_mlx::memory::clear_cache()?;
    println!(
        "image {w}×{h} → {}×{} px, {} patches → {} tokens; encoded in {encode_s:.2}s",
        px.cols * 16,
        px.rows * 16,
        px.rows * px.cols,
        img.tokens()
    );

    let text = chat(&format!("<|vision_start|>{}<|vision_end|>{prompt}", "<|image_pad|>".repeat(img.tokens())));
    let ids = tok.encode(text, false).map_err(|e| anyhow!("{e}"))?.get_ids().to_vec();
    anyhow::ensure!(
        ids.iter().filter(|&&x| x == pad).count() == img.tokens(),
        "image tokens did not tokenize one each"
    );
    let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| tok.token_to_id(t)).collect();
    let t = Instant::now();
    let mut logits = m.prefill_image(&ids, std::slice::from_ref(&img), pad, false)?;
    logits.eval()?;
    let prefill_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let mut out = Vec::with_capacity(n_gen);
    while out.len() < n_gen {
        let next = argmax_axis(&logits, -1, None)?.item_exact::<u32>();
        out.push(next);
        if eos.contains(&next) {
            break;
        }
        logits = m.step(next)?;
    }
    let decode_s = t.elapsed().as_secs_f64();
    println!("{}", tok.decode(&out, true).map_err(|e| anyhow!("{e}"))?);
    println!(
        "load {load_s:.1}s  prefill {} tok in {prefill_s:.2}s ({:.1} tok/s)  decode {} tok: {:.1} tok/s",
        ids.len(),
        ids.len() as f64 / prefill_s,
        out.len(),
        out.len() as f64 / decode_s
    );
    report(&m, out.len());
    memory()
}
