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
    m.cfg.dynamic.drop_below = ex.drop_below;
    m.cfg.dynamic.min_k = ex.min_k;
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
    memory()
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
