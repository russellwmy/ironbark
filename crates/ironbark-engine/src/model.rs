//! Maple decode, one token per step.
//!
//! Experts are either resident (stacked tensors) or streamed: kept in `experts.bin` and loaded
//! into a per-layer cache of slab slots by [`ironbark_experts::Store`]. Streaming reads each
//! layer's router choice back on the CPU (one sync per layer), pins those experts, and runs
//! the slab kernels on their slots.

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ironbark_mlx::ops::indexing::{IndexOp, TryIndexMutOp};
use ironbark_mlx::ops::{concatenate, dequantize, quantized_matmul};
use ironbark_mlx::{Array, Dtype, fast};

use ironbark_experts::{Chunks, ExpertRef, Layout, Pin, Records, Store};

use crate::flash::FlashHead;
use crate::fused::{Fused, HIDDEN};
use crate::row::{Kernels, Record, RowMat, Slab, record_mats};
use crate::{Config, Error, Result};

/// A 4-bit matrix in MLX's affine layout (embedding, LM head): `w ≈ scales · q + biases` per group.
pub(crate) struct QMat {
    pub(crate) w: Array,
    pub(crate) scales: Array,
    pub(crate) biases: Array,
    pub(crate) group: i32,
    pub(crate) bits: i32,
}

impl QMat {
    /// `x · Wᵀ`.
    pub(crate) fn apply(&self, x: &Array) -> Result<Array> {
        Ok(quantized_matmul(x, &self.w, &self.scales, &self.biases, true, self.group, self.bits)?)
    }
}

pub(crate) struct Layer {
    pub(crate) in_norm: Array,
    pub(crate) post_norm: Array,
    pub(crate) q_norm: Array,
    pub(crate) k_norm: Array,
    pub(crate) qkv: RowMat,
    pub(crate) o: RowMat,
    /// Router `[experts, hidden]` (bf16; the fused router accumulates in fp32).
    pub(crate) router: Array,
    /// Resident experts, or `None` when they are streamed.
    pub(crate) experts: Option<Experts>,
    pub(crate) sliding: bool,
}

/// Resident experts of a layer: up and gate rows fused (`[E, 2·ffn, ·]`, up first), and down.
pub(crate) struct Experts {
    pub(crate) up_gate: RowMat,
    pub(crate) down: RowMat,
}

/// Where decode steps spend wall time (nanoseconds, summed over steps; reset by
/// [`crate::bench::reset_stats`]).
#[derive(Debug, Default, Clone, Copy)]
pub struct StepTiming {
    /// Steps.
    pub steps: u64,
    /// In [`Model::step_lazy`] in total.
    pub total_ns: u64,
    /// Blocked on the per-layer router readbacks (GPU work before them, plus the round trip).
    pub sync_ns: u64,
}

/// Streaming state: the expert store and its slab as one MLX array.
pub(crate) struct Streaming {
    pub(crate) store: Arc<Store>,
    pub(crate) slab: Slab,
    /// When the cache holds every expert: the slot of expert `l · E + e`, loaded once at
    /// open. Decode maps the router's choice to slots on the GPU, with no readback.
    pub(crate) all: Option<Array>,
}

/// The record format in `experts.json`: "trit" or "2bit".
fn packed_format(dir: &Path) -> Result<String> {
    let path = dir.join("experts.json");
    let raw = std::fs::read(&path).map_err(|source| Error::Io {
        path: path.clone(),
        source,
    })?;
    let v: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
    crate::manifest::check_version(&v, &path)?;
    v["format"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| Error::Format(format!("{}: no record format", path.display())))
}

/// Slot table of a cache that holds every expert (the store preloaded them): the slot of expert
/// `l · E + e`.
fn preload(store: &Store, layers: usize, experts: usize) -> Result<Array> {
    for l in 0..layers {
        for e in 0..experts {
            store.request(l, e);
        }
    }
    // Nothing is evicted from a cache with room for every expert, so the slots stay put.
    let table = (0..layers)
        .flat_map(|l| (0..experts).map(move |e| (l, e)))
        .map(|(l, e)| Ok(store.wait(l, e)?.slot() as u32))
        .collect::<Result<Vec<u32>>>()?;
    Ok(Array::from_slice(&table, &[table.len() as i32]))
}

/// Layer `l`'s experts read from `records` (2-bit) into resident tensors: up+gate `[E, 2·ffn, ·]`
/// and down, each with its row scales, as [`Tensors::ternary`] builds them from a checkpoint.
/// Peak memory above the result is one layer of records.
fn resident_experts(records: &dyn Records, layout: &Layout, record: Record, l: usize, path: &Path) -> Result<Experts> {
    let (e, n) = (layout.experts, record.bytes());
    let mut buf = vec![0u8; e * n];
    for (i, rec) in buf.chunks_exact_mut(n).enumerate() {
        records.read(l * e + i, rec).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
    }
    let recs = Array::from_slice(&buf, &[e as i32, n as i32]).view::<u32>()?;
    let (ug, dn) = record_mats(&recs, record, e as i32)?;
    // Own, contiguous copies: the views would keep the whole layer of records alive.
    let own = |m: RowMat| -> Result<RowMat> {
        Ok(RowMat {
            w: m.w.contiguous()?,
            scales: m.scales.contiguous()?,
            n: m.n,
            k: m.k,
        })
    };
    let ex = Experts {
        up_gate: own(ug)?,
        down: own(dn)?,
    };
    ironbark_mlx::transforms::eval([&ex.up_gate.w, &ex.up_gate.scales, &ex.down.w, &ex.down.scales])?;
    Ok(ex)
}

impl Drop for Streaming {
    fn drop(&mut self) {
        // The loader threads hold the store (and its cache, up to 5 GB); let them go.
        self.store.close();
    }
}

impl Layer {
    fn eval(&self) -> Result<()> {
        let mut all = vec![&self.in_norm, &self.post_norm, &self.q_norm, &self.k_norm, &self.router];
        let mut mats = vec![&self.qkv, &self.o];
        if let Some(x) = &self.experts {
            mats.extend([&x.up_gate, &x.down]);
        }
        for r in mats {
            all.extend([&r.w, &r.scales]);
        }
        Ok(ironbark_mlx::transforms::eval(all)?)
    }
}

/// Keys and values of one layer, `[1, kv_heads, capacity, head_dim]`, written in place.
///
/// Same layout as mlx-lm's caches, so attention sums keys in the same order as the reference:
/// full-attention layers grow by 256 slots and keep time order; sliding layers are a ring of
/// `window` slots written at `pos % window` (rotated order once full).
#[derive(Default, Clone)]
pub(crate) struct Kv {
    pub(crate) k: Option<Array>,
    pub(crate) v: Option<Array>,
}

/// Most embedding chunks fetched at once (`Model::ensure_rows`).
const EMBED_FETCHES: usize = 16;

/// Growth step of full-attention caches (mlx-lm `KVCache.step`).
pub(crate) const KV_STEP: i32 = 256;

impl Kv {
    /// Store this token's `k`, `v` (`[1, kv_heads, 1, d]`) at position `pos`; return the keys and
    /// values to attend over.
    fn update(&mut self, k: &Array, v: &Array, pos: i32, window: Option<i32>) -> Result<(Array, Array)> {
        let cap = self.k.as_ref().map_or(0, |a| a.shape()[2]);
        let (slot, want) = match window {
            Some(w) => (pos % w, w),
            None => (pos, if pos < cap { cap } else { (pos / KV_STEP + 1) * KV_STEP }),
        };
        if want > cap {
            let s = k.shape();
            let grow = |old: Option<&Array>, dt| -> Result<Array> {
                let fresh = Array::zeros::<f32>(&[s[0], s[1], want - cap, s[3]])?.as_dtype(dt)?;
                Ok(match old {
                    Some(o) => concatenate(&[o, &fresh], 2)?,
                    None => fresh,
                })
            };
            self.k = Some(grow(self.k.as_ref(), k.dtype())?);
            self.v = Some(grow(self.v.as_ref(), v.dtype())?);
        }
        let (Some(ks), Some(vs)) = (&mut self.k, &mut self.v) else {
            unreachable!("allocated above")
        };
        ks.try_index_mut((.., .., slot..=slot), k)?;
        vs.try_index_mut((.., .., slot..=slot), v)?;
        let n = match window {
            Some(w) => (pos + 1).min(w),
            None => pos + 1,
        };
        Ok((ks.index((.., .., ..n)), vs.index((.., .., ..n))))
    }
}

/// A saved KV cache ([`Model::snapshot`]).
#[derive(Clone)]
pub struct KvSnapshot {
    kv: Vec<Kv>,
    pos: usize,
}

impl fmt::Debug for KvSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvSnapshot").field("pos", &self.pos).finish_non_exhaustive()
    }
}

/// Maple on the GPU.
pub struct Model {
    /// Hyperparameters.
    pub cfg: Config,
    pub(crate) embed: QMat,
    /// Where `embed.bin` is still arriving (a sparse file), if it is.
    embed_chunks: Option<Arc<dyn Chunks>>,
    /// In token order, or in FlashHead's cluster order once it was turned on.
    pub(crate) head: QMat,
    /// FlashHead, when the checkpoint has one.
    pub(crate) flash: Option<FlashHead>,
    /// `head` is in cluster order.
    permuted: bool,
    pub(crate) norm: Array,
    pub(crate) layers: Vec<Layer>,
    pub(crate) kernels: Kernels,
    pub(crate) fused: Fused,
    /// Zero residual for the first add + norm (x + 0 is exact in bf16).
    zero: Array,
    pub(crate) stream: Option<Streaming>,
    /// Streaming: also predict and prefetch the experts of the next `lookahead` layers.
    pub lookahead: usize,
    /// Where decode steps spend time.
    pub timing: StepTiming,
    /// Streaming: the last layer's experts of the previous step, pinned until the GPU has
    /// read them (the next step's first readback waits for that), with that step's logits.
    held: Vec<Pin>,
    held_until: Option<Array>,
    pub(crate) kv: Vec<Kv>,
    /// Tokens in the KV cache.
    pub pos: usize,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model").field("cfg", &self.cfg).field("pos", &self.pos).finish_non_exhaustive()
    }
}

/// All tensors of the checkpoint in `dir` (every shard listed in the index), unevaluated: a
/// tensor is read from disk only if it is evaluated.
pub(crate) fn load_tensors(dir: &Path) -> Result<HashMap<String, Array>> {
    let index = dir.join("model.safetensors.index.json");
    let raw = std::fs::read(&index).map_err(|source| Error::Io {
        path: index.clone(),
        source,
    })?;
    let v: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|e| Error::Format(format!("{}: {e}", index.display())))?;
    let mut files: Vec<String> = v["weight_map"]
        .as_object()
        .ok_or_else(|| Error::Format(format!("{}: no weight_map", index.display())))?
        .values()
        .filter_map(|s| s.as_str().map(str::to_owned))
        .collect();
    files.sort();
    files.dedup();
    let mut all = HashMap::new();
    for f in files {
        let path: PathBuf = dir.join(f);
        let t = Array::load_safetensors(&path).map_err(|e| Error::Load {
            path: path.clone(),
            msg: e.to_string(),
        })?;
        all.extend(t);
    }
    Ok(all)
}

/// Pops tensors by name with a clear error for missing ones.
struct Tensors(HashMap<String, Array>);

impl Tensors {
    fn take(&mut self, name: &str) -> Result<Array> {
        self.0.remove(name).ok_or_else(|| Error::Format(format!("missing tensor {name}")))
    }

    /// Affine-quantized matrix stored as `weight`, `scales`, `biases`.
    fn affine(&mut self, prefix: &str, group: usize, bits: usize) -> Result<QMat> {
        Ok(QMat {
            w: self.take(&format!("{prefix}.weight"))?,
            scales: self.take(&format!("{prefix}.scales"))?,
            biases: self.take(&format!("{prefix}.biases"))?,
            group: group as i32,
            bits: bits as i32,
        })
    }

    /// Ternary matrices stored as packed 2-bit `weight` + `row_alpha`, concatenated along the
    /// output rows (`axis` 0 for a plain matrix, 1 for stacked experts). Kept with one scale per
    /// row (α); the row-scale kernels use −α as the bias.
    fn ternary(&mut self, prefixes: &[String], axis: i32) -> Result<RowMat> {
        let mut ws = Vec::new();
        let mut alphas = Vec::new();
        for p in prefixes {
            ws.push(self.take(&format!("{p}.weight"))?);
            alphas.push(self.take(&format!("{p}.row_alpha"))?);
        }
        let w = concatenate(&ws, axis)?;
        let scales = concatenate(&alphas, axis)?.as_dtype(Dtype::Bfloat16)?;
        let shape = w.shape();
        let (n, k) = (shape[shape.len() - 2], shape[shape.len() - 1] * 16);
        // The kernels step 512 inputs per pass and 8 rows per threadgroup.
        if k % 512 != 0 || n % 8 != 0 {
            return Err(Error::Format(format!("{}: shape {shape:?} not supported", prefixes[0])));
        }
        Ok(RowMat { w, scales, n, k })
    }
}

impl Model {
    /// Load a Maple MLX checkpoint from `dir`, all experts resident.
    ///
    /// # Errors
    ///
    /// Fails if a file is missing or the tensors do not match the config.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with(dir, None, None)
    }

    /// Load a Maple MLX checkpoint from `dir` with experts streamed from `experts` (written by
    /// [`crate::pack`]): `per_layer` cache slots per layer, `loaders` reader threads.
    ///
    /// # Errors
    ///
    /// Fails if a file is missing or `experts` does not match the checkpoint.
    pub fn load_streaming(dir: &Path, experts: &Path, per_layer: usize, loaders: usize) -> Result<Self> {
        Self::load_experts(dir, experts, per_layer, loaders, ironbark_experts::records::local(experts)?)
    }

    /// Like [`Model::load_streaming`], with expert records read from `records` (e.g. an
    /// `ironbark-fetch` file that fetches missing ones); `dir` gives `experts.json`.
    ///
    /// # Errors
    ///
    /// Fails if a file is missing or the records do not match the checkpoint.
    pub fn load_records(dir: &Path, per_layer: usize, loaders: usize, records: Arc<dyn Records>) -> Result<Self> {
        Self::load_experts(dir, dir, per_layer, loaders, records)
    }

    /// Experts read from `records` (laid out as `experts/experts.json` says). When every expert
    /// fits and the records are 2-bit, they are read once into resident tensors in the
    /// checkpoint's layout, so decode and prefill run exactly as on the checkpoint (MLX's
    /// `gather_qmm` is ~50% slower on the slab's strided records). Otherwise they go through a
    /// store: streamed, or (trit) preloaded into its slab, which keeps them packed.
    fn load_experts(
        dir: &Path,
        experts: &Path,
        per_layer: usize,
        loaders: usize,
        records: Arc<dyn Records>,
    ) -> Result<Self> {
        let cfg = Config::read(dir)?;
        let record = Record {
            hidden: cfg.hidden,
            ffn: cfg.ffn,
            trit: packed_format(experts)? == "trit",
        };
        let layout = Layout::read(experts)?;
        if layout.layers != cfg.layers || layout.experts != cfg.experts || layout.record_bytes != record.bytes() as u64
        {
            return Err(Error::Format(format!("{}: does not match the checkpoint (re-run pack)", experts.display())));
        }
        if per_layer >= cfg.experts && !record.trit {
            let path = experts.join("experts.bin");
            let resident = (0..cfg.layers)
                .map(|l| resident_experts(&*records, &layout, record, l, &path))
                .collect::<Result<Vec<_>>>()?;
            return Self::load_with(dir, None, Some(resident));
        }
        Self::load_store(dir, record, Store::open_with(experts, per_layer, loaders, records)?)
    }

    fn load_store(dir: &Path, record: Record, store: Arc<Store>) -> Result<Self> {
        let cfg = Config::read(dir)?;
        let slab = Slab {
            array: crate::kernel::wrap_slab(&store)?,
            stride: store.slab().slot_bytes() as i32,
            record,
        };
        let all = if store.per_layer() >= cfg.experts {
            Some(preload(&store, cfg.layers, cfg.experts)?)
        } else {
            None
        };
        Self::load_with(dir, Some(Streaming { store, slab, all }), None)
    }

    /// Load the checkpoint in `dir`; its experts unless they are streamed (`stream`) or given
    /// (`resident`, one per layer).
    fn load_with(dir: &Path, stream: Option<Streaming>, resident: Option<Vec<Experts>>) -> Result<Self> {
        let cfg = Config::read(dir)?;
        // The fused kernels are built for Maple's shapes.
        if cfg.hidden as i32 != HIDDEN || cfg.experts != 256 || cfg.top_k != 8 || cfg.head_dim % 32 != 0 {
            return Err(Error::Format(format!(
                "unsupported shape: hidden {}, {} experts, top-{}, head_dim {}",
                cfg.hidden, cfg.experts, cfg.top_k, cfg.head_dim
            )));
        }
        let mut t = Tensors(load_tensors(dir)?);
        let mut resident = resident.map(Vec::into_iter);
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("model.layers.{l}");
            let attn = |n: &str| format!("{p}.self_attn.{n}");
            let exp = |n: &str| format!("{p}.mlp.switch_mlp.{n}");
            layers.push(Layer {
                in_norm: t.take(&format!("{p}.input_layernorm.weight"))?,
                post_norm: t.take(&format!("{p}.post_attention_layernorm.weight"))?,
                q_norm: t.take(&attn("q_norm.weight"))?,
                k_norm: t.take(&attn("k_norm.weight"))?,
                qkv: t.ternary(&[attn("q_proj"), attn("k_proj"), attn("v_proj")], 0)?,
                o: t.ternary(&[attn("o_proj")], 0)?,
                router: t.take(&format!("{p}.mlp.gate.weight"))?,
                // Streamed or given experts are never evaluated here, so never read from the
                // checkpoint.
                experts: match (&stream, resident.as_mut()) {
                    (Some(_), _) => None,
                    (None, Some(given)) => {
                        Some(given.next().ok_or_else(|| Error::Format(format!("layer {l}: no experts given")))?)
                    }
                    (None, None) => Some(Experts {
                        up_gate: t.ternary(&[exp("up_proj"), exp("gate_proj")], 1)?,
                        down: t.ternary(&[exp("down_proj")], 1)?,
                    }),
                },
                sliding: cfg.sliding[l],
            });
            // Evaluate each layer as it is built: its fused tensors are copies, and evaluating
            // now frees the originals before the next layer (lower peak memory while loading).
            if let Some(ly) = layers.last() {
                ly.eval()?;
            }
        }
        let flash = match (&cfg.flash_head, t.0.remove("lm_head_flash.token_map")) {
            (Some(meta), Some(map)) => {
                let centroids = t.affine("lm_head_flash.centroids", meta.group, meta.bits)?;
                ironbark_mlx::transforms::eval([&centroids.w, &centroids.scales, &centroids.biases])?;
                FlashHead::new(&cfg, &map, centroids)?
            }
            _ => None,
        };
        let model = Model {
            // Mapped from `embed.bin` in a packed model (rows are paged in as tokens use them).
            embed: match crate::embed::map(dir, &cfg)? {
                Some(e) => e,
                None => t.affine("model.word_embeddings", cfg.head_group, cfg.head_bits)?,
            },
            head: t.affine("lm_head", cfg.head_group, cfg.head_bits)?,
            embed_chunks: None,
            flash,
            permuted: false,
            norm: t.take("model.norm.weight")?,
            kv: (0..cfg.layers).map(|_| Kv::default()).collect(),
            layers,
            kernels: Kernels::new(),
            fused: Fused::new(cfg.eps, cfg.theta),
            zero: Array::zeros::<f32>(&[1, 1, cfg.hidden as i32])?.as_dtype(Dtype::Bfloat16)?,
            stream,
            lookahead: 0,
            timing: StepTiming::default(),
            held: Vec::new(),
            held_until: None,
            cfg,
            pos: 0,
        };
        // Materialize every weight now, so the first step does not pay for loading.
        let mut all = vec![&model.norm, &model.zero];
        for q in [&model.embed, &model.head] {
            all.extend([&q.w, &q.scales, &q.biases]);
        }
        ironbark_mlx::transforms::eval(all)?;
        Ok(model)
    }

    /// Exact logits of `x` `[1, L, hidden]`.
    pub(crate) fn head_logits(&self, x: &Array) -> Result<Array> {
        let logits = self.head.apply(x)?;
        match (&self.flash, self.permuted) {
            // Rows were computed in cluster order: back to token order.
            (Some(f), true) => Ok(logits.take_axis(&f.rows, -1)?),
            _ => Ok(logits),
        }
    }

    /// Decode with FlashHead (approximate) or the exact head; false if the checkpoint has no
    /// FlashHead. Turning it on puts the head in cluster order once (no extra memory after).
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails.
    pub fn set_flash_head(&mut self, on: bool) -> Result<bool> {
        let Some(f) = &mut self.flash else { return Ok(false) };
        if on && !self.permuted {
            self.head = f.permute(&self.head)?;
            self.permuted = true;
        }
        f.on = on;
        Ok(true)
    }

    /// Read the mapped `embed.bin` through `chunks` while it is still arriving (e.g. an
    /// `ironbark-fetch` sparse file): before a step or prefill reads a token's row, its chunks are
    /// made present. While incomplete, decode reads each token back first (one sync per step).
    pub fn set_embed_chunks(&mut self, chunks: Option<Arc<dyn Chunks>>) {
        self.embed_chunks = chunks;
    }

    /// Make the `embed.bin` rows of `ids` present, if the file is still arriving.
    pub(crate) fn ensure_rows(&self, ids: &[u32]) -> Result<()> {
        let Some(c) = &self.embed_chunks else { return Ok(()) };
        if c.complete() {
            return Ok(());
        }
        let mut want: Vec<usize> = ids.iter().flat_map(|&t| crate::embed::chunks_of(&self.cfg, t as usize)).collect();
        want.sort_unstable();
        want.dedup();
        want.retain(|&i| !c.has(i));
        // Missing chunks are fetched in parallel: a prompt can need hundreds, each a round trip.
        let threads = want.len().min(EMBED_FETCHES);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let r = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..threads)
                .map(|_| {
                    sc.spawn(|| -> std::io::Result<()> {
                        loop {
                            let n = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(&i) = want.get(n) else { return Ok(()) };
                            c.ensure(i)?;
                        }
                    })
                })
                .collect();
            hs.into_iter()
                .map(|h| h.join().unwrap_or_else(|_| Err(std::io::Error::other("embedding fetch panicked"))))
                .collect::<std::io::Result<Vec<()>>>()
        });
        r.map(|_| ()).map_err(|source| Error::Io {
            path: crate::embed::FILE.into(),
            source,
        })
    }

    /// The expert store, when experts are streamed.
    #[must_use]
    pub fn store(&self) -> Option<&Arc<Store>> {
        self.stream.as_ref().map(|s| &s.store)
    }

    /// Wait for the previous step's GPU work, then release the experts it held (before
    /// anything else may claim their slots: a reset, a prefill).
    pub(crate) fn release_held(&mut self) -> Result<()> {
        if let Some(a) = self.held_until.take() {
            a.eval()?;
        }
        self.held.clear();
        Ok(())
    }

    /// The KV cache as it is now, to return to with [`Model::restore`] (e.g. after a prompt
    /// every conversation starts with). Shares the arrays: MLX copies on the next write.
    #[must_use]
    pub fn snapshot(&self) -> KvSnapshot {
        KvSnapshot {
            kv: self.kv.clone(),
            pos: self.pos,
        }
    }

    /// Return the KV cache to `s`, taken from this model.
    pub fn restore(&mut self, s: &KvSnapshot) {
        // A failed wait leaves nothing to protect: the step's output is discarded with it.
        let _ = self.release_held();
        self.kv.clone_from(&s.kv);
        self.pos = s.pos;
    }

    /// Clear the KV cache.
    pub fn reset(&mut self) {
        // A failed wait leaves nothing to protect: the step's output is discarded with it.
        let _ = self.release_held();
        self.kv.iter_mut().for_each(|kv| *kv = Kv::default());
        self.pos = 0;
    }

    /// One token -> logits `[1, 1, vocab]` (lazy; evaluate or read them to run the step).
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails.
    pub fn step(&mut self, tok: u32) -> Result<Array> {
        self.step_lazy(&Array::from_slice(&[tok], &[1]))
    }

    /// The experts in cache `slots` of `slab` applied to `hn`, one output row per slot.
    pub(crate) fn slab_experts(&self, hn: &Array, slab: &Slab, slots: &Array) -> Result<Array> {
        let k = slots.shape()[0];
        let act = self.kernels.up_swiglu_slab(hn, slab, slots, k)?;
        self.kernels.qmv_gather_slab(&act, slab, slots, k)
    }

    /// Like [`Model::step`], but the token is an MLX array `[1]` (u32) that may itself be
    /// unevaluated, e.g. the argmax of the previous step. This lets the caller queue step t+1
    /// before reading token t back, so the GPU never waits for the CPU.
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails.
    pub fn step_lazy(&mut self, ids: &Array) -> Result<Array> {
        let t_step = std::time::Instant::now();
        let mut sync = std::time::Duration::ZERO;
        let c = &self.cfg;
        let (nq, nkv, d) = (c.heads as i32, c.kv_heads as i32, c.head_dim as i32);
        let (qd, kd) = (nq * d, nkv * d);
        if self.embed_chunks.as_ref().is_some_and(|c| !c.complete()) {
            // The row must be on disk before the GPU reads it: wait for the token.
            ids.eval()?;
            let v = ids.as_slice::<u32>().to_vec();
            self.ensure_rows(&v)?;
        }
        let e = &self.embed;
        let x = dequantize(
            e.w.take_axis(ids, 0)?,
            e.scales.take_axis(ids, 0)?,
            &e.biases.take_axis(ids, 0)?,
            e.group,
            e.bits,
        )?
        .reshape(&[1, 1, c.hidden as i32])?;
        // The residual stream is carried as (h, hn): every add + norm pair is one kernel.
        let (mut h, mut hn) = self.fused.add_norm(&x, &self.zero, &self.layers[0].in_norm)?;
        let k = c.top_k as i32;
        // Streaming: experts of the layer whose kernels may still run, pinned until the next
        // router readback (which needs, and so waits for, those kernels' output).
        let store = self.stream.as_ref().map(|s| s.store.clone());
        let mut pins: Vec<ExpertRef<'_>> = Vec::new();
        for i in 0..self.layers.len() {
            let ly = &self.layers[i];
            let kv = &mut self.kv[i];

            // Attention: QKV, per-head norm + RoPE (sliding layers), cache, attention, O.
            let qkv = self.kernels.qmv(&hn, &ly.qkv)?.reshape(&[qd + 2 * kd])?;
            let qk = qkv.index(..qd + kd).reshape(&[nq + nkv, d])?;
            let rope = if ly.sliding { c.rope_dims as i32 } else { 0 };
            let qk = self.fused.qk_norm_rope(&qk, &ly.q_norm, &ly.k_norm, nq, self.pos, rope)?;
            let q = qk.index(..nq).reshape(&[1, nq, 1, d])?;
            let kn = qk.index(nq..).reshape(&[1, nkv, 1, d])?;
            let v = qkv.index(qd + kd..).reshape(&[1, nkv, 1, d])?;
            let window = ly.sliding.then_some(c.window as i32);
            let (ks, vs) = kv.update(&kn, &v, self.pos as i32, window)?;
            let scale = 1.0 / (c.head_dim as f32).sqrt();
            // One query position: [1, heads, 1, d] is already the [1, 1, heads·d] layout.
            let o = fast::scaled_dot_product_attention(&q, &ks, &vs, scale, None, None)?.reshape(&[1, 1, qd])?;
            let r = self.kernels.qmv(&o, &ly.o)?;

            // Mixture of experts: add + norm, route, experts, then the weighted sum is folded
            // into the next add + norm (the next layer's input norm, or the final norm).
            (h, hn) = self.fused.add_norm(&h, &r, &ly.post_norm)?;
            let next = self.layers.get(i + 1).map_or(&self.norm, |l| &l.in_norm);
            let (ids, scores) = self.fused.router(&hn, &ly.router)?;
            if let (Some(st), None) = (&self.stream, &ly.experts)
                && let Some(all) = &st.all
            {
                let ne = c.experts as i32;
                let slots = all.index(i as i32 * ne..(i as i32 + 1) * ne).take_axis(&ids, 0)?;
                let y = self.slab_experts(&hn, &st.slab, &slots)?;
                (h, hn) = self.fused.agg_norm(&h, &y, &scores, next)?;
                continue;
            }
            let y = match (&ly.experts, &self.stream, &store) {
                (Some(x), ..) => {
                    let act = self.kernels.up_swiglu(&hn, &x.up_gate, &ids, k)?;
                    self.kernels.qmv_gather(&act, &x.down, &ids, k)?
                }
                (None, Some(st), Some(store)) => {
                    // Predict the next layers' experts from this residual (their router on
                    // their norm of it); read back with this layer's choice, in one sync.
                    let mut preds = Vec::new();
                    for (j, nl) in self.layers.iter().enumerate().skip(i + 1).take(self.lookahead) {
                        let (_, hp) = self.fused.add_norm(&h, &self.zero, &nl.post_norm)?;
                        preds.push((j, self.fused.router(&hp, &nl.router)?.0));
                    }
                    let mut wanted = vec![&ids];
                    wanted.extend(preds.iter().map(|(_, p)| p));
                    let t = std::time::Instant::now();
                    ironbark_mlx::transforms::eval(wanted)?;
                    sync += t.elapsed();
                    // The GPU is past the previous layer, and past the previous step.
                    pins.clear();
                    self.held.clear();
                    self.held_until = None;
                    let chosen: Vec<usize> = ids.as_slice::<u32>().iter().map(|&e| e as usize).collect();
                    for &e in &chosen {
                        store.request(i, e);
                    }
                    for (j, p) in &preds {
                        for &e in p.as_slice::<u32>() {
                            store.prefetch(*j, e as usize);
                        }
                    }
                    for &e in &chosen {
                        pins.push(store.wait(i, e)?);
                    }
                    let slots: Vec<u32> = pins.iter().map(|r| r.slot() as u32).collect();
                    self.slab_experts(&hn, &st.slab, &Array::from_slice(&slots, &[k]))?
                }
                _ => unreachable!("a layer without resident experts has a store"),
            };
            (h, hn) = self.fused.agg_norm(&h, &y, &scores, next)?;
        }
        self.pos += 1;
        let logits = match &self.flash {
            Some(f) if f.on => f.logits(&self.head, &hn)?,
            _ => self.head_logits(&hn)?,
        };
        if let Some(st) = &store {
            // The last layer's experts are read by the logits' computation: keep them pinned
            // until the next step's first readback instead of waiting for it here.
            self.held = pins.iter().map(|r| Pin::new(st, r)).collect();
            self.held_until = Some(logits.clone());
            pins.clear();
        }
        self.timing.steps += 1;
        self.timing.sync_ns += sync.as_nanos() as u64;
        self.timing.total_ns += t_step.elapsed().as_nanos() as u64;
        Ok(logits)
    }
}
