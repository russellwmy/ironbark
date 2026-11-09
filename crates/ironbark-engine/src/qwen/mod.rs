//! Qwen3.5 / 3.6 MoE (`qwen3_5_moe`, e.g. Qwen3.6-35B-A3B) on the GPU through MLX, as mlx-lm's
//! `qwen3_5` computes it.
//!
//! Three of every four layers mix tokens with a Gated DeltaNet (a linear-attention recurrence
//! with a fixed-size state), the fourth with gated full attention. Every layer then runs a
//! routed mixture of SwiGLU experts plus one shared expert:
//!
//! ```text
//!   h += mixer(in_norm(h))          DeltaNet (layers 0,1,2, 4,5,6, …) or attention (3, 7, …)
//!   x  = post_norm(h)
//!   h += Σ_k w_k · expert_k(x) + sigmoid(x · g) · shared(x)     top-8 of 256, softmax router
//! ```
//!
//! Attention: q and a sigmoid output gate from one projection, RMS-normed q and k, RoPE on the
//! first quarter of each head (MRoPE reduces to this for text), GQA. Logits are not capped.
//!
//! Weights are resident (an MLX 4-bit or 8-bit checkpoint), or a pack's (`ironbark pack`): the
//! non-expert weights resident and ternary ([`tern`]) or affine ([`affine`]) experts in
//! `experts.bin`, cached in full or streamed (the router's choice read back per layer, those experts pinned and loaded, the
//! next layer's predicted from this residual and prefetched). The DeltaNet recurrence is one
//! Metal kernel launch per layer and forward ([`deltanet`]).

use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ironbark_mlx::fast::ScaledDotProductAttentionMask;
use ironbark_mlx::ops::indexing::IndexOp;
use ironbark_mlx::ops::{argpartition_axis, argsort, concatenate, dequantize, floor_divide, gather_qmm, sigmoid, silu};
use ironbark_mlx::{Array, Dtype, fast};

use ironbark_experts::{ExpertRef, Layout, Pin, Records, Store};

use crate::affine::{self, AffineRecord};
use crate::kv::kv_update;
use crate::model::{QMat, load_tensors};
use crate::quant::Tensors;
use crate::tern::{self, TernRecord};
use crate::{Error, Result};

pub mod deltanet;
pub mod vision;

/// Tokens per prefill forward.
pub const CHUNK: usize = 512;

/// Shapes and hyperparameters (`config.json`, its `text_config` when there is one).
#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub layers: usize,
    /// Whether layer `l` is a full-attention layer (the rest are DeltaNet).
    pub full: Vec<bool>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Dims of each head that RoPE rotates.
    pub rotary: usize,
    pub theta: f32,
    /// MRoPE: rotary frequencies on the temporal, row and column axes ([`vision::mrope_tables`]).
    pub mrope_section: [usize; 3],
    pub delta: deltanet::Shape,
    pub experts: usize,
    pub top_k: usize,
    /// Renormalize the top-k router weights to sum to 1.
    pub norm_top_k: bool,
    /// Dynamic top-k for decode steps (off by default).
    pub dynamic: crate::dynamic::Dynamic,
    pub expert_ffn: usize,
    pub shared_ffn: usize,
    pub vocab: usize,
    pub eps: f32,
    /// Quantization group size.
    pub group: usize,
    /// Bits of the token embedding (the checkpoint's default unless it overrides them).
    pub embed_bits: usize,
    /// Experts are ternary trit records in `experts.bin` ([`pack_ternary`]).
    pub expert_ternary: bool,
    /// Experts are affine records of these bits in `experts.bin` ([`pack_affine`]).
    pub expert_bits: Option<usize>,
}

impl Config {
    /// Read `dir/config.json`.
    ///
    /// # Errors
    ///
    /// Fails if the file is missing or lacks a field.
    pub fn read(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw = std::fs::read(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let root: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
        let c = root.get("text_config").unwrap_or(&root);
        let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or_else(|| Error::Format(format!("config: no {k}")));
        let layers = u("num_hidden_layers")?;
        let full: Vec<bool> = if let Some(t) = c["layer_types"].as_array() {
            t.iter().map(|x| x.as_str() == Some("full_attention")).collect()
        } else {
            let every = u("full_attention_interval")?;
            (0..layers).map(|l| (l + 1) % every == 0).collect()
        };
        if full.len() != layers {
            return Err(Error::Format("config: layer_types does not match num_hidden_layers".into()));
        }
        let rope = &c["rope_parameters"];
        let partial = rope["partial_rotary_factor"].as_f64().or(c["partial_rotary_factor"].as_f64()).unwrap_or(0.25);
        let theta = rope["rope_theta"].as_f64().or(c["rope_theta"].as_f64()).unwrap_or(10_000_000.0);
        let head_dim = u("head_dim")?;
        let q = root.get("quantization").or(root.get("quantization_config")).unwrap_or(&serde_json::Value::Null);
        Ok(Config {
            hidden: u("hidden_size")?,
            layers,
            full,
            heads: u("num_attention_heads")?,
            kv_heads: u("num_key_value_heads")?,
            head_dim,
            rotary: (head_dim as f64 * partial) as usize,
            theta: theta as f32,
            mrope_section: match rope["mrope_section"]
                .as_array()
                .map(|a| a.iter().filter_map(serde_json::Value::as_u64).collect::<Vec<_>>())
            {
                Some(v) if v.len() == 3 => [v[0] as usize, v[1] as usize, v[2] as usize],
                _ => [11, 11, 10],
            },
            delta: deltanet::Shape {
                k_heads: u("linear_num_key_heads")?,
                v_heads: u("linear_num_value_heads")?,
                k_dim: u("linear_key_head_dim")?,
                v_dim: u("linear_value_head_dim")?,
                conv: u("linear_conv_kernel_dim")?,
            },
            experts: u("num_experts")?,
            top_k: u("num_experts_per_tok")?,
            norm_top_k: c["norm_topk_prob"].as_bool().unwrap_or(true),
            dynamic: crate::dynamic::Dynamic::default(),
            expert_ffn: u("moe_intermediate_size")?,
            shared_ffn: u("shared_expert_intermediate_size")?,
            vocab: u("vocab_size")?,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            group: q["group_size"].as_u64().unwrap_or(64) as usize,
            embed_bits: embed_bits(q),
            expert_ternary: q["experts"]["ternary"].as_bool().unwrap_or(false),
            expert_bits: q["experts"]["bits"].as_u64().map(|b| b as usize),
        })
    }
}

/// Write a pack to `out`: the non-expert weights of the MLX checkpoint `model` (4-bit), its
/// config and tokenizer, and the ternary experts of `ternary` (codes and bf16 group scales) as
/// trit records in `experts.bin`.
///
/// # Errors
///
/// Fails if `out` is `model`, a tensor is missing or does not match the config, or a file
/// cannot be written.
pub fn pack_ternary(model: &Path, out: &Path, ternary: &Path) -> Result<crate::pack::Packed> {
    let cfg = Config::read(model)?;
    if out.canonicalize().ok() == model.canonicalize().ok() {
        return Err(Error::Format("pack: out must differ from the model dir".into()));
    }
    std::fs::create_dir_all(out).map_err(|source| Error::Io {
        path: out.to_path_buf(),
        source,
    })?;
    let mut t = load_tensors(model)?;
    write_embed(out, &cfg, &mut t)?;
    crate::experts::write_slim(model, out, t, ".mlp.switch_mlp.")?;
    vision::pack(model, out, vision::PACK_BITS)?;
    tern::pack(out, ternary, cfg.layers, cfg.experts, record(&cfg))
}

/// Write a pack to `out` keeping the experts of the MLX checkpoint `model` at its precision (4
/// or 8 bits): its non-expert weights, config and tokenizer, and the experts as affine records
/// in `experts.bin`, to run in full or streamed.
///
/// # Errors
///
/// Fails if `out` is `model`, a tensor is missing or does not match the config, the experts
/// are not 4 or 8 bits, or a file cannot be written.
pub fn pack_affine(model: &Path, out: &Path) -> Result<crate::pack::Packed> {
    let cfg = Config::read(model)?;
    if out.canonicalize().ok() == model.canonicalize().ok() {
        return Err(Error::Format("pack: out must differ from the model dir".into()));
    }
    std::fs::create_dir_all(out).map_err(|source| Error::Io {
        path: out.to_path_buf(),
        source,
    })?;
    let mut t = load_tensors(model)?;
    write_embed(out, &cfg, &mut t)?;
    let prefix = |l: usize| format!("{PREFIX}.layers.{l}.mlp.switch_mlp");
    let shape = |part: &str| {
        let key = format!("{}.gate_proj.{part}", prefix(0));
        t.get(&key)
            .map(Array::shape)
            .map(|s| s[2])
            .ok_or_else(|| Error::Format(format!("{}: no {key}", model.display())))
    };
    let bits = affine::bits_of(shape("weight")?, shape("scales")?)?;
    crate::experts::write_slim(model, out, t.clone(), ".mlp.switch_mlp.")?;
    vision::pack(model, out, vision::PACK_BITS)?;
    let rec = AffineRecord {
        hidden: cfg.hidden,
        ffn: cfg.expert_ffn,
        bits,
    };
    affine::pack(out, t, prefix, cfg.layers, cfg.experts, rec)
}

/// Add the vision encoder of the checkpoint `model` to the existing pack `out` (and its
/// manifest). Returns the bytes written, or `None` for a text-only model.
///
/// # Errors
///
/// Fails if `out` is not a pack, a tensor is missing, or a file cannot be written.
pub fn pack_vision(model: &Path, out: &Path) -> Result<Option<u64>> {
    if !out.join(crate::manifest::FILE).exists() {
        return Err(Error::Format(format!("{}: not a pack", out.display())));
    }
    let bytes = vision::pack(model, out, vision::PACK_BITS)?;
    if bytes.is_some() {
        crate::manifest::add(out, vision::FILE)?;
    }
    Ok(bytes)
}

/// Bits of the token embedding in the `quantization` block `q`: its own entry, else the default.
fn embed_bits(q: &serde_json::Value) -> usize {
    q[format!("{PREFIX}.embed_tokens")]["bits"].as_u64().or(q["bits"].as_u64()).unwrap_or(4) as usize
}

/// The embedding's size and quantization, as `embed.bin` stores it.
fn embed_shape(c: &Config) -> crate::embed::Shape {
    crate::embed::Shape {
        vocab: c.vocab,
        hidden: c.hidden,
        bits: c.embed_bits,
        group: c.group,
    }
}

/// Move the token embedding out of `t` into `out/embed.bin` (memory-mapped at run time: its
/// rows are clean file-backed pages, not the app's own memory).
fn write_embed(out: &Path, cfg: &Config, t: &mut HashMap<String, Array>) -> Result<()> {
    let mut rest = Tensors(std::mem::take(t));
    let embed = rest.qmat(&format!("{PREFIX}.embed_tokens"), cfg.group);
    *t = rest.0;
    crate::embed::write(out, embed_shape(cfg), &embed?)
}

/// One expert's trit record.
const fn record(c: &Config) -> TernRecord {
    TernRecord {
        hidden: c.hidden,
        ffn: c.expert_ffn,
    }
}

/// The ternary experts in a bundle (a 4-bit checkpoint without its experts, and this file).
pub const BUNDLE_EXPERTS: &str = "experts.safetensors";
/// The subdir of a bundle that [`pack_ternary`] writes by default.
pub const BUNDLE_PACK: &str = "pack";

/// The ternary export of the bundle `dir`, if `dir` is one.
#[must_use]
pub fn bundle_experts(dir: &Path) -> Option<PathBuf> {
    let e = dir.join(BUNDLE_EXPERTS);
    e.is_file().then_some(e)
}

/// The dir to load for `dir`: its pack `dir/pack` if `dir` is a packed bundle, else `dir`.
///
/// # Errors
///
/// Fails if `dir` is a bundle that has not been packed.
pub fn packed(dir: &Path) -> Result<PathBuf> {
    let pack = dir.join(BUNDLE_PACK);
    if pack.join("experts.json").exists() {
        return Ok(pack);
    }
    if bundle_experts(dir).is_some() {
        return Err(Error::Format(format!("{}: not packed yet; run `ironbark pack --model {0}`", dir.display())));
    }
    Ok(dir.to_path_buf())
}

/// Whether `dir` holds a Qwen3.5 / 3.6 MoE checkpoint.
#[must_use]
pub fn is_qwen(dir: &Path) -> bool {
    std::fs::read_to_string(dir.join("config.json")).is_ok_and(|s| s.contains("\"qwen3_5_moe"))
}

/// Gated full attention.
struct Attention {
    /// Queries and their output gates, interleaved per head: `[heads · 2 · head_dim, hidden]`.
    q: QMat,
    k: QMat,
    v: QMat,
    o: QMat,
    q_norm: Array,
    k_norm: Array,
}

enum Mixer {
    Delta(deltanet::DeltaNet),
    Attn(Attention),
}

/// One layer's experts stacked `[experts, rows, cols]`.
struct Experts {
    gate: QMat,
    up: QMat,
    down: QMat,
}

/// The expert every token uses, and its sigmoid gate.
struct Shared {
    gate: QMat,
    up: QMat,
    down: QMat,
    /// `[1, hidden]`.
    weight: QMat,
}

struct Layer {
    in_norm: Array,
    post_norm: Array,
    mixer: Mixer,
    router: QMat,
    /// Resident experts, or `None` when they are in the store.
    experts: Option<Experts>,
    shared: Shared,
}

impl Layer {
    fn arrays(&self) -> Vec<&Array> {
        let mut all = vec![&self.in_norm, &self.post_norm];
        let mut mats = vec![
            &self.router,
            &self.shared.gate,
            &self.shared.up,
            &self.shared.down,
            &self.shared.weight,
        ];
        if let Some(x) = &self.experts {
            mats.extend([&x.gate, &x.up, &x.down]);
        }
        match &self.mixer {
            Mixer::Delta(d) => {
                all.extend(d.arrays());
                mats.extend(d.mats());
            }
            Mixer::Attn(a) => {
                all.extend([&a.q_norm, &a.k_norm]);
                mats.extend([&a.q, &a.k, &a.v, &a.o]);
            }
        }
        for m in mats {
            all.extend([&m.w, &m.scales, &m.biases]);
        }
        all
    }
}

/// What one layer carries from token to token.
#[derive(Clone)]
enum State {
    /// Keys and values `[1, kv_heads, capacity, head_dim]`.
    Kv {
        k: Option<Array>,
        v: Option<Array>,
    },
    Delta(deltanet::State),
}

impl State {
    fn fresh(full: bool) -> Self {
        if full {
            State::Kv { k: None, v: None }
        } else {
            State::Delta(deltanet::State::default())
        }
    }

    fn arrays(&self) -> Vec<&Array> {
        match self {
            State::Kv { k, v } => [k, v].into_iter().flatten().collect(),
            State::Delta(d) => d.arrays(),
        }
    }
}

/// Sequences decoding together ([`Model::batch`], [`Model::step_batch`]): per layer, their
/// caches stacked a row per sequence (keys padded after each prompt to the longest one).
pub struct Batch {
    state: Vec<State>,
    /// Each sequence's prompt length: where its positions go on from.
    lens: Vec<usize>,
    /// Each sequence's [`Model::delta`].
    deltas: Vec<i64>,
    /// The width the prompts' keys were padded to.
    pad: usize,
    /// Steps taken together.
    steps: usize,
}

impl fmt::Debug for Batch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Batch")
            .field("lens", &self.lens)
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

/// Saved caches ([`Model::snapshot`]).
#[derive(Clone)]
pub struct Snapshot {
    state: Vec<State>,
    pos: usize,
    delta: i64,
}

impl fmt::Debug for Snapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Snapshot").field("pos", &self.pos).finish_non_exhaustive()
    }
}

impl Snapshot {
    /// Tokens in the saved caches.
    #[must_use]
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// The rotary position offset after an image ([`Model::snapshot_from`] takes it back).
    #[must_use]
    pub fn delta(&self) -> i64 {
        self.delta
    }

    /// The caches by name, to save: `{layer}.k` / `{layer}.v` for full attention,
    /// `{layer}.conv` / `{layer}.ssm` for DeltaNet; a cache not started yet is left out.
    #[must_use]
    pub fn arrays(&self) -> Vec<(String, Array)> {
        let mut out = Vec::new();
        for (i, state) in self.state.iter().enumerate() {
            let named: [(&str, Option<&Array>); 2] = match state {
                State::Kv { k, v } => [("k", k.as_ref()), ("v", v.as_ref())],
                State::Delta(d) => {
                    let (conv, ssm) = d.parts();
                    [("conv", conv), ("ssm", ssm)]
                }
            };
            for (name, array) in named {
                if let Some(a) = array {
                    out.push((format!("{i}.{name}"), a.clone()));
                }
            }
        }
        out
    }
}

/// The one-token path of a pack with every expert in RAM: about a quarter of the launches.
struct FusedDecode {
    fused: crate::fused::Fused,
    delta: deltanet::FusedKernels,
    /// Ternary records: their kernels with the shared expert. Affine records use the slab's.
    tern: Option<tern::FastKernels>,
    /// Per layer: the shared expert as a record, and each expert's slot in the layer's slab.
    layers: Vec<(Array, Array)>,
    zero: Array,
}

impl FusedDecode {
    /// The path for this model, if it fits: bf16 weights of the kernels' sizes, ternary records
    /// or affine records with the shared expert's bits.
    fn new(c: &Config, layers: &[Layer], stream: Option<&Streaming>) -> Result<Option<Self>> {
        let Some(Streaming {
            slab, all: Some(all), ..
        }) = stream
        else {
            return Ok(None);
        };
        let Some(first) = layers.first() else { return Ok(None) };
        let bits = first.shared.up.bits as usize;
        if matches!(slab.kernels, RecordKernels::Affine(_)) && c.expert_bits != Some(bits) {
            return Ok(None);
        }
        let hidden = crate::fused::HIDDEN as usize;
        let same = layers
            .iter()
            .all(|l| l.shared.up.bits == bits as i32 && l.router.group == 64 && l.shared.up.group == 64);
        if c.hidden != hidden
            || c.experts != 256
            || c.top_k != 8
            || c.group != 64
            || !same
            || !deltanet::FusedKernels::fits(c.delta)
        {
            return Ok(None);
        }
        let rec = AffineRecord {
            hidden,
            ffn: c.expert_ffn,
            bits,
        };
        let tern = match &slab.kernels {
            RecordKernels::Tern(k) => Some(k.fast(rec)),
            RecordKernels::Affine(_) => None,
        };
        let ne = c.experts as i32;
        let mut per = Vec::with_capacity(layers.len());
        for (i, l) in layers.iter().enumerate() {
            let s = &l.shared;
            let m = |q: &'_ QMat| -> [Array; 3] { [q.w.clone(), q.scales.clone(), q.biases.clone()] };
            let (g, u, d) = (m(&s.gate), m(&s.up), m(&s.down));
            let shared =
                affine::record_array(rec, [&g[0], &g[1], &g[2]], [&u[0], &u[1], &u[2]], [&d[0], &d[1], &d[2]])?;
            let table = all.index(i as i32 * ne..(i as i32 + 1) * ne).as_dtype(Dtype::Uint32)?;
            ironbark_mlx::transforms::eval([&shared, &table])?;
            per.push((shared, table));
        }
        Ok(Some(Self {
            fused: crate::fused::Fused::new(c.eps, c.theta),
            delta: deltanet::FusedKernels::new(c.delta, c.eps),
            tern,
            layers: per,
            zero: Array::zeros::<f32>(&[1, 1, crate::fused::HIDDEN])?.as_dtype(Dtype::Bfloat16)?,
        }))
    }
}

/// Qwen3.5 / 3.6 MoE on the GPU.
pub struct Model {
    pub cfg: Config,
    embed: QMat,
    norm: Array,
    head: QMat,
    layers: Vec<Layer>,
    state: Vec<State>,
    /// The DeltaNet recurrence kernel; `None` runs it with MLX ops.
    step: Option<crate::kernel::Kernel>,
    /// Per layer, while capturing ([`Model::set_capture`]): each forward's expert input
    /// `[1, L, hidden]`, choices `[1, L, k]` and weights `[1, L, k]`.
    capture: Option<Vec<Vec<[Array; 3]>>>,
    /// The layers captured.
    capture_layers: std::ops::Range<usize>,
    stream: Option<Streaming>,
    /// The one-token path, when the pack fits it.
    fast: Option<FusedDecode>,
    /// Streaming: predict and prefetch the experts of this many layers ahead (0 = off).
    pub lookahead: usize,
    /// Streaming: experts the last forward's kernels may still read, until `held_until` is
    /// evaluated.
    held: Vec<Pin>,
    held_until: Option<Array>,
    /// Time spent waiting for the per-layer routing readback while streaming.
    pub sync_ns: u64,
    /// Per layer, added to the router logits: 0 for the experts kept, −∞ for the rest
    /// ([`Model::keep_experts`]; trials of a pruned model). `None`: every expert.
    keep: Option<Vec<Array>>,
    /// Tokens in the cache.
    pub pos: usize,
    /// Text's RoPE position minus its cache position: after an image, positions go on from
    /// the image's start plus its longer side, not its token count ([`Model::prefill_image`]).
    pub delta: i64,
}

/// Experts in the store: its slab as the ternary kernels read it and, when the cache holds
/// every expert, the slot of each (no readback).
/// How the kernels read a pack's expert records.
enum RecordKernels {
    Tern(tern::Kernels),
    Affine(affine::Kernels),
}

/// The slab of expert records as the u32 array each layer's kernels read: its own view
/// `[per_layer, stride / 4]` with slots numbered from 0, or the whole slab with global slots;
/// with the kernels of its layout.
struct Slab {
    layers: Vec<Array>,
    kernels: RecordKernels,
}

impl Slab {
    /// Experts of `layer` in its slots `slots` `[S]` (u32) on rows `tok` `[S]` (u32) of `x`
    /// `[T, hidden]` (bf16): `[S, hidden]`. Affine records skip the pairs where `live` `[S]`
    /// (f32) is 0.
    fn apply(&self, x: &Array, layer: usize, slots: &Array, tok: &Array, live: &Array) -> Result<Array> {
        match &self.kernels {
            RecordKernels::Tern(k) => k.apply(x, &self.layers[layer], slots, tok),
            RecordKernels::Affine(k) => k.apply(x, &self.layers[layer], slots, tok, live),
        }
    }
}

struct Streaming {
    store: Arc<Store>,
    slab: Slab,
    all: Option<Array>,
}

impl Drop for Streaming {
    fn drop(&mut self) {
        self.store.close();
    }
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("layers", &self.cfg.layers)
            .field("pos", &self.pos)
            .finish_non_exhaustive()
    }
}

const PREFIX: &str = "language_model.model";

impl Model {
    /// Load an MLX checkpoint (mlx-lm's conversion: norms already `1 + w`, experts stacked)
    /// from `dir`, every weight resident.
    ///
    /// # Errors
    ///
    /// Fails if a file is missing or the tensors do not match the config.
    pub fn load(dir: &Path) -> Result<Self> {
        Self::load_with(dir, None)
    }

    /// Load a pack's slim weights from `dir` with the ternary experts of `experts` (written by
    /// [`pack_ternary`]): `per_layer` cache slots per layer (all experts: nothing streams),
    /// `loaders` reader threads.
    ///
    /// # Errors
    ///
    /// Fails if a file is missing or `experts` does not match the checkpoint.
    pub fn load_streaming(dir: &Path, experts: &Path, per_layer: usize, loaders: usize) -> Result<Self> {
        Self::load_records(dir, experts, per_layer, loaders, ironbark_experts::records::local(experts)?)
    }

    /// Like [`Model::load_streaming`], records read from `records`.
    ///
    /// # Errors
    ///
    /// Fails if a file is missing or the records do not match the checkpoint.
    pub fn load_records(
        dir: &Path,
        experts: &Path,
        per_layer: usize,
        loaders: usize,
        records: Arc<dyn Records>,
    ) -> Result<Self> {
        let cfg = Config::read(dir)?;
        let layout = Layout::read(experts)?;
        let (want, record_bytes, kernels) = match cfg.expert_bits {
            Some(bits) => {
                let rec = AffineRecord {
                    hidden: cfg.hidden,
                    ffn: cfg.expert_ffn,
                    bits,
                };
                let kernels = RecordKernels::Affine(affine::Kernels::new(rec, layout.stride as usize));
                (affine::format(bits), rec.bytes(), kernels)
            }
            None if cfg.expert_ternary => {
                let rec = record(&cfg);
                let kernels = RecordKernels::Tern(tern::Kernels::new(rec, layout.stride as usize));
                (tern::FORMAT.to_owned(), rec.bytes(), kernels)
            }
            None => return Err(Error::Format(format!("{}: not a Qwen pack (no expert records)", dir.display()))),
        };
        if crate::experts::format(experts)? != want {
            return Err(Error::Format(format!("{}: expert records are not {want} (re-run pack)", experts.display())));
        }
        if layout.layers != cfg.layers || layout.experts != cfg.experts || layout.record_bytes != record_bytes as u64 {
            return Err(Error::Format(format!("{}: does not match the checkpoint (re-run pack)", experts.display())));
        }
        if per_layer < cfg.top_k {
            return Err(Error::Format(format!("{per_layer} cache slots per layer: need at least {}", cfg.top_k)));
        }
        let store = Store::open_with(experts, per_layer, loaders, records)?;
        // Every expert cached: one view per layer (kernels reading a large buffer pay a fixed
        // cost per call). Streaming: the whole slab, with global slots.
        let local = store.per_layer() >= cfg.experts;
        let layers = if local {
            crate::kernel::wrap_slab_layers(&store, cfg.layers)?
        } else {
            vec![crate::kernel::wrap_slab(&store)?; cfg.layers]
        };
        ironbark_mlx::transforms::eval(layers.iter())?;
        let slab = Slab { layers, kernels };
        let all = if local {
            let table = crate::model::preload(&store, cfg.layers, cfg.experts)?;
            Some(tern::layer_local(&table, cfg.experts, store.per_layer())?)
        } else {
            None
        };
        Self::load_with(dir, Some(Streaming { store, slab, all }))
    }

    fn load_with(dir: &Path, stream: Option<Streaming>) -> Result<Self> {
        let cfg = Config::read(dir)?;
        let g = cfg.group;
        let mut t = Tensors(load_tensors(dir)?);
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("{PREFIX}.layers.{l}");
            let n = |s: &str| format!("{p}.{s}");
            let mixer = if cfg.full[l] {
                Mixer::Attn(Attention {
                    q: t.qmat(&n("self_attn.q_proj"), g)?,
                    k: t.qmat(&n("self_attn.k_proj"), g)?,
                    v: t.qmat(&n("self_attn.v_proj"), g)?,
                    o: t.qmat(&n("self_attn.o_proj"), g)?,
                    q_norm: t.take(&n("self_attn.q_norm.weight"))?,
                    k_norm: t.take(&n("self_attn.k_norm.weight"))?,
                })
            } else {
                Mixer::Delta(deltanet::DeltaNet::load(&mut t, &n("linear_attn"), g, cfg.delta)?)
            };
            let layer = Layer {
                in_norm: t.take(&n("input_layernorm.weight"))?,
                post_norm: t.take(&n("post_attention_layernorm.weight"))?,
                mixer,
                router: t.qmat(&n("mlp.gate"), g)?,
                // Experts in the store are never read from the checkpoint.
                experts: match &stream {
                    Some(_) => None,
                    None => Some(Experts {
                        gate: t.qmat(&n("mlp.switch_mlp.gate_proj"), g)?,
                        up: t.qmat(&n("mlp.switch_mlp.up_proj"), g)?,
                        down: t.qmat(&n("mlp.switch_mlp.down_proj"), g)?,
                    }),
                },
                shared: Shared {
                    gate: t.qmat(&n("mlp.shared_expert.gate_proj"), g)?,
                    up: t.qmat(&n("mlp.shared_expert.up_proj"), g)?,
                    down: t.qmat(&n("mlp.shared_expert.down_proj"), g)?,
                    weight: t.qmat(&n("mlp.shared_expert_gate"), g)?,
                },
            };
            ironbark_mlx::transforms::eval(layer.arrays())?;
            layers.push(layer);
        }
        // Mapped from `embed.bin` in a pack (rows are paged in as tokens use them).
        let embed = match crate::embed::map(dir, embed_shape(&cfg))? {
            Some(e) => e,
            None => t.qmat(&format!("{PREFIX}.embed_tokens"), g)?,
        };
        let norm = t.take(&format!("{PREFIX}.norm.weight"))?;
        let head = t.qmat("language_model.lm_head", g)?;
        ironbark_mlx::transforms::eval([
            &embed.w,
            &embed.scales,
            &embed.biases,
            &norm,
            &head.w,
            &head.scales,
            &head.biases,
        ])?;
        let fast = FusedDecode::new(&cfg, &layers, stream.as_ref())?;
        Ok(Model {
            state: cfg.full.iter().map(|&f| State::fresh(f)).collect(),
            cfg,
            embed,
            norm,
            head,
            layers,
            step: Some(deltanet::step_kernel()),
            capture: None,
            capture_layers: 0..0,
            stream,
            fast,
            lookahead: 1,
            held: Vec::new(),
            held_until: None,
            sync_ns: 0,
            keep: None,
            pos: 0,
            delta: 0,
        })
    }

    /// Bytes of the caches at `context` tokens: bf16 keys and values of the attention layers,
    /// plus each DeltaNet layer's conv tail (bf16) and fp32 state, which do not grow.
    #[must_use]
    pub fn state_bytes(cfg: &Config, context: usize) -> u64 {
        let kv = (2 * cfg.kv_heads * cfg.head_dim * context * 2) as u64;
        let d = cfg.delta;
        let delta = ((d.conv - 1) * d.conv_dim() * 2 + d.v_heads * d.v_dim * d.k_dim * 4) as u64;
        cfg.full.iter().map(|&f| if f { kv } else { delta }).sum()
    }

    /// Tokens per prefill forward.
    #[must_use]
    pub fn chunk(&self) -> usize {
        CHUNK
    }

    /// The caches, to return to with [`Model::restore`].
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            state: self.state.clone(),
            pos: self.pos,
            delta: self.delta,
        }
    }

    /// A [`Snapshot`] of this model from its saved [`Snapshot::arrays`], at `pos` with
    /// `delta`.
    ///
    /// # Errors
    ///
    /// Fails if a layer's caches are missing (`pos` > 0) or do not match this model.
    pub fn snapshot_from(&self, arrays: &HashMap<String, Array>, pos: usize, delta: i64) -> Result<Snapshot> {
        let get = |i: usize, name: &str| arrays.get(&format!("{i}.{name}")).cloned();
        let mut state = Vec::with_capacity(self.state.len());
        for (i, kind) in self.state.iter().enumerate() {
            let s = match kind {
                State::Kv { .. } => State::Kv {
                    k: get(i, "k"),
                    v: get(i, "v"),
                },
                State::Delta(_) => State::Delta(deltanet::State::from_parts(get(i, "conv"), get(i, "ssm"))),
            };
            if pos > 0 && s.arrays().len() != 2 {
                return Err(Error::Format(format!("saved caches of layer {i} missing")));
            }
            state.push(s);
        }
        if arrays.len() != state.iter().map(|s| s.arrays().len()).sum::<usize>() {
            return Err(Error::Format("saved caches of another model".into()));
        }
        Ok(Snapshot { state, pos, delta })
    }

    /// Where to [`Model::rewind`] to: the position and the DeltaNet states (recurrent);
    /// full-attention layers keep their keys up to it. Cheap: arrays shared.
    #[must_use]
    pub fn rewind_point(&self) -> Snapshot {
        let state = (self.state.iter())
            .map(|s| match s {
                State::Delta(_) => s.clone(),
                State::Kv { .. } => State::Kv { k: None, v: None },
            })
            .collect();
        Snapshot {
            state,
            pos: self.pos,
            delta: self.delta,
        }
    }

    /// A snapshot of the caches as [`Model::rewind`] to `p` would leave them, without
    /// rewinding.
    #[must_use]
    pub fn snapshot_at(&self, p: &Snapshot) -> Snapshot {
        let state = (self.state.iter().zip(&p.state))
            .map(|(now, saved)| match saved {
                State::Delta(_) => saved.clone(),
                State::Kv { .. } => now.clone(),
            })
            .collect();
        Snapshot {
            state,
            pos: p.pos,
            delta: p.delta,
        }
    }

    /// Go back to `p`, a [`Model::rewind_point`] taken since the last reset or restore,
    /// forgetting the tokens after it.
    pub fn rewind(&mut self, p: &Snapshot) {
        // A failed wait leaves nothing to protect: the output is discarded with it.
        let _ = self.release_held();
        for (s, saved) in self.state.iter_mut().zip(&p.state) {
            if let State::Delta(_) = saved {
                s.clone_from(saved);
            }
        }
        self.pos = p.pos;
        self.delta = p.delta;
    }

    /// Return to a [`Snapshot`].
    pub fn restore(&mut self, s: &Snapshot) {
        // A failed wait leaves nothing to protect: the output is discarded with it.
        let _ = self.release_held();
        self.state.clone_from(&s.state);
        self.pos = s.pos;
        self.delta = s.delta;
    }

    /// The expert store, when experts are in one.
    #[must_use]
    pub fn store(&self) -> Option<&Arc<Store>> {
        self.stream.as_ref().map(|s| &s.store)
    }

    /// The sequences saved in `snaps` (each a prompt prefilled from an empty cache) as one
    /// batch, to decode together with [`Model::step_batch`].
    ///
    /// # Errors
    ///
    /// Fails if `snaps` is empty, a snapshot has no tokens, or an MLX operation fails.
    pub fn batch(&self, snaps: &[Snapshot]) -> Result<Batch> {
        let lens: Vec<usize> = snaps.iter().map(|s| s.pos).collect();
        let deltas: Vec<i64> = snaps.iter().map(|s| s.delta).collect();
        let pad = lens.iter().copied().max().filter(|&p| p > 0 && lens.iter().all(|&l| l > 0));
        let pad = pad.ok_or_else(|| Error::Format("batch: needs prefilled sequences".into()))?;
        let mut state = Vec::with_capacity(self.cfg.layers);
        for i in 0..self.cfg.layers {
            let rows: Vec<&State> = snaps.iter().map(|s| &s.state[i]).collect();
            state.push(match rows[0] {
                State::Kv { .. } => {
                    // Each row's keys and values up to its length, then zeros up to `pad`.
                    let (mut ks, mut vs) = (Vec::new(), Vec::new());
                    for (r, &len) in rows.iter().zip(&lens) {
                        let State::Kv { k: Some(k), v: Some(v) } = r else {
                            return Err(Error::Format("batch: attention cache missing".into()));
                        };
                        let fill = |a: &Array| -> Result<Array> {
                            let a = a.index((.., .., ..len as i32));
                            if len == pad {
                                return Ok(a);
                            }
                            let s = a.shape();
                            let z =
                                Array::zeros::<f32>(&[s[0], s[1], (pad - len) as i32, s[3]])?.as_dtype(a.dtype())?;
                            Ok(concatenate(&[&a, &z], 2)?)
                        };
                        ks.push(fill(k)?);
                        vs.push(fill(v)?);
                    }
                    State::Kv {
                        k: Some(concatenate(&ks, 0)?),
                        v: Some(concatenate(&vs, 0)?),
                    }
                }
                State::Delta(_) => {
                    let parts: Vec<&deltanet::State> = rows
                        .iter()
                        .map(|r| match r {
                            State::Delta(d) => Ok(d),
                            State::Kv { .. } => Err(Error::Format("batch: layer kinds differ".into())),
                        })
                        .collect::<Result<_>>()?;
                    State::Delta(deltanet::State::stack(&parts)?)
                }
            });
        }
        Ok(Batch {
            state,
            lens,
            deltas,
            pad,
            steps: 0,
        })
    }

    /// One token per sequence of `b` → logits `[B, vocab]` (lazy). Experts must be resident or
    /// all cached (a batch does not stream).
    ///
    /// # Errors
    ///
    /// Fails if `tokens` does not have one id per sequence, experts stream, or an MLX operation
    /// fails.
    pub fn step_batch(&mut self, b: &mut Batch, tokens: &[u32]) -> Result<Array> {
        let n = tokens.len();
        if n != b.lens.len() {
            return Err(Error::Format(format!("step_batch: {n} tokens for {} sequences", b.lens.len())));
        }
        let (c, bn) = (&self.cfg, n as i32);
        let e = &self.embed;
        let ids = Array::from_slice(tokens, &[bn]);
        let mut h = dequantize(
            e.w.take_axis(&ids, 0)?,
            e.scales.take_axis(&ids, 0)?,
            &e.biases.take_axis(&ids, 0)?,
            e.group,
            e.bits,
        )?
        .reshape(&[1, bn, -1])?;
        // Keys a row may see: its prompt, then everything decoded together (not the padding).
        let width = b.pad + b.steps + 1;
        let mut see = vec![false; n * width];
        for (r, &len) in b.lens.iter().enumerate() {
            for j in 0..width {
                see[r * width + j] = j < len || j >= b.pad;
            }
        }
        let mask = Array::from_slice(&see, &[bn, 1, 1, width as i32]);
        let offsets: Vec<i32> =
            b.lens.iter().zip(&b.deltas).map(|(&l, &d)| ((l + b.steps) as i64 + d) as i32).collect();
        let offsets = Array::from_slice(&offsets, &[bn]);
        let at = (b.pad + b.steps) as i32;
        for i in 0..self.layers.len() {
            let ly = &self.layers[i];
            let x = fast::rms_norm(&h, Some(&ly.in_norm), c.eps)?.reshape(&[bn, 1, -1])?;
            let m = match (&ly.mixer, &mut b.state[i]) {
                (Mixer::Attn(a), State::Kv { k, v }) => attention_batch(c, a, k, v, &x, &offsets, &mask, at)?,
                (Mixer::Delta(d), State::Delta(s)) => d.forward(&x, s, c.eps, self.step.as_ref())?,
                _ => return Err(Error::Format("layer state does not match its mixer".into())),
            };
            h = h.add(m.reshape(&[1, bn, -1])?)?;
            let x = fast::rms_norm(&h, Some(&ly.post_norm), c.eps)?;
            // A batched decode step: one token per sequence, so dynamic top-k applies.
            let (idx, w) = route_logits(c, &keep_bias(self.keep.as_deref(), i, ly.router.apply(&x)?)?, true)?;
            let routed = match (&ly.experts, &self.stream) {
                (Some(ex), _) => experts(ex, &x, &idx)?,
                (None, Some(st)) if st.all.is_some() => {
                    let ne = c.experts as i32;
                    let table = st.all.as_ref().map(|t| t.index(i as i32 * ne..(i as i32 + 1) * ne));
                    let slots = table.ok_or_else(|| Error::Format("no slot table".into()))?.take_axis(&idx, 0)?;
                    slab_experts(&st.slab, i, &x, &slots, &w)?
                }
                _ => return Err(Error::Format("step_batch: experts must be resident or all cached".into())),
            };
            h = h.add(combine(ly, &x, &routed, &w)?)?;
        }
        b.steps += 1;
        let logits = self.head.apply(&fast::rms_norm(&h, Some(&self.norm), c.eps)?)?;
        Ok(logits.reshape(&[bn, -1])?)
    }

    /// Let the slots of the last forward's experts go, once its kernels are done.
    fn release_held(&mut self) -> Result<()> {
        if let Some(a) = self.held_until.take() {
            a.eval()?;
        }
        self.held.clear();
        Ok(())
    }

    /// Keep every forward's expert inputs and routing (e.g. to calibrate a ternary export),
    /// or stop and drop them.
    pub fn set_capture(&mut self, on: bool) {
        self.set_capture_layers(on.then_some(0..self.cfg.layers));
    }

    /// Keep the expert inputs and routing of the layers `layers` only (the rest are not kept,
    /// so a long calibration text fits in memory a few layers at a time), or stop with `None`.
    pub fn set_capture_layers(&mut self, layers: Option<std::ops::Range<usize>>) {
        self.capture = layers.map(|r| {
            self.capture_layers = r;
            vec![Vec::new(); self.cfg.layers]
        });
    }

    /// Run layer `l`'s experts from the MLX checkpoint `dir` (any bits) instead of the store,
    /// or from the store again with `None`: what one layer's ternary costs.
    ///
    /// # Errors
    ///
    /// Fails if the experts are resident (not streamed), `l` is out of range, the checkpoint
    /// lacks the layer's experts or an MLX operation fails.
    pub fn float_layer(&mut self, l: usize, dir: Option<&Path>) -> Result<()> {
        if self.stream.is_none() {
            return Err(Error::Format("float_layer: the experts are resident".into()));
        }
        if l >= self.layers.len() {
            return Err(Error::Format(format!("float_layer: no layer {l}")));
        }
        let experts = match dir {
            None => None,
            Some(dir) => {
                let g = Config::read(dir)?.group;
                let mut t = Tensors(load_tensors(dir)?);
                let n = |m: &str| format!("{PREFIX}.layers.{l}.mlp.switch_mlp.{m}_proj");
                let ex = Experts {
                    gate: t.qmat(&n("gate"), g)?,
                    up: t.qmat(&n("up"), g)?,
                    down: t.qmat(&n("down"), g)?,
                };
                ironbark_mlx::transforms::eval(
                    [&ex.gate, &ex.up, &ex.down].into_iter().flat_map(|q| [&q.w, &q.scales, &q.biases]),
                )?;
                Some(ex)
            }
        };
        self.layers[l].experts = experts;
        Ok(())
    }

    /// Route only to the experts `keep[l]` of each layer `l` (as a model with the rest pruned
    /// would), or to every expert with `None`.
    ///
    /// # Errors
    ///
    /// Fails if `keep` does not have one list per layer, a list has fewer than top-k experts
    /// or an id out of range, or an MLX operation fails.
    pub fn keep_experts(&mut self, keep: Option<&[Vec<usize>]>) -> Result<()> {
        let (ne, k) = (self.cfg.experts, self.cfg.top_k);
        self.keep = match keep {
            None => None,
            Some(keep) => {
                if keep.len() != self.cfg.layers {
                    return Err(Error::Format(format!(
                        "keep_experts: {} lists for {} layers",
                        keep.len(),
                        self.cfg.layers
                    )));
                }
                let mut bias = Vec::with_capacity(keep.len());
                for (l, ids) in keep.iter().enumerate() {
                    let mut b = vec![f32::NEG_INFINITY; ne];
                    for &e in ids {
                        *b.get_mut(e)
                            .ok_or_else(|| Error::Format(format!("keep_experts: layer {l}: no expert {e}")))? = 0.0;
                    }
                    if b.iter().filter(|&&x| x == 0.0).count() < k {
                        return Err(Error::Format(format!("keep_experts: layer {l}: fewer than {k} experts")));
                    }
                    let b = Array::from_slice(&b, &[ne as i32]).as_dtype(Dtype::Bfloat16)?;
                    b.eval()?;
                    bias.push(b);
                }
                Some(bias)
            }
        };
        Ok(())
    }

    /// What was captured since [`Model::set_capture`], per layer and forward: `[x, idx, w]`
    /// (empty for the layers not captured). Capture goes on, into empty lists.
    pub fn take_capture(&mut self) -> Vec<Vec<[Array; 3]>> {
        let layers = self.cfg.layers;
        self.capture.as_mut().map(|c| std::mem::replace(c, vec![Vec::new(); layers])).unwrap_or_default()
    }

    /// Layer `l`'s experts dequantized, stacked `[experts, rows, cols]` (gate, up, down).
    ///
    /// # Errors
    ///
    /// Fails if `l` is out of range or an MLX operation fails.
    pub fn expert_weights(&self, l: usize) -> Result<[Array; 3]> {
        let ly = self.layers.get(l).ok_or_else(|| Error::Format(format!("no layer {l}")))?;
        let x = ly.experts.as_ref().ok_or_else(|| Error::Format("experts are not resident".into()))?;
        let dq = |m: &QMat| -> Result<Array> { Ok(dequantize(&m.w, &m.scales, &m.biases, m.group, m.bits)?) };
        Ok([dq(&x.gate)?, dq(&x.up)?, dq(&x.down)?])
    }

    /// The token embeddings of `ids` → `[1, L, hidden]` (lazy).
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails.
    pub fn embed(&self, ids: &[u32]) -> Result<Array> {
        Ok(self.embed_ids(&Array::from_slice(ids, &[ids.len() as i32]))?.expand_dims(0)?)
    }

    /// Layer `l` on whole sequences `h` `[1, L, hidden]` (from position 0, a fresh state, the
    /// caches left alone) with its routed experts `ex` (gate, up, down: each MLX affine
    /// weights, scales and biases at `bits`) → the residual after it (lazy). With `kernel` the
    /// DeltaNet recurrence is one Metal launch; without, MLX ops, so the whole layer is
    /// differentiable in `h` and the experts' scales and biases (training).
    ///
    /// # Errors
    ///
    /// Fails if `l` is out of range or an MLX operation fails.
    pub fn layer_with(&self, l: usize, h: &Array, ex: [[&Array; 3]; 3], bits: i32, kernel: bool) -> Result<Array> {
        let c = &self.cfg;
        let ly = self.layers.get(l).ok_or_else(|| Error::Format(format!("no layer {l}")))?;
        let x = fast::rms_norm(h, Some(&ly.in_norm), c.eps)?;
        let m = match &ly.mixer {
            Mixer::Attn(a) => attention(c, a, &mut None, &mut None, &x, 0, &Rope::Offset(0))?,
            Mixer::Delta(d) => {
                let step = if kernel { self.step.as_ref() } else { None };
                d.forward(&x, &mut deltanet::State::default(), c.eps, step)?
            }
        };
        let h = h.add(&m)?;
        let x = fast::rms_norm(&h, Some(&ly.post_norm), c.eps)?;
        let (idx, w) = route_logits(c, &keep_bias(self.keep.as_deref(), l, ly.router.apply(&x)?)?, false)?;
        let q = |[w, scales, biases]: [&Array; 3]| QMat {
            w: w.clone(),
            scales: scales.clone(),
            biases: biases.clone(),
            group: c.group as i32,
            bits,
        };
        let [gate, up, down] = ex;
        let ex = Experts {
            gate: q(gate),
            up: q(up),
            down: q(down),
        };
        Ok(h.add(&combine(ly, &x, &experts(&ex, &x, &idx)?, &w)?)?)
    }

    /// The logits of residuals `h` `[1, L, hidden]` (final norm and head; lazy).
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails.
    pub fn logits(&self, h: &Array) -> Result<Array> {
        self.head.apply(&fast::rms_norm(h, Some(&self.norm), self.cfg.eps)?)
    }

    /// The cache arrays (to evaluate with the next token, so none stays lazy).
    #[must_use]
    pub fn state_arrays(&self) -> Vec<&Array> {
        self.state.iter().flat_map(State::arrays).collect()
    }

    /// Clear the caches and drop every queued step without running it: after a failed GPU
    /// command buffer, waiting on its arrays fails again.
    pub fn discard(&mut self) {
        self.held_until = None;
        self.held.clear();
        self.reset();
    }

    /// Clear the caches.
    pub fn reset(&mut self) {
        let _ = self.release_held();
        self.state = self.cfg.full.iter().map(|&f| State::fresh(f)).collect();
        self.pos = 0;
        self.delta = 0;
    }

    /// One token → logits `[1, 1, vocab]` (lazy).
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails.
    pub fn step(&mut self, tok: u32) -> Result<Array> {
        self.step_lazy(&Array::from_slice(&[tok], &[1]))
    }

    /// Like [`Model::step`], the token an MLX array `[1]` (u32) that may be unevaluated.
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails.
    pub fn step_lazy(&mut self, ids: &Array) -> Result<Array> {
        self.forward(ids, false)
    }

    /// Prompt tokens in chunks of [`Model::chunk`]: logits of the last token `[1, 1, vocab]`,
    /// or with `all` of every token `[1, n, vocab]`.
    ///
    /// # Errors
    ///
    /// Fails if `ids` is empty or an MLX operation fails.
    pub fn prefill(&mut self, ids: &[u32], all: bool) -> Result<Array> {
        self.prefill_with(ids, None, all)
    }

    /// [`Model::prefill`] of a prompt with images: each run of `image_token` in `ids` is the
    /// next of `images` ([`vision::Vision::encode`]), whose embeddings replace those tokens'.
    /// The prompt takes 3D positions ([`vision::positions`]); text after it goes on from where
    /// the last image leaves off ([`Model::delta`]).
    ///
    /// # Errors
    ///
    /// Fails if `ids` is empty, the image tokens do not match `images`, or an MLX operation
    /// fails.
    pub fn prefill_image(
        &mut self,
        ids: &[u32],
        images: &[vision::Image],
        image_token: u32,
        all: bool,
    ) -> Result<Array> {
        if ids.is_empty() {
            return Err(Error::Format("prefill: no tokens".into()));
        }
        let start = self.pos as i64 + self.delta;
        let grids: Vec<(usize, usize)> = images.iter().map(|m| (m.rows, m.cols)).collect();
        let (p, next) = vision::positions(ids, image_token, &grids, start)?;
        let c = &self.cfg;
        let (cos, sin) = vision::mrope_tables(&p, c.rotary, c.theta, c.mrope_section);
        // Text runs embedded, image runs from the encoder.
        let text = self.embed_ids(&Array::from_slice(ids, &[ids.len() as i32]))?;
        let mut parts = Vec::new();
        let (mut i, mut next_image) = (0, images.iter());
        while i < ids.len() {
            if ids[i] == image_token {
                let m = next_image.next().ok_or_else(|| Error::Format("image tokens without an image".into()))?;
                if m.embeds.shape()[1] as usize != c.hidden {
                    return Err(Error::Format("image embeddings do not match the model".into()));
                }
                parts.push(m.embeds.as_dtype(text.dtype())?);
                i += m.tokens();
            } else {
                let run = ids[i..].iter().take_while(|&&t| t != image_token).count();
                parts.push(text.index(i as i32..(i + run) as i32));
                i += run;
            }
        }
        let h = concatenate(&parts, 0)?.expand_dims(0)?;
        let half = (c.rotary / 2) as i32;
        let table = |v: &[f32]| Array::from_slice(v, &[1, 1, ids.len() as i32, half]);
        let mm = Multimodal {
            h,
            cos: table(&cos),
            sin: table(&sin),
        };
        let logits = self.prefill_with(ids, Some(&mm), all)?;
        self.delta = next - self.pos as i64;
        Ok(logits)
    }

    /// [`Model::prefill`], the input embeddings and RoPE tables from `mm` if given.
    fn prefill_with(&mut self, ids: &[u32], mm: Option<&Multimodal>, all: bool) -> Result<Array> {
        if ids.is_empty() {
            return Err(Error::Format("prefill: no tokens".into()));
        }
        let mut outs = Vec::new();
        let chunk = self.chunk();
        for (k, c) in ids.chunks(chunk).enumerate() {
            let ids = Array::from_slice(c, &[c.len() as i32]);
            let part = match mm {
                Some(m) => {
                    let r = (k * chunk) as i32..(k * chunk + c.len()) as i32;
                    Some(Multimodal {
                        h: m.h.index((.., r.clone())),
                        cos: m.cos.index((.., .., r.clone())),
                        sin: m.sin.index((.., .., r)),
                    })
                }
                None => None,
            };
            let logits = self.forward_traced(&ids, part.as_ref(), all, None)?;
            let mut ev = self.state_arrays();
            ev.push(&logits);
            // What this chunk captured (each layer's newest entry).
            ev.extend(self.capture.iter().flatten().filter_map(|c| c.last()).flatten());
            ironbark_mlx::transforms::eval(ev)?;
            ironbark_mlx::memory::clear_cache()?;
            outs.push(logits);
        }
        if all {
            Ok(concatenate(&outs, 1)?)
        } else {
            outs.pop().ok_or_else(|| Error::Format("prefill: no tokens".into()))
        }
    }

    /// The token embeddings of `ids` `[L]` → `[L, hidden]`.
    fn embed_ids(&self, ids: &Array) -> Result<Array> {
        let e = &self.embed;
        Ok(dequantize(
            e.w.take_axis(ids, 0)?,
            e.scales.take_axis(ids, 0)?,
            &e.biases.take_axis(ids, 0)?,
            e.group,
            e.bits,
        )?)
    }

    fn forward(&mut self, ids: &Array, all: bool) -> Result<Array> {
        self.forward_traced(ids, None, all, None)
    }

    /// One token through [`FusedDecode`]; the residual stream is carried as (h, hn): every
    /// add + norm pair is one kernel.
    fn step_fused(&mut self, ids: &Array) -> Result<Array> {
        let (Some(f), Some(st)) = (&self.fast, &self.stream) else {
            return Err(Error::Format("no fused decode".into()));
        };
        let c = &self.cfg;
        let pos = self.pos as i32;
        let rope = Rope::Offset((self.pos as i64 + self.delta) as i32);
        let x = self.embed_ids(ids)?.expand_dims(0)?;
        let (mut h, mut hn) = f.fused.add_norm(&x, &f.zero, &self.layers[0].in_norm)?;
        for i in 0..self.layers.len() {
            let ly = &self.layers[i];
            let r = match (&ly.mixer, &mut self.state[i]) {
                (Mixer::Attn(a), State::Kv { k, v }) => attention(c, a, k, v, &hn, pos, &rope)?,
                (Mixer::Delta(d), State::Delta(s)) => d.step_fused(&hn, s, &f.delta)?,
                _ => return Err(Error::Format("layer state does not match its mixer".into())),
            };
            let m = &ly.router;
            let (h2, hn2, mut ids, mut scores) =
                f.fused.add_norm_affine_router(&h, &r, &ly.post_norm, &m.w, &m.scales, &m.biases, m.bits)?;
            if c.dynamic.on(c.top_k) {
                (ids, scores) = c.dynamic.apply(&ids, &scores)?;
            }
            let (shared, table) = &f.layers[i];
            let slab = &st.slab.layers[i];
            let y = match (&f.tern, &st.slab.kernels) {
                (Some(k), _) => k.apply(&hn2, slab, shared, table, &ids, &scores)?,
                (None, RecordKernels::Affine(k)) => k.apply_with_shared(&hn2, slab, shared, table, &ids, &scores)?,
                (None, RecordKernels::Tern(_)) => return Err(Error::Format("ternary records without kernels".into())),
            };
            let gate = ly.shared.weight.apply(&hn2)?;
            let next = self.layers.get(i + 1).map_or(&self.norm, |l| &l.in_norm);
            (h, hn) = f.fused.agg_shared(&h2, &y, &scores, &gate, next)?;
        }
        self.pos += 1;
        self.head.apply(&hn)
    }

    /// [`Model::forward`], pushing each layer's mixer output, MoE output and residual to
    /// `trace` (to compare with a reference layer by layer).
    #[allow(clippy::too_many_lines, reason = "one pass over the layers")]
    /// With `mm`, the input embeddings `[1, L, hidden]` replace the ids' and RoPE uses the
    /// tables; else positions go on from the cache's (plus [`Model::delta`]).
    fn forward_traced(
        &mut self,
        ids: &Array,
        mm: Option<&Multimodal>,
        all: bool,
        mut trace: Option<&mut Vec<[Array; 3]>>,
    ) -> Result<Array> {
        let n = ids.shape()[0];
        if n == 1
            && mm.is_none()
            && trace.is_none()
            && self.fast.is_some()
            && self.capture.is_none()
            && self.keep.is_none()
            && self.layers.iter().all(|l| l.experts.is_none())
        {
            return self.step_fused(ids);
        }
        // Dynamic top-k on decode steps (one token) only.
        let dynamic = n == 1;
        let mut h = match mm {
            Some(m) => m.h.clone(),
            None => self.embed_ids(ids)?.expand_dims(0)?,
        };
        let pos = self.pos as i32;
        let rope = match mm {
            Some(m) => Rope::Table(&m.cos, &m.sin),
            None => Rope::Offset((self.pos as i64 + self.delta) as i32),
        };
        let c = &self.cfg;
        let store = self.stream.as_ref().map(|s| s.store.clone());
        // Streaming: experts of the layer whose kernels may still run, pinned until the next
        // readback (which waits for those kernels).
        let mut pins: Vec<ExpertRef<'_>> = Vec::new();
        for i in 0..self.layers.len() {
            let ly = &self.layers[i];
            let x = fast::rms_norm(&h, Some(&ly.in_norm), c.eps)?;
            let m = match (&ly.mixer, &mut self.state[i]) {
                (Mixer::Attn(a), State::Kv { k, v }) => attention(c, a, k, v, &x, pos, &rope)?,
                (Mixer::Delta(d), State::Delta(s)) => d.forward(&x, s, c.eps, self.step.as_ref())?,
                _ => return Err(Error::Format("layer state does not match its mixer".into())),
            };
            h = h.add(&m)?;
            let x = fast::rms_norm(&h, Some(&ly.post_norm), c.eps)?;
            let logits = keep_bias(self.keep.as_deref(), i, ly.router.apply(&x)?)?;
            let (idx, w);
            let routed = match (&ly.experts, &self.stream, &store) {
                (Some(ex), ..) => {
                    (idx, w) = route_logits(c, &logits, dynamic)?;
                    experts(ex, &x, &idx)?
                }
                (None, Some(st), _) if st.all.is_some() => {
                    (idx, w) = route_logits(c, &logits, dynamic)?;
                    let ne = c.experts as i32;
                    let table = st.all.as_ref().map(|t| t.index(i as i32 * ne..(i as i32 + 1) * ne));
                    let slots = table.ok_or_else(|| Error::Format("no slot table".into()))?.take_axis(&idx, 0)?;
                    slab_experts(&st.slab, i, &x, &slots, &w)?
                }
                (None, Some(st), Some(store)) => {
                    (idx, w) = route_logits(c, &logits, dynamic)?;
                    // Predict the next layers' experts from this residual (their norms and
                    // routers on it); read them back with this layer's choice, in one sync.
                    let mut preds = Vec::new();
                    for (j, nl) in self.layers.iter().enumerate().skip(i + 1).take(self.lookahead) {
                        let xn = fast::rms_norm(&h, Some(&nl.post_norm), c.eps)?;
                        // As this step routes: dropped choices repeat the strongest (no prefetch).
                        preds.push((j, route_logits(c, &nl.router.apply(&xn)?, dynamic)?.0.contiguous()?));
                    }
                    let idx = idx.contiguous()?;
                    // Dynamic top-k: the weights too, so dropped choices (weight 0) skip the kernels.
                    let wf = crate::dynamic::read_weights(&c.dynamic, c.top_k, dynamic, &w)?;
                    let mut wanted = vec![&idx];
                    wanted.extend(&wf);
                    wanted.extend(preds.iter().map(|(_, p)| p));
                    let t = std::time::Instant::now();
                    ironbark_mlx::transforms::eval(wanted)?;
                    let dropped = crate::dynamic::dropped(wf.as_ref());
                    self.sync_ns += t.elapsed().as_nanos() as u64;
                    // The GPU is past the previous layer, and past the previous forward.
                    pins.clear();
                    self.held.clear();
                    self.held_until = None;
                    let chosen: Vec<usize> = idx.as_slice::<u32>().iter().map(|&e| e as usize).collect();
                    // Distinct experts by their last use, so when a long prompt needs several
                    // groups the last one (which stays cached) holds the newest tokens' experts.
                    let mut last = vec![usize::MAX; c.experts];
                    for (p, &e) in chosen.iter().enumerate() {
                        last[e] = p;
                    }
                    let mut unique: Vec<usize> = (0..c.experts).filter(|&e| last[e] != usize::MAX).collect();
                    unique.sort_unstable_by_key(|&e| last[e]);
                    // The first group's loads (all of them, unless a long prompt wants more than fit).
                    for &e in unique.iter().take(store.per_layer()) {
                        store.request(i, e);
                    }
                    for (j, p) in &preds {
                        for &e in p.as_slice::<u32>() {
                            store.prefetch(*j, e as usize);
                        }
                    }
                    // More distinct experts than the layer's slots (a long prompt): run them in
                    // groups that fit, each group's kernels done before its slots go.
                    let k = c.top_k;
                    let x2 = x.reshape(&[n, -1])?;
                    let groups: Vec<&[usize]> = unique.chunks(store.per_layer()).collect();
                    let (mut outs, mut order) = (Vec::with_capacity(groups.len()), Vec::with_capacity(chosen.len()));
                    for (gi, group) in groups.iter().enumerate() {
                        if gi > 0 {
                            pins.clear();
                            for &e in *group {
                                store.request(i, e);
                            }
                        }
                        let mut slot = vec![u32::MAX; c.experts];
                        for &e in *group {
                            let r = store.wait(i, e)?;
                            slot[e] = r.slot() as u32;
                            pins.push(r);
                        }
                        let pairs: Vec<usize> = (0..chosen.len())
                            .filter(|&p| slot[chosen[p]] != u32::MAX && !dropped.get(p).copied().unwrap_or(false))
                            .collect();
                        let slots: Vec<u32> = pairs.iter().map(|&p| slot[chosen[p]]).collect();
                        let tok: Vec<u32> = pairs.iter().map(|&p| (p / k) as u32).collect();
                        let m = pairs.len() as i32;
                        let y = st.slab.apply(
                            &x2,
                            i,
                            &Array::from_slice(&slots, &[m]),
                            &Array::from_slice(&tok, &[m]),
                            // Dropped pairs are not in this group.
                            &Array::from_slice(&vec![1.0f32; pairs.len()], &[m]),
                        )?;
                        if gi + 1 < groups.len() {
                            y.eval()?;
                        }
                        outs.push(y);
                        order.extend(pairs.iter().map(|&p| p as u32));
                    }
                    crate::dynamic::pairs_in_order(outs, &order, chosen.len())?.reshape(&[1, n, k as i32, -1])?
                }
                _ => return Err(Error::Format("a layer without resident experts needs a store".into())),
            };
            let y = combine(ly, &x, &routed, &w)?;
            h = h.add(&y)?;
            if let Some(cap) = self.capture.as_mut().filter(|_| self.capture_layers.contains(&i)) {
                cap[i].push([x, idx, w]);
            }
            if let Some(t) = trace.as_deref_mut() {
                t.push([m, y, h.clone()]);
            }
        }
        self.pos += n as usize;
        let h = if all { h } else { h.index((.., n - 1..n)) };
        let logits = self.head.apply(&fast::rms_norm(&h, Some(&self.norm), c.eps)?)?;
        if let (Some(st), Some(_)) = (&store, &self.stream) {
            self.held = pins.iter().map(|r| Pin::new(st, r)).collect();
            self.held_until = Some(logits.clone());
            pins.clear();
        }
        Ok(logits)
    }
}

/// The experts of `layer` in slab slots `slots` `[1, L, k]` with routing weights `w` on `x`
/// `[1, L, hidden]` → `[1, L, k, hidden]`.
fn slab_experts(slab: &Slab, layer: usize, x: &Array, slots: &Array, w: &Array) -> Result<Array> {
    let s = slots.shape();
    let (l, k) = (s[1], s[2]);
    let flat = slots.flatten(None, None)?.as_dtype(Dtype::Uint32)?;
    let tok: Vec<u32> = (0..(l * k) as u32).map(|i| i / k as u32).collect();
    let tok = Array::from_slice(&tok, &[l * k]);
    let live = w.flatten(None, None)?.as_dtype(Dtype::Float32)?;
    let y = slab.apply(&x.reshape(&[l, -1])?, layer, &flat, &tok, &live)?;
    Ok(y.reshape(&[1, l, k, -1])?)
}

/// The routed experts' outputs `[1, L, k, hidden]` weighted by `w` and summed, plus the gated
/// shared expert on `x`.
fn combine(ly: &Layer, x: &Array, routed: &Array, w: &Array) -> Result<Array> {
    let y = routed.multiply(w.expand_dims(-1)?)?.sum_axis(-2, false)?;
    let s = &ly.shared;
    let shared = s.down.apply(&silu(s.gate.apply(x)?)?.multiply(s.up.apply(x)?)?)?;
    Ok(y.add(sigmoid(s.weight.apply(x)?)?.multiply(&shared)?)?)
}

/// The resident routed experts plus the gated shared expert on `x` `[1, L, hidden]`, with the
/// routing (ids and weights `[1, L, k]`).
#[cfg(test)]
fn moe(c: &Config, ly: &Layer, x: &Array) -> Result<(Array, Array, Array)> {
    let (idx, w) = route(c, &ly.router, x)?;
    let ex = ly.experts.as_ref().ok_or_else(|| Error::Format("experts are not resident".into()))?;
    let y = combine(ly, x, &experts(ex, x, &idx)?, &w)?;
    Ok((y, idx, w))
}

/// Softmax over every expert (fp32 inside), the top-k and their weights (renormalized if
/// the config says so): ids `[1, L, k]` and weights.
#[cfg(test)]
fn route(c: &Config, router: &QMat, x: &Array) -> Result<(Array, Array)> {
    route_logits(c, &router.apply(x)?, false)
}

/// Router `logits` of layer `l` with the [`Model::keep_experts`] bias added, if any.
fn keep_bias(keep: Option<&[Array]>, l: usize, logits: Array) -> Result<Array> {
    Ok(match keep {
        Some(b) => logits.add(&b[l])?,
        None => logits,
    })
}

/// [`route`] from the router's logits (`−∞` for experts that may not be chosen); with `dynamic`
/// (a decode step) the weak choices dropped by the config's [`Config::dynamic`].
fn route_logits(c: &Config, logits: &Array, dynamic: bool) -> Result<(Array, Array)> {
    let p = ironbark_mlx::ops::softmax_axis(logits, -1, true)?;
    let k = c.top_k as i32;
    // The choice is not differentiable: training takes gradients through the weights only.
    let idx = ironbark_mlx::transforms::stop_gradient(argpartition_axis(&p, -k, -1)?.index((
        ..,
        ..,
        c.experts as i32 - k..,
    )))?;
    let mut w = p.take_along_axis(&idx, -1)?;
    if c.norm_top_k {
        w = w.divide(w.sum_axis(-1, true)?)?;
    }
    if dynamic && c.dynamic.on(c.top_k) {
        return c.dynamic.apply(&idx, &w);
    }
    Ok((idx, w))
}

/// mlx-lm `SwitchGLU`: `x` `[1, L, hidden]`, `idx` `[1, L, k]` → `[1, L, k, hidden]`. With 64
/// or more rows chosen, tokens are sorted by expert first (as mlx-lm does).
fn experts(ex: &Experts, x: &Array, idx: &Array) -> Result<Array> {
    let s = idx.shape();
    let (l, k) = (s[1], s[2]);
    let sort = l * k >= 64;
    let (x, rows, inv) = if sort {
        let flat = idx.flatten(None, None)?;
        let order = argsort(&flat)?;
        let inv = argsort(&order)?;
        let tokens = floor_divide(&order, Array::from_slice(&[k as u32], &[1]))?;
        let x = x.reshape(&[l, 1, -1])?.take_axis(&tokens, 0)?;
        (x, flat.take_axis(&order, 0)?, Some(inv))
    } else {
        (x.expand_dims(-2)?.expand_dims(-2)?, idx.clone(), None)
    };
    let mm = |x: &Array, m: &QMat| -> Result<Array> {
        Ok(gather_qmm(x, &m.w, &m.scales, &m.biases, None, &rows, true, m.group, m.bits, sort)?)
    };
    let up = mm(&x, &ex.up)?;
    let gate = mm(&x, &ex.gate)?;
    let y = mm(&silu(&gate)?.multiply(&up)?, &ex.down)?;
    Ok(match inv {
        Some(inv) => y.take_axis(&inv, 0)?.reshape(&[1, l, k, -1])?,
        None => y.squeeze_axes(&[-2])?,
    })
}

/// RoPE positions of a forward.
enum Rope<'a> {
    /// Consecutive from this position.
    Offset(i32),
    /// cos and sin `[1, 1, L, rotary / 2]` at each token's 3D position.
    Table(&'a Array, &'a Array),
}

/// A forward's input embeddings `[1, L, hidden]` (text and images) and its RoPE tables.
struct Multimodal {
    h: Array,
    cos: Array,
    sin: Array,
}

/// RoPE on the first `rotary` dims of `t` `[1, heads, L, head_dim]` from tables
/// `[1, 1, L, rotary / 2]` (the halves of the rotated dims paired, as `fast::rope` does).
fn rope_table(t: &Array, rotary: i32, cos: &Array, sin: &Array) -> Result<Array> {
    let half = rotary / 2;
    let f = t.as_dtype(Dtype::Float32)?;
    let x1 = f.index((.., .., .., ..half));
    let x2 = f.index((.., .., .., half..rotary));
    let o1 = x1.multiply(cos)?.subtract(x2.multiply(sin)?)?;
    let o2 = x2.multiply(cos)?.add(x1.multiply(sin)?)?;
    Ok(concatenate(&[o1, o2, f.index((.., .., .., rotary..))], -1)?.as_dtype(t.dtype())?)
}

/// Gated attention of one layer on `x` `[1, L, hidden]`, its keys written from cache position
/// `pos`, rotated by `rope`.
fn attention(
    c: &Config,
    a: &Attention,
    kc: &mut Option<Array>,
    vc: &mut Option<Array>,
    x: &Array,
    pos: i32,
    rope: &Rope<'_>,
) -> Result<Array> {
    let n = x.shape()[1];
    let (nq, nkv, d) = (c.heads as i32, c.kv_heads as i32, c.head_dim as i32);
    // Per head: the query, then its gate.
    let qg = a.q.apply(x)?.reshape(&[1, n, nq, 2 * d])?;
    let q = qg.index((.., .., .., ..d));
    let gate = qg.index((.., .., .., d..)).reshape(&[1, n, nq * d])?;
    let q = fast::rms_norm(&q, Some(&a.q_norm), c.eps)?.transpose_axes(&[0, 2, 1, 3])?;
    let k = a.k.apply(x)?.reshape(&[1, n, nkv, d])?;
    let k = fast::rms_norm(&k, Some(&a.k_norm), c.eps)?.transpose_axes(&[0, 2, 1, 3])?;
    let v = a.v.apply(x)?.reshape(&[1, n, nkv, d])?.transpose_axes(&[0, 2, 1, 3])?;
    let rope = |t: &Array| -> Result<Array> {
        match *rope {
            Rope::Offset(p) => Ok(fast::rope(t, c.rotary as i32, false, c.theta, 1.0, p, None)?),
            Rope::Table(cos, sin) => rope_table(t, c.rotary as i32, cos, sin),
        }
    };
    let (q, k) = (rope(&q)?, rope(&k)?);
    let (ks, vs) = kv_update(kc, vc, &k, &v, pos)?;
    let mask = (n > 1).then_some(ScaledDotProductAttentionMask::Causal);
    let o = fast::scaled_dot_product_attention(&q, &ks, &vs, (d as f32).powf(-0.5), mask, None)?;
    let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, n, nq * d])?;
    a.o.apply(&o.multiply(sigmoid(&gate)?)?)
}

/// Gated attention of one decode step for a batch: `x` `[B, 1, hidden]`, each row at its own
/// position (`offsets` `[B]`), its key written at column `at` of `kc`, `vc` (`[B, kv_heads, ·,
/// head_dim]`), attending to the keys `mask` (`[B, 1, 1, at + 1]`) lets it see.
#[allow(clippy::too_many_arguments, reason = "the batch's caches, positions and mask")]
fn attention_batch(
    c: &Config,
    a: &Attention,
    kc: &mut Option<Array>,
    vc: &mut Option<Array>,
    x: &Array,
    offsets: &Array,
    mask: &Array,
    at: i32,
) -> Result<Array> {
    let b = x.shape()[0];
    let (nq, nkv, d) = (c.heads as i32, c.kv_heads as i32, c.head_dim as i32);
    let qg = a.q.apply(x)?.reshape(&[b, 1, nq, 2 * d])?;
    let q = qg.index((.., .., .., ..d));
    let gate = qg.index((.., .., .., d..)).reshape(&[b, 1, nq * d])?;
    let q = fast::rms_norm(&q, Some(&a.q_norm), c.eps)?.transpose_axes(&[0, 2, 1, 3])?;
    let k = a.k.apply(x)?.reshape(&[b, 1, nkv, d])?;
    let k = fast::rms_norm(&k, Some(&a.k_norm), c.eps)?.transpose_axes(&[0, 2, 1, 3])?;
    let v = a.v.apply(x)?.reshape(&[b, 1, nkv, d])?.transpose_axes(&[0, 2, 1, 3])?;
    let rope = |t: &Array| -> Result<Array> {
        Ok(fast::rope_dynamic(t, c.rotary as i32, false, c.theta, 1.0, offsets, None)?)
    };
    let (q, k) = (rope(&q)?, rope(&k)?);
    let (ks, vs) = kv_update(kc, vc, &k, &v, at)?;
    let o = fast::scaled_dot_product_attention(
        &q,
        &ks,
        &vs,
        (d as f32).powf(-0.5),
        ScaledDotProductAttentionMask::Array(mask),
        None,
    )?;
    let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[b, 1, nq * d])?;
    a.o.apply(&o.multiply(sigmoid(&gate)?)?)
}

/// `x` as `dtype` if it is not already.
fn cast(x: &Array, dtype: Dtype) -> Result<Array> {
    Ok(if x.dtype() == dtype {
        x.clone()
    } else {
        x.as_dtype(dtype)?
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_table_matches_fast_rope_for_text() {
        let (rotary, theta, n, d) = (64usize, 1e7f32, 5, 256);
        let t = ironbark_mlx::random::normal::<f32>(&[1, 2, n, d], None, None, None)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let start = 7;
        let p: Vec<i64> = (start..start + i64::from(n)).collect();
        let (cos, sin) = vision::mrope_tables(&[p.clone(), p.clone(), p], rotary, theta, [11, 11, 10]);
        let tab = |v: &[f32]| Array::from_slice(v, &[1, 1, n, (rotary / 2) as i32]);
        let ours = rope_table(&t, rotary as i32, &tab(&cos), &tab(&sin)).unwrap();
        let want = fast::rope(&t, rotary as i32, false, theta, 1.0, start as i32, None).unwrap();
        let f = |a: &Array| {
            let a = a.as_dtype(Dtype::Float32).unwrap();
            a.eval().unwrap();
            a.as_slice::<f32>().to_vec()
        };
        let worst = f(&ours).iter().zip(f(&want)).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        // bf16 output: at most one rounding step apart.
        assert!(worst <= 1.6e-2, "{worst}");
    }

    #[test]
    fn packed_finds_a_bundles_pack() {
        let dir = std::env::temp_dir().join(format!("ironbark-qwen-bundle-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(packed(&dir).unwrap(), dir);
        assert!(bundle_experts(&dir).is_none());
        std::fs::write(dir.join(BUNDLE_EXPERTS), b"").unwrap();
        assert_eq!(bundle_experts(&dir), Some(dir.join(BUNDLE_EXPERTS)));
        packed(&dir).unwrap_err();
        std::fs::create_dir_all(dir.join(BUNDLE_PACK)).unwrap();
        std::fs::write(dir.join(BUNDLE_PACK).join("experts.json"), "{}").unwrap();
        assert_eq!(packed(&dir).unwrap(), dir.join(BUNDLE_PACK));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn config_reads_qwen36_shapes() {
        let dir = std::env::temp_dir().join(format!("ironbark-qwen-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let layer_types: Vec<&str> = (0..40)
            .map(|l| {
                if (l + 1) % 4 == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        let cfg = serde_json::json!({
            "model_type": "qwen3_5_moe",
            "quantization": {"group_size": 64, "bits": 4},
            "text_config": {
                "model_type": "qwen3_5_moe_text", "num_hidden_layers": 40, "hidden_size": 2048,
                "layer_types": layer_types, "num_attention_heads": 16, "num_key_value_heads": 2,
                "head_dim": 256, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
                "num_experts": 256, "num_experts_per_tok": 8, "moe_intermediate_size": 512,
                "shared_expert_intermediate_size": 512, "vocab_size": 248_320, "rms_norm_eps": 1e-6,
                "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10_000_000.0}
            }
        });
        std::fs::write(dir.join("config.json"), cfg.to_string()).unwrap();
        let c = Config::read(&dir).unwrap();
        assert!(is_qwen(&dir));
        assert_eq!(c.full.iter().filter(|&&f| f).count(), 10);
        assert!(c.full[3] && c.full[39] && !c.full[0] && !c.full[4]);
        assert_eq!((c.rotary, c.delta.conv_dim(), c.top_k, c.group), (64, 8192, 8, 64));
        assert!(c.norm_top_k);
        // 30 DeltaNet layers (3 × 8192 bf16 tail + 32 × 128 × 128 fp32) + 10 × KV at 4K.
        let want = 30 * (3 * 8192 * 2 + 32 * 128 * 128 * 4) + 10 * (2 * 2 * 256 * 4096 * 2);
        assert_eq!(Model::state_bytes(&c, 4096), want);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Against mlx-lm on the same MLX checkpoint (`QWEN_MODEL`), with the fixture its oracle
    /// wrote (`QWEN_FIXTURE`): the prompt's logits at every position, then 16 greedy steps.
    #[test]
    fn matches_mlx_lm() {
        let (Ok(dir), Ok(fx)) = (std::env::var("QWEN_MODEL"), std::env::var("QWEN_FIXTURE")) else {
            eprintln!("skipped: set QWEN_MODEL and QWEN_FIXTURE");
            return;
        };
        let fx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fx).unwrap()).unwrap();
        let ids: Vec<u32> = fx["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let mut m = Model::load(Path::new(&dir)).unwrap();

        // One position's logits against the oracle's top-k: (argmax agrees, mean |Δ|, max |Δ|).
        let check = |row: &[f32], want: &serde_json::Value| {
            let wid: Vec<usize> =
                want["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
            let wl: Vec<f32> = want["logits"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
            let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let d: Vec<f32> = wid.iter().zip(&wl).map(|(&i, &w)| (row[i] - w).abs()).collect();
            // The oracle's top token is among this row's maxima (ties go either way).
            (
                row[wid[0]].total_cmp(&top).is_eq(),
                d.iter().sum::<f32>() / d.len() as f32,
                d.iter().copied().fold(0.0, f32::max),
            )
        };

        let logits = m.prefill(&ids, true).unwrap().as_dtype(Dtype::Float32).unwrap();
        logits.eval().unwrap();
        let v = m.cfg.vocab;
        let flat = logits.as_slice::<f32>();
        let pre = fx["prefill"].as_array().unwrap();
        let (mut agree, mut mean, mut worst) = (0, 0.0f32, 0.0f32);
        for (t, want) in pre.iter().enumerate() {
            let (a, mu, mx) = check(&flat[t * v..(t + 1) * v], want);
            agree += usize::from(a);
            mean += mu / pre.len() as f32;
            worst = worst.max(mx);
        }
        eprintln!("prefill: argmax {agree}/{} agree, mean |Δ| {mean:.4}, max |Δ| {worst:.3}", pre.len());
        assert!(agree * 100 >= pre.len() * 97, "argmax agrees at {agree} of {} positions", pre.len());
        // The same MLX version as the oracle computes the prompt bit for bit.
        assert!(mean < 0.01 && worst < 0.1, "logits differ: mean {mean}, max {worst}");

        // Greedy decode with the caches, from a fresh prompt.
        m.reset();
        let mut last = m.prefill(&ids, false).unwrap().as_dtype(Dtype::Float32).unwrap();
        let want: Vec<u32> = fx["greedy"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let steps = fx["steps"].as_array().unwrap();
        let mut got = Vec::new();
        for (s, w) in steps.iter().enumerate() {
            last.eval().unwrap();
            let row = last.as_slice::<f32>();
            // The first of the maxima, as mlx-lm's argmax picks it.
            let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            got.push(row.iter().position(|x| x.total_cmp(&top).is_eq()).unwrap() as u32);
            // Feed the oracle's token, so one early near-tie cannot derail the rest.
            last = m.step(want[s]).unwrap().as_dtype(Dtype::Float32).unwrap();
            last.eval().unwrap();
            let (_, mu, mx) = check(last.as_slice::<f32>(), w);
            eprintln!("step {s:2}: mean |Δ| {mu:.4}, max |Δ| {mx:.3}");
            // Exact for the first steps on the same MLX; later a kernel variant can differ in the
            // last bits, below the gap between MLX's batched and one-token kernels (≈0.3 mean).
            assert!(mu < 0.3 && mx < 2.0, "step {s}: logits differ: mean {mu}, max {mx}");
        }
        let same = got.iter().zip(&want).filter(|(a, b)| a == b).count();
        eprintln!("greedy: {same}/{} tokens agree", want.len());
        assert!(same + 1 >= want.len(), "greedy tokens: {got:?} vs {want:?}");
    }

    /// Debug aid: with `QWEN_LAYERS` (the layer oracle's safetensors) as well, the first layer
    /// where this engine leaves mlx-lm, per tensor (mixer, MoE, residual).
    #[test]
    fn layers_match_mlx_lm() {
        let (Ok(dir), Ok(fx), Ok(layers)) =
            (std::env::var("QWEN_MODEL"), std::env::var("QWEN_FIXTURE"), std::env::var("QWEN_LAYERS"))
        else {
            eprintln!("skipped: set QWEN_MODEL, QWEN_FIXTURE and QWEN_LAYERS");
            return;
        };
        let fx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(fx).unwrap()).unwrap();
        let ids: Vec<u32> = fx["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let want = Array::load_safetensors(&layers).unwrap();
        let mut m = Model::load(Path::new(&dir)).unwrap();
        let mut trace = Vec::new();
        let n = ids.len() as i32;
        m.forward_traced(&Array::from_slice(&ids, &[n]), None, true, Some(&mut trace)).unwrap();
        let rel = |got: &Array, key: &str| {
            let w = &want[key];
            let g = got.reshape(&[n, -1]).unwrap().as_dtype(Dtype::Float32).unwrap();
            let d = g.subtract(w).unwrap().abs().unwrap().max(None).unwrap().item_exact::<f32>();
            let s = w.abs().unwrap().max(None).unwrap().item_exact::<f32>();
            // Worst position: the token where the residual differs most.
            let per = g.subtract(w).unwrap().abs().unwrap().max_axis(-1, false).unwrap();
            let worst = ironbark_mlx::ops::indexing::argmax_axis(&per, 0, None).unwrap().item_exact::<u32>();
            (d / s.max(1e-6), worst)
        };
        for (l, [mx, moe, h]) in trace.iter().enumerate() {
            let kind = if m.cfg.full[l] { "attn" } else { "delta" };
            let (a, _) = rel(mx, &format!("mixer.{l}"));
            let (b, _) = rel(moe, &format!("moe.{l}"));
            let (c, at) = rel(h, &format!("h.{l}"));
            eprintln!("layer {l:2} {kind:5}  mixer {a:.2e}  moe {b:.2e}  residual {c:.2e} (worst token {at})");
        }
        // Each block alone, on the oracle's own input: the mixer on in_norm(h_{l-1}), the MoE on
        // post_norm(h_{l-1} + mixer_l).
        let c = m.cfg.clone();
        for (l, ly) in m.layers.iter().enumerate() {
            let prev = if l == 0 {
                want["embed"].clone()
            } else {
                want[&format!("h.{}", l - 1)].clone()
            };
            let prev = prev.as_dtype(Dtype::Bfloat16).unwrap().expand_dims(0).unwrap();
            let x = fast::rms_norm(&prev, Some(&ly.in_norm), c.eps).unwrap();
            let mx = match &ly.mixer {
                Mixer::Attn(a) => attention(&c, a, &mut None, &mut None, &x, 0, &Rope::Offset(0)).unwrap(),
                Mixer::Delta(d) => d.forward(&x, &mut deltanet::State::default(), c.eps, m.step.as_ref()).unwrap(),
            };
            let mixed = want[&format!("mixer.{l}")].as_dtype(Dtype::Bfloat16).unwrap().expand_dims(0).unwrap();
            let x = fast::rms_norm(prev.add(&mixed).unwrap(), Some(&ly.post_norm), c.eps).unwrap();
            let (y, ..) = moe(&c, ly, &x).unwrap();
            let (a, ta) = rel(&mx, &format!("mixer.{l}"));
            let (b, tb) = rel(&y, &format!("moe.{l}"));
            eprintln!("alone {l:2}  mixer {a:.2e} (token {ta})  moe {b:.2e} (token {tb})");
        }
    }

    /// Three prompts of different lengths decoded together give the tokens each gives alone
    /// (`QWEN_MODEL`). Batched matmuls round differently, so a late near-tie may flip.
    #[test]
    fn batch_decodes_like_one_at_a_time() {
        let Ok(dir) = std::env::var("QWEN_MODEL") else {
            eprintln!("skipped: set QWEN_MODEL");
            return;
        };
        let mut m = Model::load(Path::new(&dir)).unwrap();
        let tok = tokenizers::Tokenizer::from_file(Path::new(&dir).join("tokenizer.json")).unwrap();
        let prompts = [
            "Name three primary colors.",
            "In one sentence, what does a compiler do? Keep it simple and friendly for a beginner.",
            "Hi",
        ];
        let chat = |p: &str| format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        let ids: Vec<Vec<u32>> =
            prompts.iter().map(|p| tok.encode(chat(p), false).unwrap().get_ids().to_vec()).collect();
        let v = m.cfg.vocab;
        let argmax = |l: &Array| -> Vec<u32> {
            let l = l.as_dtype(Dtype::Float32).unwrap();
            l.eval().unwrap();
            l.as_slice::<f32>()
                .chunks(v)
                .map(|row| (0..v).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap() as u32)
                .collect()
        };
        let steps = 12usize;
        let mut alone = Vec::new();
        for p in &ids {
            m.reset();
            let mut t = argmax(&m.prefill(p, false).unwrap())[0];
            let mut out = vec![t];
            for _ in 1..steps {
                t = argmax(&m.step(t).unwrap())[0];
                out.push(t);
            }
            alone.push(out);
        }
        let mut snaps = Vec::new();
        let mut next = Vec::new();
        for p in &ids {
            m.reset();
            next.push(argmax(&m.prefill(p, false).unwrap())[0]);
            snaps.push(m.snapshot());
        }
        let mut b = m.batch(&snaps).unwrap();
        let mut together: Vec<Vec<u32>> = next.iter().map(|&t| vec![t]).collect();
        for _ in 1..steps {
            next = argmax(&m.step_batch(&mut b, &next).unwrap());
            for (o, &t) in together.iter_mut().zip(&next) {
                o.push(t);
            }
        }
        for (r, (a, t)) in alone.iter().zip(&together).enumerate() {
            let same = a.iter().zip(t).take_while(|(x, y)| x == y).count();
            eprintln!("prompt {r}: {same}/{steps} tokens agree: {:?}", tok.decode(t, false).unwrap());
            assert!(same >= steps - 2, "prompt {r}: alone {a:?}, together {t:?}");
        }
    }

    #[test]
    fn embedding_bits_follow_the_override_else_the_default() {
        let default = serde_json::json!({ "group_size": 64, "bits": 4 });
        assert_eq!(embed_bits(&default), 4);
        let own = serde_json::json!({ "bits": 4, "language_model.model.embed_tokens": { "bits": 8 } });
        assert_eq!(embed_bits(&own), 8);
        assert_eq!(embed_bits(&serde_json::Value::Null), 4);
    }

    #[test]
    fn route_keeps_the_top_k_renormalized() {
        // Scores for 1 token over 4 experts; a fake router is the identity on 4 dims.
        let c = Config {
            hidden: 4,
            layers: 1,
            full: vec![true],
            heads: 1,
            kv_heads: 1,
            head_dim: 4,
            rotary: 4,
            theta: 1e4,
            mrope_section: [11, 11, 10],
            delta: deltanet::Shape {
                k_heads: 1,
                v_heads: 1,
                k_dim: 32,
                v_dim: 32,
                conv: 4,
            },
            experts: 4,
            top_k: 2,
            norm_top_k: true,
            dynamic: crate::dynamic::Dynamic::default(),
            expert_ffn: 4,
            shared_ffn: 4,
            vocab: 4,
            eps: 1e-6,
            group: 32,
            embed_bits: 4,
            expert_ternary: false,
            expert_bits: None,
        };
        let eye: Vec<f32> = (0..4 * 64).map(|i| if i / 64 == i % 64 { 1.0 } else { 0.0 }).collect();
        let w = Array::from_slice(&eye, &[4, 64]);
        let (wq, s, b) = ironbark_mlx::ops::quantize(&w, 32, 8).unwrap();
        let router = QMat {
            w: wq,
            scales: s,
            biases: b,
            group: 32,
            bits: 8,
        };
        let mut xs = vec![0.0f32; 64];
        xs[..4].copy_from_slice(&[1.0, 3.0, 0.0, 2.0]);
        let x = Array::from_slice(&xs, &[1, 1, 64]);
        let (idx, w) = route(&c, &router, &x).unwrap();
        let mut got: Vec<(u32, f32)> =
            idx.as_slice::<u32>().iter().copied().zip(w.as_slice::<f32>().iter().copied()).collect();
        got.sort_by_key(|&(e, _)| e);
        // softmax over all 4, keep experts 1 (3.0) and 3 (2.0), renormalize: e/(e+1), 1/(e+1).
        let e = std::f32::consts::E;
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), [1, 3]);
        assert!((got[0].1 - e / (e + 1.0)).abs() < 1e-2, "{got:?}");
        assert!((got[1].1 - 1.0 / (e + 1.0)).abs() < 1e-2, "{got:?}");
        // Expert 1 pruned (`Model::keep_experts`): the top-2 of the rest, 3 (2.0) and 0 (1.0).
        let keep = [Array::from_slice(&[0.0f32, f32::NEG_INFINITY, 0.0, 0.0], &[4])];
        let (idx, _) = route_logits(&c, &keep_bias(Some(&keep), 0, router.apply(&x).unwrap()).unwrap(), false).unwrap();
        let mut got = idx.as_slice::<u32>().to_vec();
        got.sort_unstable();
        assert_eq!(got, [0, 3]);
    }
}
