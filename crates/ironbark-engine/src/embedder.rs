//! Qwen3-Embedding (`qwen3`, e.g. Qwen3-Embedding-0.6B) on the GPU through MLX: a dense
//! decoder whose last token's hidden state, L2-normalized, is the text's embedding.
//!
//! ```text
//!   h = embed(ids)                                         [1, L, hidden]
//!   per layer:
//!     x  = rms_norm(h, in_norm)
//!     q  = rope(rms_norm(q·x, q_norm))                     [1, heads,    L, head_dim]
//!     k  = rope(rms_norm(k·x, k_norm))                     [1, kv_heads, L, head_dim]
//!     v  = v·x
//!     h += o·sdpa(q, k, v, causal)                         GQA
//!     x  = rms_norm(h, post_norm)
//!     h += down·(silu(gate·x) * up·x)
//!   out = l2_normalize(rms_norm(h, norm)[last])            [hidden]
//! ```
//!
//! Weights: an MLX affine checkpoint (`mlx_lm.convert -q`), all resident. No KV cache: one
//! forward per text.

use std::path::Path;

use ironbark_mlx::fast::ScaledDotProductAttentionMask;
use ironbark_mlx::ops::indexing::IndexOp;
use ironbark_mlx::ops::{dequantize, silu};
use ironbark_mlx::{Array, Dtype, fast};

use crate::model::QMat;
use crate::quant::Tensors;
use crate::{Error, Result};

/// Shapes and hyperparameters of an embedding model (`config.json`).
#[derive(Debug, Clone)]
pub struct EmbedConfig {
    pub hidden: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub theta: f32,
    pub eps: f32,
    pub vocab: usize,
    /// Quantization group size.
    pub group: usize,
    /// Quantization bits.
    pub bits: usize,
}

impl EmbedConfig {
    /// Read `dir/config.json`.
    ///
    /// # Errors
    ///
    /// Fails if the file is missing, lacks a field, or is not quantized.
    pub fn read(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw = std::fs::read(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let c: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
        let u = |k: &str| c[k].as_u64().map(|v| v as usize).ok_or_else(|| Error::Format(format!("config: no {k}")));
        let q = c.get("quantization").or(c.get("quantization_config"));
        let q = q.ok_or_else(|| Error::Format("config: not quantized (run mlx_lm.convert -q)".into()))?;
        let qu = |k: &str| {
            q[k].as_u64().map(|v| v as usize).ok_or_else(|| Error::Format(format!("config: no quantization.{k}")))
        };
        let hidden = u("hidden_size")?;
        let heads = u("num_attention_heads")?;
        Ok(Self {
            hidden,
            layers: u("num_hidden_layers")?,
            heads,
            kv_heads: u("num_key_value_heads")?,
            head_dim: u("head_dim").unwrap_or(hidden / heads),
            intermediate: u("intermediate_size")?,
            theta: c["rope_theta"].as_f64().unwrap_or(1_000_000.0) as f32,
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            vocab: u("vocab_size")?,
            group: qu("group_size")?,
            bits: qu("bits")?,
        })
    }
}

/// Whether `dir` holds a dense Qwen3 checkpoint (`model_type` `qwen3`), as an embedding model.
#[must_use]
pub fn is_embedder(dir: &Path) -> bool {
    std::fs::read(dir.join("config.json"))
        .ok()
        .and_then(|raw| serde_json::from_slice::<serde_json::Value>(&raw).ok())
        .is_some_and(|c| c["model_type"].as_str() == Some("qwen3"))
}

/// Files an embedding pack holds (beside its weights and `manifest.json`).
const PACK_FILES: [&str; 3] = ["config.json", "tokenizer.json", "LICENSE"];

/// Copy the embedding checkpoint `model` to `out` (its config, tokenizer, weights and license, if
/// any) and write `out/manifest.json`, so a copy can be checked (`ironbark verify`) or fetched.
/// `out` may be `model`: then only the manifest is written, over every file there.
///
/// # Errors
///
/// Fails if `model` is not an embedding checkpoint, or a file cannot be copied or hashed.
pub fn pack(model: &Path, out: &Path) -> Result<()> {
    if !is_embedder(model) {
        return Err(Error::Format(format!("{}: not a Qwen3 embedding checkpoint", model.display())));
    }
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |source| Error::Io { path, source }
    };
    let same = out.canonicalize().ok() == model.canonicalize().ok();
    if !same {
        std::fs::create_dir_all(out).map_err(io(out))?;
        let weights = std::fs::read_dir(model)
            .map_err(io(model))?
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.starts_with("model") && n.ends_with(".safetensors"));
        for name in weights.chain(PACK_FILES.map(str::to_owned)) {
            let src = model.join(&name);
            if src.exists() {
                std::fs::copy(&src, out.join(&name)).map_err(io(&src))?;
            }
        }
        // Keep the shard index only when the weights are split.
        let index = model.join("model.safetensors.index.json");
        if index.exists() && !model.join("model.safetensors").exists() {
            std::fs::copy(&index, out.join("model.safetensors.index.json")).map_err(io(&index))?;
        }
    }
    crate::manifest::write(out, &[], 0, 0)
}

/// One transformer layer.
struct DenseLayer {
    in_norm: Array,
    post_norm: Array,
    q: QMat,
    k: QMat,
    v: QMat,
    o: QMat,
    q_norm: Array,
    k_norm: Array,
    gate: QMat,
    up: QMat,
    down: QMat,
}

/// An embedding model, all weights resident.
pub struct EmbedModel {
    cfg: EmbedConfig,
    embed: QMat,
    layers: Vec<DenseLayer>,
    norm: Array,
}

impl std::fmt::Debug for EmbedModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedModel").field("cfg", &self.cfg).finish_non_exhaustive()
    }
}

/// The checkpoint's tensors: `model.safetensors.index.json`'s shards, else `model.safetensors`.
fn tensors(dir: &Path) -> Result<Tensors> {
    if !dir.join("model.safetensors").exists() {
        return Ok(Tensors(crate::model::load_tensors(dir)?));
    }
    let path = dir.join("model.safetensors");
    Ok(Tensors(Array::load_safetensors(&path).map_err(|e| Error::Load {
        path,
        msg: e.to_string(),
    })?))
}

impl EmbedModel {
    /// Load the checkpoint in `dir`.
    ///
    /// # Errors
    ///
    /// Fails if a file or tensor is missing, or MLX cannot load it.
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg = EmbedConfig::read(dir)?;
        let g = cfg.group;
        let mut t = tensors(dir)?;
        // `mlx_lm.convert` writes `model.`-prefixed names; a bare `Qwen3Model` has none.
        let prefix = if t.0.contains_key("model.norm.weight") { "model." } else { "" };
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let n = |s: &str| format!("{prefix}layers.{l}.{s}");
            layers.push(DenseLayer {
                in_norm: t.take(&n("input_layernorm.weight"))?,
                post_norm: t.take(&n("post_attention_layernorm.weight"))?,
                q: t.qmat(&n("self_attn.q_proj"), g)?,
                k: t.qmat(&n("self_attn.k_proj"), g)?,
                v: t.qmat(&n("self_attn.v_proj"), g)?,
                o: t.qmat(&n("self_attn.o_proj"), g)?,
                q_norm: t.take(&n("self_attn.q_norm.weight"))?,
                k_norm: t.take(&n("self_attn.k_norm.weight"))?,
                gate: t.qmat(&n("mlp.gate_proj"), g)?,
                up: t.qmat(&n("mlp.up_proj"), g)?,
                down: t.qmat(&n("mlp.down_proj"), g)?,
            });
        }
        let embed = t.qmat(&format!("{prefix}embed_tokens"), g)?;
        let norm = t.take(&format!("{prefix}norm.weight"))?;
        // The tied LM head (if any) is not used.
        let model = Self {
            cfg,
            embed,
            layers,
            norm,
        };
        ironbark_mlx::transforms::eval(model.arrays())?;
        Ok(model)
    }

    fn arrays(&self) -> Vec<&Array> {
        fn q(m: &QMat) -> [&Array; 3] {
            [&m.w, &m.scales, &m.biases]
        }
        let mut v = vec![&self.norm];
        v.extend(q(&self.embed));
        for ly in &self.layers {
            v.extend([&ly.in_norm, &ly.post_norm, &ly.q_norm, &ly.k_norm]);
            for m in [&ly.q, &ly.k, &ly.v, &ly.o, &ly.gate, &ly.up, &ly.down] {
                v.extend(q(m));
            }
        }
        v
    }

    /// Shapes and hyperparameters.
    #[must_use]
    pub fn config(&self) -> &EmbedConfig {
        &self.cfg
    }

    /// The unit-length embedding `[hidden]` (f32) of `ids`: the last token's final hidden state.
    ///
    /// # Errors
    ///
    /// Fails if `ids` is empty or an MLX operation fails.
    pub fn forward(&self, ids: &[u32]) -> Result<Array> {
        if ids.is_empty() {
            return Err(Error::Format("embed: no tokens".into()));
        }
        let c = &self.cfg;
        let n = ids.len() as i32;
        let (nq, nkv, d) = (c.heads as i32, c.kv_heads as i32, c.head_dim as i32);
        let idx = Array::from_slice(ids, &[n]);
        let e = &self.embed;
        let mut h = dequantize(
            e.w.take_axis(&idx, 0)?,
            e.scales.take_axis(&idx, 0)?,
            &e.biases.take_axis(&idx, 0)?,
            e.group,
            e.bits,
        )?
        .reshape(&[1, n, -1])?;
        let scale = (d as f32).powf(-0.5);
        for ly in &self.layers {
            let x = fast::rms_norm(&h, Some(&ly.in_norm), c.eps)?;
            let heads = |m: &QMat, norm: Option<&Array>, count: i32| -> Result<Array> {
                let t = m.apply(&x)?.reshape(&[1, n, count, d])?;
                let t = match norm {
                    Some(w) => fast::rms_norm(&t, Some(w), c.eps)?,
                    None => t,
                };
                Ok(t.transpose_axes(&[0, 2, 1, 3])?)
            };
            let rope = |t: &Array| fast::rope(t, d, false, c.theta, 1.0, 0, None);
            let q = rope(&heads(&ly.q, Some(&ly.q_norm), nq)?)?;
            let k = rope(&heads(&ly.k, Some(&ly.k_norm), nkv)?)?;
            let v = heads(&ly.v, None, nkv)?;
            let o = fast::scaled_dot_product_attention(&q, &k, &v, scale, (n > 1).then_some(ScaledDotProductAttentionMask::Causal), None)?;
            let o = o.transpose_axes(&[0, 2, 1, 3])?.reshape(&[1, n, nq * d])?;
            h = h.add(ly.o.apply(&o)?)?;
            let x = fast::rms_norm(&h, Some(&ly.post_norm), c.eps)?;
            let mlp = silu(ly.gate.apply(&x)?)?.multiply(ly.up.apply(&x)?)?;
            h = h.add(ly.down.apply(&mlp)?)?;
        }
        let last = h.index((0, n - 1, ..));
        let out = fast::rms_norm(&last, Some(&self.norm), c.eps)?.as_dtype(Dtype::Float32)?;
        let len = out.square()?.sum(None)?.sqrt()?.add(Array::from_f32(1e-12))?;
        Ok(out.divide(len)?)
    }
}
