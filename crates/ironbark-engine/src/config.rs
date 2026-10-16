//! Maple hyperparameters from `config.json`.

use std::path::Path;

use crate::{Error, Result};

/// Maple model shape.
#[derive(Debug, Clone)]
pub struct Config {
    /// Decoder layers.
    pub layers: usize,
    /// Hidden size.
    pub hidden: usize,
    /// Query heads.
    pub heads: usize,
    /// Key/value heads.
    pub kv_heads: usize,
    /// Size of one head.
    pub head_dim: usize,
    /// Dims per head that RoPE rotates (sliding layers only).
    pub rope_dims: usize,
    /// RoPE base.
    pub theta: f32,
    /// Experts per layer.
    pub experts: usize,
    /// Experts used per token.
    pub top_k: usize,
    /// Expert FFN size.
    pub ffn: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Sliding attention window (tokens kept, including the current one).
    pub window: usize,
    /// Per layer: sliding-window attention (with RoPE) or full attention (no RoPE).
    pub sliding: Vec<bool>,
    /// Ternary group size of the checkpoint's affine layout.
    pub group: usize,
    /// LM head / embedding: bits and group size.
    pub head_bits: usize,
    /// LM head / embedding group size.
    pub head_group: usize,
    /// FlashHead metadata, if the checkpoint has one.
    pub flash_head: Option<FlashMeta>,
}

/// FlashHead settings (`config.json` `flash_head`).
#[derive(Debug, Clone)]
pub struct FlashMeta {
    /// Clusters scored exactly per token.
    pub probes: usize,
    /// Tokens always scored exactly (end of turn, `</think>`).
    pub force_tokens: Vec<i32>,
    /// Centroids: bits and group size.
    pub bits: usize,
    /// Centroid group size.
    pub group: usize,
}

impl Config {
    /// Read `config.json` from `dir`.
    ///
    /// # Errors
    ///
    /// Fails if the file is missing or not a ternary Maple checkpoint.
    pub fn read(dir: &Path) -> Result<Self> {
        let path = dir.join("config.json");
        let raw = std::fs::read(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let c: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
        if c["model_type"] != "maple" {
            return Err(Error::Format(format!("{}: not a Maple checkpoint", path.display())));
        }
        let u = |k: &str| c[k].as_u64().unwrap_or(0) as usize;
        let q = &c["quantization"];
        if q["bits"] != 2 {
            return Err(Error::Format(format!("{}: expected 2-bit ternary weights", path.display())));
        }
        let layers = u("num_hidden_layers");
        let head_dim = u("head_dim");
        let sliding = (0..layers).map(|l| c["layer_types"][l] == "sliding_attention").collect();
        Ok(Config {
            layers,
            hidden: u("hidden_size"),
            heads: u("num_attention_heads"),
            kv_heads: u("num_key_value_heads"),
            head_dim,
            rope_dims: (head_dim as f64 * c["partial_rotary_factor"].as_f64().unwrap_or(1.0)) as usize,
            theta: c["rope_theta"].as_f64().unwrap_or(1e4) as f32,
            experts: u("num_experts"),
            top_k: u("num_experts_per_tok"),
            ffn: u("moe_intermediate_size"),
            vocab: u("vocab_size"),
            eps: c["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            window: u("sliding_window"),
            sliding,
            group: q["group_size"].as_u64().unwrap_or(128) as usize,
            head_bits: q["lm_head"]["bits"].as_u64().unwrap_or(4) as usize,
            head_group: q["lm_head"]["group_size"].as_u64().unwrap_or(64) as usize,
            // Older metadata has unscaled centroids (maple.py refuses it too).
            flash_head: c["flash_head"]["scaled_centroids"].as_bool().unwrap_or(false).then(|| FlashMeta {
                probes: c["flash_head"]["n_probes"].as_u64().unwrap_or(512) as usize,
                force_tokens: c["flash_head"]["force_tokens"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|t| t.as_i64().map(|t| t as i32)).collect())
                    .unwrap_or_default(),
                bits: c["flash_head"]["bits"].as_u64().unwrap_or(4) as usize,
                group: c["flash_head"]["group_size"].as_u64().unwrap_or(64) as usize,
            }),
        })
    }
}
