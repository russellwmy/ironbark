//! Size a streamed model to the device: a footprint budget from the hardware, then as many
//! expert cache slots as fit in it.
//!
//! ```text
//!   budget = min(RAM · 3/8, app limit / 2)             8 GB iPhone -> 3 GB, 64 GB Mac -> 24 GB
//!   peak   = weights + RUNTIME + KV(context) + max(PREFILL, vision) + buffer cache
//!            + slots · (layers · stride + PREFILL_SLOT)
//!   slots  = what is left / (layers · stride + PREFILL_SLOT), clamped to [MIN_SLOTS, experts]
//! ```
//!
//! The vision encoder (`vision.safetensors`) loads only to encode an image and is freed after,
//! never during a prefill: it shares that headroom instead of staying resident.
//!
//! The constants are measured footprints (M1 Max, Maple, 750–890-token prompts): tokenizer and
//! runtime ~80 MB (`RUNTIME` = 100 MB with margin), prefill temporaries 0.25 GB at 16 slots to 0.52 GB at 154 (streamed prefill
//! copies each group of cached experts), whatever the chunk size. The KV cache is computed:
//! keys and values of the full-attention layers for `context` tokens, of the sliding layers for
//! their window.

use std::path::Path;

use ironbark_experts::Layout;

use crate::device::Hardware;
use crate::{Config, Error, Result};

/// Default most tokens a session holds (its KV cache is part of the plan).
pub const CONTEXT: usize = 4096;
/// Qwen's default when the budget has room ([`QWEN_CONTEXT_SLOTS`]): its KV cache is in 10
/// of 40 layers (20 KB a token, 320 MB for 16K), and a chat with tools (results, reasoning)
/// outgrows 4096 tokens in one question.
pub const QWEN_CONTEXT: usize = 16_384;
/// Fewest cache slots per layer that [`QWEN_CONTEXT`] may leave; below it Qwen keeps
/// [`CONTEXT`] (an iPhone's 3 GB default: 23 slots instead of 16, where routing falls apart).
pub const QWEN_CONTEXT_SLOTS: usize = 48;

/// Share of physical RAM a model may use by default: far from the iOS app limit while in use,
/// and the app frees the model when it leaves the screen.
const RAM_SHARE: f64 = 0.375;
/// Share of the iOS app limit a model may use by default (the app needs the rest).
const LIMIT_SHARE: f64 = 0.5;
/// Tokenizer and MLX / Metal runtime.
const RUNTIME: u64 = 100 << 20;
/// Peak of prefill temporaries above the steady footprint: a base, plus per cache slot.
const PREFILL: u64 = 250 << 20;
const PREFILL_SLOT: u64 = 2 << 20;
/// MLX's cache of freed GPU buffers when the budget has room for it (faster prefill).
const BUFFER_CACHE: u64 = 256 << 20;
/// Fewest cache slots per layer: the top-8 a token uses, for budgets below the 16 that leave
/// room to prefetch (less memory, slower decode).
const MIN_SLOTS: usize = 8;

/// How to run a streamed model within a footprint budget.
#[derive(Debug, Clone, Copy)]
pub struct Plan {
    /// Footprint budget in bytes.
    pub budget: u64,
    /// Expert cache slots per layer; all experts means nothing streams.
    pub slots: usize,
    /// MLX buffer cache limit in bytes.
    pub buffer_cache: u64,
    /// Expected peak footprint in bytes.
    pub peak: u64,
}

/// The default budget for `hw`: 3/8 of RAM, and at most half the app limit.
#[must_use]
pub fn budget(hw: &Hardware) -> u64 {
    let ram = (hw.ram as f64 * RAM_SHARE) as u64;
    hw.app_limit.map_or(ram, |l| ram.min((l as f64 * LIMIT_SHARE) as u64))
}

impl Plan {
    /// Plan the packed model in `dir` (`ironbark pack` output) for `budget` bytes, with room for a
    /// KV cache of `context` tokens.
    ///
    /// # Errors
    ///
    /// Fails if the packed layout, the config or the weight files cannot be read.
    pub fn new(dir: &Path, budget: u64, context: usize) -> Result<Self> {
        Self::with_state(dir, budget, kv_bytes(&Config::read(dir)?, context))
    }

    /// Plan the packed model in `dir` for `budget` bytes, with `state` bytes of caches (KV,
    /// recurrent state) for the longest context.
    ///
    /// # Errors
    ///
    /// Fails if the packed layout or the weight files cannot be read.
    pub fn with_state(dir: &Path, budget: u64, state: u64) -> Result<Self> {
        let layout = Layout::read(dir).map_err(|e| Error::Format(format!("{}: {e}", dir.display())))?;
        let (weights, vision) = weight_bytes(dir)?;
        let slot = layout.layers as u64 * layout.stride + PREFILL_SLOT;
        let fixed = weights + RUNTIME + state + PREFILL.max(vision);
        let room = |buffer: u64| budget.saturating_sub(fixed + buffer) / slot;
        // The buffer cache only when it still leaves every expert room, or costs no slots; never
        // below the floor, where the budget is already short.
        let floor = room(0) < MIN_SLOTS as u64;
        let buffer_cache = if !floor && (room(BUFFER_CACHE) >= layout.experts as u64 || room(BUFFER_CACHE) == room(0)) {
            BUFFER_CACHE
        } else {
            0
        };
        let slots = (room(buffer_cache) as usize).clamp(MIN_SLOTS, layout.experts);
        Ok(Self {
            budget,
            slots,
            buffer_cache,
            peak: fixed + buffer_cache + slots as u64 * slot,
        })
    }

    /// Plan a model in `dir` whose weights are all resident (an MLX checkpoint, no
    /// `experts.bin`), with `state` bytes of caches: `slots` is 0 and `peak` may exceed
    /// `budget` (the caller decides whether to run it).
    ///
    /// # Errors
    ///
    /// Fails if the weight files cannot be read.
    pub fn resident(dir: &Path, budget: u64, state: u64) -> Result<Self> {
        let (weights, vision) = weight_bytes(dir)?;
        let fixed = weights + RUNTIME + state + PREFILL.max(vision);
        Ok(Self {
            budget,
            slots: 0,
            buffer_cache: BUFFER_CACHE,
            peak: fixed + BUFFER_CACHE,
        })
    }
}

/// Bytes of the KV cache (bf16 keys and values) for `context` tokens.
#[must_use]
pub fn kv_bytes(c: &Config, context: usize) -> u64 {
    let per_token = (2 * c.kv_heads * c.head_dim * 2) as u64;
    c.sliding.iter().map(|&s| per_token * if s { context.min(c.window) } else { context } as u64).sum()
}

/// Bytes of the non-expert weights in `dir`: the resident safetensors files, and the vision
/// encoder's (loaded only to encode an image).
fn weight_bytes(dir: &Path) -> Result<(u64, u64)> {
    let io = |source| Error::Io {
        path: dir.to_path_buf(),
        source,
    };
    let (mut resident, mut vision) = (0, 0);
    for entry in std::fs::read_dir(dir).map_err(io)? {
        let entry = entry.map_err(io)?;
        let path = entry.path();
        if path.extension().is_some_and(|x| x == "safetensors") {
            let len = entry.metadata().map_err(io)?.len();
            if path.file_name().is_some_and(|f| f == "vision.safetensors") {
                vision += len;
            } else {
                resident += len;
            }
        }
    }
    Ok((resident, vision))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_follows_the_hardware() {
        let gb = 1u64 << 30;
        let iphone = Hardware {
            ram: 8 * gb,
            app_limit: Some(6 * gb + gb / 2),
        };
        assert_eq!(budget(&iphone), 3 * gb);
        let tight = Hardware {
            ram: 8 * gb,
            app_limit: Some(3 * gb),
        };
        assert_eq!(budget(&tight), 3 * gb / 2);
        let mac = Hardware {
            ram: 64 * gb,
            app_limit: None,
        };
        assert_eq!(budget(&mac), 24 * gb);
    }
}
