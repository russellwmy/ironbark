//! Ternary MoE (Maple) decode on the Apple GPU through MLX.
//!
//! ```text
//!   token ──▶ embed (4-bit) ──▶ 24 × [ RMSNorm ─▶ QKV (ternary) ─▶ QK norm, RoPE (sliding layers)
//!                                      ─▶ attention (KV cache; window 512 on sliding layers) ─▶ O (ternary)
//!                                      RMSNorm ─▶ router (fp32, top-8 of 256)
//!                                      ─▶ 8 experts: up+gate (ternary) ─▶ clamped SwiGLU ─▶ down (ternary) ]
//!        ──▶ RMSNorm ──▶ LM head (4-bit) ──▶ logits
//! ```
//!
//! Weights come from DeepGrove's MLX checkpoint (`deepgrove/maple-preview-2bit-mlx`): ternary
//! matrices as packed 2-bit codes plus one `row_alpha` per row, read as is by the row-scale
//! kernels in `row` (custom Metal through MLX, see `kernel`).

#![allow(
    clippy::cast_possible_wrap,
    reason = "MLX shapes and indices are i32; model dimensions are far below i32::MAX"
)]

mod affine;
pub mod bench;
mod config;
pub mod device;
pub mod dynamic;
mod embed;
mod error;
mod experts;
mod flash;
mod fused;
mod kernel;
mod kv;
pub mod manifest;
mod model;
mod pack;
pub mod plan;
mod prefill;
pub mod profile;
mod quant;
pub mod qwen;
mod row;
mod tern;

pub use config::{Config, FlashMeta};
pub use error::{Error, Result};
pub use model::{KvSnapshot, Model, StepTiming};
pub use pack::{Packed, pack};
pub use prefill::CHUNK;

/// Let one Metal command buffer hold a whole decode step: MLX's defaults (20–50 operations) end a
/// buffer every few layers, and each one costs a submission (Qwen3.6 q4 on an M1 Max: 63 → 82
/// tok/s). A setting already in the environment wins.
///
/// # Safety
///
/// Call it first in `main`, before any thread starts and before any MLX call: MLX reads these
/// once, when it opens the device.
pub unsafe fn large_command_buffers() {
    for name in ["MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER"] {
        if std::env::var_os(name).is_none() {
            // SAFETY: no other thread is running (the caller's contract).
            unsafe { std::env::set_var(name, "1000") };
        }
    }
}

#[cfg(test)]
mod tests {
    use ironbark_mlx::Array;

    #[test]
    fn mlx_runs_on_gpu() {
        let a = Array::from_slice(&[1f32, 2.0, 3.0], &[3]);
        let b = (&a * &a).sum(None).unwrap();
        assert_eq!(b.item_exact::<f32>(), 14.0);
    }
}
