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

pub mod bench;
mod config;
pub mod device;
mod embed;
mod error;
mod flash;
mod fused;
mod kernel;
pub mod manifest;
mod model;
mod pack;
pub mod plan;
mod prefill;
pub mod profile;
mod row;

pub use config::{Config, FlashMeta};
pub use error::{Error, Result};
pub use model::{KvSnapshot, Model, StepTiming};
pub use pack::{Packed, pack};
pub use prefill::CHUNK;

#[cfg(test)]
mod tests {
    use mlx_rs::Array;

    #[test]
    fn mlx_runs_on_gpu() {
        let a = Array::from_slice(&[1f32, 2.0, 3.0], &[3]);
        let b = (&a * &a).sum(None).unwrap();
        assert_eq!(b.item_exact::<f32>(), 14.0);
    }
}
