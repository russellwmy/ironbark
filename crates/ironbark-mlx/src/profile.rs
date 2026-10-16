//! GPU time of each part of a decode step, to see where a device spends a token.
//!
//! Each part is queued `reps` times (independent copies) and evaluated in one go, so the
//! time per copy is GPU work without the CPU round trip.

use std::time::Instant;

use mlx_rs::{Array, Dtype};

use crate::{Model, Result};

/// Microseconds of GPU time per call of each decode part, and the per-token estimate.
#[derive(Debug, Clone)]
pub struct Profile {
    /// (part, µs per call, calls per token).
    pub parts: Vec<(&'static str, f64, usize)>,
    /// Σ µs · calls: GPU time of one token without attention, syncs or expert loads.
    pub token_us: f64,
}

/// µs per copy of `f`, `reps` copies evaluated together (after one warm-up round).
fn time(reps: usize, f: &dyn Fn() -> Result<Array>) -> Result<f64> {
    let mut us = 0.0;
    for _ in 0..2 {
        let outs = (0..reps).map(|_| f()).collect::<Result<Vec<_>>>()?;
        let t = Instant::now();
        mlx_rs::transforms::eval(outs.iter())?;
        us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
    }
    Ok(us)
}

impl Model {
    /// Time the parts of a decode step on layer 0 (`reps` copies each).
    ///
    /// # Errors
    ///
    /// Fails if an MLX op fails.
    pub fn profile(&self, reps: usize) -> Result<Profile> {
        let c = &self.cfg;
        let ly = &self.layers[0];
        let qd = (c.heads * c.head_dim) as i32;
        let (nq, nkv, d) = (c.heads as i32, c.kv_heads as i32, c.head_dim as i32);
        let v: Vec<f32> = (0..c.hidden).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.02).collect();
        let hn = Array::from_slice(&v, &[1, 1, c.hidden as i32]).as_dtype(Dtype::Bfloat16)?;
        let o_in = mlx_rs::ops::ones::<f32>(&[1, 1, qd])?.as_dtype(Dtype::Bfloat16)?;
        let qk = mlx_rs::ops::ones::<f32>(&[nq + nkv, d])?.as_dtype(Dtype::Bfloat16)?;
        let k = c.top_k as i32;
        let ids = Array::from_slice(&(0..k as u32).collect::<Vec<_>>(), &[k]);
        let layers = self.layers.len();
        let mut parts = vec![
            ("qkv", time(reps, &|| self.kernels.qmv(&hn, &ly.qkv))?, layers),
            (
                "qk norm + rope",
                time(reps, &|| self.fused.qk_norm_rope(&qk, &ly.q_norm, &ly.k_norm, nq, 7, c.rope_dims as i32))?,
                layers,
            ),
            ("o", time(reps, &|| self.kernels.qmv(&o_in, &ly.o))?, layers),
            ("add + norm", time(reps, &|| Ok(self.fused.add_norm(&hn, &hn, &ly.post_norm)?.1))?, 2 * layers),
            ("router", time(reps, &|| Ok(self.fused.router(&hn, &ly.router)?.0))?, layers),
        ];
        let experts = match (&ly.experts, &self.stream) {
            (Some(x), _) => time(reps, &|| {
                let a = self.kernels.up_swiglu(&hn, &x.up_gate, &ids, k)?;
                self.kernels.qmv_gather(&a, &x.down, &ids, k)
            })?,
            // Slots 0..k of layer 0: the timing does not depend on what they hold.
            (None, Some(st)) => time(reps, &|| self.slab_experts(&hn, &st.slab, &ids))?,
            (None, None) => 0.0,
        };
        parts.push(("experts (8)", experts, layers));
        match &self.flash {
            Some(f) if f.on => parts.push(("lm head (FlashHead)", time(reps, &|| f.logits(&self.head, &hn))?, 1)),
            _ => parts.push(("lm head", time(reps, &|| self.head_logits(&hn))?, 1)),
        }
        let token_us = parts.iter().map(|(_, us, n)| us * *n as f64).sum();
        Ok(Profile { parts, token_us })
    }
}
