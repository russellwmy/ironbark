//! Fused decode kernels for one token: fewer GPU launches per layer.
//!
//! ```text
//!   add_norm      h' = h + r (bf16 round);  hn = RMSNorm(h') · w           1 launch (4+ unfused)
//!   agg_norm      h' = h + Σ_e score_e · y_e (fp32, ordered, bf16 round);  hn = RMSNorm(h') · w
//!   qk_norm_rope  per-head RMSNorm of q and k, RoPE on the first ROPE_DIM dims (0 = none)
//!   router        logits = hn · Gᵀ (fp32); softmax; top-8; renormalize       2 launches (6 unfused)
//! ```
//!
//! Ported from DeepGrove's `maple.py` (deepgrove/maple-preview-2bit-mlx). Reductions follow
//! MLX 0.32 (`rms_single_row`, GEMV, softmax; Apple, MIT) so results match the unfused ops.
//! Shapes are Maple's: hidden 2048, 256 experts, top-8, `head_dim` 128.

use mlx_rs::{Array, Dtype};

use crate::Result;
use crate::kernel::{Kernel, Output, Template};

/// Residual + RMSNorm body. `RESIDUAL` computes `v` for element `j`; `STORE_H` / `STORE_HN`
/// write the outputs (guarded in the aggregate variant, where 8 threadgroups split the stores).
const ADD_NORM: &str = r"
    uint tid = thread_position_in_threadgroup.x;
    constexpr uint N = DIM;
    constexpr uint PT = 4u;
    float hb[PT];
    float ss = 0.0f;
    for (uint i = 0; i < PT; ++i) {
        uint j = tid * PT + i;
        RESIDUAL
        T_ vb = (T_)v;
        STORE_H
        hb[i] = (float)vb;
        ss += hb[i] * hb[i];
    }
    ss = simd_sum(ss);
    threadgroup float sums[32];
    threadgroup float inv_mean[1];
    uint sg = tid / 32u;
    uint lane = tid % 32u;
    if (sg == 0u) sums[lane] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) sums[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u) {
        float tot = simd_sum(sums[lane]);
        if (lane == 0u) inv_mean[0] = metal::precise::rsqrt(tot / (float)N + EPS_);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float scale = inv_mean[0];
    for (uint i = 0; i < PT; ++i) {
        uint j = tid * PT + i;
        STORE_HN
    }
";

const RESIDUAL_ADD: &str = "float v = (float)x[j] + (float)r[j];";

/// Σ over the 8 experts in selection order, fp32 without FMA contraction, rounded once.
const RESIDUAL_AGG: &str = r"
    float agg = 0.0f;
    {
        #pragma clang fp contract(off)
        for (uint e = 0; e < 8; ++e) {
            agg = agg + (float)r[e * N + j] * scores[e];
        }
    }
    float v = (float)x[j] + (float)(T_)agg;
";

const STORE_H: &str = "h_out[j] = vb;";
const STORE_HN: &str = "hn_out[j] = (T_)((float)w[j] * (hb[i] * scale));";
const GUARD: &str = "if (j / (N / 8u) == threadgroup_position_in_grid.x)";

const QK_NORM_ROPE: &str = r"
    uint head = thread_position_in_grid.y;
    uint lane = thread_position_in_grid.x;
    constexpr int per_lane = HEAD_DIM / 32;
    const device T_* xh = x + head * HEAD_DIM;
    device T_* oh = out + head * HEAD_DIM;
    const device T_* wh = head < (uint)NQ ? qw : kw;
    float ss = 0.0f;
    for (int i = 0; i < per_lane; ++i) {
        float v = (float)xh[lane * per_lane + i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    float pos = pos_eps[0];
    float eps = pos_eps[1];
    float scale = metal::precise::rsqrt(ss / HEAD_DIM + eps);
    for (int i = 0; i < per_lane; ++i) {
        int j = lane * per_lane + i;
        float v = (float)(T_)((float)wh[j] * ((float)xh[j] * scale));
        if (ROPE_DIM > 0 && j < ROPE_DIM) {
            constexpr int rhalf = ROPE_DIM > 0 ? ROPE_DIM / 2 : 1;
            int p = j < rhalf ? j : j - rhalf;
            float inv = metal::exp2(-(float(p) / float(rhalf)) * pos_eps[2]);
            float theta = pos * inv;
            float c = metal::fast::cos(theta);
            float s = metal::fast::sin(theta);
            int j2 = j < rhalf ? j + rhalf : j - rhalf;
            float u = (float)(T_)((float)wh[j2] * ((float)xh[j2] * scale));
            v = j < rhalf ? (v * c - u * s) : (u * s + v * c);
        }
        oh[j] = (T_)v;
    }
";

const ROUTER_GEMV: &str = r"
    constexpr int K = 2048;
    constexpr int TM = 4;
    constexpr int TN = 4;
    uint lane = thread_position_in_threadgroup.x % 32;
    uint sg = thread_position_in_threadgroup.x / 32;
    uint row = threadgroup_position_in_grid.x * 16 + sg * TM;
    float result[TM] = {0};
    for (int base = 0; base < K; base += 128) {
        float v[TN];
        #pragma clang loop unroll(full)
        for (int n = 0; n < TN; n++)
            v[n] = float(x[base + lane * TN + n]);
        #pragma clang loop unroll(full)
        for (int m = 0; m < TM; m++) {
            #pragma clang loop unroll(full)
            for (int n = 0; n < TN; n++) {
                float w = float(mat[(row + m) * K + base + lane * TN + n]);
                result[m] += w * v[n];
            }
        }
    }
    #pragma clang loop unroll(full)
    for (int m = 0; m < TM; m++) {
        #pragma clang loop unroll(full)
        for (ushort offset = 16; offset >= 1; offset >>= 1)
            result[m] += simd_shuffle_down(result[m], offset);
        if (lane == 0) logits[row + m] = result[m];
    }
";

/// Softmax over 256 logits into threadgroup `probs` (MLX 0.32 softmax arithmetic).
const ROUTER_SOFTMAX: &str = r"
    #pragma clang fp contract(off)
    uint tid = thread_position_in_threadgroup.x;
    uint lane = tid % 32;
    uint sg = tid / 32;
    threadgroup float maxima[32], sums[32], probs[256];
    float vals[4];
    if (sg == 0) { maxima[lane] = -INFINITY; sums[lane] = 0.0f; }
    float vmax = tid < 64 ? -MAXFLOAT : -INFINITY;
    for (int j = 0; j < 4; j++) {
        vals[j] = tid < 64 ? logits[tid * 4 + j] : -INFINITY;
        vmax = vmax < vals[j] ? vals[j] : vmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vmax = simd_max(vmax);
    if (lane == 0) maxima[sg] = vmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        vmax = simd_max(maxima[lane]);
        if (lane == 0) maxima[0] = vmax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    vmax = maxima[0];
    float sum = 0.0f;
    for (int j = 0; j < 4; j++) {
        vals[j] = tid < 64 ? metal::fast::exp(vals[j] - vmax) : 0.0f;
        sum += vals[j];
    }
    sum = simd_sum(sum);
    if (lane == 0) sums[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        sum = simd_sum(sums[lane]);
        if (lane == 0) sums[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f / sums[0];
    if (tid < 64)
        for (int j = 0; j < 4; j++) probs[tid * 4 + j] = vals[j] * inv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
";

/// Top-8 (picked largest-first into slots 7..0) among experts where `ALLOWED(e)`; after
/// the loop `selected` holds them. Run by SIMD group 0.
const TOP8: &str = r"
        for (int j = 0; j < 8; j++) {
            float p = probs[lane * 8 + j];
            candidates[j] = ALLOWED(lane * 8 + j) ? (metal::isnan(p) ? INFINITY : p) : -INFINITY;
        }
        for (int pick = 7; pick >= 0; pick--) {
            float best = -INFINITY;
            for (int j = 0; j < 8; j++) best = metal::max(best, candidates[j]);
            best = simd_max(best);
            valid[pick] = best != -INFINITY;
            int winner = -1;
            for (int j = 0; j < 8; j++)
                if (candidates[j] == best) winner = int(lane * 8 + j);
            winner = simd_max(winner);
            selected[pick] = uint(winner);
            for (int j = 0; j < 8; j++)
                if (lane * 8 + j == uint(winner)) candidates[j] = -INFINITY;
        }
";

/// Exact router: softmax, top-8, renormalize.
const ROUTER_SELECT: &str = r"
    ROUTER_SOFTMAX
    if (sg == 0) {
        float candidates[8];
        uint selected[8];
        bool valid[8];
        #define ALLOWED(e) true
        TOP8
        float denom = 0.0f;
        for (int j = 0; j < 8; j++) denom = probs[selected[j]] + denom;
        denom = denom + 1e-20f;
        if (lane < 8) {
            indices[lane] = selected[lane];
            scores[lane] = probs[selected[lane]] / denom;
        }
    }
";

/// Hidden size the fused kernels are built for.
pub(crate) const HIDDEN: i32 = 2048;

/// The compiled fused kernels (RMSNorm epsilon baked in).
#[derive(Debug)]
pub(crate) struct Fused {
    add_norm: Kernel,
    agg_norm: Kernel,
    qk: Kernel,
    gemv: Kernel,
    select: Kernel,
    eps: f32,
    log2_theta: f32,
}

/// Router selection source with the shared pieces filled in.
fn router(body: &str) -> String {
    body.replace("ROUTER_SOFTMAX", ROUTER_SOFTMAX).replace("TOP8", TOP8)
}

impl Fused {
    pub(crate) fn new(eps: f32, theta: f32) -> Self {
        let eps_lit = format!("{eps:.10e}f");
        let body = |residual: &str, guarded: bool| {
            let (h, hn) = if guarded {
                (format!("{GUARD} {{ {STORE_H} }}"), format!("{GUARD} {{ {STORE_HN} }}"))
            } else {
                (STORE_H.to_owned(), STORE_HN.to_owned())
            };
            ADD_NORM
                .replace("RESIDUAL", residual)
                .replace("STORE_HN", &hn)
                .replace("STORE_H", &h)
                .replace("EPS_", &eps_lit)
        };
        Fused {
            add_norm: Kernel::new(
                "ironbark_add_rms_norm",
                &["x", "r", "w"],
                &["h_out", "hn_out"],
                &body(RESIDUAL_ADD, false),
                "",
            ),
            agg_norm: Kernel::new(
                "ironbark_aggregate_add_rms_norm",
                &["x", "r", "scores", "w"],
                &["h_out", "hn_out"],
                &body(RESIDUAL_AGG, true),
                "",
            ),
            qk: Kernel::new("ironbark_qk_norm_rope", &["x", "qw", "kw", "pos_eps"], &["out"], QK_NORM_ROPE, ""),
            gemv: Kernel::new("ironbark_router_gemv", &["x", "mat"], &["logits"], ROUTER_GEMV, ""),
            select: Kernel::new(
                "ironbark_router_select",
                &["logits"],
                &["indices", "scores"],
                &router(ROUTER_SELECT),
                "",
            ),
            eps,
            log2_theta: theta.log2(),
        }
    }

    fn norm_outputs(shape: &[i32], dt: Dtype) -> [Output<'_>; 2] {
        [Output { shape, dtype: dt }, Output { shape, dtype: dt }]
    }

    /// `(h + r, RMSNorm(h + r) · w)`, `h`, `r` `[1, 1, 2048]`.
    pub(crate) fn add_norm(&self, h: &Array, r: &Array, w: &Array) -> Result<(Array, Array)> {
        let dt = h.dtype();
        let shape = [1, 1, HIDDEN];
        let mut o = self.add_norm.apply(
            &[h, r, w],
            &[Template::Dtype("T_", dt), Template::Int("DIM", HIDDEN)],
            (HIDDEN / 4, 1, 1),
            (HIDDEN / 4, 1, 1),
            &Self::norm_outputs(&shape, dt),
        )?;
        let hn = o.remove(1);
        Ok((o.remove(0), hn))
    }

    /// `h + Σ_e scores[e] · y[e]` and its RMSNorm · `w`; `y` `[8, 1, 2048]`, `scores` `[8]` f32.
    pub(crate) fn agg_norm(&self, h: &Array, y: &Array, scores: &Array, w: &Array) -> Result<(Array, Array)> {
        let dt = h.dtype();
        let shape = [1, 1, HIDDEN];
        let mut o = self.agg_norm.apply(
            &[h, y, scores, w],
            &[Template::Dtype("T_", dt), Template::Int("DIM", HIDDEN)],
            (8 * HIDDEN / 4, 1, 1),
            (HIDDEN / 4, 1, 1),
            &Self::norm_outputs(&shape, dt),
        )?;
        let hn = o.remove(1);
        Ok((o.remove(0), hn))
    }

    /// Per-head RMSNorm (`qw` for the first `nq` heads, `kw` for the rest) of `qk`
    /// `[heads, head_dim]`, then RoPE on the first `rope_dims` dims at position `pos` (0 = none).
    pub(crate) fn qk_norm_rope(
        &self,
        qk: &Array,
        qw: &Array,
        kw: &Array,
        nq: i32,
        pos: usize,
        rope_dims: i32,
    ) -> Result<Array> {
        let shape = qk.shape().to_vec();
        let pos_eps = Array::from_slice(&[pos as f32, self.eps, self.log2_theta], &[3]);
        let mut o = self.qk.apply(
            &[qk, qw, kw, &pos_eps],
            &[
                Template::Dtype("T_", qk.dtype()),
                Template::Int("HEAD_DIM", shape[1]),
                Template::Int("ROPE_DIM", rope_dims),
                Template::Int("NQ", nq),
            ],
            (32, shape[0], 1),
            (32, 1, 1),
            &[Output {
                shape: &shape,
                dtype: qk.dtype(),
            }],
        )?;
        Ok(o.remove(0))
    }

    /// Router of one token: `hn` `[1, 1, 2048]`, `gate` `[256, 2048]` -> (ids `[8]` u32, scores
    /// `[8]` f32), ids in selection order.
    pub(crate) fn router(&self, hn: &Array, gate: &Array) -> Result<(Array, Array)> {
        let logits = self.logits(hn, gate)?;
        let mut o = self.select.apply(
            &[&logits],
            &[],
            (128, 1, 1),
            (128, 1, 1),
            &[
                Output {
                    shape: &[8],
                    dtype: Dtype::Uint32,
                },
                Output {
                    shape: &[8],
                    dtype: Dtype::Float32,
                },
            ],
        )?;
        let scores = o.remove(1);
        Ok((o.remove(0), scores))
    }

    /// Router logits `[256]` f32 of `hn` `[1, 1, 2048]`.
    fn logits(&self, hn: &Array, gate: &Array) -> Result<Array> {
        Ok(self
            .gemv
            .apply(
                &[hn, gate],
                &[],
                (HIDDEN, 1, 1),
                (128, 1, 1),
                &[Output {
                    shape: &[256],
                    dtype: Dtype::Float32,
                }],
            )?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use mlx_rs::ops::indexing::{IndexOp, take_along_axis};
    use mlx_rs::ops::{argpartition_axis, softmax_axis};
    use mlx_rs::{Array, Dtype, fast};

    use super::{Fused, HIDDEN};

    const EPS: f32 = 1e-6;

    fn vec_bf16(n: i32, seed: i32, scale: f32) -> Array {
        let v: Vec<f32> = (0..n)
            .map(|i| ((i.wrapping_mul(7919).wrapping_add(seed.wrapping_mul(104_729)) % 2001) as f32 - 1000.0) * scale)
            .collect();
        Array::from_slice(&v, &[n]).as_dtype(Dtype::Bfloat16).unwrap()
    }

    fn bits(a: &Array) -> Vec<u32> {
        a.as_dtype(Dtype::Float32).unwrap().as_slice::<f32>().iter().map(|v| v.to_bits()).collect()
    }

    /// Unfused Maple RMSNorm: fp32 normalize and multiply, one rounding.
    fn rms(x: &Array, w: &Array) -> Array {
        let w = w.as_dtype(Dtype::Float32).unwrap();
        fast::rms_norm(x.as_dtype(Dtype::Float32).unwrap(), Some(&w), EPS)
            .unwrap()
            .as_dtype(x.dtype())
            .unwrap()
    }

    #[test]
    fn add_norm_matches_unfused() {
        let f = Fused::new(EPS, 10_000.0);
        let h = vec_bf16(HIDDEN, 1, 1e-3).reshape(&[1, 1, HIDDEN]).unwrap();
        let r = vec_bf16(HIDDEN, 2, 2e-3).reshape(&[1, 1, HIDDEN]).unwrap();
        let w = vec_bf16(HIDDEN, 3, 1e-3);
        let (hh, hn) = f.add_norm(&h, &r, &w).unwrap();
        let want = &h + &r;
        assert_eq!(bits(&hh), bits(&want));
        assert_eq!(bits(&hn), bits(&rms(&want, &w)));
    }

    #[test]
    fn agg_norm_matches_unfused() {
        let f = Fused::new(EPS, 10_000.0);
        let h = vec_bf16(HIDDEN, 4, 1e-3).reshape(&[1, 1, HIDDEN]).unwrap();
        let y = vec_bf16(8 * HIDDEN, 5, 3e-3).reshape(&[8, 1, HIDDEN]).unwrap();
        let s: Vec<f32> = vec![0.3, 0.2, 0.15, 0.1, 0.1, 0.08, 0.04, 0.03];
        let scores = Array::from_slice(&s, &[8]);
        let w = vec_bf16(HIDDEN, 6, 1e-3);
        let (hh, hn) = f.agg_norm(&h, &y, &scores, &w).unwrap();
        // Reference order: fp32 products summed over the 8 experts, rounded once.
        let moe = (y.as_dtype(Dtype::Float32).unwrap() * scores.reshape(&[8, 1, 1]).unwrap())
            .sum_axis(0, true)
            .unwrap()
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let want = &h + &moe;
        assert_eq!(bits(&hh), bits(&want));
        assert_eq!(bits(&hn), bits(&rms(&want, &w)));
    }

    #[test]
    fn qk_norm_rope_matches_unfused_far_into_the_sequence() {
        let f = Fused::new(EPS, 10_000.0);
        let (nq, nkv, d) = (16, 4, 128);
        let qk = vec_bf16((nq + nkv) * d, 7, 4e-3).reshape(&[nq + nkv, d]).unwrap();
        let (qw, kw) = (vec_bf16(d, 8, 1e-3), vec_bf16(d, 9, 1e-3));
        for pos in [0usize, 7, 700, 5000] {
            let got = f.qk_norm_rope(&qk, &qw, &kw, nq, pos, 64).unwrap();
            let q = rms(&qk.index(..nq), &qw).reshape(&[1, nq, 1, d]).unwrap();
            let k = rms(&qk.index(nq..), &kw).reshape(&[1, nkv, 1, d]).unwrap();
            let rope = |x: &Array| fast::rope(x, 64, false, 10_000.0, 1.0, pos as i32, None).unwrap();
            let want = mlx_rs::ops::concatenate(
                &[
                    rope(&q).reshape(&[nq, d]).unwrap(),
                    rope(&k).reshape(&[nkv, d]).unwrap(),
                ],
                0,
            )
            .unwrap();
            assert_eq!(bits(&got), bits(&want), "position {pos}");
            // No RoPE (full-attention layers): normalization only.
            let got = f.qk_norm_rope(&qk, &qw, &kw, nq, pos, 0).unwrap();
            let want = mlx_rs::ops::concatenate(&[rms(&qk.index(..nq), &qw), rms(&qk.index(nq..), &kw)], 0).unwrap();
            assert_eq!(bits(&got), bits(&want), "position {pos}, no rope");
        }
    }

    #[test]
    fn router_matches_unfused() {
        let f = Fused::new(EPS, 10_000.0);
        let hn = vec_bf16(HIDDEN, 10, 1e-3).reshape(&[1, 1, HIDDEN]).unwrap();
        let gate = vec_bf16(256 * HIDDEN, 11, 2e-3).reshape(&[256, HIDDEN]).unwrap();
        let (ids, scores) = f.router(&hn, &gate).unwrap();
        // Unfused: fp32 logits, softmax, argpartition top-8, renormalize.
        let logits = hn
            .as_dtype(Dtype::Float32)
            .unwrap()
            .matmul(gate.as_dtype(Dtype::Float32).unwrap().transpose().unwrap())
            .unwrap();
        let probs = softmax_axis(&logits, -1, true).unwrap();
        let inds = argpartition_axis(&probs, 248, -1).unwrap().index((.., .., 248..));
        let w = take_along_axis(&probs, &inds, -1).unwrap();
        let w = &w / (w.sum_axis(-1, true).unwrap() + 1e-20);
        let want_ids = inds.as_dtype(Dtype::Uint32).unwrap();
        assert_eq!(ids.as_slice::<u32>(), want_ids.as_slice::<u32>());
        assert_eq!(bits(&scores), bits(&w));
    }
}
