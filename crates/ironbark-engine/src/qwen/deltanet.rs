//! Gated DeltaNet, the linear-attention mixer of Qwen3.5 (mlx-lm `GatedDeltaNet`).
//!
//! ```text
//!   qkv = x·Wqkv   z = x·Wz   b = x·Wb   a = x·Wa
//!   c   = silu(causal depthwise conv over [conv tail ‖ qkv])      kernel 4, tail = last 3 rows
//!   q, k, v = split(c)        q, k: L2-normed per head (q also × Dk^-1/2)
//!   β = sigmoid(b)   g = exp(−exp(A_log) · softplus(a + dt_bias))   per token and value head
//!   per token, per value head (state S [Dv, Dk], fp32; value head h reads key head h / (Hv/Hk)):
//!     S = g·S;   S += ((v − S·k) · β) ⊗ k;   y = S·q
//!   out = Wo · (silu(z) · rms_norm(y))                              norm per head, fp32 product
//! ```
//!
//! The recurrence runs in one Metal kernel launch over all the tokens ([`step_kernel`], mlx-lm's
//! `gated_delta_step`): a SIMD group per (value head, value row) keeps its row of the state in
//! registers, `Dk / 32` floats per lane. [`recurrence`] is the same with MLX ops, one token at
//! a time (the reference the kernel is tested against).

use ironbark_mlx::ops::indexing::IndexOp;
use ironbark_mlx::ops::{broadcast_to, concatenate, conv1d, logaddexp, quantized_matmul, sigmoid, silu, stack};
use ironbark_mlx::{Array, Dtype, fast};

use super::cast;
use crate::kernel::{Kernel, Output, Template};
use crate::model::QMat;
use crate::quant::Tensors;
use crate::{Error, Result};

/// Head counts and sizes.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    /// Conv kernel width.
    pub conv: usize,
}

impl Shape {
    #[must_use]
    pub fn key_dim(&self) -> usize {
        self.k_heads * self.k_dim
    }

    #[must_use]
    pub fn value_dim(&self) -> usize {
        self.v_heads * self.v_dim
    }

    /// Channels of the conv: q, k and v.
    #[must_use]
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

/// A projection: affine-quantized, or dense (a checkpoint that left it unquantized).
pub(crate) enum Linear {
    Quant(QMat),
    Dense(Array),
}

impl Linear {
    pub(crate) fn load(t: &mut Tensors, prefix: &str, group: usize) -> Result<Self> {
        if t.0.contains_key(&format!("{prefix}.scales")) {
            Ok(Linear::Quant(t.qmat(prefix, group)?))
        } else {
            Ok(Linear::Dense(t.take(&format!("{prefix}.weight"))?))
        }
    }

    /// `x · Wᵀ`.
    pub(crate) fn apply(&self, x: &Array) -> Result<Array> {
        Ok(match self {
            Linear::Quant(m) => quantized_matmul(x, &m.w, &m.scales, &m.biases, true, m.group, m.bits)?,
            Linear::Dense(w) => x.matmul(w.transpose()?)?,
        })
    }
}

/// One DeltaNet layer's weights.
pub(crate) struct DeltaNet {
    qkv: Linear,
    z: Linear,
    b: Linear,
    a: Linear,
    out: Linear,
    /// Depthwise conv `[conv_dim, kernel, 1]` (MLX layout).
    conv: Array,
    dt_bias: Array,
    /// fp32.
    a_log: Array,
    /// Per-head RMSNorm weight `[v_dim]` (not zero-centered).
    norm: Array,
    shape: Shape,
}

/// A DeltaNet layer's caches: the conv's last `kernel − 1` inputs `[B, kernel − 1, conv_dim]`
/// and the recurrent state `[B, v_heads, v_dim, k_dim]` (fp32), a row per sequence. `None`
/// until the first token.
#[derive(Clone, Default)]
pub(crate) struct State {
    conv: Option<Array>,
    ssm: Option<Array>,
}

impl State {
    pub(crate) fn arrays(&self) -> Vec<&Array> {
        [&self.conv, &self.ssm].into_iter().flatten().collect()
    }

    /// Single sequences' states (each past its first token) stacked a row per sequence.
    pub(crate) fn stack(rows: &[&Self]) -> Result<Self> {
        let cat = |get: fn(&Self) -> Option<&Array>| -> Result<Array> {
            let parts: Vec<&Array> = rows
                .iter()
                .map(|r| get(r).ok_or_else(|| Error::Format("batch: DeltaNet state missing".into())))
                .collect::<Result<_>>()?;
            Ok(concatenate(&parts, 0)?)
        };
        Ok(Self {
            conv: Some(cat(|s| s.conv.as_ref())?),
            ssm: Some(cat(|s| s.ssm.as_ref())?),
        })
    }
}

/// `x · c` with `c` in `x`'s dtype (a Python scalar in mlx-lm).
fn scale(x: &Array, c: f32) -> Result<Array> {
    Ok(x.multiply(Array::from_f32(c).as_dtype(x.dtype())?)?)
}

impl DeltaNet {
    pub(crate) fn load(t: &mut Tensors, prefix: &str, group: usize, shape: Shape) -> Result<Self> {
        let n = |s: &str| format!("{prefix}.{s}");
        let conv = t.take(&n("conv1d.weight"))?;
        let want = [shape.conv_dim() as i32, shape.conv as i32, 1];
        if conv.shape() != want {
            return Err(Error::Format(format!("{}: shape {:?}, want {want:?}", n("conv1d.weight"), conv.shape())));
        }
        Ok(DeltaNet {
            qkv: Linear::load(t, &n("in_proj_qkv"), group)?,
            z: Linear::load(t, &n("in_proj_z"), group)?,
            b: Linear::load(t, &n("in_proj_b"), group)?,
            a: Linear::load(t, &n("in_proj_a"), group)?,
            out: Linear::load(t, &n("out_proj"), group)?,
            conv,
            dt_bias: t.take(&n("dt_bias"))?,
            a_log: t.take(&n("A_log"))?.as_dtype(Dtype::Float32)?,
            norm: t.take(&n("norm.weight"))?,
            shape,
        })
    }

    pub(crate) fn arrays(&self) -> Vec<&Array> {
        let mut all = vec![&self.conv, &self.dt_bias, &self.a_log, &self.norm];
        for l in [&self.qkv, &self.z, &self.b, &self.a, &self.out] {
            if let Linear::Dense(w) = l {
                all.push(w);
            }
        }
        all
    }

    pub(crate) fn mats(&self) -> Vec<&QMat> {
        [&self.qkv, &self.z, &self.b, &self.a, &self.out]
            .into_iter()
            .filter_map(|l| match l {
                Linear::Quant(m) => Some(m),
                Linear::Dense(_) => None,
            })
            .collect()
    }

    /// The mixer on `x` `[B, L, hidden]` (B sequences, each with its row of `st`), advancing
    /// `st` by `L` tokens; the recurrence with
    /// `step` ([`step_kernel`]) or, without, with MLX ops.
    pub(crate) fn forward(&self, x: &Array, st: &mut State, eps: f32, step: Option<&Kernel>) -> Result<Array> {
        let s = self.shape;
        let (bs, n) = (x.shape()[0], x.shape()[1]);
        let dt = x.dtype();
        let (hk, hv, dk, dv) = (s.k_heads as i32, s.v_heads as i32, s.k_dim as i32, s.v_dim as i32);
        let (kd, cd, tail) = (s.key_dim() as i32, s.conv_dim() as i32, s.conv as i32 - 1);

        let qkv = self.qkv.apply(x)?;
        let z = self.z.apply(x)?.reshape(&[bs, n, hv, dv])?;
        let b = self.b.apply(x)?;
        let a = self.a.apply(x)?;

        let prev = match &st.conv {
            Some(c) => c.clone(),
            None => Array::zeros::<f32>(&[bs, tail, cd])?.as_dtype(dt)?,
        };
        let conv_in = concatenate(&[&prev, &qkv], 1)?;
        let len = conv_in.shape()[1];
        st.conv = Some(conv_in.index((.., len - tail..)));
        let c = silu(conv1d(&conv_in, &self.conv, 1, 0, 1, cd)?)?;

        let q = c.index((.., .., ..kd)).reshape(&[bs, n, hk, dk])?;
        let k = c.index((.., .., kd..2 * kd)).reshape(&[bs, n, hk, dk])?;
        let v = c.index((.., .., 2 * kd..)).reshape(&[bs, n, hv, dv])?;
        // L2 norms as RMS norms: eps is added to Σx², rms_norm adds it to mean(x²).
        let inv = (dk as f32).powf(-0.5);
        let q = scale(&fast::rms_norm(&q, None, 1e-6 * inv * inv)?, inv * inv)?;
        let k = scale(&fast::rms_norm(&k, None, 1e-6 * inv * inv)?, inv)?;

        let beta = sigmoid(&b)?;
        let zero = Array::from_f32(0.0).as_dtype(dt)?;
        let softplus = logaddexp(a.add(&self.dt_bias)?, &zero)?;
        let g = self.a_log.exp()?.negative()?.multiply(&softplus)?.exp()?;

        let state = match &st.ssm {
            Some(s) => s.clone(),
            None => Array::zeros::<f32>(&[bs, hv, dv, dk])?,
        };
        let (y, state) = if let Some(kernel) = step {
            metal(kernel, s, &q, &k, &v, &g, &beta, &state)?
        } else {
            // Each key head serves Hv / Hk consecutive value heads.
            let rep = |t: &Array| -> Result<Array> {
                let e = t.expand_dims(-2)?;
                Ok(broadcast_to(&e, &[bs, n, hk, hv / hk, dk])?.reshape(&[bs, n, hv, dk])?)
            };
            recurrence(&rep(&q)?, &rep(&k)?, &v, &g, &beta, state)?
        };
        st.ssm = Some(state);

        // silu(z) · rms_norm(y) in fp32, back to the activation dtype.
        let yn = fast::rms_norm(&y, Some(&self.norm), eps)?;
        let gated = silu(cast(&z, Dtype::Float32)?)?.multiply(cast(&yn, Dtype::Float32)?)?.as_dtype(y.dtype())?;
        self.out.apply(&gated.reshape(&[bs, n, hv * dv])?)
    }
}

/// mlx-lm's `gated_delta_step` kernel (no mask, one gate per head and token).
const STEP: &str = r"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    // q, k: [B, T, Hk, Dk]
    auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
    auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

    // v, y: [B, T, Hv, Dv]
    auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
    y += b_idx * T * Hv * Dv + hv_idx * Dv;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto dv_idx = thread_position_in_grid.y;

    // state_in, state_out: [B, Hv, Dv, Dk]
    auto i_state = state_in + (n * Dv + dv_idx) * Dk;
    auto o_state = state_out + (n * Dv + dv_idx) * Dk;

    float state[n_per_t];
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      state[i] = static_cast<float>(i_state[s_idx]);
    }

    // g, beta: [B, T, Hv]
    auto g_ = g + b_idx * T * Hv;
    auto beta_ = beta + b_idx * T * Hv;

    for (int t = 0; t < T; ++t) {
      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] * g_[hv_idx];
        kv_mem += state[i] * k_[s_idx];
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = state[i] + k_[s_idx] * delta;
        out += state[i] * q_[s_idx];
      }
      out = simd_sum(out);
      if (thread_index_in_simdgroup == 0) {
        y[dv_idx] = static_cast<InT>(out);
      }
      q_ += Hk * Dk;
      k_ += Hk * Dk;
      v_ += Hv * Dv;
      y += Hv * Dv;
      g_ += Hv;
      beta_ += Hv;
    }
    for (int i = 0; i < n_per_t; ++i) {
      auto s_idx = n_per_t * dk_idx + i;
      o_state[s_idx] = static_cast<StT>(state[i]);
    }
";

/// The recurrence kernel ([`metal`]); `Dk` must be a multiple of 32.
pub(crate) fn step_kernel() -> Kernel {
    Kernel::new(
        "ironbark_gated_delta_step",
        &["q", "k", "v", "g", "beta", "state_in", "T"],
        &["y", "state_out"],
        STEP,
        "",
    )
}

/// [`recurrence`] in one launch of `kernel`: `q`, `k` `[B, L, Hk, Dk]` (one per key head), `v`
/// `[B, L, Hv, Dv]`, `g` (fp32), `beta` `[B, L, Hv]`, `state` `[B, Hv, Dv, Dk]` (fp32).
#[expect(clippy::too_many_arguments, reason = "the kernel's inputs")]
fn metal(
    kernel: &Kernel,
    s: Shape,
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array)> {
    let (b, n) = (q.shape()[0], q.shape()[1]);
    let (hk, hv, dk, dv) = (s.k_heads as i32, s.v_heads as i32, s.k_dim as i32, s.v_dim as i32);
    if dk % 32 != 0 {
        return Err(Error::Format(format!("gated delta kernel: key head dim {dk} is not a multiple of 32")));
    }
    let t = Array::from_int(n);
    let mut out = kernel.apply(
        &[q, k, v, &cast(g, Dtype::Float32)?, beta, state, &t],
        &[
            Template::Dtype("InT", q.dtype()),
            Template::Dtype("StT", Dtype::Float32),
            Template::Int("Dk", dk),
            Template::Int("Dv", dv),
            Template::Int("Hk", hk),
            Template::Int("Hv", hv),
        ],
        (32, dv, b * hv),
        (32, 4, 1),
        &[
            Output {
                shape: &[b, n, hv, dv],
                dtype: q.dtype(),
            },
            Output {
                shape: &[b, hv, dv, dk],
                dtype: Dtype::Float32,
            },
        ],
    )?;
    let state = out.pop().ok_or_else(|| Error::Format("gated delta kernel: no state".into()))?;
    let y = out.pop().ok_or_else(|| Error::Format("gated delta kernel: no output".into()))?;
    Ok((y, state))
}

/// The gated delta rule over `L` tokens: `q`, `k` `[B, L, H, Dk]` (heads already repeated to
/// the value heads), `v` `[B, L, H, Dv]`, `g`, `beta` `[B, L, H]`, `state` `[B, H, Dv, Dk]`
/// (fp32) → outputs `[B, L, H, Dv]` (in `q`'s dtype) and the new state.
pub(crate) fn recurrence(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    mut state: Array,
) -> Result<(Array, Array)> {
    let n = q.shape()[1];
    let mut ys = Vec::with_capacity(n as usize);
    for t in 0..n {
        let (qt, kt, vt) = (q.index((.., t)), k.index((.., t)), v.index((.., t)));
        let (gt, bt) = (g.index((.., t)), beta.index((.., t)));
        // [B, H] → [B, H, 1, 1]; [B, H, Dk] → [B, H, 1, Dk].
        state = state.multiply(gt.expand_dims(-1)?.expand_dims(-1)?)?;
        let kr = kt.expand_dims(-2)?;
        let kv = state.multiply(&kr)?.sum_axis(-1, false)?;
        let delta = vt.subtract(&kv)?.multiply(bt.expand_dims(-1)?)?;
        state = state.add(kr.multiply(delta.expand_dims(-1)?)?)?;
        let y = state.multiply(qt.expand_dims(-2)?)?.sum_axis(-1, false)?;
        ys.push(y.as_dtype(q.dtype())?);
    }
    Ok((stack(&ys, 1)?, state))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic pseudo-random sequence in [−1, 1).
    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn recurrence_matches_a_plain_loop() {
        let (l, h, dk, dv) = (5usize, 2usize, 4usize, 3usize);
        let q = noise(l * h * dk, 1);
        let k = noise(l * h * dk, 2);
        let v = noise(l * h * dv, 3);
        let g: Vec<f32> = noise(l * h, 4).iter().map(|x| 0.5 + 0.4 * x).collect();
        let beta: Vec<f32> = noise(l * h, 5).iter().map(|x| 0.5 + 0.5 * x).collect();
        let arr = |x: &[f32], s: &[i32]| Array::from_slice(x, s);
        let (li, hi, ki, vi) = (l as i32, h as i32, dk as i32, dv as i32);
        let (y, st) = recurrence(
            &arr(&q, &[1, li, hi, ki]),
            &arr(&k, &[1, li, hi, ki]),
            &arr(&v, &[1, li, hi, vi]),
            &arr(&g, &[1, li, hi]),
            &arr(&beta, &[1, li, hi]),
            Array::zeros::<f32>(&[1, hi, vi, ki]).unwrap(),
        )
        .unwrap();
        let (y, st) = (y.as_slice::<f32>().to_vec(), st.as_slice::<f32>().to_vec());

        let mut s = vec![0.0f32; h * dv * dk];
        for t in 0..l {
            for hh in 0..h {
                let at = |base: usize, d: usize, i: usize| base + ((t * h + hh) * d) + i;
                let gt = g[t * h + hh];
                let bt = beta[t * h + hh];
                let si = |a: usize, b: usize| (hh * dv + a) * dk + b;
                for x in &mut s[hh * dv * dk..(hh + 1) * dv * dk] {
                    *x *= gt;
                }
                for a in 0..dv {
                    let kv: f32 = (0..dk).map(|b| s[si(a, b)] * k[at(0, dk, b)]).sum();
                    let delta = (v[at(0, dv, a)] - kv) * bt;
                    for b in 0..dk {
                        s[si(a, b)] += k[at(0, dk, b)] * delta;
                    }
                }
                for a in 0..dv {
                    let want: f32 = (0..dk).map(|b| s[si(a, b)] * q[at(0, dk, b)]).sum();
                    let got = y[(t * h + hh) * dv + a];
                    assert!((got - want).abs() < 1e-5, "t {t} head {hh} row {a}: {got} vs {want}");
                }
            }
        }
        for (a, b) in st.iter().zip(&s) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    /// The kernel against the ops: 2 key heads serving 4 value heads, from a nonzero state.
    #[test]
    fn kernel_matches_the_ops() {
        let s = Shape {
            k_heads: 2,
            v_heads: 4,
            k_dim: 64,
            v_dim: 8,
            conv: 4,
        };
        let (l, hk, hv, dk, dv) = (7i32, 2i32, 4i32, 64i32, 8i32);
        let len = |d: &[i32]| d.iter().product::<i32>() as usize;
        let arr = |seed: u64, d: &[i32]| Array::from_slice(&noise(len(d), seed), d);
        let (q, k, v) = (arr(1, &[1, l, hk, dk]), arr(2, &[1, l, hk, dk]), arr(3, &[1, l, hv, dv]));
        let g = Array::from_slice(
            &noise(len(&[1, l, hv]), 4).iter().map(|x| 0.6 + 0.3 * x).collect::<Vec<_>>(),
            &[1, l, hv],
        );
        let beta = Array::from_slice(
            &noise(len(&[1, l, hv]), 5).iter().map(|x| 0.5 + 0.4 * x).collect::<Vec<_>>(),
            &[1, l, hv],
        );
        let state = arr(6, &[1, hv, dv, dk]);
        let (ym, sm) = metal(&step_kernel(), s, &q, &k, &v, &g, &beta, &state).unwrap();
        let rep = |t: &Array| {
            broadcast_to(t.expand_dims(-2).unwrap(), &[1, l, hk, hv / hk, dk])
                .unwrap()
                .reshape(&[1, l, hv, dk])
                .unwrap()
        };
        let (yo, so) = recurrence(&rep(&q), &rep(&k), &v, &g, &beta, state).unwrap();
        let diff = |a: &Array, b: &Array| a.subtract(b).unwrap().abs().unwrap().max(None).unwrap().item_exact::<f32>();
        assert!(diff(&ym, &yo) < 1e-4, "outputs differ by {}", diff(&ym, &yo));
        assert!(diff(&sm, &so) < 1e-4, "states differ by {}", diff(&sm, &so));
    }

    /// A layer with random dense weights: 2 key heads serving 4 value heads of `dim`, hidden 16.
    fn toy(dim: usize) -> DeltaNet {
        let shape = Shape {
            k_heads: 2,
            v_heads: 4,
            k_dim: dim,
            v_dim: 8,
            conv: 4,
        };
        let hidden = 16i32;
        let dense = |rows: usize, cols: i32, seed: u64| {
            Linear::Dense(Array::from_slice(&noise(rows * cols as usize, seed), &[rows as i32, cols]))
        };
        let cd = shape.conv_dim();
        DeltaNet {
            qkv: dense(cd, hidden, 11),
            z: dense(shape.value_dim(), hidden, 12),
            b: dense(shape.v_heads, hidden, 13),
            a: dense(shape.v_heads, hidden, 14),
            out: dense(hidden as usize, shape.value_dim() as i32, 15),
            conv: Array::from_slice(&noise(cd * 4, 16), &[cd as i32, 4, 1]),
            dt_bias: Array::from_slice(&noise(4, 17), &[4]),
            a_log: Array::from_slice(&noise(4, 18), &[4]),
            norm: Array::from_slice(&[1.0f32; 8], &[8]),
            shape,
        }
    }

    /// Two sequences in one batch: each row equal to that sequence alone (outputs and
    /// states), with the ops and with the kernel.
    #[test]
    fn batch_rows_are_independent() {
        let d = toy(32);
        let kernel = step_kernel();
        for step in [None, Some(&kernel)] {
            let x = Array::from_slice(&noise(2 * 5 * 16, 21), &[2, 5, 16]);
            let mut both = State::default();
            let y = d.forward(&x, &mut both, 1e-6, step).unwrap();
            for r in 0..2 {
                let mut one = State::default();
                let yr = d.forward(&x.index(r..=r), &mut one, 1e-6, step).unwrap();
                let dy = y.index(r..=r).subtract(&yr).unwrap().abs().unwrap().max(None).unwrap();
                assert!(dy.item_exact::<f32>() < 1e-4, "row {r}: {}", dy.item_exact::<f32>());
                let ds = both.ssm.as_ref().unwrap().index(r..=r).subtract(one.ssm.unwrap()).unwrap();
                assert!(ds.abs().unwrap().max(None).unwrap().item_exact::<f32>() < 1e-4);
            }
        }
    }

    /// One forward over 6 tokens must equal 6 one-token forwards (the conv tail and the state
    /// carry everything between them).
    #[test]
    fn prefill_equals_token_by_token() {
        let d = toy(8);
        let hidden = 16i32;
        let x = Array::from_slice(&noise(6 * 16, 19), &[1, 6, hidden]);
        let mut all = State::default();
        let whole = d.forward(&x, &mut all, 1e-6, None).unwrap();
        let mut one = State::default();
        let mut parts = Vec::new();
        for t in 0..6 {
            parts.push(d.forward(&x.index((.., t..=t)), &mut one, 1e-6, None).unwrap());
        }
        let parts = concatenate(&parts, 1).unwrap();
        let diff = whole.subtract(&parts).unwrap().abs().unwrap().max(None).unwrap();
        assert!(diff.item_exact::<f32>() < 1e-4, "{}", diff.item_exact::<f32>());
        let sd = all.ssm.unwrap().subtract(one.ssm.unwrap()).unwrap().abs().unwrap().max(None).unwrap();
        assert!(sd.item_exact::<f32>() < 1e-4);
    }
}
