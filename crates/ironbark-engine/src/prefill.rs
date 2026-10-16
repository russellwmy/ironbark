//! Batched prefill: a chunk of prompt tokens per pass, with matrix-matrix kernels.
//!
//! ```text
//!   ids (≤ 512) ─▶ embed ─▶ 24 × [ RMSNorm ─▶ QKV qmm ─▶ QK norm, RoPE ─▶ attention ─▶ O qmm
//!                                  RMSNorm ─▶ router ─▶ experts (sorted by expert, gather_qmm) ]
//!                          ─▶ RMSNorm ─▶ head (last position, or all for perplexity)
//! ```
//!
//! Same arithmetic as mlx-lm's prefill of Maple: MLX `quantized_matmul` / `gather_qmm` on the
//! affine group layout (row scales expanded per layer on the fly), fp32 norms and router,
//! bf16 residual. Attention sees the cached keys plus the chunk: causal on full layers, and
//! a 512-token window mask on sliding layers. Afterwards the KV cache has the same layout as
//! after token-by-token decoding, so decode continues unchanged.
//!
//! Streamed experts: per layer the chunk's routing is read back, and its distinct experts are
//! processed in groups that fit the layer's cache: pin (loading on a miss), copy the group's
//! records out of the slab into MLX's layout, run its (token, expert) pairs, unpin. When the
//! cache holds every expert, the layer's slab rows are read in place (`slab_experts`; trit
//! records are unpacked once per layer).

use ironbark_mlx::fast::ScaledDotProductAttentionMask;
use ironbark_mlx::ops::indexing::{IndexOp, TryIndexMutOp, take_along_axis};
use ironbark_mlx::ops::{
    argpartition_axis, argsort_axis, broadcast_to, clip, concatenate, dequantize, floor_divide, gather_qmm, minimum,
    quantized_matmul, silu, softmax_axis,
};
use ironbark_mlx::{Array, Dtype, fast};

use crate::model::{KV_STEP, Kv, Model, Streaming};
use crate::row::{Kernels, RowMat, record_mats};
use crate::{Error, Result};

/// Longest chunk per pass (≤ the sliding window, so a chunk's window keys are the chunk plus
/// the 511 keys before it).
pub const CHUNK: usize = 512;

/// SwiGLU clamp of the experts.
const CLAMP: f32 = 7.0;

/// Maple RMSNorm: fp32 normalize and multiply, one rounding to the input dtype.
fn rms_norm(x: &Array, w: &Array, eps: f32) -> Result<Array> {
    let w = w.as_dtype(Dtype::Float32)?;
    Ok(fast::rms_norm(x.as_dtype(Dtype::Float32)?, Some(&w), eps)?.as_dtype(x.dtype())?)
}

/// `m`'s weights with MLX's affine group layout (α and −α per 128-weight group), for the
/// stock matrix-matrix kernels. Built per use; freed with the pass.
fn grouped(m: &RowMat) -> Result<(Array, Array)> {
    let mut shape = m.scales.shape().to_vec();
    shape.push(m.k / 128);
    let s = broadcast_to(m.scales.expand_dims(-1)?, &shape)?.contiguous()?;
    let b = s.negative()?;
    Ok((s, b))
}

/// `x · Wᵀ` for a chunk `x` `[1, L, k]`.
fn qmm(x: &Array, m: &RowMat) -> Result<Array> {
    let (s, b) = grouped(m)?;
    Ok(quantized_matmul(x, &m.w, &s, &b, true, 128, 2)?)
}

impl Kv {
    /// Store a chunk's `k`, `v` (`[1, kv_heads, L, d]`) at positions `pos..pos + L`; return
    /// the keys and values the chunk attends over and its mask.
    fn update_chunk(
        &mut self,
        k: &Array,
        v: &Array,
        pos: i32,
        window: Option<i32>,
    ) -> Result<(Array, Array, Option<Array>)> {
        let l = k.shape()[2];
        let s = k.shape().to_vec();
        let cap = self.k.as_ref().map_or(0, |a| a.shape()[2]);
        let want = match window {
            Some(w) => w,
            None if pos + l <= cap => cap,
            None => (pos + l + KV_STEP - 1) / KV_STEP * KV_STEP,
        };
        if want > cap {
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
        match window {
            None => {
                ks.try_index_mut((.., .., pos..pos + l), k)?;
                vs.try_index_mut((.., .., pos..pos + l), v)?;
                let n = pos + l;
                Ok((ks.index((.., .., ..n)), vs.index((.., .., ..n)), None))
            }
            Some(w) => {
                // Keys before the chunk, oldest first (the ring holds position p at p % w).
                let old = pos.min(w - 1);
                let slots: Vec<i32> = (pos - old..pos).map(|p| p % w).collect();
                let slots = Array::from_slice(&slots, &[old]);
                let (ak, av) = if old > 0 {
                    (
                        concatenate(&[ks.take_axis(&slots, 2)?, k.clone()], 2)?,
                        concatenate(&[vs.take_axis(&slots, 2)?, v.clone()], 2)?,
                    )
                } else {
                    (k.clone(), v.clone())
                };
                // Query i (position pos + i) sees keys at positions in (pos + i - w, pos + i].
                let total = old + l;
                let mask: Vec<bool> = (0..l)
                    .flat_map(|i| {
                        (0..total).map(move |j| {
                            let (qp, kp) = (pos + i, pos - old + j);
                            kp <= qp && kp > qp - w
                        })
                    })
                    .collect();
                let mask = Array::from_slice(&mask, &[l, total]);
                // Then write the chunk into the ring (only its last w keys survive).
                for i in (l - l.min(w))..l {
                    let slot = (pos + i) % w;
                    ks.try_index_mut((.., .., slot..=slot), &k.index((.., .., i..=i)))?;
                    vs.try_index_mut((.., .., slot..=slot), &v.index((.., .., i..=i)))?;
                }
                Ok((ak, av, Some(mask)))
            }
        }
    }
}

impl Model {
    /// Run prompt tokens through the model in chunks of up to [`CHUNK`]. Returns the logits of
    /// the last token (`[1, 1, vocab]`), or of every token (`[1, n, vocab]`) with `all`.
    ///
    /// Streamed experts go through the cache in groups (see `streamed_experts`).
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails or `ids` is empty.
    pub fn prefill(&mut self, ids: &[u32], all: bool) -> Result<Array> {
        if ids.is_empty() {
            return Err(Error::Format("empty prompt".into()));
        }
        // Streamed prefill may need every slot of a layer.
        self.release_held()?;
        self.ensure_rows(ids)?;
        let mut out = Vec::new();
        for chunk in ids.chunks(CHUNK) {
            let logits = self.prefill_chunk(chunk, all)?;
            logits.eval()?;
            out.push(logits);
        }
        if all {
            Ok(concatenate(&out, 1)?)
        } else {
            Ok(out.pop().unwrap_or_else(|| unreachable!("ids is not empty")))
        }
    }

    fn prefill_chunk(&mut self, ids: &[u32], all: bool) -> Result<Array> {
        let c = self.cfg.clone();
        let l = ids.len() as i32;
        let (nq, nkv, d) = (c.heads as i32, c.kv_heads as i32, c.head_dim as i32);
        let (qd, kd) = (nq * d, nkv * d);
        let pos = self.pos as i32;
        let idx = Array::from_slice(ids, &[l]);
        let e = &self.embed;
        let mut h = dequantize(
            e.w.take_axis(&idx, 0)?,
            e.scales.take_axis(&idx, 0)?,
            &e.biases.take_axis(&idx, 0)?,
            e.group,
            e.bits,
        )?
        .reshape(&[1, l, c.hidden as i32])?;
        let k8 = c.top_k as i32;
        let (hi, lo) = (Array::from_f32(CLAMP).as_dtype(h.dtype())?, Array::from_f32(-CLAMP).as_dtype(h.dtype())?);
        for i in 0..self.layers.len() {
            let ly = &self.layers[i];
            // Attention.
            let x = rms_norm(&h, &ly.in_norm, c.eps)?;
            let qkv = qmm(&x, &ly.qkv)?;
            let q = qkv.index((.., .., ..qd)).reshape(&[1, l, nq, d])?;
            let k = qkv.index((.., .., qd..qd + kd)).reshape(&[1, l, nkv, d])?;
            let v = qkv.index((.., .., qd + kd..)).reshape(&[1, l, nkv, d])?;
            let mut q = rms_norm(&q, &ly.q_norm, c.eps)?.transpose_axes(&[0, 2, 1, 3])?;
            let mut k = rms_norm(&k, &ly.k_norm, c.eps)?.transpose_axes(&[0, 2, 1, 3])?;
            let v = v.transpose_axes(&[0, 2, 1, 3])?;
            if ly.sliding {
                q = fast::rope(&q, c.rope_dims as i32, false, c.theta, 1.0, pos, None)?;
                k = fast::rope(&k, c.rope_dims as i32, false, c.theta, 1.0, pos, None)?;
            }
            let window = ly.sliding.then_some(c.window as i32);
            let (ks, vs, mask) = self.kv[i].update_chunk(&k, &v, pos, window)?;
            let scale = 1.0 / (c.head_dim as f32).sqrt();
            let mask = match &mask {
                Some(m) => ScaledDotProductAttentionMask::Array(m),
                None => ScaledDotProductAttentionMask::Causal,
            };
            let o = fast::scaled_dot_product_attention(&q, &ks, &vs, scale, mask, None)?
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[1, l, qd])?;
            h = &h + qmm(&o, &ly.o)?;

            // Mixture of experts (mlx-lm SwitchGLU: sort the (token, expert) pairs by expert).
            let x = rms_norm(&h, &ly.post_norm, c.eps)?;
            let gates = x.as_dtype(Dtype::Float32)?.matmul(ly.router.as_dtype(Dtype::Float32)?.transpose()?)?;
            let probs = softmax_axis(&gates, -1, true)?;
            let ne = c.experts as i32;
            let inds = argpartition_axis(&probs, ne - k8, -1)?.index((.., .., ne - k8..));
            let w = take_along_axis(&probs, &inds, -1)?;
            let w = &w / (w.sum_axis(-1, true)? + 1e-20);
            let flat = inds.reshape(&[l * k8])?.as_dtype(Dtype::Uint32)?;
            let order = argsort_axis(&flat, 0)?;
            let inv = argsort_axis(&order, 0)?;
            let token = floor_divide(&order, Array::from_slice(&[k8 as u32], &[1]))?;
            let xs = x.reshape(&[l, 1, c.hidden as i32])?.take_axis(&token, 0)?;
            let sorted = flat.take_axis(&order, 0)?;
            let f = c.ffn as i32;
            let swiglu = |ug: &Array| -> Result<Array> {
                let up = ug.index((.., .., ..f));
                let gate = ug.index((.., .., f..));
                Ok(silu(minimum(&gate, &hi)?)? * clip(&up, (&lo, &hi))?)
            };
            let y = match (&ly.experts, &self.stream) {
                (Some(ex), _) => {
                    let (sug, bug) = grouped(&ex.up_gate)?;
                    let (sd, bd) = grouped(&ex.down)?;
                    let ug = gather_qmm(&xs, &ex.up_gate.w, &sug, &bug, None, &sorted, true, 128, 2, true)?;
                    gather_qmm(&swiglu(&ug)?, &ex.down.w, &sd, &bd, None, &sorted, true, 128, 2, true)?
                }
                (None, Some(st)) if st.all.is_some() => slab_experts(&self.kernels, st, i, &xs, &sorted, &swiglu)?,
                (None, Some(st)) => {
                    // MLX's tiled `gather_qmm` kernel when the resident call would use it.
                    let pairs = l * k8;
                    let tiled = pairs >= 16 && pairs / ne >= 4;
                    streamed_experts(&self.kernels, st, i, &xs, &sorted, &swiglu, tiled)?
                }
                (None, None) => unreachable!("a layer without resident experts has a store"),
            }
            .take_axis(&inv, 0)?
            .reshape(&[1, l, k8, c.hidden as i32])?;
            let moe = (y.as_dtype(Dtype::Float32)? * w.expand_dims(-1)?).sum_axis(-2, None)?.as_dtype(h.dtype())?;
            h = &h + moe;
            // One layer per evaluation bounds the temporary group scales to one layer's worth.
            h.eval()?;
        }
        self.pos += ids.len();
        let h = if all { h } else { h.index((.., l - 1.., ..)) };
        let x = rms_norm(&h, &self.norm, c.eps)?;
        self.head_logits(&x)
    }
}

/// Experts of layer `layer` for (token, expert) pairs sorted by expert (`xs` `[P, 1, H]`,
/// `sorted` `[P]` expert ids), streamed: the distinct experts go through the cache in groups of
/// at most one layer's slots. Returns `[P, 1, H]` in the same (sorted) order.
fn streamed_experts(
    kernels: &Kernels,
    st: &Streaming,
    layer: usize,
    xs: &Array,
    sorted: &Array,
    swiglu: &dyn Fn(&Array) -> Result<Array>,
    tiled: bool,
) -> Result<Array> {
    sorted.eval()?;
    let experts: Vec<u32> = sorted.as_slice::<u32>().to_vec();
    let cap = st.store.per_layer();
    let mut outs = Vec::new();
    let mut start = 0usize;
    while start < experts.len() {
        // The next run of pairs whose distinct experts fit the cache.
        let mut distinct: Vec<u32> = Vec::new();
        let mut end = start;
        while end < experts.len() {
            if distinct.last() != Some(&experts[end]) {
                if distinct.len() == cap {
                    break;
                }
                distinct.push(experts[end]);
            }
            end += 1;
        }
        for &e in &distinct {
            st.store.request(layer, e as usize);
        }
        let pins = distinct
            .iter()
            .map(|&e| st.store.wait(layer, e as usize))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let slots: Vec<u32> = pins.iter().map(|p| p.slot() as u32).collect();
        let g = slots.len() as i32;
        // Copy the group's records out of the slab (MLX's group layout needs its own arrays).
        let ids = Array::from_slice(&slots, &[g]);
        let recs = if st.slab.record.trit {
            kernels.unpack(&st.slab, &ids, g)?
        } else {
            st.slab.array.take_axis(ids, 0)?
        };
        let (ug, dn) = record_mats(&recs, st.slab.record, g)?;
        // Pairs of this group, with expert ids renumbered to the group (0..g).
        let local: Vec<u32> = experts[start..end]
            .iter()
            .map(|e| distinct.iter().position(|d| d == e).unwrap_or(0) as u32)
            .collect();
        let local = Array::from_slice(&local, &[(end - start) as i32]);
        let x = xs.index(start as i32..end as i32);
        let (sug, bug) = grouped(&ug)?;
        let (sd, bd) = grouped(&dn)?;
        let a = swiglu(&gather_like(&x, &ug.w, &sug, &bug, &local, tiled)?)?;
        let y = gather_like(&a, &dn.w, &sd, &bd, &local, tiled)?;
        // The kernels read the pinned slots: finish before unpinning.
        y.eval()?;
        drop(pins);
        outs.push(y);
        start = end;
    }
    Ok(concatenate(&outs, 0)?)
}

/// Experts of layer `layer` when the cache holds every expert. Preload put expert `e` in slot
/// `layer · slots_per_layer + e` and nothing is evicted, so the pairs index the layer's slab
/// rows directly: no routing readback, no pins, and for 2-bit records no copy. With all `E`
/// experts present `gather_qmm` picks the same kernel as the resident call. Returns `[P, 1, H]`
/// in the same (sorted) order.
fn slab_experts(
    kernels: &Kernels,
    st: &Streaming,
    layer: usize,
    xs: &Array,
    sorted: &Array,
    swiglu: &dyn Fn(&Array) -> Result<Array>,
) -> Result<Array> {
    let e = st.store.layout.experts as i32;
    let first = (layer * st.store.per_layer()) as i32;
    let recs = if st.slab.record.trit {
        let ids: Vec<u32> = (first as u32..(first + e) as u32).collect();
        kernels.unpack(&st.slab, &Array::from_slice(&ids, &[e]), e)?
    } else {
        st.slab.array.index(first..first + e)
    };
    let (ug, dn) = record_mats(&recs, st.slab.record, e)?;
    let (sug, bug) = grouped(&ug)?;
    let (sd, bd) = grouped(&dn)?;
    let a = swiglu(&gather_qmm(xs, &ug.w, &sug, &bug, None, sorted, true, 128, 2, true)?)?;
    Ok(gather_qmm(&a, &dn.w, &sd, &bd, None, sorted, true, 128, 2, true)?)
}

/// `gather_qmm` (sorted, one row per pair) forced onto the kernel the resident call uses:
/// MLX picks its tiled kernel iff `rows >= 16 && rows / experts >= 4`, and the tiled and
/// per-row kernels sum in different orders. A group of experts has fewer experts than the
/// resident tensor (256), so pad it: dummy rows (expert 0, dropped after) to reach the tiled
/// kernel, or zero experts to stay on the per-row one. Rows do not affect each other.
fn gather_like(x: &Array, w: &Array, s: &Array, b: &Array, idx: &Array, tiled: bool) -> Result<Array> {
    let rows = x.shape()[0];
    let experts = w.shape()[0];
    let picks_tiled = |r: i32, e: i32| r >= 16 && r / e >= 4;
    let gather = |x: &Array, w: &Array, s: &Array, b: &Array, idx: &Array| -> Result<Array> {
        Ok(gather_qmm(x, w, s, b, None, idx, true, 128, 2, true)?)
    };
    if picks_tiled(rows, experts) == tiled {
        return gather(x, w, s, b, idx);
    }
    if tiled {
        let pad = (4 * experts).max(16) - rows;
        let xs = concatenate(
            &[
                Array::zeros::<f32>(&[pad, 1, x.shape()[2]])?.as_dtype(x.dtype())?,
                x.clone(),
            ],
            0,
        )?;
        let ids = concatenate(&[Array::zeros::<u32>(&[pad])?, idx.clone()], 0)?;
        let y = gather(&xs, w, s, b, &ids)?;
        Ok(y.index(pad..))
    } else {
        let extra = rows / 4 + 1 - experts;
        let zeros = |a: &Array| -> Result<Array> {
            let mut shape = a.shape().to_vec();
            shape[0] = extra;
            Ok(concatenate(&[a.clone(), Array::zeros::<f32>(&shape)?.as_dtype(a.dtype())?], 0)?)
        };
        gather(x, &zeros(w)?, &zeros(s)?, &zeros(b)?, idx)
    }
}
