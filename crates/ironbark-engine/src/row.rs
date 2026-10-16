//! Row-scale ternary matvec kernels for one decode token.
//!
//! A ternary row is {−α, 0, +α}: MLX's affine 2-bit layout repeats (scale α, bias −α) for every
//! 128-weight group; these kernels read one α per row and use −α as the bias (2.01 instead of
//! 2.25 bits per weight in memory, and fewer bytes per token).
//!
//! ```text
//!   qmv       y[n] = Σ_k x[k] · w[n,k]            QKV, O; gathered by expert for down
//!   up_swiglu y[r] = silu(min(g,7)) · clip(u,±7)  u, g = fused up / gate rows of 8 experts
//! ```
//!
//! The expert kernels come in variants that differ only in where an expert's weights are:
//! stacked resident tensors indexed by expert id, or a slot of the streaming slab (one MLX
//! array over the whole slab) holding a [`Record`], in 2-bit codes or in trits.
//!
//! Trits: a lane's 16 weights are 3 bytes of 5 base-3 digits plus one 2-bit code in a plane
//! after the row (26 bits instead of 32, the record is 19% smaller). The lane decodes them to
//! the same codes and runs the same multiply-adds, so results are bit-identical.
//!
//! ```text
//!   row (K weights): [K/16 groups x 3 bytes: digits of weights 0..14][K/64 bytes: weight 15 of
//!                    each group, 2 bits]
//!   byte = ceil(v · 256 / 243), v = Σ c_i 3^(4-i); digit i = ((u8)(byte · 3^i) · 3) >> 8
//! ```
//!
//! Ported from DeepGrove's `maple.py` (deepgrove/maple-preview-2bit-mlx). The accumulation
//! order and bf16 rounding points follow MLX 0.32 `quantized.h` (Apple, MIT), so results are
//! bit-identical to `quantized_matmul` / `gather_qmm` on the expanded group layout.

use ironbark_mlx::ops::indexing::IndexOp;
use ironbark_mlx::{Array, Dtype};

use crate::Result;
use crate::kernel::{Kernel, Output, Template};

/// Per-lane dot of 16 packed 2-bit weights with pre-scaled activations (MLX `qdot`).
const QDOT: &str = r"
float maple_qdot(const device uchar* w, const thread float* x, float scale, float bias, float sum) {
    float accum = 0;
    for (int i=0;i<4;++i) {
        accum += (x[4*i]*(w[i]&0x03) + x[4*i+1]*(w[i]&0x0c) +
                  x[4*i+2]*(w[i]&0x30) + x[4*i+3]*(w[i]&0xc0));
    }
    return scale*accum+sum*bias;
}
";

/// `QD(w, row, g, a)`: lane dot of group `g` (16 weights) of `row`, 2-bit codes.
const QD_2BIT: &str = "#define QD(w,row,g,a) maple_qdot((w)+(row)*(K/4)+(g)*4,xt,(a),-(a),sum)\n";

/// Digits of every byte, 2 bits each (digit `n` at bits `2n`): `((u8)(b · 3^n) · 3) >> 8`, the
/// decoding `pack` is tested against, precomputed for all 256 bytes (`maple_trit_lut`).
fn trit_lut() -> String {
    let entries: Vec<String> = (0..=255u8)
        .map(|b| {
            let v = (0..5u32).fold(0u16, |v, n| {
                let q = b.wrapping_mul(3u8.pow(n));
                v | ((((u16::from(q) * 3) >> 8) & 3) << (2 * n))
            });
            v.to_string()
        })
        .collect();
    format!("constant ushort maple_trit_lut[256] = {{{}}};\n", entries.join(","))
}

/// `QD` for trit rows of `ROWB` bytes: decode to codes, then [`QDOT`]'s exact arithmetic.
const QD_TRIT: &str = r"
inline void maple_trits(const device uchar* t, uint extra, thread uchar* c) {
    for (int b=0;b<3;++b) {
        ushort v=maple_trit_lut[t[b]];
        for (int n=0;n<5;++n) c[5*b+n]=(v>>(2*n))&3;
    }
    c[15]=extra;
}
float maple_qdot3(const device uchar* t, uint extra, const thread float* x, float scale, float bias, float sum) {
    uchar c[16];
    maple_trits(t, extra, c);
    float accum = 0;
    for (int i=0;i<4;++i) {
        accum += (x[4*i]*c[4*i] + x[4*i+1]*(c[4*i+1]<<2) +
                  x[4*i+2]*(c[4*i+2]<<4) + x[4*i+3]*(c[4*i+3]<<6));
    }
    return scale*accum+sum*bias;
}
#define QD_EXTRA(w,row,g) (((w)[(row)*ROWB+(K/16)*3+(g)/4]>>(2*((g)%4)))&3)
#define QD(w,row,g,a) maple_qdot3((w)+(row)*ROWB+(g)*3,QD_EXTRA(w,row,g),xt,(a),-(a),sum)
";

/// Trit records of slots `ids` back to 2-bit records (codes, then the α words as they are),
/// for MLX's batched kernels in prefill. One thread per output word.
const UNPACK: &str = r"
    uint o=thread_position_in_grid.x, s=thread_position_in_grid.y;
    if (o>=W2) return;
    const device uchar* rec=(const device uchar*)slab+(ulong)ids[s]*REC;
    constexpr uint UGW=2*F*H/16, DW=H*F/16;
    uint v=0;
    if (o<UGW+DW) {
        bool up=o<UGW;
        uint K=up?H:F, ROWB=K/16*3+K/64, i=up?o:o-UGW;
        uint row=i/(K/16), g=i%(K/16);
        const device uchar* w=rec+(up?0:2*F*(H/16*3+H/64));
        uchar c[16];
        maple_trits(w+row*ROWB+g*3, QD_EXTRA(w,row,g), c);
        for (int j=0;j<16;++j) v|=uint(c[j])<<(2*j);
    } else {
        v=((const device uint*)(rec+AOFF))[o-UGW-DW];
    }
    out[(ulong)s*W2+o]=v;
";

/// Two SIMD groups per threadgroup, 4 rows each; a lane covers 16 inputs per 512-wide step.
/// `W_SETUP` defines `w` (codes of matrix `expert`) and `sc` (its row scales).
const QMV: &str = r"
    uint lid=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
    uint slot=threadgroup_position_in_grid.y;
    uint expert=GATHER?ids[slot]:0;
    uint row0=threadgroup_position_in_grid.x*8+sg*4;
    W_SETUP
    const device bfloat16_t* xv=x+(SPLIT?slot*K:0);
    float result[4]={0};
    for(int k=0;k<K;k+=512) {
        const device bfloat16_t* xx=xv+k+lid*16;
        float xt[16],sum=0;
        for(int i=0;i<16;i+=4) {
            sum+=xx[i]+xx[i+1]+xx[i+2]+xx[i+3];
            xt[i]=xx[i];xt[i+1]=xx[i+1]/4.0f;
            xt[i+2]=xx[i+2]/16.0f;xt[i+3]=xx[i+3]/64.0f;
        }
        for(int r=0;r<4;++r) {
            uint row=row0+r;
            float a=(float)sc[row];
            result[r]+=QD(w,row,k/16+lid,a);
        }
    }
    for(int r=0;r<4;++r) {
        result[r]=simd_sum(result[r]);
        if(lid==0)out[slot*N+row0+r]=(bfloat16_t)result[r];
    }
";

/// Fused up + gate (rows 0..F up, F..2F gate per expert) and clamped SwiGLU, for K = 2048.
const UP_SWIGLU: &str = r"
    constexpr int R = 4, SGS = 2, K = 2048;
    uint lid=thread_index_in_simdgroup;
    uint sg=simdgroup_index_in_threadgroup;
    uint slot=threadgroup_position_in_grid.y;
    uint expert=ids[slot];
    uint first=threadgroup_position_in_grid.x*(SGS*R)+sg*R;
    W_SETUP
    float up[R]={0},gate[R]={0};
    for(int k=0;k<K;k+=512) {
        const device bfloat16_t* xx=x+k+lid*16;
        float xt[16];
        float sum=0;
        for(int i=0;i<16;i+=4) {
            sum+=xx[i]+xx[i+1]+xx[i+2]+xx[i+3];
            xt[i]=xx[i];xt[i+1]=xx[i+1]/4.0f;
            xt[i+2]=xx[i+2]/16.0f;xt[i+3]=xx[i+3]/64.0f;
        }
        for(int r=0;r<R;++r) {
            uint row=first+r;
            uint gi=k/16+lid;
            float au=(float)sc[row], ag=(float)sc[row+F];
            up[r]+=QD(w,row,gi,au);
            gate[r]+=QD(w,row+F,gi,ag);
        }
    }
    for(int r=0;r<R;++r) {
        up[r]=simd_sum(up[r]);gate[r]=simd_sum(gate[r]);
        if(lid==0) {
            bfloat16_t u=(bfloat16_t)up[r],g=(bfloat16_t)gate[r];
            g=g<(bfloat16_t)7.0f?g:(bfloat16_t)7.0f;
            u=u>(bfloat16_t)(-7.0f)?u:(bfloat16_t)(-7.0f);
            u=u<(bfloat16_t)7.0f?u:(bfloat16_t)7.0f;
            auto t=1/(1+metal::exp(metal::abs(g)));
            bfloat16_t sig=(g<0)?t:1-t;
            bfloat16_t silu=g*sig;
            out[slot*F+first+r]=silu*u;
        }
    }
";

const QMV_RESIDENT: &str = "const device uchar* w=(const device uchar*)weight+(ulong)expert*N*(K/4);
    const device bfloat16_t* sc=scales+(ulong)expert*N;";
const UP_RESIDENT: &str = "const device uchar* w=(const device uchar*)weight+(ulong)expert*(2*F)*(K/4);
    const device bfloat16_t* sc=scales+(ulong)expert*(2*F);";
/// Slab variants: `expert` is a slab slot; the record's codes start at `WOFF`, scales at `AOFF`.
const IN_SLAB: &str = "const device uchar* rec=(const device uchar*)slab+(ulong)expert*REC;
    const device uchar* w=rec+WOFF;
    const device bfloat16_t* sc=(const device bfloat16_t*)(rec+AOFF);";

/// Byte layout of one expert record in `experts.bin` and in a slab slot.
///
/// ```text
///   [up+gate rows: 2F x row(H)][down rows: H x row(F)][up+gate α: 2F bf16][down α: H bf16]
///   row(K) = K/4 bytes (2-bit codes), or K/16·3 + K/64 (trits)
/// ```
#[derive(Debug, Clone, Copy)]
pub(crate) struct Record {
    /// Hidden size.
    pub(crate) hidden: usize,
    /// Expert FFN size.
    pub(crate) ffn: usize,
    /// Rows in trits instead of 2-bit codes.
    pub(crate) trit: bool,
}

impl Record {
    /// Bytes of a row of `k` weights.
    #[must_use]
    pub(crate) fn row_bytes(&self, k: usize) -> usize {
        if self.trit { k / 16 * 3 + k / 64 } else { k / 4 }
    }

    /// Offset of the down codes.
    #[must_use]
    pub(crate) fn down_codes(&self) -> usize {
        2 * self.ffn * self.row_bytes(self.hidden)
    }

    /// Offset of the up+gate row scales.
    #[must_use]
    pub(crate) fn up_gate_alpha(&self) -> usize {
        self.down_codes() + self.hidden * self.row_bytes(self.ffn)
    }

    /// Offset of the down row scales.
    #[must_use]
    pub(crate) fn down_alpha(&self) -> usize {
        self.up_gate_alpha() + 2 * self.ffn * 2
    }

    /// Record size (without page padding).
    #[must_use]
    pub(crate) fn bytes(&self) -> usize {
        self.down_alpha() + self.hidden * 2
    }
}

/// Ternary matrix (or stack of expert matrices) with one scale per row.
pub(crate) struct RowMat {
    /// Packed codes, `[.., n, k/16]` u32.
    pub(crate) w: Array,
    /// α per row, `[.., n]` bf16.
    pub(crate) scales: Array,
    /// Output rows (per expert).
    pub(crate) n: i32,
    /// Input columns.
    pub(crate) k: i32,
}

/// The streaming slab as one MLX array, with the layout of its records.
pub(crate) struct Slab {
    /// `[slots, stride / 4]` u32 over the slab memory (no copy).
    pub(crate) array: Array,
    /// Bytes between slots.
    pub(crate) stride: i32,
    pub(crate) record: Record,
}

/// Up+gate and down matrices of `g` records `[g, words]` (u32, 2-bit layout): views of the
/// codes, and the bf16 row scales α.
pub(crate) fn record_mats(recs: &Array, record: Record, g: i32) -> Result<(RowMat, RowMat)> {
    // Offsets in 2-bit records: trit records are unpacked to that layout for MLX's kernels.
    let rec = Record { trit: false, ..record };
    let (h, f) = (rec.hidden as i32, rec.ffn as i32);
    // Record words: up+gate codes, down codes, then α (bf16 pairs per word).
    let (ug_words, d_words) = (2 * f * h / 16, h * f / 16);
    let alpha_at = (rec.up_gate_alpha() / 4) as i32;
    let alpha_words = (2 * f + h) * 2 / 4; // bf16 scales, in u32 words
    let ug_w = recs.index((.., ..ug_words)).reshape(&[g, 2 * f, h / 16])?;
    let d_w = recs.index((.., ug_words..ug_words + d_words)).reshape(&[g, h, f / 16])?;
    let alpha = recs.index((.., alpha_at..alpha_at + alpha_words)).contiguous()?.view::<half::bf16>()?;
    let ug = RowMat {
        w: ug_w,
        scales: alpha.index((.., ..2 * f)),
        n: 2 * f,
        k: h,
    };
    let dn = RowMat {
        w: d_w,
        scales: alpha.index((.., 2 * f..)),
        n: h,
        k: f,
    };
    Ok((ug, dn))
}

/// The compiled row-scale kernels.
#[derive(Debug)]
pub(crate) struct Kernels {
    qmv: Kernel,
    up: Kernel,
    qmv_slab: Kernel,
    up_slab: Kernel,
    qmv_trit: Kernel,
    up_trit: Kernel,
    unpack: Kernel,
    zero: Array,
}

impl Kernels {
    pub(crate) fn new() -> Self {
        let resident = ["x", "weight", "scales", "ids"];
        let slab = ["x", "slab", "ids"];
        let src = |body: &str, setup: &str| body.replace("W_SETUP", setup);
        let (two, trit) = (format!("{QDOT}{QD_2BIT}"), format!("{QDOT}{}{QD_TRIT}", trit_lut()));
        Kernels {
            qmv: Kernel::new("ironbark_row_qmv", &resident, &["out"], &src(QMV, QMV_RESIDENT), &two),
            up: Kernel::new("ironbark_row_up_swiglu", &resident, &["out"], &src(UP_SWIGLU, UP_RESIDENT), &two),
            qmv_slab: Kernel::new("ironbark_row_qmv_slab", &slab, &["out"], &src(QMV, IN_SLAB), &two),
            up_slab: Kernel::new("ironbark_row_up_swiglu_slab", &slab, &["out"], &src(UP_SWIGLU, IN_SLAB), &two),
            qmv_trit: Kernel::new("ironbark_row_qmv_trit", &slab, &["out"], &src(QMV, IN_SLAB), &trit),
            up_trit: Kernel::new("ironbark_row_up_swiglu_trit", &slab, &["out"], &src(UP_SWIGLU, IN_SLAB), &trit),
            unpack: Kernel::new("ironbark_unpack_trit", &["slab", "ids"], &["out"], UNPACK, &trit),
            zero: Array::from_slice(&[0u32], &[1]),
        }
    }

    /// `x · Wᵀ` for one token `x` `[1, 1, k]` -> `[1, 1, n]`.
    pub(crate) fn qmv(&self, x: &Array, m: &RowMat) -> Result<Array> {
        let t = [
            Template::Int("K", m.k),
            Template::Int("N", m.n),
            Template::Bool("GATHER", false),
            Template::Bool("SPLIT", false),
        ];
        run(&self.qmv, &[x, &m.w, &m.scales, &self.zero], &t, (m.n / 4 * 32, 1, 1), &[1, 1, m.n])
    }

    /// Experts `ids` applied to their own rows of `x` `[slots, 1, k]` -> `[slots, 1, n]`.
    pub(crate) fn qmv_gather(&self, x: &Array, m: &RowMat, ids: &Array, slots: i32) -> Result<Array> {
        let t = [
            Template::Int("K", m.k),
            Template::Int("N", m.n),
            Template::Bool("GATHER", true),
            Template::Bool("SPLIT", true),
        ];
        run(&self.qmv, &[x, &m.w, &m.scales, ids], &t, (m.n / 4 * 32, slots, 1), &[slots, 1, m.n])
    }

    /// Fused up/gate + clamped SwiGLU of experts `ids` on one token `x` `[1, 1, 2048]`
    /// -> `[slots, 1, ffn]`. `m.n` is `2 · ffn`.
    pub(crate) fn up_swiglu(&self, x: &Array, m: &RowMat, ids: &Array, slots: i32) -> Result<Array> {
        let f = m.n / 2;
        run(
            &self.up,
            &[x, &m.w, &m.scales, ids],
            &[Template::Int("F", f)],
            (f / 8 * 64, slots, 1),
            &[slots, 1, f],
        )
    }

    /// [`Kernels::up_swiglu`] for experts held in slab slots `slot_ids`.
    pub(crate) fn up_swiglu_slab(&self, x: &Array, s: &Slab, slot_ids: &Array, slots: i32) -> Result<Array> {
        let f = s.record.ffn as i32;
        let mut t = vec![
            Template::Int("F", f),
            Template::Int("REC", s.stride),
            Template::Int("WOFF", 0),
            Template::Int("AOFF", s.record.up_gate_alpha() as i32),
        ];
        let k = if s.record.trit {
            t.push(Template::Int("ROWB", s.record.row_bytes(s.record.hidden) as i32));
            &self.up_trit
        } else {
            &self.up_slab
        };
        run(k, &[x, &s.array, slot_ids], &t, (f / 8 * 64, slots, 1), &[slots, 1, f])
    }

    /// [`Kernels::qmv_gather`] (down projection) for experts held in slab slots `slot_ids`.
    pub(crate) fn qmv_gather_slab(&self, x: &Array, s: &Slab, slot_ids: &Array, slots: i32) -> Result<Array> {
        let (k, n) = (s.record.ffn as i32, s.record.hidden as i32);
        let mut t = vec![
            Template::Int("K", k),
            Template::Int("N", n),
            Template::Bool("GATHER", true),
            Template::Bool("SPLIT", true),
            Template::Int("REC", s.stride),
            Template::Int("WOFF", s.record.down_codes() as i32),
            Template::Int("AOFF", s.record.down_alpha() as i32),
        ];
        let kern = if s.record.trit {
            t.push(Template::Int("ROWB", s.record.row_bytes(s.record.ffn) as i32));
            &self.qmv_trit
        } else {
            &self.qmv_slab
        };
        run(kern, &[x, &s.array, slot_ids], &t, (n / 4 * 32, slots, 1), &[slots, 1, n])
    }

    /// Trit records in slab slots `slot_ids` as 2-bit records `[slots, words]` (u32).
    pub(crate) fn unpack(&self, s: &Slab, slot_ids: &Array, slots: i32) -> Result<Array> {
        let r = s.record;
        let two = Record { trit: false, ..r };
        let words = (two.bytes() / 4) as i32;
        let t = [
            Template::Int("H", r.hidden as i32),
            Template::Int("F", r.ffn as i32),
            Template::Int("REC", s.stride),
            Template::Int("AOFF", r.up_gate_alpha() as i32),
            Template::Int("W2", words),
        ];
        let mut out = self.unpack.apply(
            &[&s.array, slot_ids],
            &t,
            ((words + 63) / 64 * 64, slots, 1),
            (64, 1, 1),
            &[Output {
                shape: &[slots, words],
                dtype: Dtype::Uint32,
            }],
        )?;
        Ok(out.remove(0))
    }
}

/// Launch `k` with 64-thread groups and one bf16 output.
fn run(k: &Kernel, inputs: &[&Array], t: &[Template], grid: (i32, i32, i32), shape: &[i32]) -> Result<Array> {
    let mut out = k.apply(
        inputs,
        t,
        grid,
        (64, 1, 1),
        &[Output {
            shape,
            dtype: Dtype::Bfloat16,
        }],
    )?;
    Ok(out.remove(0))
}

#[cfg(test)]
mod tests {
    use ironbark_mlx::ops::indexing::IndexOp;
    use ironbark_mlx::ops::{broadcast_to, gather_qmm, minimum, quantized_matmul, silu};
    use ironbark_mlx::{Array, Dtype};

    use super::{Kernels, Record, RowMat, Slab};
    use crate::pack::rows_to_trits;

    /// Random ternary matrices: `e` experts of `n x k` codes in {0, 1, 2} (w + 1) and row
    /// scales, plus MLX's group layout of the same weights (scales and biases per 128).
    fn ternary(e: i32, n: i32, k: i32, seed: u64) -> (RowMat, Array, Array, Vec<u32>, Vec<u16>) {
        let mut s = seed;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let words = (e * n * k / 16) as usize;
        let w: Vec<u32> =
            (0..words).map(|_| (0..16).fold(0u32, |acc, i| acc | (((next() % 3) as u32) << (2 * i)))).collect();
        let alpha: Vec<f32> = (0..(e * n) as usize).map(|_| 0.01 + (next() % 50) as f32 * 1e-3).collect();
        let wa = Array::from_slice(&w, &[e, n, k / 16]);
        let scales = Array::from_slice(&alpha, &[e, n]).as_dtype(Dtype::Bfloat16).unwrap();
        // bf16 bits of the very same (rounded) scales, for writing records.
        let alpha_bits = bits(&scales);
        let gs = broadcast_to(scales.expand_dims(-1).unwrap(), &[e, n, k / 128]).unwrap();
        let gb = gs.negative().unwrap();
        let m = RowMat { w: wa, scales, n, k };
        (m, gs, gb, w, alpha_bits)
    }

    fn input(k: i32, rows: i32) -> Array {
        let v: Vec<f32> = (0..rows * k).map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.02).collect();
        Array::from_slice(&v, &[rows, 1, k]).as_dtype(Dtype::Bfloat16).unwrap()
    }

    fn bits(a: &Array) -> Vec<u16> {
        a.as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .iter()
            .map(|v| (v.to_bits() >> 16) as u16)
            .collect()
    }

    #[test]
    fn qmv_matches_quantized_matmul_bit_for_bit() {
        let k = Kernels::new();
        let (m, gs, gb, ..) = ternary(1, 64, 2048, 1);
        let x = input(2048, 1).reshape(&[1, 1, 2048]).unwrap();
        let got = k.qmv(&x, &m).unwrap();
        let w2 = m.w.reshape(&[64, 128]).unwrap();
        let want =
            quantized_matmul(&x, &w2, gs.reshape(&[64, 16]).unwrap(), &gb.reshape(&[64, 16]).unwrap(), true, 128, 2)
                .unwrap();
        assert_eq!(bits(&got), bits(&want));
    }

    #[test]
    fn gathered_qmv_matches_gather_qmm() {
        let k = Kernels::new();
        let (m, gs, gb, ..) = ternary(4, 32, 512, 2);
        let ids = Array::from_slice(&[3u32, 0, 2], &[3]);
        let x = input(512, 3);
        let got = k.qmv_gather(&x, &m, &ids, 3).unwrap();
        let ids2 = ids.reshape(&[3, 1]).unwrap();
        let want =
            gather_qmm(x.reshape(&[3, 1, 1, 512]).unwrap(), &m.w, &gs, &gb, None, &ids2, true, 128, 2, false).unwrap();
        assert_eq!(bits(&got), bits(&want));
    }

    #[test]
    fn up_swiglu_matches_reference_composition() {
        let k = Kernels::new();
        let (m, gs, gb, ..) = ternary(4, 16, 2048, 3);
        let ids = Array::from_slice(&[1u32, 3], &[2]);
        let x = input(2048, 1).reshape(&[1, 1, 2048]).unwrap();
        let got = k.up_swiglu(&x, &m, &ids, 2).unwrap();
        // Reference: gather_qmm on the group layout, split up | gate, clamped SwiGLU in bf16.
        let ids2 = ids.reshape(&[1, 2]).unwrap();
        let ug =
            gather_qmm(x.reshape(&[1, 1, 1, 2048]).unwrap(), &m.w, &gs, &gb, None, &ids2, true, 128, 2, false).unwrap();
        let up = ug.index((.., .., .., ..8));
        let gate = ug.index((.., .., .., 8..));
        let seven = Array::from_f32(7.0).as_dtype(Dtype::Bfloat16).unwrap();
        let neg = seven.negative().unwrap();
        let act =
            silu(minimum(&gate, &seven).unwrap()).unwrap() * ironbark_mlx::ops::clip(&up, (&neg, &seven)).unwrap();
        assert_eq!(bits(&got), bits(&act));
    }

    #[test]
    fn slab_variants_match_resident_kernels() {
        slab_matches_resident(false);
    }

    #[test]
    fn trit_slab_matches_resident_kernels() {
        slab_matches_resident(true);
    }

    #[test]
    fn unpack_restores_2bit_records() {
        let (h, f) = (2048usize, 512usize);
        let (two, trit) = (records(false, 6), records(true, 6));
        let k = Kernels::new();
        let slots = Array::from_slice(&[1u32, 2], &[2]);
        let got = k.unpack(&trit.0, &slots, 2).unwrap();
        let want = two.0.array.take_axis(&slots, 0).unwrap();
        let words = (Record {
            hidden: h,
            ffn: f,
            trit: false,
        }
        .bytes()
            / 4) as i32;
        let want = want.index((.., ..words)).contiguous().unwrap();
        ironbark_mlx::transforms::eval([&got, &want]).unwrap();
        assert_eq!(got.as_slice::<u32>(), want.as_slice::<u32>());
    }

    /// A slab of 3 experts in slots 2, 0, 1 (Maple shapes), and the same experts resident.
    fn records(trit: bool, seed: u64) -> (Slab, RowMat, RowMat) {
        let (h, f, e) = (2048i32, 512i32, 3i32);
        let (ug, _, _, ug_w, ug_a) = ternary(e, 2 * f, h, seed);
        let (dn, _, _, dn_w, dn_a) = ternary(e, h, f, seed + 1);
        let rec = Record {
            hidden: h as usize,
            ffn: f as usize,
            trit,
        };
        // Records in slots 2, 0, 1 (slot order differs from expert order), 16 KiB pages.
        let stride = rec.bytes().div_ceil(16384) * 16384;
        let slot_of = [2usize, 0, 1];
        let mut slab = vec![0u8; 3 * stride];
        let (ugw, dnw) = (ug_w.len() / e as usize, dn_w.len() / e as usize);
        let (uga, dna) = (ug_a.len() / e as usize, dn_a.len() / e as usize);
        for x in 0..e as usize {
            let r = &mut slab[slot_of[x] * stride..];
            let put32 = |r: &mut [u8], at: usize, v: &[u32]| {
                v.iter()
                    .enumerate()
                    .for_each(|(i, w)| r[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes()));
            };
            let put16 = |r: &mut [u8], at: usize, v: &[u16]| {
                v.iter()
                    .enumerate()
                    .for_each(|(i, w)| r[at + 2 * i..at + 2 * i + 2].copy_from_slice(&w.to_le_bytes()));
            };
            if trit {
                rows_to_trits(&ug_w[x * ugw..(x + 1) * ugw], h as usize, r);
                rows_to_trits(&dn_w[x * dnw..(x + 1) * dnw], f as usize, &mut r[rec.down_codes()..]);
            } else {
                put32(r, 0, &ug_w[x * ugw..(x + 1) * ugw]);
                put32(r, rec.down_codes(), &dn_w[x * dnw..(x + 1) * dnw]);
            }
            put16(r, rec.up_gate_alpha(), &ug_a[x * uga..(x + 1) * uga]);
            put16(r, rec.down_alpha(), &dn_a[x * dna..(x + 1) * dna]);
        }
        let words: Vec<u32> = slab.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let s = Slab {
            array: Array::from_slice(&words, &[3, (stride / 4) as i32]),
            stride: stride as i32,
            record: rec,
        };
        (s, ug, dn)
    }

    fn slab_matches_resident(trit: bool) {
        let h = 2048;
        let k = Kernels::new();
        let (s, ug, dn) = records(trit, 4);
        let slot_of = [2usize, 0, 1];
        let experts = Array::from_slice(&[2u32, 0], &[2]);
        let slots = Array::from_slice(&[slot_of[2] as u32, slot_of[0] as u32], &[2]);
        let x = input(h, 1).reshape(&[1, 1, h]).unwrap();
        let want = k.up_swiglu(&x, &ug, &experts, 2).unwrap();
        let got = k.up_swiglu_slab(&x, &s, &slots, 2).unwrap();
        assert_eq!(bits(&got), bits(&want));
        let want = k.qmv_gather(&want, &dn, &experts, 2).unwrap();
        let got = k.qmv_gather_slab(&got, &s, &slots, 2).unwrap();
        assert_eq!(bits(&got), bits(&want));
    }
}
