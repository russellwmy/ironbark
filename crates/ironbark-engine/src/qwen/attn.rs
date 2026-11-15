//! Decode attention (one query token) over the full-attention layers' key/value caches, with
//! the query heads of a key/value head sharing each key and value read.
//!
//! ```text
//!   pass 1: one threadgroup per (block of keys, kv head), 8 simdgroups split the block;
//!           each lane holds D/32 dims of all G query heads; online softmax (exp2)
//!           ─► per block: max m, sum l, unnormalized o  [kv_heads, blocks, G, D]
//!   pass 2: per query head and dim: rescale the blocks to the global max, o = Σ o_b / Σ l_b
//! ```
//!
//! MLX's own decode kernel shares reads only for head dims 64 and 128; for Qwen3.6's 256 each
//! query head reads every key itself, at about half the memory bandwidth.

use ironbark_mlx::{Array, Dtype};

use crate::Result;
use crate::kernel::{Kernel, Output};

/// Query heads per simdgroup in pass 1 (fewer: more simdgroups read each key; more: registers
/// spill). 2 measured best on an M1 Max; 1 and 4 were slower.
const HEADS_PER_SIMD: usize = 2;
/// Simdgroups per threadgroup in pass 1.
const SIMDS: i32 = 8;
/// Keys per block, at least (fewer blocks for a short cache).
const MIN_KEYS: i32 = 256;
/// Most blocks per kv head.
const MAX_BLOCKS: i32 = 128;

const PASS1: &str = r"
    // NG groups of HG heads; each group's KS simdgroups take every KS-th key of the block.
    constexpr uint NG=G/HG, KS=SIMDS/NG;
    uint blk=threadgroup_position_in_grid.x, h=threadgroup_position_in_grid.y;
    uint nb=threadgroups_per_grid.x;
    uint sg=simdgroup_index_in_threadgroup, lane=thread_index_in_simdgroup;
    uint hg=sg%NG, ks=sg/NG;
    int NM=n[0], N=n[0]+n[1], cap=k_shape[2], tcap=kt_shape[2];
    int bk=(N+int(nb)-1)/int(nb);
    int start=int(blk)*bk, end=min(start+bk, N);
    float qv[HG][PL], m[HG], l[HG], acc[HG][PL];
    for (uint g=0;g<HG;++g) {
        for (uint i=0;i<PL;++i) { qv[g][i]=float(q[(h*G+hg*HG+g)*D+lane*PL+i])*SCALE; acc[g][i]=0; }
        m[g]=-INFINITY; l[g]=0;
    }
    // A lane's 8 dims of a key are one 16-byte load (D = 256).
    // Keys below NM from the main cache, the rest from the tail.
    const device uint4* kb=(const device uint4*)(k+size_t(h)*size_t(cap)*D)+lane;
    const device uint4* vb=(const device uint4*)(v+size_t(h)*size_t(cap)*D)+lane;
    const device uint4* ktb=(const device uint4*)(kt+size_t(h)*size_t(tcap)*D)+lane;
    const device uint4* vtb=(const device uint4*)(vt+size_t(h)*size_t(tcap)*D)+lane;
    auto kat=[&](int j) { return j<NM ? kb[size_t(j)*32] : ktb[size_t(j-NM)*32]; };
    auto vat=[&](int j) { return j<NM ? vb[size_t(j)*32] : vtb[size_t(j-NM)*32]; };
    auto unpack=[](uint4 u, thread float* f) {
        for (uint i=0;i<4;++i) { f[2*i]=as_type<float>(u[i]<<16); f[2*i+1]=as_type<float>(u[i]&0xffff0000u); }
    };
    auto step=[&](float kk[PL], float vv[PL]) {
        for (uint g=0;g<HG;++g) {
            float t=0;
            for (uint i=0;i<PL;++i) t+=qv[g][i]*kk[i];
            float s=simd_sum(t);
            float mn=max(m[g], s);
            float a=exp2(m[g]-mn), p=exp2(s-mn);
            l[g]=l[g]*a+p;
            for (uint i=0;i<PL;++i) acc[g][i]=acc[g][i]*a+p*vv[i];
            m[g]=mn;
        }
    };
    int j=start+int(ks);
    for (; j+int(KS)<end; j+=2*KS) {
        uint4 k0=kat(j), k1=kat(j+KS), v0=vat(j), v1=vat(j+KS);
        float kk[PL], vv[PL];
        unpack(k0, kk); unpack(v0, vv); step(kk, vv);
        unpack(k1, kk); unpack(v1, vv); step(kk, vv);
    }
    if (j<end) {
        float kk[PL], vv[PL];
        unpack(kat(j), kk); unpack(vat(j), vv); step(kk, vv);
    }
    threadgroup float tm[KS*G], tl[KS*G], tacc[KS*D];
    if (lane==0) for (uint g=0;g<HG;++g) { tm[ks*G+hg*HG+g]=m[g]; tl[ks*G+hg*HG+g]=l[g]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint t=sg*32+lane;
    size_t base=(size_t(h)*nb+blk)*G;
    for (uint g=0;g<G;++g) {
        float M=-INFINITY;
        for (uint s=0;s<KS;++s) M=max(M, tm[s*G+g]);
        if (g/HG==hg) {
            float mg=m[g%HG];
            float f=(mg==-INFINITY)?0.0f:exp2(mg-M);
            for (uint i=0;i<PL;++i) tacc[ks*D+lane*PL+i]=acc[g%HG][i]*f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint d=t; d<D; d+=SIMDS*32) {
            float o=0;
            for (uint s=0;s<KS;++s) o+=tacc[s*D+d];
            po[(base+g)*D+d]=o;
        }
        if (t==0) {
            float L=0;
            for (uint s=0;s<KS;++s) { float ms=tm[s*G+g]; L+=(ms==-INFINITY)?0.0f:tl[s*G+g]*exp2(ms-M); }
            pm[base+g]=M;
            pl[base+g]=L;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
";

const PASS2: &str = r"
    uint d=thread_position_in_grid.x, qh=thread_position_in_grid.y;
    uint h=qh/G, g=qh%G;
    uint nb=pm_shape[1];
    float M=-INFINITY;
    for (uint b=0;b<nb;++b) M=max(M, pm[(h*nb+b)*G+g]);
    float L=0, o=0;
    for (uint b=0;b<nb;++b) {
        size_t i=(size_t(h)*nb+b)*G+g;
        float mb=pm[i];
        float f=(mb==-INFINITY)?0.0f:exp2(mb-M);
        L+=pl[i]*f;
        o+=po[i*D+d]*f;
    }
    out[qh*D+d]=bfloat16_t(o/L);
";

/// The two kernels, sizes compiled in.
pub(crate) struct DecodeAttention {
    pass1: Kernel,
    pass2: Kernel,
    heads: i32,
    kv_heads: i32,
    dim: i32,
}

impl DecodeAttention {
    /// Whether the kernels work for this shape: 1, 2, 4 or 8 query heads per kv head and a head
    /// dim of 256 (a lane's 8 dims are one 16-byte load).
    pub(crate) fn fits(heads: usize, kv_heads: usize, dim: usize) -> bool {
        kv_heads > 0 && heads.is_multiple_of(kv_heads) && [1, 2, 4, 8].contains(&(heads / kv_heads)) && dim == 256
    }

    pub(crate) fn new(heads: usize, kv_heads: usize, dim: usize) -> Self {
        let g = heads / kv_heads;
        let hg = HEADS_PER_SIMD.min(g);
        let scale = (dim as f32).powf(-0.5) * std::f32::consts::LOG2_E;
        let header = format!(
            "constant constexpr uint G={g};constant constexpr uint D={dim};constant constexpr uint PL={};\
             constant constexpr uint SIMDS={SIMDS};constant constexpr uint HG={hg};constant constexpr float SCALE={scale:.10e}f;\n",
            dim / 32
        );
        Self {
            pass1: Kernel::new(
                "ironbark_decode_attn_1",
                &["q", "k", "v", "kt", "vt", "n"],
                &["po", "pm", "pl"],
                PASS1,
                &header,
            ),
            pass2: Kernel::new("ironbark_decode_attn_2", &["po", "pm", "pl"], &["out"], PASS2, &header),
            heads: heads as i32,
            kv_heads: kv_heads as i32,
            dim: dim as i32,
        }
    }

    /// Attention of `q` `[1, heads, 1, dim]` (bf16) over the first `nm` keys and values of the
    /// caches `k`, `v` and then the first `nt` of the tail `kt`, `vt` (all `[1, kv_heads,
    /// capacity, dim]`, bf16, row-contiguous): `[1, heads, 1, dim]`.
    #[expect(clippy::too_many_arguments, reason = "the query, two caches and their lengths")]
    pub(crate) fn apply(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        nm: i32,
        kt: &Array,
        vt: &Array,
        nt: i32,
    ) -> Result<Array> {
        let n = nm + nt;
        let (h, kvh, d) = (self.heads, self.kv_heads, self.dim);
        let g = h / kvh;
        let blocks = (n / MIN_KEYS).clamp(1, MAX_BLOCKS);
        let count = Array::from_slice(&[nm, nt], &[2]);
        let (po_shape, pm_shape) = ([kvh, blocks, g, d], [kvh, blocks, g]);
        let out = |shape, dtype| Output { shape, dtype };
        let p = self.pass1.apply(
            &[q, k, v, kt, vt, &count],
            &[],
            (blocks * SIMDS * 32, kvh, 1),
            (SIMDS * 32, 1, 1),
            &[
                out(&po_shape, Dtype::Float32),
                out(&pm_shape, Dtype::Float32),
                out(&pm_shape, Dtype::Float32),
            ],
        )?;
        let o_shape = [1, h, 1, d];
        let mut o =
            self.pass2
                .apply(&[&p[0], &p[1], &p[2]], &[], (d, h, 1), (d, 1, 1), &[out(&o_shape, Dtype::Bfloat16)])?;
        Ok(o.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ironbark_mlx::fast;
    use ironbark_mlx::ops::indexing::IndexOp;
    use ironbark_mlx::random::normal;
    use ironbark_mlx::transforms::eval;

    use super::*;

    fn rnd(s: &[i32]) -> Array {
        normal::<f32>(s, None, None, None).unwrap().as_dtype(Dtype::Bfloat16).unwrap()
    }

    /// Same output as MLX's attention, for caches with spare capacity and short to long n.
    #[test]
    fn decode_attention_matches_sdpa() {
        let a = DecodeAttention::new(16, 2, 256);
        for n in [1, 7, 255, 300, 4096, 40_000] {
            let cap = (n + 256 + 255) / 256 * 256;
            let (q, k, v) = (rnd(&[1, 16, 1, 256]), rnd(&[1, 2, cap, 256]), rnd(&[1, 2, cap, 256]));
            // The last keys (up to 100) from a tail, the rest from the main cache.
            let nt = n.min(100);
            let nm = n - nt;
            let (kt, vt) = (
                k.index((.., .., nm..)).add(Array::from_f32(0.0)).unwrap().as_dtype(Dtype::Bfloat16).unwrap(),
                v.index((.., .., nm..)).add(Array::from_f32(0.0)).unwrap().as_dtype(Dtype::Bfloat16).unwrap(),
            );
            let got = a.apply(&q, &k, &v, nm, &kt, &vt, nt).unwrap().as_dtype(Dtype::Float32).unwrap();
            let (ks, vs) = (k.index((.., .., ..n)), v.index((.., .., ..n)));
            let want = fast::scaled_dot_product_attention(&q, &ks, &vs, 1.0 / 16.0, None, None)
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap();
            let err = got.subtract(&want).unwrap().abs().unwrap().max(None).unwrap();
            eval([&err]).unwrap();
            let e = err.item_exact::<f32>();
            assert!(e < 2e-2, "n {n}: max error {e}");
        }
    }

    /// GPU time of 10 layers' decode attention at long contexts, this kernel against MLX's.
    #[test]
    #[ignore = "timing"]
    fn decode_attention_timing() {
        let a = DecodeAttention::new(16, 2, 256);
        for n in [1400, 8000, 36_800, 73_400] {
            let cap = (n + 255) / 256 * 256;
            let caches: Vec<(Array, Array)> =
                (0..10).map(|_| (rnd(&[1, 2, cap, 256]), rnd(&[1, 2, cap, 256]))).collect();
            let q = rnd(&[1, 16, 1, 256]);
            eval(caches.iter().flat_map(|(k, v)| [k, v]).chain([&q])).unwrap();
            let time = |f: &dyn Fn(&Array, &Array) -> Array| {
                let mut ms = 0.0;
                for _ in 0..3 {
                    let t = Instant::now();
                    for _ in 0..20 {
                        let outs: Vec<Array> = caches.iter().map(|(k, v)| f(k, v)).collect();
                        eval(outs.iter()).unwrap();
                    }
                    ms = t.elapsed().as_secs_f64() * 1e3 / 20.0;
                }
                ms
            };
            let tail = rnd(&[1, 2, 256, 256]);
            let ours = time(&|k, v| a.apply(&q, k, v, n, &tail, &tail, 0).unwrap());
            let mlx = time(&|k, v| {
                let (ks, vs) = (k.index((.., .., ..n)), v.index((.., .., ..n)));
                fast::scaled_dot_product_attention(&q, &ks, &vs, 1.0 / 16.0, None, None).unwrap()
            });
            let gb = 10.0 * 2.0 * 2.0 * 256.0 * 2.0 * f64::from(n) / 1e9;
            println!(
                "n {n}: ours {ours:.2} ms ({:.0} GB/s), mlx {mlx:.2} ms ({:.0} GB/s)",
                gb / ours * 1e3,
                gb / mlx * 1e3
            );
        }
    }
}

#[cfg(test)]
mod prefill_tests {
    use ironbark_mlx::fast::{self, ScaledDotProductAttentionMask};
    use ironbark_mlx::random::normal;
    use ironbark_mlx::transforms::eval;

    use super::*;

    fn rnd(s: &[i32]) -> Array {
        normal::<f32>(s, None, None, None).unwrap().as_dtype(Dtype::Bfloat16).unwrap()
    }

    /// MLX's fused kernel gives a 512-token chunk the same attention as its unfused path, with
    /// cached keys before the chunk (causal, aligned to the end).
    #[test]
    fn fused_prefill_matches_unfused() {
        let (q, k, v) = (rnd(&[1, 16, 512, 256]), rnd(&[1, 2, 4608, 256]), rnd(&[1, 2, 4608, 256]));
        let m = || Some(ScaledDotProductAttentionMask::Causal);
        let a = fast::scaled_dot_product_attention(&q, &k, &v, 1.0 / 16.0, m(), None).unwrap();
        let b = fast::scaled_dot_product_attention_fused(&q, &k, &v, 1.0 / 16.0, m(), None).unwrap();
        let d = a.as_dtype(Dtype::Float32).unwrap().subtract(b.as_dtype(Dtype::Float32).unwrap()).unwrap();
        let e = d.abs().unwrap().max(None).unwrap();
        eval([&e]).unwrap();
        assert!(e.item_exact::<f32>() < 1e-2, "max difference {}", e.item_exact::<f32>());
    }
}
