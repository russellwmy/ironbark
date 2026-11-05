//! Ternary experts: weights `a · (c − 1)` with codes `c ∈ {0, 1, 2}` and one scale `a` per
//! row and 64 inputs, stored in trits ([`crate::row`]'s rows: 16 weights in 26 bits) and
//! multiplied straight from the streaming slab by two Metal kernels.
//!
//! ```text
//!   record: [gate rows F x row(H)][up rows F x row(H)][down rows H x row(F)]
//!           [gate scales F x H/64][up scales F x H/64][down scales H x F/64]   (bf16)
//!   row(K) = K/16 x 3 bytes (weights 0..14 of each 16, base 3) + K/64 bytes (weight 15, 2 bits)
//!
//!   up_gate  act[s, r] = silu(gate_r · x[tok[s]]) · (up_r · x[tok[s]])  expert in slot slots[s]
//!   down     y[s, n]   = down_n · act[s]
//! ```
//!
//! Qwen3.6-35B-A3B: 0.74 MB per expert.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use ironbark_mlx::{Array, Dtype};

use crate::affine::AffineRecord;
use crate::kernel::{Kernel, Output};
use crate::pack::{PAGE, Packed, rows_to_trits};
use crate::{Error, Result, manifest};

/// `experts.json` format of ternary records.
pub(crate) const FORMAT: &str = "ternary-trit";

/// Inputs per scale.
pub(crate) const GROUP: usize = 64;

/// Byte layout of one ternary expert record.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TernRecord {
    pub(crate) hidden: usize,
    pub(crate) ffn: usize,
}

impl TernRecord {
    /// Bytes of one trit row of `k` weights.
    pub(crate) fn row(k: usize) -> usize {
        k / 16 * 3 + k / 64
    }

    pub(crate) fn up(self) -> usize {
        self.ffn * Self::row(self.hidden)
    }

    pub(crate) fn down(self) -> usize {
        2 * self.up()
    }

    pub(crate) fn gate_scales(self) -> usize {
        self.down() + self.hidden * Self::row(self.ffn)
    }

    pub(crate) fn up_scales(self) -> usize {
        self.gate_scales() + self.ffn * (self.hidden / GROUP) * 2
    }

    pub(crate) fn down_scales(self) -> usize {
        self.up_scales() + self.ffn * (self.hidden / GROUP) * 2
    }

    pub(crate) fn bytes(self) -> usize {
        self.down_scales() + self.hidden * (self.ffn / GROUP) * 2
    }
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |source| Error::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Write `out/experts.bin` / `experts.json` / `manifest.json` from ternary experts in
/// `ternary` (`{l}.{gate,up,down}.codes`: u8, 4 codes per byte, low bits first,
/// `[experts, rows, cols / 4]`; `{l}.{name}.scales`: bf16 `[experts, rows, cols / 64]`), and mark
/// `out/config.json` (already written with the slim weights) as having ternary experts.
pub(crate) fn pack(out: &Path, ternary: &Path, layers: usize, experts: usize, rec: TernRecord) -> Result<Packed> {
    let mut t = Array::load_safetensors(ternary).map_err(|e| Error::Load {
        path: ternary.to_path_buf(),
        msg: e.to_string(),
    })?;
    // Taken out of the map, so each layer's tensors are freed once its records are written
    // (held to the end, the whole export is resident: 9 GB for Qwen3.6-35B-A3B).
    let mut take =
        |name: String| t.remove(&name).ok_or_else(|| Error::Format(format!("{}: no {name}", ternary.display())));
    let stride = rec.bytes().div_ceil(PAGE) * PAGE;
    let bin = out.join("experts.bin");
    let file = File::create(&bin).map_err(io(&bin))?;
    ironbark_experts::layout::no_cache(&file).map_err(io(&bin))?;
    let (h, f) = (rec.hidden, rec.ffn);
    let mut buf = vec![0u8; stride];
    let mut hashes = Vec::with_capacity(layers * experts);
    for l in 0..layers {
        // Per matrix: codes as u32 words (16 codes each), scales as bf16 bits; and where the
        // record holds them.
        let mut mats = Vec::with_capacity(3);
        for (name, rows, cols, codes_at, scales_at) in [
            ("gate", f, h, 0, rec.gate_scales()),
            ("up", f, h, rec.up(), rec.up_scales()),
            ("down", h, f, rec.down(), rec.down_scales()),
        ] {
            let c = take(format!("{l}.{name}.codes"))?;
            let s = take(format!("{l}.{name}.scales"))?;
            if c.shape() != [experts as i32, rows as i32, (cols / 4) as i32]
                || s.shape() != [experts as i32, rows as i32, (cols / GROUP) as i32]
            {
                return Err(Error::Format(format!("{}: {l}.{name} does not match the config", ternary.display())));
            }
            c.eval()?;
            let bytes = c.as_slice::<u8>();
            let words: Vec<u32> = bytes.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
            mats.push((words, crate::pack::bf16_bits(&s)?, rows, cols, codes_at, scales_at));
        }
        for e in 0..experts {
            buf.fill(0);
            for (words, scales, rows, cols, codes_at, scales_at) in &mats {
                let per = rows * cols / 16;
                let n = rows * TernRecord::row(*cols);
                rows_to_trits(&words[e * per..(e + 1) * per], *cols, &mut buf[*codes_at..codes_at + n]);
                let sp = rows * cols / GROUP;
                for (i, v) in scales[e * sp..(e + 1) * sp].iter().enumerate() {
                    buf[scales_at + 2 * i..scales_at + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
                }
            }
            file.write_all_at(&buf, ((l * experts + e) * stride) as u64).map_err(io(&bin))?;
            hashes.push(manifest::hash(&buf[..rec.bytes()]));
        }
        ironbark_mlx::memory::clear_cache()?;
    }
    let marker = serde_json::json!({ "ternary": true, "group_size": GROUP });
    crate::experts::finish(out, layers, experts, rec.bytes(), stride, FORMAT, &marker, &hashes)?;
    Ok(Packed {
        records: layers * experts,
        record_bytes: rec.bytes(),
        stride,
    })
}

/// 16 codes of trit group `g` of a row (`groups` 16-weight groups long), dotted with `x`:
/// `a · (Σ c_i x_i − Σ x_i)`.
const TDOT: &str = r"
inline float tern_dot(const device uchar* row, uint groups, uint g, const thread float* x, float sum, float a) {
    const device uchar* t=row+g*3;
    float acc=0;
    for (int b=0;b<3;++b) {
        ushort v=maple_trit_lut[t[b]];
        for (int n=0;n<5;++n) acc+=x[5*b+n]*float((v>>(2*n))&3);
    }
    acc+=x[15]*float((row[groups*3+g/4]>>(2*(g%4)))&3);
    return a*(acc-sum);
}
";

/// Gate and up rows of the expert in slot `slots[s]` on `x[tok[s]]`, then SwiGLU. A threadgroup
/// takes `UR` rows; its `UKS` SIMD groups split each row's inputs (so a thread walks a short
/// chain at decode's tiny sizes, where latency, not bandwidth, bounds the time) and add their
/// partial sums in threadgroup memory. A lane covers 16 inputs per 512-wide step.
const UP_GATE: &str = r"
    uint lid=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
    uint s=threadgroup_position_in_grid.y;
    uint row0=threadgroup_position_in_grid.x*UR;
    const device uchar* rec=(const device uchar*)slab+(ulong)slots[s]*REC;
    const device bfloat16_t* sgate=(const device bfloat16_t*)(rec+GSOFF);
    const device bfloat16_t* sup=(const device bfloat16_t*)(rec+USOFF);
    const device bfloat16_t* xv=x+(ulong)tok[s]*H;
    constexpr uint ROWB=H/16*3+H/64, PART=H/UKS;
    threadgroup float part[UKS][2*UR];
    float up[UR]={0}, gate[UR]={0};
    for (uint k=sg*PART;k<(sg+1)*PART;k+=512) {
        uint kk=k+lid*16;
        if (kk<(sg+1)*PART) {
            float xt[16], sum=0;
            for (int i=0;i<16;++i) { xt[i]=float(xv[kk+i]); sum+=xt[i]; }
            uint g=kk/16, grp=kk/64;
            for (uint r=0;r<UR;++r) {
                uint row=row0+r;
                gate[r]+=tern_dot(rec+row*ROWB, H/16, g, xt, sum, float(sgate[row*(H/64)+grp]));
                up[r]+=tern_dot(rec+UOFF+row*ROWB, H/16, g, xt, sum, float(sup[row*(H/64)+grp]));
            }
        }
    }
    for (uint r=0;r<UR;++r) {
        float gs=simd_sum(gate[r]), us=simd_sum(up[r]);
        if (lid==0) { part[sg][r]=gs; part[sg][UR+r]=us; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint t=thread_index_in_threadgroup;
    if (t<UR) {
        float gs=0, us=0;
        for (uint j=0;j<UKS;++j) { gs+=part[j][t]; us+=part[j][UR+t]; }
        float g=float(bfloat16_t(gs)), u=float(bfloat16_t(us));
        out[(ulong)s*F+row0+t]=bfloat16_t(g/(1.0f+metal::precise::exp(-g))*u);
    }
";

/// Down rows of the expert in slot `slots[s]` on `act[s]`; `DR` rows per threadgroup, inputs
/// split over `DKS` SIMD groups as in [`UP_GATE`].
const DOWN: &str = r"
    uint lid=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
    uint s=threadgroup_position_in_grid.y;
    uint row0=threadgroup_position_in_grid.x*DR;
    const device uchar* rec=(const device uchar*)slab+(ulong)slots[s]*REC;
    const device bfloat16_t* sd=(const device bfloat16_t*)(rec+DSOFF);
    const device bfloat16_t* xv=act+(ulong)s*F;
    constexpr uint ROWB=F/16*3+F/64, PART=F/DKS;
    threadgroup float part[DKS][DR];
    float y[DR]={0};
    for (uint k=sg*PART;k<(sg+1)*PART;k+=512) {
        uint kk=k+lid*16;
        if (kk<(sg+1)*PART) {
            float xt[16], sum=0;
            for (int i=0;i<16;++i) { xt[i]=float(xv[kk+i]); sum+=xt[i]; }
            uint g=kk/16, grp=kk/64;
            for (uint r=0;r<DR;++r) {
                uint row=row0+r;
                y[r]+=tern_dot(rec+DOFF+row*ROWB, F/16, g, xt, sum, float(sd[row*(F/64)+grp]));
            }
        }
    }
    for (uint r=0;r<DR;++r) {
        float v=simd_sum(y[r]);
        if (lid==0) part[sg][r]=v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint t=thread_index_in_threadgroup;
    if (t<DR) {
        float v=0;
        for (uint j=0;j<DKS;++j) v+=part[j][t];
        out[(ulong)s*H+row0+t]=bfloat16_t(v);
    }
";

/// Rows per threadgroup and input splits of the two kernels.
const UR: usize = 4;
const UKS: usize = 4;
const DR: usize = 4;
const DKS: usize = 2;

/// [`UP_GATE`] on one row `x` for the fused decode: rows 0..8 the routed experts (their slots
/// `table[ids[s]]`, zeros where `live` is 0), row 8 the 4-bit `shared` record.
const FAST_UP_GATE: &str = r"
    uint lid=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
    uint s=threadgroup_position_in_grid.y;
    uint row0=threadgroup_position_in_grid.x*UR;
    bool routed=s<8;
    const device uchar* rec=routed?(const device uchar*)slab+(ulong)table[ids[s]]*REC:(const device uchar*)shared;
    const device bfloat16_t* sgate=(const device bfloat16_t*)(rec+(routed?GSOFF:SGSOFF));
    const device bfloat16_t* sup=(const device bfloat16_t*)(rec+(routed?USOFF:SUSOFF));
    const device bfloat16_t* bgate=(const device bfloat16_t*)(rec+SGBOFF);
    const device bfloat16_t* bup=(const device bfloat16_t*)(rec+SUBOFF);
    const device uint* wgate=(const device uint*)rec;
    const device uint* wup=(const device uint*)(rec+SUOFF);
    constexpr uint ROWB=H/16*3+H/64, ROWW=H*BITS/32, PART=H/FUKS;
    threadgroup float part[FUKS][2*UR];
    float up[UR]={0}, gate[UR]={0};
    bool on=!routed||live[s]!=0.0f;
    for (uint k=sg*PART;on&&k<(sg+1)*PART;k+=512) {
        uint kk=k+lid*16;
        if (kk<(sg+1)*PART) {
            float xt[16], sum=0;
            for (int i=0;i<16;++i) { xt[i]=float(x[kk+i]); sum+=xt[i]; }
            uint g=kk/16, grp=kk/64;
            if (routed) {
                for (uint r=0;r<UR;++r) {
                    uint row=row0+r;
                    gate[r]+=tern_dot(rec+row*ROWB, H/16, g, xt, sum, float(sgate[row*(H/64)+grp]));
                    up[r]+=tern_dot(rec+UOFF+row*ROWB, H/16, g, xt, sum, float(sup[row*(H/64)+grp]));
                }
            } else {
                uint at=g*W16;
                for (uint r=0;r<UR;++r) {
                    uint row=row0+r, gi=row*(H/64)+grp;
                    gate[r]+=float(sgate[gi])*aff_dot(wgate+row*ROWW+at, xt)+float(bgate[gi])*sum;
                    up[r]+=float(sup[gi])*aff_dot(wup+row*ROWW+at, xt)+float(bup[gi])*sum;
                }
            }
        }
    }
    for (uint r=0;r<UR;++r) {
        float gs=simd_sum(gate[r]), us=simd_sum(up[r]);
        if (lid==0) { part[sg][r]=gs; part[sg][UR+r]=us; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint t=thread_index_in_threadgroup;
    if (t<UR) {
        float gs=0, us=0;
        for (uint j=0;j<FUKS;++j) { gs+=part[j][t]; us+=part[j][UR+t]; }
        float g=float(bfloat16_t(gs)), u=float(bfloat16_t(us));
        out[(ulong)s*F+row0+t]=bfloat16_t(g/(1.0f+metal::precise::exp(-g))*u);
    }
";

/// [`DOWN`] for the fused decode, rows as [`FAST_UP_GATE`].
const FAST_DOWN: &str = r"
    uint lid=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
    uint s=threadgroup_position_in_grid.y;
    uint row0=threadgroup_position_in_grid.x*DR;
    bool routed=s<8;
    const device uchar* rec=routed?(const device uchar*)slab+(ulong)table[ids[s]]*REC:(const device uchar*)shared;
    const device bfloat16_t* sd=(const device bfloat16_t*)(rec+(routed?DSOFF:SDSOFF));
    const device bfloat16_t* bd=(const device bfloat16_t*)(rec+SDBOFF);
    const device uint* wd=(const device uint*)(rec+SDOFF);
    const device bfloat16_t* xv=act+(ulong)s*F;
    constexpr uint ROWB=F/16*3+F/64, ROWW=F*BITS/32, PART=F/FDKS;
    threadgroup float part[FDKS][DR];
    float y[DR]={0};
    bool on=!routed||live[s]!=0.0f;
    for (uint k=sg*PART;on&&k<(sg+1)*PART;k+=512) {
        uint kk=k+lid*16;
        if (kk<(sg+1)*PART) {
            float xt[16], sum=0;
            for (int i=0;i<16;++i) { xt[i]=float(xv[kk+i]); sum+=xt[i]; }
            uint g=kk/16, grp=kk/64;
            if (routed) {
                for (uint r=0;r<DR;++r) {
                    uint row=row0+r;
                    y[r]+=tern_dot(rec+DOFF+row*ROWB, F/16, g, xt, sum, float(sd[row*(F/64)+grp]));
                }
            } else {
                uint at=g*W16;
                for (uint r=0;r<DR;++r) {
                    uint row=row0+r, gi=row*(F/64)+grp;
                    y[r]+=float(sd[gi])*aff_dot(wd+row*ROWW+at, xt)+float(bd[gi])*sum;
                }
            }
        }
    }
    for (uint r=0;r<DR;++r) {
        float v=simd_sum(y[r]);
        if (lid==0) part[sg][r]=v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint t=thread_index_in_threadgroup;
    if (t<DR) {
        float v=0;
        for (uint j=0;j<FDKS;++j) v+=part[j][t];
        out[(ulong)s*H+row0+t]=bfloat16_t(v);
    }
";

/// Input splits of the fused kernels.
const FUKS: usize = 2;
const FDKS: usize = 1;

/// A slot table `[layers · experts]` of global slots as each layer's slots numbered from 0.
pub(crate) fn layer_local(table: &Array, experts: usize, per_layer: usize) -> Result<Array> {
    table.eval()?;
    let t: Vec<u32> = table
        .as_slice::<u32>()
        .iter()
        .enumerate()
        .map(|(i, &s)| s - ((i / experts) * per_layer) as u32)
        .collect();
    let a = Array::from_slice(&t, &[t.len() as i32]);
    a.eval()?;
    Ok(a)
}

/// The two kernels of one record layout. Its sizes and offsets are compiled into the source
/// (constants, not template arguments): MLX resolves template arguments on every launch, a
/// fixed cost of tens of µs per call at decode's tiny sizes.
pub(crate) struct Kernels {
    up_gate: Kernel,
    down: Kernel,
    rec: TernRecord,
    stride: usize,
}

/// The fused decode's two kernels: routed ternary experts and an affine shared expert in one
/// launch each.
pub(crate) struct FastKernels {
    up_gate: Kernel,
    down: Kernel,
    rec: TernRecord,
}

impl FastKernels {
    /// The 8 routed experts `ids` (`table[ids]` their slots in `slab`; skipped where `live` is 0)
    /// and the `shared` record on one row `x`: `[9, hidden]`.
    pub(crate) fn apply(
        &self,
        x: &Array,
        slab: &Array,
        shared: &Array,
        table: &Array,
        ids: &Array,
        live: &Array,
    ) -> Result<Array> {
        let (f, h) = (self.rec.ffn as i32, self.rec.hidden as i32);
        let act = self
            .up_gate
            .apply(
                &[x, slab, shared, table, ids, live],
                &[],
                (f / UR as i32 * 32 * FUKS as i32, 9, 1),
                (32 * FUKS as i32, 1, 1),
                &[Output {
                    shape: &[9, f],
                    dtype: Dtype::Bfloat16,
                }],
            )?
            .remove(0);
        Ok(self
            .down
            .apply(
                &[&act, slab, shared, table, ids, live],
                &[],
                (h / DR as i32 * 32 * FDKS as i32, 9, 1),
                (32 * FDKS as i32, 1, 1),
                &[Output {
                    shape: &[9, h],
                    dtype: Dtype::Bfloat16,
                }],
            )?
            .remove(0))
    }
}

impl Kernels {
    pub(crate) fn new(rec: TernRecord, stride: usize) -> Self {
        let consts = [
            ("H", rec.hidden),
            ("F", rec.ffn),
            ("REC", stride),
            ("UOFF", rec.up()),
            ("DOFF", rec.down()),
            ("GSOFF", rec.gate_scales()),
            ("USOFF", rec.up_scales()),
            ("DSOFF", rec.down_scales()),
            ("UR", UR),
            ("UKS", UKS),
            ("DR", DR),
            ("DKS", DKS),
        ];
        let defs: String = consts
            .iter()
            .map(|(n, v)| format!("constant constexpr uint {n} = {v};\n"))
            .collect::<Vec<_>>()
            .concat();
        let header = format!("{}{defs}{TDOT}", crate::row::trit_lut());
        Kernels {
            up_gate: Kernel::new("ironbark_tern_up_gate", &["x", "slab", "slots", "tok"], &["out"], UP_GATE, &header),
            down: Kernel::new("ironbark_tern_down", &["act", "slab", "slots"], &["out"], DOWN, &header),
            rec,
            stride,
        }
    }

    /// The fused decode's kernels, for a shared expert with this record layout.
    pub(crate) fn fast(&self, shared: AffineRecord) -> FastKernels {
        let rec = self.rec;
        let consts = [
            ("H", rec.hidden),
            ("F", rec.ffn),
            ("REC", self.stride),
            ("UOFF", rec.up()),
            ("DOFF", rec.down()),
            ("GSOFF", rec.gate_scales()),
            ("USOFF", rec.up_scales()),
            ("DSOFF", rec.down_scales()),
            ("UR", UR),
            ("DR", DR),
            ("FUKS", FUKS),
            ("FDKS", FDKS),
        ];
        let defs: String = consts
            .iter()
            .map(|(n, v)| format!("constant constexpr uint {n} = {v};\n"))
            .collect::<Vec<_>>()
            .concat();
        let header = format!("{}{defs}{}{TDOT}{}", crate::row::trit_lut(), shared.defs("S"), crate::affine::ADOT);
        let inputs = ["slab", "shared", "table", "ids", "live"];
        let named = |first: &'static str| {
            let mut v = vec![first];
            v.extend(inputs);
            v
        };
        FastKernels {
            up_gate: Kernel::new("ironbark_tern_fast_up_gate", &named("x"), &["out"], FAST_UP_GATE, &header),
            down: Kernel::new("ironbark_tern_fast_down", &named("act"), &["out"], FAST_DOWN, &header),
            rec,
        }
    }

    /// Experts in slots `slots` `[S]` (u32) of `slab` on rows `tok` `[S]` (u32) of `x`
    /// `[T, hidden]` (bf16): `[S, hidden]`.
    pub(crate) fn apply(&self, x: &Array, slab: &Array, slots: &Array, tok: &Array) -> Result<Array> {
        let act = self.up_gate(x, slab, slots, tok)?;
        self.down(&act, slab, slots)
    }

    /// Gate and up rows with the activation: `[S, ffn]`.
    pub(crate) fn up_gate(&self, x: &Array, slab: &Array, slots: &Array, tok: &Array) -> Result<Array> {
        let f = self.rec.ffn as i32;
        let n = slots.shape()[0];
        let x = x.as_dtype(Dtype::Bfloat16)?;
        Ok(self
            .up_gate
            .apply(
                &[&x, slab, slots, tok],
                &[],
                (f / UR as i32 * 32 * UKS as i32, n, 1),
                (32 * UKS as i32, 1, 1),
                &[Output {
                    shape: &[n, f],
                    dtype: Dtype::Bfloat16,
                }],
            )?
            .remove(0))
    }

    /// Down rows on `act` `[S, ffn]`: `[S, hidden]`.
    pub(crate) fn down(&self, act: &Array, slab: &Array, slots: &Array) -> Result<Array> {
        let h = self.rec.hidden as i32;
        let n = slots.shape()[0];
        Ok(self
            .down
            .apply(
                &[act, slab, slots],
                &[],
                (h / DR as i32 * 32 * DKS as i32, n, 1),
                (32 * DKS as i32, 1, 1),
                &[Output {
                    shape: &[n, h],
                    dtype: Dtype::Bfloat16,
                }],
            )?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random numbers in [0, 1).
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (*seed >> 40) as f32 / (1u64 << 24) as f32
    }

    fn bf16(v: f32) -> f32 {
        half::bf16::from_f32(v).to_f32()
    }

    #[test]
    fn kernels_match_plain_arithmetic() {
        let (h, f, ne) = (128usize, 64usize, 2usize);
        let rec = TernRecord { hidden: h, ffn: f };
        let stride = rec.bytes().div_ceil(PAGE) * PAGE;
        let mut seed = 7u64;
        // Per expert: codes (0..3) and bf16 scales of gate, up, down, and the record bytes.
        let mut slab = vec![0u8; ne * stride];
        let mut mats = Vec::new();
        for e in 0..ne {
            let mut m = Vec::new();
            for (rows, cols, codes_at, scales_at) in [
                (f, h, 0, rec.gate_scales()),
                (f, h, rec.up(), rec.up_scales()),
                (h, f, rec.down(), rec.down_scales()),
            ] {
                let codes: Vec<u32> = (0..rows * cols).map(|_| (lcg(&mut seed) * 3.0) as u32 % 3).collect();
                let scales: Vec<f32> = (0..rows * cols / GROUP).map(|_| bf16(0.01 + 0.05 * lcg(&mut seed))).collect();
                let words: Vec<u32> =
                    codes.chunks(16).map(|c| c.iter().enumerate().fold(0, |w, (i, &v)| w | (v << (2 * i)))).collect();
                let base = e * stride;
                rows_to_trits(&words, cols, &mut slab[base + codes_at..base + codes_at + rows * TernRecord::row(cols)]);
                for (i, a) in scales.iter().enumerate() {
                    let b = half::bf16::from_f32(*a).to_bits().to_le_bytes();
                    slab[base + scales_at + 2 * i..base + scales_at + 2 * i + 2].copy_from_slice(&b);
                }
                let w: Vec<f32> = (0..rows * cols).map(|i| scales[i / GROUP] * (codes[i] as f32 - 1.0)).collect();
                m.push(w);
            }
            mats.push(m);
        }
        let x: Vec<f32> = (0..h).map(|_| bf16(lcg(&mut seed) - 0.5)).collect();
        let words: Vec<u32> = slab.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
        let array = Array::from_slice(&words, &[ne as i32, (stride / 4) as i32]);
        let kernels = Kernels::new(rec, stride);
        let slots = Array::from_slice(&[1u32, 0], &[2]);
        let tok = Array::from_slice(&[0u32, 0], &[2]);
        let xa = Array::from_slice(&x, &[1, h as i32]).as_dtype(Dtype::Bfloat16).unwrap();
        let got = kernels.apply(&xa, &array, &slots, &tok).unwrap().as_dtype(Dtype::Float32).unwrap();
        let got = got.as_slice::<f32>();
        for (s, &e) in [1usize, 0].iter().enumerate() {
            let [gate, up, down] = &mats[e][..] else { unreachable!() };
            let dot = |w: &[f32], r: usize, v: &[f32]| (0..v.len()).map(|k| w[r * v.len() + k] * v[k]).sum::<f32>();
            let act: Vec<f32> = (0..f)
                .map(|r| {
                    let (g, u) = (bf16(dot(gate, r, &x)), bf16(dot(up, r, &x)));
                    bf16(g / (1.0 + (-g).exp()) * u)
                })
                .collect();
            for n in 0..h {
                let want = dot(down, n, &act);
                let g = got[s * h + n];
                assert!((g - want).abs() <= 1e-2 * want.abs().max(1e-2), "slot {s} row {n}: {g} vs {want}");
            }
        }
    }

    #[test]
    fn fast_kernels_match_the_separate_ones() {
        let (h, f, ne) = (128usize, 64usize, 2usize);
        let rec = TernRecord { hidden: h, ffn: f };
        let stride = rec.bytes().div_ceil(PAGE) * PAGE;
        let mut seed = 11u64;
        let mut slab = vec![0u8; ne * stride];
        for b in &mut slab {
            *b = (lcg(&mut seed) * 243.0) as u8;
        }
        // The extra codes of each row (weight 15, 2 bits) must not be 3; scales in range.
        for e in 0..ne {
            let base = e * stride;
            for (rows, cols, at) in [(f, h, 0), (f, h, rec.up()), (h, f, rec.down())] {
                let row = TernRecord::row(cols);
                for r in 0..rows {
                    let extra = &mut slab[base + at + r * row + cols / 16 * 3..base + at + r * row + row];
                    extra.fill(0b0110_0110);
                }
            }
            for at in (rec.gate_scales()..rec.bytes()).step_by(2) {
                let b = half::bf16::from_f32(0.01 + 0.05 * lcg(&mut seed)).to_bits().to_le_bytes();
                slab[base + at..base + at + 2].copy_from_slice(&b);
            }
        }
        let shared_rec = AffineRecord {
            hidden: h,
            ffn: f,
            bits: 4,
        };
        let shared_bytes = shared_rec.bytes().next_multiple_of(4);
        let mut bytes: Vec<u8> = (0..shared_bytes).map(|_| (lcg(&mut seed) * 255.0) as u8).collect();
        for pair in bytes[shared_rec.scales_start()..shared_rec.bytes()].as_chunks_mut::<2>().0 {
            *pair = half::bf16::from_f32(0.02 * lcg(&mut seed) - 0.01).to_bits().to_le_bytes();
        }
        let words = |b: &[u8]| -> Vec<u32> { b.as_chunks::<4>().0.iter().map(|w| u32::from_le_bytes(*w)).collect() };
        let array = Array::from_slice(&words(&slab), &[ne as i32, (stride / 4) as i32]);
        let shared = Array::from_slice(&words(&bytes), &[1, (shared_bytes / 4) as i32]);
        let x: Vec<f32> = (0..h).map(|_| bf16(lcg(&mut seed) - 0.5)).collect();
        let xa = Array::from_slice(&x, &[1, 1, h as i32]).as_dtype(Dtype::Bfloat16).unwrap();
        let flat = xa.reshape(&[1, h as i32]).unwrap();

        let kernels = Kernels::new(rec, stride);
        let fast = kernels.fast(shared_rec);
        let order = [1u32, 0, 1, 0, 1, 0, 1, 0];
        let table = Array::from_slice(&[0u32, 1], &[2]);
        let ids = Array::from_slice(&order, &[8]);
        let live = Array::from_slice(&[1.0f32, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0], &[8]);
        let got = fast.apply(&xa, &array, &shared, &table, &ids, &live).unwrap().as_dtype(Dtype::Float32).unwrap();
        let got = got.as_slice::<f32>();

        let slots = Array::from_slice(&order, &[8]);
        let tok = Array::from_slice(&[0u32; 8], &[8]);
        let want = kernels.apply(&flat, &array, &slots, &tok).unwrap().as_dtype(Dtype::Float32).unwrap();
        let want = want.as_slice::<f32>();
        for s in 0..8 {
            for n in 0..h {
                let w = if s == 3 { 0.0 } else { want[s * h + n] };
                let g = got[s * h + n];
                assert!((g - w).abs() <= 1e-2 * w.abs().max(1e-2), "row {s} output {n}: {g} vs {w}");
            }
        }
        let affine = crate::affine::Kernels::new(shared_rec, shared_bytes);
        let one = Array::from_slice(&[0u32], &[1]);
        let on = Array::from_slice(&[1.0f32], &[1]);
        let want = affine.apply(&flat, &shared, &one, &one, &on).unwrap().as_dtype(Dtype::Float32).unwrap();
        let want = want.as_slice::<f32>();
        for n in 0..h {
            let g = got[8 * h + n];
            assert!((g - want[n]).abs() <= 1e-2 * want[n].abs().max(1e-2), "shared output {n}: {g} vs {}", want[n]);
        }
    }

    #[test]
    fn record_sections_follow_each_other() {
        let r = TernRecord { hidden: 2048, ffn: 512 };
        assert_eq!(TernRecord::row(2048), 128 * 3 + 32);
        assert_eq!(r.up(), 512 * 416);
        assert_eq!(r.gate_scales(), 2 * 512 * 416 + 2048 * 104);
        assert_eq!(r.bytes(), r.gate_scales() + (2 * 512 * 32 + 2048 * 8) * 2);
        assert_eq!(r.bytes(), 737_280);
    }
}
