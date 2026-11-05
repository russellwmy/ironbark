//! Affine experts (a 4-bit or 8-bit MLX checkpoint's): weights `scale · code + bias` with one
//! scale and bias per row and 64 inputs, stored as the checkpoint holds them and multiplied
//! straight from the streaming slab by two Metal kernels.
//!
//! ```text
//!   record: [gate words F x H·bits/32][up words F x H·bits/32][down words H x F·bits/32]   (u32)
//!           [gate scales][gate biases][up scales][up biases]   F x H/64 each               (bf16)
//!           [down scales][down biases]                         H x F/64 each               (bf16)
//!
//!   up_gate  act[s, r] = silu(gate_r · x[tok[s]]) · (up_r · x[tok[s]])  expert in slot slots[s]
//!   down     y[s, n]   = down_n · act[s]
//!   a pair with live[s] == 0 (dropped by dynamic top-k) is skipped on the GPU: zeros, no readback
//! ```
//!
//! Qwen3.6-35B-A3B: 1.77 MB per expert at 4 bits, 3.34 MB at 8.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use ironbark_mlx::{Array, Dtype};

use crate::kernel::{Kernel, Output};
use crate::pack::{PAGE, Packed};
use crate::tern::GROUP;
use crate::{Error, Result, manifest};

/// `experts.json` format of affine records of `bits`.
pub(crate) fn format(bits: usize) -> String {
    format!("affine-{bits}")
}

/// Byte layout of one affine expert record.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AffineRecord {
    pub(crate) hidden: usize,
    pub(crate) ffn: usize,
    pub(crate) bits: usize,
}

impl AffineRecord {
    /// Bytes of one matrix's codes.
    fn words(self, rows: usize, cols: usize) -> usize {
        rows * cols * self.bits / 8
    }

    /// Bytes of one matrix's scales (and of its biases).
    fn scales(rows: usize, cols: usize) -> usize {
        rows * (cols / GROUP) * 2
    }

    fn up(self) -> usize {
        self.words(self.ffn, self.hidden)
    }

    fn down(self) -> usize {
        2 * self.up()
    }

    fn gate_scales(self) -> usize {
        self.down() + self.words(self.hidden, self.ffn)
    }

    fn gate_biases(self) -> usize {
        self.gate_scales() + Self::scales(self.ffn, self.hidden)
    }

    fn up_scales(self) -> usize {
        self.gate_biases() + Self::scales(self.ffn, self.hidden)
    }

    fn up_biases(self) -> usize {
        self.up_scales() + Self::scales(self.ffn, self.hidden)
    }

    fn down_scales(self) -> usize {
        self.up_biases() + Self::scales(self.ffn, self.hidden)
    }

    fn down_biases(self) -> usize {
        self.down_scales() + Self::scales(self.hidden, self.ffn)
    }

    pub(crate) fn bytes(self) -> usize {
        self.down_biases() + Self::scales(self.hidden, self.ffn)
    }

    #[cfg(test)]
    pub(crate) fn scales_start(self) -> usize {
        self.gate_scales()
    }

    /// The constants [`ADOT`] needs and this record's offsets, named `{prefix}UOFF` and so on.
    pub(crate) fn defs(self, prefix: &str) -> String {
        let shared = [
            ("UOFF", self.up()),
            ("DOFF", self.down()),
            ("GSOFF", self.gate_scales()),
            ("GBOFF", self.gate_biases()),
            ("USOFF", self.up_scales()),
            ("UBOFF", self.up_biases()),
            ("DSOFF", self.down_scales()),
            ("DBOFF", self.down_biases()),
        ];
        let dot = [
            ("BITS", self.bits),
            ("W16", self.bits / 2),
            ("EL", 32 / self.bits),
            ("MASK", (1 << self.bits) - 1),
        ];
        let line = |n: String, v: usize| format!("constant constexpr uint {n} = {v};\n");
        dot.iter()
            .map(|&(n, v)| line(n.to_owned(), v))
            .chain(shared.iter().map(|&(n, v)| line(format!("{prefix}{n}"), v)))
            .collect()
    }
}

/// The bits of the affine matrix with `words` and `scales` columns, if records can hold them.
pub(crate) fn bits_of(words: i32, scales: i32) -> Result<usize> {
    let bits = (words as usize * 32) / (scales as usize * GROUP);
    if bits == 4 || bits == 8 {
        Ok(bits)
    } else {
        Err(Error::Format(format!("affine experts: {bits} bits; records hold 4 or 8")))
    }
}

/// Write `out/experts.bin` / `experts.json` / `manifest.json` from a checkpoint's experts in
/// `t` (`{prefix(l)}.{gate,up,down}_proj.{weight,scales,biases}`, `[experts, rows, ·]`), and mark
/// `out/config.json` (already written with the slim weights) as having affine experts.
pub(crate) fn pack(
    out: &Path,
    mut t: std::collections::HashMap<String, Array>,
    prefix: impl Fn(usize) -> String,
    layers: usize,
    experts: usize,
    rec: AffineRecord,
) -> Result<Packed> {
    let io = |path: &Path| {
        let path = path.to_path_buf();
        move |source| Error::Io { path, source }
    };
    let stride = rec.bytes().div_ceil(PAGE) * PAGE;
    let bin = out.join("experts.bin");
    let file = File::create(&bin).map_err(io(&bin))?;
    ironbark_experts::layout::no_cache(&file).map_err(io(&bin))?;
    let (h, f) = (rec.hidden, rec.ffn);
    let mut buf = vec![0u8; stride];
    let mut hashes = Vec::with_capacity(layers * experts);
    for l in 0..layers {
        // Taken out of the map, so each layer's tensors are freed once its records are written.
        let mut take = |name: &str, part: &str| {
            let key = format!("{}.{name}_proj.{part}", prefix(l));
            t.remove(&key).ok_or_else(|| Error::Format(format!("checkpoint: no {key}")))
        };
        let mut mats = Vec::with_capacity(3);
        for (name, rows, cols, words_at, scales_at, biases_at) in [
            ("gate", f, h, 0, rec.gate_scales(), rec.gate_biases()),
            ("up", f, h, rec.up(), rec.up_scales(), rec.up_biases()),
            ("down", h, f, rec.down(), rec.down_scales(), rec.down_biases()),
        ] {
            let (w, s, b) = (take(name, "weight")?, take(name, "scales")?, take(name, "biases")?);
            let want = [experts as i32, rows as i32, (cols * rec.bits / 32) as i32];
            let groups = [experts as i32, rows as i32, (cols / GROUP) as i32];
            if w.shape() != want || s.shape() != groups || b.shape() != groups {
                return Err(Error::Format(format!("checkpoint: layer {l} {name} does not match the config")));
            }
            w.eval()?;
            let words = w.as_slice::<u32>().to_vec();
            let (scales, biases) = (crate::pack::bf16_bits(&s)?, crate::pack::bf16_bits(&b)?);
            mats.push((words, scales, biases, rows, cols, words_at, scales_at, biases_at));
        }
        for e in 0..experts {
            buf.fill(0);
            for (words, scales, biases, rows, cols, words_at, scales_at, biases_at) in &mats {
                let per = rows * cols * rec.bits / 32;
                for (i, v) in words[e * per..(e + 1) * per].iter().enumerate() {
                    buf[words_at + 4 * i..words_at + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
                }
                let sp = rows * cols / GROUP;
                for (at, vals) in [(scales_at, scales), (biases_at, biases)] {
                    for (i, v) in vals[e * sp..(e + 1) * sp].iter().enumerate() {
                        buf[at + 2 * i..at + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
                    }
                }
            }
            file.write_all_at(&buf, ((l * experts + e) * stride) as u64).map_err(io(&bin))?;
            hashes.push(manifest::hash(&buf[..rec.bytes()]));
        }
        ironbark_mlx::memory::clear_cache()?;
    }
    let marker = serde_json::json!({ "affine": true, "bits": rec.bits, "group_size": GROUP });
    crate::experts::finish(out, layers, experts, rec.bytes(), stride, &format(rec.bits), &marker, &hashes)?;
    Ok(Packed {
        records: layers * experts,
        record_bytes: rec.bytes(),
        stride,
    })
}

/// One expert's matrices `[weights, scales, biases]` (gate, up, down), as a record array
/// `[1, bytes / 4]` (u32) the kernels read like a slab's.
pub(crate) fn record_array(rec: AffineRecord, gate: [&Array; 3], up: [&Array; 3], down: [&Array; 3]) -> Result<Array> {
    let mut buf = vec![0u8; rec.bytes().div_ceil(4) * 4];
    for ([w, s, b], words_at, scales_at, biases_at) in [
        (gate, 0, rec.gate_scales(), rec.gate_biases()),
        (up, rec.up(), rec.up_scales(), rec.up_biases()),
        (down, rec.down(), rec.down_scales(), rec.down_biases()),
    ] {
        w.eval()?;
        for (i, v) in w.as_slice::<u32>().iter().enumerate() {
            buf[words_at + 4 * i..words_at + 4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        for (at, vals) in [
            (scales_at, crate::pack::bf16_bits(s)?),
            (biases_at, crate::pack::bf16_bits(b)?),
        ] {
            for (i, v) in vals.iter().enumerate() {
                buf[at + 2 * i..at + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
    }
    let words: Vec<u32> = buf.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
    let n = words.len() as i32;
    Ok(Array::from_slice(&words, &[1, n]))
}

/// The 16 codes of `w`, dotted with `x`.
pub(crate) const ADOT: &str = r"
inline float aff_dot(const device uint* w, const thread float* x) {
    float acc=0;
    for (uint i=0;i<W16;++i) {
        uint v=w[i];
        for (uint j=0;j<EL;++j) { acc+=x[i*EL+j]*float(v&MASK); v>>=BITS; }
    }
    return acc;
}
";

/// Gate and up rows of the expert in slot `slots[s]` on `x[tok[s]]`, then SwiGLU; zeros for a
/// dropped pair. Threadgroups and SIMD groups as [`crate::tern`]'s `up_gate`: a lane covers 16
/// inputs per 512-wide step.
const UP_GATE: &str = r"
    uint lid=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
    uint s=threadgroup_position_in_grid.y;
    uint row0=threadgroup_position_in_grid.x*UR;
    const device uchar* rec=RECORD;
    const device uint* wgate=(const device uint*)rec;
    const device uint* wup=(const device uint*)(rec+UOFF);
    const device bfloat16_t* sgate=(const device bfloat16_t*)(rec+GSOFF);
    const device bfloat16_t* bgate=(const device bfloat16_t*)(rec+GBOFF);
    const device bfloat16_t* sup=(const device bfloat16_t*)(rec+USOFF);
    const device bfloat16_t* bup=(const device bfloat16_t*)(rec+UBOFF);
    const device bfloat16_t* xv=XROW;
    constexpr uint ROWW=H*BITS/32, PART=H/UKS;
    threadgroup float part[UKS][2*UR];
    float up[UR]={0}, gate[UR]={0};
    bool on=LIVE;
    for (uint k=sg*PART;on&&k<(sg+1)*PART;k+=512) {
        uint kk=k+lid*16;
        if (kk<(sg+1)*PART) {
            float xt[16], sum=0;
            for (int i=0;i<16;++i) { xt[i]=float(xv[kk+i]); sum+=xt[i]; }
            uint at=kk/16*W16, grp=kk/64;
            for (uint r=0;r<UR;++r) {
                uint row=row0+r, gi=row*(H/64)+grp;
                gate[r]+=float(sgate[gi])*aff_dot(wgate+row*ROWW+at, xt)+float(bgate[gi])*sum;
                up[r]+=float(sup[gi])*aff_dot(wup+row*ROWW+at, xt)+float(bup[gi])*sum;
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

/// Down rows of the expert in slot `slots[s]` on `act[s]`; zeros for a dropped pair.
const DOWN: &str = r"
    uint lid=thread_index_in_simdgroup, sg=simdgroup_index_in_threadgroup;
    uint s=threadgroup_position_in_grid.y;
    uint row0=threadgroup_position_in_grid.x*DR;
    const device uchar* rec=RECORD;
    const device uint* wd=(const device uint*)(rec+DOFF);
    const device bfloat16_t* sd=(const device bfloat16_t*)(rec+DSOFF);
    const device bfloat16_t* bd=(const device bfloat16_t*)(rec+DBOFF);
    const device bfloat16_t* xv=act+(ulong)s*F;
    constexpr uint ROWW=F*BITS/32, PART=F/DKS;
    threadgroup float part[DKS][DR];
    float y[DR]={0};
    bool on=LIVE;
    for (uint k=sg*PART;on&&k<(sg+1)*PART;k+=512) {
        uint kk=k+lid*16;
        if (kk<(sg+1)*PART) {
            float xt[16], sum=0;
            for (int i=0;i<16;++i) { xt[i]=float(xv[kk+i]); sum+=xt[i]; }
            uint at=kk/16*W16, grp=kk/64;
            for (uint r=0;r<DR;++r) {
                uint row=row0+r, gi=row*(F/64)+grp;
                y[r]+=float(sd[gi])*aff_dot(wd+row*ROWW+at, xt)+float(bd[gi])*sum;
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
const UKS: usize = 2;
const DR: usize = 4;
const DKS: usize = 1;

/// The two kernels of one record layout (sizes and offsets compiled in, as [`crate::tern`]'s).
pub(crate) struct Kernels {
    up_gate: Kernel,
    down: Kernel,
    fast_up_gate: Kernel,
    fast_down: Kernel,
    rec: AffineRecord,
}

impl Kernels {
    pub(crate) fn new(rec: AffineRecord, stride: usize) -> Self {
        let consts = [
            ("H", rec.hidden),
            ("F", rec.ffn),
            ("REC", stride),
            ("BITS", rec.bits),
            ("W16", rec.bits / 2),
            ("EL", 32 / rec.bits),
            ("MASK", (1 << rec.bits) - 1),
            ("UOFF", rec.up()),
            ("DOFF", rec.down()),
            ("GSOFF", rec.gate_scales()),
            ("GBOFF", rec.gate_biases()),
            ("USOFF", rec.up_scales()),
            ("UBOFF", rec.up_biases()),
            ("DSOFF", rec.down_scales()),
            ("DBOFF", rec.down_biases()),
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
        let header = format!("{defs}{ADOT}");
        let slot = |src: &str| {
            src.replace("RECORD", "(const device uchar*)slab+(ulong)slots[s]*REC")
                .replace("LIVE", "live[s]!=0.0f")
                .replace("XROW", "x+(ulong)tok[s]*H")
        };
        // Row 8 is the shared expert; the slot of a choice comes from the layer's table.
        let fast = |src: &str| {
            src.replace("RECORD", "s<8?(const device uchar*)slab+(ulong)table[ids[s]]*REC:(const device uchar*)shared")
                .replace("LIVE", "s==8||live[s]!=0.0f")
                .replace("XROW", "x")
        };
        let up_gate = Kernel::new(
            "ironbark_aff_up_gate",
            &["x", "slab", "slots", "tok", "live"],
            &["out"],
            &slot(UP_GATE),
            &header,
        );
        let down = Kernel::new("ironbark_aff_down", &["act", "slab", "slots", "live"], &["out"], &slot(DOWN), &header);
        let fast_up_gate = Kernel::new(
            "ironbark_aff_fast_up_gate",
            &["x", "slab", "shared", "table", "ids", "live"],
            &["out"],
            &fast(UP_GATE),
            &header,
        );
        let fast_down = Kernel::new(
            "ironbark_aff_fast_down",
            &["act", "slab", "shared", "table", "ids", "live"],
            &["out"],
            &fast(DOWN),
            &header,
        );
        Kernels {
            up_gate,
            down,
            fast_up_gate,
            fast_down,
            rec,
        }
    }

    /// The 8 routed experts `ids` (`table[ids]` their slots in `slab`; skipped where `live` is 0)
    /// and the `shared` record on one row `x`: `[9, hidden]`.
    pub(crate) fn apply_with_shared(
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
            .fast_up_gate
            .apply(
                &[x, slab, shared, table, ids, live],
                &[],
                (f / UR as i32 * 32 * UKS as i32, 9, 1),
                (32 * UKS as i32, 1, 1),
                &[Output {
                    shape: &[9, f],
                    dtype: Dtype::Bfloat16,
                }],
            )?
            .remove(0);
        Ok(self
            .fast_down
            .apply(
                &[&act, slab, shared, table, ids, live],
                &[],
                (h / DR as i32 * 32 * DKS as i32, 9, 1),
                (32 * DKS as i32, 1, 1),
                &[Output {
                    shape: &[9, h],
                    dtype: Dtype::Bfloat16,
                }],
            )?
            .remove(0))
    }

    /// Experts in slots `slots` `[S]` (u32) of `slab` on rows `tok` `[S]` (u32) of `x`
    /// `[T, hidden]` (bf16), zeros where `live` `[S]` (f32) is 0: `[S, hidden]`.
    pub(crate) fn apply(&self, x: &Array, slab: &Array, slots: &Array, tok: &Array, live: &Array) -> Result<Array> {
        let (f, h) = (self.rec.ffn as i32, self.rec.hidden as i32);
        let n = slots.shape()[0];
        let x = x.as_dtype(Dtype::Bfloat16)?;
        let act = self
            .up_gate
            .apply(
                &[&x, slab, slots, tok, live],
                &[],
                (f / UR as i32 * 32 * UKS as i32, n, 1),
                (32 * UKS as i32, 1, 1),
                &[Output {
                    shape: &[n, f],
                    dtype: Dtype::Bfloat16,
                }],
            )?
            .remove(0);
        Ok(self
            .down
            .apply(
                &[&act, slab, slots, live],
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
    use ironbark_mlx::ops::indexing::IndexOp;

    use super::*;

    /// Deterministic pseudo-random numbers in [0, 1).
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (*seed >> 40) as f32 / (1u64 << 24) as f32
    }

    fn bf16(v: f32) -> f32 {
        half::bf16::from_f32(v).to_f32()
    }

    /// Against plain arithmetic: two experts in reversed slots and one dropped pair.
    fn kernels_match_plain_arithmetic(bits: usize) {
        let (h, f, ne) = (128usize, 64usize, 2usize);
        let rec = AffineRecord {
            hidden: h,
            ffn: f,
            bits,
        };
        let stride = rec.bytes().div_ceil(PAGE) * PAGE;
        let mut seed = 7u64 + bits as u64;
        let mut slab = vec![0u8; ne * stride];
        let mut mats = Vec::new();
        let levels = (1u32 << bits) as f32;
        let el = 32 / bits;
        for e in 0..ne {
            let mut m = Vec::new();
            for (rows, cols, words_at, scales_at, biases_at) in [
                (f, h, 0, rec.gate_scales(), rec.gate_biases()),
                (f, h, rec.up(), rec.up_scales(), rec.up_biases()),
                (h, f, rec.down(), rec.down_scales(), rec.down_biases()),
            ] {
                let codes: Vec<u32> = (0..rows * cols).map(|_| (lcg(&mut seed) * levels) as u32).collect();
                let scales: Vec<f32> = (0..rows * cols / GROUP).map(|_| bf16(0.002 + 0.01 * lcg(&mut seed))).collect();
                let biases: Vec<f32> = (0..rows * cols / GROUP).map(|_| bf16(-0.2 * lcg(&mut seed))).collect();
                let base = e * stride;
                for (i, c) in codes.chunks(el).enumerate() {
                    let word = c.iter().enumerate().fold(0u32, |w, (j, &v)| w | (v << (bits * j)));
                    slab[base + words_at + 4 * i..base + words_at + 4 * i + 4].copy_from_slice(&word.to_le_bytes());
                }
                for (at, vals) in [(scales_at, &scales), (biases_at, &biases)] {
                    for (i, a) in vals.iter().enumerate() {
                        let b = half::bf16::from_f32(*a).to_bits().to_le_bytes();
                        slab[base + at + 2 * i..base + at + 2 * i + 2].copy_from_slice(&b);
                    }
                }
                let w: Vec<f32> =
                    (0..rows * cols).map(|i| scales[i / GROUP] * codes[i] as f32 + biases[i / GROUP]).collect();
                m.push(w);
            }
            mats.push(m);
        }
        let x: Vec<f32> = (0..h).map(|_| bf16(lcg(&mut seed) - 0.5)).collect();
        let words: Vec<u32> = slab.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect();
        let array = Array::from_slice(&words, &[ne as i32, (stride / 4) as i32]);
        let kernels = Kernels::new(rec, stride);
        let slots = Array::from_slice(&[1u32, 0, 1], &[3]);
        let tok = Array::from_slice(&[0u32, 0, 0], &[3]);
        let live = Array::from_slice(&[1.0f32, 1.0, 0.0], &[3]);
        let xa = Array::from_slice(&x, &[1, h as i32]).as_dtype(Dtype::Bfloat16).unwrap();
        let got = kernels.apply(&xa, &array, &slots, &tok, &live).unwrap().as_dtype(Dtype::Float32).unwrap();
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
                assert!(
                    (g - want).abs() <= 2e-2 * want.abs().max(1e-2),
                    "{bits} bits, slot {s} row {n}: {g} vs {want}"
                );
            }
        }
        assert!(got[2 * h..3 * h].iter().all(|&v| v == 0.0), "a dropped pair must be zeros");

        // The fused path: same experts by id through the table, one dropped, row 8 the shared.
        let table = Array::from_slice(&[0u32, 1], &[2]);
        let ids = Array::from_slice(&[1u32, 0, 1, 1, 1, 1, 1, 1], &[8]);
        let scores = Array::from_slice(&[1.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[8]);
        let shared = array.index(1..2);
        let fused = kernels.apply_with_shared(&xa, &array, &shared, &table, &ids, &scores).unwrap();
        let fused = fused.as_dtype(Dtype::Float32).unwrap();
        let fused = fused.as_slice::<f32>();
        assert_eq!(fused[..2 * h], got[..2 * h], "{bits} bits: routed rows");
        assert_eq!(fused[8 * h..9 * h], got[..h], "{bits} bits: the shared row");
        assert!(fused[2 * h..8 * h].iter().all(|&v| v == 0.0), "dropped rows must be zeros");
    }

    #[test]
    fn four_bit_kernels_match_plain_arithmetic() {
        kernels_match_plain_arithmetic(4);
    }

    #[test]
    fn eight_bit_kernels_match_plain_arithmetic() {
        kernels_match_plain_arithmetic(8);
    }

    #[test]
    fn record_sizes_match_the_checkpoints() {
        let r = |bits| AffineRecord {
            hidden: 2048,
            ffn: 512,
            bits,
        };
        assert_eq!(r(4).bytes(), 1_769_472);
        assert_eq!(r(8).bytes(), 3_342_336);
        assert_eq!(bits_of(256, 32).unwrap(), 4);
        assert_eq!(bits_of(512, 32).unwrap(), 8);
        assert!(bits_of(128, 32).is_err());
    }
}
