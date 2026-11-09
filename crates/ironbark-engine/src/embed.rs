//! The token embedding in its own file, `embed.bin`, memory-mapped instead of loaded.
//!
//! A decode step reads one embedding row (~1 KiB) of the table (Maple: 151,936 rows, 175 MB;
//! Qwen3.6: 248,320 rows, 286 MB); the rest sits idle. Mapped from the file and wrapped as MLX arrays without a copy, rows are paged in
//! when first read and, being clean file-backed memory, can be dropped again by the system:
//! they do not count as the app's own memory.
//!
//! ```text
//!   embed.bin: [codes: R x H·bits/8][scales: R x H/group bf16][biases: R x H/group bf16]
//!   R = vocab rounded up so every region is whole 16 KiB pages (extra rows are zero)
//!
//! The file may be sparse, filling from a server ([`ironbark_experts::Chunks`], e.g.
//! `ironbark-fetch`): the model then makes the chunks of each token's row present before the
//! GPU reads them (`Model::set_embed_chunks`).
//! ```

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use ironbark_mlx::{Array, Dtype};

use crate::kernel::Mapped;
use crate::model::QMat;
use crate::pack::PAGE;
use crate::{Config, Error, Result};

/// An embedding table's size and affine quantization: what fixes the layout of `embed.bin`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Shape {
    /// Rows (vocab).
    pub(crate) vocab: usize,
    /// Width (hidden size).
    pub(crate) hidden: usize,
    pub(crate) bits: usize,
    pub(crate) group: usize,
}

impl Shape {
    /// Maple's embedding: quantized like its LM head.
    pub(crate) fn maple(cfg: &Config) -> Self {
        Shape {
            vocab: cfg.vocab,
            hidden: cfg.hidden,
            bits: cfg.head_bits,
            group: cfg.head_group,
        }
    }
}

/// File name in a packed model.
pub(crate) const FILE: &str = "embed.bin";

/// Bytes per chunk of an `embed.bin` of `bytes` in the manifest: the unit a sparse copy
/// fetches and checks (`ironbark-fetch`). 64 KiB, or the largest smaller power of two down to
/// a page that divides the file (always a page: regions are whole pages).
pub(crate) fn chunk(bytes: usize) -> usize {
    [64 << 10, 32 << 10, PAGE].into_iter().find(|c| bytes.is_multiple_of(*c)).unwrap_or(PAGE)
}

/// Byte layout of `embed.bin` for a [`Shape`].
#[derive(Debug, Clone, Copy)]
struct Layout {
    /// Rows stored (vocab, padded).
    rows: usize,
    /// Bytes of a row of codes.
    code_row: usize,
    /// Bytes of a row of scales (and of biases).
    scale_row: usize,
}

impl Layout {
    fn new(s: Shape) -> Self {
        let code_row = s.hidden * s.bits / 8;
        let scale_row = s.hidden / s.group * 2;
        // Rows per whole page of the smaller region (row sizes are powers of two here).
        let step = [code_row, scale_row].iter().map(|&b| PAGE / gcd(PAGE, b)).max().unwrap_or(1);
        Layout {
            rows: s.vocab.div_ceil(step) * step,
            code_row,
            scale_row,
        }
    }

    fn scales_at(&self) -> usize {
        self.rows * self.code_row
    }

    fn biases_at(&self) -> usize {
        self.scales_at() + self.rows * self.scale_row
    }

    fn bytes(&self) -> usize {
        self.biases_at() + self.rows * self.scale_row
    }
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |source| Error::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Raw bytes of an evaluated array.
fn bytes(a: &Array) -> Result<Vec<u8>> {
    let a = a.contiguous()?;
    a.eval()?;
    Ok(match a.dtype() {
        Dtype::Uint32 => a.as_slice::<u32>().iter().flat_map(|w| w.to_le_bytes()).collect(),
        Dtype::Bfloat16 => a.as_slice::<half::bf16>().iter().flat_map(|w| w.to_bits().to_le_bytes()).collect(),
        d => return Err(Error::Format(format!("embedding: unexpected dtype {d:?}"))),
    })
}

/// Write the embedding `e` (MLX affine layout, of shape `s`) to `dir/embed.bin`.
pub(crate) fn write(dir: &Path, s: Shape, e: &QMat) -> Result<()> {
    if (e.bits as usize, e.group as usize) != (s.bits, s.group) {
        return Err(Error::Format(format!(
            "embedding: {}-bit groups of {}, expected {}-bit groups of {}",
            e.bits, e.group, s.bits, s.group
        )));
    }
    let l = Layout::new(s);
    let path = dir.join(FILE);
    let file = File::create(&path).map_err(io(&path))?;
    file.set_len(l.bytes() as u64).map_err(io(&path))?;
    for (a, at, row) in [
        (&e.w, 0, l.code_row),
        (&e.scales, l.scales_at(), l.scale_row),
        (&e.biases, l.biases_at(), l.scale_row),
    ] {
        let b = bytes(a)?;
        if b.len() != s.vocab * row {
            return Err(Error::Format(format!("embedding: {} bytes, expected {}", b.len(), s.vocab * row)));
        }
        file.write_all_at(&b, at as u64).map_err(io(&path))?;
    }
    Ok(())
}

/// The chunks of `embed.bin` that hold row `row`: its codes, scales and biases.
pub(crate) fn chunks_of(s: Shape, row: usize) -> [usize; 3] {
    let l = Layout::new(s);
    let c = chunk(l.bytes());
    [
        row * l.code_row / c,
        (l.scales_at() + row * l.scale_row) / c,
        (l.biases_at() + row * l.scale_row) / c,
    ]
}

/// The embedding of the packed model in `dir`, mapped from `embed.bin`; `None` without one.
pub(crate) fn map(dir: &Path, s: Shape) -> Result<Option<QMat>> {
    let path = dir.join(FILE);
    if !path.exists() {
        return Ok(None);
    }
    let l = Layout::new(s);
    let m = Mapped::open(&path)?;
    if m.len() != l.bytes() {
        return Err(Error::Format(format!("{}: {} bytes, expected {}", path.display(), m.len(), l.bytes())));
    }
    let (rows, h) = (l.rows as i32, s.hidden as i32);
    let groups = (s.hidden / s.group) as i32;
    Ok(Some(QMat {
        w: m.array(0, &[rows, h * s.bits as i32 / 32], Dtype::Uint32)?,
        scales: m.array(l.scales_at(), &[rows, groups], Dtype::Bfloat16)?,
        biases: m.array(l.biases_at(), &[rows, groups], Dtype::Bfloat16)?,
        group: s.group as i32,
        bits: s.bits as i32,
    }))
}

#[cfg(test)]
mod tests {
    use ironbark_mlx::ops::dequantize;

    use super::*;

    #[test]
    fn mapped_rows_match_the_table() {
        let (vocab, hidden) = (1000usize, 2048usize);
        let cfg = Config {
            layers: 1,
            hidden,
            heads: 16,
            kv_heads: 4,
            head_dim: 128,
            rope_dims: 64,
            theta: 1e4,
            experts: 256,
            top_k: 8,
            ffn: 512,
            vocab,
            eps: 1e-6,
            window: 512,
            sliding: vec![true],
            group: 128,
            head_bits: 4,
            head_group: 64,
            flash_head: None,
            dynamic: crate::dynamic::Dynamic::default(),
        };
        let v: Vec<f32> = (0..vocab * hidden).map(|i| ((i * 7919 % 2003) as f32 - 1000.0) * 1e-3).collect();
        let table = Array::from_slice(&v, &[vocab as i32, hidden as i32]).as_dtype(Dtype::Bfloat16).unwrap();
        let (w, scales, biases) = ironbark_mlx::ops::quantize(&table, 64, 4).unwrap();
        let e = QMat {
            w,
            scales,
            biases,
            group: 64,
            bits: 4,
        };
        let dir = std::env::temp_dir().join(format!("ironbark-embed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        write(&dir, Shape::maple(&cfg), &e).unwrap();
        // Fails if MLX had to copy the mapping.
        let m = map(&dir, Shape::maple(&cfg)).unwrap().unwrap();
        let ids = Array::from_slice(&[0u32, 999, 17, 512], &[4]);
        let rows = |q: &QMat| {
            let r = dequantize(
                q.w.take_axis(&ids, 0).unwrap(),
                q.scales.take_axis(&ids, 0).unwrap(),
                &q.biases.take_axis(&ids, 0).unwrap(),
                64,
                4,
            )
            .unwrap()
            .as_dtype(Dtype::Float32)
            .unwrap();
            r.eval().unwrap();
            r.as_slice::<f32>().to_vec()
        };
        assert_eq!(rows(&m), rows(&e));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
