//! Qwen3.5 / 3.6's vision encoder: an image (RGB) → embeddings the language model reads in
//! place of its `<|image_pad|>` tokens, as transformers' `Qwen3_5MoeVisionModel` and
//! `Qwen2VLImageProcessorFast` compute them.
//!
//! ```text
//!   RGB ─► resize (sides multiples of 32, bicubic) ─► (x/255 − 0.5)/0.5
//!       ─► 16×16 patches, 2×2 blocks of them adjacent, each patch twice (temporal)  [N, 1536]
//!       ─► patch_embed ─► + learned positions (48×48 grid, bilinear to the image's grid)
//!       ─► 27 × { x += attn(LN(x)) with 2D RoPE (rows, cols);  x += MLP(LN(x)) }
//!       ─► merger: LN, each 2×2 block's 4 patches as one row, MLP ─► [N/4, 2048]
//! ```
//!
//! The language model places the image's tokens with 3D positions ([`positions`]): text
//! tokens count on as usual (all three axes equal), an image's tokens share a temporal
//! position and take rows and columns from it, and the text after the image goes on from
//! the image's start plus its longer side.
//!
//! A checkpoint's vision weights are bf16; [`pack`] writes them 4-bit (`vision.safetensors`
//! in a pack), patch embedding, positions, norms and biases kept bf16.

use std::collections::HashMap;
use std::path::Path;

use ironbark_mlx::ops::indexing::IndexOp;
use ironbark_mlx::ops::{concatenate, erf, quantize, tanh};
use ironbark_mlx::{Array, Dtype, fast};

use super::deltanet::Linear;
use crate::model::load_tensors;
use crate::quant::Tensors;
use crate::{Error, Result};

/// The vision weights' file in a pack.
pub const FILE: &str = "vision.safetensors";
/// Tensor name prefix (mlx-vlm's checkpoint layout).
const PREFIX: &str = "vision_tower";
/// Bits of the vision matrices `ironbark pack` writes.
pub const PACK_BITS: i32 = 6;
/// Group size of quantized vision matrices.
const GROUP: i32 = 64;

/// The encoder's shapes (`config.json` `vision_config`).
#[derive(Debug, Clone, Copy)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub heads: usize,
    pub patch: usize,
    /// Frames per patch (an image is repeated to fill them).
    pub temporal: usize,
    /// Patches per side merged into one token.
    pub merge: usize,
    /// The language model's hidden size.
    pub out: usize,
    /// Side of the learned position grid.
    pub side: usize,
    pub channels: usize,
}

impl VisionConfig {
    /// The `vision_config` of `dir/config.json`, or `None` for a text-only model.
    ///
    /// # Errors
    ///
    /// Fails if the file is unreadable or a field is missing.
    pub fn read(dir: &Path) -> Result<Option<Self>> {
        let path = dir.join("config.json");
        let raw = std::fs::read(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let root: serde_json::Value =
            serde_json::from_slice(&raw).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
        let c = &root["vision_config"];
        if !c.is_object() {
            return Ok(None);
        }
        let u =
            |k: &str| c[k].as_u64().map(|v| v as usize).ok_or_else(|| Error::Format(format!("vision_config: no {k}")));
        let positions = u("num_position_embeddings")?;
        let side = positions.isqrt();
        if side * side != positions {
            return Err(Error::Format("vision_config: num_position_embeddings is not a square".into()));
        }
        Ok(Some(VisionConfig {
            depth: u("depth")?,
            hidden: u("hidden_size")?,
            heads: u("num_heads")?,
            patch: u("patch_size")?,
            temporal: u("temporal_patch_size")?,
            merge: u("spatial_merge_size")?,
            out: u("out_hidden_size")?,
            side,
            channels: u("in_channels")?,
        }))
    }

    /// Image sides are multiples of this.
    #[must_use]
    pub const fn factor(&self) -> usize {
        self.patch * self.merge
    }

    /// Values of one patch: channels × frames × patch × patch.
    #[must_use]
    pub const fn patch_dim(&self) -> usize {
        self.channels * self.temporal * self.patch * self.patch
    }

    const fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }
}

/// An image as the encoder reads it: patches `[N, patch_dim]` (fp32, normalized, 2×2 blocks
/// adjacent) on a grid of `rows` × `cols` patches.
#[derive(Debug, Clone)]
pub struct Pixels {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

/// An encoded image: embeddings `[tokens, hidden]` on a grid of `rows` × `cols` tokens (after
/// merging).
#[derive(Debug, Clone)]
pub struct Image {
    pub embeds: Array,
    pub rows: usize,
    pub cols: usize,
}

impl Image {
    /// Tokens the image takes in the prompt.
    #[must_use]
    pub const fn tokens(&self) -> usize {
        self.rows * self.cols
    }
}

/// Fewest pixels an image is scaled up to (`preprocessor_config.json` `shortest_edge`).
pub const MIN_PIXELS: usize = 65_536;
/// Most pixels the checkpoint's processor allows (`longest_edge`); devices pass less.
pub const MAX_PIXELS: usize = 16_777_216;

/// The size an image of `height` × `width` is resized to: sides multiples of `factor`, pixels
/// within `[min, max]`, close to the aspect ratio (`smart_resize`).
///
/// # Errors
///
/// Fails for an empty image or one more than 200 times longer than wide.
pub fn target_size(height: usize, width: usize, factor: usize, min: usize, max: usize) -> Result<(usize, usize)> {
    if height == 0 || width == 0 || height.max(width) > 200 * height.min(width) {
        return Err(Error::Format(format!("image {width}×{height}: empty or too narrow")));
    }
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    let round = |x: f64| ((x / f).round_ties_even() * f) as usize;
    let (mut hb, mut wb) = (round(h), round(w));
    if hb * wb > max {
        let beta = (h * w / max as f64).sqrt();
        hb = factor.max(((h / beta / f).floor() * f) as usize);
        wb = factor.max(((w / beta / f).floor() * f) as usize);
    } else if hb * wb < min {
        let beta = (min as f64 / (h * w)).sqrt();
        hb = ((h * beta / f).ceil() * f) as usize;
        wb = ((w * beta / f).ceil() * f) as usize;
    }
    Ok((hb, wb))
}

/// Resize, normalize and cut an RGB image (`rgb`: `height` × `width` × 3 bytes, rows top to
/// bottom) into the encoder's patches, at most `max_pixels` after resizing.
///
/// # Errors
///
/// Fails if `rgb` is not `height · width · 3` bytes or the size is not usable.
pub fn preprocess(c: &VisionConfig, rgb: &[u8], width: usize, height: usize, max_pixels: usize) -> Result<Pixels> {
    if rgb.len() != width * height * 3 || c.channels != 3 {
        return Err(Error::Format(format!("image: {} bytes for {width}×{height} RGB", rgb.len())));
    }
    let (th, tw) = target_size(height, width, c.factor(), MIN_PIXELS, max_pixels.max(MIN_PIXELS))?;
    let img = resize(rgb, width, height, tw, th);
    let (p, m, t) = (c.patch, c.merge, c.temporal);
    let (rows, cols) = (th / p, tw / p);
    let mut data = Vec::with_capacity(rows * cols * c.patch_dim());
    // Patch order: block row, block column, row in block, column in block; values in a patch:
    // channel, frame (the image again), pixel row, pixel column.
    for br in 0..rows / m {
        for bc in 0..cols / m {
            for ir in 0..m {
                for ic in 0..m {
                    let (r0, c0) = ((br * m + ir) * p, (bc * m + ic) * p);
                    for ch in 0..3 {
                        for _ in 0..t {
                            for y in 0..p {
                                for x in 0..p {
                                    let v = f32::from(img[((r0 + y) * tw + c0 + x) * 3 + ch]);
                                    data.push((v / 255.0 - 0.5) / 0.5);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(Pixels { data, rows, cols })
}

/// Bicubic resampling with antialiasing (PIL's, which torchvision matches for `uint8`): the
/// Keys kernel with `a = −0.5`, widened by the scale when shrinking; rows first, then columns,
/// rounded to bytes after each pass. In PIL's 8-bit fixed point (weights in 22 fractional bits,
/// integer sums), so the bytes match PIL's exactly.
pub(crate) fn resize(rgb: &[u8], w: usize, h: usize, tw: usize, th: usize) -> Vec<u8> {
    let horizontal = resample_axis(rgb, w, h, tw, true);
    resample_axis(&horizontal, tw, h, th, false)
}

/// Fractional bits of PIL's 8-bit resampling weights.
const PRECISION: u32 = 22;

/// One pass of [`resize`] along x (`along_x`) or y: `src` is `w` × `h`, the result `n` long on
/// that axis.
fn resample_axis(src: &[u8], w: usize, h: usize, n: usize, along_x: bool) -> Vec<u8> {
    let len = if along_x { w } else { h };
    // PIL's `normalize_coeffs_8bpc`: each weight in PRECISION fractional bits, rounded away
    // from zero.
    let one = f64::from(1u32 << PRECISION);
    let taps: Vec<(usize, Vec<i64>)> = filter_taps(len, n)
        .into_iter()
        .map(|(start, ws)| {
            (
                start,
                ws.iter()
                    .map(|&k| {
                        if k < 0.0 {
                            (k * one - 0.5) as i64
                        } else {
                            (k * one + 0.5) as i64
                        }
                    })
                    .collect(),
            )
        })
        .collect();
    let (ow, oh) = if along_x { (n, h) } else { (w, n) };
    let mut out = vec![0u8; ow * oh * 3];
    for y in 0..oh {
        for x in 0..ow {
            let (start, weights) = if along_x { &taps[x] } else { &taps[y] };
            for ch in 0..3 {
                let mut acc = 1i64 << (PRECISION - 1);
                for (k, &wt) in weights.iter().enumerate() {
                    let (sx, sy) = if along_x { (*start + k, y) } else { (x, *start + k) };
                    acc += wt * i64::from(src[(sy * w + sx) * 3 + ch]);
                }
                out[(y * ow + x) * 3 + ch] = (acc >> PRECISION).clamp(0, 255) as u8;
            }
        }
    }
    out
}

/// For each of `n` outputs resampled from `len` inputs: the first input it reads and the
/// normalized weights (PIL's `precompute_coeffs`, bicubic `a = −0.5`).
fn filter_taps(len: usize, n: usize) -> Vec<(usize, Vec<f64>)> {
    let cubic = |x: f64| {
        let (a, x) = (-0.5, x.abs());
        if x < 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            (((x - 5.0) * x + 8.0) * x - 4.0) * a
        } else {
            0.0
        }
    };
    let scale = len as f64 / n as f64;
    let fscale = scale.max(1.0);
    let support = 2.0 * fscale;
    (0..n)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support + 0.5).floor().max(0.0)) as usize;
            let hi = ((center + support + 0.5).floor() as usize).min(len);
            let mut w: Vec<f64> = (lo..hi).map(|j| cubic((j as f64 - center + 0.5) / fscale)).collect();
            let total: f64 = w.iter().sum();
            if total != 0.0 {
                for x in &mut w {
                    *x /= total;
                }
            }
            (lo, w)
        })
        .collect()
}

/// A linear layer with its bias.
struct Dense {
    w: Linear,
    b: Array,
}

impl Dense {
    fn load(t: &mut Tensors, prefix: &str) -> Result<Self> {
        Ok(Dense {
            w: Linear::load(t, prefix, GROUP as usize)?,
            b: t.take(&format!("{prefix}.bias"))?,
        })
    }

    fn apply(&self, x: &Array) -> Result<Array> {
        Ok(self.w.apply(x)?.add(&self.b)?)
    }
}

/// Layer norm weight and bias.
struct Norm {
    w: Array,
    b: Array,
}

impl Norm {
    fn load(t: &mut Tensors, prefix: &str) -> Result<Self> {
        Ok(Norm {
            w: t.take(&format!("{prefix}.weight"))?,
            b: t.take(&format!("{prefix}.bias"))?,
        })
    }

    fn apply(&self, x: &Array) -> Result<Array> {
        Ok(fast::layer_norm(x, &self.w, &self.b, 1e-6)?)
    }
}

struct Block {
    norm1: Norm,
    norm2: Norm,
    qkv: Dense,
    proj: Dense,
    fc1: Dense,
    fc2: Dense,
}

/// The vision encoder on the GPU.
pub struct Vision {
    pub cfg: VisionConfig,
    /// Patch embedding as a matrix `[hidden, patch_dim]` (values in [`Pixels`] order).
    patch: Array,
    patch_b: Array,
    /// Learned positions `[side², hidden]`.
    pos: Array,
    blocks: Vec<Block>,
    merge_norm: Norm,
    merge_fc1: Dense,
    merge_fc2: Dense,
}

impl std::fmt::Debug for Vision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Vision").field("cfg", &self.cfg).finish_non_exhaustive()
    }
}

impl Vision {
    /// Load the encoder of `dir`: a pack's [`FILE`], else the checkpoint's own weights.
    /// `None` if the model has no vision config or the pack no vision weights.
    ///
    /// # Errors
    ///
    /// Fails if a file is unreadable or a tensor is missing.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let Some(cfg) = VisionConfig::read(dir)? else {
            return Ok(None);
        };
        let file = dir.join(FILE);
        let mut all = if file.exists() {
            Array::load_safetensors(&file)
                .map_err(|e| Error::Load {
                    path: file.clone(),
                    msg: e.to_string(),
                })?
                .into_iter()
                .collect()
        } else if dir.join("experts.json").exists() {
            // A pack without vision weights.
            return Ok(None);
        } else {
            load_tensors(dir)?
        };
        all.retain(|k, _| k.starts_with(PREFIX));
        if all.is_empty() {
            return Ok(None);
        }
        let mut t = Tensors(all);
        let n = |s: &str| format!("{PREFIX}.{s}");
        let patch = patch_matrix(&t.take(&n("patch_embed.proj.weight"))?, &cfg)?;
        let patch_b = t.take(&n("patch_embed.proj.bias"))?;
        let pos = t.take(&n("pos_embed.weight"))?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let b = |s: &str| n(&format!("blocks.{i}.{s}"));
            blocks.push(Block {
                norm1: Norm::load(&mut t, &b("norm1"))?,
                norm2: Norm::load(&mut t, &b("norm2"))?,
                qkv: Dense::load(&mut t, &b("attn.qkv"))?,
                proj: Dense::load(&mut t, &b("attn.proj"))?,
                fc1: Dense::load(&mut t, &b("mlp.linear_fc1"))?,
                fc2: Dense::load(&mut t, &b("mlp.linear_fc2"))?,
            });
        }
        let v = Vision {
            cfg,
            patch,
            patch_b,
            pos,
            blocks,
            merge_norm: Norm::load(&mut t, &n("merger.norm"))?,
            merge_fc1: Dense::load(&mut t, &n("merger.linear_fc1"))?,
            merge_fc2: Dense::load(&mut t, &n("merger.linear_fc2"))?,
        };
        ironbark_mlx::transforms::eval(v.arrays())?;
        Ok(Some(v))
    }

    fn arrays(&self) -> Vec<&Array> {
        let mut a = vec![&self.patch, &self.patch_b, &self.pos];
        let mut dense = vec![&self.merge_fc1, &self.merge_fc2];
        let mut norms = vec![&self.merge_norm];
        for b in &self.blocks {
            dense.extend([&b.qkv, &b.proj, &b.fc1, &b.fc2]);
            norms.extend([&b.norm1, &b.norm2]);
        }
        for d in dense {
            a.push(&d.b);
            match &d.w {
                Linear::Quant(m) => a.extend([&m.w, &m.scales, &m.biases]),
                Linear::Dense(w) => a.push(w),
            }
        }
        for n in norms {
            a.extend([&n.w, &n.b]);
        }
        a
    }

    /// The dense weights as `dt` (to compare with an fp32 reference).
    #[cfg(test)]
    fn cast(&mut self, dt: Dtype) {
        let c = |a: &mut Array| *a = a.as_dtype(dt).unwrap();
        c(&mut self.patch);
        c(&mut self.patch_b);
        c(&mut self.pos);
        let mut dense = vec![&mut self.merge_fc1, &mut self.merge_fc2];
        let mut norms = vec![&mut self.merge_norm];
        for b in &mut self.blocks {
            dense.extend([&mut b.qkv, &mut b.proj, &mut b.fc1, &mut b.fc2]);
            norms.extend([&mut b.norm1, &mut b.norm2]);
        }
        for d in dense {
            c(&mut d.b);
            if let Linear::Dense(w) = &mut d.w {
                c(w);
            }
        }
        for n in norms {
            c(&mut n.w);
            c(&mut n.b);
        }
    }

    /// Encode `px` → its embeddings for the language model.
    ///
    /// # Errors
    ///
    /// Fails if `px` does not match the config or an MLX operation fails.
    pub fn encode(&self, px: &Pixels) -> Result<Image> {
        let (out, _) = self.encode_traced(px)?;
        Ok(out)
    }

    /// [`Vision::encode`], also returning the encoder's output before the merger `[N, hidden]`.
    fn encode_traced(&self, px: &Pixels) -> Result<(Image, Array)> {
        let c = &self.cfg;
        let (m, n) = (c.merge, px.rows * px.cols);
        if !px.rows.is_multiple_of(m) || !px.cols.is_multiple_of(m) || px.data.len() != n * c.patch_dim() {
            return Err(Error::Format("vision: pixels do not match the config".into()));
        }
        let dt = self.patch.dtype();
        let x = Array::from_slice(&px.data, &[n as i32, c.patch_dim() as i32]).as_dtype(dt)?;
        let mut h = x.matmul(self.patch.transpose()?)?.add(&self.patch_b)?;
        let (idx, wts) = interpolation(c.side, px.rows, px.cols, m);
        let taps = self.pos.take_axis(Array::from_slice(&idx, &[(n * 4) as i32]), 0)?.as_dtype(Dtype::Float32)?;
        let taps = taps.reshape(&[n as i32, 4, c.hidden as i32])?;
        let wts = Array::from_slice(&wts, &[n as i32, 4, 1]);
        h = h.add(taps.multiply(&wts)?.sum_axis(1, false)?.as_dtype(dt)?)?;
        let (cos, sin) = rotary(c, px.rows, px.cols);
        let (cos, sin) = (
            Array::from_slice(&cos, &[n as i32, 1, c.head_dim() as i32]),
            Array::from_slice(&sin, &[n as i32, 1, c.head_dim() as i32]),
        );
        for b in &self.blocks {
            h = h.add(self.attention(b, &b.norm1.apply(&h)?, &cos, &sin)?)?;
            let y = b.fc1.apply(&b.norm2.apply(&h)?)?;
            h = h.add(b.fc2.apply(&gelu_tanh(&y)?)?)?;
        }
        let y = self.merge_norm.apply(&h)?.reshape(&[(n / (m * m)) as i32, -1])?;
        let y = self.merge_fc2.apply(&gelu(&self.merge_fc1.apply(&y)?)?)?;
        y.eval()?;
        let img = Image {
            embeds: y,
            rows: px.rows / m,
            cols: px.cols / m,
        };
        Ok((img, h))
    }

    /// Full (non-causal) attention over the patches `x` `[N, hidden]`, 2D RoPE on q and k.
    fn attention(&self, b: &Block, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        let c = &self.cfg;
        let n = x.shape()[0];
        let (nh, d) = (c.heads as i32, c.head_dim() as i32);
        let qkv = b.qkv.apply(x)?.reshape(&[n, 3, nh, d])?;
        let part = |i: i32| qkv.index((.., i));
        let rope = |t: &Array| -> Result<Array> {
            let tf = t.as_dtype(Dtype::Float32)?;
            let half = d / 2;
            let rot = concatenate(&[tf.index((.., .., half..)).negative()?, tf.index((.., .., ..half))], -1)?;
            Ok(tf.multiply(cos)?.add(rot.multiply(sin)?)?.as_dtype(t.dtype())?)
        };
        let heads = |t: Array| -> Result<Array> { Ok(t.transpose_axes(&[1, 0, 2])?.expand_dims(0)?) };
        let q = heads(rope(&part(0))?)?;
        let k = heads(rope(&part(1))?)?;
        let v = heads(part(2))?;
        let o = fast::scaled_dot_product_attention(&q, &k, &v, (d as f32).powf(-0.5), None, None)?;
        let o = o.squeeze_axes(&[0])?.transpose_axes(&[1, 0, 2])?.reshape(&[n, nh * d])?;
        b.proj.apply(&o)
    }
}

/// The checkpoint's patch convolution as a matrix in [`Pixels`] value order (channel, frame,
/// row, column): from mlx-vlm's `[hidden, frames, p, p, channels]` or a flat `[hidden, dim]`.
fn patch_matrix(w: &Array, c: &VisionConfig) -> Result<Array> {
    let s = w.shape();
    let dim = c.patch_dim() as i32;
    Ok(match s.len() {
        5 => w.transpose_axes(&[0, 4, 1, 2, 3])?.reshape(&[s[0], dim])?,
        2 if s[1] == dim => w.clone(),
        _ => return Err(Error::Format(format!("vision: patch embedding of shape {s:?}"))),
    })
}

/// The patch's row and column for each patch in [`Pixels`] order.
fn patch_coords(rows: usize, cols: usize, m: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..rows / m).flat_map(move |br| {
        (0..cols / m).flat_map(move |bc| (0..m).flat_map(move |ir| (0..m).map(move |ic| (br * m + ir, bc * m + ic))))
    })
}

/// Bilinear (corners aligned) taps into the `side` × `side` learned positions for each
/// patch: 4 indices and weights per patch.
fn interpolation(side: usize, rows: usize, cols: usize, m: usize) -> (Vec<u32>, Vec<f32>) {
    let axis = |i: usize, size: usize| -> ([usize; 2], [f32; 2]) {
        let src = i as f32 * (side - 1) as f32 / (size.max(2) - 1) as f32;
        let f = src.floor();
        let t = src - f;
        let lo = (f as usize).min(side - 1);
        ([lo, (lo + 1).min(side - 1)], [1.0 - t, t])
    };
    let (mut idx, mut wts) = (Vec::new(), Vec::new());
    for (r, c) in patch_coords(rows, cols, m) {
        let ((hi, hw), (wi, ww)) = (axis(r, rows), axis(c, cols));
        for a in 0..2 {
            for b in 0..2 {
                idx.push((hi[a] * side + wi[b]) as u32);
                wts.push(hw[a] * ww[b]);
            }
        }
    }
    (idx, wts)
}

/// 2D RoPE tables for the patches `[N · head_dim]`: a quarter of the frequencies on the row,
/// a quarter on the column, the whole repeated so it spans the head.
fn rotary(c: &VisionConfig, rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let d = c.head_dim();
    let spatial = d / 2;
    let inv: Vec<f32> = (0..spatial / 2).map(|i| 1.0 / 10_000f32.powf((2 * i) as f32 / spatial as f32)).collect();
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for (r, col) in patch_coords(rows, cols, c.merge) {
        let f: Vec<f32> = inv.iter().map(|&w| r as f32 * w).chain(inv.iter().map(|&w| col as f32 * w)).collect();
        for _ in 0..2 {
            cos.extend(f.iter().map(|x| x.cos()));
            sin.extend(f.iter().map(|x| x.sin()));
        }
    }
    (cos, sin)
}

/// `gelu_pytorch_tanh`.
fn gelu_tanh(x: &Array) -> Result<Array> {
    let s = |v: f32| Array::from_f32(v).as_dtype(x.dtype());
    let inner = x
        .add(x.multiply(x)?.multiply(x)?.multiply(s(0.044_715)?)?)?
        .multiply(s((2.0 / std::f32::consts::PI).sqrt())?)?;
    Ok(x.multiply(s(0.5)?)?.multiply(s(1.0)?.add(tanh(&inner)?)?)?)
}

/// Exact GELU: `x · (1 + erf(x / √2)) / 2`.
fn gelu(x: &Array) -> Result<Array> {
    let s = |v: f32| Array::from_f32(v).as_dtype(x.dtype());
    Ok(x.multiply(s(0.5)?)?
        .multiply(s(1.0)?.add(erf(x.multiply(s(std::f32::consts::FRAC_1_SQRT_2)?)?)?)?)?)
}

/// The language model's 3D positions (temporal, row, column) for `ids`, starting at `start`:
/// text tokens count on; each run of `image` tokens is the next of `images` (its token grid),
/// laid out row by row. Returns the positions `[3][L]` and where the text after `ids` goes on.
///
/// # Errors
///
/// Fails if the image tokens do not match `images`.
pub fn positions(ids: &[u32], image: u32, images: &[(usize, usize)], start: i64) -> Result<([Vec<i64>; 3], i64)> {
    let mut p: [Vec<i64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let (mut cur, mut i, mut next) = (start, 0, images.iter());
    while i < ids.len() {
        if ids[i] != image {
            for a in &mut p {
                a.push(cur);
            }
            cur += 1;
            i += 1;
            continue;
        }
        let &(rows, cols) = next.next().ok_or_else(|| Error::Format("image tokens without an image".into()))?;
        let run = ids[i..].iter().take_while(|&&t| t == image).count();
        if run < rows * cols {
            return Err(Error::Format(format!("image of {} tokens has {run} in the prompt", rows * cols)));
        }
        for r in 0..rows {
            for c in 0..cols {
                p[0].push(cur);
                p[1].push(cur + r as i64);
                p[2].push(cur + c as i64);
            }
        }
        cur += rows.max(cols) as i64;
        i += rows * cols;
    }
    if next.next().is_some() {
        return Err(Error::Format("more images than image tokens".into()));
    }
    Ok((p, cur))
}

/// The text model's RoPE angles at 3D positions `p` (`[3][L]`): `rotary / 2` frequencies of
/// `theta`, interleaved over the axes (`section`: how many go to temporal, row and column;
/// frequency `i` is row if `i % 3 == 1` within `3 · section[1]`, column if `i % 3 == 2` within
/// `3 · section[2]`, else temporal). Returns cos and sin `[L · rotary / 2]`.
#[must_use]
pub fn mrope_tables(p: &[Vec<i64>; 3], rotary: usize, theta: f32, section: [usize; 3]) -> (Vec<f32>, Vec<f32>) {
    let half = rotary / 2;
    let axis = |i: usize| {
        if i % 3 == 1 && i < 3 * section[1] {
            1
        } else if i % 3 == 2 && i < 3 * section[2] {
            2
        } else {
            0
        }
    };
    let inv: Vec<f32> = (0..half).map(|i| 1.0 / theta.powf((2 * i) as f32 / rotary as f32)).collect();
    let (mut cos, mut sin) = (Vec::new(), Vec::new());
    for l in 0..p[0].len() {
        for (i, &w) in inv.iter().enumerate() {
            let a = p.get(axis(i)).map_or(0, |a| a[l]) as f32 * w;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    (cos, sin)
}

/// Write the vision weights of the checkpoint `model` to `out/`[`FILE`]: matrices 4-bit (an
/// MLP whose width is not a multiple of the group is padded with zero units, which add
/// nothing), the rest as they are. Returns the bytes written, or `None` for a text-only model.
///
/// # Errors
///
/// Fails if a tensor is missing or the file cannot be written.
pub fn pack(model: &Path, out: &Path, bits: i32) -> Result<Option<u64>> {
    let Some(cfg) = VisionConfig::read(model)? else {
        return Ok(None);
    };
    let mut t = load_tensors(model)?;
    t.retain(|k, _| k.starts_with(PREFIX));
    if t.is_empty() {
        return Ok(None);
    }
    let mut w: HashMap<String, Array> = HashMap::new();
    let quant = |w: &mut HashMap<String, Array>, name: &str, m: &Array| -> Result<()> {
        let (q, s, b) = quantize(m, GROUP, bits)?;
        w.insert(format!("{name}.weight"), q);
        w.insert(format!("{name}.scales"), s);
        w.insert(format!("{name}.biases"), b);
        Ok(())
    };
    let pad_to = |n: i32| (n + GROUP - 1) / GROUP * GROUP;
    let mut names: Vec<String> = t.keys().cloned().collect();
    names.sort();
    for k in names {
        let Some(a) = t.remove(&k) else { continue };
        let base = k.trim_end_matches(".weight");
        let matrix = k.ends_with(".weight") && a.ndim() == 2 && !k.contains("pos_embed");
        if k.ends_with("patch_embed.proj.weight") {
            w.insert(k, patch_matrix(&a, &cfg)?);
        } else if matrix && k.ends_with("mlp.linear_fc1.weight") {
            // Pad the MLP's width (rows of fc1, its bias, columns of fc2) to the group.
            let rows = a.shape()[0];
            let a = ironbark_mlx::ops::pad(&a, &[(0, pad_to(rows) - rows), (0, 0)][..], None, None)?;
            quant(&mut w, base, &a)?;
        } else if matrix && k.ends_with("mlp.linear_fc2.weight") {
            let cols = a.shape()[1];
            let a = ironbark_mlx::ops::pad(&a, &[(0, 0), (0, pad_to(cols) - cols)][..], None, None)?;
            quant(&mut w, base, &a)?;
        } else if k.ends_with("mlp.linear_fc1.bias") {
            let n = a.shape()[0];
            w.insert(k, ironbark_mlx::ops::pad(&a, &[(0, pad_to(n) - n)][..], None, None)?);
        } else if matrix {
            quant(&mut w, base, &a)?;
        } else {
            w.insert(k, a);
        }
    }
    let path = out.join(FILE);
    Array::save_safetensors(&w, None, &path).map_err(|e| Error::Load {
        path: path.clone(),
        msg: e.to_string(),
    })?;
    let bytes = std::fs::metadata(&path).map_err(|source| Error::Io { path, source })?.len();
    Ok(Some(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_size_matches_smart_resize() {
        // 400×300 → 384×288 (12.5 rounds to even), small images grow, large ones shrink.
        assert_eq!(target_size(300, 400, 32, MIN_PIXELS, MAX_PIXELS).unwrap(), (288, 384));
        assert_eq!(target_size(100, 100, 32, MIN_PIXELS, MAX_PIXELS).unwrap(), (256, 256));
        let (h, w) = target_size(3000, 4000, 32, MIN_PIXELS, 512 * 512).unwrap();
        assert!(h * w <= 512 * 512 && h % 32 == 0 && w % 32 == 0, "{h}×{w}");
        assert!(target_size(1, 300, 32, MIN_PIXELS, MAX_PIXELS).is_err());
    }

    #[test]
    fn resize_keeps_flat_colours_and_identity() {
        let rgb: Vec<u8> = (0..20 * 10).flat_map(|_| [10u8, 200, 77]).collect();
        let out = resize(&rgb, 20, 10, 7, 13);
        assert!(out.chunks(3).all(|p| p == [10, 200, 77]));
        let ramp: Vec<u8> = (0..8 * 4).flat_map(|i| [(i * 7) as u8, 0, 255]).collect();
        assert_eq!(resize(&ramp, 8, 4, 8, 4), ramp);
    }

    #[test]
    fn positions_lay_images_out_in_3d() {
        let img = 9;
        // 2 text, a 2×3 image, 2 text.
        let ids = [1, 2, img, img, img, img, img, img, 3, 4];
        let (p, next) = positions(&ids, img, &[(2, 3)], 0).unwrap();
        assert_eq!(p[0], [0, 1, 2, 2, 2, 2, 2, 2, 5, 6]);
        assert_eq!(p[1], [0, 1, 2, 2, 2, 3, 3, 3, 5, 6]);
        assert_eq!(p[2], [0, 1, 2, 3, 4, 2, 3, 4, 5, 6]);
        assert_eq!(next, 7);
        assert!(positions(&ids, img, &[], 0).is_err());
        assert!(positions(&ids, img, &[(2, 3), (1, 1)], 0).is_err());
    }

    #[test]
    fn mrope_reduces_to_rope_for_text() {
        let p = [vec![5, 6], vec![5, 6], vec![5, 6]];
        let (cos, _) = mrope_tables(&p, 64, 1e7, [11, 11, 10]);
        let want: Vec<f32> = [5.0f32, 6.0]
            .iter()
            .flat_map(|&x| (0..32).map(move |i| (x * (1.0 / 1e7f32.powf((2 * i) as f32 / 64.0))).cos()))
            .collect();
        assert_eq!(cos, want);
    }

    /// Cosine similarity and largest absolute difference of two arrays (as fp32).
    fn compare(a: &Array, b: &Array) -> (f32, f32) {
        let (a, b) = (a.as_dtype(Dtype::Float32).unwrap(), b.as_dtype(Dtype::Float32).unwrap());
        a.eval().unwrap();
        b.eval().unwrap();
        let (x, y) = (a.as_slice::<f32>(), b.as_slice::<f32>());
        assert_eq!(x.len(), y.len());
        let dot: f64 = x.iter().zip(y).map(|(p, q)| f64::from(*p) * f64::from(*q)).sum();
        let norm = |v: &[f32]| v.iter().map(|p| f64::from(*p).powi(2)).sum::<f64>().sqrt();
        let max = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
        ((dot / (norm(x) * norm(y))) as f32, max)
    }

    /// Against transformers (fp32) on the BF16 original's vision weights, through a checkpoint
    /// that holds them (`QWEN_MODEL`, the 4-bit MLX one keeps them bf16), with the fixture of
    /// the scratchpad's `qwen_vision_oracle.py` (`QWEN_VISION_FIXTURE`, a dir). Then 4-bit, as
    /// [`pack`] writes them.
    #[test]
    fn vision_matches_transformers() {
        let (Ok(dir), Ok(fx)) = (std::env::var("QWEN_MODEL"), std::env::var("QWEN_VISION_FIXTURE")) else {
            eprintln!("skipped: set QWEN_MODEL and QWEN_VISION_FIXTURE");
            return;
        };
        let dir = Path::new(&dir);
        let f: HashMap<String, Array> =
            Array::load_safetensors(Path::new(&fx).join("vision.safetensors")).unwrap().into_iter().collect();
        let v = Vision::load(dir).unwrap().expect("vision weights");
        let c = v.cfg;

        // Preprocessing: the same patches, within a byte of resampling.
        let rgb = &f["rgb"];
        rgb.eval().unwrap();
        let (h, w) = (rgb.shape()[0] as usize, rgb.shape()[1] as usize);
        let px = preprocess(&c, rgb.as_slice::<u8>(), w, h, MAX_PIXELS).unwrap();
        let want = &f["pixel_values"];
        want.eval().unwrap();
        assert_eq!(px.data.len(), want.as_slice::<f32>().len());
        let off = px.data.iter().zip(want.as_slice::<f32>()).filter(|(a, b)| (*a - *b).abs() > 0.5 / 255.0).count();
        let worst = px.data.iter().zip(want.as_slice::<f32>()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        eprintln!("pixels: {off} of {} differ, worst {:.4} ({:.1} levels)", px.data.len(), worst, worst * 127.5);
        assert!(worst <= 2.5 / 127.5, "resampling differs by {worst}");

        // The encoder on the oracle's own pixels.
        let theirs = Pixels {
            data: want.as_slice::<f32>().to_vec(),
            rows: px.rows,
            cols: px.cols,
        };
        let (img, last) = v.encode_traced(&theirs).unwrap();
        let (cos_l, max_l) = compare(&last, &f["last_hidden"]);
        let (cos_m, max_m) = compare(&img.embeds, &f["merged"]);
        eprintln!("bf16: last hidden cos {cos_l:.6} max {max_l:.3}; merged cos {cos_m:.6} max {max_m:.3}");
        let mut v32 = Vision::load(dir).unwrap().expect("vision weights");
        v32.cast(Dtype::Float32);
        let (img32, last32) = v32.encode_traced(&theirs).unwrap();
        let (cos_32, max_32) = compare(&img32.embeds, &f["merged"]);
        let (cos_l32, _) = compare(&last32, &f["last_hidden"]);
        eprintln!("fp32: last hidden cos {cos_l32:.7}; merged cos {cos_32:.7} max {max_32:.5}");
        drop(v32);
        assert!(cos_32 > 0.99999, "fp32 encoder differs: {cos_32}");
        assert!(cos_m > 0.995, "bf16 encoder: {cos_m}");
        assert_eq!((img.rows, img.cols, img.tokens()), (9, 12, 108));

        // Ours end to end (our resize).
        let (cos_e, _) = compare(&v.encode(&px).unwrap().embeds, &f["merged"]);
        eprintln!("bf16, our preprocessing: merged cos {cos_e:.6}");
        assert!(cos_e > 0.995);

        // 3D positions and the text RoPE at them.
        let ids = &f["ids"];
        ids.eval().unwrap();
        let ids: Vec<u32> = ids.as_slice::<i32>().iter().map(|&x| x as u32).collect();
        let (p, next) = positions(&ids, 248_056, &[(img.rows, img.cols)], 0).unwrap();
        let want = &f["positions"];
        want.eval().unwrap();
        let flat: Vec<i64> = p.iter().flatten().copied().collect();
        assert_eq!(flat, want.as_slice::<i32>().iter().map(|&x| i64::from(x)).collect::<Vec<_>>());
        assert_eq!(next - ids.len() as i64, -96);
        let (cos, sin) = mrope_tables(&p, 64, 1e7, [11, 11, 10]);
        for (ours, theirs) in [(cos, &f["cos"]), (sin, &f["sin"])] {
            theirs.eval().unwrap();
            let t = theirs.as_slice::<f32>();
            let worst = (0..ids.len())
                .flat_map(|l| (0..32).map(move |i| (l, i)))
                .map(|(l, i)| {
                    (ours[l * 32 + i] - t[l * 64 + i]).abs().max((ours[l * 32 + i] - t[l * 64 + 32 + i]).abs())
                })
                .fold(0.0f32, f32::max);
            assert!(worst < 1e-4, "mrope table differs by {worst}");
        }

        // Quantized, as packed.
        let tmp = std::env::temp_dir().join(format!("ironbark-vision-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        std::fs::copy(dir.join("config.json"), tmp.join("config.json")).unwrap();
        std::fs::write(tmp.join("experts.json"), "{}").unwrap();
        for bits in [8, 6, 5, 4] {
            let bytes = pack(dir, &tmp, bits).unwrap().unwrap();
            let q = Vision::load(&tmp).unwrap().expect("packed vision weights");
            let (cos_q, _) = compare(&q.encode(&theirs).unwrap().embeds, &f["merged"]);
            eprintln!("{bits}-bit ({:.0} MB): merged cos {cos_q:.6}", bytes as f64 / 1e6);
            if bits == PACK_BITS {
                assert!(cos_q > 0.99, "{bits}-bit encoder: {cos_q}");
            }
        }
        std::fs::remove_dir_all(&tmp).unwrap();
    }
}
