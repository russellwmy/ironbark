//! `experts.bin` / `experts.json` from a Maple MLX checkpoint, for streaming experts from SSD.
//!
//! Record `l · E + e` holds expert `e` of layer `l` in the [`Record`] layout (what the slab
//! kernels read), padded to whole 16 KiB pages:
//!
//! ```text
//!   [up codes, gate codes: 2F x H/4][down codes: H x F/4][α up, α gate: 2F bf16][α down: H bf16]
//! ```
//!
//! With `trit`, rows are stored as trits (see `row`): 19% fewer bytes per record.
//!
//! It also writes the token embedding to `embed.bin` (memory-mapped at run time, see `embed`)
//! and the rest of the checkpoint (attention, routers, norms, head) as `model.safetensors`
//! with the config and tokenizer, so `out` alone is a complete streaming model: what a phone
//! needs, without the 5 GB of expert tensors twice.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use ironbark_mlx::{Array, Dtype};

use crate::manifest;
use crate::model::{QMat, load_tensors};
use crate::row::Record;
use crate::{Config, Error, Result};

/// Page size records are padded to (Apple Silicon).
pub(crate) const PAGE: usize = 16384;

/// What [`pack`] wrote.
#[derive(Debug, Clone, Copy)]
pub struct Packed {
    /// Records (layers x experts).
    pub records: usize,
    /// Bytes per record without padding.
    pub record_bytes: usize,
    /// Bytes between records.
    pub stride: usize,
}

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |source| Error::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// u32 words of an evaluated array.
fn words(a: &Array) -> Result<Vec<u32>> {
    a.eval()?;
    Ok(a.as_slice::<u32>().to_vec())
}

/// bf16 bits of an evaluated bf16 array (bf16 -> f32 is exact, so the high half is the value).
fn bf16_bits(a: &Array) -> Result<Vec<u16>> {
    let f = a.as_dtype(Dtype::Float32)?;
    f.eval()?;
    Ok(f.as_slice::<f32>().iter().map(|v| (v.to_bits() >> 16) as u16).collect())
}

/// Write a packed model to `out`: the experts to `experts.bin` / `experts.json`, the other
/// weights, config and tokenizer, then `manifest.json` (see [`crate::manifest`]).
///
/// # Errors
///
/// Fails if a tensor is missing or has an unexpected size, or a file cannot be written.
pub fn pack(model: &Path, out: &Path, trit: bool) -> Result<Packed> {
    let cfg = Config::read(model)?;
    if out.canonicalize().ok() == model.canonicalize().ok() {
        return Err(Error::Format("pack: out must differ from the model dir".into()));
    }
    let (h, f, ne) = (cfg.hidden, cfg.ffn, cfg.experts);
    let rec = Record {
        hidden: h,
        ffn: f,
        trit,
    };
    let stride = rec.bytes().div_ceil(PAGE) * PAGE;
    std::fs::create_dir_all(out).map_err(io(out))?;
    let bin = out.join("experts.bin");
    let file = File::create(&bin).map_err(io(&bin))?;
    // Written uncached, so the first streaming reads come from storage, not the page cache.
    ironbark_experts::layout::no_cache(&file).map_err(io(&bin))?;

    // Lazily loaded: only the tensors evaluated below are read from disk.
    let mut t = load_tensors(model)?;
    let mut take = |name: String| t.remove(&name).ok_or_else(|| Error::Format(format!("missing tensor {name}")));
    let mut buf = vec![0u8; stride];
    let mut hashes = Vec::with_capacity(cfg.layers * ne);
    for l in 0..cfg.layers {
        let p = |m: &str, k: &str| format!("model.layers.{l}.mlp.switch_mlp.{m}.{k}");
        let up = words(&take(p("up_proj", "weight"))?)?;
        let gate = words(&take(p("gate_proj", "weight"))?)?;
        let down = words(&take(p("down_proj", "weight"))?)?;
        let a_up = bf16_bits(&take(p("up_proj", "row_alpha"))?)?;
        let a_gate = bf16_bits(&take(p("gate_proj", "row_alpha"))?)?;
        let a_down = bf16_bits(&take(p("down_proj", "row_alpha"))?)?;
        // Per expert: f x h/16 words (up, gate), h x f/16 (down); f, f, h scales.
        let (wu, wd) = (f * h / 16, h * f / 16);
        if up.len() != ne * wu || gate.len() != ne * wu || down.len() != ne * wd || a_down.len() != ne * h {
            return Err(Error::Format(format!("layer {l}: expert tensors do not match the config")));
        }
        for e in 0..ne {
            buf.fill(0);
            let put32 = |buf: &mut [u8], at: usize, v: &[u32]| {
                for (i, w) in v.iter().enumerate() {
                    buf[at + 4 * i..at + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
                }
            };
            let put16 = |buf: &mut [u8], at: usize, v: &[u16]| {
                for (i, w) in v.iter().enumerate() {
                    buf[at + 2 * i..at + 2 * i + 2].copy_from_slice(&w.to_le_bytes());
                }
            };
            if trit {
                rows_to_trits(&up[e * wu..(e + 1) * wu], h, &mut buf);
                rows_to_trits(&gate[e * wu..(e + 1) * wu], h, &mut buf[f * rec.row_bytes(h)..]);
                rows_to_trits(&down[e * wd..(e + 1) * wd], f, &mut buf[rec.down_codes()..]);
            } else {
                put32(&mut buf, 0, &up[e * wu..(e + 1) * wu]);
                put32(&mut buf, 4 * wu, &gate[e * wu..(e + 1) * wu]);
                put32(&mut buf, rec.down_codes(), &down[e * wd..(e + 1) * wd]);
            }
            put16(&mut buf, rec.up_gate_alpha(), &a_up[e * f..(e + 1) * f]);
            put16(&mut buf, rec.up_gate_alpha() + 2 * f, &a_gate[e * f..(e + 1) * f]);
            put16(&mut buf, rec.down_alpha(), &a_down[e * h..(e + 1) * h]);
            file.write_all_at(&buf, ((l * ne + e) * stride) as u64).map_err(io(&bin))?;
            hashes.push(manifest::hash(&buf[..rec.bytes()]));
        }
    }
    write_slim(model, out, &cfg, t)?;
    let meta = serde_json::json!({
        "version": manifest::VERSION,
        "layers": cfg.layers,
        "experts": ne,
        "record_bytes": rec.bytes(),
        "format": if trit { "trit" } else { "2bit" },
        "stride": stride,
    });
    let json = out.join("experts.json");
    std::fs::write(&json, meta.to_string()).map_err(io(&json))?;
    // Last: a pack with a manifest is complete.
    manifest::write(out, &hashes, rec.bytes(), stride)?;
    Ok(Packed {
        records: cfg.layers * ne,
        record_bytes: rec.bytes(),
        stride,
    })
}

/// Files copied next to the slim weights: what [`Config::read`] and the tokenizer need, the
/// chat template, and the license and model card (MIT asks for the notice with every copy).
const SIDE_FILES: [&str; 8] = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "LICENSE",
    "README.md",
];

/// Write the embedding to `out/embed.bin`, every other tensor except the experts to
/// `out/model.safetensors` with its index, and copy the config and tokenizer files.
fn write_slim(model: &Path, out: &Path, cfg: &Config, mut t: HashMap<String, Array>) -> Result<()> {
    t.retain(|name, _| !name.contains(".mlp.switch_mlp."));
    let take = |t: &mut HashMap<String, Array>, name: &str| {
        t.remove(name).ok_or_else(|| Error::Format(format!("missing tensor {name}")))
    };
    let embed = QMat {
        w: take(&mut t, "model.word_embeddings.weight")?,
        scales: take(&mut t, "model.word_embeddings.scales")?,
        biases: take(&mut t, "model.word_embeddings.biases")?,
        group: cfg.head_group as i32,
        bits: cfg.head_bits as i32,
    };
    crate::embed::write(out, cfg, &embed)?;
    drop(embed);
    let file = "model.safetensors";
    let path = out.join(file);
    Array::save_safetensors(&t, None, &path).map_err(|e| Error::Load {
        path: path.clone(),
        msg: e.to_string(),
    })?;
    let map: serde_json::Map<String, serde_json::Value> = t.keys().map(|k| (k.clone(), file.into())).collect();
    let index = out.join("model.safetensors.index.json");
    let json = serde_json::json!({ "metadata": {}, "weight_map": map });
    std::fs::write(&index, json.to_string()).map_err(io(&index))?;
    for name in SIDE_FILES {
        let src = model.join(name);
        if src.exists() {
            std::fs::copy(&src, out.join(name)).map_err(io(&src))?;
        }
    }
    Ok(())
}

/// Five codes in {0, 1, 2} (first = most significant digit) as one byte, read back by
/// `digit`: `ceil(v · 256 / 243)`.
fn trit_byte(c: &[u32]) -> u8 {
    let v = c.iter().fold(0u32, |v, &d| v * 3 + d);
    ((v * 256).div_ceil(243)) as u8
}

/// Digit `i` (0 = most significant) of a [`trit_byte`], as the kernels compute it.
#[cfg(test)]
fn digit(b: u8, i: u32) -> u32 {
    let q = b.wrapping_mul(3u8.pow(i));
    (u32::from(q) * 3) >> 8
}

/// Rows of `k` 2-bit codes (16 per u32, code `i` at bits `2i`) as trit rows into `out`.
pub(crate) fn rows_to_trits(words: &[u32], k: usize, out: &mut [u8]) {
    let groups = k / 16;
    let row_bytes = groups * 3 + k / 64;
    for (r, row) in words.chunks(groups).enumerate() {
        let o = &mut out[r * row_bytes..(r + 1) * row_bytes];
        o.fill(0);
        for (g, w) in row.iter().enumerate() {
            let c: Vec<u32> = (0..16).map(|i| (w >> (2 * i)) & 3).collect();
            for b in 0..3 {
                o[g * 3 + b] = trit_byte(&c[5 * b..5 * b + 5]);
            }
            o[groups * 3 + g / 4] |= (c[15] as u8) << (2 * (g % 4));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_trit_byte_decodes_to_its_digits() {
        for v in 0..243u32 {
            let c: Vec<u32> = (0..5).map(|i| v / 3u32.pow(4 - i) % 3).collect();
            let b = trit_byte(&c);
            let back: Vec<u32> = (0..5).map(|i| digit(b, i)).collect();
            assert_eq!(back, c, "v = {v}, byte = {b}");
        }
    }
}
