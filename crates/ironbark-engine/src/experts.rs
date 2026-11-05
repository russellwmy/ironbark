//! Packing an MoE checkpoint: the record format a pack's `experts.json` names, and the slim
//! weights (everything but the experts) written next to `experts.bin`.

use std::path::Path;

use ironbark_mlx::Array;

use crate::{Error, Result};

/// The record format in `dir/experts.json`.
pub(crate) fn format(dir: &Path) -> Result<String> {
    let path = dir.join("experts.json");
    let raw = std::fs::read(&path).map_err(io(&path))?;
    let v: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
    Ok(v["format"].as_str().unwrap_or_default().to_owned())
}

/// Files copied next to the slim weights.
const SIDE_FILES: [&str; 7] = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "chat_template.jinja",
    "LICENSE",
    "README.md",
];

fn io(path: &Path) -> impl FnOnce(std::io::Error) -> Error + '_ {
    move |source| Error::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Mark `out/config.json` with `marker` as how its experts are stored, and write `experts.json`
/// and `manifest.json` for the records just written to `experts.bin`.
#[expect(clippy::too_many_arguments, reason = "a pack's layout, each with its own meaning")]
pub(crate) fn finish(
    out: &Path,
    layers: usize,
    experts: usize,
    record_bytes: usize,
    stride: usize,
    format: &str,
    marker: &serde_json::Value,
    hashes: &[u64],
) -> Result<()> {
    let path = out.join("config.json");
    let raw = std::fs::read(&path).map_err(io(&path))?;
    let mut c: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|e| Error::Format(format!("{}: {e}", path.display())))?;
    c["quantization"]["experts"] = marker.clone();
    let json = serde_json::to_string_pretty(&c).map_err(|e| Error::Format(e.to_string()))?;
    std::fs::write(&path, json).map_err(io(&path))?;
    let meta = serde_json::json!({
        "version": crate::manifest::VERSION,
        "layers": layers,
        "experts": experts,
        "record_bytes": record_bytes,
        "format": format,
        "stride": stride,
    });
    let json = out.join("experts.json");
    std::fs::write(&json, meta.to_string()).map_err(io(&json))?;
    crate::manifest::write(out, hashes, record_bytes, stride)
}

/// Every text tensor except the experts to `out/model.safetensors` with its index, and the
/// config and tokenizer files.
pub(crate) fn write_slim(
    model: &Path,
    out: &Path,
    mut t: std::collections::HashMap<String, Array>,
    marker: &str,
) -> Result<()> {
    t.retain(|name, _| name.starts_with("language_model.") && !name.contains(marker));
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
