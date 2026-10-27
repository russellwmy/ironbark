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
