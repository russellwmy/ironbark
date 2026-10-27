//! Helpers shared by the model commands: logits and memory reports.

use anyhow::Result;
use ironbark_mlx::Array;

use ironbark_engine::device::footprint;

pub(crate) fn memory() -> Result<()> {
    println!(
        "MLX memory: active {:.2} GB, peak {:.2} GB",
        ironbark_mlx::memory::active_memory()? as f64 / 1e9,
        ironbark_mlx::memory::peak_memory()? as f64 / 1e9
    );
    if let Some(f) = footprint() {
        println!("footprint: {:.2} GB, peak {:.2} GB", f.now as f64 / 1e9, f.peak as f64 / 1e9);
    }
    Ok(())
}

/// The `n` largest logits of `[1, 1, vocab]` as (token, logit).
pub(crate) fn top_n(logits: &Array, n: usize) -> Result<Vec<(u32, f32)>> {
    let l = logits.as_dtype(ironbark_mlx::Dtype::Float32)?;
    let v: Vec<f32> = l.as_slice::<f32>().to_vec();
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|a, b| v[*b].total_cmp(&v[*a]));
    Ok(idx.into_iter().take(n).map(|i| (i as u32, v[i])).collect())
}
