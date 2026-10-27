//! A growing key/value cache (`[1, kv_heads, capacity, head_dim]` per layer) for the
//! full-attention layers of Qwen3.5 / 3.6.

use ironbark_mlx::Array;
use ironbark_mlx::ops::concatenate;
use ironbark_mlx::ops::indexing::{IndexOp, TryIndexMutOp};

use crate::Result;
use crate::model::KV_STEP;

/// Write `k`, `v` `[1, kv_heads, L, d]` at `pos` into the caches `kc`, `vc` (grown by
/// [`KV_STEP`]); the keys and values to attend over.
pub(crate) fn kv_update(
    kc: &mut Option<Array>,
    vc: &mut Option<Array>,
    k: &Array,
    v: &Array,
    pos: i32,
) -> Result<(Array, Array)> {
    let n = k.shape()[2];
    let cap = kc.as_ref().map_or(0, |a| a.shape()[2]);
    let end = pos + n;
    let (mut ks, mut vs) = (kc.take(), vc.take());
    if end > cap {
        let want = (end + KV_STEP - 1) / KV_STEP * KV_STEP;
        let s = k.shape();
        let grow = |old: Option<Array>, dt| -> Result<Array> {
            let fresh = Array::zeros::<f32>(&[s[0], s[1], want - cap, s[3]])?.as_dtype(dt)?;
            Ok(match old {
                Some(o) => concatenate(&[&o, &fresh], 2)?,
                None => fresh,
            })
        };
        ks = Some(grow(ks, k.dtype())?);
        vs = Some(grow(vs, v.dtype())?);
    }
    let (Some(mut ks), Some(mut vs)) = (ks, vs) else {
        unreachable!("allocated above")
    };
    ks.try_index_mut((.., .., pos..end), k)?;
    vs.try_index_mut((.., .., pos..end), v)?;
    let out = (ks.index((.., .., ..end)), vs.index((.., .., ..end)));
    *kc = Some(ks);
    *vc = Some(vs);
    Ok(out)
}
