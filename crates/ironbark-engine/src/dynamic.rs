//! Dynamic top-k: each token keeps only the router choices that carry weight.
//!
//! ```text
//!   router: top-k choices, weights w ─► shares w / Σw ─► drop shares < drop_below
//!                                                        (never the min_k largest)
//!   dropped choice ─► the token's strongest expert, weight 0   (loaded anyway: no read)
//!   kept weights   ─► rescaled to the total they had (renorm), or left as they were
//! ```
//!
//! Easy tokens keep few experts, hard ones all of them; streaming reads only the kept.

use ironbark_mlx::ops::indexing::argmax_axis;
use ironbark_mlx::ops::{argsort, concatenate, select};
use ironbark_mlx::{Array, Dtype};

use crate::{Error, Result};

/// Dynamic top-k settings (off by default: the model as trained).
#[derive(Debug, Clone, Copy)]
pub struct Dynamic {
    /// Drop a choice whose share of its token's routing weight is under this; 0: off.
    pub drop_below: f32,
    /// Fewest experts a token keeps.
    pub min_k: usize,
    /// Rescale the kept weights to the total of all the choices (else the dropped weight is
    /// simply missing, the kept experts at their trained weights).
    pub renorm: bool,
}

impl Default for Dynamic {
    fn default() -> Self {
        Self {
            drop_below: 0.0,
            min_k: 3,
            renorm: true,
        }
    }
}

impl Dynamic {
    /// Whether it drops anything for a router choosing `top_k` experts.
    #[must_use]
    pub fn on(&self, top_k: usize) -> bool {
        self.drop_below > 0.0 && self.min_k < top_k
    }

    /// The choices `idx` `[.., k]` with weights `w` after dropping the weak ones (see the
    /// module doc).
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails.
    pub fn apply(&self, idx: &Array, w: &Array) -> Result<(Array, Array)> {
        let total = w.sum_axis(-1, true)?;
        let share = w.divide(&total)?;
        // Rank of each choice within its token: how many of the others weigh more.
        let rank = share.expand_dims(-1)?.lt(share.expand_dims(-2)?)?.sum_axis(-1, false)?;
        let weak = share.lt(Array::from_f32(self.drop_below).as_dtype(w.dtype())?)?;
        let drop = weak.logical_and(rank.ge(Array::from_int(self.min_k as i32))?)?;
        let top = idx.take_along_axis(&argmax_axis(w, -1, true)?, -1)?;
        let idx = select(&drop, &top, idx)?;
        let kept = select(&drop, Array::from_f32(0.0).as_dtype(w.dtype())?, w)?;
        let w = if self.renorm {
            kept.multiply(total.divide(kept.sum_axis(-1, true)?)?)?
        } else {
            kept
        };
        Ok((idx, w))
    }
}

/// Streaming: the routing weights `w` as f32 to read back with the choices, when `d` drops
/// any on this forward (`active`: a decode step); dropped choices have weight 0 and skip the
/// expert kernels ([`dropped`]).
pub(crate) fn read_weights(d: &Dynamic, top_k: usize, active: bool, w: &Array) -> Result<Option<Array>> {
    Ok(if active && d.on(top_k) {
        Some(w.as_dtype(Dtype::Float32)?.contiguous()?)
    } else {
        None
    })
}

/// Per (token, choice) pair, whether it was dropped (from [`read_weights`], evaluated).
pub(crate) fn dropped(wf: Option<&Array>) -> Vec<bool> {
    wf.map(|a| a.as_slice::<f32>().iter().map(|&x| x == 0.0).collect()).unwrap_or_default()
}

/// Expert outputs computed per (token, choice) pair (`outs`, rows in the pair order `order`)
/// back in pair order `0..total`; a pair not computed (dropped, weight 0) gets a zero row.
pub(crate) fn pairs_in_order(mut outs: Vec<Array>, order: &[u32], total: usize) -> Result<Array> {
    // One group covering every pair is already in order.
    if outs.len() == 1 && order.len() == total {
        return Ok(outs.remove(0));
    }
    if order.len() == total {
        let inv = argsort(Array::from_slice(order, &[order.len() as i32]))?;
        return Ok(concatenate(&outs, 0)?.take_axis(&inv, 0)?);
    }
    let first = outs.first().ok_or_else(|| Error::Format("no expert outputs".into()))?;
    let zero = Array::zeros::<f32>(&[1, first.shape()[1]])?.as_dtype(first.dtype())?;
    let mut row = vec![order.len() as u32; total];
    for (r, &p) in order.iter().enumerate() {
        row[p as usize] = r as u32;
    }
    outs.push(zero);
    Ok(concatenate(&outs, 0)?.take_axis(Array::from_slice(&row, &[total as i32]), 0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choices() -> (Array, Array) {
        (
            Array::from_slice(&[10u32, 11, 12, 13, 14], &[1, 1, 5]),
            Array::from_slice(&[0.3f32, 0.5, 0.1, 0.06, 0.04], &[1, 1, 5]),
        )
    }

    #[test]
    fn drops_weak_choices_past_the_floor() {
        let mut d = Dynamic {
            drop_below: 0.08,
            min_k: 2,
            ..Dynamic::default()
        };
        let (idx, w) = choices();
        let (idx, w) = d.apply(&idx, &w).unwrap();
        // 0.06 and 0.04 are under 0.08 and past the 2 largest: they go to expert 11 (the
        // strongest) with weight 0; the rest share the whole weight.
        assert_eq!(idx.as_slice::<u32>(), &[10, 11, 12, 11, 11]);
        let got = w.as_slice::<f32>().to_vec();
        let want = [0.3f32 / 0.9, 0.5 / 0.9, 0.1 / 0.9, 0.0, 0.0];
        assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-6), "{got:?}");

        // Without renormalizing, the kept weights stay as they were.
        d.renorm = false;
        let (idx, w) = choices();
        let (_, w) = d.apply(&idx, &w).unwrap();
        assert_eq!(w.as_slice::<f32>(), &[0.3, 0.5, 0.1, 0.0, 0.0]);

        // Every share under the threshold: the floor still keeps the 2 largest.
        d.drop_below = 0.9;
        let (idx, w) = choices();
        let (idx, _) = d.apply(&idx, &w).unwrap();
        assert_eq!(idx.as_slice::<u32>(), &[10, 11, 11, 11, 11]);
    }

    #[test]
    fn shares_not_weights_decide() {
        // Weights summing to 0.5 (a router without top-k renormalization): the shares are
        // twice the weights, and the kept weights are rescaled to the same 0.5.
        let d = Dynamic {
            drop_below: 0.15,
            min_k: 1,
            ..Dynamic::default()
        };
        let idx = Array::from_slice(&[1u32, 2, 3], &[1, 1, 3]);
        let w = Array::from_slice(&[0.3f32, 0.15, 0.05], &[1, 1, 3]);
        let (idx, w) = d.apply(&idx, &w).unwrap();
        assert_eq!(idx.as_slice::<u32>(), &[1, 2, 1]);
        let got = w.as_slice::<f32>().to_vec();
        let want = [0.3f32 / 0.45 * 0.5, 0.15 / 0.45 * 0.5, 0.0];
        assert!(got.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-6), "{got:?}");
    }

    #[test]
    fn pairs_in_order_fills_skipped_pairs_with_zeros() {
        // Pairs 2 and 0 computed (in that order), pair 1 dropped.
        let outs = vec![Array::from_slice(&[2.0f32, 2.0, 0.0, 0.0], &[2, 2])];
        let y = pairs_in_order(outs, &[2, 0], 3).unwrap();
        assert_eq!(y.as_slice::<f32>(), &[0.0, 0.0, 0.0, 0.0, 2.0, 2.0]);
        // Two groups, every pair computed: back in pair order.
        let outs = vec![
            Array::from_slice(&[1.0f32], &[1, 1]),
            Array::from_slice(&[0.0f32], &[1, 1]),
        ];
        let y = pairs_in_order(outs, &[1, 0], 2).unwrap();
        assert_eq!(y.as_slice::<f32>(), &[0.0, 1.0]);
    }
}
