//! FlashHead, DeepGrove's approximate LM head for decode (`maple.py` `FlashHead`): score the
//! vocabulary's cluster centroids, then compute exact logits only for the tokens of the top
//! `probes` clusters and a few forced control tokens (end of turn, `</think>`); every other
//! logit is −∞. Greedy decode matches the exact head whenever its argmax is in a probed
//! cluster. Opt-in: [`crate::Model::set_flash_head`].
//!
//! ```text
//!   hn ─▶ centroids (4748 × 2048, 4-bit) ─▶ top 512 clusters ─▶ gather_qmm over their 32-row
//!         blocks of the head (16,384 rows of 151,936) ─▶ scatter into −∞ logits
//! ```
//!
//! The clusters partition the vocabulary (4748 × 32 = 151,936), so the LM head is kept in
//! cluster order once FlashHead is on: a probed cluster is a contiguous block, and the exact
//! head reads every row and puts them back in token order (same dot products, so still exact).

use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{arange, argpartition_axis, concatenate, gather_qmm};
use mlx_rs::{Array, Dtype};

use crate::model::QMat;
use crate::{Config, Error, Result};

/// FlashHead tensors of a checkpoint (`lm_head_flash.*`, config `flash_head`).
pub(crate) struct FlashHead {
    /// Scaled cluster centroids `[clusters, hidden]`.
    pub(crate) centroids: QMat,
    /// Token of each cluster-order row `[vocab]`.
    pub(crate) token_map: Array,
    /// Cluster-order row of each token `[vocab]` (the inverse of `token_map`).
    pub(crate) rows: Array,
    clusters: i32,
    size: i32,
    probes: i32,
    /// Forced control tokens and their cluster-order rows.
    force_tokens: Array,
    force_rows: Array,
    /// In use for decode.
    pub(crate) on: bool,
}

impl FlashHead {
    /// From the checkpoint's `token_map` `[clusters, size]` (a permutation of the vocabulary)
    /// and `centroids`; `None` if the config has no FlashHead.
    pub(crate) fn new(cfg: &Config, token_map: &Array, centroids: QMat) -> Result<Option<Self>> {
        let Some(meta) = &cfg.flash_head else { return Ok(None) };
        let shape = token_map.shape();
        let (clusters, size) = (shape[0], shape[1]);
        let map: Vec<i32> = token_map.as_dtype(Dtype::Int32)?.reshape(&[-1])?.as_slice::<i32>().to_vec();
        let vocab = map.len();
        let mut rows = vec![-1i32; vocab];
        for (r, &t) in map.iter().enumerate() {
            match rows.get_mut(t as usize) {
                Some(slot @ -1) => *slot = r as i32,
                _ => {
                    return Err(Error::Format("lm_head_flash.token_map is not a permutation of the vocabulary".into()));
                }
            }
        }
        let force_rows: Vec<i32> = meta.force_tokens.iter().map(|&t| rows[t as usize]).collect();
        Ok(Some(Self {
            centroids,
            token_map: Array::from_slice(&map, &[vocab as i32]),
            rows: Array::from_slice(&rows, &[vocab as i32]),
            clusters,
            size,
            probes: meta.probes.min(clusters as usize) as i32,
            force_tokens: Array::from_slice(&meta.force_tokens, &[meta.force_tokens.len() as i32]),
            force_rows: Array::from_slice(&force_rows, &[force_rows.len() as i32]),
            on: false,
        }))
    }

    /// `head` (token order) in cluster order.
    pub(crate) fn permute(&self, head: &QMat) -> Result<QMat> {
        let take = |a: &Array| -> Result<Array> { Ok(a.take_axis(&self.token_map, 0)?) };
        let q = QMat {
            w: take(&head.w)?,
            scales: take(&head.scales)?,
            biases: take(&head.biases)?,
            group: head.group,
            bits: head.bits,
        };
        mlx_rs::transforms::eval([&q.w, &q.scales, &q.biases])?;
        Ok(q)
    }

    /// Approximate logits `[1, 1, vocab]` of one position `hn` `[1, 1, hidden]`, from `head` in
    /// cluster order.
    pub(crate) fn logits(&self, head: &QMat, hn: &Array) -> Result<Array> {
        let hidden = *hn.shape().last().unwrap_or(&0);
        let vocab = head.w.shape()[0];
        let scores = self.centroids.apply(&hn.reshape(&[1, hidden])?)?;
        let top = argpartition_axis(&scores, -self.probes, -1)?.index((.., -self.probes..));
        let block = |a: &Array| a.reshape(&[self.clusters, self.size, -1]);
        let sub = gather_qmm(
            hn.reshape(&[1, 1, 1, 1, hidden])?,
            block(&head.w)?,
            block(&head.scales)?,
            &block(&head.biases)?,
            None,
            &top.reshape(&[1, 1, self.probes])?,
            true,
            head.group,
            head.bits,
            false,
        )?
        .reshape(&[-1])?;
        // Rows of `sub`: each probed cluster's block, in order.
        let size = Array::from_int(self.size);
        let rows = top
            .as_dtype(Dtype::Int32)?
            .reshape(&[self.probes, 1])?
            .multiply(&size)?
            .add(&arange::<_, i32>(None, self.size, None)?)?;
        let tokens = self.token_map.take_axis(&rows.reshape(&[-1])?, 0)?;
        let one_row = |a: &Array| a.reshape(&[vocab, 1, -1]);
        let forced = gather_qmm(
            hn.reshape(&[1, 1, hidden])?,
            one_row(&head.w)?,
            one_row(&head.scales)?,
            &one_row(&head.biases)?,
            None,
            &self.force_rows.as_dtype(Dtype::Uint32)?,
            true,
            head.group,
            head.bits,
            false,
        )?
        .reshape(&[-1])?;
        let ids = concatenate(&[tokens, self.force_tokens.clone()], 0)?;
        let vals = concatenate(&[sub, forced], 0)?;
        let none = Array::full::<f32>(&[vocab], Array::from_f32(f32::NEG_INFINITY))?.as_dtype(vals.dtype())?;
        Ok(none.put_along_axis(&ids, &vals, 0)?.reshape(&[1, 1, vocab])?)
    }
}
