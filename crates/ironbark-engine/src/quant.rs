//! Affine-quantized matrices of a checkpoint: loading them by name (bits inferred from the
//! shapes) and row views of a stacked matrix.

use ironbark_mlx::Array;

use crate::model::QMat;
use crate::{Error, Result};

pub(crate) struct Tensors(pub(crate) std::collections::HashMap<String, Array>);

impl Tensors {
    pub(crate) fn take(&mut self, name: &str) -> Result<Array> {
        self.0.remove(name).ok_or_else(|| Error::Format(format!("missing tensor {name}")))
    }

    /// Affine-quantized matrix `prefix.{weight, scales, biases}`; bits from the shapes (the
    /// router and shared-expert gate are 8-bit, the rest 4-bit).
    pub(crate) fn qmat(&mut self, prefix: &str, group: usize) -> Result<QMat> {
        let w = self.take(&format!("{prefix}.weight"))?;
        let scales = self.take(&format!("{prefix}.scales"))?;
        let biases = self.take(&format!("{prefix}.biases"))?;
        Ok(qmat(w, scales, biases, group))
    }
}

/// A quantized matrix with its bits inferred: `cols · 32 / (groups · group)`.
fn qmat(w: Array, scales: Array, biases: Array, group: usize) -> QMat {
    let (wc, sc) = (w.shape()[w.ndim() - 1], scales.shape()[scales.ndim() - 1]);
    QMat {
        bits: wc * 32 / (sc * group as i32),
        w,
        scales,
        biases,
        group: group as i32,
    }
}
