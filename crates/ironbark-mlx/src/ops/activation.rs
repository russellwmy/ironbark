//! Activation functions.

use crate::{Array, error::Result, ops::sigmoid, transforms::compile::compile};

/// Applies the Sigmoid Linear Unit (Swish): `x * sigmoid(x)`, compiled into one kernel.
pub fn silu(x: impl AsRef<Array>) -> Result<Array> {
    let f = |x: &Array| x.multiply(sigmoid(x)?);
    let mut compiled = compile(f, true);
    compiled(x.as_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    #[test]
    fn test_silu() {
        let x = array!([-1.0f32, 0.0, 1.0, 2.0]);
        let y = silu(&x).unwrap();
        let expected = [-0.268_941_43f32, 0.0, 0.731_058_6, 1.761_594_2];
        for (a, b) in y.as_slice::<f32>().iter().zip(expected) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }
}
