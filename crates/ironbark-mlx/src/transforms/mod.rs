//! Function transforms: evaluation and compilation.
//!
//! **WARN**: Because function transforms including compilation works on
//! the computation graph, the user must ensure that all `Array`s are passed
//! as inputs to the function/closure. Closures with captured `Array`s may
//! not work as expected and may lead to undefined behavior.

use crate::{
    Array,
    error::Result,
    utils::{VectorArray, guard::Guarded},
};

pub mod compile;

/// Evaluate an iterator of [`Array`]s.
pub fn eval<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<()> {
    let vec = VectorArray::try_from_iter(outputs.into_iter())?;
    <() as Guarded>::try_from_op(|_| unsafe { ironbark_mlx_sys::mlx_eval(vec.as_ptr()) })
}

/// Asynchronously evaluate an iterator of [`Array`]s.
///
/// Please note that this is not a rust async function.
pub fn async_eval<'a>(outputs: impl IntoIterator<Item = &'a Array>) -> Result<()> {
    let vec = VectorArray::try_from_iter(outputs.into_iter())?;
    <() as Guarded>::try_from_op(|_| unsafe { ironbark_mlx_sys::mlx_async_eval(vec.as_ptr()) })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array;

    #[test]
    fn test_eval() {
        let a = array!([1.0f32, 2.0]);
        let b = &a + &a;
        eval([&b]).unwrap();
        assert_eq!(b.as_slice::<f32>(), &[2.0, 4.0]);
    }

    #[test]
    fn test_async_eval() {
        let a = array!([1.0f32, 2.0]);
        let b = &a * &a;
        async_eval([&b]).unwrap();
        assert_eq!(b.as_slice::<f32>(), &[1.0, 4.0]);
    }
}
