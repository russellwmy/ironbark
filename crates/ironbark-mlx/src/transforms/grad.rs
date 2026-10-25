//! Reverse-mode gradients: [`value_and_grad`] of a function of arrays, and [`stop_gradient`].
//!
//! ironbark never trains; this is for tools that fine-tune models for it (ironwood).

use crate::{
    Array, Stream,
    error::{Exception, Result},
    utils::{Closure, VectorArray, guard::Guarded},
};

/// `a` with no gradient flowing back through it (e.g. indices chosen from scores).
///
/// # Errors
///
/// Fails if the MLX operation fails.
pub fn stop_gradient(a: impl AsRef<Array>) -> Result<Array> {
    let stream = Stream::thread_local_or_default();
    // SAFETY: `a` and `stream` are valid for the call.
    Array::try_from_op(|res| unsafe {
        ironbark_mlx_sys::mlx_stop_gradient(res, a.as_ref().as_ptr(), stream.as_ref().as_ptr())
    })
}

/// The outputs of `f(inputs)` and the gradients of its first output (a scalar) with respect
/// to `inputs[i]` for each `i` in `argnums`, in that order.
///
/// Only `f`'s arguments are differentiated: arrays it captures are constants.
///
/// # Errors
///
/// Fails if `f` fails, its first output is not a scalar, or an op has no gradient.
pub fn value_and_grad<F>(f: F, inputs: &[Array], argnums: &[i32]) -> Result<(Vec<Array>, Vec<Array>)>
where
    F: FnMut(&[Array]) -> std::result::Result<Vec<Array>, Exception>,
{
    let closure = Closure::new_fallible(f);
    // SAFETY: a new, empty closure object; freed below.
    let mut vag = unsafe { ironbark_mlx_sys::mlx_closure_value_and_grad_new() };
    // SAFETY: `closure` is a valid closure; `argnums` outlives the call.
    let made = <() as Guarded>::try_from_op(|_| unsafe {
        ironbark_mlx_sys::mlx_value_and_grad(&mut vag, closure.as_ptr(), argnums.as_ptr(), argnums.len())
    });
    let out = made.and_then(|()| {
        let input = VectorArray::try_from_iter(inputs.iter())?;
        // SAFETY: `vag` was made above and `closure` (which it calls) is still alive.
        <(Vec<Array>, Vec<Array>) as Guarded>::try_from_op(|(values, grads)| unsafe {
            ironbark_mlx_sys::mlx_closure_value_and_grad_apply(values, grads, vag, input.as_ptr())
        })
    });
    // SAFETY: `vag` is not used after this.
    unsafe { ironbark_mlx_sys::mlx_closure_value_and_grad_free(vag) };
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fast;
    use crate::ops::{dequantize, gather_qmm, quantize, quantized_matmul};

    fn close(a: &Array, b: &Array, tol: f32) -> bool {
        let d = a.subtract(b).unwrap().abs().unwrap().max(None).unwrap();
        d.item_exact::<f32>() < tol
    }

    #[test]
    fn grad_of_a_sum_of_squares_is_twice_x() {
        let x = Array::from_slice(&[1.0f32, -2.0, 3.0], &[3]);
        let (v, g) = value_and_grad(|a| Ok(vec![a[0].square()?.sum(None)?]), std::slice::from_ref(&x), &[0]).unwrap();
        assert!((v[0].item_exact::<f32>() - 14.0).abs() < 1e-5);
        assert!(close(&g[0], &x.multiply(Array::from_f32(2.0)).unwrap(), 1e-5));
    }

    #[test]
    fn stop_gradient_blocks_the_gradient() {
        let x = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let f = |a: &[Array]| Ok(vec![a[0].multiply(stop_gradient(&a[0])?)?.sum(None)?]);
        let (_, g) = value_and_grad(f, std::slice::from_ref(&x), &[0]).unwrap();
        assert!(close(&g[0], &x, 1e-6));
    }

    #[test]
    fn grads_of_two_args_and_captured_constants() {
        let (x, y) = (Array::from_slice(&[1.0f32, 2.0], &[2]), Array::from_slice(&[3.0f32, 5.0], &[2]));
        let c = Array::from_slice(&[10.0f32, 20.0], &[2]);
        let f = |a: &[Array]| Ok(vec![a[0].multiply(&a[1])?.multiply(&c)?.sum(None)?]);
        let (_, g) = value_and_grad(f, &[x.clone(), y.clone()], &[0, 1]).unwrap();
        assert!(close(&g[0], &y.multiply(&c).unwrap(), 1e-5));
        assert!(close(&g[1], &x.multiply(&c).unwrap(), 1e-5));
    }

    /// Training goes through 4-bit matmuls and the experts' gather: their gradient with
    /// respect to the input must equal that of the dequantized matmul.
    #[test]
    fn grad_flows_through_quantized_matmuls_and_rms_norm() {
        let w = crate::random::normal::<f32>(&[64, 128], None, None, None).unwrap();
        let (wq, s, b) = quantize(&w, 64, 4).unwrap();
        let wd = dequantize(&wq, &s, &b, 64, 4).unwrap();
        let x = crate::random::normal::<f32>(&[3, 128], None, None, None).unwrap();
        let norm = Array::from_slice(&vec![1.0f32; 128], &[128]);
        let q = |a: &[Array]| {
            let h = fast::rms_norm(&a[0], Some(&norm), 1e-6)?;
            Ok(vec![quantized_matmul(&h, &wq, &s, &b, true, 64, 4)?.square()?.sum(None)?])
        };
        let d = |a: &[Array]| {
            let h = fast::rms_norm(&a[0], Some(&norm), 1e-6)?;
            Ok(vec![h.matmul(wd.transpose()?)?.square()?.sum(None)?])
        };
        let (_, gq) = value_and_grad(q, std::slice::from_ref(&x), &[0]).unwrap();
        let (_, gd) = value_and_grad(d, std::slice::from_ref(&x), &[0]).unwrap();
        assert!(close(&gq[0], &gd[0], 1e-2));

        // Two "experts": the same weights stacked; each row of x routed to expert 1.
        let (ws, ss, bs) = (
            crate::ops::stack(&[&wq, &wq], 0).unwrap(),
            crate::ops::stack(&[&s, &s], 0).unwrap(),
            crate::ops::stack(&[&b, &b], 0).unwrap(),
        );
        let idx = Array::from_slice(&[1u32, 1, 1], &[3, 1]);
        let g = |a: &[Array]| {
            let xe = a[0].reshape(&[3, 1, 1, 128])?;
            Ok(vec![gather_qmm(&xe, &ws, &ss, &bs, None, &idx, true, 64, 4, false)?.square()?.sum(None)?])
        };
        let (_, gg) = value_and_grad(g, std::slice::from_ref(&x), &[0]).unwrap();
        let plain = |a: &[Array]| Ok(vec![a[0].matmul(wd.transpose()?)?.square()?.sum(None)?]);
        let (_, gp) = value_and_grad(plain, std::slice::from_ref(&x), &[0]).unwrap();
        assert!(close(&gg[0], &gp[0], 1e-2));
    }
}
