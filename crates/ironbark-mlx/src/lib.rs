//! Safe Rust API over [MLX](https://github.com/ml-explore/mlx) for inference on Apple silicon
//! (macOS and iOS).
//!
//! # Table of Contents
//!
//! - [Threading](#threading)
//! - [Basics](#basics)
//! - [Lazy evaluation](#lazy-evaluation)
//! - [Unified memory](#unified-memory)
//! - [Modules](#modules)
//!
//! # Threading
//!
//! [`Array`] values may move between threads. MLX 0.32.2 keeps default streams per thread, so each
//! worker must perform operations through its own default or thread-local scoped override.
//! [`Stream`] and compiled closures are thread-affine and cannot move or be shared across threads.
//! The scoped Rust override is thread-local, not asynchronous task-local, and does not propagate
//! across `.await`.
//!
//! # Basics
//!
//! ```rust
//! use ironbark_mlx::{array, Dtype};
//!
//! let a = array!([1, 2, 3, 4]);
//! assert_eq!(a.shape(), &[4]);
//! assert_eq!(a.dtype(), Dtype::Int32);
//!
//! let b = array!([1.0, 2.0, 3.0, 4.0]);
//! assert_eq!(b.dtype(), Dtype::Float32);
//! ```
//!
//! # Lazy evaluation
//!
//! Operations only record a compute graph. It runs on [`Array::eval`] or [`transforms::eval`], or
//! when the data is read ([`Array::item_exact`], [`Array::item_cast`], [`Array::as_slice`],
//! printing, saving).
//!
//! ```rust
//! use ironbark_mlx::array;
//!
//! let a = array!([1, 2, 3, 4]);
//! let b = array!([1.0, 2.0, 3.0, 4.0]);
//!
//! let c = &a + &b; // not evaluated yet
//! c.eval().unwrap(); // evaluates c
//!
//! let d = &a + &b;
//! let d_slice: &[f32] = d.as_slice(); // evaluates d
//! ```
//!
//! | Rule | Why |
//! |---|---|
//! | Evaluate once per step (e.g. once per decoded token) | each evaluation has a fixed cost; a few tens to thousands of ops per graph is fine |
//! | Avoid reading scalars for control flow inside a step | every read forces an evaluation |
//! | Outputs that are never used are never computed | but their graph is still built |
//!
//! # Unified memory
//!
//! CPU and GPU share one memory pool, so arrays have no device. The device is chosen per
//! operation (default stream, or a [`Stream`] passed to the `*_device` variants).
//!
//! # Modules
//!
//! | Module | What |
//! |---|---|
//! | [`ops`] | array operations, [`ops::indexing`], quantize / dequantize |
//! | [`fast`] | fused kernels: RMS norm, layer norm, RoPE, scaled dot-product attention |
//! | [`transforms`] | [`transforms::eval`], [`transforms::async_eval`], [`transforms::compile`] |
//! | [`memory`] | active / peak / cache memory, cache limit |
//! | [`random`], [`linalg`], [`fft`] | sampling and math |
//! | [`Array::load_safetensors`], [`Array::save_safetensors`], [`Array::load_numpy`] | file I/O |

#![deny(unused_unsafe, missing_debug_implementations, missing_docs)]
#![cfg_attr(test, allow(clippy::approx_constant))]

#[macro_use]
pub mod macros; // Must be first to ensure the other modules can use the macros

mod array;
pub mod builder;
mod device;
mod dtype;
pub mod error;
pub mod fast;
pub mod fft;
pub mod io;
pub mod linalg;
pub mod memory;
#[cfg(feature = "metal")]
pub mod metal;
pub mod nested;
pub mod ops;
mod options;
pub mod random;
mod stream;
pub mod transforms;
pub mod utils;

/// Test-only assertion support shared with the workspace integration tests.
#[doc(hidden)]
pub mod test_utils;

pub use array::*;
pub use device::*;
pub use dtype::*;
pub use options::*;
pub use stream::*;

pub(crate) mod constants {
    /// The default length of the stack-allocated vector in `SmallVec<[T; DEFAULT_STACK_VEC_LEN]>`
    pub(crate) const DEFAULT_STACK_VEC_LEN: usize = 4;
}

pub(crate) mod sealed {
    /// A marker trait to prevent external implementations of the `Sealed` trait.
    pub trait Sealed {}

    impl Sealed for () {}

    impl<A> Sealed for (A,) where A: Sealed {}
    impl<A, B> Sealed for (A, B)
    where
        A: Sealed,
        B: Sealed,
    {
    }
    impl<A, B, C> Sealed for (A, B, C)
    where
        A: Sealed,
        B: Sealed,
        C: Sealed,
    {
    }
}
