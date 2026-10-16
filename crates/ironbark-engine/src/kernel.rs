//! Custom Metal kernels compiled and scheduled by MLX (`mx.fast.metal_kernel`), through the
//! MLX C API. `mlx-rs` has no safe wrapper for it, so all of the FFI lives here.
//!
//! ```text
//!   Kernel::new(name, inputs, outputs, source, header)    once; MLX compiles per template
//!   kernel.apply(inputs, templates, grid, group, outputs)  per call; returns lazy arrays
//! ```
//!
//! Each C object (string vector, array vector, call config) has a small owner that frees it
//! on drop, so early returns cannot leak.

use std::ffi::{CString, c_char};
use std::fmt;

use ironbark_mlx::{Array, Dtype, Stream};
use ironbark_mlx_sys as sys;

use crate::{Error, Result};

/// A compile-time template argument of a kernel.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Template {
    /// `constexpr int`.
    Int(&'static str, i32),
    /// `constexpr bool`.
    Bool(&'static str, bool),
    /// A type name (e.g. `T_` = `bfloat16_t`).
    Dtype(&'static str, Dtype),
}

/// One output of a kernel call.
pub(crate) struct Output<'a> {
    pub(crate) shape: &'a [i32],
    pub(crate) dtype: Dtype,
}

fn c_string(s: &str) -> CString {
    CString::new(s).unwrap_or_default()
}

/// Owned `mlx_vector_string`.
struct Names(sys::mlx_vector_string);

impl Names {
    fn new(names: &[&str]) -> Self {
        let owned: Vec<CString> = names.iter().map(|n| c_string(n)).collect();
        let mut ptrs: Vec<*const c_char> = owned.iter().map(|s| s.as_ptr()).collect();
        // SAFETY: `ptrs` points to NUL-terminated strings that live until the end of this
        // function; MLX copies them into the vector.
        Names(unsafe { sys::mlx_vector_string_new_data(ptrs.as_mut_ptr(), ptrs.len()) })
    }
}

impl Drop for Names {
    fn drop(&mut self) {
        // SAFETY: created by `mlx_vector_string_new_data`, freed only here.
        unsafe { sys::mlx_vector_string_free(self.0) };
    }
}

/// Owned `mlx_vector_array`.
struct Arrays(sys::mlx_vector_array);

impl Arrays {
    fn new() -> Self {
        // SAFETY: no arguments; returns an empty vector.
        Arrays(unsafe { sys::mlx_vector_array_new() })
    }

    fn from(arrays: &[&Array]) -> Self {
        let handles: Vec<sys::mlx_array> = arrays.iter().map(|a| a.as_ptr()).collect();
        // SAFETY: the handles are valid (borrowed arrays) for the call; MLX copies them (taking
        // its own references).
        Arrays(unsafe { sys::mlx_vector_array_new_data(handles.as_ptr(), handles.len()) })
    }

    fn get(&self, i: usize) -> Option<Array> {
        // SAFETY: no arguments; returns an empty array handle.
        let mut a = unsafe { sys::mlx_array_new() };
        // SAFETY: `a` is a valid handle to overwrite and `self.0` a valid vector.
        let rc = unsafe { sys::mlx_vector_array_get(&raw mut a, self.0, i) };
        // SAFETY: `a` is a valid handle either way; the Array takes ownership and frees it.
        let a = unsafe { Array::from_ptr(a) };
        (rc == 0).then_some(a)
    }
}

impl Drop for Arrays {
    fn drop(&mut self) {
        // SAFETY: created by `mlx_vector_array_new*`, freed only here.
        unsafe { sys::mlx_vector_array_free(self.0) };
    }
}

/// Owned `mlx_fast_metal_kernel_config` (one call's templates, grid and outputs).
struct Config(sys::mlx_fast_metal_kernel_config);

impl Config {
    fn new() -> Self {
        // SAFETY: no arguments; returns a fresh config.
        Config(unsafe { sys::mlx_fast_metal_kernel_config_new() })
    }

    fn template(&self, t: Template) -> bool {
        match t {
            Template::Int(name, v) => {
                let name = c_string(name);
                // SAFETY: valid config; `name` is NUL-terminated and copied by MLX.
                unsafe { sys::mlx_fast_metal_kernel_config_add_template_arg_int(self.0, name.as_ptr(), v) == 0 }
            }
            Template::Bool(name, v) => {
                let name = c_string(name);
                // SAFETY: as above.
                unsafe { sys::mlx_fast_metal_kernel_config_add_template_arg_bool(self.0, name.as_ptr(), v) == 0 }
            }
            Template::Dtype(name, v) => {
                let name = c_string(name);
                // SAFETY: as above; `v` is a valid MLX dtype.
                unsafe {
                    sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(self.0, name.as_ptr(), v as sys::mlx_dtype)
                        == 0
                }
            }
        }
    }

    fn grid(&self, (x, y, z): (i32, i32, i32)) -> bool {
        // SAFETY: valid config; plain integers.
        unsafe { sys::mlx_fast_metal_kernel_config_set_grid(self.0, x, y, z) == 0 }
    }

    fn group(&self, (x, y, z): (i32, i32, i32)) -> bool {
        // SAFETY: valid config; plain integers.
        unsafe { sys::mlx_fast_metal_kernel_config_set_thread_group(self.0, x, y, z) == 0 }
    }

    fn output(&self, o: &Output<'_>) -> bool {
        // SAFETY: valid config; `shape` points to `len` ints, copied by MLX.
        unsafe {
            sys::mlx_fast_metal_kernel_config_add_output_arg(
                self.0,
                o.shape.as_ptr(),
                o.shape.len(),
                o.dtype as sys::mlx_dtype,
            ) == 0
        }
    }
}

impl Drop for Config {
    fn drop(&mut self) {
        // SAFETY: created by `mlx_fast_metal_kernel_config_new`, freed only here.
        unsafe { sys::mlx_fast_metal_kernel_config_free(self.0) };
    }
}

/// A Metal kernel (source + argument names); MLX compiles it per template instance.
pub(crate) struct Kernel {
    name: &'static str,
    raw: sys::mlx_fast_metal_kernel,
}

impl fmt::Debug for Kernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Kernel({})", self.name)
    }
}

impl Kernel {
    /// Define a kernel. `source` is the body of the Metal function; `header` is included above
    /// it. Inputs are made row-contiguous before each call.
    pub(crate) fn new(name: &'static str, inputs: &[&str], outputs: &[&str], source: &str, header: &str) -> Self {
        let (ins, outs) = (Names::new(inputs), Names::new(outputs));
        let (c_name, c_source, c_header) = (c_string(name), c_string(source), c_string(header));
        // SAFETY: all strings are NUL-terminated and alive for the call; MLX copies the names,
        // source and header into the kernel.
        let raw = unsafe {
            sys::mlx_fast_metal_kernel_new(
                c_name.as_ptr(),
                ins.0,
                outs.0,
                c_source.as_ptr(),
                c_header.as_ptr(),
                true,
                false,
            )
        };
        Kernel { name, raw }
    }

    /// Schedule the kernel: `grid` is in threads (not threadgroups), as in `mx.fast.metal_kernel`.
    /// The outputs are lazy MLX arrays.
    pub(crate) fn apply(
        &self,
        inputs: &[&Array],
        templates: &[Template],
        grid: (i32, i32, i32),
        group: (i32, i32, i32),
        outputs: &[Output<'_>],
    ) -> Result<Vec<Array>> {
        let fail = |what: &str| Error::Format(format!("metal kernel {}: {what} failed", self.name));
        let config = Config::new();
        let mut ok = config.grid(grid) && config.group(group);
        for &t in templates {
            ok &= config.template(t);
        }
        for o in outputs {
            ok &= config.output(o);
        }
        if !ok {
            return Err(fail("configure"));
        }
        let (ins, mut outs) = (Arrays::from(inputs), Arrays::new());
        let stream = Stream::thread_local_or_default();
        // SAFETY: kernel, inputs, config and stream are valid; MLX stores the outputs in the
        // vector owned by `outs`, which frees it on drop.
        let rc =
            unsafe { sys::mlx_fast_metal_kernel_apply(&raw mut outs.0, self.raw, ins.0, config.0, stream.as_ptr()) };
        if rc != 0 {
            return Err(fail("apply"));
        }
        (0..outputs.len()).map(|i| outs.get(i).ok_or_else(|| fail("read output"))).collect()
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        // SAFETY: `raw` was created by `mlx_fast_metal_kernel_new` and is freed only here.
        unsafe { sys::mlx_fast_metal_kernel_free(self.raw) };
    }
}

/// Releases the `Arc<T>` handed to MLX by [`wrap`] when MLX frees the array.
unsafe extern "C" fn release<T>(payload: *mut std::ffi::c_void) {
    // SAFETY: `payload` came from `Arc::into_raw` in `wrap` and MLX calls this exactly once.
    drop(unsafe { std::sync::Arc::from_raw(payload.cast_const().cast::<T>()) });
}

/// `bytes` at `ptr` as an MLX array of `shape` and `dtype` over the same memory: MLX's Metal
/// allocator wraps page-aligned memory of whole pages without copying (`newBuffer(bytesNoCopy:)`).
/// The array holds a reference to `owner`, which keeps the memory valid.
///
/// # Safety
///
/// `ptr` must be page-aligned and cover `shape · dtype` bytes (whole pages) for as long as
/// `owner` lives.
unsafe fn wrap<T: Send + Sync>(
    ptr: *const u8,
    shape: &[i32],
    dtype: Dtype,
    owner: &std::sync::Arc<T>,
) -> Result<Array> {
    let payload = std::sync::Arc::into_raw(owner.clone()).cast_mut().cast::<std::ffi::c_void>();
    // SAFETY: the caller keeps `ptr` valid while `payload` holds the owner; MLX calls
    // `release` once when the array's buffer is freed.
    let raw = unsafe {
        sys::mlx_array_new_data_managed_payload(
            ptr.cast_mut().cast(),
            shape.as_ptr(),
            shape.len() as i32,
            dtype as sys::mlx_dtype,
            payload,
            Some(release::<T>),
        )
    };
    // SAFETY: `raw` is a fresh array handle; the Array takes ownership.
    let a = unsafe { Array::from_ptr(raw) };
    // MLX copies when it cannot wrap the memory: the point is not to.
    // SAFETY: `a` is a valid array with data.
    let data = unsafe { sys::mlx_array_data_uint8(a.as_ptr()) };
    if a.ndim() != shape.len() || data != ptr {
        return Err(Error::Format("could not wrap memory as an MLX array without a copy".into()));
    }
    Ok(a)
}

/// The slab of `store` as one MLX array `[slots, stride / 4]` (u32) over the same memory (no
/// copy). The array holds a reference to `store`, so the memory outlives every kernel that
/// reads it.
///
/// Kernels may only read slots pinned by an [`ironbark_experts::ExpertRef`]; loader threads write
/// other slots concurrently.
pub(crate) fn wrap_slab(store: &std::sync::Arc<ironbark_experts::Store>) -> Result<Array> {
    let slab = store.slab();
    let stride = slab.slot_bytes();
    let shape = [slab.len() as i32, (stride / 4) as i32];
    // SAFETY: the slab covers `len * slot_bytes` bytes (whole pages, page-aligned) and stays
    // valid while the store lives.
    unsafe { wrap(slab.as_ptr().cast(), &shape, Dtype::Uint32, store) }
}

/// A read-only memory map of a whole file, unmapped on drop.
pub(crate) struct Mapped {
    ptr: *mut libc::c_void,
    len: usize,
}

// SAFETY: the mapping is read-only and owned; nothing writes through `ptr`.
unsafe impl Send for Mapped {}
// SAFETY: as above.
unsafe impl Sync for Mapped {}

impl Mapped {
    /// Map `path` read-only. Pages are read from the file when first touched, and as clean
    /// file-backed memory the system may drop them again under pressure.
    pub(crate) fn open(path: &std::path::Path) -> Result<std::sync::Arc<Self>> {
        use std::os::fd::AsRawFd;
        let io = |source| Error::Io {
            path: path.to_path_buf(),
            source,
        };
        let file = std::fs::File::open(path).map_err(io)?;
        let len = file.metadata().map_err(io)?.len() as usize;
        // SAFETY: a fresh read-only shared mapping of an open file; checked below.
        let ptr =
            unsafe { libc::mmap(std::ptr::null_mut(), len, libc::PROT_READ, libc::MAP_SHARED, file.as_raw_fd(), 0) };
        if ptr == libc::MAP_FAILED {
            return Err(io(std::io::Error::last_os_error()));
        }
        Ok(std::sync::Arc::new(Mapped { ptr, len }))
    }

    /// File length in bytes.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// `shape` of `dtype` at byte `offset` as an MLX array over the mapping (no copy).
    pub(crate) fn array(self: &std::sync::Arc<Self>, offset: usize, shape: &[i32], dtype: Dtype) -> Result<Array> {
        let item = match dtype {
            Dtype::Uint8 | Dtype::Int8 | Dtype::Bool => 1,
            Dtype::Uint16 | Dtype::Int16 | Dtype::Float16 | Dtype::Bfloat16 => 2,
            Dtype::Uint64 | Dtype::Int64 | Dtype::Float64 | Dtype::Complex64 => 8,
            _ => 4,
        };
        let bytes = shape.iter().map(|&d| d as usize).product::<usize>() * item;
        let page = crate::pack::PAGE;
        if !offset.is_multiple_of(page) || !bytes.is_multiple_of(page) || offset + bytes > self.len {
            return Err(Error::Format(format!("mapped region {offset}+{bytes} is not whole pages of the file")));
        }
        // SAFETY: `offset` lies in the mapping (checked above).
        let ptr = unsafe { self.ptr.cast::<u8>().add(offset) };
        // SAFETY: the region lies in the mapping (checked), page-aligned (mmap returns a page
        // boundary), whole pages, and the array keeps the mapping alive.
        unsafe { wrap(ptr, shape, dtype, self) }
    }
}

impl Drop for Mapped {
    fn drop(&mut self) {
        // SAFETY: mapped in `open`, unmapped only here, after every array over it is gone.
        unsafe { libc::munmap(self.ptr, self.len) };
    }
}
