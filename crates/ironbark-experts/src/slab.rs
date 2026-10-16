//! Page-aligned expert memory: one allocation split into fixed-size slots.
//!
//! All `unsafe` for expert memory lives here. The slab itself does not know who may touch a
//! slot; the store's state machine (under its mutex) guarantees that a slot is either written
//! by exactly one loader (`Loading`) or read by any number of compute tasks (`Ready`), never both.

use std::ptr::NonNull;

/// Bytes per page on Apple Silicon. Slots start on page boundaries, so a slot can later be
/// wrapped as a no-copy GPU buffer and read with `F_NOCACHE` without bounce copies.
pub const PAGE_BYTES: usize = 16384;

/// Fixed-size, page-aligned slots in one zeroed anonymous mapping.
///
/// Pages are mapped lazily by the OS, so RSS grows only as slots are filled. The mapping is
/// unmapped on drop, so the memory leaves the process footprint at once (the allocator would
/// keep a freed gigabyte cached).
#[derive(Debug)]
pub struct Slab {
    ptr: NonNull<u8>,
    size: usize,
    slot_bytes: usize,
    slots: usize,
}

// SAFETY: the slab is plain memory; access rules are enforced by the store's state machine
// (see the module doc), so sharing the pointer across threads is sound.
unsafe impl Send for Slab {}
// SAFETY: as above; `&Slab` only hands out slices through `unsafe` methods whose callers
// guarantee no write overlaps a read of the same slot.
unsafe impl Sync for Slab {}

impl Slab {
    /// Allocate `slots` slots of `slot_bytes` each (`slot_bytes` must be a multiple of
    /// [`PAGE_BYTES`]).
    ///
    /// # Panics
    ///
    /// If `slot_bytes` is not page-aligned, or the memory cannot be mapped.
    #[must_use]
    pub fn new(slots: usize, slot_bytes: usize) -> Self {
        assert!(slot_bytes > 0 && slot_bytes.is_multiple_of(PAGE_BYTES), "slot size must be whole pages");
        let size = slots.max(1).saturating_mul(slot_bytes);
        // SAFETY: a fresh private anonymous mapping; no existing memory is touched.
        let raw = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert!(raw != libc::MAP_FAILED, "cannot map a slab of {size} bytes");
        // Anonymous mappings are zeroed and page-aligned (16 KiB pages on Apple Silicon).
        let Some(ptr) = NonNull::new(raw.cast::<u8>()) else {
            unreachable!("mmap returned null")
        };
        Slab {
            ptr,
            size,
            slot_bytes,
            slots,
        }
    }

    /// Number of slots.
    #[must_use]
    pub fn len(&self) -> usize {
        self.slots
    }

    /// Whether the slab has no slots.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots == 0
    }

    /// Bytes per slot.
    #[must_use]
    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// Start of the allocation (page-aligned; `len() * slot_bytes()` bytes, whole pages), for
    /// wrapping the slab as a no-copy GPU buffer. Whoever reads through it must follow the
    /// slot rules in the module doc: read only slots that are pinned `Ready`.
    #[must_use]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr.as_ptr()
    }

    /// Start of slot `i`.
    fn base(&self, i: usize) -> *mut u8 {
        assert!(i < self.slots, "slot {i} out of range ({})", self.slots);
        // SAFETY: `i < slots`, so the offset stays inside the allocation.
        unsafe { self.ptr.as_ptr().add(i * self.slot_bytes) }
    }

    /// Writable bytes of slot `i`.
    ///
    /// # Safety
    ///
    /// The caller must have exclusive access to slot `i` for the lifetime of the slice:
    /// no other `slot_mut` or `slot` of the same slot may be live.
    #[expect(clippy::mut_from_ref, reason = "exclusivity per slot is the caller's contract")]
    #[must_use]
    pub unsafe fn slot_mut(&self, i: usize) -> &mut [u8] {
        let p = self.base(i);
        // SAFETY: `p` points to `slot_bytes` bytes inside the allocation, initialized (zeroed
        // at allocation); the caller guarantees exclusive access.
        unsafe { std::slice::from_raw_parts_mut(p, self.slot_bytes) }
    }

    /// The first `len` bytes of slot `i`.
    ///
    /// # Safety
    ///
    /// No `slot_mut` of slot `i` may be live while the returned slice is.
    ///
    /// # Panics
    ///
    /// If `len` bytes do not fit in a slot.
    #[must_use]
    pub unsafe fn slot(&self, i: usize, len: usize) -> &[u8] {
        assert!(len <= self.slot_bytes, "{len} bytes do not fit in a slot");
        let p = self.base(i);
        // SAFETY: `len` bytes from `p` are inside the slot and initialized; the caller
        // guarantees no concurrent writer.
        unsafe { std::slice::from_raw_parts(p, len) }
    }
}

impl Drop for Slab {
    fn drop(&mut self) {
        // SAFETY: `ptr` and `size` are the mapping made in `new`, unmapped once.
        unsafe { libc::munmap(self.ptr.as_ptr().cast(), self.size) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_page_aligned_and_disjoint() {
        let s = Slab::new(3, PAGE_BYTES * 2);
        // SAFETY: single-threaded test; each slice is dropped before the next is created.
        let a = unsafe { s.slot_mut(0) }.as_ptr() as usize;
        // SAFETY: as above.
        let c = unsafe { s.slot_mut(2) }.as_ptr() as usize;
        assert_eq!(a % PAGE_BYTES, 0);
        assert_eq!(c - a, 2 * 2 * PAGE_BYTES);
    }

    #[test]
    fn write_then_read() {
        let s = Slab::new(2, PAGE_BYTES);
        // SAFETY: single-threaded test, no other view of slot 1 is live.
        let w = unsafe { s.slot_mut(1) };
        w[..4].copy_from_slice(&[1, 0, 2, 0]);
        // SAFETY: the writable slice is no longer used.
        let r = unsafe { s.slot(1, 4) };
        assert_eq!(r, [1, 0, 2, 0]);
    }

    #[test]
    #[should_panic(expected = "whole pages")]
    fn rejects_unaligned_slot_size() {
        let _ = Slab::new(1, 100);
    }
}
