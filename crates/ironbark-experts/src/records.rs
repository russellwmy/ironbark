//! Where the loaders read expert records: by default the local `experts.bin`; any other
//! [`Records`] (e.g. one that fetches missing records from a server, `ironbark-fetch`) can stand in.

use std::fmt;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::Arc;

use crate::{Error, Layout, Result};

/// Expert records by index (`l * experts + e`), each `record_bytes` long.
pub trait Records: Send + Sync + fmt::Debug {
    /// Read record `i` into `buf` (`record_bytes` long). Called from several loader threads.
    ///
    /// # Errors
    ///
    /// Any read error; the store then stops serving (see [`crate::Store::wait`]).
    fn read(&self, i: usize, buf: &mut [u8]) -> io::Result<()>;
}

/// A complete `experts.bin`, read uncached.
#[derive(Debug)]
pub(crate) struct Local {
    pub(crate) file: File,
    pub(crate) stride: u64,
}

impl Records for Local {
    fn read(&self, i: usize, buf: &mut [u8]) -> io::Result<()> {
        self.file.read_exact_at(buf, i as u64 * self.stride)
    }
}

/// The complete `experts.bin` in `dir` (laid out as `dir/experts.json` says), read uncached.
///
/// # Errors
///
/// Fails if `experts.json` / `experts.bin` cannot be read, or the file is shorter than the
/// layout needs (e.g. still being copied).
pub fn local(dir: &Path) -> Result<Arc<dyn Records>> {
    let layout = Layout::read(dir)?;
    let path = dir.join("experts.bin");
    let file = File::open(&path).map_err(Error::io(&path))?;
    // Bypass the OS page cache: every miss costs a real storage read, as on a phone
    // where the cache would be the same RAM we are trying to save.
    crate::layout::no_cache(&file).map_err(Error::io(&path))?;
    // A truncated file (e.g. still being copied) would fail reads in a loader thread, where
    // a waiting task cannot see the error: refuse it here.
    let need = ((layout.layers * layout.experts - 1) as u64) * layout.stride + layout.record_bytes;
    let have = file.metadata().map_err(Error::io(&path))?.len();
    if have < need {
        return Err(Error::format(
            &path,
            format!("{have} bytes, {need} expected: incomplete (still copying?) or not packed for this layout"),
        ));
    }
    Ok(Arc::new(Local {
        file,
        stride: layout.stride,
    }))
}

/// Fixed-size chunks of a file that may still be arriving, e.g. a sparse `embed.bin` that fills
/// from a server (`ironbark-fetch`). The engine makes the chunks it is about to read present.
pub trait Chunks: Send + Sync + fmt::Debug {
    /// Make chunk `i` present in the local file, fetching it if missing. Called from the decode
    /// thread; blocks until the chunk is on disk.
    ///
    /// # Errors
    ///
    /// Fails if the chunk cannot be fetched.
    fn ensure(&self, i: usize) -> io::Result<()>;

    /// Whether chunk `i` is present.
    fn has(&self, i: usize) -> bool;

    /// Whether every chunk is present (then nothing needs ensuring).
    fn complete(&self) -> bool;
}
