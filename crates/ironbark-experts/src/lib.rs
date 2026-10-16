//! Event-driven MoE expert store: experts stay in `experts.bin` and are streamed into a
//! page-aligned slab when the router (or a predictor) asks for them.
//!
//! ```text
//!   layout   experts.bin / experts.json: record size and stride
//!   slab     page-aligned slot memory (all unsafe memory access)
//!   records  where loaders read records: experts.bin, or any `Records` (e.g. ironbark-fetch)
//!   store    events, loader pool, per-layer LRU, pinning
//! ```

pub mod error;
pub mod layout;
pub mod records;
pub mod slab;
pub mod store;

pub use error::{Error, Result};
pub use layout::Layout;
pub use records::{Chunks, Records};
pub use slab::{PAGE_BYTES, Slab};
pub use store::{ExpertRef, Pin, Stats, Store};
