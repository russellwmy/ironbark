//! Error type.

use std::path::PathBuf;

/// Engine error.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// MLX operation failed.
    #[error("mlx: {0}")]
    Mlx(#[from] ironbark_mlx::error::Exception),
    /// Reading a checkpoint file failed.
    #[error("{path}: {source}")]
    Io {
        /// File.
        path: PathBuf,
        /// Cause.
        source: std::io::Error,
    },
    /// Loading a safetensors shard failed.
    #[error("{path}: {msg}")]
    Load {
        /// File.
        path: PathBuf,
        /// Cause.
        msg: String,
    },
    /// The expert store failed (opening it, or loading an expert).
    #[error("expert store: {0}")]
    Store(#[from] ironbark_experts::Error),
    /// The checkpoint does not match what the engine expects.
    #[error("{0}")]
    Format(String),
}

/// Result with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;
