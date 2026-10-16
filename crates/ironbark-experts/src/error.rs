//! Error type for the expert store.

use std::path::PathBuf;

/// Result alias for this crate.
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Why the expert files could not be read.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A file could not be opened or read.
    #[error("read {path}")]
    Io {
        /// File that failed.
        path: PathBuf,
        /// Underlying I/O error.
        source: std::io::Error,
    },
    /// `experts.json` is not valid JSON.
    #[error("parse {path}")]
    Json {
        /// File that failed.
        path: PathBuf,
        /// Underlying parse error.
        source: serde_json::Error,
    },
    /// An expert could not be loaded; the store stops serving (see [`crate::Store::wait`]).
    #[error("expert load failed: {0}")]
    Load(String),
    /// The file is valid JSON but not the expected shape.
    #[error("{path}: {what}")]
    Format {
        /// File that failed.
        path: PathBuf,
        /// What was wrong.
        what: String,
    },
}

impl Error {
    /// Wrap an I/O error with the file it came from.
    pub fn io(path: impl Into<PathBuf>) -> impl FnOnce(std::io::Error) -> Self {
        let path = path.into();
        move |source| Error::Io { path, source }
    }

    /// Wrap a JSON error with the file it came from.
    pub fn json(path: impl Into<PathBuf>) -> impl FnOnce(serde_json::Error) -> Self {
        let path = path.into();
        move |source| Error::Json { path, source }
    }

    /// A shape error in `path`.
    pub fn format(path: impl Into<PathBuf>, what: impl Into<String>) -> Self {
        Error::Format {
            path: path.into(),
            what: what.into(),
        }
    }
}
