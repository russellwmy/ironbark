//! Text embeddings with Qwen3-Embedding ([`Embedder`]).
//!
//! ```text
//!   text ──▶ (query: instruction prefix) ──▶ tokens + <|endoftext|>, cut to max_tokens
//!        ──▶ EmbedModel::forward ──▶ unit vector [1024] ──▶ first `dims`, renormalized
//! ```

use std::path::Path;

use ironbark_engine::EmbedModel;
use tokenizers::Tokenizer;

use crate::{Error, Result, tok_err};

/// The instruction the model was trained with for queries (sentence-transformers' `query`
/// prompt); documents get none.
pub const QUERY_PREFIX: &str = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";

/// Token that ends every text: its hidden state is the embedding.
const END: &str = "<|endoftext|>";

/// MLX's buffer cache limit while an embedder is open: freed activations go back to the system.
const BUFFER_CACHE: usize = 64 << 20;

/// What a text is, for retrieval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedKind {
    /// A search query: prefixed with [`QUERY_PREFIX`].
    Query,
    /// A passage to search: embedded as is.
    Document,
}

/// How to open an [`Embedder`].
#[derive(Debug, Clone, Copy)]
pub struct EmbedOptions {
    /// Most tokens of a text (longer ones are cut, keeping the end token).
    pub max_tokens: usize,
    /// Dims of each vector (Matryoshka: the first `dims`, renormalized). `None`: all (1024).
    pub dims: Option<usize>,
}

impl Default for EmbedOptions {
    fn default() -> Self {
        Self {
            max_tokens: 512,
            dims: None,
        }
    }
}

/// A Qwen3-Embedding model open for embedding. Dropping it frees its memory.
#[derive(Debug)]
pub struct Embedder {
    /// Taken on drop, so its buffers are freed before MLX's cache is cleared.
    model: Option<EmbedModel>,
    tok: Tokenizer,
    end: u32,
    max_tokens: usize,
    dims: usize,
}

impl Embedder {
    /// Open the model in `dir` (an `mlx_lm.convert -q` output of Qwen3-Embedding).
    ///
    /// # Errors
    ///
    /// Fails if a file is missing, `dims` is 0 or above the model's, or MLX cannot load it.
    pub fn open(dir: impl AsRef<Path>, options: EmbedOptions) -> Result<Self> {
        let dir = dir.as_ref();
        // Lower the limit, never raise it (a session may have set a smaller one).
        let before = ironbark_mlx::memory::set_cache_limit(BUFFER_CACHE)?;
        if before < BUFFER_CACHE {
            ironbark_mlx::memory::set_cache_limit(before)?;
        }
        let model = EmbedModel::load(dir)?;
        let hidden = model.config().hidden;
        let dims = options.dims.unwrap_or(hidden);
        if dims == 0 || dims > hidden {
            return Err(Error::Model(ironbark_engine::Error::Format(format!("dims {dims}: 1 to {hidden}"))));
        }
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(tok_err)?;
        let end = tok.token_to_id(END).ok_or_else(|| Error::Tokenizer(format!("no {END} token")))?;
        Ok(Self {
            model: Some(model),
            tok,
            end,
            max_tokens: options.max_tokens.max(1),
            dims,
        })
    }

    /// Dims of each vector.
    #[must_use]
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// The token ids embedded for `text`: [`QUERY_PREFIX`] for a query, the text, cut to
    /// `max_tokens - 1`, then `<|endoftext|>`.
    ///
    /// # Errors
    ///
    /// Fails if the tokenizer fails.
    pub fn tokens(&self, text: &str, kind: EmbedKind) -> Result<Vec<u32>> {
        let text = match kind {
            EmbedKind::Query => format!("{QUERY_PREFIX}{text}"),
            EmbedKind::Document => text.to_owned(),
        };
        let mut ids = self.tok.encode(text, false).map_err(tok_err)?.get_ids().to_vec();
        ids.truncate(self.max_tokens - 1);
        ids.push(self.end);
        Ok(ids)
    }

    /// One unit vector of [`Embedder::dims`] per text; their dot product is the cosine
    /// similarity.
    ///
    /// # Errors
    ///
    /// Fails if the tokenizer or the model fails.
    pub fn embed(&mut self, texts: &[&str], kind: EmbedKind) -> Result<Vec<Vec<f32>>> {
        let model = self.model.as_ref().ok_or(Error::NoTurn)?;
        let hidden = model.config().hidden;
        texts
            .iter()
            .map(|t| {
                let v = model.forward(&self.tokens(t, kind)?)?;
                v.eval()?;
                let mut v = v.as_slice::<f32>()[..self.dims].to_vec();
                if self.dims < hidden {
                    let len = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
                    for x in &mut v {
                        *x /= len;
                    }
                }
                Ok(v)
            })
            .collect()
    }
}

impl Drop for Embedder {
    fn drop(&mut self) {
        drop(self.model.take());
        // Return the weights' buffers, now in MLX's cache, to the system.
        let _ = ironbark_mlx::memory::clear_cache();
    }
}

/// True if `dir` holds a Qwen3 dense checkpoint, opened by [`Embedder::open`].
#[must_use]
pub fn is_embedder(dir: &Path) -> bool {
    ironbark_engine::is_embedder(dir)
}
