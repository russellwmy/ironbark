//! The next token: greedy, or sampled with temperature, top-k, top-p and a presence penalty.
//! All of it stays lazy on the GPU, so decode keeps queueing step t+1 before reading token t.
//!
//! ```text
//!   logits ─▶ − penalty · seen ─▶ top-k ─▶ ÷ temperature ─▶ top-p ─▶ categorical ─▶ token
//!                    ▲                                                                │
//!                    └──────────────────── seen[token] = 1 ◀──────────────────────────┘
//! ```

use ironbark_mlx::ops::indexing::{IndexOp, argmax_axis};
use ironbark_mlx::ops::{argpartition_axis, argsort_axis, cumsum, select, softmax_axis};
use ironbark_mlx::{Array, Dtype, random};

use crate::Result;

/// How the next token is chosen ([`crate::Session::set_sampling`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    /// Softmax temperature; 0: greedy (the most likely token).
    pub temperature: f32,
    /// Sample among this many most likely tokens; 0: all.
    pub top_k: usize,
    /// Then among the fewest tokens whose probability reaches this; 1: all.
    pub top_p: f32,
    /// Taken off the logit of each token the reply already holds (repeats less).
    pub presence_penalty: f32,
}

impl Sampling {
    /// The most likely token every step (Maple: bit-exact with DeepGrove's reference).
    pub const GREEDY: Self = Self {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        presence_penalty: 0.0,
    };

    /// Qwen3.5 / 3.6's non-thinking settings, thinking or not. Greedy, the ternary models loop,
    /// repeating a paragraph until the context fills (also with every expert loaded). The model
    /// card's thinking settings (temperature 1.0, top-p 0.95) end replies to tool results with
    /// an empty answer (2 of 16 on the Mac, 0 of 16 with these).
    pub const QWEN: Self = Self {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.8,
        presence_penalty: 1.5,
    };

    fn greedy(&self) -> bool {
        self.temperature <= 0.0
    }
}

/// [`Sampling`] with its random key and the tokens of the current reply.
#[derive(Debug)]
pub(crate) struct Sampler {
    key: Array,
    /// `[vocab]` f32: 1 for each token the reply holds; `None` before its first token.
    seen: Option<Array>,
    /// Tokens never picked ([`Sampler::ban`]), and their `[vocab]` mask once the vocabulary is
    /// known (0, or −∞ for a banned token; evaluated, so any thread can use it).
    banned: Vec<u32>,
    ban_mask: Option<Array>,
}

impl Sampler {
    pub(crate) fn new(seed: u64) -> Result<Self> {
        Ok(Self {
            key: random::key(seed)?,
            seen: None,
            banned: Vec::new(),
            ban_mask: None,
        })
    }

    /// Its state, lazy after [`Sampler::pick`]: evaluate it with the step, as the next call
    /// may come from another thread.
    pub(crate) fn pending(&self) -> impl Iterator<Item = &Array> {
        std::iter::once(&self.key).chain(&self.seen)
    }

    /// Never pick `tokens` (none: no ban).
    pub(crate) fn ban(&mut self, tokens: &[u32]) {
        if self.banned != tokens {
            self.banned = tokens.to_vec();
            self.ban_mask = None;
        }
    }

    /// The ban as a mask over `vocab` logits, if any.
    fn ban_mask(&mut self, vocab: i32) -> Result<Option<&Array>> {
        if self.banned.is_empty() {
            return Ok(None);
        }
        if self.ban_mask.as_ref().is_none_or(|m| m.shape()[0] != vocab) {
            let ids: Vec<u32> = self.banned.iter().copied().filter(|&t| (t as usize) < vocab as usize).collect();
            let n = i32::try_from(ids.len()).unwrap_or(i32::MAX);
            let mask = Array::zeros::<f32>(&[vocab])?.put_along_axis(
                Array::from_slice(&ids, &[n]),
                Array::full::<f32>(&[n], Array::from_f32(f32::NEG_INFINITY))?,
                0,
            )?;
            mask.eval()?;
            self.ban_mask = Some(mask);
        }
        Ok(self.ban_mask.as_ref())
    }

    /// A new reply: no token seen yet.
    pub(crate) fn restart(&mut self) {
        self.seen = None;
    }

    /// The next token of logits `[1, 1, vocab]` as a lazy `[1]` u32, marked seen.
    pub(crate) fn pick(&mut self, s: &Sampling, logits: &Array) -> Result<Array> {
        let mut l = logits.reshape(&[-1])?;
        let vocab = l.shape()[0];
        if let Some(mask) = self.ban_mask(vocab)? {
            l = l.as_dtype(Dtype::Float32)?.add(mask)?;
        }
        if s.greedy() && s.presence_penalty == 0.0 {
            return Ok(argmax_axis(&l, -1, None)?.reshape(&[1])?.as_dtype(Dtype::Uint32)?);
        }
        l = l.as_dtype(Dtype::Float32)?;
        if let Some(seen) = &self.seen
            && s.presence_penalty != 0.0
        {
            l = l.subtract(seen.multiply(Array::from_f32(s.presence_penalty))?)?;
        }
        let tok = if s.greedy() {
            argmax_axis(&l, -1, None)?.reshape(&[1])?.as_dtype(Dtype::Uint32)?
        } else {
            self.sample(s, &l, vocab)?
        };
        let seen = match self.seen.take() {
            Some(seen) => seen,
            None => Array::zeros::<f32>(&[vocab])?,
        };
        self.seen = Some(seen.put_along_axis(&tok, Array::from_slice(&[1.0f32], &[1]), 0)?);
        Ok(tok)
    }

    /// Sample from f32 logits `[vocab]`.
    fn sample(&mut self, s: &Sampling, l: &Array, vocab: i32) -> Result<Array> {
        let k = if s.top_k == 0 {
            vocab
        } else {
            i32::try_from(s.top_k).map_or(vocab, |k| k.min(vocab))
        };
        // The top k (unordered), then most likely first.
        let ids = if k < vocab {
            argpartition_axis(l, -k, -1)?.index(vocab - k..)
        } else {
            argsort_axis(l, -1)?
        };
        let vals = l.take_axis(&ids, 0)?;
        let order = argsort_axis(vals.negative()?, -1)?;
        let ids = ids.take_axis(&order, 0)?;
        let mut vals = vals.take_axis(&order, 0)?.divide(Array::from_f32(s.temperature))?;
        if s.top_p < 1.0 {
            // Keep a token while the ones before it hold less than top_p (the first always).
            let p = softmax_axis(&vals, -1, true)?;
            let before = cumsum(&p, -1, None, false)?;
            vals = select(before.lt(Array::from_f32(s.top_p))?, &vals, Array::from_f32(f32::NEG_INFINITY))?;
        }
        let (key, next) = random::split(&self.key, 2)?;
        self.key = next;
        let j = random::categorical(&vals, -1, None, &key)?.reshape(&[1])?;
        Ok(ids.take_axis(&j, 0)?.as_dtype(Dtype::Uint32)?)
    }
}
