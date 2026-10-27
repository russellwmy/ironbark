//! The model behind a [`crate::Session`]: Maple or Qwen3.5 / 3.6, with the few
//! operations a session needs from any of them.
//!
//! ```text
//!   Session ──▶ Engine::Maple(Model)          ternary MoE, FlashHead, profile
//!           └─▶ Engine::Qwen(qwen::Model)     Qwen3.5 / 3.6 MoE (pack, or MLX checkpoint, resident)
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use ironbark_engine::bench::Decode;
use ironbark_engine::dynamic::Dynamic;
use ironbark_engine::{KvSnapshot, Model, StepTiming, qwen};
use ironbark_experts::{Records, Store};
use ironbark_mlx::Array;

use crate::{ExpertStats, Options, Plan, Profile, Result};

/// True if `dir` holds a Qwen3.5 / 3.6 MoE checkpoint.
#[must_use]
pub fn is_qwen(dir: &Path) -> bool {
    qwen::is_qwen(dir)
}

pub(crate) enum Engine {
    Maple(Box<Model>),
    Qwen(Box<qwen::Model>),
}

/// Saved caches of an [`Engine`].
#[derive(Debug, Clone)]
pub(crate) enum Cache {
    Maple(KvSnapshot),
    Qwen(qwen::Snapshot),
}

impl Cache {
    /// The caches by name, the position and the rotary offset (Maple: 0), to save.
    pub(crate) fn parts(&self) -> (Vec<(String, Array)>, usize, i64) {
        match self {
            Cache::Maple(c) => (c.arrays(), c.pos(), 0),
            Cache::Qwen(c) => (c.arrays(), c.pos(), c.delta()),
        }
    }
}

impl std::fmt::Debug for Engine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Engine::Maple(m) => m.fmt(f),
            Engine::Qwen(m) => m.fmt(f),
        }
    }
}

impl Engine {
    /// Plan the packed model in `dir` for `budget` bytes.
    pub(crate) fn plan(dir: &Path, budget: u64, context: usize) -> Result<Plan> {
        if is_qwen(dir) {
            let state = qwen::Model::state_bytes(&qwen::Config::read(dir)?, context);
            if dir.join("experts.json").exists() {
                Ok(Plan::with_state(dir, budget, state)?)
            } else {
                Ok(Plan::resident(dir, budget, state)?)
            }
        } else {
            Ok(Plan::new(dir, budget, context)?)
        }
    }

    /// Load the packed model in `dir` with `slots` expert cache slots per layer.
    pub(crate) fn load(dir: &Path, slots: usize, options: Options) -> Result<Self> {
        if is_qwen(dir) {
            if options.embed.is_some() {
                return Err(ironbark_engine::Error::Format("Qwen: no embed option".into()).into());
            }
            if !dir.join("experts.json").exists() {
                if options.records.is_some() {
                    return Err(ironbark_engine::Error::Format("Qwen: records need a pack".into()).into());
                }
                return Ok(Engine::Qwen(Box::new(qwen::Model::load(dir)?)));
            }
            let records: Arc<dyn Records> = match options.records {
                Some(r) => r,
                None => ironbark_experts::records::local(dir).map_err(ironbark_engine::Error::from)?,
            };
            let mut m = qwen::Model::load_records(dir, dir, slots, options.loaders, records)?;
            m.lookahead = 1;
            return Ok(Engine::Qwen(Box::new(m)));
        }
        let mut m = match options.records {
            Some(r) => Model::load_records(dir, slots, options.loaders, r)?,
            None => Model::load_streaming(dir, dir, slots, options.loaders)?,
        };
        m.lookahead = 1;
        m.set_embed_chunks(options.embed);
        Ok(Engine::Maple(Box::new(m)))
    }

    /// Qwen: prefill `ids` with `image` in place of its `token`s ([`qwen::Model::prefill_image`]).
    pub(crate) fn prefill_image(&mut self, ids: &[u32], image: &qwen::vision::Image, token: u32) -> Result<Array> {
        match self {
            Engine::Qwen(m) => Ok(m.prefill_image(ids, std::slice::from_ref(image), token, false)?),
            Engine::Maple(_) => Err(ironbark_engine::Error::Format("images: Qwen3.5 / 3.6 only".into()).into()),
        }
    }

    pub(crate) fn is_qwen(&self) -> bool {
        matches!(self, Engine::Qwen(_))
    }

    /// Caches of this engine from saved [`Cache::parts`].
    pub(crate) fn cache_from(&self, arrays: &HashMap<String, Array>, pos: usize, delta: i64) -> Result<Cache> {
        Ok(match self {
            Engine::Maple(m) => Cache::Maple(m.snapshot_from(arrays, pos)?),
            Engine::Qwen(m) => Cache::Qwen(m.snapshot_from(arrays, pos, delta)?),
        })
    }

    /// Tokens in the caches.
    pub(crate) fn pos(&self) -> usize {
        match self {
            Engine::Maple(m) => m.pos,
            Engine::Qwen(m) => m.pos,
        }
    }

    pub(crate) fn snapshot(&self) -> Cache {
        match self {
            Engine::Maple(m) => Cache::Maple(m.snapshot()),
            Engine::Qwen(m) => Cache::Qwen(m.snapshot()),
        }
    }

    /// Return to `c`, taken from this engine.
    pub(crate) fn restore(&mut self, c: &Cache) {
        match (self, c) {
            (Engine::Maple(m), Cache::Maple(c)) => m.restore(c),
            (Engine::Qwen(m), Cache::Qwen(c)) => m.restore(c),
            _ => unreachable!("a cache is restored into the model it came from"),
        }
    }

    pub(crate) fn set_lookahead(&mut self, layers: usize) {
        match self {
            Engine::Maple(m) => m.lookahead = layers,
            Engine::Qwen(m) => m.lookahead = layers,
        }
    }

    /// Clear the caches without running any queued step (after a failed GPU command buffer).
    pub(crate) fn discard(&mut self) {
        match self {
            Engine::Maple(m) => m.discard(),
            Engine::Qwen(m) => m.discard(),
        }
    }

    /// Dynamic top-k (Qwen, Maple). Returns true (every model has it).
    pub(crate) fn set_dynamic(&mut self, d: Dynamic) -> bool {
        match self {
            Engine::Maple(m) => m.cfg.dynamic = d,
            Engine::Qwen(m) => m.cfg.dynamic = d,
        }
        true
    }

    /// FlashHead on or off; false if the model has none.
    pub(crate) fn set_flash_head(&mut self, on: bool) -> Result<bool> {
        match self {
            Engine::Maple(m) => Ok(m.set_flash_head(on)?),
            Engine::Qwen(_) => Ok(false),
        }
    }

    pub(crate) fn profile(&self, reps: usize) -> Result<Profile> {
        match self {
            Engine::Maple(m) => Ok(m.profile(reps)?),
            Engine::Qwen(_) => Err(ironbark_engine::Error::Format("profile: Maple only".into()).into()),
        }
    }

    /// Zero the step timing and the expert store counters.
    pub(crate) fn reset_stats(&mut self) {
        self.reset_timing();
        if let Some(s) = Decode::store(self) {
            ironbark_engine::bench::reset_store(s);
        }
    }

    /// Expert cache counters over `tokens` decoded tokens.
    pub(crate) fn expert_stats(&self, tokens: usize) -> Option<ExpertStats> {
        match self {
            Engine::Maple(m) => ExpertStats::of(m, tokens),
            Engine::Qwen(m) => m.store().map(|s| ExpertStats::of_store(s, tokens)),
        }
    }
}

impl Decode for Engine {
    fn reset(&mut self) {
        match self {
            Engine::Maple(m) => m.reset(),
            Engine::Qwen(m) => m.reset(),
        }
    }
    fn prefill(&mut self, ids: &[u32]) -> ironbark_engine::Result<Array> {
        match self {
            Engine::Maple(m) => Decode::prefill(m.as_mut(), ids),
            Engine::Qwen(m) => Decode::prefill(m.as_mut(), ids),
        }
    }
    fn step_lazy(&mut self, ids: &Array) -> ironbark_engine::Result<Array> {
        match self {
            Engine::Maple(m) => m.step_lazy(ids),
            Engine::Qwen(m) => m.step_lazy(ids),
        }
    }
    fn store(&self) -> Option<&Arc<Store>> {
        match self {
            Engine::Maple(m) => m.store(),
            Engine::Qwen(m) => Decode::store(m.as_ref()),
        }
    }
    fn timing(&self) -> StepTiming {
        match self {
            Engine::Maple(m) => Decode::timing(m.as_ref()),
            Engine::Qwen(m) => Decode::timing(m.as_ref()),
        }
    }
    fn reset_timing(&mut self) {
        match self {
            Engine::Maple(m) => Decode::reset_timing(m.as_mut()),
            Engine::Qwen(m) => Decode::reset_timing(m.as_mut()),
        }
    }
    fn pending(&self) -> Vec<&Array> {
        match self {
            Engine::Maple(m) => m.pending(),
            Engine::Qwen(m) => Decode::pending(m.as_ref()),
        }
    }
}
