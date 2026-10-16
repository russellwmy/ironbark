//! Run Maple (DeepGrove's 20B ternary MoE) on Apple Silicon (macOS, iOS) with its experts
//! streamed from storage, sized to the device's memory.
//!
//! A model is a dir written by `ironbark pack`. [`Session::open`] plans a memory budget for the
//! device, loads the non-expert weights, and caches as many experts per layer as fit; the rest
//! stream from storage as the router picks them. Output is greedy and bit-exact with
//! DeepGrove's reference.
//!
//! ```no_run
//! # fn main() -> ironbark::Result<()> {
//! let mut s = ironbark::Session::open("models/maple", ironbark::Options::default())?;
//! for piece in s.chat("What is the capital of Germany?")? {
//!     print!("{}", piece?);
//! }
//! println!("\n{:.1} tok/s", s.stats().decode_tps);
//! # Ok(())
//! # }
//! ```
//!
//! ```text
//!   Session ── chat / start + next_piece ──▶ Model (ironbark-engine: MLX, custom Metal kernels)
//!      │                                        └─▶ Store (ironbark-experts: expert cache, loader threads)
//!      └── plan: budget from the device -> experts per layer
//! ```

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use ironbark_mlx::ops::indexing::argmax_axis;
use ironbark_mlx::transforms::async_eval;
use ironbark_mlx::{Array, Dtype};
use tokenizers::Tokenizer;

pub use ironbark_engine::bench::{ExpertStats, Generation, ReadProbe};
pub use ironbark_engine::device::{Footprint, Hardware};
pub use ironbark_engine::manifest::{Report, verify};
pub use ironbark_engine::plan::Plan;
pub use ironbark_engine::profile::Profile;
pub use ironbark_engine::{Config, KvSnapshot, Model, pack};
pub use ironbark_experts::{Chunks, Records};
pub use serde_json::{Value, json};
pub use tools::{Message, Parsed, ToolCall, parse};

pub mod tools;

/// Errors of a session.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Loading or running the model failed.
    #[error(transparent)]
    Model(#[from] ironbark_engine::Error),
    /// An MLX operation failed.
    #[error(transparent)]
    Mlx(#[from] ironbark_mlx::error::Exception),
    /// The tokenizer failed.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// The text does not fit the context left: [`Session::reset`] and retry.
    #[error("context full: {need} tokens needed, {max} allowed")]
    ContextFull {
        /// Tokens the conversation would hold.
        need: usize,
        /// [`Options::max_context`].
        max: usize,
    },
    /// [`Session::next_piece`] without [`Session::start`].
    #[error("no reply started")]
    NoTurn,
}

/// Result of this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// How to open a session.
#[derive(Debug, Clone)]
pub struct Options {
    /// Process footprint budget in bytes; `None`: from the device (3/8 of RAM, at most half
    /// the iOS app limit).
    pub budget: Option<u64>,
    /// Expert cache slots per layer; `None`: as many as fit the budget (all 256 keeps every
    /// expert loaded, nothing streams).
    pub slots: Option<usize>,
    /// Most tokens a conversation holds; the budget reserves its KV cache.
    pub max_context: usize,
    /// Threads reading experts from storage.
    pub loaders: usize,
    /// Where to read expert records; `None`: the dir's `experts.bin`, which must be complete.
    /// E.g. an `ironbark_fetch::Sparse` that fetches missing records from a server.
    pub records: Option<Arc<dyn Records>>,
    /// Where `embed.bin` is still arriving; `None`: the dir's `embed.bin`, which must be
    /// complete. E.g. an `ironbark_fetch::Sparse::open_embed` that fetches missing chunks.
    pub embed: Option<Arc<dyn Chunks>>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            budget: None,
            slots: None,
            max_context: ironbark_engine::plan::CONTEXT,
            loaders: 8,
            records: None,
            embed: None,
        }
    }
}

/// Why a reply ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    /// The model ended its turn.
    Stop,
    /// The context is full.
    Context,
}

/// Speed and memory of the current turn.
#[derive(Debug, Clone)]
pub struct TurnStats {
    /// Tokens read for this turn.
    pub prompt_tokens: usize,
    /// Prompt tokens per second.
    pub prefill_tps: f64,
    /// Reply tokens so far.
    pub tokens: usize,
    /// Reply tokens per second.
    pub decode_tps: f64,
    /// Tokens in the conversation.
    pub context: usize,
    /// Expert cache counters per reply token (all hits when every expert is resident).
    pub experts: Option<ExpertStats>,
    /// Process footprint now and at its peak.
    pub footprint: Option<Footprint>,
}

/// The reply being generated.
#[derive(Debug)]
struct Turn {
    /// Next token, possibly still on the GPU.
    next: Array,
    tokens: Vec<u32>,
    /// Bytes of the decoded reply already handed out.
    shown: usize,
    /// Text the reply starts with that the prompt already holds (handed out first).
    prefix: String,
    prompt: usize,
    prefill_s: f64,
    decode_start: Instant,
    decode_s: f64,
    finish: Option<Finish>,
}

/// A model open for generation, with its tokenizer and the plan that sized it.
#[derive(Debug)]
pub struct Session {
    model: Model,
    tok: Tokenizer,
    plan: Plan,
    hw: Hardware,
    max_context: usize,
    /// Tokens that end the model's turn.
    stop: Vec<u32>,
    turn: Option<Turn>,
    /// The KV cache holds a conversation in Maple's chat format.
    chatting: bool,
    /// System turn that starts each conversation (see [`Session::set_system`]).
    system: String,
    /// The KV cache after the system turn, so a new conversation does not read it again.
    system_kv: Option<KvSnapshot>,
    /// Replies reason first (see [`Session::set_thinking`]).
    think: bool,
}

fn tok_err(e: impl std::fmt::Display) -> Error {
    Error::Tokenizer(e.to_string())
}

/// Greedy next token of logits `[1, 1, vocab]`, as a lazy `[1]` u32.
fn pick(logits: &Array) -> Result<Array> {
    Ok(argmax_axis(logits, -1, None)?.reshape(&[1])?.as_dtype(Dtype::Uint32)?)
}

impl Session {
    /// Open the packed model in `dir` (`ironbark pack` output).
    ///
    /// # Errors
    ///
    /// Fails if a file is missing, incomplete or does not match, or MLX cannot load it.
    pub fn open(dir: impl AsRef<Path>, options: Options) -> Result<Self> {
        let dir = dir.as_ref();
        let hw = Hardware::detect();
        let budget = options.budget.unwrap_or_else(|| ironbark_engine::plan::budget(&hw));
        let mut plan = Plan::new(dir, budget, options.max_context)?;
        if let Some(slots) = options.slots {
            plan.slots = slots;
        }
        // MLX keeps freed GPU buffers for reuse; unlimited, they hold prefill temporaries (4–5
        // GB of footprint at 64 experts per layer), which iOS counts against the app.
        ironbark_mlx::memory::set_cache_limit(plan.buffer_cache as usize)?;
        let mut model = match options.records {
            Some(r) => Model::load_records(dir, plan.slots, options.loaders, r)?,
            None => Model::load_streaming(dir, dir, plan.slots, options.loaders)?,
        };
        model.lookahead = 1;
        model.set_embed_chunks(options.embed);
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(tok_err)?;
        let stop = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| tok.token_to_id(t)).collect();
        Ok(Self {
            model,
            tok,
            plan,
            hw,
            max_context: options.max_context,
            stop,
            turn: None,
            chatting: false,
            system: String::new(),
            system_kv: None,
            think: true,
        })
    }

    /// How the session was sized.
    #[must_use]
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// The device it was sized for.
    #[must_use]
    pub fn hardware(&self) -> &Hardware {
        &self.hw
    }

    /// The model, for lower-level use (`ironbark-engine`).
    #[must_use]
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Tokens in the conversation (the KV cache).
    #[must_use]
    pub fn context(&self) -> usize {
        self.model.pos
    }

    /// Token ids of `text` (no special tokens added).
    ///
    /// # Errors
    ///
    /// Fails if the tokenizer fails.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.tok.encode(text, false).map_err(tok_err)?.get_ids().to_vec())
    }

    /// Text of token ids.
    ///
    /// # Errors
    ///
    /// Fails if the tokenizer fails.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tok.decode(ids, true).map_err(tok_err)
    }

    /// Forget the conversation.
    pub fn reset(&mut self) {
        self.model.reset();
        self.turn = None;
        self.chatting = false;
    }

    /// Benchmark: from an empty context, read `prompt` and generate `tokens` tokens greedily,
    /// with timings. Forgets any conversation.
    ///
    /// # Errors
    ///
    /// Fails if the prompt is empty or does not fit, or the model fails.
    pub fn generate(&mut self, prompt: &str, tokens: usize) -> Result<Generation> {
        self.reset();
        let ids = self.encode(prompt)?;
        if ids.len() + tokens > self.max_context {
            return Err(Error::ContextFull {
                need: ids.len() + tokens,
                max: self.max_context,
            });
        }
        let g = ironbark_engine::bench::generate(&mut self.model, &ids, tokens)?;
        self.model.reset();
        Ok(g)
    }

    /// Start every conversation with a system turn: `system` and, with `tools` (JSON schemas,
    /// `{"type": "function", "function": {"name", "description", "parameters"}}`), the tool-call
    /// format (see [`tools`]). Forgets the conversation.
    pub fn set_system(&mut self, system: Option<&str>, tools: &[Value]) {
        self.system = tools::system_turn(system, tools);
        self.system_kv = None;
        self.reset();
    }

    /// Streaming: layers ahead whose experts each step predicts and prefetches (default 1;
    /// 0 turns prediction off).
    pub fn set_lookahead(&mut self, layers: usize) {
        self.model.lookahead = layers;
    }

    /// Decode with FlashHead, DeepGrove's approximate LM head (exact logits only for the top
    /// clusters of the vocabulary), or the exact head (the default); false if the model has no
    /// FlashHead.
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails.
    pub fn set_flash_head(&mut self, on: bool) -> Result<bool> {
        Ok(self.model.set_flash_head(on)?)
    }

    /// Whether replies reason before answering (the default). Off, a reply to a user message
    /// starts with an empty `<think></think>` and answers at once (far fewer tokens, weaker on
    /// hard problems); its text starts with that `</think>`, so [`parse`] splits it as usual.
    /// Replies to tool results still reason.
    pub fn set_thinking(&mut self, on: bool) {
        self.think = on;
    }

    /// Send `message` in the conversation (Maple's chat format) and stream the reply: its
    /// reasoning, `</think>`, then the answer or tool calls ([`parse`] splits them).
    ///
    /// # Errors
    ///
    /// Fails if the message does not fit the context left, or the model fails.
    pub fn chat(&mut self, message: &str) -> Result<Reply<'_>> {
        self.send(&[Message::User(message)])
    }

    /// Send the results of the tool calls of the last reply, in order, and stream the reply.
    ///
    /// # Errors
    ///
    /// Fails if the results do not fit the context left, or the model fails.
    pub fn tool_results(&mut self, results: &[&str]) -> Result<Reply<'_>> {
        self.send(&[Message::ToolResults(results)])
    }

    /// Read `messages` after the conversation (after the system turn in a new one) and stream
    /// the reply; the last is a user message or tool results. To rebuild a conversation (e.g.
    /// after reopening the model), [`Session::reset`] and send all of its messages.
    ///
    /// # Errors
    ///
    /// Fails if the messages do not fit the context left, or the model fails.
    pub fn send(&mut self, messages: &[Message<'_>]) -> Result<Reply<'_>> {
        let turns: String = messages.iter().map(Message::turn).collect();
        self.open_turn(&turns)
    }

    /// Read user turns `turn` after the conversation and open the model's turn.
    fn open_turn(&mut self, turn: &str) -> Result<Reply<'_>> {
        // Close the previous turn: its end token is in the KV cache, unless the reply was
        // dropped before the model ended it.
        let open = self.turn.as_ref().is_some_and(|t| t.finish.is_none());
        let close = match (self.chatting, open) {
            (false, _) => "",
            (true, true) => "<|im_end|>\n",
            (true, false) => "\n",
        };
        // A new conversation starts from the system turn, read once and then restored.
        if !self.chatting && !self.system.is_empty() {
            if let Some(kv) = &self.system_kv {
                self.model.restore(kv);
            } else {
                let system = self.system.clone();
                self.start(&system, true)?;
                self.system_kv = Some(self.model.snapshot());
            }
        }
        // After tool results the model reasons even when off: without it, it called the same
        // tool again instead of answering.
        let think = self.think || turn.contains("<tool_response>");
        let reasoning = if think { "<think>\n" } else { "<think>\n\n</think>\n\n" };
        let text = format!("{close}{turn}<|im_start|>assistant\n{reasoning}");
        let reset = !self.chatting && self.system.is_empty();
        self.start(&text, reset)?;
        if !think {
            // The reply is shown from the end of its (empty) reasoning, as parse expects.
            if let Some(t) = &mut self.turn {
                t.prefix = "</think>\n\n".into();
            }
        }
        self.chatting = true;
        Ok(Reply { session: self })
    }

    /// Read `text` after the conversation (or from scratch with `reset`); the reply starts at
    /// the next [`Session::next_piece`]. For prompt formats other than [`Session::chat`]'s.
    ///
    /// # Errors
    ///
    /// Fails if the text is empty or does not fit the context left, or the model fails.
    pub fn start(&mut self, text: &str, reset: bool) -> Result<usize> {
        let ids = self.encode(text)?;
        if ids.is_empty() {
            return Err(ironbark_engine::Error::Format("empty text".into()).into());
        }
        if reset {
            self.reset();
        }
        let need = self.model.pos + ids.len();
        if need > self.max_context {
            return Err(Error::ContextFull {
                need,
                max: self.max_context,
            });
        }
        let t = Instant::now();
        let logits = self.model.prefill(&ids, false)?;
        let next = pick(&logits)?;
        next.eval()?;
        let prefill_s = t.elapsed().as_secs_f64();
        ironbark_engine::bench::reset_stats(&mut self.model);
        self.turn = Some(Turn {
            next,
            tokens: Vec::new(),
            shown: 0,
            prefix: String::new(),
            prompt: ids.len(),
            prefill_s,
            decode_start: Instant::now(),
            decode_s: 0.0,
            finish: None,
        });
        Ok(ids.len())
    }

    /// The next piece of the reply: `Some("")` while a character is still incomplete, `None`
    /// once it ended ([`Session::finish`] says why).
    ///
    /// # Errors
    ///
    /// Fails without a started turn, or if the model fails.
    pub fn next_piece(&mut self) -> Result<Option<String>> {
        let turn = self.turn.as_mut().ok_or(Error::NoTurn)?;
        if !turn.prefix.is_empty() {
            return Ok(Some(std::mem::take(&mut turn.prefix)));
        }
        if turn.finish.is_some() {
            return Ok(None);
        }
        if self.model.pos >= self.max_context {
            turn.finish = Some(Finish::Context);
            return Ok(None);
        }
        // Queue the step after this token before reading it, so the GPU keeps going. The step
        // puts this token in the KV cache, also when it ends the turn.
        let following = pick(&self.model.step_lazy(&turn.next)?)?;
        async_eval([&following])?;
        let tok = turn.next.item_exact::<u32>();
        turn.next = following;
        turn.decode_s = turn.decode_start.elapsed().as_secs_f64();
        if self.stop.contains(&tok) {
            turn.finish = Some(Finish::Stop);
            return Ok(None);
        }
        turn.tokens.push(tok);
        let text = self.tok.decode(&turn.tokens, true).map_err(tok_err)?;
        // A multi-byte character split across tokens decodes as U+FFFD until it completes.
        if text.ends_with('\u{FFFD}') || text.len() < turn.shown {
            return Ok(Some(String::new()));
        }
        let piece = text[turn.shown..].to_owned();
        turn.shown = text.len();
        Ok(Some(piece))
    }

    /// Why the current reply ended, if it did.
    #[must_use]
    pub fn finish(&self) -> Option<Finish> {
        self.turn.as_ref().and_then(|t| t.finish)
    }

    /// Speed and memory of the current turn.
    #[must_use]
    pub fn stats(&self) -> TurnStats {
        let (prompt, prefill_s, tokens, decode_s) =
            self.turn.as_ref().map_or((0, 0.0, 0, 0.0), |t| (t.prompt, t.prefill_s, t.tokens.len(), t.decode_s));
        TurnStats {
            prompt_tokens: prompt,
            prefill_tps: prompt as f64 / prefill_s.max(1e-9),
            tokens,
            decode_tps: tokens as f64 / decode_s.max(1e-9),
            context: self.model.pos,
            experts: ExpertStats::of(&self.model, tokens),
            footprint: ironbark_engine::device::footprint(),
        }
    }

    /// GPU time of each part of a decode step.
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails.
    pub fn profile(&self, reps: usize) -> Result<Profile> {
        Ok(self.model.profile(reps)?)
    }
}

/// A reply streamed piece by piece (from [`Session::chat`]); dropping it early leaves the
/// conversation usable.
#[derive(Debug)]
pub struct Reply<'a> {
    session: &'a mut Session,
}

impl Reply<'_> {
    /// Why the reply ended, once it did.
    #[must_use]
    pub fn finish(&self) -> Option<Finish> {
        self.session.finish()
    }
}

impl Iterator for Reply<'_> {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        self.session.next_piece().transpose()
    }
}
