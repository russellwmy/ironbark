//! Run Maple (DeepGrove's 20B ternary MoE) on Apple Silicon (macOS, iOS) with its experts
//! streamed from storage, sized to the device's memory.
//!
//! A model is a dir written by `ironbark pack`. [`Session::open`] plans a memory budget for the
//! device, loads the non-expert weights, and caches as many experts per layer as fit; the rest
//! stream from storage as the router picks them. Maple's output is greedy and bit-exact with
//! DeepGrove's reference; Qwen samples with its recommended settings ([`Sampling`]).
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

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use engine::{Cache, Engine};
use ironbark_engine::bench::Decode;
use ironbark_engine::plan::{CONTEXT, QWEN_CONTEXT, QWEN_CONTEXT_SLOTS};
use ironbark_mlx::Array;
use ironbark_mlx::transforms::async_eval;
use sample::Sampler;
use tokenizers::Tokenizer;
use tools::Format;

pub use engine::is_qwen;
pub use ironbark_engine::bench::{ExpertStats, Generation, ReadProbe};
pub use ironbark_engine::device::{Footprint, Hardware};
pub use ironbark_engine::dynamic::Dynamic;
pub use ironbark_engine::manifest::{Report, verify};
pub use ironbark_engine::plan::Plan;
pub use ironbark_engine::profile::Profile;
pub use ironbark_engine::{Config, KvSnapshot, Model, pack};
pub use ironbark_experts::{Chunks, Records};
pub use sample::Sampling;
pub use serde_json::{Value, json};
pub use tools::{Message, Parsed, ToolCall, parse};

mod engine;
mod sample;
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
    /// Most tokens a conversation holds; the budget reserves its KV cache. `None`: the model's
    /// default (Maple 4096; Qwen 16384 when the budget still leaves 48 expert slots per layer,
    /// else 4096).
    pub max_context: Option<usize>,
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
            max_context: None,
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

/// An image encoded for the next user message ([`Session::set_image`]).
#[derive(Debug, Clone)]
pub struct ImageInfo {
    /// Tokens it takes in the prompt.
    pub tokens: usize,
    /// The size it was resized to (pixels).
    pub width: usize,
    pub height: usize,
    /// Seconds to load the encoder, resize and encode.
    pub encode_s: f64,
}

/// Most pixels [`Session::set_image`] resizes an image to by default: 512 × 512, ~260 prompt
/// tokens (a few seconds of prefill on a phone).
pub const IMAGE_PIXELS: usize = 512 * 512;

/// Most system turns kept read ([`Session::set_system`]), so switching between them is cheap.
const SYSTEM_KVS: usize = 4;

/// A saved conversation ([`Session::snapshot`]): its KV cache, its system turn, and how the
/// next turn closes the last one.
#[derive(Debug, Clone)]
pub struct Snapshot {
    kv: Cache,
    /// The tokens in `kv` (to read again after a GPU failure).
    ids: Vec<u32>,
    image: bool,
    context: usize,
    system: String,
    chatting: bool,
    close: &'static str,
}

impl Snapshot {
    /// Tokens in the saved conversation.
    #[must_use]
    pub fn context(&self) -> usize {
        self.context
    }
}

/// A model open for generation, with its tokenizer and the plan that sized it.
#[derive(Debug)]
pub struct Session {
    model: Engine,
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
    /// The KV cache after each recent system turn (oldest first, at most [`SYSTEM_KVS`]), so
    /// a new conversation, also after switching system turns, does not read it again.
    system_kvs: Vec<(String, Cache, Vec<u32>)>,
    /// How the next turn closes the last one, when set by [`Session::read`] or
    /// [`Session::restore`] rather than by the last reply.
    close: Option<&'static str>,
    /// Replies reason first (see [`Session::set_thinking`]).
    think: bool,
    /// The model's chat format.
    format: Format,
    /// The model's dir (its vision encoder loads from it on demand).
    dir: PathBuf,
    /// Qwen: an encoded image for the next user message ([`Session::set_image`]).
    image: Option<ironbark_engine::qwen::vision::Image>,
    /// Set by [`Session::set_sampling`]; `None`: the model's own ([`Session::sampling`]).
    sampling: Option<Sampling>,
    sampler: Sampler,
    /// The tokens in the KV cache, to read again after a GPU failure ([`Session::recover`]).
    ids: Vec<u32>,
    /// The KV cache holds an image (its embeddings are gone after a GPU failure).
    image_read: bool,
}

fn tok_err(e: impl std::fmt::Display) -> Error {
    Error::Tokenizer(e.to_string())
}

impl Session {
    /// Open the packed model in `dir` (`ironbark pack` output, or a bundle packed into
    /// `dir/pack`).
    ///
    /// # Errors
    ///
    /// Fails if a file is missing, incomplete or does not match, `dir` is a bundle not packed
    /// yet, or MLX cannot load it.
    pub fn open(dir: impl AsRef<Path>, options: Options) -> Result<Self> {
        let dir = &ironbark_engine::qwen::packed(dir.as_ref())?;
        let hw = Hardware::detect();
        let budget = options.budget.unwrap_or_else(|| ironbark_engine::plan::budget(&hw));
        let (max_context, mut plan) = match options.max_context {
            Some(c) => (c, Engine::plan(dir, budget, c)?),
            None if is_qwen(dir) => {
                // A long context only while it leaves enough expert slots (0: all resident).
                let long = Engine::plan(dir, budget, QWEN_CONTEXT)?;
                if long.slots == 0 || long.slots >= QWEN_CONTEXT_SLOTS {
                    (QWEN_CONTEXT, long)
                } else {
                    (CONTEXT, Engine::plan(dir, budget, CONTEXT)?)
                }
            }
            None => (CONTEXT, Engine::plan(dir, budget, CONTEXT)?),
        };
        if let Some(slots) = options.slots {
            plan.slots = slots;
        }
        // MLX keeps freed GPU buffers for reuse; unlimited, they hold prefill temporaries (4–5
        // GB of footprint at 64 experts per layer), which iOS counts against the app.
        ironbark_mlx::memory::set_cache_limit(plan.buffer_cache as usize)?;
        let model = Engine::load(dir, plan.slots, options)?;
        let tok = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(tok_err)?;
        let format = if model.is_qwen() { Format::Qwen } else { Format::Maple };
        let stop = ["<|im_end|>", "<|endoftext|>"].iter().filter_map(|t| tok.token_to_id(t)).collect();
        Ok(Self {
            model,
            tok,
            plan,
            hw,
            max_context,
            stop,
            turn: None,
            chatting: false,
            system: String::new(),
            system_kvs: Vec::new(),
            close: None,
            think: true,
            format,
            dir: dir.clone(),
            image: None,
            sampling: None,
            sampler: Sampler::new(seed())?,
            ids: Vec::new(),
            image_read: false,
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

    /// The Maple model, for lower-level use (`ironbark-engine`); `None` for the other models.
    #[must_use]
    pub fn model(&self) -> Option<&Model> {
        match &self.model {
            Engine::Maple(m) => Some(m),
            Engine::Qwen(_) => None,
        }
    }

    /// The model is Qwen3.5 / 3.6.
    #[must_use]
    pub fn is_qwen(&self) -> bool {
        self.model.is_qwen()
    }

    /// Expert cache counters over `tokens` decoded tokens.
    #[must_use]
    pub fn expert_stats(&self, tokens: usize) -> Option<ExpertStats> {
        self.model.expert_stats(tokens)
    }

    /// Tokens in the conversation (the KV cache).
    #[must_use]
    pub fn context(&self) -> usize {
        self.model.pos()
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
        self.ids.clear();
        self.image_read = false;
        self.turn = None;
        self.chatting = false;
        self.close = None;
    }

    /// Save the conversation, to come back to it with [`Session::restore`] (e.g. after a
    /// side conversation on another system turn). Cheap: the KV cache is shared, not copied.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            kv: self.model.snapshot(),
            ids: self.ids.clone(),
            image: self.image_read,
            context: self.model.pos(),
            system: self.system.clone(),
            chatting: self.chatting,
            close: self.closing(),
        }
    }

    /// Return to a conversation saved by [`Session::snapshot`], with its system turn; the
    /// next message continues it.
    pub fn restore(&mut self, snapshot: &Snapshot) {
        self.model.restore(&snapshot.kv);
        self.ids.clone_from(&snapshot.ids);
        self.image_read = snapshot.image;
        self.system.clone_from(&snapshot.system);
        self.turn = None;
        self.chatting = snapshot.chatting;
        self.close = Some(snapshot.close);
    }

    /// Read `messages` into the conversation (after the system turn in a new one) without a
    /// reply, e.g. few-shot samples to [`Session::snapshot`] once and restore for each
    /// question. Returns the tokens read.
    ///
    /// # Errors
    ///
    /// Fails if the messages are empty or do not fit the context left, or the model fails.
    pub fn read(&mut self, messages: &[Message<'_>]) -> Result<usize> {
        let turns: String = messages.iter().map(Message::turn).collect();
        let close = self.closing();
        self.open_system()?;
        let reset = !self.chatting && self.system.is_empty();
        let read = self.start(&format!("{close}{turns}"), reset)?;
        self.turn = None;
        self.chatting = true;
        // The turns end with their end token and line break.
        self.close = Some("");
        Ok(read)
    }

    /// How the next turn closes the last one: nothing in a new conversation; its end token
    /// when the last reply was dropped before the model ended it; else the line break after
    /// the end token the reply put in the KV cache.
    fn closing(&self) -> &'static str {
        if let Some(close) = self.close {
            return close;
        }
        let open = self.turn.as_ref().is_some_and(|t| t.finish.is_none());
        match (self.chatting, open) {
            (false, _) => "",
            (true, true) => "<|im_end|>\n",
            (true, false) => "\n",
        }
    }

    /// In a new conversation, put the system turn in the KV cache: restored when it was read
    /// recently, else read now. Returns the tokens read and the time it took (zero when
    /// restored or not needed).
    fn open_system(&mut self) -> Result<(usize, f64)> {
        if self.chatting || self.system.is_empty() {
            return Ok((0, 0.0));
        }
        if let Some(i) = self.system_kvs.iter().position(|(text, ..)| *text == self.system) {
            let entry = self.system_kvs.remove(i);
            self.model.restore(&entry.1);
            self.ids.clone_from(&entry.2);
            self.image_read = false;
            self.system_kvs.push(entry);
            return Ok((0, 0.0));
        }
        let system = self.system.clone();
        let tokens = self.start(&system, true)?;
        let read_s = self.turn.as_ref().map_or(0.0, |t| t.prefill_s);
        self.turn = None;
        if self.system_kvs.len() == SYSTEM_KVS {
            self.system_kvs.remove(0);
        }
        self.system_kvs.push((system, self.model.snapshot(), self.ids.clone()));
        Ok((tokens, read_s))
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
    /// format (see [`tools`]). Forgets the conversation. The last few system turns stay read,
    /// so switching back to one does not read it again.
    pub fn set_system(&mut self, system: Option<&str>, tools: &[Value]) {
        self.system = tools::system_turn(system, tools, self.format);
        self.reset();
    }

    /// Streaming: layers ahead whose experts each step predicts and prefetches (default 1;
    /// 0 turns prediction off).
    pub fn set_lookahead(&mut self, layers: usize) {
        self.model.set_lookahead(layers);
    }

    /// Dynamic top-k: decode drops each token's weak expert choices (fewer reads), within
    /// `d` ([`Dynamic`]; its default is off). Every model (Qwen, Maple). Returns true.
    pub fn set_dynamic_k(&mut self, d: Dynamic) -> bool {
        self.model.set_dynamic(d)
    }

    /// Decode with FlashHead, DeepGrove's approximate LM head (exact logits only for the top
    /// clusters of the vocabulary), or the exact head (the default); false if the model has no
    /// FlashHead.
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails.
    pub fn set_flash_head(&mut self, on: bool) -> Result<bool> {
        self.model.set_flash_head(on)
    }

    /// Whether replies reason before answering (the default). Off, a reply to a user message
    /// starts with an empty `<think></think>` and answers at once (far fewer tokens, weaker on
    /// hard problems); its text starts with that `</think>`, so [`parse`] splits it as usual.
    /// Maple's replies to tool results still reason.
    pub fn set_thinking(&mut self, on: bool) {
        self.think = on;
    }

    /// Whether replies may call tools (the default). Off, the model cannot start a
    /// `<tool_call>`, so it answers with what it has (e.g. after a limit of tool rounds).
    pub fn set_tool_calls(&mut self, on: bool) {
        let call = self.tok.token_to_id("<tool_call>");
        self.sampler.ban(&if on { None } else { call }.into_iter().collect::<Vec<_>>());
    }

    /// Choose next tokens this way; `None`: the model's own (greedy for Maple, [`Sampling::QWEN`]
    /// for Qwen, which loops when greedy).
    pub fn set_sampling(&mut self, sampling: Option<Sampling>) {
        self.sampling = sampling;
    }

    /// How next tokens are chosen ([`Session::set_sampling`]).
    #[must_use]
    pub fn sampling(&self) -> Sampling {
        self.sampling.unwrap_or_else(|| {
            if self.model.is_qwen() {
                Sampling::QWEN
            } else {
                Sampling::GREEDY
            }
        })
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
        let turns = match (&self.image, messages.iter().rposition(|m| matches!(m, Message::User(_)))) {
            // The image goes before the text of the last user message, as the template puts it.
            (Some(img), Some(i)) => {
                let Message::User(text) = messages[i] else {
                    unreachable!("found a user message")
                };
                let pads = "<|image_pad|>".repeat(img.tokens());
                format!(
                    "{}<|im_start|>user\n<|vision_start|>{pads}<|vision_end|>{text}<|im_end|>\n{}",
                    messages[..i].iter().map(Message::turn).collect::<String>(),
                    messages[i + 1..].iter().map(Message::turn).collect::<String>()
                )
            }
            _ => messages.iter().map(Message::turn).collect(),
        };
        let after_results = matches!(messages.last(), Some(Message::ToolResults(_)));
        self.open_turn(&turns, after_results)
    }

    /// Read user turns `turn` after the conversation and open the model's turn
    /// (`after_results`: they end with tool results).
    fn open_turn(&mut self, turn: &str, after_results: bool) -> Result<Reply<'_>> {
        // Close the previous turn (see `closing`).
        let close = self.closing();
        // A new conversation starts from the system turn, read once and then restored.
        let (system_tokens, system_s) = self.open_system()?;
        // After tool results Maple reasons even when off: without it, it called the same tool
        // again instead of answering. Qwen 3.6 answers without (as in the evals).
        let think = self.think || (after_results && !self.model.is_qwen());
        let reasoning = if think { "<think>\n" } else { "<think>\n\n</think>\n\n" };
        let text = format!("{close}{turn}<|im_start|>assistant\n{reasoning}");
        let reset = !self.chatting && self.system.is_empty();
        self.start(&text, reset)?;
        if let Some(t) = &mut self.turn {
            // The turn's stats count the system turn when it was read for it.
            t.prompt += system_tokens;
            t.prefill_s += system_s;
            if !think {
                // The reply is shown from the end of its (empty) reasoning, as parse expects.
                t.prefix = "</think>\n\n".into();
            }
        }
        self.chatting = true;
        self.close = None;
        Ok(Reply { session: self })
    }

    /// Read `text` after the conversation (or from scratch with `reset`); the reply starts at
    /// the next [`Session::next_piece`]. For prompt formats other than [`Session::chat`]'s.
    ///
    /// # Errors
    ///
    /// Fails if the text is empty or does not fit the context left, or the model fails.
    pub fn start(&mut self, text: &str, reset: bool) -> Result<usize> {
        match self.start_once(text, reset) {
            Err(e) if self.recoverable(&e) => {
                self.recover()?;
                self.start_once(text, reset)
            }
            r => r,
        }
    }

    fn start_once(&mut self, text: &str, reset: bool) -> Result<usize> {
        let ids = self.encode(text)?;
        if ids.is_empty() {
            return Err(ironbark_engine::Error::Format("empty text".into()).into());
        }
        if reset {
            self.reset();
        }
        let need = self.model.pos() + ids.len();
        if need > self.max_context {
            return Err(Error::ContextFull {
                need,
                max: self.max_context,
            });
        }
        let t = Instant::now();
        // An image set for this text: its tokens are in `ids` (the system turn has none).
        let pad = self.tok.token_to_id("<|image_pad|>").filter(|p| ids.contains(p));
        let image = if pad.is_some() { self.image.take() } else { None };
        let with_image = image.is_some();
        let logits = match (image, pad) {
            (Some(img), Some(pad)) => self.model.prefill_image(&ids, &img, pad)?,
            _ => self.model.prefill(&ids)?,
        };
        let sampling = self.sampling();
        self.sampler.restart();
        let next = self.sampler.pick(&sampling, &logits)?;
        // With the caches: none may stay lazy for the next call (see `next_piece`).
        let mut ready = self.model.pending();
        ready.push(&next);
        ready.extend(self.sampler.pending());
        ironbark_mlx::transforms::eval(ready)?;
        let prefill_s = t.elapsed().as_secs_f64();
        self.ids.extend(&ids);
        self.image_read |= with_image;
        self.model.reset_stats();
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
        match self.next_piece_once() {
            Err(e) if self.recoverable(&e) => {
                self.recover()?;
                self.next_piece_once()
            }
            r => r,
        }
    }

    fn next_piece_once(&mut self) -> Result<Option<String>> {
        let sampling = self.sampling();
        let turn = self.turn.as_mut().ok_or(Error::NoTurn)?;
        if !turn.prefix.is_empty() {
            return Ok(Some(std::mem::take(&mut turn.prefix)));
        }
        if turn.finish.is_some() {
            return Ok(None);
        }
        if self.model.pos() >= self.max_context {
            turn.finish = Some(Finish::Context);
            return Ok(None);
        }
        // Queue the step after this token before reading it, so the GPU keeps going. The step
        // puts this token in the KV cache, also when it ends the turn. Its caches are queued
        // too: the next call may come from another thread, where a lazy array cannot run.
        let following = self.sampler.pick(&sampling, &self.model.step_lazy(&turn.next)?)?;
        let mut queued = self.model.pending();
        queued.push(&following);
        queued.extend(self.sampler.pending());
        async_eval(queued)?;
        let tok = turn.next.item_exact::<u32>();
        self.ids.push(tok);
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

    /// A failed GPU command buffer (e.g. GPU work submitted while an iOS app was in the
    /// background), with a conversation [`Session::recover`] can read again.
    fn recoverable(&self, e: &Error) -> bool {
        !self.image_read && e.to_string().contains("Command buffer execution failed")
    }

    /// After a failed GPU command buffer, which poisons its MLX stream for good: run on a
    /// fresh stream ([`reset_gpu`]), drop the caches and queued steps (their arrays fail
    /// too), read the conversation's tokens again, and pick an open reply's next token anew.
    /// Weights and expert slots were evaluated before the failure and stay.
    fn recover(&mut self) -> Result<()> {
        reset_gpu();
        self.model.discard();
        self.sampler = Sampler::new(seed())?;
        self.system_kvs.clear();
        if self.ids.is_empty() {
            return Ok(());
        }
        let logits = self.model.prefill(&self.ids)?;
        let open = self.turn.as_ref().is_some_and(|t| t.finish.is_none());
        let next = if open {
            Some(self.sampler.pick(&self.sampling(), &logits)?)
        } else {
            None
        };
        let mut ready = self.model.pending();
        ready.extend(&next);
        ready.extend(self.sampler.pending());
        ironbark_mlx::transforms::eval(ready)?;
        if let (Some(turn), Some(next)) = (self.turn.as_mut(), next) {
            turn.next = next;
        }
        Ok(())
    }

    /// Why the current reply ended, if it did.
    #[must_use]
    pub fn finish(&self) -> Option<Finish> {
        self.turn.as_ref().and_then(|t| t.finish)
    }

    /// Qwen: encode an image (`rgb`: `height` rows of `width` RGB pixels, top to bottom) for
    /// the next user message ([`Session::send`], [`Session::chat`]), which then shows it before
    /// its text. Resized to at most `max_pixels` (sides multiples of 32; e.g.
    /// [`IMAGE_PIXELS`]). The encoder loads for this and is freed after.
    ///
    /// # Errors
    ///
    /// Fails if the model has no vision encoder (a text-only pack: `ironbark pack
    /// --vision-only` adds it), the image is not `width · height · 3` bytes, or MLX fails.
    pub fn set_image(&mut self, rgb: &[u8], width: usize, height: usize, max_pixels: usize) -> Result<ImageInfo> {
        use ironbark_engine::qwen::vision::{self, Vision};
        if !self.model.is_qwen() {
            return Err(ironbark_engine::Error::Format("images: Qwen3.5 / 3.6 only".into()).into());
        }
        let t = Instant::now();
        let v = Vision::load(&self.dir)?.ok_or_else(|| {
            ironbark_engine::Error::Format("no vision encoder in this pack (ironbark pack --vision-only)".into())
        })?;
        let px = vision::preprocess(&v.cfg, rgb, width, height, max_pixels)?;
        let img = v.encode(&px)?;
        let patch = v.cfg.patch;
        drop(v);
        ironbark_mlx::memory::clear_cache()?;
        let info = ImageInfo {
            tokens: img.tokens(),
            width: px.cols * patch,
            height: px.rows * patch,
            encode_s: t.elapsed().as_secs_f64(),
        };
        self.image = Some(img);
        Ok(info)
    }

    /// Drop an image set with [`Session::set_image`] and not sent yet.
    pub fn clear_image(&mut self) {
        self.image = None;
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
            context: self.model.pos(),
            experts: self.model.expert_stats(tokens),
            footprint: ironbark_engine::device::footprint(),
        }
    }

    /// GPU time of each part of a decode step.
    ///
    /// # Errors
    ///
    /// Fails if an MLX operation fails.
    pub fn profile(&self, reps: usize) -> Result<Profile> {
        self.model.profile(reps)
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

/// Run GPU work on a fresh default stream. A command buffer that failed (e.g. GPU work
/// submitted while an iOS app was in the background) poisons its stream for good: every later
/// operation on it fails. A [`Session`] does this itself when a step fails, and reads its
/// conversation again.
pub fn reset_gpu() {
    use ironbark_mlx::{Device, Stream};
    Stream::set_default(&Stream::new_with_device(&Device::gpu()));
}

/// A seed that differs per session, so the same prompt can get another reply.
fn seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "tests")]

    use super::*;

    /// After a GPU failure the session reads its conversation again: a greedy reply interrupted
    /// by [`Session::recover`] goes on token for token as one that was not. Needs
    /// `IRONBARK_MODEL` (skipped otherwise).
    #[test]
    fn recover_goes_on_where_the_reply_was() {
        let Some(dir) = std::env::var_os("IRONBARK_MODEL") else {
            return;
        };
        let mut s = Session::open(
            dir,
            Options {
                budget: Some(2_000_000_000),
                ..Options::default()
            },
        )
        .unwrap();
        s.set_sampling(Some(Sampling::GREEDY));
        s.set_thinking(false);
        s.set_system(Some("You are terse."), &[]);
        let ask = |s: &mut Session, recover_at: Option<usize>| -> String {
            s.reset();
            s.chat("Name three rivers in Europe.").unwrap();
            let mut text = String::new();
            for i in 0.. {
                if Some(i) == recover_at {
                    s.recover().unwrap();
                }
                match s.next_piece().unwrap() {
                    Some(p) => text += &p,
                    None => break,
                }
            }
            text
        };
        let plain = ask(&mut s, None);
        let recovered = ask(&mut s, Some(6));
        assert_eq!(plain, recovered);
        assert_eq!(s.ids.len(), s.context());
    }
}
