//! The model thread: one reply at a time, reusing the conversation it holds when it can.

use std::sync::mpsc::{Receiver, Sender};

use ironbark::{Finish, Message, Sampling, Session, ToolCall, Value};

use crate::openai::{ChatRequest, Piece, Splitter, Turn, render};

/// Why the model refused or failed.
#[derive(Debug, Clone)]
pub struct BackendError {
    /// For the client.
    pub message: String,
    /// The conversation does not fit the context.
    pub context_full: bool,
}

/// Tokens of the turn that just ended.
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    /// Tokens read for this turn (the rest of the prompt was already in the caches).
    pub read: usize,
    /// Tokens written.
    pub written: usize,
    /// Tokens in the conversation.
    pub context: usize,
    /// Prompt tokens per second.
    pub prefill_tps: f64,
    /// Reply tokens per second.
    pub decode_tps: f64,
}

/// What the server needs of a model.
pub trait Backend {
    /// Whether calls are written as Qwen's XML, not JSON.
    fn is_qwen(&self) -> bool;
    /// Set the system turn and forget the conversation.
    fn set_system(&mut self, system: Option<&str>, tools: &[Value]);
    /// Whether replies reason first.
    fn set_thinking(&mut self, on: bool);
    /// Whether replies may call tools.
    fn set_tool_calls(&mut self, on: bool);
    /// The model's own sampling.
    fn default_sampling(&self) -> Sampling;
    /// Sample this way from now on.
    fn set_sampling(&mut self, sampling: Sampling);
    /// Read `messages` after the conversation and start a reply.
    ///
    /// # Errors
    ///
    /// Fails if they do not fit or the model fails.
    fn send(&mut self, messages: &[Message<'_>]) -> Result<(), BackendError>;
    /// The next piece of the reply: `Some("")` when a character is incomplete, `None` at its end.
    ///
    /// # Errors
    ///
    /// Fails if the model fails.
    fn next_piece(&mut self) -> Result<Option<String>, BackendError>;
    /// Whether the context filled up (rather than the model ending its turn).
    fn context_ran_out(&self) -> bool;
    /// Tokens of the current turn.
    fn stats(&self) -> Stats;
}

fn error(e: &ironbark::Error) -> BackendError {
    BackendError {
        message: e.to_string(),
        context_full: matches!(e, ironbark::Error::ContextFull { .. }),
    }
}

impl Backend for Session {
    fn is_qwen(&self) -> bool {
        Self::is_qwen(self)
    }

    fn set_system(&mut self, system: Option<&str>, tools: &[Value]) {
        Self::set_system(self, system, tools);
    }

    fn set_thinking(&mut self, on: bool) {
        Self::set_thinking(self, on);
    }

    fn set_tool_calls(&mut self, on: bool) {
        Self::set_tool_calls(self, on);
    }

    fn default_sampling(&self) -> Sampling {
        self.sampling()
    }

    fn set_sampling(&mut self, sampling: Sampling) {
        Self::set_sampling(self, Some(sampling));
    }

    fn send(&mut self, messages: &[Message<'_>]) -> Result<(), BackendError> {
        Self::send(self, messages).map(drop).map_err(|e| error(&e))
    }

    fn next_piece(&mut self) -> Result<Option<String>, BackendError> {
        Self::next_piece(self).map_err(|e| error(&e))
    }

    fn context_ran_out(&self) -> bool {
        matches!(self.finish(), Some(Finish::Context))
    }

    fn stats(&self) -> Stats {
        let s = Self::stats(self);
        // The end token is in the caches, but neither prompt nor reply.
        let end_token = usize::from(matches!(self.finish(), Some(Finish::Stop)));
        Stats {
            read: s.prompt_tokens,
            written: s.tokens,
            context: s.context.saturating_sub(end_token),
            prefill_tps: s.prefill_tps,
            decode_tps: s.decode_tps,
        }
    }
}

/// How a reply ended, for the client.
#[derive(Debug, Clone)]
pub(crate) struct Done {
    pub(crate) reasoning: String,
    pub(crate) text: String,
    pub(crate) calls: Vec<ToolCall>,
    /// `stop`, `tool_calls` or `length`.
    pub(crate) finish: &'static str,
    pub(crate) stats: Stats,
}

/// What the model thread says about a job.
#[derive(Debug)]
pub(crate) enum Event {
    /// The prompt fits and was read.
    Started,
    Piece(Piece),
    Done(Box<Done>),
    Failed(BackendError),
}

/// A request for the model thread; dropping the event receiver cancels it.
#[derive(Debug)]
pub(crate) struct Job {
    pub(crate) request: ChatRequest,
    pub(crate) events: Sender<Event>,
}

/// The conversation the model holds.
#[derive(Debug, PartialEq)]
struct Held {
    system: Option<String>,
    tools: Vec<Value>,
    think: bool,
    turns: Vec<Turn>,
}

/// The model thread's state.
#[derive(Debug)]
pub(crate) struct Worker<B> {
    backend: B,
    /// `None`: what the model holds is not a conversation a client can continue.
    held: Option<Held>,
    base_sampling: Sampling,
    think: bool,
    max_tokens: usize,
}

impl<B: Backend> Worker<B> {
    pub(crate) fn new(backend: B, think: bool, max_tokens: usize) -> Self {
        let base_sampling = backend.default_sampling();
        Self {
            backend,
            held: None,
            base_sampling,
            think,
            max_tokens,
        }
    }

    /// Serve jobs until the channel closes.
    pub(crate) fn run(&mut self, jobs: &Receiver<Job>, log: &(dyn Fn(&str) + Send + Sync)) {
        while let Ok(job) = jobs.recv() {
            let events = job.events.clone();
            match self.reply(job.request, &events) {
                Ok(Some(line)) => log(&line),
                Ok(None) => log("cancelled: the client went away"),
                Err(e) => {
                    log(&format!("failed: {}", e.message));
                    // The receiver may be gone too; there is nobody else to tell.
                    drop(events.send(Event::Failed(e)));
                }
            }
        }
    }

    /// Why `request` does not continue the conversation held; `None` if it does.
    fn mismatch(&self, request: &ChatRequest, think: bool) -> Option<String> {
        let Some(h) = &self.held else {
            return Some("nothing held".to_owned());
        };
        if h.system != request.system {
            return Some("system turn changed".to_owned());
        }
        if h.tools != request.tools {
            return Some("tools changed".to_owned());
        }
        if h.think != think {
            return Some("reasoning switched".to_owned());
        }
        if request.turns.len() <= h.turns.len() {
            return Some(format!("{} turns, {} held", request.turns.len(), h.turns.len()));
        }
        h.turns.iter().zip(&request.turns).position(|(a, b)| a != b).map(|i| {
            let short = |t: &Turn| format!("{t:?}").chars().take(160).collect::<String>();
            format!("turn {i} differs: held {} sent {}", short(&h.turns[i]), short(&request.turns[i]))
        })
    }

    /// The turns to read for `request`, and why from scratch if so: only the new ones when it
    /// continues the conversation.
    fn new_turns<'a>(&mut self, request: &'a ChatRequest, think: bool) -> (&'a [Turn], Option<String>) {
        match self.mismatch(request, think) {
            None => (&request.turns[self.held.as_ref().map_or(0, |h| h.turns.len())..], None),
            Some(why) => {
                self.backend.set_system(request.system.as_deref(), &request.tools);
                (&request.turns, Some(why))
            }
        }
    }

    /// One reply: a line for the log, or `None` if the client went away.
    fn reply(&mut self, request: ChatRequest, events: &Sender<Event>) -> Result<Option<String>, BackendError> {
        let think = request.think.unwrap_or(self.think);
        self.backend.set_thinking(think);
        self.backend.set_tool_calls(!request.no_tools);
        self.backend.set_sampling(request.sampling.over(self.base_sampling));
        let (new, scratch) = self.new_turns(&request, think);
        let rendered = render(new, self.backend.is_qwen());
        let messages: Vec<Message<'_>> = rendered.iter().map(crate::openai::Rendered::message).collect();
        // Whatever goes wrong from here, the caches no longer match `held`.
        self.held = None;
        self.backend.send(&messages)?;
        if events.send(Event::Started).is_err() {
            return Ok(None);
        }
        let limit = request.max_tokens.unwrap_or(self.max_tokens);
        let mut splitter = Splitter::new(request.stop.clone());
        let (mut full, mut text, mut steps, mut cut) = (String::new(), String::new(), 0usize, false);
        loop {
            if steps >= limit && self.backend.stats().written >= limit {
                cut = true;
                break;
            }
            let Some(piece) = self.backend.next_piece()? else {
                break;
            };
            steps += 1;
            full.push_str(&piece);
            for part in splitter.feed(&piece) {
                if let Piece::Text(t) = &part {
                    text.push_str(t);
                }
                if events.send(Event::Piece(part)).is_err() {
                    return Ok(None);
                }
            }
            if splitter.stopped() {
                break;
            }
        }
        for part in splitter.finish() {
            if let Piece::Text(t) = &part {
                text.push_str(t);
            }
            if events.send(Event::Piece(part)).is_err() {
                return Ok(None);
            }
        }
        let ran_out = self.backend.context_ran_out();
        let mut parsed = ironbark::parse(&full);
        // Calls the model got wrong: shown as text, not lost.
        for m in &parsed.malformed {
            let raw = format!("<tool_call>{m}</tool_call>");
            if events.send(Event::Piece(Piece::Text(raw.clone()))).is_err() {
                return Ok(None);
            }
            text.push_str(&raw);
        }
        if splitter.stopped() || cut || ran_out {
            parsed.calls.clear();
        }
        let finish = if cut || ran_out {
            "length"
        } else if parsed.calls.is_empty() {
            "stop"
        } else {
            "tool_calls"
        };
        let stats = self.backend.stats();
        let complete = !cut && !ran_out && !splitter.stopped();
        let mut turns = request.turns;
        let trimmed = text.trim().to_owned();
        turns.push(Turn::Assistant {
            text: trimmed.clone(),
            calls: parsed.calls.clone(),
        });
        let line = format!(
            "{} turns, read {} tokens ({} tok/s){}, wrote {} ({} tok/s), context {}, {finish}",
            turns.len(),
            stats.read,
            stats.prefill_tps as u64,
            scratch.map_or_else(String::new, |why| format!(" from scratch ({why})")),
            stats.written,
            stats.decode_tps as u64,
            stats.context,
        );
        if complete {
            self.held = Some(Held {
                system: request.system,
                tools: request.tools,
                think,
                turns,
            });
        }
        let done = Done {
            reasoning: parsed.thinking,
            text: trimmed,
            calls: parsed.calls,
            finish,
            stats,
        };
        Ok(events.send(Event::Done(Box::new(done))).is_ok().then_some(line))
    }
}
