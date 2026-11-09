//! An OpenAI-compatible server for ironbark models, for agents such as Goose.
//!
//! ```text
//!   client ─HTTP─▶ connection thread ─job─▶ model thread (owns the model; one reply at a time)
//!          ◀─SSE── (one per request)  ◀─events─┘
//!
//!   each request: same system turn, tools and history as the conversation held?
//!     yes ─▶ read only the new messages        no ─▶ reset and read them all
//! ```
//!
//! Agents send the whole conversation every turn, so the second path would read it all again
//! each time; the first is what makes long sessions usable.
//!
//! Endpoints: `POST /v1/chat/completions` (streaming or not, tools, reasoning as
//! `reasoning_content`), `GET /v1/models`, `GET /health`. One model, so the request's `model` is
//! ignored. Not supported, and refused rather than ignored: images, `n` above 1, forced tool
//! choice, `response_format`. `seed` is ignored.

mod http;
mod openai;
mod server;
mod worker;

pub use server::{Config, serve};
pub use worker::{Backend, BackendError, Stats};
