//! Connections: a thread each, talking to the model thread through channels.

use std::io::{self, Write as _};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, Sender};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use crate::http::{self, ReadError};
use crate::openai::{self, Piece, calls_json, error_json};
use crate::worker::{Backend, Done, Event, Job, Stats, Worker};

/// Open connections; more are refused, each costs a thread.
const MAX_CONNECTIONS: usize = 64;
/// How often a waiting connection checks that its client is still there.
const POLL: Duration = Duration::from_millis(200);
/// How long a client may take to send its request, or to take what is written to it.
const IO_TIMEOUT: Duration = Duration::from_secs(60);

/// How to serve.
#[derive(Debug, Clone)]
pub struct Config {
    /// The id clients see.
    pub model_id: String,
    /// Clients must send this bearer token, if set.
    pub api_key: Option<String>,
    /// Whether replies reason first, unless a request says otherwise.
    pub think: bool,
    /// Longest reply, unless a request says otherwise.
    pub max_tokens: usize,
}

/// A logger.
type Log = Arc<dyn Fn(&str) + Send + Sync>;

/// Serve on `listener`, the model on this thread (MLX stays on one). Does not return unless the
/// accept thread cannot start.
///
/// # Errors
///
/// Fails if a thread cannot be started.
pub fn serve<B: Backend>(listener: TcpListener, backend: B, config: Config, log: &Log) -> io::Result<()> {
    let (jobs, queue) = mpsc::channel();
    let shared = Arc::new(config);
    let (accept_log, accept_config) = (Arc::clone(log), Arc::clone(&shared));
    std::thread::Builder::new()
        .name("accept".into())
        .spawn(move || accept(&listener, &jobs, &accept_config, &accept_log))?;
    Worker::new(backend, shared.think, shared.max_tokens).run(&queue, log.as_ref());
    Ok(())
}

fn accept(listener: &TcpListener, jobs: &Sender<Job>, config: &Arc<Config>, log: &Log) {
    let open = Arc::new(AtomicUsize::new(0));
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                log(&format!("accept: {e}"));
                continue;
            }
        };
        if open.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
            open.fetch_sub(1, Ordering::SeqCst);
            // The client is told if it can be; a failure here changes nothing.
            drop(http::write_json(&mut stream, 503, &error_json("too many connections", "server_error", None)));
            continue;
        }
        let (jobs, config, thread_log, opened) = (jobs.clone(), Arc::clone(config), Arc::clone(log), Arc::clone(&open));
        let spawned = std::thread::Builder::new().name("connection".into()).spawn(move || {
            if let Err(e) = connection(&mut stream, &jobs, &config) {
                // A client that left is not news; anything else is.
                if !matches!(e.kind(), io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset) {
                    thread_log(&format!("connection: {e}"));
                }
            }
            opened.fetch_sub(1, Ordering::SeqCst);
        });
        if let Err(e) = spawned {
            open.fetch_sub(1, Ordering::SeqCst);
            log(&format!("cannot start a connection thread: {e}"));
        }
    }
}

/// Whether two tokens are equal, in time that does not depend on where they differ.
fn same_token(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |d, (x, y)| d | (x ^ y)) == 0
}

fn connection(s: &mut TcpStream, jobs: &Sender<Job>, config: &Config) -> io::Result<()> {
    s.set_read_timeout(Some(IO_TIMEOUT))?;
    s.set_write_timeout(Some(IO_TIMEOUT))?;
    s.set_nodelay(true)?;
    let request = match http::read_request(s) {
        Ok(r) => r,
        Err(ReadError::Io) => return Ok(()),
        Err(ReadError::Bad(status, message)) => {
            return http::write_json(s, status, &error_json(message, "invalid_request_error", None));
        }
    };
    if request.method == "GET" && request.path == "/health" {
        return http::write_json(s, 200, &json!({"status": "ok"}).to_string());
    }
    if let Some(key) = &config.api_key {
        let sent = request.authorization.as_deref().and_then(|a| a.strip_prefix("Bearer "));
        if !sent.is_some_and(|t| same_token(t.trim(), key)) {
            let body = error_json("missing or wrong API key", "invalid_request_error", Some("invalid_api_key"));
            return http::write_json(s, 401, &body);
        }
    }
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/v1/models" | "/models") => {
            let model = json!({"id": config.model_id, "object": "model", "created": now(), "owned_by": "ironbark"});
            http::write_json(s, 200, &json!({"object": "list", "data": [model]}).to_string())
        }
        ("POST", "/v1/chat/completions" | "/chat/completions") => match openai::parse_request(&request.body) {
            Ok(chat) => completion(s, chat, jobs, config),
            Err(message) => http::write_json(s, 400, &error_json(&message, "invalid_request_error", None)),
        },
        (_, "/v1/models" | "/models" | "/v1/chat/completions" | "/chat/completions" | "/health") => {
            http::write_json(s, 405, &error_json("method not allowed", "invalid_request_error", None))
        }
        _ => http::write_json(s, 404, &error_json("no such endpoint", "invalid_request_error", None)),
    }
}

fn now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Ids of completions and their calls, unique in this process.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Whether the client has closed the connection.
fn client_gone(s: &TcpStream) -> bool {
    let mut byte = [0u8; 1];
    if s.set_nonblocking(true).is_err() {
        return true;
    }
    let gone = match s.peek(&mut byte) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => e.kind() != io::ErrorKind::WouldBlock,
    };
    s.set_nonblocking(false).is_err() || gone
}

fn usage(stats: &Stats) -> Value {
    let prompt = stats.context.saturating_sub(stats.written);
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": stats.written,
        "total_tokens": stats.context,
        "prompt_tokens_details": {"cached_tokens": prompt.saturating_sub(stats.read)},
    })
}

/// A streamed chunk.
fn chunk(id: &str, model: &str, delta: &Value, finish: Option<&str>) -> String {
    json!({
        "id": id, "object": "chat.completion.chunk", "created": now(), "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    })
    .to_string()
}

fn status_of(e: &crate::worker::BackendError) -> (u16, String) {
    let (status, kind, code) = if e.context_full {
        (400, "invalid_request_error", Some("context_length_exceeded"))
    } else {
        (500, "server_error", None)
    };
    (status, error_json(&e.message, kind, code))
}

/// Run a chat completion on the model thread and answer with its events.
fn completion(s: &mut TcpStream, chat: openai::ChatRequest, jobs: &Sender<Job>, config: &Config) -> io::Result<()> {
    let number = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let id = format!("chatcmpl-{number}");
    let (stream, include_usage) = (chat.stream, chat.include_usage);
    let (events, replies) = mpsc::channel();
    let job = Job { request: chat, events };
    if jobs.send(job).is_err() {
        return http::write_json(s, 503, &error_json("the model thread has stopped", "server_error", None));
    }
    let (mut started, mut reasoning, mut text) = (false, String::new(), String::new());
    loop {
        let event = match replies.recv_timeout(POLL) {
            Ok(e) => e,
            Err(RecvTimeoutError::Timeout) => {
                if client_gone(s) {
                    // Dropping the receiver stops the reply at its next piece.
                    return Ok(());
                }
                continue;
            }
            Err(RecvTimeoutError::Disconnected) => {
                return http::write_json(s, 500, &error_json("the reply ended unexpectedly", "server_error", None));
            }
        };
        match event {
            Event::Started => {
                started = true;
                if stream {
                    http::write_stream_head(s)?;
                    let delta = json!({"role": "assistant", "content": ""});
                    http::write_event(s, &chunk(&id, &config.model_id, &delta, None))?;
                }
            }
            Event::Piece(piece) if stream => {
                let delta = match &piece {
                    Piece::Reasoning(t) => json!({"reasoning_content": t}),
                    Piece::Text(t) => json!({"content": t}),
                };
                http::write_event(s, &chunk(&id, &config.model_id, &delta, None))?;
            }
            Event::Piece(Piece::Reasoning(t)) => reasoning.push_str(&t),
            Event::Piece(Piece::Text(t)) => text.push_str(&t),
            Event::Done(done) => {
                if stream {
                    finish_stream(s, &id, config, &done, number, include_usage)?;
                } else {
                    let body = whole_reply(&id, config, &done, number);
                    http::write_json(s, 200, &body)?;
                }
                return s.shutdown(Shutdown::Write);
            }
            Event::Failed(e) => {
                let (status, body) = status_of(&e);
                if started && stream {
                    // Too late for a status: the error goes in the stream.
                    http::write_event(s, &body)?;
                    return s.flush();
                }
                return http::write_json(s, status, &body);
            }
        }
    }
}

fn finish_stream(
    s: &mut TcpStream,
    id: &str,
    config: &Config,
    done: &Done,
    number: u64,
    include_usage: bool,
) -> io::Result<()> {
    let model = &config.model_id;
    if !done.calls.is_empty() {
        let delta = json!({"tool_calls": calls_json(&done.calls, number)});
        http::write_event(s, &chunk(id, model, &delta, None))?;
    }
    http::write_event(s, &chunk(id, model, &json!({}), Some(done.finish)))?;
    if include_usage {
        let last = json!({
            "id": id, "object": "chat.completion.chunk", "created": now(), "model": model,
            "choices": [], "usage": usage(&done.stats),
        });
        http::write_event(s, &last.to_string())?;
    }
    http::write_event(s, "[DONE]")
}

fn whole_reply(id: &str, config: &Config, done: &Done, number: u64) -> String {
    let mut message = json!({
        "role": "assistant",
        "content": if done.text.is_empty() && !done.calls.is_empty() { Value::Null } else { json!(done.text) },
    });
    if !done.reasoning.is_empty() {
        message["reasoning_content"] = json!(done.reasoning);
    }
    if !done.calls.is_empty() {
        message["tool_calls"] = json!(calls_json(&done.calls, number));
    }
    json!({
        "id": id, "object": "chat.completion", "created": now(), "model": config.model_id,
        "choices": [{"index": 0, "message": message, "finish_reason": done.finish}],
        "usage": usage(&done.stats),
    })
    .to_string()
}
