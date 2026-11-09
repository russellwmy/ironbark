//! The server over real TCP, the model scripted.

#![expect(clippy::unwrap_used, reason = "tests")]

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ironbark::{Message, Sampling};
use ironbark_serve::{Backend, BackendError, Config, Stats, serve};
use serde_json::{Value, json};

/// What the scripted model was told, and its replies to come.
#[derive(Default)]
struct Script {
    replies: VecDeque<Vec<String>>,
    /// Each `send`: its messages as text.
    sends: Vec<Vec<String>>,
    /// Each `set_system`: the system turn and how many tools.
    systems: Vec<(Option<String>, usize)>,
    thinking: Vec<bool>,
    tool_calls: Vec<bool>,
    sampling: Vec<Sampling>,
    pieces_given: usize,
    refuse_next: bool,
}

struct Scripted {
    script: Arc<Mutex<Script>>,
    reply: VecDeque<String>,
    written: usize,
    read: usize,
    context: usize,
}

impl Backend for Scripted {
    fn is_qwen(&self) -> bool {
        true
    }

    fn set_system(&mut self, system: Option<&str>, tools: &[Value]) {
        self.script.lock().unwrap().systems.push((system.map(str::to_owned), tools.len()));
        self.context = 0;
    }

    fn set_thinking(&mut self, on: bool) {
        self.script.lock().unwrap().thinking.push(on);
    }

    fn set_tool_calls(&mut self, on: bool) {
        self.script.lock().unwrap().tool_calls.push(on);
    }

    fn default_sampling(&self) -> Sampling {
        Sampling {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            presence_penalty: 1.5,
        }
    }

    fn set_sampling(&mut self, sampling: Sampling) {
        self.script.lock().unwrap().sampling.push(sampling);
    }

    fn send(&mut self, messages: &[Message<'_>]) -> Result<(), BackendError> {
        let mut s = self.script.lock().unwrap();
        if std::mem::take(&mut s.refuse_next) {
            return Err(BackendError {
                message: "context full: 99 tokens needed, 9 allowed".into(),
                context_full: true,
            });
        }
        s.sends.push(messages.iter().map(|m| format!("{m:?}")).collect());
        self.reply = s.replies.pop_front().unwrap_or_default().into();
        self.read = messages.len() * 10;
        self.context += self.read;
        self.written = 0;
        Ok(())
    }

    fn next_piece(&mut self) -> Result<Option<String>, BackendError> {
        let Some(piece) = self.reply.pop_front() else {
            return Ok(None);
        };
        self.script.lock().unwrap().pieces_given += 1;
        self.written += 1;
        self.context += 1;
        Ok(Some(piece))
    }

    fn context_ran_out(&self) -> bool {
        false
    }

    fn stats(&self) -> Stats {
        Stats {
            read: self.read,
            written: self.written,
            context: self.context,
            prefill_tps: 100.0,
            decode_tps: 50.0,
        }
    }
}

struct Server {
    addr: SocketAddr,
    script: Arc<Mutex<Script>>,
}

fn start(api_key: Option<&str>, think: bool, max_tokens: usize) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let script = Arc::new(Mutex::new(Script::default()));
    let backend = Scripted {
        script: Arc::clone(&script),
        reply: VecDeque::new(),
        written: 0,
        read: 0,
        context: 0,
    };
    let config = Config {
        model_id: "test-model".into(),
        api_key: api_key.map(str::to_owned),
        think,
        max_tokens,
    };
    std::thread::spawn(move || {
        let log: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(|_: &str| {});
        serve(listener, backend, config, &log)
    });
    Server { addr, script }
}

impl Server {
    fn reply(&self, pieces: &[&str]) {
        self.script.lock().unwrap().replies.push_back(pieces.iter().map(|p| (*p).to_owned()).collect());
    }

    fn raw(&self, method: &str, path: &str, headers: &str, body: &str) -> (u16, String, String) {
        let mut s = TcpStream::connect(self.addr).unwrap();
        s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
        write!(s, "{method} {path} HTTP/1.1\r\nHost: x\r\n{headers}Content-Length: {}\r\n\r\n{body}", body.len())
            .unwrap();
        let mut out = String::new();
        s.read_to_string(&mut out).unwrap();
        let (head, body) = out.split_once("\r\n\r\n").unwrap();
        let status = head.split(' ').nth(1).unwrap().parse().unwrap();
        (status, head.to_owned(), body.to_owned())
    }

    fn chat(&self, body: &Value) -> (u16, Value) {
        let (status, _, text) = self.raw("POST", "/v1/chat/completions", "", &body.to_string());
        (status, serde_json::from_str(&text).unwrap())
    }

    /// A streamed request's events, the `[DONE]` required at the end and left out.
    fn stream(&self, body: &Value) -> Vec<Value> {
        let (status, head, text) = self.raw("POST", "/v1/chat/completions", "", &body.to_string());
        assert_eq!(status, 200, "{text}");
        assert!(head.contains("text/event-stream"), "{head}");
        let mut events: Vec<&str> = text.split("\n\n").filter(|e| !e.is_empty()).collect();
        assert_eq!(events.pop(), Some("data: [DONE]"));
        events.iter().map(|e| serde_json::from_str(e.strip_prefix("data: ").unwrap()).unwrap()).collect()
    }

    fn sends(&self) -> Vec<Vec<String>> {
        self.script.lock().unwrap().sends.clone()
    }
}

fn user(text: &str) -> Value {
    json!({"role": "user", "content": text})
}

#[test]
fn a_plain_reply_with_reasoning_and_usage() {
    let srv = start(None, true, 1000);
    srv.reply(&["\nLet me ", "think.\n</thi", "nk>\n\nIt is ", "27 °C."]);
    let (status, r) = srv.chat(&json!({"model": "anything", "messages": [user("Weather?")]}));
    assert_eq!(status, 200);
    assert_eq!(r["object"], "chat.completion");
    assert_eq!(r["model"], "test-model");
    let choice = &r["choices"][0];
    assert_eq!(choice["message"]["content"], "It is 27 °C.");
    assert_eq!(choice["message"]["reasoning_content"], "Let me think.");
    assert_eq!(choice["finish_reason"], "stop");
    assert_eq!(r["usage"]["completion_tokens"], 4);
    assert_eq!(r["usage"]["prompt_tokens"], 10);
    assert_eq!(r["usage"]["total_tokens"], 14);
}

#[test]
fn tool_calls_come_back_in_openai_form() {
    let srv = start(None, true, 1000);
    srv.reply(&[
        "why\n</think>\n\nLooking.\n<tool_call>\n<function=bash>\n<parameter=command>\nls -l\n</parameter>\n",
        "<parameter=n>\n3\n</parameter>\n</function>\n</tool_call>",
    ]);
    let (_, r) = srv.chat(&json!({
        "messages": [user("List")],
        "tools": [{"type": "function", "function": {"name": "bash", "parameters": {}}}],
    }));
    let choice = &r["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert_eq!(choice["message"]["content"], "Looking.");
    let call = &choice["message"]["tool_calls"][0];
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "bash");
    let args: Value = serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"command": "ls -l", "n": 3}));
    assert!(call["id"].as_str().unwrap().starts_with("call_"));
    assert_eq!(srv.script.lock().unwrap().systems[0].1, 1, "the tools reach the model");
}

#[test]
fn a_malformed_call_is_shown_not_lost() {
    let srv = start(None, true, 1000);
    srv.reply(&["</think>\n\n<tool_call>\n{oops}\n</tool_call>"]);
    let (_, r) = srv.chat(&json!({"messages": [user("x")]}));
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    assert!(r["choices"][0]["message"]["content"].as_str().unwrap().contains("{oops}"));
}

#[test]
fn streams_pieces_and_never_shows_a_tag() {
    let srv = start(None, true, 1000);
    srv.reply(&[
        "hm",
        "m</th",
        "ink>\n\nLoo",
        "king.\n<tool",
        "_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n</function>\n</tool_call>",
    ]);
    let events = srv.stream(&json!({
        "stream": true, "stream_options": {"include_usage": true},
        "messages": [user("x")],
    }));
    let delta = |e: &Value| e["choices"][0]["delta"].clone();
    assert_eq!(delta(&events[0])["role"], "assistant");
    let collect =
        |key: &str| events.iter().filter_map(|e| delta(e)[key].as_str().map(str::to_owned)).collect::<String>();
    assert_eq!(collect("reasoning_content"), "hmm");
    assert_eq!(collect("content"), "Looking.\n");
    let all = serde_json::to_string(&events).unwrap();
    assert!(!all.contains("</th") && !all.contains("<tool"), "{all}");
    let calls: Vec<&Value> = events.iter().filter(|e| e["choices"][0]["delta"]["tool_calls"].is_array()).collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"], "f");
    let finish: Vec<&Value> = events.iter().filter(|e| !e["choices"][0]["finish_reason"].is_null()).collect();
    assert_eq!(finish.len(), 1);
    assert_eq!(finish[0]["choices"][0]["finish_reason"], "tool_calls");
    let last = events.last().unwrap();
    assert_eq!(last["choices"], json!([]));
    assert_eq!(last["usage"]["completion_tokens"], 5);
}

#[test]
fn a_continued_conversation_reads_only_what_is_new() {
    let srv = start(None, true, 1000);
    let tools = json!([{"type": "function", "function": {"name": "bash"}}]);
    let system = json!({"role": "system", "content": "Be brief."});
    srv.reply(&[
        "</think>\n\n<tool_call>\n<function=bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>",
    ]);
    let (_, first) = srv.chat(&json!({"tools": tools, "messages": [system, user("List")]}));
    let call = first["choices"][0]["message"]["tool_calls"][0].clone();

    // The client sends everything back, with its call re-encoded and its own id.
    let args = serde_json::to_string_pretty(&json!({"command": "ls"})).unwrap();
    let assistant = json!({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "other-id", "type": "function", "function": {"name": call["function"]["name"], "arguments": args}}],
    });
    let tool = json!({"role": "tool", "tool_call_id": "other-id", "content": "a.txt"});
    srv.reply(&["</think>\n\nOne file."]);
    let (_, second) = srv.chat(&json!({"tools": tools, "messages": [system, user("List"), assistant, tool]}));
    assert_eq!(second["choices"][0]["message"]["content"], "One file.");
    assert_eq!(second["usage"]["prompt_tokens_details"]["cached_tokens"], 11);

    let sends = srv.sends();
    assert_eq!(sends.len(), 2);
    assert_eq!(sends[1].len(), 1, "only the tool result: {:?}", sends[1]);
    assert!(sends[1][0].contains("a.txt"));
    assert_eq!(srv.script.lock().unwrap().systems.len(), 1, "the conversation was not reset");

    // A third turn continues from there, with the model's text as the client got it.
    let said = json!({"role": "assistant", "content": "One file."});
    srv.reply(&["</think>Yes."]);
    srv.chat(&json!({"tools": tools, "messages": [system, user("List"), assistant, tool, said, user("Sure?")]}));
    assert_eq!(srv.sends()[2].len(), 1);
    assert_eq!(srv.script.lock().unwrap().systems.len(), 1);
}

#[test]
fn anything_else_starts_over_with_the_whole_history() {
    let srv = start(None, true, 1000);
    let said = |t: &str| json!({"role": "assistant", "content": t});
    srv.reply(&["</think>one"]);
    srv.chat(&json!({"messages": [user("a")]}));

    // An edited reply: the model must not go on from what it really said.
    srv.reply(&["</think>two"]);
    srv.chat(&json!({"messages": [user("a"), said("EDITED"), user("b")]}));
    let sends = srv.sends();
    assert_eq!(sends[1].len(), 3, "{:?}", sends[1]);
    assert!(sends[1][1].contains("EDITED"));
    assert_eq!(srv.script.lock().unwrap().systems.len(), 2);

    // Another system turn.
    srv.reply(&["</think>three"]);
    srv.chat(&json!({"messages": [
        {"role": "system", "content": "New."}, user("a"), said("EDITED"), user("b"), said("two"), user("c"),
    ]}));
    assert_eq!(srv.sends()[2].len(), 5);

    // The same request again (a retry) holds nothing new: it starts over too.
    srv.reply(&["</think>four"]);
    srv.chat(&json!({"messages": [user("a")]}));
    srv.reply(&["</think>five"]);
    srv.chat(&json!({"messages": [user("a")]}));
    let sends = srv.sends();
    assert_eq!(sends.last().unwrap().len(), 1);
    assert_eq!(srv.script.lock().unwrap().systems.len(), 5);

    // Reasoning switched for a request changes the prompt: from scratch as well.
    srv.reply(&["</think>six"]);
    srv.chat(&json!({
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [user("a"), said("five"), user("b")],
    }));
    assert_eq!(srv.sends().last().unwrap().len(), 3);
    assert_eq!(srv.script.lock().unwrap().thinking.last(), Some(&false));
}

#[test]
fn max_tokens_cuts_the_reply_and_the_next_request_starts_over() {
    let srv = start(None, true, 1000);
    srv.reply(&["</think>", "a", "b", "c", "d", "e"]);
    let (_, r) = srv.chat(&json!({"max_tokens": 3, "messages": [user("x")]}));
    assert_eq!(r["choices"][0]["finish_reason"], "length");
    assert_eq!(r["choices"][0]["message"]["content"], "ab");
    assert_eq!(srv.script.lock().unwrap().pieces_given, 3);

    srv.reply(&["</think>ok"]);
    srv.chat(&json!({"messages": [user("x"), {"role": "assistant", "content": "ab"}, user("more")]}));
    assert_eq!(srv.sends()[1].len(), 3, "a cut reply is not a conversation to go on from");
}

#[test]
fn a_cut_tool_call_is_not_reported_as_a_call() {
    let srv = start(None, true, 1000);
    srv.reply(&[
        "</think><tool_call>\n",
        "<function=f>\n",
        "</function>\n",
        "</tool_call>",
    ]);
    let (_, r) = srv.chat(&json!({"max_tokens": 3, "messages": [user("x")]}));
    assert_eq!(r["choices"][0]["finish_reason"], "length");
    assert!(r["choices"][0]["message"]["tool_calls"].is_null());
}

#[test]
fn the_server_default_limits_a_reply() {
    let srv = start(None, true, 2);
    srv.reply(&["</think>", "a", "b", "c"]);
    let (_, r) = srv.chat(&json!({"messages": [user("x")]}));
    assert_eq!(r["choices"][0]["finish_reason"], "length");
}

#[test]
fn stop_strings_end_the_reply() {
    let srv = start(None, true, 1000);
    srv.reply(&["</think>one EN", "D two", " three"]);
    let (_, r) = srv.chat(&json!({"stop": ["END"], "messages": [user("x")]}));
    assert_eq!(r["choices"][0]["message"]["content"], "one");
    assert_eq!(r["choices"][0]["finish_reason"], "stop");
    assert_eq!(srv.script.lock().unwrap().pieces_given, 2, "generation stops at the stop string");
}

#[test]
fn sampling_reasoning_and_tool_choice_reach_the_model() {
    let srv = start(None, false, 1000);
    srv.reply(&["</think>a"]);
    srv.chat(&json!({"temperature": 0.6, "top_p": 0.95, "tool_choice": "none", "messages": [user("x")]}));
    {
        let s = srv.script.lock().unwrap();
        assert_eq!(s.thinking, [false], "the server's default");
        assert_eq!(s.tool_calls, [false]);
        assert_eq!(
            s.sampling[0],
            Sampling {
                temperature: 0.6,
                top_k: 20,
                top_p: 0.95,
                presence_penalty: 1.5,
            }
        );
    }
    // The next request is back to the model's own: settings do not leak between requests.
    srv.reply(&["</think>b"]);
    srv.chat(&json!({"messages": [user("y")]}));
    let s = srv.script.lock().unwrap();
    assert_eq!(s.tool_calls, [false, true]);
    assert_eq!(s.sampling[1].temperature, 0.7);
}

#[test]
fn errors_use_openai_form_and_status() {
    let srv = start(None, true, 1000);
    let (status, r) = srv.chat(&json!({"messages": []}));
    assert_eq!(status, 400);
    assert_eq!(r["error"]["type"], "invalid_request_error");
    assert!(r["error"]["message"].as_str().unwrap().contains("last message"));

    srv.script.lock().unwrap().refuse_next = true;
    let (status, r) = srv.chat(&json!({"messages": [user("x")]}));
    assert_eq!(status, 400);
    assert_eq!(r["error"]["code"], "context_length_exceeded");

    // Streaming too: it is refused with a status, not an empty stream.
    srv.script.lock().unwrap().refuse_next = true;
    let (status, _, text) =
        srv.raw("POST", "/v1/chat/completions", "", &json!({"stream": true, "messages": [user("x")]}).to_string());
    assert_eq!(status, 400);
    assert!(text.contains("context_length_exceeded"));

    assert_eq!(srv.raw("GET", "/nope", "", "").0, 404);
    assert_eq!(srv.raw("GET", "/v1/chat/completions", "", "").0, 405);
    assert_eq!(srv.raw("POST", "/v1/chat/completions", "", "{oops").0, 400);
}

#[test]
fn models_health_and_the_api_key() {
    let srv = start(Some("sekret"), true, 1000);
    assert_eq!(srv.raw("GET", "/health", "", "").0, 200, "health needs no key");
    assert_eq!(srv.raw("GET", "/v1/models", "", "").0, 401);
    assert_eq!(srv.raw("GET", "/v1/models", "Authorization: Bearer wrong!\r\n", "").0, 401);
    let body = json!({"messages": [user("x")]}).to_string();
    let (status, _, text) = srv.raw("POST", "/v1/chat/completions", "", &body);
    assert_eq!(status, 401);
    assert!(text.contains("invalid_api_key"));
    assert!(srv.sends().is_empty(), "nothing reached the model");
    let (status, _, text) = srv.raw("GET", "/v1/models", "Authorization: Bearer sekret\r\n", "");
    assert_eq!(status, 200);
    let models: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(models["data"][0]["id"], "test-model");
}

#[expect(clippy::disallowed_methods, reason = "a test waiting on another thread's progress")]
fn pause(ms: u64) {
    std::thread::sleep(Duration::from_millis(ms));
}

fn wait_until(what: &str, done: impl Fn() -> bool) {
    let start = Instant::now();
    while !done() {
        assert!(start.elapsed() < Duration::from_secs(10), "timed out: {what}");
        pause(20);
    }
}

/// A very long reply, to see it stop when its client goes.
#[test]
fn a_client_that_leaves_stops_its_reply() {
    let srv = start(None, true, 1000);
    let long: Vec<String> = (0..2_000_000).map(|_| "x".to_owned()).collect();
    srv.script.lock().unwrap().replies.push_back(long);
    let mut s = TcpStream::connect(srv.addr).unwrap();
    let body = json!({"stream": true, "messages": [user("x")]}).to_string();
    write!(s, "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
    wait_until("the reply starts", || srv.script.lock().unwrap().pieces_given > 0);
    drop(s);
    let stopped_soon = || {
        let before = srv.script.lock().unwrap().pieces_given;
        pause(300);
        srv.script.lock().unwrap().pieces_given == before
    };
    wait_until("the reply stops", stopped_soon);
    assert!(srv.script.lock().unwrap().pieces_given < 2_000_000, "it did not run to the end");

    // And the server goes on, from scratch.
    srv.reply(&["</think>hello"]);
    let (status, r) = srv.chat(&json!({"messages": [user("again")]}));
    assert_eq!(status, 200);
    assert_eq!(r["choices"][0]["message"]["content"], "hello");
}

#[test]
fn requests_are_served_one_at_a_time_in_order() {
    let srv = Arc::new(start(None, true, 1000));
    for _ in 0..8 {
        srv.reply(&["</think>ok"]);
    }
    let threads: Vec<_> = (0..8)
        .map(|i| {
            let srv = Arc::clone(&srv);
            std::thread::spawn(move || {
                let (status, r) = srv.chat(&json!({"messages": [user(&format!("q{i}"))]}));
                assert_eq!(status, 200);
                assert_eq!(r["choices"][0]["message"]["content"], "ok");
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    assert_eq!(srv.sends().len(), 8);
}
