//! `OpenAI`'s chat format: requests to conversation turns, replies to JSON, and splitting a
//! reply's stream into reasoning and text.

use ironbark::{Message, ToolCall};
use serde_json::{Value, json};

const THINK_END: &str = "</think>";
const CALL_START: &str = "<tool_call>";

/// A conversation turn, as compared with the conversation the model holds.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Turn {
    User(String),
    Assistant { text: String, calls: Vec<ToolCall> },
    ToolResults(Vec<String>),
}

/// Sampling a request asks for; what it leaves out stays the model's own.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct SamplingRequest {
    pub(crate) temperature: Option<f32>,
    pub(crate) top_p: Option<f32>,
    pub(crate) top_k: Option<usize>,
    pub(crate) presence_penalty: Option<f32>,
}

impl SamplingRequest {
    /// `base` with what the request sets.
    pub(crate) fn over(self, base: ironbark::Sampling) -> ironbark::Sampling {
        ironbark::Sampling {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_k: self.top_k.unwrap_or(base.top_k),
            top_p: self.top_p.unwrap_or(base.top_p),
            presence_penalty: self.presence_penalty.unwrap_or(base.presence_penalty),
        }
    }
}

/// A chat completion request.
#[derive(Debug, Clone)]
pub(crate) struct ChatRequest {
    pub(crate) system: Option<String>,
    pub(crate) tools: Vec<Value>,
    pub(crate) turns: Vec<Turn>,
    pub(crate) stream: bool,
    pub(crate) include_usage: bool,
    pub(crate) max_tokens: Option<usize>,
    pub(crate) stop: Vec<String>,
    pub(crate) sampling: SamplingRequest,
    pub(crate) think: Option<bool>,
    pub(crate) no_tools: bool,
}

/// A message's text: a string, null, or text parts.
fn content_text(content: &Value) -> Result<String, String> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut text = String::new();
            for part in parts {
                match (part["type"].as_str(), part["text"].as_str()) {
                    (Some("text" | "input_text"), Some(t)) => text.push_str(t),
                    (kind, _) => {
                        return Err(format!("content part of type {} is not supported", kind.unwrap_or("(none)")));
                    }
                }
            }
            Ok(text)
        }
        _ => Err("content must be a string or an array of parts".into()),
    }
}

/// An assistant message's tool calls.
fn message_calls(message: &Value) -> Result<Vec<ToolCall>, String> {
    let Some(calls) = message["tool_calls"].as_array() else {
        return Ok(Vec::new());
    };
    calls
        .iter()
        .map(|c| {
            let f = &c["function"];
            let name = f["name"].as_str().ok_or("a tool call has no function name")?;
            let arguments = match &f["arguments"] {
                Value::Null => json!({}),
                Value::String(s) if s.trim().is_empty() => json!({}),
                Value::String(s) => {
                    serde_json::from_str(s).map_err(|e| format!("tool call {name}: arguments are not JSON: {e}"))?
                }
                other => other.clone(),
            };
            Ok(ToolCall {
                name: name.to_owned(),
                arguments,
            })
        })
        .collect()
}

fn optional_f32(body: &Value, key: &str, range: std::ops::RangeInclusive<f32>) -> Result<Option<f32>, String> {
    match &body[key] {
        Value::Null => Ok(None),
        v => {
            let x = v.as_f64().ok_or_else(|| format!("{key} must be a number"))? as f32;
            if range.contains(&x) {
                Ok(Some(x))
            } else {
                Err(format!("{key} must be in {}..={}", range.start(), range.end()))
            }
        }
    }
}

/// The request in `body`; the error is the message for the client.
pub(crate) fn parse_request(body: &[u8]) -> Result<ChatRequest, String> {
    let body: Value = serde_json::from_slice(body).map_err(|e| format!("body is not JSON: {e}"))?;
    let messages = body["messages"].as_array().ok_or("messages must be an array")?;
    if body["n"].as_u64().is_some_and(|n| n > 1) {
        return Err("n above 1 is not supported".into());
    }
    let format = body["response_format"]["type"].as_str();
    if format.is_some_and(|t| t != "text") {
        return Err("response_format is not supported".into());
    }
    let no_tools = match &body["tool_choice"] {
        Value::Null => false,
        Value::String(s) if s == "auto" => false,
        Value::String(s) if s == "none" => true,
        _ => return Err("tool_choice must be auto or none".into()),
    };
    let (mut system, mut turns): (Vec<String>, Vec<Turn>) = (Vec::new(), Vec::new());
    for m in messages {
        let text = || content_text(&m["content"]);
        match m["role"].as_str() {
            Some("system" | "developer") => system.push(text()?),
            Some("user") => turns.push(Turn::User(text()?)),
            Some("assistant") => turns.push(Turn::Assistant {
                text: text()?.trim().to_owned(),
                calls: message_calls(m)?,
            }),
            Some("tool") => {
                if let Some(Turn::ToolResults(results)) = turns.last_mut() {
                    results.push(text()?);
                } else {
                    turns.push(Turn::ToolResults(vec![text()?]));
                }
            }
            other => return Err(format!("message role {} is not supported", other.unwrap_or("(none)"))),
        }
    }
    if !matches!(turns.last(), Some(Turn::User(_) | Turn::ToolResults(_))) {
        return Err("the last message must be from the user or a tool".into());
    }
    let tools = body["tools"].as_array().cloned().unwrap_or_default();
    if let Some(bad) = tools.iter().find(|t| t["function"]["name"].as_str().is_none()) {
        return Err(format!("a tool has no function name: {bad}"));
    }
    let stop = match &body["stop"] {
        Value::Null => Vec::new(),
        Value::String(s) => vec![s.clone()],
        Value::Array(a) => a.iter().filter_map(|s| s.as_str().map(str::to_owned)).collect(),
        _ => return Err("stop must be a string or an array".into()),
    };
    let max_tokens = match (&body["max_completion_tokens"], &body["max_tokens"]) {
        (Value::Null, Value::Null) => None,
        (Value::Null, v) | (v, _) => Some(v.as_u64().ok_or("max_tokens must be a whole number")? as usize),
    };
    let think = body["chat_template_kwargs"]["enable_thinking"].as_bool();
    Ok(ChatRequest {
        system: (!system.is_empty()).then(|| system.join("\n\n")),
        tools,
        turns,
        stream: body["stream"].as_bool().unwrap_or(false),
        include_usage: body["stream_options"]["include_usage"].as_bool().unwrap_or(false),
        max_tokens,
        stop: stop.into_iter().filter(|s| !s.is_empty()).collect(),
        sampling: SamplingRequest {
            temperature: optional_f32(&body, "temperature", 0.0..=2.0)?,
            top_p: optional_f32(&body, "top_p", 0.0..=1.0)?,
            top_k: body["top_k"].as_u64().map(|k| k as usize),
            presence_penalty: optional_f32(&body, "presence_penalty", -2.0..=2.0)?,
        },
        think,
        no_tools,
    })
}

/// A call as the model writes it.
fn render_call(call: &ToolCall, qwen: bool) -> String {
    if !qwen {
        let body = json!({"name": call.name, "arguments": call.arguments});
        return format!("<tool_call>\n{body}\n</tool_call>");
    }
    let mut s = format!("<tool_call>\n<function={}>\n", call.name);
    if let Some(args) = call.arguments.as_object() {
        for (key, value) in args {
            let v = value.as_str().map_or_else(|| value.to_string(), str::to_owned);
            s.push_str("<parameter=");
            s.push_str(key);
            s.push_str(">\n");
            s.push_str(&v);
            s.push_str("\n</parameter>\n");
        }
    }
    s.push_str("</function>\n</tool_call>");
    s
}

/// Turns ready to hand to the model ([`Rendered::messages`]).
#[derive(Debug)]
pub(crate) enum Rendered<'a> {
    User(&'a str),
    Assistant(String),
    ToolResults(Vec<&'a str>),
}

impl Rendered<'_> {
    pub(crate) fn message(&self) -> Message<'_> {
        match self {
            Self::User(s) => Message::User(s),
            Self::Assistant(s) => Message::Assistant(s),
            Self::ToolResults(r) => Message::ToolResults(r),
        }
    }
}

/// `turns` in the model's chat format.
pub(crate) fn render(turns: &[Turn], qwen: bool) -> Vec<Rendered<'_>> {
    turns
        .iter()
        .map(|t| match t {
            Turn::User(s) => Rendered::User(s),
            Turn::ToolResults(r) => Rendered::ToolResults(r.iter().map(String::as_str).collect()),
            Turn::Assistant { text, calls } => {
                let mut parts = Vec::with_capacity(calls.len() + 1);
                if !text.is_empty() {
                    parts.push(text.clone());
                }
                parts.extend(calls.iter().map(|c| render_call(c, qwen)));
                Rendered::Assistant(parts.join("\n"))
            }
        })
        .collect()
}

/// Part of a reply for the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Piece {
    Reasoning(String),
    Text(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Reasoning,
    Text,
    /// Tool calls, or past a stop string: the rest is not shown.
    Hidden,
}

/// Splits a reply's stream at `</think>`, `<tool_call>` and stop strings, holding back text that
/// may be the start of one of them. A reply starts inside its reasoning.
#[derive(Debug)]
pub(crate) struct Splitter {
    stops: Vec<String>,
    held: String,
    phase: Phase,
    /// Nothing shown yet in this phase: leading white space is dropped.
    fresh: bool,
    stopped: bool,
}

impl Splitter {
    pub(crate) fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            held: String::new(),
            phase: Phase::Reasoning,
            fresh: true,
            stopped: false,
        }
    }

    /// A stop string ended the reply.
    pub(crate) const fn stopped(&self) -> bool {
        self.stopped
    }

    fn markers(&self) -> Vec<&str> {
        match self.phase {
            Phase::Reasoning => vec![THINK_END, CALL_START],
            Phase::Text => {
                let mut m = vec![CALL_START];
                m.extend(self.stops.iter().map(String::as_str));
                m
            }
            Phase::Hidden => Vec::new(),
        }
    }

    fn push(&mut self, out: &mut Vec<Piece>, text: &str) {
        let text = if self.fresh { text.trim_start() } else { text };
        if text.is_empty() {
            return;
        }
        self.fresh = false;
        match self.phase {
            Phase::Reasoning => out.push(Piece::Reasoning(text.to_owned())),
            Phase::Text => out.push(Piece::Text(text.to_owned())),
            Phase::Hidden => {}
        }
    }

    /// The parts of `piece` that can be shown now.
    pub(crate) fn feed(&mut self, piece: &str) -> Vec<Piece> {
        let mut out = Vec::new();
        if self.phase == Phase::Hidden {
            return out;
        }
        self.held.push_str(piece);
        loop {
            let found = self
                .markers()
                .iter()
                .filter_map(|m| self.held.find(m).map(|i| (i, m.to_string())))
                .min_by_key(|(i, _)| *i);
            let Some((at, marker)) = found else { break };
            let before = self.held[..at].to_owned();
            let after = self.held[at + marker.len()..].to_owned();
            self.push(&mut out, &before);
            self.held.clear();
            if marker == THINK_END {
                self.phase = Phase::Text;
                self.fresh = true;
                self.held = after;
            } else {
                self.stopped = marker != CALL_START;
                self.phase = Phase::Hidden;
                return out;
            }
        }
        // Everything but what may start a marker.
        let keep = self.markers().iter().map(|m| partial_suffix(&self.held, m)).max().unwrap_or(0);
        let shown = self.held.len() - keep;
        let text = self.held[..shown].to_owned();
        self.held.drain(..shown);
        self.push(&mut out, &text);
        out
    }

    /// What was held back, at the end of the reply.
    pub(crate) fn finish(&mut self) -> Vec<Piece> {
        let mut out = Vec::new();
        let held = std::mem::take(&mut self.held);
        self.push(&mut out, &held);
        out
    }
}

/// Bytes at the end of `text` that are a proper start of `marker`.
fn partial_suffix(text: &str, marker: &str) -> usize {
    marker
        .char_indices()
        .map(|(i, _)| i)
        .filter(|&i| i > 0 && text.ends_with(&marker[..i]))
        .max()
        .unwrap_or(0)
}

/// The JSON of tool calls; `call_base` makes their ids unique.
pub(crate) fn calls_json(calls: &[ToolCall], call_base: u64) -> Vec<Value> {
    calls
        .iter()
        .enumerate()
        .map(|(i, c)| {
            json!({
                "index": i,
                "id": format!("call_{call_base}_{i}"),
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments.to_string()},
            })
        })
        .collect()
}

/// An error body.
pub(crate) fn error_json(message: &str, kind: &str, code: Option<&str>) -> String {
    json!({"error": {"message": message, "type": kind, "code": code, "param": null}}).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(v: &Value) -> Result<ChatRequest, String> {
        parse_request(v.to_string().as_bytes())
    }

    #[test]
    fn messages_become_turns_and_tool_results_merge() {
        let r = parse(&json!({
            "stream": true, "stream_options": {"include_usage": true}, "max_tokens": 5,
            "temperature": 0.6, "stop": "END",
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "messages": [
                {"role": "system", "content": "a"},
                {"role": "developer", "content": [{"type": "text", "text": "b"}]},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "x", "type": "function", "function": {"name": "f", "arguments": "{\"k\": 1}"}},
                    {"id": "y", "type": "function", "function": {"name": "g", "arguments": ""}},
                ]},
                {"role": "tool", "tool_call_id": "x", "content": "one"},
                {"role": "tool", "tool_call_id": "y", "content": "two"},
            ],
        }))
        .unwrap();
        assert_eq!(r.system.as_deref(), Some("a\n\nb"));
        assert_eq!(r.turns.len(), 3);
        assert_eq!(r.turns[2], Turn::ToolResults(vec!["one".into(), "two".into()]));
        let Turn::Assistant { calls, .. } = &r.turns[1] else {
            unreachable!()
        };
        assert_eq!(calls[0].arguments, json!({"k": 1}));
        assert_eq!(calls[1].arguments, json!({}));
        assert!(r.stream && r.include_usage);
        assert_eq!(r.max_tokens, Some(5));
        assert_eq!(r.stop, ["END"]);
        assert_eq!(r.sampling.temperature, Some(0.6));
        assert_eq!(r.tools.len(), 1);
    }

    #[test]
    fn what_cannot_be_honoured_is_refused() {
        let user = json!([{"role": "user", "content": "x"}]);
        for (body, want) in [
            (json!({}), "messages"),
            (json!({"messages": []}), "last message"),
            (json!({"messages": [{"role": "assistant", "content": "x"}]}), "last message"),
            (json!({"messages": user, "n": 2}), "n above 1"),
            (json!({"messages": user, "tool_choice": "required"}), "tool_choice"),
            (json!({"messages": user, "response_format": {"type": "json_object"}}), "response_format"),
            (json!({"messages": user, "temperature": 3.0}), "temperature"),
            (json!({"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}), "image_url"),
            (json!({"messages": [{"role": "function", "content": "x"}]}), "function"),
            (json!({"messages": user, "tools": [{"type": "function"}]}), "function name"),
        ] {
            let e = parse(&body).unwrap_err();
            assert!(e.contains(want), "{body}: {e}");
        }
        assert!(parse_request(b"{oops").unwrap_err().contains("not JSON"));
    }

    #[test]
    fn calls_render_in_the_models_format() {
        let call = ToolCall {
            name: "bash".into(),
            arguments: json!({"command": "ls -l", "n": 3}),
        };
        let turns = [Turn::Assistant {
            text: "Looking.".into(),
            calls: vec![call],
        }];
        let Rendered::Assistant(q) = &render(&turns, true)[0] else {
            unreachable!()
        };
        assert_eq!(
            q,
            "Looking.\n<tool_call>\n<function=bash>\n<parameter=command>\nls -l\n</parameter>\n<parameter=n>\n3\n\
             </parameter>\n</function>\n</tool_call>"
        );
        let parsed = ironbark::parse(&format!("</think>{q}"));
        let Turn::Assistant { calls, .. } = &turns[0] else {
            unreachable!()
        };
        assert_eq!(&parsed.calls, calls, "what is rendered reads back the same");
        let Rendered::Assistant(m) = &render(&turns, false)[0] else {
            unreachable!()
        };
        assert_eq!(&ironbark::parse(&format!("</think>{m}")).calls, calls);
    }

    /// Every way of cutting `reply` into pieces gives the same parts.
    fn split_all_ways(reply: &str, stops: &[&str]) -> (String, String, bool) {
        let stops: Vec<String> = stops.iter().map(|s| (*s).to_owned()).collect();
        let run = |size: usize| {
            let mut s = Splitter::new(stops.clone());
            let chars: Vec<char> = reply.chars().collect();
            let mut pieces = Vec::new();
            for c in chars.chunks(size) {
                pieces.extend(s.feed(&c.iter().collect::<String>()));
            }
            pieces.extend(s.finish());
            let (mut reasoning, mut text) = (String::new(), String::new());
            for p in pieces {
                match p {
                    Piece::Reasoning(t) => reasoning.push_str(&t),
                    Piece::Text(t) => text.push_str(&t),
                }
            }
            (reasoning, text, s.stopped())
        };
        let first = run(1);
        for size in 2..=reply.chars().count().max(2) {
            assert_eq!(run(size), first, "pieces of {size}");
        }
        first
    }

    #[test]
    fn the_splitter_never_shows_a_marker() {
        assert_eq!(
            split_all_ways("\nhmm a < b\n</think>\n\nIt is <b>27</b> °C.", &[]),
            ("hmm a < b\n".into(), "It is <b>27</b> °C.".into(), false)
        );
        assert_eq!(
            split_all_ways("why\n</think>\n\nLook.\n<tool_call>\n<function=f>\n</function>\n</tool_call>", &[]),
            ("why\n".into(), "Look.\n".into(), false)
        );
        // No reasoning at all.
        assert_eq!(split_all_ways("<tool_call>x</tool_call>", &[]), (String::new(), String::new(), false));
        // Reasoning off: the reply starts with the end of its (empty) reasoning.
        assert_eq!(split_all_ways("</think>\n\nHi", &[]), (String::new(), "Hi".into(), false));
        // A marker's start that turns out to be text comes through at the end.
        assert_eq!(split_all_ways("</think>a <tool_c", &[]), (String::new(), "a <tool_c".into(), false));
    }

    #[test]
    fn stop_strings_end_the_text_but_not_the_reasoning() {
        assert_eq!(
            split_all_ways("END here\n</think>one END two", &["END"]),
            ("END here\n".into(), "one ".into(), true)
        );
        assert_eq!(split_all_ways("</think>héllo wörld", &["wö"]), (String::new(), "héllo ".into(), true));
    }

    #[test]
    fn requested_sampling_overrides_only_what_it_sets() {
        let base = ironbark::Sampling {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            presence_penalty: 1.5,
        };
        let r = SamplingRequest {
            temperature: Some(0.0),
            ..SamplingRequest::default()
        };
        assert_eq!(
            r.over(base),
            ironbark::Sampling {
                temperature: 0.0,
                ..base
            }
        );
    }
}
