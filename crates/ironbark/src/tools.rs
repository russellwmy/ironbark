//! Tool use in Maple's chat format (`chat_template.jinja`): tools are listed as JSON in the
//! system turn, the model answers with `<tool_call>` blocks, and the app sends the results back
//! as `<tool_response>` blocks in a user turn.
//!
//! ```text
//!   system  : # Tools … <tools>{"type":"function","function":{…}}</tools> …
//!   user    : what's the weather in Adelaide?
//!   model   : <think>…</think>
//!             <tool_call>{"name": "weather", "arguments": {"city": "Adelaide"}}</tool_call>
//!   app     : runs the tool -> Session::tool_results
//!   user    : <tool_response>{"temp_c": 22}</tool_response>
//!   model   : <think>…</think> It is 22 °C in Adelaide.
//! ```

use serde_json::Value;

/// A call the model asked for.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    /// Tool name.
    pub name: String,
    /// Arguments (a JSON object).
    pub arguments: Value,
}

/// A reply split into its parts.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Parsed {
    /// The reasoning between `<think>` and `</think>`.
    pub thinking: String,
    /// The answer: the reply without reasoning and tool calls.
    pub text: String,
    /// Tool calls, in order.
    pub calls: Vec<ToolCall>,
    /// `<tool_call>` blocks that are not `{"name": …, "arguments": …}` JSON.
    pub malformed: Vec<String>,
}

/// Split a reply (the reasoning opener `<think>\n` is part of the prompt, so the reply starts
/// inside the reasoning).
#[must_use]
pub fn parse(reply: &str) -> Parsed {
    let (thinking, rest) = match reply.split_once("</think>") {
        Some((t, rest)) => (t.trim_start_matches("<think>").trim(), rest),
        // Still reasoning (cut off), or the model skipped it.
        None if reply.contains("<tool_call>") => ("", reply),
        None => (reply.trim_start_matches("<think>").trim(), ""),
    };
    let mut p = Parsed {
        thinking: thinking.to_owned(),
        ..Parsed::default()
    };
    let mut text = String::new();
    let mut rest = rest;
    while let Some((before, after)) = rest.split_once("<tool_call>") {
        text.push_str(before);
        let (body, after) = after.split_once("</tool_call>").unwrap_or((after, ""));
        match serde_json::from_str::<Value>(body.trim()) {
            Ok(Value::Object(mut o)) if o.get("name").is_some_and(Value::is_string) => p.calls.push(ToolCall {
                name: o.remove("name").and_then(|n| n.as_str().map(str::to_owned)).unwrap_or_default(),
                arguments: o.remove("arguments").unwrap_or(Value::Object(serde_json::Map::new())),
            }),
            _ => p.malformed.push(body.trim().to_owned()),
        }
        rest = after;
    }
    text.push_str(rest);
    Parsed {
        text: text.trim().to_owned(),
        ..p
    }
}

/// The system turn: `system` and, with `tools`, their JSON schemas and the call format.
pub(crate) fn system_turn(system: Option<&str>, tools: &[Value]) -> String {
    if tools.is_empty() {
        return system.map_or_else(String::new, |s| format!("<|im_start|>system\n{s}<|im_end|>\n"));
    }
    let mut t = String::from("<|im_start|>system\n");
    if let Some(s) = system {
        t.push_str(s);
        t.push_str("\n\n");
    }
    t.push_str(
        "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with \
         function signatures within <tools></tools> XML tags:\n<tools>",
    );
    for tool in tools {
        t.push('\n');
        py_json(tool, &mut t);
    }
    t.push_str(
        "\n</tools>\n\nFor each function call, return a json object with function name and arguments within \
         <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": \
         <args-json-object>}\n</tool_call><|im_end|>\n",
    );
    t
}

/// `v` as the template's `tojson` writes it (Python's `json.dumps`: `", "` and `": "`).
fn py_json(v: &Value, out: &mut String) {
    match v {
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json(x, out);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&Value::String(k.clone()).to_string());
                out.push_str(": ");
                py_json(x, out);
            }
            out.push('}');
        }
        _ => out.push_str(&v.to_string()),
    }
}

/// A message of a conversation, read with [`crate::Session::send`].
#[derive(Debug, Clone, Copy)]
pub enum Message<'a> {
    /// What the user wrote.
    User(&'a str),
    /// An earlier reply as the model streamed it; its reasoning is left out, as the chat
    /// template leaves it out of earlier turns.
    Assistant(&'a str),
    /// Results of the tool calls of the reply before, in order.
    ToolResults(&'a [&'a str]),
}

impl Message<'_> {
    /// The message as a closed turn of the chat format.
    pub(crate) fn turn(&self) -> String {
        match *self {
            Message::User(m) => format!("<|im_start|>user\n{m}<|im_end|>\n"),
            Message::Assistant(reply) => {
                let answer = reply.split_once("</think>").map_or(reply, |(_, rest)| rest);
                format!("<|im_start|>assistant\n{}<|im_end|>\n", answer.trim())
            }
            Message::ToolResults(results) => results_turn(results),
        }
    }
}

/// A user turn holding tool results.
fn results_turn(results: &[&str]) -> String {
    let mut t = String::from("<|im_start|>user");
    for r in results {
        t.push_str("\n<tool_response>\n");
        t.push_str(r);
        t.push_str("\n</tool_response>");
    }
    t.push_str("<|im_end|>\n");
    t
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "tests")]

    use serde_json::json;

    use super::*;

    #[test]
    fn parses_reasoning_calls_and_text() {
        let p = parse(
            "\nLet me look it up.\n</think>\n\n<tool_call>\n{\"name\": \"weather\", \"arguments\": {\"city\": \
             \"HK\"}}\n</tool_call>\n<tool_call>\n{oops}\n</tool_call>",
        );
        assert_eq!(p.thinking, "Let me look it up.");
        assert_eq!(p.text, "");
        assert_eq!(
            p.calls,
            vec![ToolCall {
                name: "weather".into(),
                arguments: json!({"city": "HK"})
            }]
        );
        assert_eq!(p.malformed, vec!["{oops}".to_owned()]);

        let p = parse("hmm\n</think>\n\nIt is 27 °C.");
        assert_eq!((p.thinking.as_str(), p.text.as_str(), p.calls.len()), ("hmm", "It is 27 °C.", 0));
    }

    #[test]
    fn turns_match_the_chat_template() {
        assert_eq!(system_turn(None, &[]), "");
        assert_eq!(system_turn(Some("Be brief."), &[]), "<|im_start|>system\nBe brief.<|im_end|>\n");
        let t = system_turn(Some("Be brief."), &[json!({"type": "function", "f": [1, "x"]})]);
        assert!(t.starts_with("<|im_start|>system\nBe brief.\n\n# Tools\n"));
        assert!(t.contains("<tools>\n{\"type\": \"function\", \"f\": [1, \"x\"]}\n</tools>"), "{t}");
        assert_eq!(
            Message::ToolResults(&["a", "b"]).turn(),
            "<|im_start|>user\n<tool_response>\na\n</tool_response>\n<tool_response>\nb\n</tool_response><|im_end|>\n"
        );
        assert_eq!(Message::User("hi").turn(), "<|im_start|>user\nhi<|im_end|>\n");
        assert_eq!(
            Message::Assistant("hmm\n</think>\n\n<tool_call>\n{}\n</tool_call>").turn(),
            "<|im_start|>assistant\n<tool_call>\n{}\n</tool_call><|im_end|>\n"
        );
    }
}
