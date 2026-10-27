//! Tool use in the models' chat formats (`chat_template.jinja`): tools are listed as JSON in
//! the system turn, the model answers with `<tool_call>` blocks, and the app sends the results
//! back as `<tool_response>` blocks in a user turn. Maple's calls are JSON; Qwen3.5 / 3.6 write
//! them as XML (`<function=name><parameter=arg>value</parameter></function>`). [`parse`] reads
//! both.
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
    /// `<tool_call>` blocks that are neither `{"name": …, "arguments": …}` JSON nor a
    /// `<function=…>` block.
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
        if body.trim_start().starts_with("<function=") {
            match xml_call(body.trim()) {
                Some(c) => p.calls.push(c),
                None => p.malformed.push(body.trim().to_owned()),
            }
            rest = after;
            continue;
        }
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

/// A Qwen3.5 / 3.6 call: `<function=name>` then `<parameter=arg>\nvalue\n</parameter>` blocks.
/// A value is JSON when it parses as anything but a string (the template writes non-strings
/// as JSON), else the text itself.
fn xml_call(body: &str) -> Option<ToolCall> {
    // The name ends at `>` or, when a model drops the `>`, at the line end.
    let after = body.strip_prefix("<function=")?;
    let (name, rest) = after.split_at(after.find(['>', '\n'])?);
    let mut rest = rest.strip_prefix('>').unwrap_or(rest);
    let mut args = serde_json::Map::new();
    while let Some((_, after)) = rest.split_once("<parameter=") {
        let (key, after) = after.split_once('>')?;
        let (raw, after) = after.split_once("</parameter>")?;
        let raw = raw.strip_prefix('\n').unwrap_or(raw);
        let raw = raw.strip_suffix('\n').unwrap_or(raw);
        let value = match serde_json::from_str::<Value>(raw) {
            Ok(v) if !v.is_string() => v,
            _ => Value::String(raw.to_owned()),
        };
        args.insert(key.trim().to_owned(), value);
        rest = after;
    }
    Some(ToolCall {
        name: name.trim().to_owned(),
        arguments: Value::Object(args),
    })
}

/// The chat format of a model: Maple and Qwen3.5 / 3.6 use `ChatML` turns
/// (`<|im_start|>role … <|im_end|>`) and differ in the tool instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Format {
    Maple,
    Qwen,
}

/// The system turn: `system` and, with `tools`, their JSON schemas and the call format of
/// the model's chat `format`.
pub(crate) fn system_turn(system: Option<&str>, tools: &[Value], format: Format) -> String {
    if format == Format::Qwen && !tools.is_empty() {
        return qwen_system_turn(system, tools);
    }
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

/// Qwen3.5 / 3.6's system turn with tools, as its `chat_template.jinja` writes it.
fn qwen_system_turn(system: Option<&str>, tools: &[Value]) -> String {
    let mut t = String::from("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for tool in tools {
        t.push('\n');
        py_json(tool, &mut t);
    }
    t.push_str("\n</tools>");
    t.push_str(QWEN_TOOLS);
    if let Some(s) = system.map(str::trim).filter(|s| !s.is_empty()) {
        t.push_str("\n\n");
        t.push_str(s);
    }
    t.push_str("<|im_end|>\n");
    t
}

/// The call instructions after Qwen3.5 / 3.6's tool list.
const QWEN_TOOLS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO \
                          suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n\
                          value_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second \
                          parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n\
                          <IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner \
                          <function=...></function> block must be nested within <tool_call></tool_call> XML \
                          tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for \
                          your function call in natural language BEFORE the function call, but NOT after\n- If \
                          there is no function call available, answer the question like normal with your current \
                          knowledge and do not tell the user about function calls\n</IMPORTANT>";

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
    fn parses_qwen_xml_calls() {
        let p = parse(
            "\nNeed the weather.\n</think>\n\n<tool_call>\n<function=weather>\n<parameter=city>\nHong Kong\n\
             </parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=opts>\n{\"unit\": \"c\"}\n</parameter>\n\
             </function>\n</tool_call>\n<tool_call>\n<function=now>\n</function>\n</tool_call>",
        );
        assert_eq!(p.thinking, "Need the weather.");
        assert_eq!(p.text, "");
        assert!(p.malformed.is_empty(), "{:?}", p.malformed);
        assert_eq!(
            p.calls,
            vec![
                ToolCall {
                    name: "weather".into(),
                    arguments: json!({"city": "Hong Kong", "days": 3, "opts": {"unit": "c"}})
                },
                ToolCall {
                    name: "now".into(),
                    arguments: json!({})
                }
            ]
        );
        // A dropped `>` after the name: the name ends at the line end.
        let p = parse(
            "<tool_call>\n<function=get_weather\n<parameter=city>\nOslo\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(
            p.calls,
            vec![ToolCall {
                name: "get_weather".into(),
                arguments: json!({"city": "Oslo"})
            }]
        );
    }

    /// Byte-exact against the Jinja template as the oracle rendered it (`QWEN_FIXTURE`, written
    /// with the model's tokenizer): system turn with a tool, user turn, generation prompt.
    #[test]
    fn qwen_turns_match_the_rendered_template() {
        let Ok(path) = std::env::var("QWEN_FIXTURE") else {
            eprintln!("skipped: set QWEN_FIXTURE");
            return;
        };
        let fx: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let t = &fx["template"];
        let system = t["system"].as_str().unwrap();
        let head = system_turn(Some(system), std::slice::from_ref(&t["tool"]), Format::Qwen)
            + &Message::User(t["user"].as_str().unwrap()).turn()
            + "<|im_start|>assistant\n";
        assert_eq!(format!("{head}<think>\n\n</think>\n\n"), t["no_think"].as_str().unwrap());
        assert_eq!(format!("{head}<think>\n"), t["think"].as_str().unwrap());
    }

    #[test]
    fn qwen_system_turn_matches_the_template() {
        let tool = json!({"type": "function", "function": {"name": "now", "parameters": {}}});
        let t = system_turn(Some("Be brief."), std::slice::from_ref(&tool), Format::Qwen);
        assert!(t.starts_with(
            "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \
             \"function\", \"function\": {\"name\": \"now\", \"parameters\": {}}}\n</tools>\n\nIf you choose"
        ));
        assert!(
            t.ends_with("do not tell the user about function calls\n</IMPORTANT>\n\nBe brief.<|im_end|>\n"),
            "{t}"
        );
        assert!(t.contains("multiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- "));
        assert_eq!(system_turn(Some("Be brief."), &[], Format::Qwen), "<|im_start|>system\nBe brief.<|im_end|>\n");
    }

    #[test]
    fn turns_match_the_chat_template() {
        assert_eq!(system_turn(None, &[], Format::Maple), "");
        assert_eq!(system_turn(Some("Be brief."), &[], Format::Maple), "<|im_start|>system\nBe brief.<|im_end|>\n");
        let t = system_turn(Some("Be brief."), &[json!({"type": "function", "f": [1, "x"]})], Format::Maple);
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
