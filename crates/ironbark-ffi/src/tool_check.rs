//! Does the model call tools when it should, with sensible arguments? A fixed set of cases
//! (web search, weather, a calculator, and one that needs no tool), each from a fresh
//! conversation; when the model calls a tool it gets a canned result, for up to [`ROUNDS`]
//! rounds of calls.

use std::time::Instant;

use ironbark::{Session, Value, json, parse};

/// The tools offered in every case.
fn tools() -> Vec<Value> {
    let tool = |name: &str, description: &str, arg: &str, arg_description: &str| {
        json!({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": { arg: { "type": "string", "description": arg_description } },
                    "required": [arg],
                },
            },
        })
    };
    vec![
        tool(
            "web_search",
            "Search the web for current information. Returns the top results with title, url and snippet.",
            "query",
            "The search query",
        ),
        tool("get_weather", "Current weather in a city.", "city", "City name, e.g. Paris"),
        tool("calculator", "Evaluate an arithmetic expression exactly.", "expression", "e.g. 2 * (3 + 4)"),
    ]
}

/// Most tool-call rounds per case.
const ROUNDS: usize = 3;

/// The model assumes a date from its training; the cases are about later events.
const SYSTEM: &str = "You are a helpful assistant. Today is 2026-10-15.";

/// A case: the user message, the tool it should call (`None`: no tool), and the canned result.
struct Case {
    message: &'static str,
    expect: Option<&'static str>,
    result: &'static str,
}

const CASES: [Case; 4] = [
    Case {
        message: "What's the weather in Adelaide right now?",
        expect: Some("get_weather"),
        result: r#"{"city": "Adelaide", "temp_c": 22, "condition": "sunny", "humidity": 0.55}"#,
    },
    Case {
        message: "Who won the Formula 1 Japanese Grand Prix in 2026?",
        expect: Some("web_search"),
        result: r#"[{"title": "2026 Japanese Grand Prix - race report", "url": "https://news.example.org/f1/japan-2026", "snippet": "Andrea Rossi won the 2026 Japanese Grand Prix at Suzuka ahead of Lena Park and Tom Hale."}]"#,
    },
    Case {
        message: "What is 48213 * 9177? Use the calculator.",
        expect: Some("calculator"),
        result: "442450701",
    },
    Case {
        message: "What is the capital of Germany?",
        expect: None,
        result: "",
    },
];

/// Stream a reply of at most `max` tokens; returns its text and seconds.
fn collect(reply: ironbark::Reply<'_>, max: usize) -> ironbark::Result<(String, f64)> {
    let t = Instant::now();
    let mut text = String::new();
    for piece in reply.take(max) {
        text.push_str(&piece?);
    }
    Ok((text, t.elapsed().as_secs_f64()))
}

fn reply_json(s: &Session, text: &str, secs: f64) -> Value {
    let p = parse(text);
    let st = s.stats();
    json!({
        "finished": s.finish().is_some(),
        "prompt_tokens": st.prompt_tokens,
        "prefill_tps": st.prefill_tps,
        "tokens": st.tokens,
        "decode_tps": st.decode_tps,
        "seconds": secs,
        "thinking_chars": p.thinking.chars().count(),
        "calls": p.calls.iter().map(|c| json!({ "name": c.name, "arguments": c.arguments })).collect::<Vec<_>>(),
        "malformed": p.malformed,
        "text": p.text,
        "raw": text,
    })
}

/// Run every case with replies of at most `max_tokens`; JSON with a result per case.
///
/// # Errors
///
/// Fails if the model fails.
pub fn run(s: &mut Session, max_tokens: usize) -> ironbark::Result<Value> {
    s.set_system(Some(SYSTEM), &tools());
    let mut cases = Vec::new();
    for c in &CASES {
        s.reset();
        let (text, secs) = collect(s.chat(c.message)?, max_tokens)?;
        let first = reply_json(s, &text, secs);
        let calls = parse(&text).calls;
        let called = calls.first().map(|c| c.name.as_str());
        let mut case = json!({
            "message": c.message,
            "expect": c.expect,
            "called": called,
            "pass": called == c.expect && parse(&text).malformed.is_empty(),
            "reply": first,
        });
        // Answer each round of calls with the canned result until the model answers.
        let mut rounds = Vec::new();
        let mut calls = calls.len();
        while c.expect.is_some() && calls > 0 && rounds.len() < ROUNDS {
            let results = vec![c.result; calls];
            let (text, secs) = collect(s.tool_results(&results)?, max_tokens)?;
            calls = parse(&text).calls.len();
            rounds.push(reply_json(s, &text, secs));
        }
        case["rounds"] = rounds.into();
        cases.push(case);
    }
    s.set_system(None, &[]);
    let passed = cases.iter().filter(|c| c["pass"] == true).count();
    Ok(json!({ "passed": passed, "cases": cases.len(), "max_tokens": max_tokens, "results": cases }))
}
