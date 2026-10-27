//! Answer prompts through `Session`, each in a fresh conversation, with optional tools:
//! `PROMPTS.json` is `[{"prompt": "...", "tools": [...]?}, ...]`; `OUT.json` gets
//! `[{"text", "calls": [{"name", "arguments"}], "raw", "tokens", "decode_tps"}, ...]`.
//! Thinking off, replies of at most `MAX_TOKENS`.
//!
//! usage: `cargo run --release -p ironbark --example eval_prompts -- MODEL_DIR PROMPTS.json OUT.json [MAX_TOKENS]`
#![allow(clippy::print_stdout, reason = "an example")]

use std::io::{Seek, Write};

use ironbark::{Options, Session, Value, json, parse};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let ([_, dir, prompts, out] | [_, dir, prompts, out, _]) = &args[..] else {
        return Err("usage: eval_prompts MODEL_DIR PROMPTS.json OUT.json [MAX_TOKENS]".into());
    };
    let max_tokens: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(1280);
    let budget = std::env::var("IRONBARK_BUDGET_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|gb| (gb * 1e9) as u64);
    let items: Vec<Value> = serde_json::from_slice(&std::fs::read(prompts)?)?;
    let mut s = Session::open(
        dir,
        Options {
            budget,
            ..Options::default()
        },
    )?;
    s.set_thinking(false);
    // IRONBARK_GREEDY=1: the most likely token every step (Qwen samples by default).
    if std::env::var("IRONBARK_GREEDY").is_ok_and(|v| v == "1") {
        s.set_sampling(Some(ironbark::Sampling::GREEDY));
    }
    let mut replies = Vec::with_capacity(items.len());
    let mut file = std::fs::File::create(out)?;
    for (i, item) in items.iter().enumerate() {
        let tools: Vec<Value> = item["tools"].as_array().cloned().unwrap_or_default();
        s.set_system(None, &tools);
        let prompt = item["prompt"].as_str().unwrap_or_default();
        s.chat(prompt)?;
        let mut raw = String::new();
        for _ in 0..max_tokens {
            match s.next_piece()? {
                Some(piece) => raw += &piece,
                None => break,
            }
        }
        let st = s.stats();
        let p = parse(&raw);
        let calls: Vec<Value> = p.calls.iter().map(|c| json!({"name": c.name, "arguments": c.arguments})).collect();
        replies.push(
            json!({"text": p.text, "calls": calls, "raw": raw, "tokens": st.tokens, "decode_tps": st.decode_tps}),
        );
        if i % 50 == 0 {
            println!("{i}/{} done", items.len());
            // Partial results survive an interruption.
            file.set_len(0)?;
            file.rewind()?;
            write!(file, "{}", Value::Array(replies.clone()))?;
        }
    }
    file.set_len(0)?;
    file.rewind()?;
    write!(file, "{}", Value::Array(replies))?;
    Ok(())
}
