//! A tool-call round trip through `Session`: ask for the weather with a `get_weather` tool,
//! answer the call, print the reply. Works with any model the session supports.
//!
//! usage: `cargo run --release -p ironbark --example tool_chat -- MODEL_DIR`
#![allow(clippy::print_stdout, reason = "an example")]

use ironbark::{Options, Session, json, parse};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1).ok_or("usage: tool_chat MODEL_DIR")?;
    let mut s = Session::open(dir, Options::default())?;
    s.set_thinking(false);
    let weather = json!({"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
        "required": ["city"]}}});
    s.set_system(Some("You are terse."), &[weather]);
    let collect =
        |reply: ironbark::Reply<'_>| -> Result<String, ironbark::Error> { reply.collect::<Result<String, _>>() };
    let first = collect(s.chat("Is it warmer in Oslo or in Rome right now?")?)?;
    let p = parse(&first);
    println!("reply 1: {first:?}\ncalls: {:?}", p.calls);
    let results: Vec<String> = p
        .calls
        .iter()
        .map(|c| {
            format!(
                "{{\"city\": {}, \"temp_c\": {}}}",
                c.arguments["city"],
                if c.arguments["city"] == "Oslo" { 4 } else { 18 }
            )
        })
        .collect();
    let refs: Vec<&str> = results.iter().map(String::as_str).collect();
    let second = collect(s.tool_results(&refs)?)?;
    println!("reply 2: {:?}", parse(&second).text);
    let third = collect(s.chat("Thanks. And which of the two is the capital of Italy? One word.")?)?;
    println!("reply 3: {:?}", parse(&third).text);
    println!("context {} tokens, {:.1} tok/s", s.context(), s.stats().decode_tps);
    Ok(())
}
