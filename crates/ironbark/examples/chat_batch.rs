//! Answer each prompt of a JSON list in a fresh conversation (thinking off, at most 300
//! tokens); write the replies with decode speed and peak footprint as JSON.
//!
//! usage: `cargo run --release -p ironbark --example chat_batch -- MODEL_DIR PROMPTS.json OUT.json`

use ironbark::{Options, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, dir, prompts, out] = &args[..] else {
        return Err("usage: chat_batch MODEL_DIR PROMPTS.json OUT.json".into());
    };
    let prompts: Vec<String> = serde_json::from_slice(&std::fs::read(prompts)?)?;
    // IRONBARK_BUDGET_GB: footprint budget (default: from the device).
    let budget = std::env::var("IRONBARK_BUDGET_GB")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(|gb| (gb * 1e9) as u64);
    let mut s = Session::open(
        dir,
        Options {
            budget,
            ..Options::default()
        },
    )?;
    s.set_thinking(false);
    let mut replies = Vec::new();
    for p in &prompts {
        s.set_system(None, &[]);
        s.chat(p)?;
        let mut text = String::new();
        for _ in 0..300 {
            match s.next_piece()? {
                Some(piece) => text += &piece,
                None => break,
            }
        }
        let st = s.stats();
        replies.push(serde_json::json!({
            "text": text,
            "tokens": st.tokens,
            "decode_tps": st.decode_tps,
            "peak_footprint": st.footprint.map(|f| f.peak),
        }));
    }
    std::fs::write(out, serde_json::Value::Array(replies).to_string())?;
    Ok(())
}
