//! An image and a question through `Session` (Qwen3.5 / 3.6), with thinking off and
//! on: prints each reply's reasoning and answer.
//!
//! usage: `cargo run --release -p ironbark --example image_chat -- MODEL_DIR IMAGE ["QUESTION"]`
#![allow(clippy::print_stdout, reason = "an example")]

use ironbark::{IMAGE_PIXELS, Options, Session, parse};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, dir, image, rest @ ..] = args.as_slice() else {
        return Err("usage: image_chat MODEL_DIR IMAGE [QUESTION]".into());
    };
    let question = rest.first().map_or("What is in this photo? One sentence.", String::as_str);
    let rgb = image::open(image)?.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut s = Session::open(dir, Options::default())?;
    for think in [false, true] {
        s.set_thinking(think);
        s.set_system(None, &[]);
        let info = s.set_image(rgb.as_raw(), w, h, IMAGE_PIXELS)?;
        let reply: String = s.chat(question)?.collect::<Result<String, _>>()?;
        let p = parse(&reply);
        println!(
            "thinking {}: image {}×{} → {} tokens in {:.2}s; prompt {} tokens\n  reasoning: {:?}\n  answer: {:?}\n",
            if think { "on" } else { "off" },
            info.width,
            info.height,
            info.tokens,
            info.encode_s,
            s.stats().prompt_tokens,
            p.thinking.chars().take(200).collect::<String>(),
            p.text
        );
    }
    Ok(())
}
