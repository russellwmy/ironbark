//! A Qwen pack answers about an image through `Session` (model-gated: `QWEN_PACK`, a pack with
//! its vision encoder; `QWEN_VISION_FIXTURE`, the dir of the vision oracle's fixture, whose
//! `rgb` is a 400×300 image reading "IRONBARK 42").

use std::collections::HashMap;
use std::path::Path;

use ironbark::{Message, Options, Session};
use ironbark_mlx::Array;

#[test]
fn qwen_reads_the_text_in_an_image() {
    let (Ok(pack), Ok(fx)) = (std::env::var("QWEN_PACK"), std::env::var("QWEN_VISION_FIXTURE")) else {
        eprintln!("skipped: set QWEN_PACK and QWEN_VISION_FIXTURE");
        return;
    };
    let f: HashMap<String, Array> =
        Array::load_safetensors(Path::new(&fx).join("vision.safetensors")).unwrap().into_iter().collect();
    let rgb = &f["rgb"];
    rgb.eval().unwrap();
    let (h, w) = (rgb.shape()[0] as usize, rgb.shape()[1] as usize);

    let mut s = Session::open(&pack, Options::default()).unwrap();
    s.set_thinking(false);
    let info = s.set_image(rgb.as_slice::<u8>(), w, h, ironbark::IMAGE_PIXELS).unwrap();
    assert_eq!((info.width, info.height, info.tokens), (384, 288, 108));
    let reply: String = s
        .send(&[Message::User("What text is in this image? Reply with the text only.")])
        .unwrap()
        .map(|p| p.unwrap())
        .collect();
    eprintln!("reply: {reply:?}  ({:.1} tok/s prefill)", s.stats().prefill_tps);
    assert!(reply.contains("IRONBARK 42"), "{reply}");

    // The conversation goes on in text, positions after the image intact.
    let more: String = s
        .send(&[Message::User("And what number is in it? Digits only.")])
        .unwrap()
        .map(|p| p.unwrap())
        .collect();
    eprintln!("follow-up: {more:?}");
    assert!(more.contains("42"), "{more}");
}
