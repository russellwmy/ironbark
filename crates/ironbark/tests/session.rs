//! Sessions on a real packed model: set `IRONBARK_MODEL` to an `ironbark pack` dir (skipped otherwise).

#![allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]

use ironbark::{Error, Finish, Options, Session};

fn open(options: Options) -> Option<Session> {
    let dir = std::env::var_os("IRONBARK_MODEL")?;
    Some(
        Session::open(
            dir,
            Options {
                budget: Some(2_000_000_000),
                ..options
            },
        )
        .expect("open"),
    )
}

fn reply(s: &mut Session, message: &str) -> String {
    s.chat(message).unwrap().collect::<ironbark::Result<String>>().unwrap()
}

#[test]
fn chat_remembers_the_conversation() {
    let Some(mut s) = open(Options::default()) else { return };
    let first = reply(&mut s, "What is the capital of Germany? One word.");
    assert!(first.contains("Berlin"), "{first}");
    assert_eq!(s.finish(), Some(Finish::Stop));
    let before = s.context();
    let second = reply(&mut s, "And of France? One word.");
    assert!(second.contains("Paris"), "{second}");
    assert!(s.context() > before);
}

#[test]
fn a_dropped_reply_leaves_the_conversation_usable() {
    let Some(mut s) = open(Options::default()) else { return };
    let partial: String = s.chat("Count from 1 to 50.").unwrap().take(5).map(Result::unwrap).collect();
    assert_ne!(partial, "");
    let next = reply(&mut s, "What is the capital of Italy? One word.");
    assert!(next.contains("Rome"), "{next}");
}

#[test]
fn context_cap_is_enforced() {
    let Some(mut s) = open(Options {
        max_context: 64,
        ..Options::default()
    }) else {
        return;
    };
    let long = "word ".repeat(200);
    assert!(matches!(s.chat(&long), Err(Error::ContextFull { max: 64, .. })));
    let mut r = s.chat("Hi").unwrap();
    while r.next().is_some() {}
    assert_eq!(r.finish().map(|f| f == Finish::Stop || f == Finish::Context), Some(true));
    assert!(s.context() <= 64);
}

#[test]
fn generate_is_greedy_and_repeatable() {
    let Some(mut s) = open(Options::default()) else { return };
    let a = s.generate("The capital of Germany is", 16).unwrap();
    let b = s.generate("The capital of Germany is", 16).unwrap();
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(s.context(), 0);
}

#[test]
fn a_new_conversation_restores_the_system_turn() {
    let Some(mut s) = open(Options::default()) else { return };
    s.set_system(Some("You are terse. Today is 2026-10-15."), &[]);
    let first = reply(&mut s, "Name a primary colour. One word.");
    let held = s.context();
    s.reset();
    // Restored from the snapshot: the same reply, token for token, after the same context.
    let again = reply(&mut s, "Name a primary colour. One word.");
    assert_eq!(first, again);
    assert_eq!(s.context(), held);
}

#[test]
fn without_thinking_a_reply_starts_after_its_reasoning() {
    let Some(mut s) = open(Options::default()) else { return };
    s.set_thinking(false);
    let r = reply(&mut s, "What is the capital of Germany? One word.");
    assert!(r.starts_with("</think>"), "{r}");
    let p = ironbark::parse(&r);
    assert_eq!(p.thinking, "");
    assert!(p.text.contains("Berlin"), "{r}");
}

#[test]
fn embed_fetched_on_demand_decodes_the_same() {
    let Some(model) = std::env::var_os("IRONBARK_MODEL").map(std::path::PathBuf::from) else {
        return;
    };
    let mut local = open(Options::default()).unwrap();
    let want = local.generate("The capital of Germany is", 16).unwrap();
    drop(local);

    // A client dir with everything but embed.bin, which fills from the model's copy.
    let client = std::env::temp_dir().join(format!("ironbark-embed-client-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&client);
    std::fs::create_dir_all(&client).unwrap();
    for e in std::fs::read_dir(&model).unwrap() {
        let name = e.unwrap().file_name();
        if name != "embed.bin" {
            std::os::unix::fs::symlink(model.join(&name), client.join(&name)).unwrap();
        }
    }
    let remote = ironbark_fetch::FileFetch::open(&model.join("embed.bin"), std::time::Duration::ZERO).unwrap();
    let embed = ironbark_fetch::Sparse::open_embed(&client, std::sync::Arc::new(remote)).unwrap();
    let mut s = Session::open(
        &client,
        Options {
            budget: Some(2_000_000_000),
            embed: Some(embed.clone()),
            ..Options::default()
        },
    )
    .unwrap();
    let got = s.generate("The capital of Germany is", 16).unwrap();
    assert_eq!(got.tokens, want.tokens);
    // Only the rows these tokens use were fetched.
    let p = embed.progress();
    assert!(p.present > 0 && p.present < p.total / 10, "{p:?}");
    drop(s);
    std::fs::remove_dir_all(&client).unwrap();
}
