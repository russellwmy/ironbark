//! Sessions on a real packed model: set `IRONBARK_MODEL` to an `ironbark pack` dir (skipped otherwise).

#![allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]

use ironbark::{Error, Finish, Options, Sampling, Session, SystemTurn};

/// Greedy, so replies repeat token for token (Qwen samples by default).
fn open(options: Options) -> Option<Session> {
    let dir = std::env::var_os("IRONBARK_MODEL")?;
    let mut s = Session::open(
        dir,
        Options {
            budget: Some(2_000_000_000),
            ..options
        },
    )
    .expect("open");
    s.set_sampling(Some(Sampling::GREEDY));
    Some(s)
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
fn a_reply_dropped_while_reasoning_leaves_no_reasoning() {
    let Some(mut s) = open(Options::default()) else { return };
    let ask = "What is the capital of Italy? One word.";
    for cut in [5, 30] {
        s.reset();
        let partial: String = s.chat("Count from 1 to 50.").unwrap().take(cut).map(Result::unwrap).collect();
        assert!(!partial.contains("</think>"), "{partial}");
        let next = reply(&mut s, ask);
        assert!(next.contains("Rome"), "{next}");
        let context = s.context();
        // As if the reply had been an empty reasoning and answer, read in the same two parts.
        s.reset();
        s.start("<|im_start|>user\nCount from 1 to 50.<|im_end|>\n<|im_start|>assistant\n<think>\n")
            .unwrap();
        let rebuilt: String = s
            .start(&format!(
                "\n</think>\n\n<|im_end|>\n<|im_start|>user\n{ask}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            ))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(next, rebuilt);
        assert_eq!(s.context(), context);
    }
}

#[test]
fn a_snapshot_of_a_reply_dropped_while_reasoning_leaves_no_reasoning() {
    let Some(mut s) = open(Options::default()) else { return };
    let ask = "What is the capital of Italy? One word.";
    let partial: String = s.chat("Count from 1 to 50.").unwrap().take(5).map(Result::unwrap).collect();
    assert!(!partial.contains("</think>"), "{partial}");
    let saved = s.snapshot();
    let direct = reply(&mut s, ask);
    s.reset();
    s.restore(&saved);
    assert_eq!(reply(&mut s, ask), direct);
}

#[test]
fn context_cap_is_enforced() {
    let Some(mut s) = open(Options {
        max_context: Some(64),
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
    let a = s.bench().generate("The capital of Germany is", 16).unwrap();
    let b = s.bench().generate("The capital of Germany is", 16).unwrap();
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
    // Restored from the snapshot: the same reply after the same context.
    let again = reply(&mut s, "Name a primary colour. One word.");
    assert_eq!(first, again);
    assert_eq!(s.context(), held);
}

#[test]
fn switching_system_turns_keeps_both_read() {
    let Some(mut s) = open(Options::default()) else { return };
    let a = Some("You are terse. Today is 2026-10-15.");
    let b = Some("You sort messages. Reply with one word.");
    s.set_system(a, &[]);
    let first = reply(&mut s, "Name a primary colour. One word.");
    let read_first = s.stats().prompt_tokens;
    s.set_system(b, &[]);
    reply(&mut s, "hello");
    s.set_system(a, &[]);
    // System turn a was kept: the same reply, and only the message read this time.
    let again = reply(&mut s, "Name a primary colour. One word.");
    assert_eq!(first, again);
    assert!(s.stats().prompt_tokens < read_first);
}

#[test]
fn a_snapshot_survives_a_side_conversation() {
    let Some(mut s) = open(Options::default()) else { return };
    s.set_system(Some("You are terse."), &[]);
    reply(&mut s, "My name is Ada. Say ok.");
    let held = s.snapshot();
    assert_eq!(held.context(), s.context());
    s.set_system(Some("You sort messages. Reply with one word."), &[]);
    reply(&mut s, "hello");
    s.restore(&held);
    let name = reply(&mut s, "What is my name? One word.");
    assert!(name.contains("Ada"), "{name}");
}

#[test]
fn read_samples_once_then_restore_them() {
    let Some(mut s) = open(Options::default()) else { return };
    s.set_system(Some("Reply with the colour of the thing, one word."), &[]);
    let read = s
        .read(&[
            ironbark::Message::User("grass"),
            ironbark::Message::Assistant("green"),
            ironbark::Message::User("the sky"),
            ironbark::Message::Assistant("blue"),
        ])
        .unwrap();
    assert!(read > 0);
    let samples = s.snapshot();
    s.set_thinking(false);
    let snow = reply(&mut s, "snow");
    s.restore(&samples);
    let again = reply(&mut s, "snow");
    assert_eq!(snow, again);
    assert!(ironbark::parse(&snow).text.to_lowercase().contains("white"), "{snow}");
}

/// A file for a saved snapshot, removed first.
fn snapshot_file(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("ironbark-{}-{name}.safetensors", std::process::id()));
    let _ = std::fs::remove_file(&path);
    path
}

#[test]
fn saved_samples_load_in_a_new_session() {
    let path = snapshot_file("samples");
    let system = Some("Reply with the colour of the thing, one word.");
    let samples = [
        ironbark::Message::User("grass"),
        ironbark::Message::Assistant("green"),
        ironbark::Message::User("the sky"),
        ironbark::Message::Assistant("blue"),
    ];
    let snow = {
        let Some(mut s) = open(Options::default()) else { return };
        s.set_system(system, &[]);
        s.read(&samples).unwrap();
        s.save(&path).unwrap();
        s.set_thinking(false);
        reply(&mut s, "snow")
    };
    let mut s = open(Options::default()).unwrap();
    assert!(s.load(&path).unwrap(), "same model");
    s.set_thinking(false);
    assert_eq!(reply(&mut s, "snow"), snow);
    // Only the message was read.
    assert!(s.stats().prompt_tokens < 20, "{}", s.stats().prompt_tokens);
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn a_saved_system_turn_stays_read_for_new_conversations() {
    let path = snapshot_file("system");
    let system = Some("You are terse. Today is 2026-10-15.");
    let (first, full) = {
        let Some(mut s) = open(Options::default()) else { return };
        s.set_system(system, &[]);
        assert!(matches!(s.read_system_cached(&path).unwrap(), SystemTurn::Read(n) if n > 0));
        assert_eq!(s.read_system_cached(&path).unwrap(), SystemTurn::Restored);
        let first = reply(&mut s, "Name a primary colour. One word.");
        (first, s.context())
    };
    let mut s = open(Options::default()).unwrap();
    s.set_system(system, &[]);
    assert_eq!(s.read_system_cached(&path).unwrap(), SystemTurn::Loaded);
    // A new conversation after the load still finds the system turn read.
    s.reset();
    assert_eq!(s.read_system().unwrap(), SystemTurn::Restored);
    assert_eq!(reply(&mut s, "Name a primary colour. One word."), first);
    assert_eq!(s.context(), full);
    // Another system turn is read, and replaces the file.
    s.set_system(Some("You are verbose."), &[]);
    assert!(matches!(s.read_system_cached(&path).unwrap(), SystemTurn::Read(_)));
    std::fs::remove_file(&path).unwrap();
}

#[test]
fn a_missing_file_does_not_load() {
    let Some(mut s) = open(Options::default()) else { return };
    assert!(!s.load(snapshot_file("missing")).unwrap());
}

#[test]
fn a_broken_file_is_a_snapshot_error() {
    let Some(mut s) = open(Options::default()) else { return };
    let path = snapshot_file("broken");
    std::fs::write(&path, b"not safetensors").unwrap();
    assert!(matches!(s.load(&path), Err(Error::Snapshot(_))));
    // read_system_cached reads the system turn instead, and replaces the file.
    s.set_system(Some("You are terse."), &[]);
    assert!(matches!(s.read_system_cached(&path).unwrap(), SystemTurn::Read(_)));
    assert!(s.load(&path).unwrap());
    std::fs::remove_file(&path).unwrap();
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
    // Maple only: a Qwen pack has no embed.bin.
    if ironbark::is_qwen(&model) {
        return;
    }
    let mut local = open(Options::default()).unwrap();
    let want = local.bench().generate("The capital of Germany is", 16).unwrap();
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
    let got = s.bench().generate("The capital of Germany is", 16).unwrap();
    assert_eq!(got.tokens, want.tokens);
    // Only the rows these tokens use were fetched.
    let p = embed.progress();
    assert!(p.present > 0 && p.present < p.total / 10, "{p:?}");
    drop(s);
    std::fs::remove_dir_all(&client).unwrap();
}

#[test]
fn a_session_moves_between_threads() {
    // As through the C API: an app opens the model on one thread and may call it from another
    // each time. An array left lazy between calls is tied to its thread's GPU stream and fails
    // on the next ("There is no Stream(gpu, 0) in current thread").
    struct Moved(Session);
    // SAFETY: the session is used by one thread at a time (moved, then joined).
    unsafe impl Send for Moved {}
    fn on_thread<T: Send + 'static>(s: Session, f: impl FnOnce(&mut Session) -> T + Send + 'static) -> (Session, T) {
        let moved = Moved(s);
        let back = std::thread::spawn(move || {
            // The whole wrapper, not just its field (closures capture fields separately).
            let moved = moved;
            let Moved(mut s) = moved;
            let out = f(&mut s);
            (Moved(s), out)
        })
        .join()
        .unwrap();
        (back.0.0, back.1)
    }
    let Some(mut s) = open(Options::default()) else { return };
    s.set_thinking(false);
    // The model's own sampling: its state too must be ready for the next thread.
    s.set_sampling(None);
    for (question, answer) in [
        ("What is the capital of Japan? One word.", "Tokyo"),
        ("And of Italy? One word.", "Rome"),
    ] {
        let started;
        (s, started) = on_thread(s, move |s| s.chat(question).map(drop));
        started.unwrap();
        let mut text = String::new();
        loop {
            let piece;
            (s, piece) = on_thread(s, Session::next_piece);
            match piece.unwrap() {
                Some(p) => text += &p,
                None => break,
            }
        }
        assert!(text.contains(answer), "{text}");
    }
}

#[test]
fn without_tool_calls_the_reply_answers() {
    let Some(mut s) = open(Options::default()) else { return };
    s.set_thinking(false);
    let weather = ironbark::json!({"type": "function", "function": {"name": "get_weather",
        "description": "Current weather for a city", "parameters": {"type": "object",
        "properties": {"city": {"type": "string"}}, "required": ["city"]}}});
    s.set_system(None, &[weather]);
    let called = reply(&mut s, "What is the weather in Oslo right now?");
    assert!(called.contains("<tool_call>"), "{called}");
    s.reset();
    s.set_tool_calls(false);
    let answered = reply(&mut s, "What is the weather in Oslo right now?");
    assert!(!answered.contains("<tool_call>"), "{answered}");
    assert_eq!(s.finish(), Some(Finish::Stop));
}
