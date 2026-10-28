//! The C API on a real packed model: set `IRONBARK_MODEL` to an `ironbark pack` dir (skipped
//! otherwise).

#![allow(clippy::unwrap_used, clippy::expect_used, unsafe_code, reason = "tests of the C API")]

use std::ffi::{CStr, CString, c_char};
use std::ptr::null;

use ironbark_ffi::{
    Session, ironbark_clear_image, ironbark_close, ironbark_free_string, ironbark_load, ironbark_next, ironbark_open,
    ironbark_read_system_cached, ironbark_save, ironbark_send, ironbark_set_dynamic_k, ironbark_set_sampling,
    ironbark_set_system,
};
use serde_json::{Value, json};

/// A string a call returned, freed; `None` for null.
fn take(p: *mut c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    // SAFETY: a NUL-terminated string from this library.
    let text = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_owned();
    // SAFETY: freed once, by the library that made it.
    unsafe { ironbark_free_string(p) };
    Some(text)
}

/// The JSON a call returned.
fn json_of(p: *mut c_char) -> Value {
    serde_json::from_str(&take(p).expect("a JSON reply")).unwrap()
}

fn c(s: &str) -> CString {
    CString::new(s).unwrap()
}

/// A temporary snapshot file, removed first.
fn file(name: &str) -> CString {
    let path = std::env::temp_dir().join(format!("ironbark-ffi-{}-{name}.safetensors", std::process::id()));
    let _ = std::fs::remove_file(&path);
    c(path.to_str().unwrap())
}

/// An open session, used by one thread; every call passes NUL-terminated strings that
/// outlive it.
struct S(*mut Session);

impl S {
    fn open() -> Option<Self> {
        let dir = c(&std::env::var("IRONBARK_MODEL").ok()?);
        // SAFETY: a NUL-terminated dir; no error out-pointer.
        let s = unsafe { ironbark_open(dir.as_ptr(), 2_000_000_000, 0, std::ptr::null_mut()) };
        assert!(!s.is_null(), "open");
        Some(Self(s))
    }

    fn sampling(&self, json: Option<&str>) -> Value {
        let json = json.map(c);
        // SAFETY: see `S`.
        json_of(unsafe { ironbark_set_sampling(self.0, json.as_ref().map_or(null(), |j| j.as_ptr())) })
    }

    fn system(&self, text: &str) {
        // SAFETY: see `S`.
        json_of(unsafe { ironbark_set_system(self.0, c(text).as_ptr(), null()) });
    }

    fn read_system_cached(&self, path: &CString) -> Value {
        // SAFETY: see `S`.
        json_of(unsafe { ironbark_read_system_cached(self.0, path.as_ptr()) })
    }

    fn save(&self, path: &CString) -> Value {
        // SAFETY: see `S`.
        json_of(unsafe { ironbark_save(self.0, path.as_ptr()) })
    }

    fn load(&self, path: &CString) -> Value {
        // SAFETY: see `S`.
        json_of(unsafe { ironbark_load(self.0, path.as_ptr()) })
    }

    /// Send a user message and stream the reply to its end.
    fn chat(&self, message: &str) -> String {
        let messages = c(&json!([{ "role": "user", "content": message }]).to_string());
        // SAFETY: see `S`.
        let stats = json_of(unsafe { ironbark_send(self.0, messages.as_ptr(), false) });
        assert!(stats.get("error").is_none(), "{stats}");
        // SAFETY: see `S`; the turn was started above.
        std::iter::from_fn(|| take(unsafe { ironbark_next(self.0) })).collect()
    }
}

impl Drop for S {
    fn drop(&mut self) {
        // SAFETY: the session from `open`, closed once.
        unsafe { ironbark_close(self.0) };
    }
}

#[test]
fn saved_conversations_and_settings_through_the_c_api() {
    let Some(s) = S::open() else { return };
    let (system, saved, broken) = (file("system"), file("chat"), file("broken"));
    // Sampling: a field set, the others the model's own; null: all the model's own.
    let own = s.sampling(None);
    let greedy = s.sampling(Some(r#"{"temperature": 0}"#));
    assert_eq!(greedy["temperature"], 0.0);
    assert_eq!(greedy["top_k"], own["top_k"]);

    s.system("You are terse.");
    let first = s.read_system_cached(&system);
    assert_eq!(first["system"], "read");
    assert!(first["read"].as_u64().unwrap() > 0);
    assert_eq!(s.read_system_cached(&system)["system"], "restored");

    // SAFETY: see `S`.
    unsafe { ironbark_set_dynamic_k(s.0, 0.0, 3) };
    // SAFETY: see `S`.
    unsafe { ironbark_clear_image(s.0) };
    assert_ne!(s.chat("Name a primary colour. One word."), "");

    assert_eq!(s.save(&saved), json!({}));
    assert_eq!(s.load(&file("missing"))["loaded"], false);
    assert_eq!(s.load(&saved)["loaded"], true);
    std::fs::write(broken.to_str().unwrap(), b"not safetensors").unwrap();
    let err = s.load(&broken);
    assert!(err["error"].as_str().unwrap().starts_with("snapshot: "), "{err}");
    drop(s);

    // A new session loads the system turn from the file.
    let s = S::open().unwrap();
    s.system("You are terse.");
    assert_eq!(s.read_system_cached(&system)["system"], "loaded");
    for f in [system, saved, broken] {
        std::fs::remove_file(f.to_str().unwrap()).unwrap();
    }
}
