//! C API for running Maple from an iOS or macOS app (`include/ironbark.h`).
//!
//! A session is a model streamed from a packed dir (`ironbark pack`). Calls that report something
//! return a JSON string the caller frees with [`ironbark_free_string`]; on failure the JSON is
//! `{"error": "..."}`.
//!
//! ```text
//!   Swift app ──▶ ironbark_open(dir, budget, slots) ──▶ ironbark_start(text, reset) ──▶ ironbark_next … ──▶ ironbark_stats
//!             ──▶ ironbark_set_system(system, tools) ──▶ ironbark_send(messages, reset) ──▶ ironbark_next …
//!             ──▶ ironbark_parse(reply) ──▶ JSON reasoning, answer, tool calls
//!             ──▶ ironbark_generate(prompt, n) ──▶ JSON timings, stats, text
//!             ──▶ ironbark_probe(dir) ──▶ JSON GPU round trip, storage reads, memory
//! ```

use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use ironbark::{Chunks, ExpertStats, Hardware, Options, Records};
use ironbark_engine::bench;
use ironbark_fetch::{Fetch, Sparse};
use serde_json::{Value, json};

pub mod tool_check;

/// An open model (an [`ironbark::Session`]).
#[derive(Debug)]
pub struct Session(ironbark::Session);

type Result<T> = std::result::Result<T, String>;

fn open(
    dir: &Path,
    budget: u64,
    slots: usize,
    records: Option<Arc<dyn Records>>,
    embed: Option<Arc<dyn Chunks>>,
) -> Result<Session> {
    let options = Options {
        budget: (budget > 0).then_some(budget),
        slots: (slots > 0).then_some(slots),
        records,
        embed,
        ..Options::default()
    };
    ironbark::Session::open(dir, options).map(Session).map_err(|e| e.to_string())
}

fn plan_json(s: &ironbark::Session) -> Value {
    let (p, hw) = (s.plan(), s.hardware());
    json!({
        "ram": hw.ram,
        "app_limit": hw.app_limit,
        "budget": p.budget,
        "slots": p.slots,
        "buffer_cache": p.buffer_cache,
        "expected_peak": p.peak,
    })
}

fn experts_json(x: &ExpertStats) -> Value {
    json!({
        "per_layer": x.per_layer,
        "experts": x.experts,
        "cache_bytes": x.cache_bytes,
        "hit": x.hit,
        "loads_per_tok": x.loads,
        "read_bytes_per_tok": x.read,
        "wait_ms_per_tok": x.wait_ms,
        "prefetch_issued_per_tok": x.prefetch_issued,
        "prefetch_used": x.prefetch_used,
        "late_per_tok": x.late,
    })
}

fn generate(s: &mut ironbark::Session, prompt: &str, n: usize) -> Result<Value> {
    let g = s.generate(prompt, n).map_err(|e| e.to_string())?;
    let text = s.decode(&g.tokens).map_err(|e| e.to_string())?;
    let experts = ExpertStats::of(s.model(), g.tokens.len()).map(|x| experts_json(&x));
    Ok(json!({
        "prompt_tokens": g.prompt,
        "prefill_s": g.prefill_s,
        "prefill_tps": g.prompt as f64 / g.prefill_s,
        "tokens": g.tokens.len(),
        "decode_s": g.decode_s,
        "decode_tps": g.tokens.len() as f64 / g.decode_s,
        "text": text,
        "experts": experts,
        "mlx_trace": { "active": g.active, "prefill_peak": g.prefill_peak },
        "footprint_peak": g.footprint_peak,
        "timing_ms_per_tok": {
            "step": g.timing.total_ns as f64 / 1e6 / n.max(1) as f64,
            "sync": g.timing.sync_ns as f64 / 1e6 / n.max(1) as f64,
            "read": g.read_ns as f64 / 1e6 / n.max(1) as f64,
        },
        "plan": plan_json(s),
        "memory": memory(),
    }))
}

fn stats(s: &ironbark::Session) -> Value {
    let st = s.stats();
    json!({
        "prompt_tokens": st.prompt_tokens,
        "prefill_tps": st.prefill_tps,
        "tokens": st.tokens,
        "decode_tps": st.decode_tps,
        "context": st.context,
        "finish": s.finish().map(|f| format!("{f:?}").to_lowercase()),
        "experts": st.experts.map(|x| experts_json(&x)),
        "memory": memory(),
        "plan": plan_json(s),
    })
}

fn probe(dir: Option<&Path>) -> Result<Value> {
    let round_trip_us = bench::round_trip_us(200).map_err(|e| e.to_string())?;
    let read = match dir {
        Some(d) => {
            let r = bench::read_probe(d, 64, 8).map_err(|e| e.to_string())?;
            json!({
                "record_bytes": r.record_bytes,
                "latency_us": r.latency_us,
                "serial_bytes_per_s": r.serial,
                "parallel_bytes_per_s": r.parallel,
                "threads": r.threads,
            })
        }
        None => Value::Null,
    };
    Ok(json!({ "round_trip_us": round_trip_us, "read": read, "memory": memory() }))
}

/// MLX's GPU memory, the process footprint (what iOS limits), and on iOS the app limit.
fn memory() -> Value {
    let fp = ironbark_engine::device::footprint();
    json!({
        "mlx_active": ironbark_mlx::memory::active_memory().ok(),
        "mlx_peak": ironbark_mlx::memory::peak_memory().ok(),
        "mlx_cache": ironbark_mlx::memory::cache_memory().ok(),
        "footprint": fp.map(|f| f.now),
        "footprint_peak": fp.map(|f| f.peak),
        "app_limit": Hardware::detect().app_limit,
    })
}

/// JSON as a C string the caller frees with [`ironbark_free_string`].
fn to_c(v: &Value) -> *mut c_char {
    // serde_json escapes NUL in strings, so the text has no interior NUL.
    CString::new(v.to_string()).map_or(std::ptr::null_mut(), CString::into_raw)
}

/// Run `f`, turning errors and panics into `{"error": ...}`.
fn report(f: impl FnOnce() -> Result<Value>) -> *mut c_char {
    let v = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => json!({ "error": e }),
        Err(_) => json!({ "error": "panic" }),
    };
    to_c(&v)
}

/// A UTF-8 path or string from C.
///
/// # Safety
///
/// `p` is null or a NUL-terminated string valid for the call.
unsafe fn str_arg<'a>(p: *const c_char) -> Result<&'a str> {
    if p.is_null() {
        return Err("null string".into());
    }
    // SAFETY: the caller guarantees a NUL-terminated string.
    let c = unsafe { CStr::from_ptr(p) };
    c.to_str().map_err(|e| e.to_string())
}

/// Like [`str_arg`], with null as `None`.
///
/// # Safety
///
/// `p` is null or a NUL-terminated string valid for the call.
unsafe fn opt_str_arg<'a>(p: *const c_char) -> Result<Option<&'a str>> {
    if p.is_null() {
        Ok(None)
    } else {
        // SAFETY: forwarded from this function's contract.
        unsafe { str_arg(p) }.map(Some)
    }
}

/// Open the packed model in `dir` sized to a footprint of `budget` bytes (0: from the device,
/// 3/8 of RAM and at most half the iOS app limit). `slots` overrides the expert cache
/// slots per layer the budget gives (0: from the budget; 256 keeps every expert loaded).
/// Returns null on failure and, when `error` is not null, stores a message there (free it
/// with [`ironbark_free_string`]).
///
/// # Safety
///
/// `dir` is a NUL-terminated string; `error` is null or valid for one pointer write.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_open(
    dir: *const c_char,
    budget: u64,
    slots: u32,
    error: *mut *mut c_char,
) -> *mut Session {
    // SAFETY: forwarded from this function's contract.
    let dir = unsafe { str_arg(dir) };
    let r = dir.and_then(|d| {
        catch_unwind(|| open(Path::new(d), budget, slots as usize, None, None)).unwrap_or_else(|_| Err("panic".into()))
    });
    match r {
        Ok(s) => Box::into_raw(Box::new(s)),
        Err(e) => {
            if !error.is_null() {
                // SAFETY: the caller guarantees `error` is valid for a write.
                unsafe { *error = to_c(&Value::String(e)) };
            }
            std::ptr::null_mut()
        }
    }
}

/// Fills `buf` with `len` bytes of the remote file (`experts.bin`, or `embed.bin` for an embed
/// fetcher) from `offset`; returns 0 on success. Called from session threads and fetcher
/// threads, possibly several at once.
pub type IronbarkFetch = unsafe extern "C" fn(ctx: *mut c_void, offset: u64, buf: *mut u8, len: usize) -> i32;

/// An app's fetch callback as a [`Fetch`].
struct CFetch {
    f: IronbarkFetch,
    ctx: *mut c_void,
}

// SAFETY: `ironbark_fetcher_open` requires the callback and `ctx` to be usable from any thread.
unsafe impl Send for CFetch {}
// SAFETY: as above, including concurrent calls.
unsafe impl Sync for CFetch {}

impl Fetch for CFetch {
    fn fetch(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        // SAFETY: `buf` is valid for `buf.len()` bytes of writes for the call; the callback's
        // contract is that of `ironbark_fetcher_open`.
        match unsafe { (self.f)(self.ctx, offset, buf.as_mut_ptr(), buf.len()) } {
            0 => Ok(()),
            rc => Err(std::io::Error::other(format!("fetch callback returned {rc}"))),
        }
    }
}

/// Builds a local `experts.bin` or `embed.bin` from a remote copy (an
/// [`ironbark_fetch::Sparse`]), apart from any session.
#[derive(Debug)]
pub struct Fetcher {
    sparse: Arc<Sparse>,
    /// Fills `embed.bin` (from [`ironbark_embed_fetcher_open`]), else `experts.bin`.
    embed: bool,
}

/// Store a message in `error` if it is not null.
///
/// # Safety
///
/// `error` is null or valid for one pointer write.
unsafe fn set_error(error: *mut *mut c_char, e: String) {
    if !error.is_null() {
        // SAFETY: the caller guarantees `error` is valid for a write.
        unsafe { *error = to_c(&Value::String(e)) };
    }
}

/// Open the fetcher of the packed dir `dir`, whose `experts.bin` may be missing or partial
/// (`experts.json` and `manifest.json` must be there): records come through `fetch`, are
/// checked against `manifest.json` and kept on disk. Nothing is fetched until
/// [`ironbark_fetcher_start`] or a session from [`ironbark_open_fetcher`] needs a record. Returns null
/// on failure (message in `error` as for [`ironbark_open`]).
///
/// # Safety
///
/// `dir` is a NUL-terminated string; `error` is null or valid for one pointer write; `fetch`
/// and `ctx` stay valid until [`ironbark_fetcher_close`] and every session using the fetcher are
/// closed, and may be called from any thread, concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_fetcher_open(
    dir: *const c_char,
    fetch: IronbarkFetch,
    ctx: *mut c_void,
    error: *mut *mut c_char,
) -> *mut Fetcher {
    // SAFETY: forwarded from this function's contract.
    unsafe { fetcher_open(dir, fetch, ctx, error, false) }
}

/// Like [`ironbark_fetcher_open`] for the `embed.bin` of `dir` (`manifest.json` must be there):
/// `fetch` reads the remote `embed.bin`, in chunks the manifest checks. A session from
/// [`ironbark_open_fetchers`] fetches the rows each token needs; start background threads (e.g.
/// once the experts are in: they compete for the network) to complete it, after which decode
/// no longer waits for each token on the CPU.
///
/// # Safety
///
/// As [`ironbark_fetcher_open`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_embed_fetcher_open(
    dir: *const c_char,
    fetch: IronbarkFetch,
    ctx: *mut c_void,
    error: *mut *mut c_char,
) -> *mut Fetcher {
    // SAFETY: forwarded from this function's contract.
    unsafe { fetcher_open(dir, fetch, ctx, error, true) }
}

/// [`ironbark_fetcher_open`] (`embed` false) or [`ironbark_embed_fetcher_open`].
///
/// # Safety
///
/// As [`ironbark_fetcher_open`].
unsafe fn fetcher_open(
    dir: *const c_char,
    fetch: IronbarkFetch,
    ctx: *mut c_void,
    error: *mut *mut c_char,
    embed: bool,
) -> *mut Fetcher {
    // SAFETY: forwarded from the caller's contract.
    let dir = unsafe { str_arg(dir) };
    let fetch: Arc<dyn Fetch> = Arc::new(CFetch { f: fetch, ctx });
    let r = dir.and_then(|d| {
        catch_unwind(AssertUnwindSafe(|| {
            let d = Path::new(d);
            if embed {
                Sparse::open_embed(d, fetch)
            } else {
                Sparse::open(d, fetch)
            }
            .map_err(|e| format!("{e:#}"))
        }))
        .unwrap_or_else(|_| Err("panic".into()))
    });
    match r {
        Ok(sparse) => Box::into_raw(Box::new(Fetcher { sparse, embed })),
        Err(e) => {
            // SAFETY: forwarded from this function's contract.
            unsafe { set_error(error, e) };
            std::ptr::null_mut()
        }
    }
}

/// Start `threads` background threads fetching every missing record (they hold off while a
/// session is waiting on a fetch). Restarts them if running.
///
/// # Safety
///
/// `f` comes from [`ironbark_fetcher_open`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_fetcher_start(f: *const Fetcher, threads: u32) {
    // SAFETY: the caller guarantees `f` is a live fetcher.
    if let Some(f) = unsafe { f.as_ref() } {
        f.sparse.start(threads as usize, &[]);
    }
}

/// Stop the background threads and wait for them (a session still fetches what it needs).
///
/// # Safety
///
/// `f` comes from [`ironbark_fetcher_open`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_fetcher_stop(f: *const Fetcher) {
    // SAFETY: the caller guarantees `f` is a live fetcher.
    if let Some(f) = unsafe { f.as_ref() } {
        f.sparse.stop();
    }
}

/// Progress (JSON): `present` and `total` records, `complete`, `fetched_bytes`, `errors`
/// (retried fetches) and `last_error`.
///
/// # Safety
///
/// `f` comes from [`ironbark_fetcher_open`]; any thread may call this.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_fetcher_progress(f: *const Fetcher) -> *mut c_char {
    // SAFETY: the caller guarantees `f` is a live fetcher.
    let f = unsafe { f.as_ref() };
    report(|| {
        let f = &f.ok_or("null fetcher")?.sparse;
        let p = f.progress();
        Ok(json!({
            "present": p.present,
            "total": p.total,
            "complete": p.present == p.total,
            "fetched_bytes": p.fetched_bytes,
            "errors": p.errors,
            "last_error": f.last_error(),
        }))
    })
}

/// Stop the background threads and release the fetcher; sessions using it keep working.
///
/// # Safety
///
/// `f` is null or came from [`ironbark_fetcher_open`] and is not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_fetcher_close(f: *mut Fetcher) {
    if !f.is_null() {
        // SAFETY: the caller hands back ownership of a box from ironbark_fetcher_open.
        let f = unsafe { Box::from_raw(f) };
        f.sparse.stop();
    }
}

/// Like [`ironbark_open`], reading expert records through `fetcher`: one not on disk yet is
/// fetched when the router first needs it.
///
/// # Safety
///
/// As [`ironbark_open`]; `fetcher` comes from [`ironbark_fetcher_open`] (the session keeps its own
/// reference, so the fetcher may be closed first).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_open_fetcher(
    dir: *const c_char,
    budget: u64,
    slots: u32,
    fetcher: *const Fetcher,
    error: *mut *mut c_char,
) -> *mut Session {
    if fetcher.is_null() {
        // SAFETY: forwarded from this function's contract.
        unsafe { set_error(error, "null fetcher".into()) };
        return std::ptr::null_mut();
    }
    // SAFETY: forwarded from this function's contract.
    unsafe { ironbark_open_fetchers(dir, budget, slots, fetcher, std::ptr::null(), error) }
}

/// Like [`ironbark_open`], with `experts.bin` read through `experts` (from
/// [`ironbark_fetcher_open`]) and `embed.bin` through `embed` (from
/// [`ironbark_embed_fetcher_open`]): what is not on disk yet is fetched when first needed. Either
/// may be null: that file must then be complete.
///
/// # Safety
///
/// As [`ironbark_open`]; `experts` and `embed` are null or live fetchers of their kind (the
/// session keeps its own references, so they may be closed first).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_open_fetchers(
    dir: *const c_char,
    budget: u64,
    slots: u32,
    experts: *const Fetcher,
    embed: *const Fetcher,
    error: *mut *mut c_char,
) -> *mut Session {
    // SAFETY: forwarded from this function's contract.
    let dir = unsafe { str_arg(dir) };
    // SAFETY: the caller guarantees `experts` is null or a live fetcher.
    let experts = unsafe { experts.as_ref() };
    // SAFETY: as above, for `embed`.
    let embed = unsafe { embed.as_ref() };
    let r = dir.and_then(|d| {
        if experts.is_some_and(|f| f.embed) || embed.is_some_and(|f| !f.embed) {
            return Err(
                "fetcher of the wrong file (experts: ironbark_fetcher_open, embed: ironbark_embed_fetcher_open)".into(),
            );
        }
        let records = experts.map(|f| Arc::clone(&f.sparse) as Arc<dyn Records>);
        let chunks = embed.map(|f| Arc::clone(&f.sparse) as Arc<dyn Chunks>);
        catch_unwind(AssertUnwindSafe(|| open(Path::new(d), budget, slots as usize, records, chunks)))
            .unwrap_or_else(|_| Err("panic".into()))
    });
    match r {
        Ok(s) => Box::into_raw(Box::new(s)),
        Err(e) => {
            // SAFETY: forwarded from this function's contract.
            unsafe { set_error(error, e) };
            std::ptr::null_mut()
        }
    }
}

/// Greedy-generate `tokens` tokens after `prompt`. Returns JSON with prefill and decode speed,
/// the text, expert cache counters and memory.
///
/// # Safety
///
/// `s` comes from [`ironbark_open`] and is not used concurrently; `prompt` is a NUL-terminated
/// string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_generate(s: *mut Session, prompt: *const c_char, tokens: u32) -> *mut c_char {
    // SAFETY: forwarded from this function's contract.
    let prompt = unsafe { str_arg(prompt) };
    // SAFETY: the caller guarantees `s` is a live session used by this thread only.
    let s = unsafe { s.as_mut() };
    report(|| match s {
        Some(s) => generate(&mut s.0, prompt?, tokens as usize),
        None => Err("null session".into()),
    })
}

/// Measure the GPU round trip, expert reads from `dir/experts.bin` (skipped when `dir` is null)
/// and memory. Returns JSON.
///
/// # Safety
///
/// `dir` is null or a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_probe(dir: *const c_char) -> *mut c_char {
    let dir = if dir.is_null() {
        None
    } else {
        // SAFETY: forwarded from this function's contract.
        Some(unsafe { str_arg(dir) })
    };
    report(|| probe(dir.transpose()?.map(PathBuf::from).as_deref()))
}

/// GPU time of each part of a decode step (see `ironbark_engine::profile`). Returns JSON.
///
/// # Safety
///
/// `s` comes from [`ironbark_open`] and is not used concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_profile(s: *mut Session) -> *mut c_char {
    // SAFETY: the caller guarantees `s` is a live session used by this thread only.
    let s = unsafe { s.as_ref() };
    report(|| {
        let s = s.ok_or("null session")?;
        let p = s.0.profile(200).map_err(|e| e.to_string())?;
        let parts: Vec<Value> =
            p.parts.iter().map(|(name, us, n)| json!({ "part": name, "us": us, "per_token": n })).collect();
        Ok(json!({ "parts": parts, "token_us": p.token_us, "plan": plan_json(&s.0) }))
    })
}

/// Run the tool-calling cases ([`tool_check`]) with replies of at most `max_tokens`. Forgets
/// the conversation. Returns JSON.
///
/// # Safety
///
/// `s` comes from [`ironbark_open`] and is not used concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_tool_check(s: *mut Session, max_tokens: u32) -> *mut c_char {
    // SAFETY: the caller guarantees `s` is a live session used by this thread only.
    let s = unsafe { s.as_mut() };
    report(|| {
        let s = &mut s.ok_or("null session")?.0;
        tool_check::run(s, max_tokens as usize).map_err(|e| e.to_string())
    })
}

/// Read `text` (a formatted chat turn) into the conversation; `reset` starts a new one.
/// Returns the turn's stats JSON (as [`ironbark_stats`]).
///
/// # Safety
///
/// `s` comes from [`ironbark_open`] and is not used concurrently; `text` is a NUL-terminated
/// string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_start(s: *mut Session, text: *const c_char, reset: bool) -> *mut c_char {
    // SAFETY: forwarded from this function's contract.
    let text = unsafe { str_arg(text) };
    // SAFETY: the caller guarantees `s` is a live session used by this thread only.
    let s = unsafe { s.as_mut() };
    report(|| {
        let s = &mut s.ok_or("null session")?.0;
        s.start(text?, reset).map_err(|e| e.to_string())?;
        Ok(stats(s))
    })
}

/// Start every conversation with a system turn: `system` (may be null) and the tools in `tools`
/// (a JSON array of function schemas, may be null). Forgets the conversation. Returns `{}`.
///
/// # Safety
///
/// `s` comes from [`ironbark_open`] and is not used concurrently; `system` and `tools` are
/// null or NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_set_system(
    s: *mut Session,
    system: *const c_char,
    tools: *const c_char,
) -> *mut c_char {
    // SAFETY: forwarded from this function's contract.
    let system = unsafe { opt_str_arg(system) };
    // SAFETY: forwarded from this function's contract.
    let tools = unsafe { opt_str_arg(tools) };
    // SAFETY: the caller guarantees `s` is a live session used by this thread only.
    let s = unsafe { s.as_mut() };
    report(|| {
        let s = &mut s.ok_or("null session")?.0;
        let tools: Vec<Value> = match tools? {
            Some(t) => serde_json::from_str(t).map_err(|e| format!("tools: {e}"))?,
            None => Vec::new(),
        };
        s.set_system(system?, &tools);
        Ok(json!({}))
    })
}

/// Whether replies reason before answering (default on). Off, a reply to a user message
/// answers at once and its text starts with `</think>`; replies to tool results still reason.
///
/// # Safety
///
/// `s` comes from [`ironbark_open`] and is not used concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_set_thinking(s: *mut Session, on: bool) {
    // SAFETY: the caller guarantees `s` is a live session used by this thread only.
    if let Some(s) = unsafe { s.as_mut() } {
        s.0.set_thinking(on);
    }
}

/// Streaming: layers ahead whose experts each step predicts and prefetches (default 1).
///
/// # Safety
///
/// `s` comes from [`ironbark_open`] and is not used concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_set_lookahead(s: *mut Session, layers: u32) {
    // SAFETY: the caller guarantees `s` is a live session used by this thread only.
    if let Some(s) = unsafe { s.as_mut() } {
        s.0.set_lookahead(layers as usize);
    }
}

/// Decode with FlashHead (approximate LM head) or the exact head (default). Returns
/// `{"available": bool}` (false: the model has no FlashHead).
///
/// # Safety
///
/// `s` comes from [`ironbark_open`] and is not used concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_set_flash_head(s: *mut Session, on: bool) -> *mut c_char {
    // SAFETY: the caller guarantees `s` is a live session used by this thread only.
    let s = unsafe { s.as_mut() };
    report(|| {
        let s = &mut s.ok_or("null session")?.0;
        Ok(json!({ "available": s.set_flash_head(on).map_err(|e| e.to_string())? }))
    })
}

/// Read `messages` into the conversation (`reset` first forgets it, so a transcript can be
/// rebuilt) and open the model's reply; stream it with [`ironbark_next`]. `messages` is a JSON
/// array of `{"role": "user" | "assistant", "content": text}` and `{"role": "tool", "content":
/// [result, …]}`, ending in a user or tool message. Returns the turn's stats JSON.
///
/// # Safety
///
/// `s` comes from [`ironbark_open`] and is not used concurrently; `messages` is a
/// NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_send(s: *mut Session, messages: *const c_char, reset: bool) -> *mut c_char {
    // SAFETY: forwarded from this function's contract.
    let messages = unsafe { str_arg(messages) };
    // SAFETY: the caller guarantees `s` is a live session used by this thread only.
    let s = unsafe { s.as_mut() };
    report(|| {
        let s = &mut s.ok_or("null session")?.0;
        let raw: Vec<Value> = serde_json::from_str(messages?).map_err(|e| format!("messages: {e}"))?;
        // Tool results as string lists, kept alive while the messages borrow them.
        let results: Vec<Vec<&str>> = raw
            .iter()
            .map(|m| m["content"].as_array().map_or_else(Vec::new, |a| a.iter().filter_map(Value::as_str).collect()))
            .collect();
        let list = raw
            .iter()
            .zip(&results)
            .map(|(m, r)| match (m["role"].as_str(), m["content"].as_str()) {
                (Some("user"), Some(c)) => Ok(ironbark::Message::User(c)),
                (Some("assistant"), Some(c)) => Ok(ironbark::Message::Assistant(c)),
                (Some("tool"), _) => Ok(ironbark::Message::ToolResults(r)),
                _ => Err(format!("bad message: {m}")),
            })
            .collect::<Result<Vec<_>>>()?;
        if reset {
            s.reset();
        }
        // The reply streams through ironbark_next.
        s.send(&list).map_err(|e| e.to_string())?;
        Ok(stats(s))
    })
}

/// Split a reply into its reasoning, its answer and its tool calls. Returns JSON `thinking`,
/// `text`, `calls` (`[{name, arguments}]`) and `malformed`.
///
/// # Safety
///
/// `reply` is a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_parse(reply: *const c_char) -> *mut c_char {
    // SAFETY: forwarded from this function's contract.
    let reply = unsafe { str_arg(reply) };
    report(|| {
        let p = ironbark::parse(reply?);
        Ok(json!({
            "thinking": p.thinking,
            "text": p.text,
            "calls": p.calls.iter().map(|c| json!({ "name": c.name, "arguments": c.arguments })).collect::<Vec<_>>(),
            "malformed": p.malformed,
        }))
    })
}

/// The next piece of the reply (UTF-8, possibly empty), or null once the model ends its turn
/// or on error. Free it with [`ironbark_free_string`].
///
/// # Safety
///
/// `s` comes from [`ironbark_open`], has a started turn, and is not used concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_next(s: *mut Session) -> *mut c_char {
    // SAFETY: the caller guarantees `s` is a live session used by this thread only.
    let Some(s) = (unsafe { s.as_mut() }) else {
        return std::ptr::null_mut();
    };
    match catch_unwind(AssertUnwindSafe(|| s.0.next_piece())) {
        Ok(Ok(Some(piece))) => CString::new(piece).map_or(std::ptr::null_mut(), CString::into_raw),
        _ => std::ptr::null_mut(),
    }
}

/// Speed, expert cache, memory and plan of the current turn. Returns JSON.
///
/// # Safety
///
/// `s` comes from [`ironbark_open`] and is not used concurrently.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_stats(s: *mut Session) -> *mut c_char {
    // SAFETY: the caller guarantees `s` is a live session used by this thread only.
    let s = unsafe { s.as_ref() };
    report(|| Ok(stats(&s.ok_or("null session")?.0)))
}

/// Close a session from [`ironbark_open`]; null is ignored.
///
/// # Safety
///
/// `s` is null or came from [`ironbark_open`] and is not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_close(s: *mut Session) {
    if !s.is_null() {
        // SAFETY: the caller hands back ownership of a box from ironbark_open.
        drop(unsafe { Box::from_raw(s) });
    }
}

/// Free a string returned by this library; null is ignored.
///
/// # Safety
///
/// `p` is null or came from this library and is not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ironbark_free_string(p: *mut c_char) {
    if !p.is_null() {
        // SAFETY: the caller hands back ownership of a CString from this library.
        drop(unsafe { CString::from_raw(p) });
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "tests")]

    use std::os::unix::fs::FileExt;

    use super::*;

    const CHUNK: usize = 16384;

    /// The C fetch callback: reads the file `ctx` points to.
    unsafe extern "C" fn read_file(ctx: *mut c_void, offset: u64, buf: *mut u8, len: usize) -> i32 {
        // SAFETY: `ctx` is the test's `File`, alive for the fetcher.
        let file = unsafe { &*ctx.cast::<std::fs::File>() };
        // SAFETY: `buf` holds `len` bytes for the call.
        let buf = unsafe { std::slice::from_raw_parts_mut(buf, len) };
        i32::from(file.read_exact_at(buf, offset).is_err())
    }

    fn c(s: &Path) -> CString {
        CString::new(s.to_str().unwrap()).unwrap()
    }

    /// Takes a C string from this library.
    fn take(p: *mut c_char) -> String {
        // SAFETY: `p` is a fresh string from this library, freed here.
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
        // SAFETY: as above.
        unsafe { ironbark_free_string(p) };
        s
    }

    #[test]
    fn embed_fetcher_fills_through_the_callback() {
        let dir = std::env::temp_dir().join(format!("ironbark-ffi-embed-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (server, client) = (dir.join("server"), dir.join("client"));
        std::fs::create_dir_all(&server).unwrap();
        std::fs::create_dir_all(&client).unwrap();
        let bin: Vec<u8> = (0..3 * CHUNK).map(|i| (i * 7) as u8).collect();
        std::fs::write(server.join("embed.bin"), &bin).unwrap();
        let hashes: Vec<String> =
            bin.chunks(CHUNK).map(|d| format!("\"{:016x}\"", ironbark_engine::manifest::hash(d))).collect();
        let manifest = format!(
            r#"{{"version": 1, "files": {{"embed.bin": {{"bytes": {}, "record_bytes": {CHUNK}, "stride": {CHUNK}, "records": [{}]}}}}}}"#,
            bin.len(),
            hashes.join(", ")
        );
        std::fs::write(client.join("manifest.json"), manifest).unwrap();
        let mut remote = std::fs::File::open(server.join("embed.bin")).unwrap();
        let ctx = (&raw mut remote).cast::<c_void>();

        let mut err = std::ptr::null_mut();
        // SAFETY: valid strings; `ctx` outlives the fetcher.
        let f = unsafe { ironbark_embed_fetcher_open(c(&client).as_ptr(), read_file, ctx, &raw mut err) };
        assert!(!f.is_null());
        // A fetcher of one file cannot stand in for the other; checked before any model is read.
        // SAFETY: valid strings and fetchers.
        let s = unsafe { ironbark_open_fetchers(c(&client).as_ptr(), 0, 0, f, std::ptr::null(), &raw mut err) };
        assert!(s.is_null());
        assert!(take(err).contains("wrong file"));

        // SAFETY: `f` is live.
        unsafe { ironbark_fetcher_start(f, 2) };
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let progress = loop {
            // SAFETY: `f` is live.
            let p: Value = serde_json::from_str(&take(unsafe { ironbark_fetcher_progress(f) })).unwrap();
            if p["complete"] == true || std::time::Instant::now() > deadline {
                break p;
            }
            std::thread::yield_now();
        };
        assert_eq!(progress["complete"], true, "{progress}");
        assert_eq!(progress["total"], 3);
        // SAFETY: `f` is live and not used again.
        unsafe { ironbark_fetcher_close(f) };
        assert_eq!(std::fs::read(client.join("embed.bin")).unwrap(), bin);
        assert!(!client.join("embed.present").exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
