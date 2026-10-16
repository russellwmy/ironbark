//! A sparse `experts.bin` that fills from a remote copy, alone and under the engine's store.

#![allow(clippy::unwrap_used, reason = "tests")]

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::time::Duration;

use ironbark_experts::layout::hash;
use ironbark_experts::{Chunks, PAGE_BYTES, Store};
use ironbark_fetch::{Fetch, FileFetch, Sparse};

const LAYERS: usize = 2;
const EXPERTS: usize = 4;
const REC: usize = 1000;

fn record(i: usize) -> Vec<u8> {
    (0..REC).map(|j| (i * 31 + j * 7) as u8).collect()
}

/// A pack in `dir`: the full `experts.bin` in `dir/server`, and only `experts.json` and
/// `manifest.json` in `dir/client`.
fn pack(name: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("ironbark-remote-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (server, client) = (dir.join("server"), dir.join("client"));
    std::fs::create_dir_all(&server).unwrap();
    std::fs::create_dir_all(&client).unwrap();
    let n = LAYERS * EXPERTS;
    let mut bin = vec![0u8; (n - 1) * PAGE_BYTES + REC];
    for i in 0..n {
        bin[i * PAGE_BYTES..i * PAGE_BYTES + REC].copy_from_slice(&record(i));
    }
    std::fs::write(server.join("experts.bin"), &bin).unwrap();
    let layout =
        format!(r#"{{"layers": {LAYERS}, "experts": {EXPERTS}, "record_bytes": {REC}, "stride": {PAGE_BYTES}}}"#);
    let hashes: Vec<String> = (0..n).map(|i| format!("\"{:016x}\"", hash(&record(i)))).collect();
    let manifest = format!(r#"{{"version": 1, "files": {{"experts.bin": {{"records": [{}]}}}}}}"#, hashes.join(", "));
    std::fs::write(client.join("experts.json"), layout).unwrap();
    std::fs::write(client.join("manifest.json"), manifest).unwrap();
    (server, client)
}

/// Fetches from a file, corrupting the first `bad` fetches and counting all of them.
struct Flaky {
    inner: FileFetch,
    bad: AtomicUsize,
    calls: AtomicUsize,
}

impl Flaky {
    fn new(server: &Path, bad: usize, delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            inner: FileFetch::open(&server.join("experts.bin"), delay).unwrap(),
            bad: AtomicUsize::new(bad),
            calls: AtomicUsize::new(0),
        })
    }
}

impl Fetch for Flaky {
    fn fetch(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
        self.calls.fetch_add(1, Relaxed);
        self.inner.fetch(offset, buf)?;
        if self.bad.try_update(Relaxed, Relaxed, |b| b.checked_sub(1)).is_ok() {
            buf[0] ^= 0xff;
        }
        Ok(())
    }
}

#[test]
fn fetches_on_demand_fills_and_reopens_local() {
    let (server, client) = pack("fill");
    // One corrupted fetch: the hash check rejects it and the retry succeeds.
    let fetch = Flaky::new(&server, 1, Duration::from_millis(20));
    let sparse = Sparse::open(&client, fetch.clone()).unwrap();
    sparse.start(1, &[7, 6]);
    let store = Store::open_with(&client, 2, 2, sparse.clone()).unwrap();
    store.request(1, 2);
    let r = store.wait(1, 2).unwrap();
    assert_eq!(*r, record(EXPERTS + 2));
    drop(r);

    assert!(sparse.wait_filled(Duration::from_secs(10)), "{:?}", sparse.progress());
    let p = sparse.progress();
    assert!(p.errors >= 1, "{p:?}");
    assert!(!client.join("experts.present").exists());
    store.close();
    sparse.stop();

    // Complete: a plain local store, no more fetches.
    let calls = fetch.calls.load(Relaxed);
    let store = Store::open(&client, 2, 2).unwrap();
    for e in 0..EXPERTS {
        store.request(0, e);
        assert_eq!(*store.wait(0, e).unwrap(), record(e));
    }
    assert_eq!(fetch.calls.load(Relaxed), calls);
    store.close();
    std::fs::remove_dir_all(client.parent().unwrap()).unwrap();
}

#[test]
fn bad_remote_fails_the_load_and_resumes_after_reopen() {
    let (server, client) = pack("bad");
    // Every fetch corrupted: the demand load fails after its retries.
    let sparse = Sparse::open(&client, Flaky::new(&server, usize::MAX, Duration::ZERO)).unwrap();
    let store = Store::open_with(&client, 2, 1, sparse).unwrap();
    store.request(0, 1);
    assert!(store.wait(0, 1).is_err());
    store.close();
    assert!(client.join("experts.present").exists());

    // A good remote: the records come in, and those fetched stay.
    let fetch = Flaky::new(&server, 0, Duration::ZERO);
    let sparse = Sparse::open(&client, fetch.clone()).unwrap();
    let store = Store::open_with(&client, 2, 1, sparse.clone()).unwrap();
    store.request(0, 1);
    assert_eq!(*store.wait(0, 1).unwrap(), record(1));
    assert_eq!(sparse.progress().present, 1);
    store.close();
    let store = Store::open_with(&client, 2, 1, Sparse::open(&client, fetch.clone()).unwrap()).unwrap();
    store.request(0, 1);
    assert_eq!(*store.wait(0, 1).unwrap(), record(1));
    assert_eq!(fetch.calls.load(Relaxed), 1);
    store.close();
    std::fs::remove_dir_all(client.parent().unwrap()).unwrap();
}

#[test]
fn fills_without_an_engine() {
    let (server, client) = pack("alone");
    let sparse = Sparse::open(&client, Flaky::new(&server, 0, Duration::ZERO)).unwrap();
    sparse.start(3, &[]);
    assert!(sparse.wait_filled(Duration::from_secs(10)), "{:?}", sparse.progress());
    sparse.stop();
    assert!(!client.join("experts.present").exists());
    let bin = std::fs::read(client.join("experts.bin")).unwrap();
    assert_eq!(bin.len(), LAYERS * EXPERTS * PAGE_BYTES);
    for i in 0..LAYERS * EXPERTS {
        assert_eq!(&bin[i * PAGE_BYTES..i * PAGE_BYTES + REC], record(i).as_slice());
    }
    std::fs::remove_dir_all(client.parent().unwrap()).unwrap();
}

const CHUNK: usize = 16384;
const CHUNKS: usize = 5;

fn chunk(i: usize) -> Vec<u8> {
    (0..CHUNK).map(|j| (i * 13 + j * 3) as u8).collect()
}

/// An `embed.bin` of [`CHUNKS`] chunks in `dir/server`, and only `manifest.json` in `dir/client`.
fn embed_pack(name: &str) -> (PathBuf, PathBuf) {
    let dir = std::env::temp_dir().join(format!("ironbark-remote-embed-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let (server, client) = (dir.join("server"), dir.join("client"));
    std::fs::create_dir_all(&server).unwrap();
    std::fs::create_dir_all(&client).unwrap();
    let bin: Vec<u8> = (0..CHUNKS).flat_map(chunk).collect();
    std::fs::write(server.join("embed.bin"), &bin).unwrap();
    let hashes: Vec<String> = (0..CHUNKS).map(|i| format!("\"{:016x}\"", hash(&chunk(i)))).collect();
    let manifest = format!(
        r#"{{"version": 1, "files": {{"embed.bin": {{"bytes": {}, "record_bytes": {CHUNK}, "stride": {CHUNK}, "records": [{}]}}}}}}"#,
        bin.len(),
        hashes.join(", ")
    );
    std::fs::write(client.join("manifest.json"), manifest).unwrap();
    (server, client)
}

#[test]
fn embed_chunks_arrive_on_demand_then_fill() {
    let (server, client) = embed_pack("demand");
    let calls = |f: &Flaky| f.calls.load(Relaxed);
    // One corrupted fetch: the hash check rejects it and the retry succeeds.
    let fetch = Arc::new(Flaky {
        inner: FileFetch::open(&server.join("embed.bin"), Duration::ZERO).unwrap(),
        bad: AtomicUsize::new(1),
        calls: AtomicUsize::new(0),
    });
    let sparse = Sparse::open_embed(&client, fetch.clone()).unwrap();
    let path = client.join("embed.bin");
    assert_eq!(std::fs::metadata(&path).unwrap().len(), (CHUNKS * CHUNK) as u64);
    assert!(client.join("embed.present").exists());

    // Only the chunk asked for is fetched, and it is in the file (what the engine maps).
    sparse.ensure(3).unwrap();
    assert!(sparse.has(3) && !sparse.has(2) && !sparse.complete());
    assert_eq!(calls(&fetch), 2);
    let bin = std::fs::read(&path).unwrap();
    assert_eq!(&bin[3 * CHUNK..4 * CHUNK], chunk(3).as_slice());
    assert!(bin[..3 * CHUNK].iter().all(|&b| b == 0));
    sparse.ensure(3).unwrap();
    assert_eq!(calls(&fetch), 2);

    sparse.start(2, &[]);
    assert!(sparse.wait_filled(Duration::from_secs(10)), "{:?}", sparse.progress());
    sparse.stop();
    assert!(sparse.complete());
    assert!(!client.join("embed.present").exists());
    let bin = std::fs::read(&path).unwrap();
    for i in 0..CHUNKS {
        assert_eq!(&bin[i * CHUNK..(i + 1) * CHUNK], chunk(i).as_slice());
    }

    // Complete: reopens local, never fetches.
    let n = calls(&fetch);
    let again = Sparse::open_embed(&client, fetch.clone()).unwrap();
    assert!(again.complete());
    (0..CHUNKS).for_each(|i| again.ensure(i).unwrap());
    assert_eq!(calls(&fetch), n);
    std::fs::remove_dir_all(client.parent().unwrap()).unwrap();
}
