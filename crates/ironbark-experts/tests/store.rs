//! Expert store against a small synthetic `experts.bin` (no model files needed).

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::Ordering::Relaxed;

use ironbark_experts::{Layout, Store};

const LAYERS: usize = 2;
const EXPERTS: usize = 4;
const HIDDEN: usize = 8;
const FFN: usize = 4;
const STRIDE: u64 = 16384;

/// Bytes per record.
const REC: usize = 3 * HIDDEN * FFN * 2;

/// Value every byte of record (l, e) is filled with.
fn tag(l: usize, e: usize) -> u8 {
    (l * 100 + e) as u8
}

fn fixture(name: &str) -> std::io::Result<PathBuf> {
    let dir = std::env::temp_dir().join(format!("ironbark-experts-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let f = File::create(dir.join("experts.bin"))?;
    for l in 0..LAYERS {
        for e in 0..EXPERTS {
            let bytes: Vec<u8> = vec![tag(l, e); REC];
            f.write_all_at(&bytes, (l * EXPERTS + e) as u64 * STRIDE)?;
        }
    }
    f.set_len((LAYERS * EXPERTS) as u64 * STRIDE)?;
    let meta = format!(r#"{{"layers":{LAYERS},"experts":{EXPERTS},"record_bytes":{REC},"stride":{STRIDE}}}"#);
    std::fs::write(dir.join("experts.json"), meta)?;
    Ok(dir)
}

#[test]
fn layout_reads_json() {
    let dir = fixture("layout").unwrap();
    let l = Layout::read(&dir).unwrap();
    assert_eq!((l.layers, l.experts, l.stride), (LAYERS, EXPERTS, STRIDE));
    assert_eq!(l.record_bytes, REC as u64);
}

#[test]
fn missing_json_is_an_error() {
    let dir = std::env::temp_dir().join(format!("ironbark-experts-none-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    assert!(Store::open(&dir, 2, 1).is_err());
}

#[test]
fn streamed_expert_has_its_own_data() {
    let s = Store::open(&fixture("data").unwrap(), 2, 2).unwrap();
    for (l, e) in [(0, 1), (1, 3), (0, 2)] {
        s.request(l, e);
        let w = s.wait(l, e).expect("loaded");
        assert_eq!(w.len(), REC);
        assert!(w.iter().all(|&v| v == tag(l, e)));
    }
}

#[test]
fn full_cache_preloads_and_always_hits() {
    let s = Store::open(&fixture("full").unwrap(), EXPERTS, 1).unwrap();
    assert_eq!(s.resident(), LAYERS * EXPERTS);
    s.request(1, 2);
    drop(s.wait(1, 2).expect("loaded"));
    assert_eq!(s.stats.hits.load(Relaxed), 1);
    assert_eq!(s.stats.demand_loads.load(Relaxed), 0);
}

#[test]
fn per_layer_budget_evicts_idle_lru() {
    let s = Store::open(&fixture("evict").unwrap(), 2, 2).unwrap();
    for e in 0..EXPERTS {
        s.request(0, e);
        drop(s.wait(0, e).expect("loaded")); // release so it can be evicted
    }
    // Two slots per layer, all idle: never more than the budget.
    assert_eq!(s.resident(), 2);
    // The two most recent are still cached.
    s.request(0, 3);
    drop(s.wait(0, 3).expect("loaded"));
    s.request(0, 2);
    drop(s.wait(0, 2).expect("loaded"));
    assert_eq!(s.stats.hits.load(Relaxed), 2);
}

#[test]
fn in_use_expert_is_not_evicted() {
    let s = Store::open(&fixture("pin").unwrap(), 2, 2).unwrap();
    s.request(0, 0);
    let held = s.wait(0, 0).expect("loaded"); // keep it pinned
    for e in 1..EXPERTS {
        s.request(0, e);
        drop(s.wait(0, e).expect("loaded"));
    }
    assert!(held.iter().all(|&v| v == tag(0, 0)));
    // Still cached: a new wait is a hit.
    let hits = s.stats.hits.load(Relaxed);
    drop(s.wait(0, 0).expect("loaded"));
    assert_eq!(s.stats.hits.load(Relaxed), hits + 1);
}

#[test]
fn demand_waits_for_a_pinned_slot_to_be_released() {
    let s = Store::open(&fixture("budget").unwrap(), 2, 2).unwrap();
    s.request(0, 0);
    let a = s.wait(0, 0).expect("loaded");
    s.request(0, 1);
    let b = s.wait(0, 1).expect("loaded");
    // Both slots of layer 0 are pinned: a third expert must wait, never exceed the budget.
    std::thread::scope(|sc| {
        let h = sc.spawn(|| {
            s.request(0, 2);
            let c = s.wait(0, 2).expect("loaded");
            c.iter().all(|&v| v == tag(0, 2))
        });
        while s.stats.slot_waits.load(Relaxed) == 0 {
            std::thread::yield_now();
        }
        assert_eq!(s.resident(), 2);
        drop(a);
        assert!(h.join().unwrap());
    });
    assert_eq!(s.resident(), 2);
    assert!(b.iter().all(|&v| v == tag(0, 1)));
}

#[test]
fn many_threads_read_correct_data() {
    let s = Store::open(&fixture("stress").unwrap(), 2, 4).unwrap();
    std::thread::scope(|sc| {
        for t in 0..4 {
            let s = &s;
            sc.spawn(move || {
                for i in 0..200 {
                    let (l, e) = ((t + i) % LAYERS, (t * 7 + i) % EXPERTS);
                    s.request(l, e);
                    let w = s.wait(l, e).expect("loaded");
                    assert!(w.iter().all(|&v| v == tag(l, e)), "wrong data for ({l}, {e})");
                }
            });
        }
    });
    assert!(s.resident() <= LAYERS * 2);
}

#[test]
fn a_failed_read_is_an_error_not_a_hang() {
    let dir = fixture("fail").unwrap();
    let s = Store::open(&dir, 2, 2).unwrap();
    // The file shrinks after open (as if the storage failed): reads of the tail fail.
    File::options().write(true).open(dir.join("experts.bin")).unwrap().set_len(STRIDE).unwrap();
    s.request(1, 3);
    let err = s.wait(1, 3).expect_err("read past the end fails");
    assert!(err.to_string().contains("expert 3 of layer 1"), "{err}");
    // Every later wait reports it too.
    assert!(s.wait(0, 0).is_err());
    s.close();
}
