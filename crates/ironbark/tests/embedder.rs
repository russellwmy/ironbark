//! Embeddings on a real model: set `IRONBARK_EMBED_MODEL` to a Qwen3-Embedding-0.6B
//! `mlx_lm.convert -q` dir (skipped otherwise). Run with `--test-threads=1`: the drop test
//! reads MLX's process-wide memory.
//!
//! `data/embed_golden.json`: vectors from sentence-transformers (`Qwen/Qwen3-Embedding-0.6B`,
//! float32), queries with its `query` prompt.

#![allow(clippy::unwrap_used, clippy::expect_used, reason = "tests")]

use ironbark::{EmbedKind, EmbedOptions, Embedder, Value};

fn open(options: EmbedOptions) -> Option<Embedder> {
    let dir = std::env::var_os("IRONBARK_EMBED_MODEL")?;
    assert!(ironbark::is_embedder(dir.as_ref()));
    Some(Embedder::open(dir, options).expect("open"))
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn kind(k: &str) -> EmbedKind {
    if k == "query" { EmbedKind::Query } else { EmbedKind::Document }
}

/// Cosine to the float32 reference that 4-bit (group 64) meets: it measures 0.93–0.95, as
/// mlx-lm's own forward of the same weights does (6-bit 0.99, 8-bit 0.999).
const MIN_COSINE: f32 = 0.92;

#[test]
fn matches_the_reference_vectors() {
    let Some(mut e) = open(EmbedOptions::default()) else { return };
    assert_eq!(e.dims(), 1024);
    let golden: Value = serde_json::from_str(include_str!("data/embed_golden.json")).unwrap();
    let items = golden["items"].as_array().unwrap();
    let (mut got, mut want) = (Vec::new(), Vec::new());
    for item in items {
        let text = item["text"].as_str().unwrap();
        let w: Vec<f32> = item["vector"].as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let g = e.embed(&[text], kind(item["kind"].as_str().unwrap())).unwrap().remove(0);
        assert!((dot(&g, &g) - 1.0).abs() < 1e-3);
        let sim = dot(&g, &w);
        assert!(sim >= MIN_COSINE, "{text}: cosine {sim}");
        got.push(g);
        want.push(w);
    }
    // The query ranks the documents as the reference does.
    let order = |v: &[Vec<f32>]| {
        let mut i: Vec<usize> = (1..v.len()).collect();
        i.sort_by(|&a, &b| dot(&v[0], &v[b]).total_cmp(&dot(&v[0], &v[a])));
        i
    };
    assert_eq!(order(&got), order(&want));
}

#[test]
fn ranks_the_answering_passage_first() {
    let Some(mut e) = open(EmbedOptions::default()) else { return };
    let q = e.embed(&["how do I end the lease"], EmbedKind::Query).unwrap().remove(0);
    let docs = [
        "The tenant pays rent on the first day of each month by bank transfer.",
        "Termination: either party may end this lease with 60 days' written notice to the other.",
        "Photosynthesis turns light, water and carbon dioxide into sugar and oxygen.",
        "Berlin is the capital and largest city of Germany.",
    ];
    let d = e.embed(&docs, EmbedKind::Document).unwrap();
    let scores: Vec<f32> = d.iter().map(|v| dot(&q, v)).collect();
    let best = scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(best, 1, "{scores:?}");
}

#[test]
fn fewer_dims_are_unit_vectors() {
    let Some(mut e) = open(EmbedOptions {
        dims: Some(256),
        ..EmbedOptions::default()
    }) else {
        return;
    };
    assert_eq!(e.dims(), 256);
    let v = e.embed(&["hello"], EmbedKind::Document).unwrap().remove(0);
    assert_eq!(v.len(), 256);
    assert!((dot(&v, &v) - 1.0).abs() < 1e-4);
}

#[test]
fn long_texts_are_cut_to_max_tokens() {
    let Some(mut e) = open(EmbedOptions {
        max_tokens: 16,
        ..EmbedOptions::default()
    }) else {
        return;
    };
    let long = "word ".repeat(200);
    let ids = e.tokens(&long, EmbedKind::Document).unwrap();
    assert_eq!(ids.len(), 16);
    assert_eq!(ids.last(), Some(&151_643));
    assert_eq!(e.embed(&[&long], EmbedKind::Document).unwrap()[0].len(), 1024);
}

#[test]
fn dropping_frees_memory() {
    let Some(mut e) = open(EmbedOptions::default()) else { return };
    e.embed(&["warm up"], EmbedKind::Document).unwrap();
    let held = ironbark_mlx_memory();
    drop(e);
    let after = ironbark_mlx_memory();
    assert!(after + 300_000_000 < held, "held {held}, after {after}");
}

/// MLX's active plus cached bytes.
fn ironbark_mlx_memory() -> usize {
    ironbark_mlx::memory::active_memory().unwrap() + ironbark_mlx::memory::cache_memory().unwrap()
}
