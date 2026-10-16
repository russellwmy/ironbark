# Rust Style Guide

Rules for all code in this workspace. Tools enforce most of them; this file explains the rest and the reasons.

## Table of Contents

1. [Toolchain and checks](#1-toolchain-and-checks)
2. [Formatting](#2-formatting)
3. [Lints](#3-lints)
4. [Naming](#4-naming)
5. [Crates and modules](#5-crates-and-modules)
6. [Errors](#6-errors)
7. [Panics](#7-panics)
8. [Unsafe code](#8-unsafe-code)
9. [Concurrency](#9-concurrency)
10. [Performance](#10-performance)
11. [Documentation](#11-documentation)
12. [Testing](#12-testing)
13. [Dependencies](#13-dependencies)
14. [Commits and review](#14-commits-and-review)

## 1. Toolchain and checks

| File | Purpose |
|---|---|
| `rust-toolchain.toml` | Pins Rust **1.99.0** stable (minimal profile) with `rustfmt`, `clippy`; targets macOS, iOS, iOS simulator (all aarch64) |
| `rustfmt.toml` | Formatting (stable options only) |
| `clippy.toml` | Lint thresholds (lint levels live in `Cargo.toml`) |
| `Cargo.toml` `[workspace.lints]` | Lint levels for every crate |
| `deny.toml` | Licenses, advisories, banned crates, allowed sources |

Every change must pass, in this order:

```
 cargo fmt --all --check
        │
        ▼
 cargo clippy --workspace --all-targets --release -- -D warnings
        │
        ▼
 IRONBARK_MODEL=<packed dir> cargo test --workspace --release -- --test-threads=1
        │
        ▼
 cargo deny check
```

Toolchain upgrades are their own commit: bump `rust-toolchain.toml`, `rust-version` in `Cargo.toml` and `msrv` in `clippy.toml` together, then fix new lints.

## 2. Formatting

- `cargo fmt` decides. Do not hand-format, and do not use `#[rustfmt::skip]` without a comment saying why.
- Max line width is 120.
- Imports: one `use` block per item group, in this order. rustfmt sorts within a group; keep the groups by hand (the nightly-only `group_imports` option is not used).

```rust
use std::sync::Arc;            // 1. std / core / alloc

use anyhow::Result;            // 2. external crates
use ironbark_mlx::Array;

use ironbark_experts::{ExpertRef, Store};   // 3. workspace crates

use crate::row::Kernels;       // 4. this crate
```

## 3. Lints

Levels are set once in `[workspace.lints]`. Each crate opts in with `[lints] workspace = true`.

| Group / lint | Level | Why |
|---|---|---|
| `clippy::all` | deny | Correctness, suspicious, style, complexity, perf |
| `clippy::pedantic` | warn | Stricter style. Noisy numeric lints are allowed (see below) |
| `unwrap_used`, `expect_used`, `panic` | warn | Library code returns errors (see [Panics](#7-panics)) |
| `undocumented_unsafe_blocks`, `multiple_unsafe_ops_per_block` | warn | Every unsafe operation has its own `// SAFETY:` comment |
| `print_stdout`, `print_stderr`, `dbg_macro` | warn | Only the CLI prints; libraries return data |
| `unsafe_op_in_unsafe_fn` | deny | Unsafe calls inside an `unsafe fn` still need an `unsafe {}` block |
| `unreachable_pub`, `missing_debug_implementations` | warn | Keep APIs small and debuggable |
| `todo`, `unimplemented`, `mem_forget`, `rc_mutex`, `lossy_float_literal`, `float_cmp_const`, `unused_qualifications`, `rust_2018_idioms` | warn | No placeholders, leaks, or sloppy floats and paths |

Allowed on purpose:

| Lint | Reason |
|---|---|
| `cast_precision_loss`, `cast_possible_truncation`, `cast_sign_loss` | Kernels cast between `usize`, `f32`, `f64` all the time |
| `many_single_char_names`, `similar_names` | Math code uses `x`, `w`, `l`, `e`, `h`, `f` to match the formulas |

Local exceptions:

- Use `#[expect(clippy::lint_name, reason = "...")]`, not `#[allow]`. `expect` fails when the lint no longer fires, so stale exceptions get removed.
- Put the exception on the smallest item possible. Crate level only for these, always with a `reason`:
  - `ironbark-cli`: `#![expect(clippy::print_stdout, clippy::print_stderr)]`, printing is its job.
  - `ironbark-engine`: `#![allow(clippy::cast_possible_wrap)]`, MLX shapes and indices are `i32`.
  - Integration tests: `#![allow(...)]` for `unwrap` and `expect`.
- The `ironbark-mlx*` crates (from mlx-rs, [DR-0002](docs/decisions/0002-own-mlx-bindings.md)) do not opt into the workspace lints and stay on edition 2021; they must pass default rustc and clippy lints. New code there follows this guide.

## 4. Naming

| Item | Style | Example |
|---|---|---|
| Crates | `ironbark-<area>` | `ironbark-experts` |
| Types, traits | `UpperCamelCase`, nouns | `Store`, `ExpertRef` |
| Functions | `snake_case`, verbs | `request`, `prefetch` |
| Constants | `SCREAMING_SNAKE_CASE`, with unit | `PAGE_BYTES` |
| Conversions | `as_` (cheap borrow), `to_` (costly), `into_` (consumes) | `as_bytes`, `to_f32`, `into_inner` |
| Getters | no `get_` prefix | `slot_bytes()`, not `get_slot_bytes()` |

- Short math names (`x`, `w`, `hd`, `ffn`) are fine inside kernels and model code when the module doc shows the formula they come from.
- Put units in names when the type does not carry them: `wait_ns`, `bytes_read`, `cache_mb`.

## 5. Crates and modules

```
 ironbark-cli ─────────────────────▶ ironbark-engine ──▶ ironbark-experts
 ironbark-ffi ──▶ ironbark (lib) ──▶ ironbark-engine
      │               └────────────────────────────▶ ironbark-experts
      ├────────────────────────────▶ ironbark-engine
      └─────────▶ ironbark-fetch ──▶ ironbark-experts
```

- Dependencies point one way, left to right. No cycles. A lower crate never names a higher one.
- `lib.rs` holds the module list, re-exports and the crate doc. Logic lives in modules.
- Keep visibility as small as possible: private, then `pub(crate)`, then `pub`.
- One main type per module. Split a file when it passes ~400 lines.
- Every crate's `Cargo.toml` uses `version.workspace = true`, `edition.workspace = true`, `[lints] workspace = true`, and takes dependencies from `[workspace.dependencies]`.

## 6. Errors

| Where | Use | Why |
|---|---|---|
| Library crates | `thiserror` enum (`ironbark_experts::Error`, `ironbark_engine::Error`) | Callers can match on the cause |
| `ironbark-cli` | `anyhow::Result` with `.context(...)` | Only needs a readable message |

- Each library crate has its own `Error`, except `ironbark-fetch` (reuses `ironbark_experts::Error`) and `ironbark-ffi` (messages as `String` for C). Wrap a lower crate's error with `#[from]` (e.g. `ironbark_engine::Error::Store`).
- Every error names its file: use the helpers `Error::io(path)`, `Error::json(path)`, `Error::format(path, what)` in `ironbark-experts`, or the crate's own `io` helper.
- Error messages are lowercase with no trailing period: `"read experts.json"`.
- Add context where the error is raised (file path, layer, expert id), not where it is printed.
- Never discard errors silently. `let _ = x;` and `.ok()` need a comment saying why it is safe to ignore.

## 7. Panics

A panic in this workspace means a bug, never bad input or a failed read.

| Case | Do |
|---|---|
| I/O, parsing, user input, config | Return `Result` |
| A broken internal invariant | Make it impossible by construction (check in the constructor, then `let ... else { unreachable!("why") }`). If you must `expect`, put `#[expect(clippy::expect_used, reason = "...")]` on that function |
| `Mutex` poisoning | Recover: `m.lock().unwrap_or_else(PoisonError::into_inner)` (see `lock()` in `ironbark-experts`). The panic that poisoned it already reports the bug |
| Indexing in hot loops | Allowed when the bounds are checked once before the loop; say so in a comment |
| Tests | `unwrap()` is fine |

Public functions that can panic need a `# Panics` doc section.

## 8. Unsafe code

- Allowed only where needed: FFI (`libc`, Metal), raw memory views of buffers, aligned allocation.
- Wrap it in a small safe type in one module (for example `Slab` in `ironbark-experts`). The rest of the code stays safe.
- Every `unsafe` block has a `// SAFETY:` comment just above it that says why each condition holds.
- One unsafe operation per block.
- Prefer safe crates or std APIs when they exist (`bytemuck`-style casts, `as_chunks`, `<[T]>::as_ptr().cast()`).
- `unsafe fn` needs a `# Safety` doc section listing what the caller must guarantee.

```rust
// SAFETY: `p` points to `slot_bytes` bytes inside the allocation, initialized (zeroed
// at allocation); the caller guarantees exclusive access.
unsafe { std::slice::from_raw_parts_mut(p, self.slot_bytes) }
```

## 9. Concurrency

The expert store is event driven. These rules keep it fast and free of deadlocks.

| Rule | Why |
|---|---|
| Never hold a lock during I/O, a blocking wait, or a callback | A held lock serializes every loader |
| Never poll with `sleep` or a timeout loop | Use a `Condvar` or a channel. `std::thread::sleep` is banned in `clippy.toml`. Exception: a fixed-rate sampler (`PeakSampler`, 1 ms `recv_timeout`) |
| Wake only the waiters that care | No `notify_all` on a shared `Condvar` per load |
| One lock order, written in the module doc | Prevents deadlocks |
| Atomics: `Relaxed` for counters only | Anything that publishes data uses `Acquire` / `Release`, with a comment |
| Name threads | `thread::Builder::new().name("ironbark-loader-3")` shows up in Instruments |
| Threads have an owner that joins them (`Store::close`) | No detached threads that outlive the store |

## 10. Performance

- No allocation in the per-token hot path (`step`, expert compute, loaders). Reuse buffers.
- Small functions called across crates in hot loops get `#[inline]`.
- Every Metal kernel has a reference built from MLX ops and a unit test comparing the two.
- Every performance change comes with before/after numbers from `ironbark gen` (speed, memory) and `ironbark ppl` (quality), same model and prompt.
- Expert files are written and read with `F_NOCACHE` (`layout::no_cache`), so cache misses are real SSD reads.

## 11. Documentation

- Every crate has a `//!` doc: what it does, and an ASCII diagram when there is data flow.
- Every `pub` item has a `///` doc. The first line is one sentence.
- Sections when they apply: `# Errors`, `# Panics`, `# Safety`, `# Examples`.
- Put formulas in the module doc, in a `text` code block, matching the variable names in the code.
- Comments say *why*, not *what*.
- Decisions that are hard to reverse go in `docs/decisions/` (see its README).

## 12. Testing

| Kind | Where | What |
|---|---|---|
| Unit | `#[cfg(test)] mod tests` in the same file | Cache eviction, pinning, kernels against MLX reference ops |
| Integration | `crates/<crate>/tests/` | Store with a temp `experts.bin`, end to end |
| Golden | manual, needs model files: `ironbark ppl` | Perplexity matches the reference (Maple, batched prefill: 72.7078 / 30.0218 at 400 / 1024 tokens) |
| Benchmark | `ironbark gen`, `ironbark ppl` | Speed, memory, perplexity |

- Tests must be deterministic: fixed seeds, no timing asserts.
- Concurrency tests run many iterations and assert on final state, not on ordering.
- Floating-point checks use a stated tolerance, never `==`.

## 13. Dependencies

- Add a dependency only when it saves real work. Prefer std.
- Declare it once in `[workspace.dependencies]`; crates use `dep.workspace = true`.
- Turn off default features you don't need.
- `cargo deny check` must pass: allowed licenses only, no yanked crates, no unknown sources, no git dependencies.
- Commit `Cargo.lock` (this workspace builds a binary).

## 14. Commits and review

- One logical change per commit. Formatting-only and toolchain bumps are separate commits.
- Message: an imperative summary under 72 characters, then a body saying *why*.
- A commit that changes speed or memory includes the benchmark numbers in the body.
- Never commit with failing checks from [section 1](#1-toolchain-and-checks).
