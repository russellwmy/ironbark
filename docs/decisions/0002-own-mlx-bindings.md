# DR-0002: Own the MLX bindings in house (ironbark-mlx)

| Field | Value |
|---|---|
| Status | Accepted (2026-10-16) |
| Date | 2026-10-16 |
| Deciders | project owner |
| Consulted | – |
| Related | [DR-0001](0001-ternary-moe-streaming.md), ml-explore/mlx#3915, [crates/ironbark-mlx-sys](../../crates/ironbark-mlx-sys/README.md) |

## Table of Contents

- [Context](#context)
- [Decision drivers](#decision-drivers)
- [Options considered](#options-considered)
- [Decision matrix](#decision-matrix)
- [Decision](#decision)
- [Consequences](#consequences)
- [Follow-ups](#follow-ups)
- [References](#references)

## Context

ironbark reached MLX through mlx-rs 0.32 / mlx-sys 0.6 from crates.io. MLX 0.32.2 compiles its Metal kernels with `-sdk macosx` even for iOS, so ironbark patched mlx-sys (`vendor/mlx-sys`, through `[patch.crates-io]`). Cargo applies `[patch]` only in the top workspace, so anyone using ironbark from crates.io would get the unpatched mlx-sys and a broken iOS build. Upstream mlx-rs (now oxiglade/mlx-rs) has no iOS support, and the MLX fix (#3915) is agreed but not merged.

```
before:  ironbark ──▶ mlx-rs (crates.io) ──▶ mlx-sys (crates.io) ──▶ MLX
                                  ▲
                    [patch] vendor/mlx-sys (only in this workspace)
```

## Decision drivers

| # | Driver | Weight (1–3) |
|---|---|---|
| D1 | iOS works for crates.io users | 3 |
| D2 | Speed (same MLX as today) | 3 |
| D3 | Control: fix and update without waiting on others | 2 |
| D4 | Maintenance effort | 1 |

## Options considered

### Option A: Wait for upstream

Keep `[patch]`, ship to crates.io only after MLX #3915 and an mlx-rs release with iOS.

- Pros: no code to own.
- Cons: no iOS for crates.io users until two upstream releases land; no date.
- Cost / effort: S

### Option B: Switch to safemlx

safemlx 0.3.1 / safemlx-sys 0.2.3 builds for iOS already (MLX v0.32.0 plus 14 patches).

- Pros: iOS from crates.io today.
- Cons: measured on M1 Max (branch `safemlx-spike`): prefill ~20% slower, trit decode ~6% slower. Causes are in its older MLX: a `std::regex` built on every templated `metal_kernel` call (+6.5 µs per custom kernel), and older quantized kernel selection (`quantized_matmul` +30%, `gather_qmm` +55%). Newer MLX depends on one maintainer rebasing 14 patches. ~244 call sites change (explicit streams).
- Cost / effort: M

### Option C: Vendor and rename the packages only

Copy mlx-rs and mlx-sys, publish as `ironbark-mlx` / `ironbark-mlx-sys`, keep the library names `mlx_rs` / `mlx_sys`, keep mlx-macros and mlx-internal-macros from crates.io.

- Pros: easy to diff against new mlx-rs releases; easy to drop.
- Cons: still depends on mlx-rs crates (macros); the code says `mlx_rs` while the crate is `ironbark-mlx`.
- Cost / effort: S

### Option D: Move in house

The code becomes two ironbark crates, `ironbark-mlx` and `ironbark-mlx-sys`, with `ironbark_mlx*` paths, ironbark's version, style and lints. No mlx-rs dependency left. ironbark is inference only, so the training API is cut: `nn`, `module`, `optimizers`, `losses`, `quantization` (the `nn` quantized layers), autodiff (`grad`, `value_and_grad`, `jvp`, `vjp`) and mlx-macros (the derives for `nn`). The 245 deprecated `*_device` shims go too (use `with_stream` / `with_device`), and with them mlx-internal-macros, which only generated their `op!()` macros.

- Pros: iOS from crates.io; same MLX 0.32.2; one set of crates to publish; full control.
- Cons: ~26k lines to own; mlx-rs fixes come only by porting them by hand.
- Cost / effort: M

## Decision matrix

| Option | D1 (×3) | D2 (×3) | D3 (×2) | D4 (×1) | Total |
|---|---|---|---|---|---|
| A: wait | 1 | 5 | 1 | 5 | 25 |
| B: safemlx | 5 | 2 | 2 | 3 | 28 |
| C: vendor, rename packages | 5 | 5 | 3 | 4 | 40 |
| D: in house | 5 | 5 | 5 | 2 | 42 |

## Decision

We choose **Option D** because it gives iOS to crates.io users at today's speed (D1, D2), and ironbark controls every part of the MLX stack (D3). C scores close, but it keeps a dependency on mlx-rs crates and two names for the same code.

```
after:  ironbark crates ──▶ ironbark-mlx ──▶ ironbark-mlx-sys ──▶ mlx-c ──▶ MLX v0.32.2 (+ iOS patch)
```

Base: mlx-rs 0.32.0, mlx-sys 0.6.0, mlx-macros 0.32.0, mlx-internal-macros 0.32.0 (MIT OR Apache-2.0, the mlx-rs authors).

## Consequences

| Type | Item |
|---|---|
| Good | iOS builds from crates.io with no `[patch]`; no `vendor/` dir |
| Good | Same speed and output as before: Mac benchmarks within ±4%, iPhone 17 within ±3% of the README tables, perplexity 67.8979 unchanged |
| Good | One workspace: fmt, clippy and 820 tests (with the kept mlx-rs suite) run on all of it |
| Bad | ~26k lines of mlx-rs code we now maintain |
| Bad | No training in Rust: fine-tuning or LoRA would need `nn`, optimizers and autodiff back (from mlx-rs 0.32) |
| Bad | mlx-rs fixes and new MLX versions are ported by hand |
| Risk | MLX update breaks the iOS patch → mitigation: the patch is one CMake file; rebuild for iOS on every MLX bump |

## Follow-ups

- [ ] Send the MLX iOS fix upstream (ml-explore/mlx#3915); drop `patch-mlx-ios.cmake` once released
- [ ] Publish order: `ironbark-mlx-sys`, `ironbark-mlx`, then the ironbark crates

## References

- mlx-rs: https://github.com/oxiglade/mlx-rs
- MLX iOS Metal SDK issue: https://github.com/ml-explore/mlx/issues/3915
- safemlx: https://github.com/jbg/eredu
