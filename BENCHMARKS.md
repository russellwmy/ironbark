# Benchmarks

Measured speed, memory and accuracy of ironbark on Apple Silicon. How the budget is planned: [Memory budget](README.md#memory-budget).

## Table of Contents

- [Summary](#summary)
- [Mac (M1 Max)](#mac-m1-max)
  - [Checkpoint vs reference](#checkpoint-vs-reference)
  - [Prefill](#prefill)
  - [Streaming decode](#streaming-decode)
  - [Readback cost](#readback-cost)
  - [Lookahead](#lookahead)
  - [Packs by budget](#packs-by-budget)
  - [On-demand experts](#on-demand-experts)
  - [On-demand embedding](#on-demand-embedding)
- [iPhone 17](#iphone-17)

## Summary

| Where | Setup | Decode | Peak memory |
|---|---|---|---|
| M1 Max | Checkpoint, all experts in RAM | 232 tok/s | 5.55 GB (DeepGrove's `maple.py`: 6.81 GB) |
| M1 Max | Trit pack, experts streamed from SSD, 1.25 GB budget | 26.6 tok/s | 1.15 GB |
| iPhone 17 | Trit pack, device budget (3 GB) | 30–31 tok/s | 2.8 GB |
| iPhone 17 | Trit pack, 1.5 GB budget | 13.7 tok/s | 1.3 GB |

Greedy decode; M1 Max checkpoint row after a short prompt, the others after a 747–887-token prompt (iPhone: prose).

## Mac (M1 Max)

### Checkpoint vs reference

M1 Max (64 GB), measured 2026-10-15. Prompts are the first 3,000 characters of `prompts/prose.txt` and `rust.txt` (746 and 887 tokens); the long prompt is 10,000 characters of prose (2,327 tokens). Decode = greedy tokens/s after a short prompt (256 tokens) and after the prose prompt (128 tokens). DeepGrove's numbers are from mlx-lm `stream_generate` with a 512-token prompt.

| Engine | Decode, short | Decode, 746-token context | Peak GPU memory | Perplexity 400 / 1024 tok |
|---|---|---|---|---|
| DeepGrove `maple.py` (mlx-lm, fused Metal kernels) | – | 217 tok/s | 6.81 GB | 67.8978 / 31.2153 |
| DeepGrove + FlashHead (approximate head) | – | 233 tok/s | 7.30 GB | – |
| ironbark (checkpoint) | **232 tok/s** | 205 tok/s | **5.55 GB** | 67.8979 / 31.2154 |

### Prefill

Prefill runs in chunks of up to 512 tokens (`prefill.rs`: MLX `quantized_matmul` / `gather_qmm` with row scales expanded per layer, experts sorted by id, causal or 512-window masks): 1,143 tok/s on the prose prompt and 1,180 tok/s on the long one (reference ~1,200). Its perplexity matches the reference's batched path exactly (72.7078 / 30.0218 at 400 / 1024 tokens vs 72.7077 / 30.0218). Maple's batched and token-by-token arithmetic differ by ±5% in perplexity in the reference too.

Streaming models prefill in chunks too: per layer the chunk's routing is read back and its distinct experts go through the cache in groups that fit (pin, copy the records into MLX's layout, `gather_qmm`, unpin). Each group is forced onto the same MLX kernel the resident call would use (MLX switches between a tiled and a per-row kernel on rows ÷ experts), so the result equals resident prefill bit for bit at any cache size. When every expert fits, none of that is needed:

| Pack | How the experts are held | Prefill, 887 tokens | Decode | Peak footprint |
|---|---|---|---|---|
| 2-bit | Read once from `experts.bin` into resident tensors in the checkpoint's layout; no store | 1,095 tok/s | 197.7 tok/s | 5.76 GB |
| trit | Preloaded in the slab (stays packed, 0.9 GB smaller); prefill reads each layer's slab rows in place (one unpack per layer), no routing readback | 757 tok/s | ~172 tok/s | 5.16 GB |
| Checkpoint, for reference | Resident tensors | ~1,110 tok/s | 198.5 tok/s | 6.06 GB |

MLX's `gather_qmm` is ~50% slower on the slab's records than on contiguous tensors (14.5 vs 9.4 ms per layer): each record interleaves up+gate codes, down codes and row scales at a 784 KiB stride. Perplexity is unchanged in every mode (72.7078 / 30.0218 batched, 67.8979 token by token).

Streamed prefill (trit pack, lookahead 1):

| Cache per layer | Prefill, 746 tokens | Prefill, 2,327 tokens | Peak GPU memory |
|---|---|---|---|
| 128 | 322 tok/s | 348 tok/s | 2.84 GB |
| 64 | 305 tok/s | 330 tok/s | 1.72 GB |
| 32 | 285 tok/s | 306 tok/s | 1.18 GB |

### Streaming decode

Trit pack, M1 Max internal SSD, 256 decoded tokens, lookahead 1, one run each; perplexity unchanged at 67.8979:

| Cache per layer | Peak GPU memory | Prose: hit / read / decode | Rust: hit / read / decode |
|---|---|---|---|
| all (see above) | 5.12 GB | – / – / ~172 tok/s | – / – / ~172 tok/s |
| 128 | 2.79 GB | 96.1% / 12.4 MB/tok / 68 tok/s | 95.0% / 18.3 MB/tok / 39 tok/s |
| 64 | 1.68 GB | 89.4% / 44.1 MB/tok / 34 tok/s | 87.8% / 44.7 MB/tok / 32 tok/s |
| 32 | 1.14 GB | 81.8% / 74.0 MB/tok / 28 tok/s | 78.0% / 78.7 MB/tok / 26 tok/s |
| 16 | 0.89 GB | 72.5% / 106.1 MB/tok / 23 tok/s | 64.3% / 128.0 MB/tok / 21 tok/s |

The per-layer readback (24 GPU→CPU round trips per token) costs more than the SSD: at 128 per layer, prose waits 2.5 ms per token for loads but decodes at 68 tok/s, against ~172 when every expert fits and no readback is needed. A bare MLX round trip (one tiny op, eval, read) takes ~225 µs on this Mac, so that cost is MLX/Metal latency, not ironbark.

### Readback cost

Skipping readbacks while staying exact would mean running a layer on the GPU's own slot map and redoing it when one of its 8 experts was not cached. How often a layer has all 8 cached (Mac, 128 decoded tokens):

| Cache per layer | Layers with all 8 cached, prose / Rust | Expert hit, prose / Rust |
|---|---|---|
| 119 (iPhone, device budget) | 73.7 / 64.3% | 95.6 / 94.0% |
| 62 (iPhone, 2 GB) | 47.6 / 40.1% | 89.3 / 87.1% |
| 34 (iPhone, 1.5 GB) | 30.5 / 24.9% | 83.2 / 79.5% |

Two layers in a row are both cached about half the time at 119 (0.7²), less below. On the iPhone a skipped sync saves 0.2–0.5 ms, and a wrong guess wastes a layer's GPU work (~0.8 ms) plus the sync anyway, so it does not pay at these cache sizes. With every expert cached there is no readback already.

The round trip itself is Metal's, not MLX's. Raw Metal on the M1 Max (one 1-thread kernel, no MLX; GPU busy 8 µs), 2026-10-15:

| How the CPU waits for the GPU, or resumes it | Round trip |
|---|---|
| New command buffer, `waitUntilCompleted` (what MLX does) | 219 µs |
| New command buffer, CPU spins on its status | 203 µs |
| Shared event, CPU spins on its value | 182 µs |
| One command buffer per token: GPU signals, then waits on an event the CPU signals | 206 µs per handoff |
| One kernel: GPU polls a flag in shared memory the CPU writes | never sees the write (GPU → CPU is visible after ~0.5 ms; CPU → GPU is not, and Metal has no system-scope atomics) |

Spinning in MLX's `EventImpl::wait` (a test patch) left MLX's round trip at 206 µs and slowed streamed decode (prose 48–62 → 45 tok/s at 119 per layer, the spinning core competing with the expert loaders), so it was dropped. Each readback needs the GPU to resume after a CPU decision, which costs ~200 µs of scheduling on Apple GPUs however it is expressed; only fewer CPU decisions per token cut it.

### Lookahead

Lookahead prefetch (same prompts as [Streaming decode](#streaming-decode), decode tok/s):

| Lookahead | Prose 128 | Prose 64 | Prose 32 | Rust 128 | Rust 64 | Rust 32 |
|---|---|---|---|---|---|---|
| 0 | 44.0 | 29.3 | 25.3 | 38.5 | 29.2 | 26.3 |
| 1 (default) | **68.4** | 33.7 | 28.1 | 39.3 | 32.3 | **26.4** |
| 2 | 50.7 | **35.9** | **28.9** | **46.8** | **33.1** | 26.3 |

It raises the effective hit rate (prose at 64: 74.7% → 89.4%) at the cost of more reads (31 → 44 MB per token).

### Packs by budget

M1 Max, 2026-10-16 (Rust prompt, 887 tokens, 128 decoded, one run per row; footprint = what iOS counts; packs on the internal SSD: from a disk image the load wait was ~2x and streamed decode up to 2x slower). In brackets: before `embed.bin` (embedding loaded into memory).

Trit records (`ironbark pack` default):

| Budget | Experts per layer | Hit | Load wait | Peak footprint | Prefill | Decode |
|---|---|---|---|---|---|---|
| 1.25 GB | 30 | 77% | 16.5 ms/tok | 1.15 GB | 226 tok/s | 26.6 tok/s |
| 2 GB | 72 | 89% | 8.9 ms/tok | 1.90 GB | 306 tok/s | 32.1 tok/s |
| 3 GB | 128 | 95% | 4.2 ms/tok | 2.90 GB | 299 tok/s | 39.2 tok/s |
| 4 GB | 184 | 98% | 1.3 ms/tok | 3.85 GB | 299 tok/s | 77.9 tok/s |
| 6 GB | 256 | all | – | 5.14–5.23 GB | 691–758 tok/s | 170–172 tok/s |

2-bit records (`ironbark pack --two-bit`):

| Budget | Experts per layer | Hit | Load wait | Peak footprint | Prefill | Decode |
|---|---|---|---|---|---|---|
| 1.25 GB | 25 (18) | 72% | 22.2 ms/tok | 1.14 GB (1.15) | 217 tok/s (191) | 22.8 tok/s (19.0) |
| 2 GB | 60 (53) | 86% | 11.2 ms/tok | 1.90 GB (1.89) | 273 tok/s (280) | 33.8 tok/s (30.7) |
| 3 GB | 107 (99) | 93% | 6.2 ms/tok | 2.92 GB (2.98) | 273 tok/s (287) | 40.2 tok/s (37.1) |
| 4 GB | 154 (146) | 96% | 3.1 ms/tok | 3.90 GB (3.85) | 273 tok/s (295) | 61.3 tok/s (61.0) |
| 6 GB | 247 (240) | 99.8% | 0.2 ms/tok | 5.23 GB (5.77) | 339 tok/s (384) | 82.4 tok/s (84.9) |
| device (24 GiB = 25.8 GB) | 256 | all | – | 5.76 GB (5.93) | 1,095 tok/s (1,081) | 197.7 tok/s (198.6) |

All 256 loaded from the packed dir is exact (perplexity 67.8979 / batched 72.7078).

### On-demand experts

How it works: [On-demand experts](README.md#on-demand-experts).

Mac test, 2026-10-15 (the trit pack as the remote, 30 ms per fetch, 4 fill threads, 64 slots):

| Phase | Result |
|---|---|
| Open | 0.3 s |
| 1st / 2nd / 3rd prompt | 2.1 / 4.8 / 11.6 tok/s decode while filling (3667, 5163, 5765 of 6144 records after each) |
| Fill | complete ~59 s after open, 3.96 GB fetched (each record once, no errors), `ironbark verify`: intact |
| Output | Same text as the local pack (64-token Rust reply) |

### On-demand embedding

How it works: [On-demand embedding](README.md#on-demand-embedding).

Mac test, 2026-10-16 (M1 Max, 2 GB budget, 30 ms per fetch, 4 expert fill threads, embed fill off; `ironbark gen --fetch-from`):

| First query | `embed.bin` | Download before | Embed fetched | Prefill | Decode |
|---|---|---|---|---|---|
| 887-token Rust prompt | shipped | 435 MB | – | 58.2 / 59.1 s | 15.6 tok/s |
| 887-token Rust prompt | on demand | 268 MB | 170 chunks, 11 MB | 59.8 s | 15.6 tok/s |
| 12-token question | shipped | 435 MB | – | 15.1 / 15.4 s | 1.1 tok/s |
| 12-token question | on demand | 268 MB | 45 chunks, 2.9 MB | 15.1 / 16.0 s | 1.1 tok/s |

Output is the same as the local pack in every run. The first query's time goes to expert fetches, not the embedding.

## iPhone 17

iPhone 17 (A19, 8 GB RAM, app limit 6.44 GB with the increased-memory-limit entitlement), embedding mapped from `embed.bin`, measured 2026-10-16: 747-token prose and 887-token Rust prompts, 128 decoded, one run per row (device budget: 3 runs). Peak = footprint as iOS counts it, sampled every 1 ms. In brackets: before `embed.bin` (embedding loaded into memory, 2026-10-15).

Trit records (`ironbark pack` default):

| Budget | Experts per layer | Hit prose / Rust | Load wait | Prefill | Decode prose / Rust | Peak |
|---|---|---|---|---|---|---|
| device (3.02 GB) | 129 (119) | 96 / 94% | 6–10 ms/tok | 136–141 tok/s | **30.2–31.0 / 23.1–23.6 tok/s** (23.5–25.1 / 21.2–21.5) | 2.79–2.85 GB |
| 2 GB | 72 (62) | 90 / 88% | 18–24 ms/tok | 141–143 tok/s | **17.7 / 16.2 tok/s** (16.5 / 15.5) | 1.78–1.80 GB |
| 1.5 GB | 44 (34) | 85 / 83% | 30–34 ms/tok | 127–130 tok/s | **13.7 / 12.8 tok/s** (12.6 / 11.4) | 1.27–1.29 GB |

2-bit records (`--two-bit`):

| Budget | Experts per layer | Hit prose / Rust | Load wait | Prefill | Decode prose / Rust | Peak |
|---|---|---|---|---|---|---|
| device (3.02 GB) | 108 (99) | 94 / 92% | 12–16 ms/tok | 128–132 tok/s | **21.7–21.9 / 19.5–19.7 tok/s** (21.2 / 18.3) | 2.79–2.86 GB |
| 2 GB | 60 (52) | 88 / 86% | 28–32 ms/tok | 131–133 tok/s | **14.7 / 13.7 tok/s** (12.3 / 13.1) | 1.77–1.78 GB |
| 1.5 GB | 37 (28) | 83 / 80% | 44–49 ms/tok | 114–118 tok/s | **11.3 / 10.7 tok/s** (10.7 / 9.6) | 1.28–1.30 GB |

Mapping the embedding frees ~0.19 GB of the budget for experts (+8 to +10 per layer), so fewer loads and less waiting at the same peak. Output text is the same as on the Mac.

Lookahead and FlashHead on the iPhone (trit, decode tok/s prose / Rust, 128 tokens, `-lookahead 1,2,3 -flash`, before `embed.bin`, 2026-10-15):

| Budget | Lookahead 1 (default) | 2 | 3 | Lookahead 1 + FlashHead |
|---|---|---|---|---|
| device (119 per layer) | **29.3 / 26.5** | 29.3 / 25.8 | 27.0 / 22.0 | 31.8 / 24.0 |
| 2 GB (62) | **16.6 / 15.2** | 15.1 / 14.5 | 13.7 / 12.5 | 19.0 / 15.0 |
| 1.5 GB (34) | 11.9 / 11.1 | – | – | 14.2 / – |

Lookahead 1 stays the default: more layers ahead read more (load wait 7.8 → 9.6 ms per token at the device budget) without more hits. FlashHead (`Session::set_flash_head`, off by default) changed the prose text after 50 characters, so its prose speed is not comparable; on the Rust prompt (same text) it was slower (24.0 vs 26.5). On the Mac it matched the exact head on Rust but missed the exact argmax on 2 of the first 40 prose tokens (the right token's cluster ranked ~1290th of 4748, outside the 512 probed).

Phone vs M1 Max: GPU round trip 200–277 µs (two runs) vs 225 µs; uncached expert reads 1.0 / 1.6 GB/s (1 / 8 threads) vs 1.9 / 6.1 GB/s. Load wait is ~40% of a token at 2 GB; the rest is A19 compute.
