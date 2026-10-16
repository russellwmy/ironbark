# DR-0001: Save RAM with a ternary MoE streamed by ironbark

| Field | Value |
|---|---|
| Status | Accepted (2026-10-14) |
| Date | 2026-10-14 |
| Deciders | project owner |
| Consulted | – |
| Related | [README](../../README.md) (design, raw numbers) |

## Table of Contents

- [Context](#context)
- [Decision drivers](#decision-drivers)
- [Options considered](#options-considered)
- [Decision matrix](#decision-matrix)
- [Decision](#decision)
- [Target model: maple-preview](#target-model-maple-preview)
- [Lessons from DeepGrove's inference write-up](#lessons-from-deepgroves-inference-write-up)
- [Findings from streaming Maple on MLX](#findings-from-streaming-maple-on-mlx)
- [Consequences](#consequences)
- [Follow-ups](#follow-ups)
- [References](#references)

## Context

The goal is **lowest RAM at usable speed** for a strong local model. Two levers exist and they multiply:

| Lever | Cuts | Example |
|---|---|---|
| Fewer bits (trained) | every byte: RAM and bytes per token | ternary: ~2 bits per weight |
| Sparsity (MoE) | weights touched per token | 35B-A3B: ~3B active |

A MoE can run below its own size by keeping cold experts on SSD (ironbark). Training it at low bits shrinks each expert, so every miss costs fewer bytes.

```
              RAM ▲
         (small)  │   ternary MoE + ironbark ●   ← target
                  │                          MoE Q4 ●
         (large)  │                                      bf16 MoE ●
                  └────────────────────────────────────────────▶ speed
```

## Decision drivers

| # | Driver | Weight (1–3) |
|---|---|---|
| D1 | RAM at usable speed (≥ 10 tok/s decode) | 3 |
| D2 | Quality | 2 |
| D3 | Available now (no training budget) | 2 |
| D4 | Fits ironbark's streaming design | 1 |

## Options considered

### Option A: Q4 MoE + ironbark streaming (e.g. 35B-A3B)

- Pros: 30B-class quality; RAM set by cache size.
- Cons: 9–20 MB experts make misses expensive; a long tail of rarely used experts keeps the miss rate high at small caches; ironbark needs a Q4 format.

### Option B: Ternary MoE + ironbark streaming

- Pros: experts under 1 MB, so misses are ~10x cheaper; RAM and speed both improve.
- Cons: needs a ternary-trained MoE (training is GPU-days) — unless one is released.

### Option C: Do nothing (bf16 MoE streaming)

- Cons: bf16 experts are several MiB each, so small caches are SSD-bandwidth-bound.

## Decision matrix

| Option | D1 (×3) | D2 (×2) | D3 (×2) | D4 (×1) | Total |
|---|---|---|---|---|---|
| A: Q4 MoE + streaming | 3 | 4 | 4 | 4 | **29** |
| B: Ternary MoE + streaming | 5 | 3 | 4 | 5 | **34** |
| C: Do nothing | 1 | 4 | 5 | 5 | **26** |

D3 for B is 4 because a ternary MoE exists: maple-preview (below).

## Decision

We choose **Option B** and build it on **deepgrove/maple-preview**, run on the GPU through MLX.

## Target model: maple-preview

Verified 2026-10-14 by reading rows of the checkpoint: every expert and attention row holds exactly three values {−s, 0, +s} (one scale per row); the router is full precision. Packing to 2 bits per weight is **lossless**.

| Item | Value |
|---|---|
| Size | 20B total, ~1B active, MIT license |
| Layers / hidden | 24 / 2048 |
| Experts | 256 per layer, top-8, expert FFN 512, no shared expert |
| Attention | 16 q heads, 4 kv heads, head_dim 128, per-head QK norm, partial RoPE (64 of 128 dims), sliding window 512 on 3 of 4 layers, no RoPE on global layers |
| MLP | SwiGLU with clamps: gate ≤ 7, up in [−7, 7] |
| Vendor claims | 5.31 GB checkpoint, 218 tok/s on an M4 Mac mini |

```
 maple-preview-2bit-mlx
        │ ironbark pack
        ▼
 experts.bin: 24 × 256 records [codes (trit, or 2-bit)][row scales]: ~4.0 GB trit / ~4.9 GB 2-bit
        │ Store streams records from SSD into a page-aligned slab
        ▼
 Metal kernels read slab[slot] directly; attention, LM head stay resident; embedding is memory-mapped (embed.bin)
```

## Lessons from DeepGrove's inference write-up

Source: "Running a 20B Language Model at 218 tokens/s on a Mac mini" (deepgrove.ai/preview, 2026-08-04). Their runtime: MLX on the GPU, base M4 Mac mini (16 GB, 120 GB/s), everything resident, no streaming.

Their bytes read per decoded token (640-token context):

| Bucket | Per token |
|---|---|
| Ternary layers (QKV, O, 8 experts) × 24 | ~240.6 MB |
| LM head, FlashHead (4-bit) | ~24.3 MB (full 4-bit head: 175 MB) |
| KV cache, router, norms | ~52 MB |
| Total | ~317 MB → 2.64 ms at 120 GB/s → 379 tok/s ceiling; measured 218 |

| # | Lesson | ironbark |
|---|---|---|
| L1 | **The LM head dominates.** 151,936 × 2048 rows; bf16 = 622 MB read per token | Done: 4-bit head |
| L2 | **FlashHead**: cluster head rows offline (4,748 clusters of 32), score centroids, compute logits only in the top-512 clusters. 169.6 → 218 tok/s; approximate | Done: opt-in (`Session::set_flash_head`), approximate |
| L3 | Embedding stored 4-bit; only one row is read per token | Done: 4-bit embedding |
| L4 | Their unshipped "row-scale expert QMV" is **our record format** (codes + one scale per row). Their production MLX path uses affine groups of 128 (2.25 bits) | Kept, plus trit rows (default: ~1.6 bits/weight, 19% smaller than 2-bit) |
| L5 | **Masked-multiply kernel**: codes stored as w+1 (0,1,2); activations pre-scaled per lane so `x · (byte & mask)` needs no shift; subtract Σx once per row | Done: the 2-bit kernel works this way; trit rows decode to the same codes |
| L6 | Router stays bf16 with fp32 scores: routing is discrete, quantizing it changes which experts fire | Done |
| L7 | Fuse launches: QKV into one matvec, up + gate, the router into one kernel | Done: QKV one matvec, up + gate one kernel, router 2 launches |
| L8 | Batched prefill (1,075 tok/s) | Done: chunked prefill |
| L9 | Speculative decoding is costly for sparse MoE: draft tokens route to different experts, so verification reads more experts | Do not plan on it for streaming |

## Findings from streaming Maple on MLX

Current numbers are in the [README](../../README.md#results).

| # | Finding |
|---|---|
| S1 | RAM goal met: streaming cuts GPU memory to a fraction of the resident model with identical output |
| S2 | Exact streaming needs the router's choice on the CPU every layer; 24 GPU round trips per token cap decode well below resident speed, even when almost every expert is cached |
| S3 | Prose has better expert locality than code: code misses more and streams more bytes at the same cache size |
| S4 | Lookahead prefetch on the same readback raises the hit rate and recovers part of the loss, at the cost of extra reads |
| S5 | Routing around absent experts (substitute or drop) is fast but breaks quality: Maple is very sensitive to expert choice |
| S6 | The round-trip cost is the GPU driver, not MLX: in raw Metal, a CPU wait per command buffer and an `MTLSharedEvent` signal/wait inside one command buffer cost about the same. A GPU kernel spinning on CPU-written shared memory never sees the write mid-pass |

## Consequences

| Type | Item |
|---|---|
| Good | Lossless 2-bit packing: ~8x smaller experts than bf16, no quality loss vs the checkpoint |
| Good | Each miss costs under 1 MB, so streaming at small caches is practical |
| Good | Sliding-window layers cap most of the KV cache at 512 tokens |
| Bad | Depends on one preview model; vendor notes weak agentic performance |
| Bad | Exact streaming with a partial cache is capped by the per-layer readback (S2, S6) |
| Risk | Code and other low-locality text stream more bytes (S3) → mitigation: lookahead, larger cache when RAM allows |

## Follow-ups

- [x] 2-bit ternary format (pack, kernels, resident attention)
- [x] Trit records, the `ironbark pack` default (L4)
- [x] Fused QKV, up + gate, router (L7); chunked prefill (L8)
- [x] Masked-multiply code encoding (L5)
- [x] Maple architecture (sliding window, partial RoPE, NoPE, clamps), bit-exact with the reference
- [x] 4-bit LM head and embedding (L1, L3)
- [x] Bench: tok/s and memory at caches 256 / 128 / 64 / 32 per layer
- [x] FlashHead, opt-in (L2)
- [x] Embedding memory-mapped from `embed.bin`: 0.19 GB lower peak, same speed (M1 Max, 2026-10-16). Trit attention rows were tried and dropped: 12 MB smaller, 3% slower decode with all experts resident
- [ ] Per-domain expert histogram on Maple (256 experts): do domains share a small hot set?

## References

- deepgrove/maple-preview: https://huggingface.co/deepgrove/maple-preview
- DeepGrove inference write-up: https://deepgrove.ai/preview/index.html (linked as deepgrove.ai/maple-inference)
- FlashHead (Tranheden et al., arXiv 2603.14591, 2026)
- MoTE: Mixture of Ternary Experts (arXiv 2506.14435)
- BitNet b1.58 (Ma et al., 2024)
