# DR-0005: Keep lookahead prefetch for Qwen; do not adopt a trained prerouter

| Field | Value |
|---|---|
| Status | Accepted (2026-11-09) |
| Date | 2026-11-09 |
| Deciders | project owner |
| Consulted | – |
| Related | [DR-0001](0001-ternary-moe-streaming.md); [edge0](https://github.com/Edge0-AI/edge0) and its report ([arXiv 2609.18063](https://arxiv.org/abs/2609.18063)); commits `c2bc945` (route stats, `--lookahead-k`), `b7dc13a` (Qwen `embed.bin`) |

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

edge0 runs Qwen3.6-35B-A3B at 4 bits from SSD. A small trained head per layer predicts the
next layer's experts one token ahead, and the prediction **is** the routing (K=4, not the
trained 8). A LoRA distilled from the fp16 model wins back the quality. It reports +80% decode
against on-demand loading with **no** prediction.

ironbark already predicts: `lookahead` 1 runs the next layer's router on the current residual
and prefetches its top-k, in the same readback as this layer's choice. On the iPhone 17, Qwen
keeps 31 experts per layer in a 3.02 GB budget, hits 70–80%, and waits for loads in 33–40% of
each token. The question: would a trained prerouter cut that wait, or the RAM?

```
 ironbark:  layer l readback ─► chosen(l) ─► load/wait
                              └► router(l+1) on h_l ─► prefetch top-k(l+1)   (no training)
 edge0:     head(l) at token t ─► experts(l+1) at t+1 ─► used as the routing  (trained + LoRA)
```

Measured on the general ternary pack (`gen --route-stats`, M1 Max, 256 decoded tokens):

| Guess of the experts a layer runs | Prose | Rust |
|---|---|---|
| Lookahead, top 8 (now) | 71% | 61% |
| Lookahead, top 12 / 16 / 32 | 83 / 88 / 96% | 73 / 80 / 91% |
| Previous token's experts, same layer | 30% | 20% |

Of the cache misses, the lookahead catches 56–63%, and 27–34% of its prefetches go unused.
Prefetching the top 16 cuts demand loads by 33–43% but reads 1.6–2.1× the bytes. Output is the
same at every setting.

## Decision drivers

| # | Driver | Weight (1–3) |
|---|---|---|
| D1 | Quality: output unchanged, experts stay ternary | 3 |
| D2 | iPhone speed and battery (load wait, bytes read) | 2 |
| D3 | Effort and new parts to maintain (training, file formats) | 1 |

## Options considered

### Option A: edge0's prerouter (prediction is the routing, K=4, LoRA)

- Pros:
  - No per-layer readback and no demand loads by construction
  - K=4 halves expert reads (edge0: 3.3 → 6.4 tok/s from K=8 to K=4)
- Cons:
  - Changes the model; edge0 needs a LoRA to recover, and still loses 3.9 points on average
  - LoRA weights next to ternary experts break the fully-ternary rule
  - Training pipeline in ironwood plus a new head and adapter format in ironbark
  - Its +80% is against no prediction; ironbark's baseline already has lookahead
- Cost / effort: L

### Option B: Trained head for prefetch only (routing unchanged)

- Pros:
  - Output unchanged; heads are small dense layers, not experts
- Cons:
  - Ceiling is small: removes 19–61 demand loads per token and ~20% of reads at best
  - Unknown whether it beats 71% / 61% at top 8; edge0 reports no recall figures and calls
    cross-token prediction "information-limited"
  - Needs residual dumps, training, and a head file in packs
- Cost / effort: M

### Option C: Untrained wider prefetch (`--lookahead-k 12`)

- Pros:
  - Output unchanged; no training; already in the CLI
  - Top 12 catches 83% / 73%, demand loads −23 to −25%
- Cons:
  - ~1.3–1.5× the bytes read; on the phone (~1.6 GB/s reads, heat-limited) this may cost more
    than it saves
- Cost / effort: S (FFI knob and a phone run)

### Option D: Keep lookahead 1 as is (do nothing)

- Pros:
  - Nothing to build; measured and stable
- Cons:
  - Load wait stays at 33–40% of each token on the phone
- Cost / effort: none

## Decision matrix

Score 1 (poor) to 5 (great) per driver.

| Option | D1 (×3) | D2 (×2) | D3 (×1) | Total |
|---|---|---|---|---|
| A: edge0 prerouter | 1 | 5 | 1 | 14 |
| B: trained prefetch head | 5 | 3 | 2 | 23 |
| C: wider prefetch | 5 | 3 (unmeasured) | 5 | 26 |
| D: keep lookahead | 5 | 2 | 5 | 24 |

## Decision

We choose **Option D now, with C as the next trial**. The lookahead already does most of what
a predictor can do here, and the parts of edge0 that give its speed (prediction as routing,
K=4, LoRA) change the model, which D1 rules out. A trained prefetch head (B) is not worth its
pipeline until C shows how much the phone gains from more hits per byte.

For RAM, the free saving was taken instead: Qwen packs now memory-map the embedding from
`embed.bin` (`b7dc13a`), 0.29 GB less footprint with the same output and speed (peak 3.42 →
3.13 GB at 31 experts per layer on the Mac; at a 3.02 GB budget the plan fits ~9 more experts
per layer). Both release packs were repacked with it.

## Consequences

| Type | Item |
|---|---|
| Good | Output and quality unchanged; experts stay fully ternary |
| Good | Route stats and `--lookahead-k` stay in the CLI to measure future predictors |
| Good | Qwen footprint 0.29 GB lower (`embed.bin`), spent on more expert slots at the same budget |
| Bad | Load wait on the phone is unchanged until C is tried |
| Bad | edge0's K=4 speed (half the reads) is given up |
| Risk | Mac timings were taken with other builds running → only counts were used; phone speed must be measured on the phone |

## Follow-ups

- [ ] iPhone: plan and speed with the repacked Qwen packs (`embed.bin`), at 3.02 GB
- [ ] iPhone: `lookahead_k` 12 through the FFI; decode speed, bytes read and battery against 8
- [ ] Revisit B if C helps on the phone but reads cost too much: a head trained on residual dumps, scored by recall at top 8
- [ ] Revisit A only if the fully-ternary rule changes

## References

- edge0 README, `docs/prerouter.md`, report §3–5 (prerouter, Table 3–4)
- Route stats logs (2026-11-09, session scratchpad; numbers recorded here)
- [BENCHMARKS.md: Qwen on the Mac](../../BENCHMARKS.md#qwen-on-the-mac) (`embed.bin` table)
