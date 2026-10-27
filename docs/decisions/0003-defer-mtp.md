# DR-0003: Defer MTP speculative decoding for Qwen

| Field | Value |
|---|---|
| Status | Accepted (2026-10-25) |
| Date | 2026-10-25 |
| Deciders | project owner |
| Consulted | – |
| Related | [DR-0001](0001-ternary-moe-streaming.md), commit 222218c (Qwen sampling) |

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

Qwen3.6-35B-A3B has a multi-token prediction (MTP) layer (`mtp_num_hidden_layers: 1` in `config.json`). It drafts the token after next. The model then checks the draft in a 2-token pass. Each kept draft saves one decode step.

ironbark has no MTP today:

| Layer | MTP |
|---|---|
| Model architecture | Yes, 1 layer |
| Our pack (`qwen3.6-tern-full-pack`) | No `mtp.*` weights (dropped at conversion) |
| ironbark engine | No code |

Question: would MTP make decode faster in our design (a ternary MoE with experts streamed from storage, DR-0001)?

```
normal decode:  [step] -> 1 token
MTP cycle:      [draft][check 2 tokens] -> 1 + a tokens    (a = share of drafts kept)
```

### Measured cost of a 2-token pass

On a Mac with `qwen3.6-tern-full-pack`, after a short context. The 2-token prefill stands in for the MTP check.

| Setup | Decode step | 1-token pass | 2-token pass | 2nd token adds |
|---|---|---|---|---|
| All experts loaded | 14.0 ms | 19.2 ms | 23.9 ms | +4.8 ms (+34%) |
| 4.4 GB, cache-aware (like the phone) | 19.1 ms | 20.4 ms | 74.6 ms | +54 ms* |
| 4.4 GB, exact routing (streaming) | 57.8 ms | 58.7 ms | 75.2 ms | +17 ms (+29%) |

\* Multi-token passes use exact routing, so the 2nd token's experts are read from storage. An MTP check could use cache-aware routing too. That should bring it down to about +35%.

### Estimated gain with all experts loaded

Assumes the check costs about 19 ms (decode step + 4.8 ms) and the draft about 2–3 ms (estimated, not measured).

```
speed = (1 + a) / ~22 ms        break-even: a ≈ 0.57
```

| Share of drafts kept (a) | Speed | vs now (71 tok/s) |
|---|---|---|
| 0.5 | ~68 tok/s | −4% |
| 0.6 | ~73 tok/s | +3% |
| 0.7 | ~77 tok/s | +9% |
| 0.8 | ~82 tok/s | +15% |

Full-precision Qwen MTP usually keeps 80–90% of drafts with greedy decoding. Ours is ternary and now samples (commit 222218c), so we expect 50–70%, which is near break-even.

## Decision drivers

| # | Driver | Weight (1–3) |
|---|---|---|
| D1 | Decode speed on the phone (streamed experts) | 3 |
| D2 | Pressure on the expert cache / RAM | 3 |
| D3 | Engine complexity | 2 |
| D4 | Decode speed on the Mac (all experts loaded) | 1 |

## Options considered

### Option A: Add MTP now

Keep the `mtp.*` weights at conversion, load the MTP layer in ironbark, then draft and check every step.

- Pros:
  - Up to +15% on the Mac with all experts loaded, if 80% of drafts are kept.
  - The model already ships the layer, so there is nothing to train.
- Cons:
  - Each check routes 2 tokens, so up to ~16 experts per layer instead of 8. On the phone this means more cache misses and storage reads.
  - Most layers are DeltaNet, which keeps a running state instead of a KV cache. On a rejected draft that state must be saved and restored. That adds memory traffic every cycle.
  - Sampling needs the speculative-sampling acceptance rule, not a simple top-token match.
  - A ternary MTP layer drafts worse, so fewer drafts are kept. Keeping it in full precision costs RAM.
- Cost / effort: L

### Option B: Do nothing; improve the expert cache instead

Keep one token per step. Put the effort into cache hit rate and the experts kept per layer.

- Pros:
  - Targets the phone's real bottleneck (storage reads, RAM).
  - No new state to roll back, no new weights.
- Cons:
  - No gain on the Mac with all experts loaded.
- Cost / effort: S–M (ongoing work)

## Decision matrix

| Option | D1 (×3) | D2 (×3) | D3 (×2) | D4 (×1) | Total |
|---|---|---|---|---|---|
| A: MTP now | 2 | 1 | 1 | 4 | 17 |
| B: Do nothing / improve the cache | 4 | 4 | 5 | 2 | 36 |

## Decision

We choose **Option B**. MTP does not help where our design is constrained. On the phone, speed depends on expert reads from storage, and MTP doubles the experts each step needs. On the Mac the expected gain is 0–15%, which does not justify the DeltaNet save/restore and the extra weights.

```
phone time per token   ≈ compute + expert misses × read time
MTP                    → fewer steps, but ~2× experts per step → more misses
better expert cache    → fewer misses
```

## Consequences

| Type | Item |
|---|---|
| Good | The engine stays simple: one token per step, no DeltaNet save/restore |
| Good | The expert cache budget is not split with MTP |
| Bad | We give up a possible 0–15% on the Mac with all experts loaded |
| Risk | The share of kept drafts may be higher than we expect → mitigation: measure it before building (follow-up) |

## Follow-ups

- [ ] Revisit if a cheap check shows ≥ 75% of drafts kept: keep `mtp.*` (full precision) at conversion and measure on the Mac with all experts loaded
- [ ] Revisit if a device can hold all experts in RAM (decode limited by compute, not storage)
- [ ] Revisit if a 2-token pass with cache-aware routing costs less than +25%

## References

- Qwen3.6-35B-A3B `config.json`: `mtp_num_hidden_layers`, `mtp_use_dedicated_embeddings`
- [DR-0001](0001-ternary-moe-streaming.md): ternary MoE streamed by ironbark
- Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023): the acceptance rule for sampled drafts
