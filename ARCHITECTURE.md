# Architecture

How ironbark runs Maple (a 20B ternary MoE) on Apple Silicon with little RAM. For commands and measured numbers see [README.md](README.md); for why see [DR-0001](docs/decisions/0001-ternary-moe-streaming.md).

## Table of Contents

- [Big picture](#big-picture)
- [Repository layout](#repository-layout)
- [Crates and dependencies](#crates-and-dependencies)
- [Model on disk](#model-on-disk)
- [Opening a session](#opening-a-session)
- [Decode step](#decode-step)
- [Expert store](#expert-store)
- [Prefill](#prefill)
- [Session, chat and tools](#session-chat-and-tools)
- [On-demand experts](#on-demand-experts)
- [C API and iOS app](#c-api-and-ios-app)
- [Where the time goes](#where-the-time-goes)
- [Rules the code keeps](#rules-the-code-keeps)
- [Where to start reading](#where-to-start-reading)

## Big picture

Maple has 24 layers with 256 experts each, and each token uses 8 of them per layer. All experts take ~4–5 GB, but one token touches only ~1B parameters. ironbark keeps everything except the experts and the embedding in memory (~0.27 GB), maps the embedding from its file (one row read per token), and keeps a per-layer cache of experts that is as big as the memory budget allows. A miss is read from SSD.

```
            ┌──────────── RAM (budget, e.g. 3 GB on an 8 GB iPhone) ────────────┐
            │  weights 0.27 GB   KV cache   runtime   expert slab (N per layer) │
            └─────────────────────────────────────▲─────────────────────────────┘
                                                  │ miss: pread 630 KiB record
 SSD:  model.safetensors   embed.bin (mapped)   experts.bin (6,144 records, 4.0 GB trit)
```

| Device | Experts per layer | Mode |
|---|---|---|
| 8 GB iPhone (3 GB budget) | ~119 of 256 | streamed |
| 6 GB iPhone | ~76 | streamed |
| 16 GB+ Mac | 256 | all loaded once, no streaming |

## Repository layout

```
ironbark/
├── crates/
│   ├── ironbark/           library: Session, chat, tools
│   ├── ironbark-mlx/       model, Metal kernels, prefill, plan, pack
│   ├── ironbark-experts/   expert store: slab, loaders, LRU, pinning
│   ├── ironbark-fetch/     fills experts.bin / embed.bin from a remote copy
│   ├── ironbark-ffi/       C API (include/ironbark.h)
│   └── ironbark-cli/       `ironbark` binary: pack, verify, gen, ppl, profile
├── ios/                    IronbarkBench app (SwiftUI, xcodegen)
├── vendor/mlx-sys/         mlx-sys 0.6 patched for iOS (IRONBARK.md)
└── docs/decisions/         decision records
```

## Crates and dependencies

```
 ironbark-cli ─────────────────────┐
 ironbark-ffi ──▶ ironbark (lib) ──┼▶ ironbark-mlx ──▶ mlx-rs (MLX 0.32, Metal)
    ├──────────────────────────────┘      └──▶ ironbark-experts
    └──▶ ironbark-fetch ──▶ ironbark-experts (Records trait, layout, hash only)
```

| Crate | Owns | Key types |
|---|---|---|
| `ironbark` | The public API: open sized to the device, chat turns, tool calls | `Session`, `Options`, `Reply`, `Message`, `parse` |
| `ironbark-mlx` | Everything on the GPU and the file formats | `Model`, `Plan`, `Config`, `pack`, `KvSnapshot` |
| `ironbark-experts` | Expert memory and loading, no GPU code | `Store`, `Slab`, `Layout`, `Records` |
| `ironbark-fetch` | A sparse `experts.bin` that fills from a server | `Sparse`, `Fetch` |
| `ironbark-ffi` | C functions returning JSON, for Swift | `ironbark_open`, `ironbark_send`, … |
| `ironbark-cli` | Developer tools | `ironbark pack / verify / gen / ppl / profile` |

`ironbark-experts` knows nothing about MLX: it hands out pinned slots of a byte slab. `ironbark-mlx` wraps that slab as one MLX array, so kernels read the slots in place.

## Model on disk

`ironbark pack` turns DeepGrove's MLX checkpoint into a streaming dir. That dir is what goes on the phone.

| File | Content |
|---|---|
| `model.safetensors` | Everything except experts and embedding: attention, routers, norms, LM head, FlashHead (0.27 GB) |
| `embed.bin` | Token embedding, 4-bit: codes, scales, biases, each region whole 16 KiB pages (0.18 GB). Memory-mapped and wrapped as MLX arrays without a copy: rows are paged in as tokens use them and do not count as app footprint |
| `experts.bin` | One record per (layer, expert), no header |
| `experts.json` | `format` (trit / 2bit), `layers`, `experts`, `record_bytes`, `stride`, `version` |
| `manifest.json` | Size and hash of every file, and one hash per record |
| `model.safetensors.index.json` | Tensor → file map for `model.safetensors` |
| `config.json`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `added_tokens.json`, `chat_template.jinja`, `LICENSE`, `README.md` | Copied from the checkpoint (MIT asks for the notice with every copy) |

```
experts.bin    record (l, e) at offset (l · 256 + e) · stride       stride = 640 KiB trit / 784 KiB 2-bit (page multiple)

┌──────────────────── record (645,120 B trit / 2-bit is 23% larger) ────────────────────┬─ pad ─┐
│ up + gate codes (2F × H) │ down codes (H × F) │ α up, α gate (bf16) │ α down (bf16)     │       │
└───────────────────────────────────────────────────────────────────────────────────────┴───────┘
```

- A fixed stride means no index lookup: one `pread` (or one HTTP Range request) per expert.
- Page-aligned records allow `F_NOCACHE` reads straight into page-aligned slots.
- Ternary rows keep one scale α per row (1.64 bits per weight trit, 2.01 with 2-bit codes), not MLX's per-group scale and bias (2.25).
- Trit records pack 16 weights in 26 bits; kernels decode them to the same codes, so results are bit-identical to 2-bit.

## Opening a session

```
Session::open(dir, Options)
  │
  ├─ Hardware::detect()        RAM, iOS app limit
  ├─ Plan::new()               budget → expert slots per layer
  ├─ set MLX buffer cache limit (0 or 256 MB)
  ├─ Model::load_streaming()   or load_records() for a fetching source
  │     ├─ Store::open_with()  slab + loader threads
  │     ├─ wrap_slab()         slab as one MLX array (no copy)
  │     └─ load_with()         layers evaluated one by one (low load peak)
  └─ Tokenizer
```

The plan (`ironbark-mlx/src/plan.rs`):

```
budget = min(RAM · 3/8, app limit / 2)
peak   = weights + runtime (100 MB) + KV(context 4096) + prefill (250 MB) + buffer cache
         + slots · (24 · stride + 2 MB)
slots  = what is left / (24 · stride + 2 MB), clamped to [16, 256]
```

| Slots | What `load_experts` builds |
|---|---|
| < 256 | Streaming store; decode reads router choices back per layer |
| 256, trit | Store preloaded once; a GPU slot table maps router choice → slot, no readback |
| 256, 2-bit | Resident tensors in the checkpoint layout (fastest `gather_qmm`), no store |

## Decode step

`Model::step_lazy` runs one token. The residual is carried as `(h, hn)` so each add + norm is a single kernel.

```
token ─▶ embed ─▶ 24 × [ QKV (ternary qmv) ─▶ QK norm + RoPE ─▶ KV cache ─▶ attention ─▶ O
                         add + norm ─▶ router (fp32, top-8)
                         ─▶ 8 experts: up+gate+SwiGLU ─▶ down ─▶ weighted sum + add + next norm ]
      ─▶ LM head ─▶ logits
```

| Piece | Where | Note |
|---|---|---|
| Ternary matvec | `row.rs` | Custom Metal via MLX's `metal_kernel` (`kernel.rs`) |
| Fused norms, RoPE, router | `fused.rs` | ~300 → ~13 launches per layer |
| Attention | MLX `scaled_dot_product_attention` | KV: full layers grow by 256, sliding layers are a 512-slot ring |
| LM head | MLX `quantized_matmul` (4-bit) | Or FlashHead (`flash.rs`, opt-in, approximate) |

**Pipelining.** The caller queues step t+1 from the *lazy* argmax of step t, then reads token t back. The GPU never waits for the CPU to read a token.

**Streamed layers.** The router's choice must reach the CPU before experts can run:

```
GPU: … router ─┐                                         ┌─▶ experts(slab[slots]) ─▶ next layer …
               │ eval (readback, ~200–490 µs)             │
CPU:           └─▶ ids ─▶ Store.request / wait ─▶ slots ─┘
                         (+ predicted ids of the next layer → Store.prefetch)
```

- `lookahead` (default 1): the same readback carries the next layer's predicted experts (its router on the current residual), so loaders start early.
- The last layer's experts stay pinned until the next step's first readback (`held`), so the logits kernel can still read them.

## Expert store

`ironbark-experts` is an event-driven cache over one page-aligned slab.

```
compute thread                         loader threads (8)
request(l, e) ──demand──▶ ┐            take demand jobs first, else prefetch
prefetch(l, e) ─prefetch─▶ ┼──▶         read the record into its slot (Records: pread, or fetch)
wait(l, e) ◀── loaded[slot] ──┘         mark Ready, wake that slot's waiters
  └─▶ ExpertRef (pins the slot; unpins on drop)
```

Slot states (changed only under the store's mutex):

```
Free ──claim──▶ Loading ──loaded──▶ Ready ──evict (no pins, LRU)──▶ Loading (another expert)
               one writer                  any number of readers
```

- Each layer owns its own slots: layers are visited in a cycle, which defeats one global LRU.
- The budget is hard. When every slot of a layer is pinned or loading, a demand load waits and a prefetch is dropped.
- All `unsafe` memory access is in `slab.rs`; the state machine guarantees one writer or many readers per slot.
- The slab is an anonymous mapping, unmapped on drop, so its memory leaves the footprint at once.

## Prefill

`prefill.rs` reads the prompt in chunks of up to 512 tokens with matrix-matrix ops (MLX `quantized_matmul`, `gather_qmm`), the same arithmetic as mlx-lm.

```
chunk ─▶ embed ─▶ 24 × [ QKV qmm ─▶ norm, RoPE ─▶ attention (causal / 512 window) ─▶ O
                         router ─▶ experts sorted by id ─▶ gather_qmm ]
      ─▶ head (last position)
```

With streamed experts, per layer the chunk's distinct experts go through the cache in groups that fit: pin, copy the records into MLX's layout, run, unpin. Each group is forced onto the same MLX kernel the resident call would use, so results equal resident prefill bit for bit at any cache size. Afterwards the KV cache has the same layout as after token-by-token decode.

## Session, chat and tools

`Session` (`crates/ironbark/src/lib.rs`) keeps one conversation in the KV cache and writes Maple's chat format.

```
set_system(system, tools)  ─▶ system turn (tools as JSON, call format)
send([Message…])           ─▶ close previous turn ─▶ user / tool turns ─▶ "<|im_start|>assistant\n<think>\n"
next_piece() …             ─▶ reasoning </think> answer | <tool_call>{…}</tool_call>
parse(reply)               ─▶ thinking, text, calls
```

| Feature | How |
|---|---|
| Messages | `Message::User`, `Assistant` (reasoning dropped, as in the template), `ToolResults` |
| Rebuild | `reset()` then `send(all messages)`, e.g. after the app reopened the model |
| System turn cache | Prefilled once; each new conversation restores its KV (`Model::snapshot` / `restore`). MLX arrays are copy-on-write, so a snapshot only holds references. |
| Thinking off | The prompt ends with an empty `<think>\n\n</think>\n\n`, and the reply text starts with `</think>` so `parse` works unchanged. Replies to tool results still reason. |
| Dropped reply | The next `send` closes it with `<|im_end|>`, so the cache always matches the transcript |
| Stop tokens | `<|im_end|>`, `<|endoftext|>` |

The app runs the tools; the library only writes turns and parses calls.

## On-demand experts

`ironbark-fetch` builds `experts.bin` from a server, separately from the engine. The engine reads through the `Records` trait and does not know where records come from.

```
engine loader: read record i
  present bit set? ──yes──▶ pread local experts.bin
      │ no
      ▼
  Fetch::fetch(i · stride, record_bytes)  (HTTP Range GET via the app's callback)
  hash == manifest hash? ──no──▶ 4 tries, then fail
      │ yes
      ▼
  serve ─▶ pwrite + fsync ─▶ set bit i in experts.present
```

Background fill threads fetch the remaining records and hold off while the engine is waiting on a fetch.

`embed.bin` can fill the same way (`Sparse::open_embed`, 64 KiB chunks with hashes in the manifest). The engine sees it through the `Chunks` trait: before prefill or a decode step reads a token's row, `Model::ensure_rows` makes the row's three chunks present (missing ones fetched in parallel), then the GPU reads the mapped file. While it is incomplete, decode reads each token back first, so the pipelined step waits one round trip.

## C API and iOS app

`ironbark-ffi` is a static library with C functions. Functions that report something return JSON that the caller frees with `ironbark_free_string`.

| Group | Functions |
|---|---|
| Open / close | `ironbark_open`, `ironbark_open_fetcher`, `ironbark_open_fetchers` (experts and embed), `ironbark_close` |
| Chat | `ironbark_set_system`, `ironbark_set_thinking`, `ironbark_send`, `ironbark_next`, `ironbark_parse`, `ironbark_stats` |
| Raw turns | `ironbark_start`, `ironbark_next` |
| Bench | `ironbark_generate`, `ironbark_probe`, `ironbark_profile`, `ironbark_tool_check`, `ironbark_set_lookahead`, `ironbark_set_flash_head` |
| Fetch | `ironbark_fetcher_open`, `ironbark_embed_fetcher_open`; `ironbark_fetcher_start / progress / stop / close` |

The iOS app (`ios/IronbarkBench`):

| File | Role |
|---|---|
| `App.swift` | Chat (tool loop, thinking switch, off-screen handling) and Bench (runs, profile, per-run peak) |
| `Tools.swift` | `calculator` (on device), `get_weather` (Open-Meteo), `wikipedia_search` |
| `Markdown.swift` | Block Markdown renderer (headings, lists, code, tables, quotes) |

```
Send ─▶ engine queue ─▶ ironbark_send ─▶ ironbark_next … (stream to the UI)
            └─ calls? ─▶ Tools.run ─▶ ironbark_send([tool results]) … up to 3 rounds
```

- All library calls run on one serial queue. Only one model is in memory: the bench frees the chat's model first.
- iOS runs no GPU work off screen. A reply stops at its next token (unless it runs as a continued-processing task, see `ios/project.yml`), and the model is freed after 20 s unless the app comes back.
- `ios/build.sh` builds the Rust library for `aarch64-apple-ios`, copies MLX's metallib, runs xcodegen and xcodebuild; `--install` installs on the iPhone.

## Where the time goes

iPhone 17, device budget, ~23–31 tok/s (~32–43 ms per token):

```
├─ GPU compute            ~20 ms
├─ expert load wait       6–10 ms   (18–34 ms at 2 / 1.5 GB)
└─ 24 router readbacks    6–12 ms   (24 × ~200–490 µs)
```

| Limit | Why it stays |
|---|---|
| Readback round trip | Metal's CPU → GPU resume costs ~200 µs however it is expressed (new command buffer, shared event); the GPU does not see CPU writes mid-kernel |
| Skipping readbacks on a guess | Only 64–74% of layers have all 8 experts cached at 119 per layer, so redoing misses costs more than it saves |
| Load wait | SSD reads ~1.0 / 1.6 GB/s (1 / 8 threads) on the phone; a bigger budget means fewer misses |

The fast path is "every expert fits": no readback, ~172–199 tok/s on an M1 Max.

## Rules the code keeps

| Rule | Where it shows |
|---|---|
| Bit-exact with DeepGrove's reference | Kernels follow MLX 0.32's accumulation order and bf16 rounding; unit tests compare fused against unfused ops; perplexity is checked |
| `unsafe` confined | Slab memory in `slab.rs` (called from `store.rs`); `F_NOCACHE` in `layout.rs`; mach / sysctl in `device.rs`; the MLX C API in `kernel.rs`; C entry points in `ironbark-ffi` |
| Lock order | Store: `inner`, then `jobs`; never held during a read |
| No polling | `thread::sleep` is banned by clippy; waits use condvars or channels |
| Memory is the budget | The plan's peak model is checked against measured footprints; slabs are unmapped on drop |

See [STYLE.md](STYLE.md) for the full style and lint rules.

## Where to start reading

| To understand | Read |
|---|---|
| The public API | `crates/ironbark/src/lib.rs` (`Session`) |
| One decode step | `crates/ironbark-mlx/src/model.rs` (`step_lazy`) |
| The kernels | `crates/ironbark-mlx/src/row.rs`, `fused.rs` |
| Expert caching | `crates/ironbark-experts/src/store.rs` |
| Memory sizing | `crates/ironbark-mlx/src/plan.rs` |
| The file format | `crates/ironbark-mlx/src/pack.rs`, `crates/ironbark-experts/src/layout.rs` |
| Chat and tools | `crates/ironbark/src/tools.rs`, `ios/IronbarkBench/App.swift` |
