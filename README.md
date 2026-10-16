# ironbark

Ternary MoE inference on Apple Silicon. The target model is **Maple** (DeepGrove, 20B total / ~1B active, 256 experts top-8, ternary weights), run on the GPU through MLX from Rust. Goal: lowest RAM at usable speed ([DR-0001](docs/decisions/0001-ternary-moe-streaming.md)).

## Table of Contents

- [Workspace](#workspace)
- [Design](#design)
- [Commands](#commands)
- [Usage](#usage)
- [Tool use](#tool-use)
- [On-demand experts](#on-demand-experts)
- [iOS](#ios)
- [Results](#results)
- [License](#license)

Architecture: [ARCHITECTURE.md](ARCHITECTURE.md).

## Workspace

```
ironbark/
├── Cargo.toml          workspace: members, shared deps, profiles
├── crates/
│   ├── ironbark/           library: Session (open sized to the device, chat, tools)
│   ├── ironbark-engine/    Maple decode on the GPU
│   ├── ironbark-experts/   event-driven expert store (SSD streaming), for the RAM work
│   ├── ironbark-fetch/     builds a local experts.bin from a remote copy (separate from the engine)
│   ├── ironbark-ffi/       C API (ironbark.h) for iOS / macOS apps
│   ├── ironbark-cli/       binary `ironbark`: pack, verify, gen, ppl, profile
│   ├── ironbark-mlx/       safe Rust API over MLX
│   └── ironbark-mlx-sys/   mlx-c bindings; builds MLX for macOS and iOS
├── ios/                IronbarkBench: SwiftUI benchmark app (xcodegen)
├── models/             (gitignored) Maple checkpoint and packs, from scripts/setup.sh
├── prompts/            prompt texts for perplexity and benchmarks (prose, rust)
├── scripts/setup.sh    download Maple, build, pack, verify
└── docs/decisions/     decision records
```

```
 ironbark-cli ─────────────────────┐
 ironbark-ffi ──▶ ironbark (lib) ──┼▶ ironbark-engine ──▶ ironbark-mlx (MLX 0.32, Metal)
    ├──────────────────────────────┘         └──▶ ironbark-experts (streamed experts: slab wrapped as one MLX array, no copy)
    └──▶ ironbark-fetch ──▶ ironbark-experts (only the Records trait, layout, hash)
```

| Crate | What |
|---|---|
| `ironbark` | Library: `Session::open` sized to the device, chat and tool turns, `parse` for tool calls |
| `ironbark-engine` | Loads `deepgrove/maple-preview-2bit-mlx` or a packed dir; one-token decode step on custom Metal kernels (`row.rs` ternary matvec, `fused.rs` norms / RoPE / router, via the MLX C API in `kernel.rs`) plus MLX attention; bit-exact with DeepGrove's `maple.py` |
| `ironbark-experts` | `experts.bin` records streamed from SSD into a page-aligned slab; per-layer LRU cache, loader pool, pinning; reads through a `Records` trait |
| `ironbark-fetch` | Sparse `experts.bin` filled from a remote copy: on-demand records, hash checks, presence bitmap, background fill; runs with or without the engine |
| `ironbark-ffi` | C API (`ironbark.h`), static library for the iOS / macOS apps |
| `ironbark-cli` | `ironbark pack` (experts -> `experts.bin`), `verify`, `gen` (greedy decode, speed, memory), `ppl` (perplexity), `profile`; `--experts DIR` streams experts (`--cache N`, `--budget GB`, `--loaders`, `--lookahead`) |
| `ironbark-mlx` | Safe Rust API over MLX for inference (arrays, ops, fused kernels, eval, compile) |
| `ironbark-mlx-sys` | Bindings to mlx-c; builds MLX v0.32.2 and its Metal kernels for macOS and iOS |

## Design

One token per step; the residual stream is carried as (h, hn) so every add + norm is one kernel:

```
 token ──▶ embed (4-bit) ──▶ 24 × [ RMSNorm ─▶ QKV (ternary) ─▶ QK norm, RoPE (sliding layers only)
                                    ─▶ attention (window 512 on 18 sliding layers, full on 6) ─▶ O (ternary)
                                    RMSNorm ─▶ router (fp32, top-8 of 256)
                                    ─▶ 8 experts: up+gate (ternary) ─▶ clamped SwiGLU ─▶ down (ternary) ]
       ──▶ RMSNorm ──▶ LM head (4-bit) ──▶ logits
```

| Part | Choice | Why |
|---|---|---|
| Ternary weights | Packed 2-bit codes + one scale α per row, read by custom kernels: matvec (QKV, O, gathered down) and fused up+gate+clamped SwiGLU | MLX's affine layout repeats (α, −α) per 128-weight group: 0.54 GB more memory and more bytes per token |
| Embedding, LM head | 4-bit affine, stock MLX | Not ternary |
| Embedding in RAM | Packed models keep it in `embed.bin`, memory-mapped and wrapped as MLX arrays without a copy (`embed.rs`) | One 1 KiB row is read per token; the 175 MB table is clean file-backed memory, not app footprint: 0.19 GB lower peak, same speed |
| Fused kernels | add + RMSNorm; expert weighted sum + add + next RMSNorm; per-head QK norm + RoPE; router GEMV + softmax + top-8 | ~300 → ~13 GPU launches per layer; their reductions follow MLX 0.32, so results stay bit-exact (unit tests compare each against the unfused ops) |
| KV cache | Preallocated, written in place: full layers grow by 256 slots, sliding layers are a 512-slot ring | Same layout as mlx-lm, so attention sums in the same order (a time-ordered window drifted 0.7% in perplexity past 512 tokens) |
| Numerics | bf16 activations; RMSNorm and router in fp32; clamp in bf16 | Bit-exact with the reference (an f32 clamp constant promoted the experts to f32 and moved perplexity 8%) |
| Loading | Each layer is evaluated as it is built | Frees the checkpoint's split tensors early: peak memory = steady-state memory |
| Decode loop | Step t+1 is built from the lazy argmax of step t, then `async_eval` | The GPU never waits for the CPU to read a token |

Streaming experts (`--experts`): per layer, the router's choice is read back on the CPU, the 8 experts are pinned in the store (loaded from `experts.bin` on a miss), and the slab kernels read their slots directly; the slab is wrapped as one MLX array without copying (Metal `bytesNoCopy`). With `--lookahead N` (default 1), the same readback also carries the next N layers' predicted experts (their router on the current residual), which loaders fetch while the GPU works.

```
 router (GPU) ──ids──▶ CPU: Store.request / wait (hit, or pread into a free slot)
                          │ slot numbers (pinned until the next layer's readback)
                          ▼
 up+gate+SwiGLU, down kernels read slab[slot]
```

## Commands

One step from a fresh clone (needs `cargo` and `hf`: `brew install huggingface-cli`):

```bash
scripts/setup.sh              # download to models/maple-2bit-mlx, build, pack to models/maple, verify
scripts/setup.sh --two-bit    # also the 2-bit pack in models/maple-2bit
```

Or step by step:

```bash
cargo build --release
M=$PWD/models/maple-2bit-mlx      # hf download deepgrove/maple-preview-2bit-mlx --local-dir $M
target/release/ironbark gen --model $M --prompt "The capital of Germany is" --gen 128
target/release/ironbark ppl --model $M --text prompts/prose.txt --tokens 400
P=$PWD/models/maple                # a packed model: what the library and the phone use
target/release/ironbark pack --model $M --out $P           # trit experts (4.0 GB) + other weights, tokenizer, manifest
target/release/ironbark verify $P                          # sizes, hashes, every expert record (e.g. after copying)
target/release/ironbark gen --model $P --prompt "..." --experts $P           # sized to this machine
target/release/ironbark gen --model $P --prompt "..." --experts $P --budget 2  # or to a 2 GB footprint
```

To use the library from your own code, see [Usage](#usage).

Checks (toolchain pinned in `rust-toolchain.toml`; `rustfmt.toml`, `clippy.toml`, `deny.toml`):

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --release -- -D warnings
IRONBARK_MODEL=$P cargo test --workspace --release -- --test-threads=1   # without IRONBARK_MODEL the model tests are skipped
cargo deny check
```

The first build compiles MLX from source (about 15 min).

## Usage

The crates are not on crates.io (`publish = false`), so depend on them by path or git. They need Rust 1.99+. iOS needs nothing extra: `ironbark-mlx-sys` builds MLX's kernels for iOS.

```toml
[dependencies]
ironbark = { path = "../ironbark/crates/ironbark" }
ironbark-fetch = { path = "../ironbark/crates/ironbark-fetch" }   # only for on-demand experts
```

```
 Session::open(dir, Options) ──▶ set_system(date, tools) ──▶ chat(msg) ──▶ Reply (iterator of text pieces)
                                                                │ parse(reply).calls not empty
                                                                ▼
                                     app runs the tools ──▶ tool_results(&[..]) ──▶ Reply …
```

`dir` is a packed model (`ironbark pack`, see [Commands](#commands)).

### Chat

```rust
fn main() -> ironbark::Result<()> {
    let mut s = ironbark::Session::open("maple", ironbark::Options::default())?; // sized to the device
    s.set_system(Some("Today is 2026-10-15."), &[]);
    for piece in s.chat("What is the capital of Germany?")? {
        print!("{}", piece?); // streamed as it decodes
    }
    let st = s.stats();
    println!("\n{} tokens, {:.1} tok/s", st.tokens, st.decode_tps);
    Ok(())
}
```

### Options

| Field | Default | What |
|---|---|---|
| `budget` | `None`: 3/8 of RAM, at most half the iOS app limit | Footprint budget in bytes |
| `slots` | `None`: as many as fit the budget | Expert cache slots per layer (256 = all loaded, no streaming) |
| `max_context` | 4096 | Most tokens per conversation; the budget reserves its KV cache |
| `loaders` | 8 | Threads reading experts from storage |
| `records` | `None`: the dir's `experts.bin` (must be complete) | Where expert records come from, e.g. an `ironbark_fetch::Sparse` |

```rust
let opts = ironbark::Options { budget: Some(2_000_000_000), ..Default::default() }; // 2 GB footprint
```

### Tool calls

```rust
let tools = [ironbark::json!({"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})];
s.set_system(Some("Today is 2026-10-15."), &tools); // give the date, see Tool use
s.set_thinking(false);                              // optional: no reasoning on user turns

let mut reply: String = s.chat("Weather in Adelaide?")?.collect::<ironbark::Result<_>>()?;
for _ in 0..3 {                                     // cap the rounds of calls
    let parsed = ironbark::parse(&reply);           // .thinking, .text, .calls, .malformed
    if parsed.calls.is_empty() {
        println!("{}", parsed.text);
        break;
    }
    // run_tool(&ironbark::ToolCall) -> String is your code: call.name, call.arguments (JSON)
    let results: Vec<String> = parsed.calls.iter().map(run_tool).collect();
    let results: Vec<&str> = results.iter().map(String::as_str).collect();
    reply = s.tool_results(&results)?.collect::<ironbark::Result<_>>()?;
}
```

### Conversations

| Call | What |
|---|---|
| `s.reset()` | New conversation; the system turn is restored from its KV snapshot, not read again |
| `s.send(&[Message::User(..), Message::Assistant(..), Message::ToolResults(..)])` | Rebuild a conversation from a transcript (after `reset`) |
| `s.stats()` | Prompt / reply tokens, prefill and decode tok/s, expert hits, footprint |
| `s.set_lookahead(n)` | Layers of experts prefetched ahead (default 1) |

### Experts from a server

The pack ships without `experts.bin`; records come from your server (see [On-demand experts](#on-demand-experts)).

```rust
use std::sync::Arc;

struct Http; // your HTTP client
impl ironbark_fetch::Fetch for Http {
    fn fetch(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        todo!("Range GET bytes offset .. offset + buf.len() of the remote experts.bin")
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::path::Path::new("maple");
    let sparse = ironbark_fetch::Sparse::open(dir, Arc::new(Http))?;
    sparse.start(4, &[]); // 4 background fill threads
    let opts = ironbark::Options { records: Some(sparse.clone()), ..Default::default() };
    let mut s = ironbark::Session::open(dir, opts)?; // missing records are fetched on demand
    // ... chat as above; sparse.progress() gives present / total records
    Ok(())
}
```

## Tool use

Maple's chat template (Qwen/Hermes style) lists tools as JSON in the system turn; the model replies with `<tool_call>` blocks and gets results back as `<tool_response>` blocks. The library writes the turns and parses replies; the app runs the tools (search API keys stay in the app).

```
 set_system(Some("Today is …"), &tools) ──▶ chat("weather in Adelaide?") ──▶ <think>…</think><tool_call>{…}</tool_call>
                                                                                 │ parse(reply).calls
 app runs get_weather ──▶ tool_results(&["{\"temp_c\": 22}"]) ──▶ "It is 22 °C in Adelaide."
```

Code: [Usage → Tool calls](#tool-calls).

| Case (Mac, trit pack, replies ≤ 1024 tokens, 2026-10-16) | Called | Answer |
|---|---|---|
| Weather in Adelaide | `get_weather {"city": "Adelaide"}` | Uses the result |
| 2026 Japanese GP winner | `web_search {"query": "2026 Formula 1 Japanese Grand Prix winner"}` | Uses the result |
| 48213 × 9177, "use the calculator" | `calculator {"expression": "48213 * 9177"}` | 442,450,701 (right) |
| Capital of Germany | none | Berlin, without a tool |

Notes:

- Every reply starts inside `<think>` (the template's generation prompt); `parse` splits reasoning, answer and calls.
- `set_thinking(false)` skips the reasoning of replies to user messages: the prompt ends with an empty `<think>\n\n</think>\n\n` and the reply text starts with that `</think>`, so `parse` works unchanged. Replies to tool results still reason: without it, Maple called the same tool again instead of answering (calculator 3× in a row). Mac, tool cases above: 4 of 4 pass either way; "capital of Germany" takes 7 tokens instead of 46. The iOS chat has a Think switch (off by default).
- The system turn (with the tools, ~400 tokens) is read once per session; each new conversation restores its KV cache (`Model::snapshot` / `restore`, ~24 MB) instead of reading it again. Same reply, token for token (`crates/ironbark/tests/session.rs`).
- Give the date in the system turn: without it Maple assumed 2025, doubted a 2026 search result and searched again for 2025.
- Reasoning can run long (a haiku spent 1024+ tokens counting syllables); cap reply tokens in the app.
- `Session::send` takes a list of messages (`Message::User`, `Assistant`, `ToolResults`): after `reset`, it rebuilds a conversation from a transcript (earlier replies lose their reasoning, as in the template).
- The iOS chat offers three tools it runs itself, with no API keys: `calculator` (on device), `get_weather` (Open-Meteo) and `wikipedia_search` (Wikipedia API). It answers up to 3 rounds of calls per message, shows each call and result, folds the reasoning away, and renders the answer's Markdown (headings, lists, quotes, code, tables). C API: `ironbark_set_system`, `ironbark_send`, `ironbark_parse`.
- The model card notes minimal agentic post-training: measure on your own tools. The iOS app's Bench tab runs the cases above (`-tools 1024`; iPhone 17, 2026-10-15: 4 of 4 pass at 21–23 tok/s).

## On-demand experts

A model can ship without `experts.bin` (4 GB) and build it locally from a server. The fetcher (`ironbark-fetch`) is separate from the engine: it fills the file on its own (e.g. at first launch, before any model is loaded), and the engine only sees a `Records` source, so a record the router needs before it arrives is fetched at once.

```
 app ──▶ Sparse::open(dir, fetch) ──▶ start(threads) ──▶ progress() … stop()      ironbark-fetch
                │ Records
                ▼
 Session::open(dir, Options { records: Some(sparse), .. })                         ironbark (engine)
                │ loader: read record i
                ▼
 bit set in experts.present? ──yes──▶ pread experts.bin
      │ no
      ▼
 Fetch::fetch(i × stride, record_bytes) ──▶ hash ok? ──▶ serve, pwrite + fsync, set bit
 (app callback / HTTP Range GET)              no: 4 tries, then fail the read
```

| Piece | Rust | C / Swift |
|---|---|---|
| Fetcher | `ironbark_fetch::Sparse::open(dir, fetch)` | `ironbark_fetcher_open(dir, fetch_cb, ctx, &err)` |
| Background fill | `start(threads, order)`, `stop()`, `wait_filled(t)` | `ironbark_fetcher_start` / `_stop` / `_close` |
| Progress | `progress()`, `last_error()` | `ironbark_fetcher_progress` (JSON) |
| Engine reads through it | `Options { records: Some(sparse), .. }` | `ironbark_open_fetcher(dir, budget, slots, f, &err)` |
| Engine without it | `Options::default()`: `experts.bin` must be complete | `ironbark_open` |

The dir needs every other file of the pack (incl. `experts.json`, `manifest.json`). A record is fetched once even if the engine and a fill thread want it at the same time; the engine goes first. When every record is present, `experts.present` is removed and the dir is a normal pack.

Mac test, 2026-10-15 (the trit pack as the remote, 30 ms per fetch, 4 fill threads, 64 slots):

| Phase | Result |
|---|---|
| Open | 0.3 s |
| 1st / 2nd / 3rd prompt | 2.1 / 4.8 / 11.6 tok/s decode while filling (3667, 5163, 5765 of 6144 records after each) |
| Fill | complete ~59 s after open, 3.96 GB fetched (each record once, no errors), `ironbark verify`: intact |
| Output | Same text as the local pack (64-token Rust reply) |

### On-demand embedding

`embed.bin` (175 MB) can arrive the same way, so the first download is ~268 MB instead of ~435 MB. The manifest holds one hash per 64 KiB chunk (2,673 chunks). Before prefill or a decode step reads a token's row, the engine makes its three chunks (codes, scales, biases) present, fetching missing ones in parallel; the GPU then reads the mapped file as usual.

```
 token ids ──▶ chunks of their rows ──▶ present? ──yes──▶ GPU reads the mapped embed.bin
                                           │ no
                                           ▼
                        fetch in parallel (≤ 16), hash check, pwrite, set bit in embed.present
```

| Piece | Rust | C / Swift |
|---|---|---|
| Fetcher | `Sparse::open_embed(dir, fetch)` (`fetch` reads the remote `embed.bin`) | `ironbark_embed_fetcher_open(dir, fetch_cb, ctx, &err)` |
| Engine reads through it | `Options { embed: Some(sparse), .. }` | `ironbark_open_fetchers(dir, budget, slots, experts_f, embed_f, &err)` (either may be null) |
| Fill, progress, close | as for experts | `ironbark_fetcher_start` / `_progress` / `_close` |

While `embed.bin` is incomplete, decode reads each token back before queuing the next step (one sync per token). Start its background fill after the experts are in: during the first query, 4 embed fill threads competed with expert fetches and cost ~6 s.

Mac test, 2026-10-16 (M1 Max, 2 GB budget, 30 ms per fetch, 4 expert fill threads, embed fill off; `ironbark gen --fetch-from`):

| First query | `embed.bin` | Download before | Embed fetched | Prefill | Decode |
|---|---|---|---|---|---|
| 887-token Rust prompt | shipped | 435 MB | – | 58.2 / 59.1 s | 15.6 tok/s |
| 887-token Rust prompt | on demand | 268 MB | 170 chunks, 11 MB | 59.8 s | 15.6 tok/s |
| 12-token question | shipped | 435 MB | – | 15.1 / 15.4 s | 1.1 tok/s |
| 12-token question | on demand | 268 MB | 45 chunks, 2.9 MB | 15.1 / 16.0 s | 1.1 tok/s |

Output is the same as the local pack in every run. The first query's time goes to expert fetches, not the embedding.

## iOS

`ironbark pack` writes a self-contained streaming dir: `experts.bin` (4.0 GB trit; 4.9 GB with `--two-bit`) plus the embedding as `embed.bin` (0.18 GB, memory-mapped), the rest of the checkpoint as `model.safetensors` (0.27 GB), `experts.json`, `manifest.json`, config and tokenizer. That dir is what goes on the phone.

| Step | Command |
|---|---|
| Pack | `ironbark pack --model $M --out $P` (trit); add `--two-bit --out models/maple-2bit` for the 2-bit pack |
| Build app | `IRONBARK_TEAM=<Apple Team ID> ios/build.sh` (Rust for `aarch64-apple-ios`, iOS metallib, xcodegen, xcodebuild); `--install` installs on the connected iPhone and warns if the app's models are gone (an install can come with an empty data folder) |
| Copy model | `xcrun devicectl device copy to --device $DEV --domain-type appDataContainer --domain-identifier dev.ironbark.bench --source $P --destination Documents/maple`; the 2-bit pack goes to `Documents/maple-2bit` |
| Benchmark | Put the arguments in `auto.txt` (e.g. `-models maple,maple-2bit -budgets 0,2,1.5`), copy it like the model to `Documents/auto.txt`, then `xcrun devicectl device process launch --terminate-existing --device $DEV dev.ironbark.bench`; results go to `Documents/results.jsonl` (new each run). Don't pass `-auto` on the launch line: Xcode 26's `devicectl` reads it as its own options |

| Piece | Note |
|---|---|
| MLX for iOS | MLX 0.32.2 hardcodes `-sdk macosx` for its kernels; `ironbark-mlx-sys` patches it to `iphoneos`. The app targets iOS 26.2+ (Metal 4, for MLX's NAX prefill kernels; `IOS_MIN` in `ios/build.sh`); `ironbark-mlx-sys` alone defaults to iOS 18 |
| Metal kernels | `mlx.metallib` (136 MB) sits next to the app binary, where MLX looks first |
| Memory | iOS limits the process footprint. MLX's cache of freed buffers held 2.5–3.5 GB of prefill temporaries (footprint 5.2 GB at 64 experts per layer); the plan caps it (0 or 256 MB), same speed |

### Memory budget

The model is sized to the device (`plan.rs`, `device.rs`). Budget = 3/8 of RAM, at most half the iOS app limit; `--budget GB` / `ironbark_open(dir, budget_bytes, 0, &err)` override it. Far from the app limit while on screen; the app frees the model when it leaves the screen (see [Off screen](#off-screen)). The expert cache gets what the rest leaves:

```
 peak = weights 0.27 GB + runtime 0.1 GB + KV(context) + prefill 0.25 GB + buffer cache (0 or 256 MB)
        + slots × (15.7 MB trit / 19.3 MB 2-bit cache + 2 MB prefill)
```

| Device | Budget | Experts per layer | Mode |
|---|---|---|---|
| 8 GB iPhone (app limit 6.44 GB) | 3.0 GB | ~129 (trit) | streamed |
| 6 GB iPhone | 2.25 GB | ~86 (trit) | streamed |
| 16 GB Mac | 6 GB | 256 (trit) | all loaded once |
| 32+ GB Mac | 12+ GB | 256 | all loaded once, no streaming: 2-bit packs as resident tensors (as from the checkpoint), trit packs in the slab with the router's choice mapped to slots on the GPU; no readback either way |

Measured on M1 Max, 2026-10-16 (Rust prompt, 887 tokens, 128 decoded, one run per row; footprint = what iOS counts; packs on the internal SSD: from a disk image the load wait was ~2x and streamed decode up to 2x slower). In brackets: before `embed.bin` (embedding loaded into memory).

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

### Off screen

iOS runs no GPU work for an app off screen. When IronbarkBench leaves the screen, a reply stops at its next token and the model is closed (footprint ~0.1 GB, so iOS has little reason to kill the suspended app); the next message reopens it and rebuilds the conversation from the transcript. The chat keeps the model loaded for 20 s first (a background task), so a quick switch away and back needs no reload. With the Background GPU Access entitlement (paid developer accounts; see `ios/project.yml`) and an iOS 26 device that supports background GPU, a reply instead runs as a `BGContinuedProcessingTask` and finishes off screen.

Without a debugger connection (`devicectl` needs the developer disk image), install with `ios-deploy --bundle`, upload the arguments as `Documents/auto.txt` (`ios-deploy --upload auto.txt --to Documents/auto.txt`), open the app, and download `Documents/results.jsonl`.

## Results

M1 Max (64 GB), measured 2026-10-15. Prompts are the first 3,000 characters of `prompts/prose.txt` and `rust.txt` (746 and 887 tokens); the long prompt is 10,000 characters of prose (2,327 tokens). Decode = greedy tokens/s after a short prompt (256 tokens) and after the prose prompt (128 tokens). DeepGrove's numbers are from mlx-lm `stream_generate` with a 512-token prompt.

| Engine | Decode, short | Decode, 746-token context | Peak GPU memory | Perplexity 400 / 1024 tok |
|---|---|---|---|---|
| DeepGrove `maple.py` (mlx-lm, fused Metal kernels) | – | 217 tok/s | 6.81 GB | 67.8978 / 31.2153 |
| DeepGrove + FlashHead (approximate head) | – | 233 tok/s | 7.30 GB | – |
| ironbark (checkpoint) | **232 tok/s** | 205 tok/s | **5.55 GB** | 67.8979 / 31.2154 |

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

Streaming decode (trit pack, M1 Max internal SSD, 256 decoded tokens, lookahead 1, one run each; perplexity unchanged at 67.8979):

| Cache per layer | Peak GPU memory | Prose: hit / read / decode | Rust: hit / read / decode |
|---|---|---|---|
| all (see above) | 5.12 GB | – / – / ~172 tok/s | – / – / ~172 tok/s |
| 128 | 2.79 GB | 96.1% / 12.4 MB/tok / 68 tok/s | 95.0% / 18.3 MB/tok / 39 tok/s |
| 64 | 1.68 GB | 89.4% / 44.1 MB/tok / 34 tok/s | 87.8% / 44.7 MB/tok / 32 tok/s |
| 32 | 1.14 GB | 81.8% / 74.0 MB/tok / 28 tok/s | 78.0% / 78.7 MB/tok / 26 tok/s |
| 16 | 0.89 GB | 72.5% / 106.1 MB/tok / 23 tok/s | 64.3% / 128.0 MB/tok / 21 tok/s |

The per-layer readback (24 GPU→CPU round trips per token) costs more than the SSD: at 128 per layer, prose waits 2.5 ms per token for loads but decodes at 68 tok/s, against ~172 when every expert fits and no readback is needed. A bare MLX round trip (one tiny op, eval, read) takes ~225 µs on this Mac, so that cost is MLX/Metal latency, not ironbark.

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

Lookahead prefetch (same prompts, decode tok/s):

| Lookahead | Prose 128 | Prose 64 | Prose 32 | Rust 128 | Rust 64 | Rust 32 |
|---|---|---|---|---|---|---|
| 0 | 44.0 | 29.3 | 25.3 | 38.5 | 29.2 | 26.3 |
| 1 (default) | **68.4** | 33.7 | 28.1 | 39.3 | 32.3 | **26.4** |
| 2 | 50.7 | **35.9** | **28.9** | **46.8** | **33.1** | 26.3 |

It raises the effective hit rate (prose at 64: 74.7% → 89.4%) at the cost of more reads (31 → 44 MB per token).

## License

| What | License |
|---|---|
| ironbark code | MIT OR Apache-2.0, at your option ([LICENSE-MIT](LICENSE-MIT), [LICENSE-APACHE](LICENSE-APACHE)) |
| `crates/ironbark-mlx*` | MIT OR Apache-2.0; based on mlx-rs 0.32 by the mlx-rs authors ([DR-0002](docs/decisions/0002-own-mlx-bindings.md)) |
| Maple weights (not in this repo) | MIT, DeepGrove; `ironbark pack` copies its `LICENSE` into every pack |
