# ironbark

Ternary MoE inference on Apple Silicon, in Rust, run on the GPU through MLX. Goal: lowest RAM at usable speed ([DR-0001](https://github.com/russellwmy/ironbark/blob/main/docs/decisions/0001-ternary-moe-streaming.md)).

| Model | Size | Layers | Experts | Weights |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | 35B total / ~3B active | 40 (30 Gated DeltaNet, 10 attention) | 256 per layer, top-8 | Ternary experts, the rest 4-bit; text, images, tool calls |
| Maple (DeepGrove) | 20B total / ~1B active | 24 | 256 per layer, top-8 | Ternary; text, tool calls |

| Model | Where | Setup | Decode | Peak memory |
|---|---|---|---|---|
| Qwen3.6-35B-A3B | M1 Max | Ternary pack, all experts in RAM | 69–72 tok/s | 10.2 GB |
| Qwen3.6-35B-A3B | M1 Max | Ternary pack, 3 GB budget | 10.4–11.7 tok/s | 3.0 GB |
| Qwen3.6-35B-A3B | iPhone 17 | Ternary pack, device budget (3 GB) | 7.4–7.6 tok/s | 2.9 GB |
| Maple | M1 Max | Checkpoint, all experts in RAM | 232 tok/s | 5.55 GB (DeepGrove's `maple.py`: 6.81 GB) |
| Maple | iPhone 17 | Trit pack, device budget (3 GB) | 30–31 tok/s | 2.8 GB |
| Maple | iPhone 17 | Trit pack, 1.5 GB budget | 13.7 tok/s | 1.3 GB |

Greedy decode; the Maple checkpoint row after a short prompt, the others after a 733–962-token prompt (iPhone: prose). Conditions and all measurements: [BENCHMARKS.md](https://github.com/russellwmy/ironbark/blob/main/BENCHMARKS.md).

- Runs a 35B MoE in a 3 GB budget on an iPhone: experts that don't fit are streamed from SSD ([Memory budget](#memory-budget)).
- Qwen: images in chat (`Session::set_image`), tool calls, dynamic top-k on by default in the iOS app ([Dynamic top-k](#dynamic-top-k)).
- Matches the reference on Maple: perplexity 67.8979 vs 67.8978 ([BENCHMARKS.md](https://github.com/russellwmy/ironbark/blob/main/BENCHMARKS.md#checkpoint-vs-reference)).
- Ships small: an app can download the non-expert weights and fetch the experts on demand (Maple: ~268 MB, then 4 GB; [On-demand experts](#on-demand-experts)).
- Rust library, C API for iOS / macOS apps, and the `ironbark` CLI.

More: [BENCHMARKS.md](https://github.com/russellwmy/ironbark/blob/main/BENCHMARKS.md) (measurements), [ARCHITECTURE.md](https://github.com/russellwmy/ironbark/blob/main/ARCHITECTURE.md) (design), [STYLE.md](https://github.com/russellwmy/ironbark/blob/main/STYLE.md) (code style), [docs/decisions](https://github.com/russellwmy/ironbark/blob/main/docs/decisions/README.md) (decision records).

## Table of Contents

- [Getting started](#getting-started)
- [Commands](#commands)
  - [Download options](#download-options)
  - [Expert options](#expert-options)
  - [Examples](#examples)
  - [Qwen bundles](#qwen-bundles)
  - [Maple checkpoints](#maple-checkpoints)
  - [Development](#development)
  - [Release](#release)
- [Usage](#usage)
  - [Chat](#chat) · [Options](#options) · [Images](#images) · [Tool calls](#tool-calls) · [Conversations](#conversations) · [Dynamic top-k](#dynamic-top-k) · [Experts from a server](#experts-from-a-server)
- [Tool use](#tool-use)
- [On-demand experts](#on-demand-experts)
  - [On-demand embedding](#on-demand-embedding)
- [iOS](#ios)
  - [Use in an iOS app](#use-in-an-ios-app)
  - [Bench app](#bench-app)
  - [Memory budget](#memory-budget)
  - [Off screen](#off-screen)
- [Design](#design)
- [Workspace](#workspace)
- [License](#license)

## Getting started

Requirements:

| Need | Notes |
|---|---|
| Apple Silicon Mac | The GPU path (MLX, Metal) only runs on M-series chips |
| Xcode + Metal toolchain | Compiles MLX's Metal kernels; Xcode 26: `xcodebuild -downloadComponent MetalToolchain` |
| Rust via `rustup` | `rust-toolchain.toml` pins 1.99.0; `rustup` installs it on the first `cargo` call |
| CMake | Builds MLX: `brew install cmake` |
| Disk | Qwen: 11.4 GB bundle + ~9 GB pack (`~/.ironbark/models`); Maple: ~5 GB checkpoint + 4.0 GB pack; ~10 GB `target/` |

Steps:

```
 cargo build ──▶ download ──▶ pack ──▶ verify ──▶ gen
  (~15 min,      (11.4 GB,    (~20 s,    (hashes,   (first
   MLX once)      resumes)    D/pack)    records)   tokens)
```

```bash
git clone https://github.com/russellwmy/ironbark && cd ironbark
cargo build --release                                 # first build compiles MLX, about 15 min
B=target/release/ironbark
D=~/.ironbark/models/qwen3.6-35b-a3b-ternary
$B download russellwmy/qwen3.6-35b-a3b-ternary         # -> $D
$B pack --model $D                                    # -> $D/pack
$B verify $D/pack
$B gen --model $D --chat --prompt "What is the capital of Germany?"
```

For Maple instead: [Maple checkpoints](#maple-checkpoints).

Models go to `~/.ironbark/models` unless you pass `--dir` or set `IRONBARK_MODELS`. Set `HF_TOKEN` for private repos. All commands and options: [Commands](#commands); the library: [Usage](#usage); the phone: [iOS](#ios).

## Commands

Requirements and the short path: [Getting started](#getting-started). Every command has `--help`.

| Command | What |
|---|---|
| `ironbark download [REPO]` | Download a model from Hugging Face into `~/.ironbark/models` |
| `ironbark pack --model M --out P` | Turn an MLX checkpoint into a streaming model dir (`experts.bin`, `embed.bin`, other weights, tokenizer, manifest); `--two-bit` for 2-bit expert records; for a [Qwen bundle](#qwen-bundles), `--model M` alone packs into `M/pack` |
| `ironbark verify P` | Check a packed model against its manifest: sizes, file hashes, every expert record |
| `ironbark gen --model M --prompt "…"` | Greedy decode, with prefill / decode speed and GPU memory; `--gen N` tokens (default 128), `--top N` prints top-5 logits |
| `ironbark profile --model M` | GPU time of each part of a decode step |

```
 download ──▶ M: MLX checkpoint ──▶ pack ──▶ P: packed model ──▶ verify ──▶ gen / profile, library, phone
```

### Download options

| Option | Default |
|---|---|
| `REPO` | `deepgrove/maple-preview-2bit-mlx` (Maple); Qwen: `russellwmy/qwen3.6-35b-a3b-ternary` |
| `--dir` | `$IRONBARK_MODELS`, else `~/.ironbark/models` |
| `--name` | Repo name without the owner |
| `--revision` | `main` |

Reruns resume partial files and skip complete ones. Set `HF_TOKEN` for private repos.

### Expert options

For `gen` and `profile`:

| Option | Default | What |
|---|---|---|
| `--experts DIR` | Off: all experts in RAM | Stream experts from a packed dir |
| `--budget GB` | 3/8 of RAM | Footprint budget |
| `--cache N` | What fits the budget | Expert cache slots per layer (of 256) |
| `--loaders N` | 8 | Threads reading experts |
| `--lookahead N` | 1 | Layers ahead to predict and prefetch (0 = off) |
| `--fetch-from DIR` | Off | Fill `--experts`' `experts.bin` and `embed.bin` on demand from this complete pack (stands in for a server) |
| `--fetch-ms MS` | 30 | Delay per fetch, standing in for a network round trip |
| `--fill N` / `--embed-fill N` | 4 / as `--fill` | Background fill threads (0: fetch only what the model needs) |

### Examples

```bash
B=target/release/ironbark
D=~/.ironbark/models/qwen3.6-35b-a3b-ternary   # a packed bundle: what the library and the phone use
$B gen --model $D --chat --prompt "..."                     # sized to this machine
$B gen --model $D --chat --prompt "..." --budget 3          # or to a 3 GB footprint
$B profile --model $D --budget 3                            # GPU time per part of a step
```

To use the library from your own code, see [Usage](#usage).

### Qwen bundles

A bundle is how ternary Qwen models are published: the 4-bit MLX checkpoint
without its experts, and the ternary experts as `experts.safetensors`. Neither format changes
with the pack version, so you pack it yourself.

```bash
$B download russellwmy/qwen3.6-35b-a3b-ternary   # ~11 GB
$B pack --model ~/.ironbark/models/qwen3.6-35b-a3b-ternary
$B gen  --model ~/.ironbark/models/qwen3.6-35b-a3b-ternary --chat --prompt "Hello"
```

```
 qwen3.6-35b-a3b-ternary/
 ├── config.json, tokenizer*, model.safetensors, …   from the 4-bit checkpoint, no experts
 ├── experts.safetensors                            ternary experts (codes, group scales)
 └── pack/                                          written by `pack` (~9 GB)
```

| Command | On a bundle dir `D` |
|---|---|
| `pack --model D` | `--ternary D/experts.safetensors`, `--out D/pack` (both can be given) |
| `gen` / `profile --model D`, `Session::open(D)` | Load `D/pack`; error if not packed yet |

Packing Qwen3.6-35B-A3B takes about 20 s and 2.9 GB of memory. After a pack version change,
run `pack` again: it rebuilds `D/pack` from the bundle. `pack --vision-only` adds just the
vision encoder (6-bit) to an existing pack.

### Maple checkpoints

Maple is published as a 2-bit MLX checkpoint; `pack` writes the streaming dir.

```bash
M=~/.ironbark/models/maple-preview-2bit-mlx   # from: $B download (the default repo)
P=~/.ironbark/models/maple
$B download
$B gen --model $M --prompt "The capital of Germany is" --gen 128   # the checkpoint itself
$B pack --model $M --out $P                   # trit experts (4.0 GB) + other weights, tokenizer, manifest
$B verify $P                                  # e.g. after copying
$B gen --model $P --prompt "..." --experts $P             # sized to this machine
$B gen --model $P --prompt "..." --experts $P --budget 2  # or to a 2 GB footprint
```

### Development

Toolchain pinned in `rust-toolchain.toml`; config in `rustfmt.toml`, `clippy.toml`, `deny.toml`; rules in [STYLE.md](https://github.com/russellwmy/ironbark/blob/main/STYLE.md).

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --release -- -D warnings
IRONBARK_MODEL=$D cargo test --workspace --release -- --test-threads=1   # without IRONBARK_MODEL the model tests are skipped
cargo deny check
```

### Release

Pushing a version tag publishes the crates to crates.io (`.github/workflows/publish.yml`, Linux runner, no build: `--no-verify` uploads the source):

```
 bump version ──▶ dry run ──▶ commit, push ──▶ git tag vX.Y.Z ──▶ git push origin vX.Y.Z
                                                                       │
                                                                       ▼
          tag == workspace version? ──▶ each crate in dependency order (all but ironbark-ffi):
                                        already published? skip; else cargo publish --no-verify;
                                        429? wait for the time crates.io gives, retry
```

| Step | How |
|---|---|
| Once | Add a crates.io API token as the repo secret `CARGO_REGISTRY_TOKEN` (scopes: publish-new, publish-update) |
| Bump | `version` in `[workspace.package]` and in `[workspace.dependencies]` of the root `Cargo.toml` |
| Dry run | `cargo publish --workspace --dry-run`: builds each package as crates.io gets it (CI does not) |
| Tag | `git tag v0.1.0 && git push origin v0.1.0`: the tag must equal the version, or the job fails before publishing |
| Rerun | Actions > Publish > Run workflow with the tag: publishes only the crates not yet on crates.io |

A crates.io version can be yanked, not deleted. New crates are rate limited (a burst, then one per 10 minutes), so a first release can wait between crates; the job waits and retries.

## Usage

Needs Rust 1.99+, macOS or iOS on Apple Silicon, Xcode and CMake ([Getting started](#getting-started)). The first build downloads MLX v0.32.4 from GitHub and compiles it with its Metal kernels (about 15 min); iOS needs nothing extra.

```toml
[dependencies]
ironbark = "0.3"
ironbark-fetch = "0.3"   # only for on-demand experts
```

Install the CLI: `cargo install ironbark-cli` (binary `ironbark`). The C API (`ironbark-ffi`) is not on crates.io; build it from this repo.

```
 Session::open(dir, Options) ──▶ set_system(date, tools) ──▶ chat(msg) ──▶ Reply (iterator of text pieces)
                                                                │ parse(reply).calls not empty
                                                                ▼
                                     app runs the tools ──▶ tool_results(&[..]) ──▶ Reply …
```

`dir` is a packed model or a packed Qwen bundle (`ironbark pack`, see [Commands](#commands)).

### Chat

```rust
fn main() -> ironbark::Result<()> {
    let mut s = ironbark::Session::open("qwen3.6-35b-a3b-ternary", ironbark::Options::default())?; // sized to the device
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
| `max_context` | `None`: Maple 4096; Qwen 16384 if that leaves 48+ expert slots per layer, else 4096 | Most tokens per conversation; the budget reserves its KV cache |
| `loaders` | 8 | Threads reading experts from storage |
| `records` | `None`: the dir's `experts.bin` (must be complete) | Where expert records come from, e.g. an `ironbark_fetch::Sparse` |

```rust
let opts = ironbark::Options { budget: Some(2_000_000_000), ..Default::default() }; // 2 GB footprint
```

### Images

Qwen only, and the pack needs its vision encoder. The image goes with the next user message.

```rust
let info = s.set_image(&rgb, width, height, ironbark::IMAGE_PIXELS)?; // RGB bytes; resized to ≤ 512 × 512
println!("{} image tokens, encoded in {:.1} s", info.tokens, info.encode_s);
let reply: String = s.chat("Describe this photo.")?.collect::<ironbark::Result<_>>()?;
```

C API: `ironbark_set_image`. iPhone 17: a 576 × 416 photo is 234 tokens, encoded in 0.6 s ([BENCHMARKS.md](https://github.com/russellwmy/ironbark/blob/main/BENCHMARKS.md#qwen-on-the-iphone)).

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
| `s.set_dynamic_k(Dynamic { drop_below: 0.08, ..Dynamic::default() })` | Decode drops each token's expert choices under that share of its routing weight, keeping at least 3 (default off); fewer expert reads, different text ([Dynamic top-k](#dynamic-top-k)) |
| `s.read_system()` | Read the system turn now (restored when read recently), so the first message does not wait for it |
| `s.read_system_cached("sys.safetensors")` | Same, kept across sessions: load the system turn from the file, else read it and save the file |
| `s.snapshot()` / `s.restore(&snap)` | Save and return to a conversation in memory (cheap: arrays shared) |
| `s.save("x.safetensors")` / `s.load(path)` | Save and return to a conversation across sessions: its caches as safetensors, with the token ids, system turn and a model fingerprint (ironbark version + `manifest.json`). `load` returns false when the file is missing or from another model or version |

A long prompt read once (a system turn with tools, few-shot samples) loads from a file
instead of being read again after the model reopens; on a phone, where a cold prompt reads at
45–59 tok/s, that is the difference between seconds and a minute. Qwen3.6 files hold ~63 MB of
fixed DeltaNet state plus the full-attention KV (2,000 tokens: ~100 MB).

```
 system turn:  set_system ─▶ read_system_cached(file)     first run: read + save; later: load
 samples:      load(file) ─▶ false ─▶ read(samples) ─▶ save(file)
                         └─▶ true: samples already read
```

### Dynamic top-k

Decode can skip the experts a token barely uses. Off by default (the model as trained); prompts always use all 8.

```
 router: top-8 choices, weights w ─► shares w / Σw ─► drop shares < drop_below (never the 3 largest)
 dropped choice ─► the token's strongest expert, weight 0 (already loaded: no read;
                   Qwen streaming also skips its kernels)
 kept weights   ─► rescaled to the total they had
```

A higher `drop_below` drops more experts: faster, further from the model as trained.

| | 0.08 | 0.10 |
|---|---|---|
| Drops a choice under | 8% of the token's routing weight | 10% |
| Qwen3.6, iPhone 17 chat (off: 10.9 tok/s, 100 MB read per token) | 12.0 tok/s, 87 MB | 18.4 tok/s, 43 MB |
| Qwen3.6 quality vs off | Tools 45 / 46, tasks 72 / 80, IFEval (120) 80.0% | Tools 46 / 46, tasks 72 / 80, IFEval (120) 80.0%, full IFEval 74.7% (off: 76.2%) |
| Maple tasks (off: 59 / 80) | 57 | 52 |
| Default in the iOS app | Qwen | – |

- Maple keeps it off: it loses tasks at both settings.
- The 0.08 speed row was measured before lookahead prefetches followed the drop; the 0.10 row after. On raw prompts after that change, Qwen at 0.08 decoded 8–10% slower than off ([BENCHMARKS.md](https://github.com/russellwmy/ironbark/blob/main/BENCHMARKS.md#dynamic-top-k)), so re-measure before choosing by speed.
- The reply text changes within about the first 20 tokens; each setting is repeatable run to run.
- Evals (tools, tasks, IFEval): external runs on the ternary packs.

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

Both chat templates list tools in the system turn and take results back as `<tool_response>` blocks. The library writes the turns and parses replies into the same `ToolCall` (name, JSON arguments); the app runs the tools (search API keys stay in the app).

| Model | A call in the reply |
|---|---|
| Qwen3.6 | `<tool_call><function=name><parameter=arg>value</parameter>…</function></tool_call>` |
| Maple (Hermes style) | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` |

```
 set_system(Some("Today is …"), &tools) ──▶ chat("weather in Adelaide?") ──▶ <think>…</think><tool_call>{…}</tool_call>
                                                                                 │ parse(reply).calls
 app runs get_weather ──▶ tool_results(&["{\"temp_c\": 22}"]) ──▶ "It is 22 °C in Adelaide."
```

Code: [Usage → Tool calls](#tool-calls).

| Case (Maple, Mac, trit pack, replies ≤ 1024 tokens, 2026-10-16; Qwen3.6 passes the same 4 on the iPhone) | Called | Answer |
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
- Maple's model card notes minimal agentic post-training: measure on your own tools. The iOS app's Bench tab runs the cases above (`-tools 1024`; iPhone 17: Qwen3.6 4 of 4 at 7–9 tok/s, 2026-10-27; Maple 4 of 4 at 21–23 tok/s, 2026-10-15).

## On-demand experts

A model can ship without `experts.bin` (Maple: 4 GB) and build it locally from a server. The fetcher (`ironbark-fetch`) is separate from the engine: it fills the file on its own (e.g. at first launch, before any model is loaded), and the engine only sees a `Records` source, so a record the router needs before it arrives is fetched at once.

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

Measured with Maple on the Mac: [BENCHMARKS.md](https://github.com/russellwmy/ironbark/blob/main/BENCHMARKS.md#on-demand-experts).

### On-demand embedding

`embed.bin` can arrive the same way (Maple: 175 MB, so the first download is ~268 MB instead of ~435 MB). The manifest holds one hash per 64 KiB chunk (2,673 chunks). Before prefill or a decode step reads a token's row, the engine makes its three chunks (codes, scales, biases) present, fetching missing ones in parallel; the GPU then reads the mapped file as usual.

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

Measured with Maple on the Mac: [BENCHMARKS.md](https://github.com/russellwmy/ironbark/blob/main/BENCHMARKS.md#on-demand-embedding).

## iOS

`ironbark pack` writes a self-contained streaming dir: `experts.bin`, the embedding as `embed.bin` (memory-mapped), the rest of the checkpoint as `model.safetensors`, `experts.json`, `manifest.json`, config and tokenizer. That dir is what goes on the phone: `D/pack` for Qwen3.6 (~9 GB); for Maple 4.0 GB of trit experts (4.9 GB with `--two-bit`), 0.18 GB embedding, 0.27 GB other weights.

### Use in an iOS app

```
 cargo build (ironbark-ffi, aarch64-apple-ios) ──▶ libironbark_ffi.a + ironbark.h ──▶ Xcode app
                                                     + mlx.metallib (app resource)
 app: ironbark_open(dir) ──▶ ironbark_set_system ──▶ ironbark_send(messages) ──▶ ironbark_next … NULL
```

| Step | What |
|---|---|
| Build the library | `IPHONEOS_DEPLOYMENT_TARGET=26.2 cargo build --release -p ironbark-ffi --target aarch64-apple-ios` → `target/aarch64-apple-ios/release/libironbark_ffi.a` |
| Header | `crates/ironbark-ffi/include/ironbark.h`; import it in the bridging header (Swift) |
| Link | `-lironbark_ffi -lc++ -framework Metal -framework Accelerate` |
| Metal kernels | Add `~/.mlx/lib/*-ios26.2/mlx.metallib` (made by the build) to the app's resources: MLX looks next to the app binary first |
| Entitlement | `com.apple.developer.kernel.increased-memory-limit` (raises the app limit; the budget is half of it) |
| Deployment target | iOS 26.2+ (Metal 4) |
| Model | A pack dir in the app's Documents (or download it there; see [On-demand experts](#on-demand-experts)) |

`ios/project.yml` and `ios/build.sh` do all of this for the bench app; copy from them.

Every call returns JSON as a `char *` (free it with `ironbark_free_string`); a failure is `{"error": "..."}`. Make calls on one session from one serial queue.

```swift
var err: UnsafeMutablePointer<CChar>?
guard let s = ironbark_open(modelDir, 0, 0, &err) else { /* String(cString: err!) */ return }
defer { ironbark_close(s) }

func take(_ p: UnsafeMutablePointer<CChar>?) -> String {
    defer { ironbark_free_string(p) }
    return p.map { String(cString: $0) } ?? ""
}

_ = take(ironbark_set_system(s, "You are helpful.", nil))      // tools: JSON array or nil
_ = take(ironbark_send(s, #"[{"role":"user","content":"Hi"}]"#, false))  // turn stats
var reply = ""
while let p = ironbark_next(s) {                                 // NULL: the reply ended
    reply += String(cString: p)
    ironbark_free_string(p)
}
let parsed = take(ironbark_parse(reply))                         // {thinking, text, calls, malformed}
```

| Call | Use |
|---|---|
| `ironbark_open(dir, budget, slots, &err)` | Open a pack; `budget` 0 sizes it to the device |
| `ironbark_set_system(s, system, tools)` | System turn and tool schemas; forgets the conversation |
| `ironbark_send(s, messages, reset)` | Read the new messages (`user`, `assistant`, `tool`); `reset` rebuilds from a full transcript |
| `ironbark_next(s)` | Next piece of the reply, NULL at its end; stop calling to stop the reply |
| `ironbark_parse(reply)` | Split a reply into thinking, text and tool calls |
| `ironbark_set_thinking(s, on)` | Reason before answering (default on) |
| `ironbark_stats(s)` | The turn's speed and memory |
| `ironbark_open_fetchers(...)` | Open with experts / embedding fetched on demand |

When the app leaves the screen, close the session (iOS runs no GPU work off screen); see [Off screen](#off-screen).

### Bench app

`ios/IronbarkBench` chats with the models in its Documents and benchmarks them on the phone. It opens Qwen when present (with Expert drop 0.08), else Maple.

| Step | Command |
|---|---|
| Pack | Qwen: `ironbark pack --model $D`; Maple: `ironbark pack --model $M --out $P` (trit), add `--two-bit --out ~/.ironbark/models/maple-2bit` for the 2-bit pack |
| Build app | `IRONBARK_TEAM=<Apple Team ID> ios/build.sh` (Rust for `aarch64-apple-ios`, iOS metallib, xcodegen, xcodebuild); `--install` installs on the connected iPhone and warns if the app's models are gone (an install can come with an empty data folder) |
| Copy model | `xcrun devicectl device copy to --device $DEV --domain-type appDataContainer --domain-identifier dev.ironbark.bench --source $D/pack --destination Documents/qwen3.6-35b-a3b-ternary` (a name starting with `qwen`); Maple: `--source $P --destination Documents/maple`, the 2-bit pack to `Documents/maple-2bit` |
| Benchmark | Put the arguments in `auto.txt` (e.g. `-models qwen3.6-35b-a3b-ternary,maple -budgets 0,2,1.5`), copy it like the model to `Documents/auto.txt`, then `xcrun devicectl device process launch --terminate-existing --device $DEV dev.ironbark.bench`; results go to `Documents/results.jsonl` (new each run). Don't pass `-auto` on the launch line: Xcode 26's `devicectl` reads it as its own options |

| Piece | Note |
|---|---|
| MLX for iOS | MLX 0.32.4 hardcodes `-sdk macosx` for its kernels; `ironbark-mlx-sys` patches it to `iphoneos`. The app targets iOS 26.2+ (Metal 4, for MLX's NAX prefill kernels; `IOS_MIN` in `ios/build.sh`); `ironbark-mlx-sys` alone defaults to iOS 18 |
| Metal kernels | `mlx.metallib` (136 MB) sits next to the app binary, where MLX looks first |
| Memory | iOS limits the process footprint. MLX's cache of freed buffers held 2.5–3.5 GB of prefill temporaries (footprint 5.2 GB at 64 experts per layer); the plan caps it (0 or 256 MB), same speed |

### Memory budget

The model is sized to the device (`plan.rs`, `device.rs`). Budget = 3/8 of RAM, at most half the iOS app limit; `--budget GB` / `ironbark_open(dir, budget_bytes, 0, &err)` override it. Far from the app limit while on screen; the app frees the model when it leaves the screen (see [Off screen](#off-screen)). The expert cache gets what the rest leaves.

Qwen3.6 (measured):

| Device | Budget | Experts per layer | Mode |
|---|---|---|---|
| iPhone 17 (8 GB) | 3.02 GB | 31 | streamed |
| Mac, `--budget 3` | 3 GB | 18 | streamed |
| 32+ GB Mac | default | 256 | all loaded once (7.55 GB expert cache, 10.2 GB peak) |

Maple:

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

Measured footprint, hit rate and speed per budget on the Mac and the iPhone 17: [BENCHMARKS.md](https://github.com/russellwmy/ironbark/blob/main/BENCHMARKS.md#packs-by-budget).

### Off screen

iOS runs no GPU work for an app off screen. When IronbarkBench leaves the screen, a reply stops at its next token and the model is closed (footprint ~0.1 GB, so iOS has little reason to kill the suspended app); the next message reopens it and rebuilds the conversation from the transcript. The chat keeps the model loaded for 20 s first (a background task), so a quick switch away and back needs no reload. With the Background GPU Access entitlement (paid developer accounts; see `ios/project.yml`) and an iOS 26 device that supports background GPU, a reply instead runs as a `BGContinuedProcessingTask` and finishes off screen.

Without a debugger connection (`devicectl` needs the developer disk image), install with `ios-deploy --bundle`, upload the arguments as `Documents/auto.txt` (`ios-deploy --upload auto.txt --to Documents/auto.txt`), open the app, and download `Documents/results.jsonl`.

## Design

The section below is Maple's decode step. Qwen3.6 shares the expert path (router top-8 of 256, ternary experts in the slab, streamed the same way) over 40 layers: 30 Gated DeltaNet and 10 attention, the rest of its weights 4-bit.

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

## Workspace

```
ironbark/
├── .github/workflows/  publish.yml: crates.io release on a version tag
├── Cargo.toml          workspace: members, shared deps, profiles
├── ARCHITECTURE.md     design
├── STYLE.md            code style and checks
├── crates/
│   ├── ironbark/           library: Session (open sized to the device, chat, tools)
│   ├── ironbark-engine/    Qwen3.6 and Maple decode on the GPU
│   ├── ironbark-experts/   event-driven expert store (SSD streaming), for the RAM work
│   ├── ironbark-fetch/     builds a local experts.bin from a remote copy (separate from the engine)
│   ├── ironbark-ffi/       C API (ironbark.h) for iOS / macOS apps
│   ├── ironbark-cli/       binary `ironbark`: download, pack, verify, gen, profile
│   ├── ironbark-mlx/       safe Rust API over MLX
│   └── ironbark-mlx-sys/   mlx-c bindings; builds MLX for macOS and iOS
├── ios/                IronbarkBench: SwiftUI benchmark app (xcodegen)
├── prompts/            prompt texts for perplexity and benchmarks (prose, rust)
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
| `ironbark-engine` | Loads a packed dir (Qwen3.6 bundle or Maple pack) or `deepgrove/maple-preview-2bit-mlx`; Qwen3.6 vision encoder; one-token decode step on custom Metal kernels (`row.rs` ternary matvec, `fused.rs` norms / RoPE / router, via the MLX C API in `kernel.rs`) plus MLX attention; bit-exact with DeepGrove's `maple.py` |
| `ironbark-experts` | `experts.bin` records streamed from SSD into a page-aligned slab; per-layer LRU cache, loader pool, pinning; reads through a `Records` trait |
| `ironbark-fetch` | Sparse `experts.bin` filled from a remote copy: on-demand records, hash checks, presence bitmap, background fill; runs with or without the engine |
| `ironbark-ffi` | C API (`ironbark.h`), static library for the iOS / macOS apps |
| `ironbark-cli` | The `ironbark` binary: `download`, `pack`, `verify`, `gen`, `profile` (see [Commands](#commands)) |
| `ironbark-mlx` | Safe Rust API over MLX for inference (arrays, ops, fused kernels, eval, compile) |
| `ironbark-mlx-sys` | Bindings to mlx-c; builds MLX v0.32.4 and its Metal kernels for macOS and iOS |

## License

| What | License |
|---|---|
| ironbark code | MIT OR Apache-2.0, at your option ([LICENSE-MIT](https://github.com/russellwmy/ironbark/blob/main/LICENSE-MIT), [LICENSE-APACHE](https://github.com/russellwmy/ironbark/blob/main/LICENSE-APACHE)) |
| `crates/ironbark-mlx*` | MIT OR Apache-2.0; based on mlx-rs 0.32 by the mlx-rs authors ([DR-0002](https://github.com/russellwmy/ironbark/blob/main/docs/decisions/0002-own-mlx-bindings.md)) |
| Qwen3.6 weights (not in this repo) | Apache-2.0, Qwen team (Alibaba) |
| Maple weights (not in this repo) | MIT, DeepGrove; `ironbark pack` copies its `LICENSE` into every pack |
