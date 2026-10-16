//! `ironbark`: run Maple (ternary MoE) on the Apple GPU through MLX.

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "printing results is this binary's job"
)]

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

mod maple;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// Expert placement flags shared by `gen`, `profile` and `ppl`.
#[derive(Debug, clap::Args)]
struct ExpertArgs {
    /// Stream experts from this packed dir (`ironbark pack`) instead of keeping all 256 per layer
    #[arg(long)]
    experts: Option<PathBuf>,
    /// Expert cache slots per layer when streaming (default: what fits the budget)
    #[arg(long)]
    cache: Option<usize>,
    /// Footprint budget in GB when streaming (default: from the device, 3/8 of RAM)
    #[arg(long)]
    budget: Option<f64>,
    /// Loader threads when streaming
    #[arg(long, default_value_t = 8)]
    loaders: usize,
    /// Streaming: predict and prefetch the experts of this many layers ahead (0 = off)
    #[arg(long, default_value_t = 1)]
    lookahead: usize,
    /// Fill `--experts`' experts.bin and embed.bin on demand from this complete pack (a stand-in
    /// for a server; delete them in `--experts` first)
    #[arg(long)]
    fetch_from: Option<PathBuf>,
    /// Delay per fetch in ms, standing in for a network round trip
    #[arg(long, default_value_t = 30)]
    fetch_ms: u64,
    /// Background fill threads (0: fetch only what the model needs)
    #[arg(long, default_value_t = 4)]
    fill: usize,
    /// Background fill threads for embed.bin (default: as `--fill`)
    #[arg(long)]
    embed_fill: Option<usize>,
}

impl ExpertArgs {
    fn into_experts(self) -> maple::Experts {
        maple::Experts {
            dir: self.experts,
            cache: self.cache,
            budget: self.budget.map(|g| (g * 1e9) as u64),
            loaders: self.loaders,
            lookahead: self.lookahead,
            fetch: self.fetch_from.map(|from| maple::Remote {
                from,
                delay: std::time::Duration::from_millis(self.fetch_ms),
                fill: self.fill,
                embed_fill: self.embed_fill.unwrap_or(self.fill),
            }),
        }
    }
}

#[derive(Debug, Subcommand)]
enum Cmd {
    /// Write a streaming model dir: experts to <out>/experts.bin, the other weights, config and
    /// tokenizer next to it
    Pack {
        /// MLX checkpoint dir
        #[arg(long)]
        model: PathBuf,
        /// Output dir (put it on the internal SSD)
        #[arg(long)]
        out: PathBuf,
        /// Store expert rows as 2-bit codes instead of trits (23% larger)
        #[arg(long)]
        two_bit: bool,
    },
    /// Greedy decode from a prompt, with prefill / decode speed and GPU memory
    Gen {
        /// MLX checkpoint dir (deepgrove/maple-preview-2bit-mlx)
        #[arg(long)]
        model: PathBuf,
        /// Prompt text
        #[arg(long)]
        prompt: String,
        /// Tokens to generate
        #[arg(long = "gen", default_value_t = 128)]
        n_gen: usize,
        /// Print the top 5 logits of the first N decode steps (to compare with a reference)
        #[arg(long, default_value_t = 0)]
        top: usize,
        #[command(flatten)]
        experts: ExpertArgs,
    },
    /// Check a packed model against its manifest (sizes, file hashes, every expert record)
    Verify {
        /// Packed model dir
        dir: PathBuf,
    },
    /// GPU time of each part of a decode step (no syncs, no expert loads)
    Profile {
        /// MLX checkpoint dir
        #[arg(long)]
        model: PathBuf,
        #[command(flatten)]
        experts: ExpertArgs,
    },
    /// Perplexity of a text file: the first half of --tokens is context, the second half is scored
    Ppl {
        /// MLX checkpoint dir
        #[arg(long)]
        model: PathBuf,
        /// Text file
        #[arg(long)]
        text: PathBuf,
        /// Tokens to read from the text
        #[arg(long, default_value_t = 512)]
        tokens: usize,
        /// Score with one batched forward (prefill kernels) instead of token by token
        #[arg(long)]
        batch: bool,
        #[command(flatten)]
        experts: ExpertArgs,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Pack { model, out, two_bit } => {
            let p = ironbark_mlx::pack(&model, &out, !two_bit)?;
            println!(
                "packed {} experts x {:.2} MB (stride {} B) -> {}",
                p.records,
                p.record_bytes as f64 / 1e6,
                p.stride,
                out.display()
            );
            if !out.join("LICENSE").exists() {
                eprintln!(
                    "warning: {} has no LICENSE; add the model's license to {} before shipping it",
                    model.display(),
                    out.display()
                );
            }
            Ok(())
        }
        Cmd::Gen {
            model,
            prompt,
            n_gen,
            top,
            experts,
        } => maple::run(&model, &experts.into_experts(), &prompt, n_gen, top),
        Cmd::Verify { dir } => {
            let r = ironbark_mlx::manifest::verify(&dir, 8)?;
            for p in &r.problems {
                println!("BAD  {p}");
            }
            println!(
                "{} files, {} records (experts, embed chunks), {:.2} GB read: {}",
                r.files,
                r.records,
                r.bytes as f64 / 1e9,
                if r.problems.is_empty() { "intact" } else { "damaged" }
            );
            anyhow::ensure!(r.problems.is_empty(), "{} problems", r.problems.len());
            Ok(())
        }
        Cmd::Profile { model, experts } => maple::profile(&model, &experts.into_experts()),
        Cmd::Ppl {
            model,
            text,
            tokens,
            batch,
            experts,
        } => maple::ppl(&model, &experts.into_experts(), &text, tokens, batch),
    }
}
