//! `ironbark`: run Maple (ternary MoE) on the Apple GPU through MLX.

#![expect(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "printing results is this binary's job"
)]

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};

mod download;
mod maple;
mod qwen;
mod util;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

/// `pack --vision-only`: add the vision encoder of `model` to the pack `out`.
fn pack_vision(model: &std::path::Path, out: &std::path::Path) -> Result<()> {
    anyhow::ensure!(ironbark_engine::qwen::is_qwen(model), "--vision-only: Qwen3.5 / 3.6 only");
    let bytes = ironbark_engine::qwen::pack_vision(model, out)?
        .ok_or_else(|| anyhow::anyhow!("{}: no vision weights", model.display()))?;
    println!("vision encoder {:.0} MB -> {}", bytes as f64 / 1e6, out.display());
    Ok(())
}

/// Expert placement flags shared by `gen` and `profile`.
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
    /// Download a model from Hugging Face into <dir>/<name> (resumes; complete files are skipped)
    Download {
        /// Hugging Face repo
        #[arg(default_value = download::DEFAULT_REPO)]
        repo: String,
        /// Models dir (default: `$IRONBARK_MODELS`, else `~/.ironbark/models`)
        #[arg(long)]
        dir: Option<PathBuf>,
        /// Subdir to write (default: the repo name without the owner)
        #[arg(long)]
        name: Option<String>,
        /// Branch, tag or commit
        #[arg(long, default_value = "main")]
        revision: String,
    },
    /// Write a streaming model dir: experts to <out>/experts.bin, the other weights, config and
    /// tokenizer next to it
    Pack {
        /// MLX checkpoint dir, or a bundle (a 4-bit checkpoint without its experts, plus
        /// experts.safetensors: packs that)
        #[arg(long)]
        model: PathBuf,
        /// Output dir (put it on the internal SSD; default for a bundle: <model>/pack)
        #[arg(long)]
        out: Option<PathBuf>,
        /// Store expert rows as 2-bit codes instead of trits (23% larger)
        #[arg(long)]
        two_bit: bool,
        /// Qwen: ternary experts from this file (codes and group scales) as trit records (default for a bundle: its experts.safetensors)
        #[arg(long)]
        ternary: Option<PathBuf>,
        /// Qwen: only add the vision encoder (6-bit) to the existing pack <out>
        #[arg(long, conflicts_with_all = ["ternary", "two_bit"])]
        vision_only: bool,
    },
    /// Greedy decode from a prompt, with prefill / decode speed and GPU memory
    Gen {
        /// MLX checkpoint dir (deepgrove/maple-preview-2bit-mlx), pack, or packed bundle
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
        /// Qwen: wrap the prompt as one chat turn (thinking off) and stop at the end of the turn
        #[arg(long)]
        chat: bool,
        /// Qwen: an image (PNG or JPEG) before the prompt, in a chat turn
        #[arg(long)]
        image: Option<PathBuf>,
        /// Qwen: most pixels the image is resized to (sides multiples of 32)
        #[arg(long, default_value_t = 1 << 20)]
        max_pixels: usize,
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
        /// MLX checkpoint dir, pack, or packed bundle
        #[arg(long)]
        model: PathBuf,
        #[command(flatten)]
        experts: ExpertArgs,
    },
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Download {
            repo,
            dir,
            name,
            revision,
        } => {
            let name = name.unwrap_or_else(|| repo.rsplit('/').next().unwrap_or(&repo).to_owned());
            let out = download::models_dir(dir)?.join(name);
            download::download(&repo, &revision, &out)?;
            let next = if ironbark_engine::qwen::bundle_experts(&out).is_some() {
                ""
            } else {
                " --out <packed dir>"
            };
            println!("done: {}\nnext: ironbark pack --model {0}{next}", out.display());
            Ok(())
        }
        Cmd::Pack {
            model,
            out,
            two_bit,
            ternary,
            vision_only,
        } => {
            let bundle = ironbark_engine::qwen::bundle_experts(&model);
            let out = match (out, &bundle) {
                (Some(o), _) => o,
                (None, Some(_)) => model.join(ironbark_engine::qwen::BUNDLE_PACK),
                (None, None) => anyhow::bail!("--out: required ({} is not a bundle)", model.display()),
            };
            if vision_only {
                return pack_vision(&model, &out);
            }
            let ternary = ternary.or(bundle);
            let p = if let Some(t) = &ternary {
                anyhow::ensure!(ironbark_engine::qwen::is_qwen(&model), "--ternary: Qwen3.5 / 3.6 only");
                ironbark_engine::qwen::pack_ternary(&model, &out, t)?
            } else {
                ironbark_engine::pack(&model, &out, !two_bit)?
            };
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
            chat,
            image,
            max_pixels,
            mut experts,
        } => {
            let model = ironbark_engine::qwen::packed(&model)?;
            experts.experts = experts.experts.as_deref().map(ironbark_engine::qwen::packed).transpose()?;
            if image.is_some() && !ironbark_engine::qwen::is_qwen(&model) {
                anyhow::bail!("--image: Qwen3.5 / 3.6 only");
            }
            if let Some(img) = image {
                qwen::run_image(&model, &experts.into_experts(), &img, max_pixels, &prompt, n_gen)
            } else if ironbark_engine::qwen::is_qwen(&model) {
                qwen::run(&model, &experts.into_experts(), &prompt, n_gen, top, chat)
            } else {
                maple::run(&model, &experts.into_experts(), &prompt, n_gen, top)
            }
        }
        Cmd::Verify { dir } => {
            let r = ironbark_engine::manifest::verify(&dir, 8)?;
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
        Cmd::Profile { model, mut experts } => {
            experts.experts = experts.experts.as_deref().map(ironbark_engine::qwen::packed).transpose()?;
            maple::profile(&ironbark_engine::qwen::packed(&model)?, &experts.into_experts())
        }
    }
}
