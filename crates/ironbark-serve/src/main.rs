//! `ironbark-serve`: an OpenAI-compatible server for a packed model.

use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::Parser;
use ironbark_serve::{Config, serve};

/// Serve a packed model over an OpenAI-compatible HTTP API
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Packed model dir (`ironbark pack`)
    #[arg(long)]
    model: PathBuf,
    /// Address to listen on; anything but loopback needs --api-key
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8000)]
    port: u16,
    /// Bearer token clients must send (default: `IRONBARK_API_KEY` in the environment, else none)
    #[arg(long, env = "IRONBARK_API_KEY", hide_env_values = true)]
    api_key: Option<String>,
    /// Name clients see (default: the model dir's name)
    #[arg(long)]
    model_id: Option<String>,
    /// Memory budget in GB (default: from the device)
    #[arg(long)]
    budget: Option<f64>,
    /// Expert cache slots per layer (default: as many as fit; 256 keeps every expert in RAM)
    #[arg(long)]
    slots: Option<usize>,
    /// Most tokens a conversation holds (default: the model's own limit)
    #[arg(long)]
    max_context: Option<usize>,
    /// Longest reply, unless a request asks for another length
    #[arg(long, default_value_t = 16_384)]
    max_tokens: usize,
    /// Reply without reasoning first, unless a request asks for it
    #[arg(long)]
    no_think: bool,
    /// Sampling, unless a request says otherwise (default: the model's own)
    #[arg(long)]
    temperature: Option<f32>,
    #[arg(long)]
    top_p: Option<f32>,
    #[arg(long)]
    top_k: Option<usize>,
    #[arg(long)]
    presence_penalty: Option<f32>,
}

#[expect(clippy::print_stderr, reason = "the server's log")]
fn main() -> Result<()> {
    // SAFETY: first in main; nothing else is running.
    unsafe { ironbark_engine::large_command_buffers() };
    let args = Args::parse();
    let loopback = matches!(args.host.as_str(), "127.0.0.1" | "::1" | "localhost");
    let api_key = args.api_key.filter(|k| !k.is_empty());
    if !loopback && api_key.is_none() {
        bail!("listening on {} would let anyone on the network run this model: set --api-key", args.host);
    }
    // Before the model: a taken port should not cost a load.
    let listener = TcpListener::bind((args.host.as_str(), args.port))
        .with_context(|| format!("cannot listen on {}:{}", args.host, args.port))?;
    let model_id = args.model_id.unwrap_or_else(|| {
        args.model.file_name().map_or_else(|| "model".to_owned(), |n| n.to_string_lossy().into_owned())
    });
    let max_context = match args.max_context {
        Some(c) => c,
        None => model_context(&args.model)?,
    };
    let options = ironbark::Options {
        budget: args.budget.map(|gb| (gb * 1e9) as u64),
        slots: args.slots,
        max_context: Some(max_context),
        ..ironbark::Options::default()
    };
    eprintln!("loading {}", args.model.display());
    let mut session =
        ironbark::Session::open(&args.model, options).with_context(|| args.model.display().to_string())?;
    let own = session.sampling();
    session.set_sampling(Some(ironbark::Sampling {
        temperature: args.temperature.unwrap_or(own.temperature),
        top_p: args.top_p.unwrap_or(own.top_p),
        top_k: args.top_k.unwrap_or(own.top_k),
        presence_penalty: args.presence_penalty.unwrap_or(own.presence_penalty),
    }));
    eprintln!(
        "serving {model_id} on http://{}:{}/v1 ({} experts per layer, context {})",
        args.host,
        args.port,
        session.plan().slots,
        max_context,
    );
    let config = Config {
        model_id,
        api_key,
        think: !args.no_think,
        max_tokens: args.max_tokens,
    };
    let log: Arc<dyn Fn(&str) + Send + Sync> = Arc::new(|line: &str| eprintln!("{line}"));
    serve(listener, session, config, &log)?;
    Ok(())
}

/// The context the model at `dir` was trained for: `max_position_embeddings` in its
/// `config.json` (under `text_config` for Qwen), in the pack a bundle holds if there is one.
fn model_context(dir: &Path) -> Result<usize> {
    let pack = ironbark_engine::qwen::packed(dir).with_context(|| dir.display().to_string())?;
    let path = pack.join("config.json");
    let text = std::fs::read_to_string(&path).with_context(|| path.display().to_string())?;
    let config: serde_json::Value = serde_json::from_str(&text).with_context(|| path.display().to_string())?;
    let limit = config.get("text_config").unwrap_or(&config).get("max_position_embeddings");
    match limit.and_then(serde_json::Value::as_u64) {
        Some(n) if n > 0 => usize::try_from(n).context("max_position_embeddings does not fit usize"),
        _ => bail!("{}: no max_position_embeddings; set --max-context", path.display()),
    }
}
