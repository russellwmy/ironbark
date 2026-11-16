//! Embedding speed: load time, then tokens per second for one ~300-token document (median of
//! runs after a warm-up), and MLX memory.
//!
//! usage: `cargo run --release -p ironbark --example embed_bench -- MODEL_DIR [RUNS]`

#![allow(clippy::print_stdout, reason = "an example")]

use std::time::Instant;

use ironbark::{EmbedKind, EmbedOptions, Embedder};

/// About 300 tokens of plain prose.
const TEXT: &str = "A lease is a contract in which the owner of a property lets another party use it for a set \
time in return for rent. The lease names the parties, the property, the rent and when it is due, the deposit, \
and what each side must repair. Most leases also say how either party may end the agreement early: usually \
by written notice some weeks before the date, sometimes with a fee. If the tenant leaves without notice, the \
owner may keep the deposit or claim the rent still owed. If the owner wants the property back, the law of \
most places requires a notice period and, for some reasons, a court order. When a lease ends, the tenant \
returns the keys, the owner inspects the property, and the deposit is returned less the cost of any damage \
beyond normal wear. Disputes about deposits are among the most common between owners and tenants, so both \
should record the state of the property, with dated photos, when the lease starts and when it ends. A \
lease may be renewed on the same terms, renewed with new terms, or left to run on from month to month, in \
which case either side can usually end it with one month's notice. Before signing, read the clauses on \
rent increases, subletting, pets, and who pays for utilities and insurance, and ask for any promise made in \
person to be written into the lease.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let (Some(dir), runs) = (args.get(1), args.get(2).and_then(|r| r.parse().ok()).unwrap_or(20usize)) else {
        return Err("usage: embed_bench MODEL_DIR [RUNS]".into());
    };
    let t = Instant::now();
    let mut e = Embedder::open(dir, EmbedOptions::default())?;
    let load_s = t.elapsed().as_secs_f64();
    let tokens = e.tokens(TEXT, EmbedKind::Document)?.len();
    e.embed(&[TEXT], EmbedKind::Document)?;
    let mut secs: Vec<f64> = (0..runs)
        .map(|_| {
            let t = Instant::now();
            e.embed(&[TEXT], EmbedKind::Document).map(|_| t.elapsed().as_secs_f64())
        })
        .collect::<Result<_, _>>()?;
    secs.sort_by(f64::total_cmp);
    let median = secs[secs.len() / 2];
    let mb = |b: usize| b as f64 / 1e6;
    println!("load {load_s:.2} s");
    println!("{tokens} tokens: {:.1} ms, {:.0} tok/s (median of {runs})", median * 1e3, tokens as f64 / median);
    println!(
        "MLX memory: active {:.0} MB, peak {:.0} MB",
        mb(ironbark_mlx::memory::active_memory()?),
        mb(ironbark_mlx::memory::peak_memory()?)
    );
    Ok(())
}
