//! Download a model repo from Hugging Face with `curl` (ships with macOS; no HTTP stack in the
//! binary).

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, anyhow, bail, ensure};

/// The MLX checkpoint `ironbark pack` reads.
pub(crate) const DEFAULT_REPO: &str = "deepgrove/maple-preview-2bit-mlx";

const HUB: &str = "https://huggingface.co";

/// Models dir: `--dir`, else `$IRONBARK_MODELS`, else `~/.ironbark/models`.
pub(crate) fn models_dir(flag: Option<PathBuf>) -> Result<PathBuf> {
    if let Some(d) = flag {
        return Ok(d);
    }
    if let Some(d) = std::env::var_os("IRONBARK_MODELS").filter(|d| !d.is_empty()) {
        return Ok(PathBuf::from(d));
    }
    let home = std::env::home_dir().ok_or_else(|| anyhow!("no home dir; pass --dir"))?;
    Ok(home.join(".ironbark").join("models"))
}

/// One file of the repo.
#[derive(Debug)]
struct Entry {
    path: String,
    size: u64,
}

/// Download every file of `repo` at `revision` into `out`. Files already there with the right
/// size are skipped; partial files (`*.part`) resume.
pub(crate) fn download(repo: &str, revision: &str, out: &Path) -> Result<()> {
    let token = std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty());
    let files = list(repo, revision, token.as_deref())?;
    let total: u64 = files.iter().map(|f| f.size).sum();
    println!("{repo}@{revision}: {} files, {:.2} GB -> {}", files.len(), total as f64 / 1e9, out.display());
    for (i, f) in files.iter().enumerate() {
        let dest = out.join(&f.path);
        // The hub lists repo-relative paths; refuse anything that would land outside `out`.
        ensure!(
            Path::new(&f.path).components().all(|c| matches!(c, std::path::Component::Normal(_))),
            "unsafe path in repo listing: {}",
            f.path
        );
        let tag = format!("[{}/{}] {}", i + 1, files.len(), f.path);
        if fs::metadata(&dest).is_ok_and(|m| m.len() == f.size) {
            println!("{tag}: already there");
            continue;
        }
        if let Some(dir) = dest.parent() {
            fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        println!("{tag} ({:.1} MB)", f.size as f64 / 1e6);
        let part = dest.with_file_name(format!(
            "{}.part",
            dest.file_name().map_or_else(Default::default, |n| n.to_string_lossy())
        ));
        // A complete .part from an interrupted run: curl -C - would get a 416, so just rename.
        if !fs::metadata(&part).is_ok_and(|m| m.len() == f.size) {
            let url = format!("{HUB}/{repo}/resolve/{revision}/{}", f.path);
            let mut c = curl(token.as_deref());
            c.args(["--progress-bar", "-C", "-", "-o"]).arg(&part).arg(&url);
            run(c, token.as_deref()).with_context(|| format!("download {url}"))?;
        }
        let got = fs::metadata(&part)?.len();
        ensure!(got == f.size, "{}: got {got} bytes, expected {} (rerun to resume)", f.path, f.size);
        fs::rename(&part, &dest).with_context(|| format!("rename to {}", dest.display()))?;
    }
    Ok(())
}

/// The repo's files (recursive) from the hub API.
fn list(repo: &str, revision: &str, token: Option<&str>) -> Result<Vec<Entry>> {
    let url = format!("{HUB}/api/models/{repo}/tree/{revision}?recursive=true");
    let mut c = curl(token);
    c.args(["--silent", "--show-error"]).arg(&url);
    let body = run(c, token).with_context(|| format!("list {repo} (does it exist? private repos need HF_TOKEN)"))?;
    let v: serde_json::Value = serde_json::from_slice(&body).with_context(|| format!("parse listing from {url}"))?;
    let items = v.as_array().ok_or_else(|| anyhow!("unexpected listing from {url}: {v}"))?;
    let mut files = Vec::new();
    for e in items {
        if e["type"] != "file" {
            continue;
        }
        let path = e["path"].as_str().ok_or_else(|| anyhow!("listing entry without path: {e}"))?;
        let size = e["size"].as_u64().ok_or_else(|| anyhow!("{path}: no size in listing"))?;
        files.push(Entry {
            path: path.to_owned(),
            size,
        });
    }
    ensure!(!files.is_empty(), "{repo}@{revision} has no files");
    Ok(files)
}

/// `curl` that follows redirects, fails on HTTP errors and retries; the token (if any) goes in
/// on stdin so it doesn't show in `ps`.
fn curl(token: Option<&str>) -> Command {
    let mut c = Command::new("curl");
    c.args(["--location", "--fail", "--retry", "5", "--retry-delay", "2"]);
    if token.is_some() {
        c.args(["--header", "@-"]);
    }
    c
}

/// Run `c`, feeding the auth header on stdin; returns stdout (progress goes to the terminal).
fn run(mut c: Command, token: Option<&str>) -> Result<Vec<u8>> {
    c.stdin(if token.is_some() { Stdio::piped() } else { Stdio::null() })
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut child = c.spawn().context("run curl")?;
    if let Some(t) = token
        && let Some(mut stdin) = child.stdin.take()
    {
        writeln!(stdin, "Authorization: Bearer {t}")?;
    }
    let out = child.wait_with_output()?;
    if !out.status.success() {
        bail!("curl failed ({})", out.status);
    }
    Ok(out.stdout)
}
