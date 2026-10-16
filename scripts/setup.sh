#!/bin/zsh
# Download Maple from Hugging Face, build the CLI, pack the model and verify the pack.
#   scripts/setup.sh              trit pack in models/maple
#   scripts/setup.sh --two-bit    also the 2-bit pack in models/maple-2bit
#   IRONBARK_MODELS=/path scripts/setup.sh   other models dir (default models in the repo)
# Each step is skipped when its output is already there; rerun after a failure.
set -euo pipefail
cd "${0:A:h}/.."

models=${${IRONBARK_MODELS:-models}:A}   # absolute: cargo test runs in each crate dir
ckpt=$models/maple-2bit-mlx   # DeepGrove's MLX checkpoint (~5 GB)
repo=deepgrove/maple-preview-2bit-mlx
two_bit=0
for a in "$@"; do
  case $a in
    --two-bit) two_bit=1 ;;
    *) echo "usage: $0 [--two-bit]" >&2; exit 2 ;;
  esac
done

for t in cargo hf; do
  command -v $t >/dev/null || { echo "missing $t (hf: brew install huggingface-cli)" >&2; exit 1; }
done

mkdir -p $models
echo "== download $repo -> $ckpt (resumes; files already there are skipped)"
hf download $repo --local-dir $ckpt

echo "== build (the first build compiles MLX, ~15 min)"
cargo build --release -p ironbark-cli
bin=target/release/ironbark

pack() {  # pack <out> [--two-bit]
  local out=$1; shift
  if [[ -f $out/manifest.json ]]; then
    echo "== pack: $out (already there)"
  else
    echo "== pack -> $out $*"
    $bin pack --model $ckpt --out $out "$@"
  fi
  echo "== verify $out"
  $bin verify $out
}

pack $models/maple
(( two_bit )) && pack $models/maple-2bit --two-bit

cat <<EOF

Done. Try:
  $bin gen --model $models/maple --experts $models/maple --prompt "The capital of Germany is"
  IRONBARK_MODEL=$models/maple cargo test --release --workspace -- --test-threads=1
EOF
