# ironbark-mlx

Safe Rust API over [MLX](https://github.com/ml-explore/mlx) for inference: arrays, ops, fused kernels, evaluation and compilation. Runs on the Apple GPU (Metal) on macOS and iOS.

```
ironbark-mlx ──▶ ironbark-mlx-sys ──▶ mlx-c ──▶ MLX v0.32.4
```

| Module | What |
|---|---|
| `ops` | array operations, indexing, quantize / dequantize |
| `fast` | RMS norm, layer norm, RoPE, scaled dot-product attention |
| `transforms` | `eval`, `async_eval`, `compile`, `value_and_grad` |
| `memory` | active / peak / cache memory, cache limit |
| `random`, `linalg`, `fft` | sampling and math |

No training API (no `nn` layers, optimizers or losses; `value_and_grad` exists for tools that fine-tune models for ironbark, such as ironwood) and no deprecated `*_device` shims: run an op on another stream with `with_stream` / `with_device`.

License: MIT OR Apache-2.0.
