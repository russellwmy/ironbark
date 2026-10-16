# ironbark-mlx-sys

Rust bindings to the [mlx-c](https://github.com/ml-explore/mlx-c) API, generated with bindgen. The build compiles mlx-c and MLX v0.32.2 (fetched by CMake) for macOS or iOS.

## Contents

- [iOS](#ios)
- [Metal library location](#metal-library-location)
- [docs.rs](#docsrs)

## iOS

MLX 0.32.2 compiles its Metal kernels with `-sdk macosx`. For iOS targets (`aarch64-apple-ios`, `aarch64-apple-ios-sim`) this crate changes that:

| File | Change |
|---|---|
| `build.rs` | CMake `CMAKE_SYSTEM_NAME=iOS`, SDK, arm64, deployment target from `IPHONEOS_DEPLOYMENT_TARGET` (default 18.0); bindgen with the iOS target and sysroot; no `clang_rt.osx` (Xcode links `clang_rt.ios`); no mlx-c examples |
| `src/mlx-c/CMakeLists.txt` | `PATCH_COMMAND` on the MLX fetch |
| `src/mlx-c/patch-mlx-ios.cmake` | `iphoneos` / `iphonesimulator` SDK and `-mios-version-min`; Metal and SDK versions probed from that SDK, so a 26.2+ deployment target builds the NAX kernels |

macOS builds are unchanged. Upstream MLX issue: ml-explore/mlx#3915.

## Metal library location

| Target | `mlx.metallib` in |
|---|---|
| macOS | `~/.mlx/lib/<key>/` |
| iOS device | `~/.mlx/lib/<key>-ios<deployment target>/` |
| iOS Simulator | `~/.mlx/lib/<key>-iossim<deployment target>/` |

`<key>` is a hash of the mlx-c headers and CMake files. The stable location lets binaries from `cargo install` keep loading it after Cargo removes the build directory. `MLX_RS_METAL_PATH` overrides the directory (no `HOME` read or write, for sandboxed and Nix builds).

## docs.rs

docs.rs builds with no network and no Apple SDK, so it can't fetch MLX or run bindgen against Xcode. With `DOCS_RS` set, `build.rs` skips both and copies `src/docs_bindings.rs` (the macOS bindings, kept in the repo) to `OUT_DIR`.

```
 normal build ──▶ CMake (MLX) ──▶ bindgen ──▶ OUT_DIR/bindings.rs ──▶ differs from src/docs_bindings.rs? ──▶ warning
 DOCS_RS=1    ─────────────────────────────▶ copy src/docs_bindings.rs ──▶ OUT_DIR/bindings.rs
```

After updating mlx-c, a macOS build warns until you copy the generated bindings over `src/docs_bindings.rs`.
