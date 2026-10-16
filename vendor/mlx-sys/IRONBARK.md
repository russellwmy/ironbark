# mlx-sys 0.6.0, patched for iOS

Copy of crates.io `mlx-sys` 0.6.0 (MIT OR Apache-2.0), used through `[patch.crates-io]`.

| File | Change |
|---|---|
| `build.rs` | iOS target: CMake `CMAKE_SYSTEM_NAME=iOS`, SDK, arm64, deployment target (`IPHONEOS_DEPLOYMENT_TARGET`, default 18.0); metallib in `~/.mlx/lib/<key>-ios<deployment target>` (`-iossim<target>` for the simulator); bindgen with the iOS target and sysroot; no `clang_rt.osx` (Xcode links `clang_rt.ios`); no mlx-c examples |
| `src/mlx-c/CMakeLists.txt` | `PATCH_COMMAND` on the MLX fetch |
| `src/mlx-c/patch-mlx-ios.cmake` | MLX 0.32.2 compiles its Metal kernels with `-sdk macosx`; on iOS use `iphoneos` / `iphonesimulator`, `-mios-version-min`, and probe the Metal and SDK versions from that SDK (so a 26.2+ deployment target builds the NAX kernels) |
| `Cargo.toml` | dropped the example (not vendored) |

macOS builds are unchanged.
