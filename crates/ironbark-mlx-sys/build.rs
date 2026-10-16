extern crate cmake;

use cmake::Config;
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

#[cfg(feature = "metal")]
use std::fs;

#[path = "src/bindgen_config.rs"]
mod bindgen_config;

/// Whether the target is iOS (device or simulator).
fn target_ios() -> bool {
    env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("ios")
}

/// Whether the target is the iOS simulator.
fn target_sim() -> bool {
    env::var("TARGET").is_ok_and(|t| t.ends_with("-sim"))
}

/// Find the clang runtime library path dynamically using xcrun
fn find_clang_rt_path() -> Option<String> {
    // Use xcrun to find the active toolchain path
    let output = Command::new("xcrun").args(["--show-sdk-platform-path"]).output().ok()?;

    if !output.status.success() {
        return None;
    }

    // Get the developer directory which contains the toolchain
    let output = Command::new("xcode-select").args(["--print-path"]).output().ok()?;

    if !output.status.success() {
        return None;
    }

    let developer_dir = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let toolchain_base = format!("{}/Toolchains/XcodeDefault.xctoolchain/usr/lib/clang", developer_dir);

    // Find the clang version directory (it varies by Xcode version)
    let clang_dir = std::fs::read_dir(&toolchain_base).ok()?;
    for entry in clang_dir.flatten() {
        let darwin_path = entry.path().join("lib/darwin");
        let clang_rt_lib = darwin_path.join("libclang_rt.osx.a");
        if clang_rt_lib.exists() {
            return Some(darwin_path.to_string_lossy().to_string());
        }
    }

    None
}

#[cfg(feature = "metal")]
fn mlx_c_key(mlx_c_root: &Path) -> String {
    if mlx_c_root.join(".git").exists() {
        let output = Command::new("git").arg("-C").arg(mlx_c_root).args(["rev-parse", "HEAD"]).output();
        if let Ok(output) = output {
            if output.status.success() {
                let commit = String::from_utf8_lossy(&output.stdout);
                let commit = commit.trim();
                if commit.len() >= 12 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
                    return commit[..12].to_owned();
                }
            }
        }
    }

    let mut files =
        bindgen_config::discover_headers(mlx_c_root).expect("Unable to discover mlx-c headers for the metallib key");
    files.push(mlx_c_root.join("CMakeLists.txt"));
    files.sort();

    let mut hash = 0xcbf29ce484222325_u64;
    for file in files {
        let relative = file.strip_prefix(mlx_c_root).unwrap_or(&file);
        for byte in relative.to_string_lossy().bytes().chain([0]) {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        for byte in fs::read(&file).expect("Unable to hash mlx-c source for the metallib key") {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")[..12].to_owned()
}

#[cfg(feature = "metal")]
fn metallib_dir(mlx_c_root: &Path) -> PathBuf {
    if let Some(path) = env::var_os("MLX_RS_METAL_PATH") {
        return PathBuf::from(path);
    }

    let home = env::var_os("HOME").expect("HOME must be set when MLX_RS_METAL_PATH is not provided");
    PathBuf::from(home).join(".mlx").join("lib").join(if target_ios() {
        // The iOS metallib is compiled with the iOS SDK; keep it apart from the macOS one.
        // Per deployment target too: from iOS 26.2 the metallib holds the NAX kernels.
        let min = env::var("IPHONEOS_DEPLOYMENT_TARGET").unwrap_or_else(|_| "18.0".into());
        format!("{}-{}{min}", mlx_c_key(mlx_c_root), if target_sim() { "iossim" } else { "ios" })
    } else {
        mlx_c_key(mlx_c_root)
    })
}

fn build_and_link_mlx_c() {
    let mlx_c_root = Path::new("src/mlx-c");
    let mut config = Config::new(mlx_c_root);
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");

    // Use Xcode's clang to ensure compatibility with the macOS SDK
    config.define("CMAKE_C_COMPILER", "/usr/bin/cc");
    config.define("CMAKE_CXX_COMPILER", "/usr/bin/c++");

    if target_ios() {
        let sdk = if target_sim() { "iphonesimulator" } else { "iphoneos" };
        let min = env::var("IPHONEOS_DEPLOYMENT_TARGET").unwrap_or_else(|_| "18.0".into());
        println!("cargo:rerun-if-env-changed=IPHONEOS_DEPLOYMENT_TARGET");
        config.define("CMAKE_SYSTEM_NAME", "iOS");
        config.define("CMAKE_OSX_SYSROOT", sdk);
        config.define("CMAKE_OSX_ARCHITECTURES", "arm64");
        config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &min);
        // Read by mlx-c's patch step: which SDK compiles the Metal kernels.
        config.define("MLX_METAL_SDK", sdk);
    }

    // The C examples are removed from the vendored copy.
    config.define("MLX_C_BUILD_EXAMPLES", "OFF");

    #[cfg(debug_assertions)]
    {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    #[cfg(not(debug_assertions))]
    {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    config.define("MLX_BUILD_METAL", "OFF");
    config.define("MLX_BUILD_ACCELERATE", "OFF");

    #[cfg(feature = "metal")]
    {
        config.define("MLX_BUILD_METAL", "ON");
        let metallib_dir = metallib_dir(mlx_c_root);
        fs::create_dir_all(&metallib_dir).expect("Unable to create the MLX metallib directory");
        config.define("MLX_METAL_PATH", &metallib_dir);
    }

    #[cfg(feature = "accelerate")]
    {
        config.define("MLX_BUILD_ACCELERATE", "ON");
    }

    // build the mlx-c project
    let dst = config.build();

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
    // mlx's GGUF io depends on the vendored gguflib archive, which cmake leaves
    // in the mlx build tree instead of installing next to libmlx.
    println!("cargo:rustc-link-search=native={}/build/_deps/mlx-build/mlx/io", dst.display());
    println!("cargo:rustc-link-lib=static=mlx");
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=gguflib");

    println!("cargo:rustc-link-lib=c++");
    println!("cargo:rustc-link-lib=dylib=objc");
    println!("cargo:rustc-link-lib=framework=Foundation");

    #[cfg(feature = "metal")]
    {
        println!("cargo:rustc-link-lib=framework=Metal");
        let metallib = metallib_dir(mlx_c_root).join("mlx.metallib");
        if !metallib.exists() {
            println!(
                "cargo:warning=mlx.metallib was not created at {}; Metal operations may fail at runtime",
                metallib.display()
            );
        }
    }

    #[cfg(feature = "accelerate")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    // On iOS the app's link (Xcode's clang driver) adds clang_rt.ios itself, and the archive is
    // fat, which rustc cannot bundle.
    if target_ios() {
        return;
    }
    // Link against Xcode's clang runtime for ___isPlatformVersionAtLeast symbol
    // This is needed on macOS 26+ where the bundled LLVM runtime may be outdated
    // See: https://github.com/conda-forge/llvmdev-feedstock/issues/244
    if let Some(clang_rt_path) = find_clang_rt_path() {
        println!("cargo:rustc-link-search={}", clang_rt_path);
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=MLX_RS_METAL_PATH");
    build_and_link_mlx_c();

    let mlx_c_root = PathBuf::from("src/mlx-c");
    let headers = bindgen_config::discover_headers(&mlx_c_root).expect("Unable to discover headers");
    for header in &headers {
        let relative = bindgen_config::relative_header(&mlx_c_root, header).expect("Unable to record header");
        println!("cargo:rerun-if-changed={}", mlx_c_root.join(relative).display());
    }
    let mut builder = bindgen_config::builder(&mlx_c_root, &headers);
    if target_ios() {
        // libclang parses the headers for the host unless told the target and its SDK.
        let sdk = if target_sim() { "iphonesimulator" } else { "iphoneos" };
        let path = Command::new("xcrun")
            .args(["--sdk", sdk, "--show-sdk-path"])
            .output()
            .expect("Unable to run xcrun for the iOS SDK path");
        let triple = if target_sim() {
            "arm64-apple-ios-simulator"
        } else {
            "arm64-apple-ios"
        };
        builder = builder
            .clang_arg(format!("--target={triple}"))
            .clang_arg("-isysroot")
            .clang_arg(String::from_utf8_lossy(&path.stdout).trim());
    }
    let bindings = builder
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).expect("Couldn't write bindings!");
}
