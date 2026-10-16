#!/bin/zsh
# Build the Rust library for iOS, stage the Metal kernels and prompts, generate the Xcode
# project, and build (and with --install, install) the app on the first connected iPhone.
set -euo pipefail
cd "${0:A:h}"
root=..
: ${IRONBARK_TEAM:?set IRONBARK_TEAM to your Apple Team ID (Xcode > Settings > Accounts)}

# iOS 26.2+ and Metal 4: MLX's NAX kernels (GPU neural accelerators, A19 / M5) for prefill.
export IPHONEOS_DEPLOYMENT_TARGET=${IOS_MIN:-26.2}
cargo build --release -p ironbark-ffi --target aarch64-apple-ios --manifest-path $root/Cargo.toml

mkdir -p build/prompts
metallib=$(ls -t ~/.mlx/lib/*-ios$IPHONEOS_DEPLOYMENT_TARGET/mlx.metallib | head -1)
cp "$metallib" build/mlx.metallib
for d in prose rust; do cp $root/prompts/$d.txt build/prompts/; done

xcodegen generate --quiet
xcodebuild -project IronbarkBench.xcodeproj -scheme IronbarkBench -configuration Release \
  -destination 'generic/platform=iOS' -derivedDataPath build/dd -allowProvisioningUpdates \
  -quiet build

app=build/dd/Build/Products/Release-iphoneos/IronbarkBench.app
echo "built $app"
if [[ ${1:-} == --install ]]; then
  dev=$(xcrun devicectl list devices | awk '/iPhone/ && /available|connected/ {print $3; exit}')
  xcrun devicectl device install app --device "$dev" "$app"
  # An install can come with a new, empty data container: check the packs are still there.
  if ! xcrun devicectl device info files --device "$dev" --domain-type appDataContainer \
      --domain-identifier dev.ironbark.bench --subdirectory Documents 2>/dev/null | grep -q '/experts.json'; then
    print -u2 "warning: no pack in the app's Documents (the models are gone); copy one back:"
    print -u2 "  xcrun devicectl device copy to --device $dev --domain-type appDataContainer" \
      "--domain-identifier dev.ironbark.bench --source ${root:A}/models/maple --destination Documents/maple"
  fi
fi
