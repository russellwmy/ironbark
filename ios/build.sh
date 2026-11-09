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
  # The first paired physical iPhone, by UDID from devicectl's JSON (the text table's columns
  # shift when a device has no hostname).
  devices=$(mktemp)
  xcrun devicectl list devices --quiet --json-output $devices >/dev/null
  dev=$(jq -r 'first(.result.devices[] | select(.hardwareProperties.reality == "physical"
    and .hardwareProperties.deviceType == "iPhone" and .connectionProperties.pairingState == "paired")
    | .hardwareProperties.udid) // empty' $devices)
  rm -f $devices
  [[ -n $dev ]] || { print -u2 "no paired iPhone found (xcrun devicectl list devices)"; exit 1; }
  xcrun devicectl device install app --device "$dev" "$app"
  # An install can come with a new, empty data container: check the packs are still there.
  # Listing right after an install can fail, so retry; read it whole (grep -q in a pipe can
  # SIGPIPE devicectl, which pipefail counts as a failure).
  files=""
  for try in 1 2 3 4 5; do
    files=$(xcrun devicectl device info files --device "$dev" --domain-type appDataContainer \
      --domain-identifier dev.ironbark.bench --subdirectory Documents 2>/dev/null) && break
    files=""
    sleep 2
  done
  if [[ -z $files ]]; then
    print -u2 "warning: could not list the app's Documents to check for packs"
  elif [[ $files != */experts.json* ]]; then
    print -u2 "warning: no pack in the app's Documents (the models are gone); copy one back:"
    print -u2 "  xcrun devicectl device copy to --device $dev --domain-type appDataContainer" \
      "--domain-identifier dev.ironbark.bench --source ~/.ironbark/models/maple --destination Documents/maple"
  fi
fi
