# ironbark: MLX 0.32.2 hardcodes the macOS SDK when compiling its Metal kernels.
# When MLX_METAL_SDK is iphoneos or iphonesimulator, compile them with that SDK
# and probe the Metal version and SDK version from it (with a deployment target of iOS 26.2+
# and Metal 4, MLX then builds its NAX kernels). Runs in the MLX source dir; idempotent.
if(NOT MLX_METAL_SDK OR MLX_METAL_SDK STREQUAL "macosx")
  return()
endif()
if(MLX_METAL_SDK STREQUAL "iphonesimulator")
  set(min "-mios-simulator-version-min=")
else()
  set(min "-mios-version-min=")
endif()

foreach(f mlx/backend/metal/kernels/CMakeLists.txt cmake/extension.cmake)
  file(READ ${f} s)
  string(REPLACE "-sdk macosx metal" "-sdk ${MLX_METAL_SDK} metal" s "${s}")
  string(REPLACE "-mmacosx-version-min=" "${min}" s "${s}")
  file(WRITE ${f} "${s}")
endforeach()

file(READ CMakeLists.txt s)
string(
  REPLACE
    "set(MLX_METAL_VERSION 0)"
    "execute_process(COMMAND zsh -c \"echo __METAL_VERSION__ | xcrun -sdk ${MLX_METAL_SDK} metal ${min}\${CMAKE_OSX_DEPLOYMENT_TARGET} -E -x metal -P - | tail -1\" OUTPUT_VARIABLE MLX_METAL_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)\n    message(STATUS \"Metal version: \${MLX_METAL_VERSION}\")\n    # The iOS SDK version, under the name MLX's NAX kernel checks read (they need >= 26.2).\n    execute_process(COMMAND xcrun -sdk ${MLX_METAL_SDK} --show-sdk-version OUTPUT_VARIABLE MACOS_SDK_VERSION OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)"
    s
    "${s}")
file(WRITE CMakeLists.txt "${s}")
