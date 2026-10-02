#!/bin/bash
# Linux HIP build of the standalone ghost-miner (no Boost, no nheqminer).
# Same pipeline as build.bat, without the MSVC runtime flag.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ARCH="${1:-gfx1201}"
HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
HIP_SRC="$HERE/../../amd_hip/hip_impl.hip"
OUT="$HERE/ghost-miner"

echo "[build-linux] arch=$ARCH hipcc=$HIPCC"
"$HIPCC" --offload-arch="$ARCH" -O3 -std=c++17 -fPIC \
  -c -o "$HERE/hip_impl.o" "$HIP_SRC"
"$HIPCC" --offload-arch="$ARCH" -O3 -std=c++17 \
  -o "$OUT" "$HERE/hip_impl.o" "$HERE/main.cpp" -pthread
echo "[build-linux] built $OUT"
