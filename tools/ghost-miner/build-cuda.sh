#!/bin/bash
# NVIDIA build of the same Equihash kernels. gpu_compat.h maps the HIP
# calls onto CUDA when nvcc compiles hip_impl.hip as CUDA.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NVCC="${NVCC:-nvcc}"
HIP_SRC="$HERE/../../amd_hip/hip_impl.hip"
OUT="$HERE/ghost-miner"

echo "[build-cuda] nvcc=$NVCC arch=native"
"$NVCC" -x cu -arch=native -O3 -std=c++17 \
  -c -o "$HERE/hip_impl.o" "$HIP_SRC"
"$NVCC" -arch=native -O3 -std=c++17 \
  -o "$OUT" "$HERE/hip_impl.o" "$HERE/main.cpp" -pthread
echo "[build-cuda] built $OUT"
