#!/bin/bash
# Build ghost-miner on this machine and mine at the lab crosslink pool.
# The pool is on the lab tailnet (100.86.64.115:1234). Coinbase pays the
# address in the node template; the name after the dot is only the worker.
#
#   ./handoff.sh
#   GHOST_GPU_DUTY=99 ./handoff.sh
#
# Duty is how much of each half-second the GPU mines. 100 is full speed.
# Laptops default to 40.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
POOL_HOST="${POOL_HOST:-100.86.64.115}"
POOL_PORT="${POOL_PORT:-1234}"
WALLET="${WALLET:-tmDiXKm6zVgoBCn2A5Yccx74x9MszJa7UjA}"
WORKER="${WORKER:-$(hostname -s)}"
DUTY="${GHOST_GPU_DUTY:-40}"
DEVICE="${DEVICE:-0}"
BIN="$ROOT/tools/ghost-miner/ghost-miner"

if ! timeout 4 bash -c "echo >/dev/tcp/${POOL_HOST}/${POOL_PORT}" 2>/dev/null; then
  echo "Cannot reach ${POOL_HOST}:${POOL_PORT}."
  echo "Connect Tailscale so this machine can see the lab, then run this again."
  exit 1
fi

build_hip() {
  local hipcc="${HIPCC:-}"
  if [ -z "$hipcc" ]; then
    if command -v hipcc >/dev/null 2>&1; then
      hipcc="$(command -v hipcc)"
    elif [ -x /opt/rocm/bin/hipcc ]; then
      hipcc=/opt/rocm/bin/hipcc
    else
      return 1
    fi
  fi
  local arch="${AMDGPU_ARCH:-}"
  if [ -z "$arch" ]; then
    local rocminfo=""
    command -v rocminfo >/dev/null 2>&1 && rocminfo="$(command -v rocminfo)"
    [ -z "$rocminfo" ] && [ -x /opt/rocm/bin/rocminfo ] && rocminfo=/opt/rocm/bin/rocminfo
    if [ -n "$rocminfo" ]; then
      arch="$("$rocminfo" 2>/dev/null | awk '/Name:[[:space:]]+gfx/{print $2; exit}')"
    fi
  fi
  if [ -z "$arch" ]; then
    echo "ROCm is installed but rocminfo did not report a gfx name."
    echo "Set AMDGPU_ARCH (for example gfx1103 or gfx1201) and run this again."
    exit 1
  fi
  echo "[handoff] AMD $arch"
  HIPCC="$hipcc" "$ROOT/tools/ghost-miner/build-linux.sh" "$arch"
}

build_cuda() {
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "NVIDIA GPU found, but nvcc is not installed."
    echo "Install the CUDA toolkit, then run this again."
    exit 1
  fi
  echo "[handoff] NVIDIA (nvcc -arch=native)"
  "$ROOT/tools/ghost-miner/build-cuda.sh"
}

if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  build_cuda
elif build_hip; then
  :
else
  echo "No HIP or CUDA toolchain found."
  echo "AMD: install ROCm so hipcc and rocminfo are on PATH."
  echo "NVIDIA: install the CUDA toolkit so nvcc is on PATH."
  exit 1
fi

echo "[handoff] worker ${WALLET}.${WORKER} duty ${DUTY} -> ${POOL_HOST}:${POOL_PORT}"
export GHOST_GPU_DUTY="$DUTY"
exec "$BIN" -l "${POOL_HOST}:${POOL_PORT}" -u "${WALLET}.${WORKER}" -p x -d "$DEVICE"
