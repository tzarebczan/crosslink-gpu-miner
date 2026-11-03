# crosslink-gpu-miner

Equihash 200,9 miner for a Crosslink featurenet node. It speaks the extra
`mining.notify` fields from
[crosslink-snomp](https://github.com/tzarebczan/crosslink-snomp): the BFT
fat pointer and the block target. A stock Equihash miner does not, and it
will not land blocks on this chain.

The stratum username is a worker name. The coinbase is paid by the node,
from `mining.miner_address` in that node's config (or the wallet address
the node derived, if that key is unset). It is not paid to the username.

There is no prebuilt binary. AMD needs ROCm (`hipcc` and `rocminfo`).
NVIDIA needs `nvcc`. An Intel GPU cannot build this miner. One process per
GPU.

## Your own node

Run the pool from crosslink-snomp against the local node first, then:

```bash
git clone https://github.com/tzarebczan/crosslink-gpu-miner.git
cd crosslink-gpu-miner
POOL_HOST=127.0.0.1 WALLET=tmYourTransparentAddress ./handoff.sh
```

`WALLET` is the worker name the pool displays. Use the same transparent
address the node pays, plus a dot and a worker suffix. The script appends
`.<hostname>` itself, so pass the address only.

`GHOST_GPU_DUTY` is how much of each half-second the GPU mines, from 1 to
100. The script defaults to 40 so a laptop stays usable. A dedicated card:

```bash
GHOST_GPU_DUTY=99 POOL_HOST=127.0.0.1 WALLET=tmYourTransparentAddress ./handoff.sh
```

Stop it with Ctrl-C. If Ctrl-C does not return, the process is inside a
GPU call; stop it with SIGTERM. Do not start a second copy on the same GPU.

`./handoff.sh` with no variables uses the lab pool on Tailscale
(`100.86.64.115:1234`) and a lab worker prefix. That pool is not part of
setting up a new node. Set `POOL_HOST` and `WALLET`.

## Build

AMD: the script reads the gfx name from `rocminfo`. If that is empty, set
`AMDGPU_ARCH` (`gfx1201`, `gfx1103`, and so on).

NVIDIA: `nvcc -arch=native` through `amd_hip/gpu_compat.h`.

The on-screen Sol/s count is every Equihash solution, taken before the
pool's share check. At share difficulty 0.001 about one solution in eight
is submitted, so the pool's accepted-share rate is lower than Sol/s.
That gap is the filter, not a failed miner.
