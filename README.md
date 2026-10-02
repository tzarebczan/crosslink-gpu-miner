# crosslink-gpu-miner

Equihash 200,9 miner for the lab crosslink pool. The beast desktop already
runs this against `100.86.64.115:1234` as worker `tmDiX....gpu`. This tree
is the same source, including the fat-pointer block check, so another
machine can build it and join.

The pool address is only reachable on Tailscale. The stratum username is a
worker name. The coinbase pays the address baked into the node template.

## On a laptop

```bash
git clone https://github.com/tzarebczan/crosslink-gpu-miner.git
cd crosslink-gpu-miner
./handoff.sh
```

`handoff.sh` checks the pool, builds for the GPU it finds, and mines in
the foreground. The worker name is this machine's hostname, so a ThinkPad
`t16` shows up as `tmDiX....t16` and does not share the desktop's `.gpu` slot.

AMD needs ROCm (`hipcc` and `rocminfo`). The script reads the gfx name
itself. Override it with `AMDGPU_ARCH=gfx1103` if rocminfo is quiet.

NVIDIA needs `nvcc`. The kernels compile through `gpu_compat.h`. That path
is in the source and has not been timed on a laptop yet.

`GHOST_GPU_DUTY` is how much of each half-second the GPU mines. The script
uses 40 so a laptop stays usable. Full speed:

```bash
GHOST_GPU_DUTY=99 ./handoff.sh
```

Stop it with Ctrl-C. One miner per GPU. Do not start a second copy on a
machine that is already mining.
