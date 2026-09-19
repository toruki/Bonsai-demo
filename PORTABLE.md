# Portable bundle

`./scripts/make_portable_bundle.sh` packs a self-contained, movable copy of this demo:
the locally built binaries, the CUDA runtime they need, the tracked repository files,
and the sources to rebuild from. The result runs on a machine that has only the NVIDIA
driver — no CUDA toolkit, no compiler, no network.

```bash
./scripts/build_cuda_linux.sh --output cuda-e8    # once, to produce bin/cuda-e8
./scripts/make_portable_bundle.sh                 # -> dist/bonsai-e8.tar.gz
```

Only tracked files travel, so `models/`, `bin/` and untracked local files such as
`.bonsai_token` are excluded by construction; no credentials can ride along. Model
weights are not included (7+ GB) — see below.

The rest of this file is what the receiving side needs to know.

---

## What you received

Self-contained build of the `e8-kv` llama.cpp fork, with the compressed KV-cache
presets (`BONSAI_KV=rk8v4 | rk4v4 | rk4v4-e8 | rk2v4-e8`) wired into the launcher.

## Requirements

- **NVIDIA driver only.** The CUDA runtime (`libcudart`, `libcublas`, `libcublasLt`,
  all v13) is bundled in `bin/cuda-e8/`; no CUDA toolkit installation is needed.
- **A Blackwell GPU (compute capability 12.0).** `bin/cuda-e8` is compiled for
  `sm_120a` only. RTX 5090 and RTX PRO 6000 Blackwell are both GB202 / CC 12.0 and
  are covered. Check any other target with:

  ```bash
  nvidia-smi --query-gpu=name,compute_cap --format=csv
  ```

  If it does not say `12.0`, the bundled binaries will not run; rebuild from
  `src-bundles/` (see below).
- Linux or WSL2, x86_64.

## Running

```bash
tar xzf bonsai-e8.tar.gz
cd bonsai-e8

# fetch the 27B weights (~7.3 GB) -- skip if you already have models/
./scripts/download_models.sh

BONSAI_KV=rk8v4 ./scripts/start_llama_server.sh
```

The launcher finds `bin/cuda-e8` on its own. Then open http://localhost:8080.

If you already have the model elsewhere, point at it directly:

```bash
BONSAI_GGUF=/path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
BONSAI_MMPROJ=/path/to/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf \
BONSAI_KV=rk8v4 ./scripts/start_llama_server.sh
```

## Which preset

Measured on an RTX 5090, 27B, KV at 262,144 tokens, perplexity over 16 chunks (±0.030):

| `BONSAI_KV` | K / V | KV @262k | PPL | notes |
|---|---|---:|---:|---|
| *(unset)* | f16 / f16 | 16384 MiB | 3.0913 | default |
| `rk8v4` | q8_0 / q4_0 | 6656 MiB | 3.0928 | **recommended** — 2.5x smaller, and faster than rk4v4 |
| `rk4v4` | q4_0 / q4_0 | 4608 MiB | 3.0971 | the old `BONSAI_KV4=1` |
| `rk4v4-e8` | q4_0_e8 / q4_0 | 4608 MiB | 3.1001 | **do not use** — same size as rk4v4, strictly worse |
| `rk2v4-e8` | q2_e8 / q4_0 | 3456 MiB | 3.2142 | smallest cache available, ~4% perplexity |

Full background, including why `rk4v4-e8` cannot win: `KV-CACHE.md`.

## Rebuilding (other GPU architectures, or to keep developing)

`src/` holds the sources: a git bundle of this repository (branch `kv-e8-presets`,
full history) and the llama.cpp work as three patches on top of upstream.

```bash
# needs: CUDA toolkit 13.x, cmake, ninja, patchelf, git

SRC=$PWD/src                                     # remember where the sources are

git clone $SRC/bonsai-demo.bundle bonsai-demo    # checks out kv-e8-presets
cd bonsai-demo

git clone -b prism --depth 1 https://github.com/PrismML-Eng/llama.cpp.git
cd llama.cpp && git checkout -b e8-kv && git am $SRC/llama.cpp-patches/*.patch && cd ..

./scripts/build_cuda_linux.sh --output cuda-e8                    # all supported archs
./scripts/build_cuda_linux.sh --output cuda-e8 --archs "89;120a"  # or pick your own
```

`--archs 89` covers Ada (RTX 6000 Ada, L40S, 4090), `120a` covers Blackwell.
The patches apply to `PrismML-Eng/llama.cpp@9a9394a` (tag `prism-b10709-9a9394a`);
if that branch has moved on, `git am -3` will three-way merge them.

## What is in the fork

Branch `e8-kv`, three commits on top of `PrismML-Eng/llama.cpp@9a9394a` (branch `prism`):

- two new ggml KV-cache types, `q4_0_e8` (144) and `q2_e8` (145), ported from ninfer
- their CUDA kernels, plus a whitelist that lets FlashAttention take a narrow K next
  to a `q4_0` V without compiling the full quant matrix
- `--cache-type-k/-v` support, validation, and test coverage

Nothing is pushed anywhere; these bundles are the only copy.
