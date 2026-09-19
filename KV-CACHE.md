# Compressed KV cache (experimental)

⚠️ **Experimental, llama.cpp backend only.** A memory tool, not a speed tool: decode is slightly slower than the default FP16 KV cache.

`BONSAI_KV=<preset>` stores the KV cache in a narrower format than FP16, which is what decides how much context fits on a given card. Per token on the 27B (16 attention layers × 4 KV heads × 256 dims):

| `BONSAI_KV` | K type | V type | per token | 262,144 tokens | PPL | needs |
|---|---|---|---:|---:|---:|---|
| *(unset)* | f16 | f16 | 64 KiB | 16.0 GiB | 3.0913 | — |
| `rk8v4` | `q8_0` | `q4_0` | 26 KiB | 6.5 GiB | 3.0928 | fork build |
| `rk4v4` | `q4_0` | `q4_0` | 18 KiB | 4.5 GiB | 3.0971 | published binaries |
| `rk4v4-e8` | `q4_0_e8` | `q4_0` | 18 KiB | 4.5 GiB | 3.1001 | fork build |
| `rk2v4-e8` | `q2_e8` | `q4_0` | 13.5 KiB | 3.4 GiB | 3.2142 | fork build |

*PPL: Ternary-Bonsai-2-27B-PQ2_0, `llama-perplexity -c 4096` over 16 chunks of this repo's markdown (±0.030), RTX 5090. Lower is better; these are a sanity check on one corpus, not a benchmark.*

**Recommendation:** `rk8v4` is the sweet spot — 2.5x smaller than f16 for a 0.05% perplexity cost. `rk4v4` is the best option on the published binaries. Reach for `rk2v4-e8` only when the extra 25% of KV capacity is what decides whether your context fits at all. **Do not use `rk4v4-e8`**: it is the same size as `rk4v4` and strictly worse (see below).

```bash
BONSAI_KV=rk4v4 ./scripts/start_llama_server.sh
```

`BONSAI_KV4=1` still works and is an alias for `BONSAI_KV=rk4v4`.

Under the hood this passes `--cache-type-k <K> --cache-type-v <V>` (quantized KV requires flash attention, which the scripts already enable). The 27B's hybrid attention already keeps the cache small, so reach for this only at very long contexts on tight machines.

## The E8 presets

`rk4v4-e8` and `rk2v4-e8` come from [ninfer](https://github.com/Neroued/ninfer)'s compressed KV work (original design: [UDPSendToFailed/ninfer-4090](https://github.com/UDPSendToFailed/ninfer-4090), Don-Chad/ninfer-3090 lineage). Both rely on the Hadamard K/V rotation that llama.cpp already applies to any quantized KV cache, and quantize the rotated K vector 8 dimensions at a time:

- **`rk4v4-e8`** projects each rotated 8-vector onto the nearest point of the **E8 lattice** before packing it into the same 4-bit block layout `q4_0` uses (same scale rule as `q4_0`, so the quantizer is the only difference). Same size as `rk4v4`. See the warning below: it does not pay off.
- **`rk2v4-e8`** stores each rotated 8-vector as an **E8 root index (1 byte, 240 roots) plus a packed 4-bit log-radius and 4-bit residual axis** — 2.25 bits per weight, the smallest K cache available here. V stays at `q4_0`.

Both presets keep V at `q4_0`, so they need a build whose FlashAttention kernels accept a K type different from the V type — see below.

### Measured quality, honestly

**`rk4v4-e8` loses to `rk4v4` on every metric, and it always will.** Measured on Hadamard-rotated gaussian vectors (`test-e8-codec`):

| | mean cosine | relative L2 error | PPL |
|---|---:|---:|---:|
| `q4_0` | 0.9964 | 0.0855 | 3.0971 |
| `q4_0_e8` | 0.9920 | 0.1269 | 3.1001 |

This is not a tuning problem, it is arithmetic. `q4_0` rounds each element to the nearest representable value — that *is* the nearest point of the representable grid. `q4_0_e8` projects onto E8 first and only then rounds onto the same grid, so it necessarily lands on a point no closer than the one `q4_0` picked. E8 = D8 ∪ (D8 + ½), and the payoff lives in the half-integer coset, which this block layout has no bit to record. ninfer flags the same limitation in its own source ("a source-compatible approximation, not an exact E8 lattice representation").

Recovering it would need a wider block: one coset bit per 8 values is +3% storage against E8's ≈0.65 dB (≈7% RMS) packing gain over the cubic lattice — a net win on paper, but a thin one, and a new format. **The preset is kept for completeness and for parity with ninfer; there is no reason to prefer it over `rk4v4`.**

**`rk2v4-e8` is the one that earns its place**: at 2.25 bits per K weight it costs about 4% perplexity, and no other preset gets the cache that small. ninfer quotes ≈96.2% cosine for it on real attention activations; on isotropic gaussian input it lands near 0.89, and an exhaustive search over all 240 roots × 16 axes reaches only 0.897 on the same data — so that is the codebook's intrinsic limit, not an implementation gap.

## Building the fork

Everything except `rk4v4` needs a locally built llama.cpp with the E8 types compiled in. The launcher probes `llama-server --help` and refuses the preset if the binary does not have them.

```bash
./scripts/build_cuda_linux.sh --output cuda-e8            # builds ./llama.cpp into bin/cuda-e8
BONSAI_BIN_DIR=bin/cuda-e8 BONSAI_KV=rk4v4-e8 ./scripts/start_llama_server.sh
```

`build_cuda_linux.sh` clones `PrismML-Eng/llama.cpp` (branch `prism`) into `./llama.cpp` if it is not there; pass `--repo-url` / `--branch` to point it somewhere else. Windows has no fork build script yet, so the `.ps1` launcher accepts `rk4v4` only.

## Better quality for `rk4v4`: the mean-centering bias

4-bit quantization of the K cache loses a little accuracy on channels whose activations have a nonzero mean. A small **model-specific calibration bias** fixes most of that at zero decode-time cost (one subtract when the cache is written). Build it once:

```bash
./scripts/make_kv_bias.sh
BONSAI_KV=rk4v4 ./scripts/start_llama_server.sh
```

The script runs `llama-kv-mean-center` (included in the prebuilt binaries) over a calibration text and writes `<Model>-kv-bias.gguf` next to the model weights. The server picks the bias up automatically whenever `BONSAI_KV=rk4v4` is set; without a bias, the 4-bit cache still runs, just with slightly lower quality.

Notes:

- **The bias applies to `rk4v4` only.** It is calibrated with the K-rotation *disabled*, while the E8 presets depend on that rotation being *on*, so the two cannot be combined; the loader rejects the mismatch by design.
- **Calibration does not need much data.** The script ships a tiny built-in synthetic corpus; for best results pass your own text file as the first argument, representative of your workload:

  ```bash
  ./scripts/make_kv_bias.sh my_corpus.txt
  ```

- **The bias is model-specific.** Re-run the script after switching `BONSAI_FAMILY` / `BONSAI_MODEL`.

Full background and the manual command flow: [PrismML-Eng/llama.cpp#54](https://github.com/PrismML-Eng/llama.cpp/issues/54).
