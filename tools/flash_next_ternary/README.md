# Flash-Next on the fork — build and run

Qwen3.8-Flash-Next (unsloth `UD-IQ3_XXS` GGUF, used as published, no conversion) on the PrismML llama.cpp fork
branch `qwen4exp-port` (github.com/toruki/llama.cpp) with the host expert cache. Verified on one machine: RTX 5090
32 GB, WSL2, CUDA 13.3, compute capability 12.0 (`120a`). Only the CUDA backend has the expert cache and the batched top-k.

1. Build into any directory (clones the fork branch, configures with CUDA, builds llama-server/llama-cli/llama-perplexity):

       tools/flash_next_ternary/setup_flash_next.sh /path/to/dir 120a

2. Get the model: `huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF`, folder `UD-IQ3_XXS/` (3 files, ~30 GB).

3. Run:

       FLASH_BIN_DIR=/path/to/dir/llama.cpp/build/bin FLASH_MODEL=/path/to/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf \
       FLASH_CTX=131072 tools/flash_next_ternary/start_flash_next_server.sh [--reasoning-budget 2048 ...]

   `FLASH_CTX` (default 262144), `FLASH_SLOTS` (expert cache slots, 1.88 MiB each; default 8175 / 7000 / 6500 by context),
   `FLASH_UB` (2048 up to 128k, 1024 above), `FLASH_HOST` / `FLASH_PORT`. Extra arguments go to llama-server.
   Peak VRAM at 256k with 6,500 slots is 28.5 GB; keep other GPU processes off while running.

4. Check (optional): `q4x_eval`-style tests live in `qsa_edit_test/` (KV edits) and the KL recipe in docs/expert_cache_plan.md.

Measured (this machine): decode 2k 73 t/s, 32k 60, 128k 53, 256k 45 t/s; prefill 128k 862 t/s (ub 2048), 256k 498 t/s.
Details and history: docs/expert_cache_plan.md.
