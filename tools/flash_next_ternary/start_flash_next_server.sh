#!/usr/bin/env bash
# llama-server for Qwen3.8-Flash-Next (unsloth UD-IQ3_XXS) on the fork build (build-q4x) with the
# host expert cache. Measured on an RTX 5090 (32 GB, WSL2): 2k 73 t/s, 32k 60 t/s, 128k 53 t/s,
# 256k 45 t/s decode; prefill 420-510 t/s; peak VRAM at 256k with 6,500 slots = 28.5 GB.
#
#   FLASH_CTX    context (default 262144)
#   FLASH_SLOTS  expert cache slots (1.88 MiB each; default by context: <=32k 8175, <=128k 7000, else 6500)
#   FLASH_HOST / FLASH_PORT (default 127.0.0.1 / 8080)
#   FLASH_MODEL  GGUF path
# Any extra arguments are passed to llama-server (e.g. --reasoning-budget 2048, --alias name).
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN="$ROOT/llama.cpp/build-q4x/bin"
MODEL=${FLASH_MODEL:-/data/models/qwen3.8-flash-next/hf-cache/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS/Qwen3.8-Flash-Next-UD-IQ3_XXS-00001-of-00003.gguf}
CTX=${FLASH_CTX:-262144}
if [ -z "${FLASH_SLOTS:-}" ]; then
    if   [ "$CTX" -le 32768 ];  then FLASH_SLOTS=8175
    elif [ "$CTX" -le 131072 ]; then FLASH_SLOTS=7000
    else                             FLASH_SLOTS=6500
    fi
fi

# page-lock the mapped expert weights: the cache fills from them at PCIe speed and prefill is 2x faster
export GGML_CUDA_REGISTER_HOST=1
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

echo "Flash-Next: ctx $CTX, expert cache $FLASH_SLOTS slots ($(awk "BEGIN{printf \"%.1f\", $FLASH_SLOTS*1.88/1024}") GiB)"

exec "$BIN/llama-server" -m "$MODEL" \
    -c "$CTX" -b 1024 -ub 1024 -ngl 99 -fa on \
    --expert-cache-slots "$FLASH_SLOTS" \
    -np 1 -t 4 -tb 16 \
    --jinja \
    --host "${FLASH_HOST:-127.0.0.1}" --port "${FLASH_PORT:-8080}" \
    "$@"
