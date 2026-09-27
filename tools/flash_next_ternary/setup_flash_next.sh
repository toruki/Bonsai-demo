#!/usr/bin/env bash
# Build the Flash-Next fork (branch qwen4exp-port of toruki/llama.cpp) into a directory of your choice.
#   usage: setup_flash_next.sh <target-dir> [cuda-arch]      (default arch 120a = RTX 5090)
# Result: <target-dir>/llama.cpp/build/bin/llama-server; run it with
#   FLASH_BIN_DIR=<target-dir>/llama.cpp/build/bin tools/flash_next_ternary/start_flash_next_server.sh
# Needs: git, cmake >= 3.18, a CUDA toolkit (nvcc on PATH or /usr/local/cuda*/bin), a C++17 compiler.
# The model is not converted: the unsloth UD-IQ3_XXS GGUF is used as published
#   (huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF, UD-IQ3_XXS/, 3 files, ~30 GB); point FLASH_MODEL at the first file.
set -euo pipefail
TARGET=${1:?target directory}
ARCH=${2:-120a}
REPO=${FLASH_REPO:-git@github.com:toruki/llama.cpp.git}
BRANCH=${FLASH_BRANCH:-qwen4exp-port}

mkdir -p "$TARGET"
if [ ! -d "$TARGET/llama.cpp/.git" ]; then
    git clone --branch "$BRANCH" --single-branch "$REPO" "$TARGET/llama.cpp"
else
    git -C "$TARGET/llama.cpp" fetch -q origin "$BRANCH" && git -C "$TARGET/llama.cpp" checkout -q "$BRANCH" && git -C "$TARGET/llama.cpp" pull -q --ff-only
fi

# prefer a toolkit directory (/usr/local/cuda-X.Y) over a bare nvcc symlink in /usr/local/bin, and resolve symlinks
NVCC=$(ls -d /usr/local/cuda*/bin/nvcc 2>/dev/null | sort -V | tail -1 || true)
[ -n "$NVCC" ] || NVCC=$(command -v nvcc || true)
[ -n "$NVCC" ] || { echo "nvcc not found"; exit 1; }
NVCC=$(readlink -f "$NVCC")

CUDA_ROOT=$(cd "$(dirname "$NVCC")/.." && pwd)
echo "using CUDA toolkit at $CUDA_ROOT"
export PATH="$CUDA_ROOT/bin:$PATH"

cd "$TARGET/llama.cpp"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
      -DCMAKE_CUDA_COMPILER="$NVCC" -DCUDAToolkit_ROOT="$CUDA_ROOT" -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
      > build-configure.log 2>&1 || { tail -20 build-configure.log; exit 1; }
cmake --build build -j"$(nproc)" --target llama-server llama-cli llama-perplexity > build-build.log 2>&1 || { grep -E "error" build-build.log | head; exit 1; }
echo "built: $TARGET/llama.cpp/build/bin ($(git rev-parse --short HEAD), arch $ARCH)"
