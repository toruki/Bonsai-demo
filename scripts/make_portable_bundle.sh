#!/bin/bash
# Pack a self-contained, movable copy of this demo: the locally built binaries, the CUDA
# runtime they need, the tracked repository files, and the sources to rebuild from.
#
# The result runs on a machine that has only the NVIDIA driver -- no CUDA toolkit, no
# compiler, no network. Model weights are NOT included (7+ GB); the receiving side runs
# ./scripts/download_models.sh or points BONSAI_GGUF at an existing file.
#
# Usage:
#   ./scripts/make_portable_bundle.sh [options]
#
# Options:
#   --bin-dir NAME     bin/ subdirectory to ship (default: cuda-e8)
#   --output FILE      archive to write (default: dist/<name>.tar.gz)
#   --name NAME        top-level directory inside the archive (default: bonsai-e8)
#   --repo-dir DIR     llama.cpp checkout to take patches from (default: ./llama.cpp)
#   --llama-base REF   base the llama.cpp patches are cut against (default: origin/prism)
#   --no-cuda-libs     do not bundle the CUDA runtime (target then needs the toolkit)
#   --no-src           do not include sources for rebuilding
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/common.sh"
DEMO_DIR="$(resolve_demo_dir)"
cd "$DEMO_DIR"

BIN_DIR="cuda-e8"
OUTPUT=""
NAME="bonsai-e8"
REPO_DIR="./llama.cpp"
LLAMA_BASE="origin/prism"
CUDA_LIBS=1
INCLUDE_SRC=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --bin-dir)      BIN_DIR="$2"; shift 2 ;;
        --output)       OUTPUT="$2"; shift 2 ;;
        --name)         NAME="$2"; shift 2 ;;
        --repo-dir)     REPO_DIR="$2"; shift 2 ;;
        --llama-base)   LLAMA_BASE="$2"; shift 2 ;;
        --no-cuda-libs) CUDA_LIBS=0; shift ;;
        --no-src)       INCLUDE_SRC=0; shift ;;
        -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
        *)              err "unknown option: $1"; exit 1 ;;
    esac
done

OUTPUT="${OUTPUT:-dist/$NAME.tar.gz}"
SRC_BIN="bin/$BIN_DIR"

[ -x "$SRC_BIN/llama-server" ] || {
    err "$SRC_BIN/llama-server not found."
    echo "  Build it first:  ./scripts/build_cuda_linux.sh --output $BIN_DIR"
    exit 1
}

# git archive ships the committed tree, so anything uncommitted -- including brand new,
# untracked files -- would silently not travel. Say so loudly rather than shipping a
# bundle that is quietly missing the change you just made.
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    warn "working tree is not clean; the bundle ships the committed tree ($(git rev-parse --short HEAD)) only."
    git status --short | sed 's/^/    /'
    echo ""
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
ROOT="$STAGE/$NAME"
mkdir -p "$ROOT"

echo "=== Staging tracked files ==="
# Tracked files only: this deliberately excludes models/, bin/, .venv/ and untracked
# local files such as .bonsai_token, so no credentials can ride along.
git archive HEAD | tar -x -C "$ROOT"
echo "  $(git rev-parse --short HEAD) ($(git branch --show-current 2>/dev/null || echo detached))"

echo ""
echo "=== Staging binaries ==="
mkdir -p "$ROOT/bin"
cp -a "$SRC_BIN" "$ROOT/bin/"
echo "  $SRC_BIN -> bin/$BIN_DIR ($(du -sh "$SRC_BIN" | cut -f1))"

if [ "$CUDA_LIBS" -eq 1 ]; then
    echo ""
    echo "=== Bundling the CUDA runtime ==="
    # Whatever ldd actually resolves outside the bin directory, so the list cannot drift.
    # libcuda.so.1 is the driver and is deliberately left out: it belongs to the host.
    _found=0
    for lib in $(ldd "$SRC_BIN/libggml-cuda.so" 2>/dev/null \
                 | awk '/=> \//{print $3}' \
                 | grep -E '/lib(cudart|cublas|cublasLt|nvrtc|nvJitLink|cufft|curand|cusparse|cusolver)\.so' \
                 | sort -u); do
        cp -L "$lib" "$ROOT/bin/$BIN_DIR/"
        echo "  $(basename "$lib") ($(du -shL "$lib" | cut -f1))"
        _found=$((_found + 1))
    done
    if [ "$_found" -eq 0 ]; then
        warn "no CUDA runtime libraries resolved; the target will need a CUDA toolkit."
    fi
fi

if [ "$INCLUDE_SRC" -eq 1 ]; then
    echo ""
    echo "=== Staging sources for rebuilding ==="
    mkdir -p "$ROOT/src"
    _branch="$(git branch --show-current 2>/dev/null || true)"
    git bundle create "$ROOT/src/bonsai-demo.bundle" HEAD ${_branch:+"$_branch"} 2>/dev/null >&2
    echo "  bonsai-demo.bundle ($(du -sh "$ROOT/src/bonsai-demo.bundle" | cut -f1))"

    # The llama.cpp checkout is usually a shallow clone, and a bundle of a shallow repo
    # cannot be cloned back. Ship the work as patches against its upstream base instead.
    if [ -d "$REPO_DIR/.git" ]; then
        _base="$(git -C "$REPO_DIR" rev-parse --verify --quiet "$LLAMA_BASE" || true)"
        if [ -n "$_base" ] && [ "$_base" != "$(git -C "$REPO_DIR" rev-parse HEAD)" ]; then
            mkdir -p "$ROOT/src/llama.cpp-patches"
            git -C "$REPO_DIR" format-patch -o "$(cd "$ROOT/src/llama.cpp-patches" && pwd)" \
                "$_base..HEAD" >/dev/null
            echo "  llama.cpp-patches/ ($(ls "$ROOT/src/llama.cpp-patches" | wc -l) patches on $(echo "$_base" | cut -c1-7))"
        else
            warn "$REPO_DIR has no commits beyond $LLAMA_BASE; no patches included."
        fi
    else
        warn "$REPO_DIR is not a git checkout; no llama.cpp sources included."
    fi
fi

echo ""
echo "=== Packing ==="
mkdir -p "$(dirname "$OUTPUT")"
tar -czf "$OUTPUT" -C "$STAGE" "$NAME"

echo ""
echo "Done: $OUTPUT ($(du -sh "$OUTPUT" | cut -f1))"
echo "  sha256: $(sha256sum "$OUTPUT" | cut -d' ' -f1)"
echo ""
echo "On the target (NVIDIA driver only, no CUDA toolkit needed):"
echo "  tar xzf $(basename "$OUTPUT") && cd $NAME"
echo "  ./scripts/download_models.sh"
echo "  BONSAI_KV=rk8v4 ./scripts/start_llama_server.sh"
