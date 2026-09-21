#!/usr/bin/env bash
# Evaluate one GGUF: wikitext-2 perplexity, optional KL vs a base logits file,
# and a per-layer hidden dump on the fixed probe prompt.
#   run_eval.sh NAME MODEL.gguf [NGL] [extra llama args...]
# Env: EVAL_DIR (default /data/eval), KL_BASE (path to base logits file; if it does not
#      exist it is *written* by this run, otherwise it is *compared against*).
set -euo pipefail
NAME=$1; MODEL=$2; NGL=${3:-99}; shift 3 || shift $#
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
BIN=$ROOT/bin/cuda-e8
EVAL_DIR=${EVAL_DIR:-/data/eval}
export LD_LIBRARY_PATH=$BIN
WIKI=$EVAL_DIR/wikitext-2-raw/wiki.test.raw
KL_CHUNKS=${KL_CHUNKS:-16}
PPL_CHUNKS=${PPL_CHUNKS:-64}

echo "== [$NAME] perplexity c=512 chunks=$PPL_CHUNKS"
$BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 512 -b 512 --chunks $PPL_CHUNKS -ngl "$NGL" "$@" \
  > "$EVAL_DIR/ppl_${NAME}.log" 2>&1
grep -E "Final estimate" "$EVAL_DIR/ppl_${NAME}.log"

if [[ -n "${KL_BASE:-}" ]]; then
  if [[ ! -f "$KL_BASE" ]]; then
    echo "== [$NAME] writing KL base logits ($KL_CHUNKS chunks) -> $KL_BASE"
    $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 512 -b 512 --chunks $KL_CHUNKS -ngl "$NGL" \
      --kl-divergence-base "$KL_BASE" "$@" > "$EVAL_DIR/klbase_${NAME}.log" 2>&1
  else
    echo "== [$NAME] KL divergence vs $KL_BASE"
    $BIN/llama-perplexity -m "$MODEL" -f "$WIKI" -c 512 -b 512 --chunks $KL_CHUNKS -ngl "$NGL" \
      --kl-divergence-base "$KL_BASE" --kl-divergence "$@" > "$EVAL_DIR/kl_${NAME}.log" 2>&1
    grep -E "Mean|Median|KLD|Same top|Maximum|99.0%|PPL" "$EVAL_DIR/kl_${NAME}.log" | tail -25
  fi
fi

echo "== [$NAME] hidden dump"
rm -rf "$EVAL_DIR/hidden_${NAME}"
$ROOT/tools/flash_next_ternary/dump_hidden/dump_hidden -m "$MODEL" -f "$EVAL_DIR/probe_prompt.txt" \
  -c 2048 -b 2048 -ub 2048 -ngl "$NGL" --dump-dir "$EVAL_DIR/hidden_${NAME}" "$@" \
  > "$EVAL_DIR/dump_${NAME}.log" 2>&1
grep -E "dumped" "$EVAL_DIR/dump_${NAME}.log"
