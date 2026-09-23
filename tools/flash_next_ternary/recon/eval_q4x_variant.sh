#!/usr/bin/env bash
# Evaluate a Flash-Next variant whose routed experts in some layers were replaced by
# trained/PTQ ternary values (experts_L{N}.npz from q4x_progressive.py).
#   eval_q4x_variant.sh <name> <values-dir> <first> <last>      (name=base: IQ3_XXS reference only)
# Writes PPL + KL (32 x 512 wikitext-2 test, vs the IQ3_XXS logits) and ffn_moe_topk dumps.
set -uo pipefail
export LD_LIBRARY_PATH=/usr/local/cuda-13.3/lib64
NAME=$1; VAL=${2:-}; FIRST=${3:-0}; LAST=${4:-0}
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
B=$ROOT/llama.cpp/build-q4x/bin; T=$HERE/../dump_hidden/dump_hidden_q4x
PY=${PY:-python3}
SRC=/data/models/qwen3.8-flash-next/hf-cache/hub/models--unsloth--Qwen3.8-Flash-Next-GGUF/snapshots/c8b5954a88c2775c546b92593eda40ea041d3176/UD-IQ3_XXS
EV=${EVAL_DIR:-/home/sohey/AI/LLM/q4x_eval}; KLB=$EV/kl_base_iq3.bin
TXT=/data/eval/wikitext-2-raw/wiki.test.raw
S=Qwen3.8-Flash-Next-UD-IQ3_XXS
step() { echo "[$(date +%H:%M:%S)] $*"; }

if [ "$NAME" = base ]; then
  M=$SRC/$S-00001-of-00003.gguf
  step "base: PPL + KL base logits"
  $B/llama-perplexity -m $M -f $TXT -c 512 -b 512 --chunks 32 -ngl 12 --kl-divergence-base $KLB > $EV/ppl_base.log 2>&1
  grep "Final estimate" $EV/ppl_base.log
else
  D=$EV/model_$NAME; mkdir -p $D
  LAYERS=$(seq $FIRST $LAST)
  N=""; for l in $LAYERS; do N="$N blk.$l.ffn_down_exps.weight blk.$l.ffn_gate_exps.weight blk.$l.ffn_up_exps.weight"; done
  if [ -f $D/$S-00001-of-00003.gguf ] && [ -f $D/$S-00002-of-00003.gguf ] && [ -z "${REINJECT:-}" ]; then
    step "$NAME: model already built, skipping injection (REINJECT=1 to redo)"
  else
  step "$NAME: inject shard 2 (layers $FIRST-$LAST)"
  $PY -u $HERE/../gguf_inject_ternary.py --src $SRC/$S-00002-of-00003.gguf --dst $D/$S-00002-of-00003.gguf \
      --layers $LAYERS --values-dir $VAL > $EV/inject_$NAME.log 2>&1 || { step "inject FAILED"; tail -5 $EV/inject_$NAME.log; exit 1; }
  $PY -u $HERE/../gguf_inject_ternary.py --src $SRC/$S-00001-of-00003.gguf --dst $D/$S-00001-of-00003.gguf \
      --kv-only --names $N --widths 640 2560 >> $EV/inject_$NAME.log 2>&1 || { step "kv FAILED"; exit 1; }
  ln -sf $SRC/$S-00003-of-00003.gguf $D/$S-00003-of-00003.gguf
  fi
  M=$D/$S-00001-of-00003.gguf
  step "$NAME: PPL + KL"
  $B/llama-perplexity -m $M -f $TXT -c 512 -b 512 --chunks 32 -ngl 12 --kl-divergence-base $KLB --kl-divergence > $EV/ppl_$NAME.log 2>&1
  grep -E "Final estimate|^Mean PPL|Mean *KLD|Same top" $EV/ppl_$NAME.log
fi
step "$NAME: routing dump"
rm -rf $EV/route_$NAME
$T -m $M -f $TXT -c 512 -b 512 -ub 512 -ngl 12 --no-logits --n-chunks 8 --chunk-len 512 \
   --dump-filter '^ffn_moe_topk-[0-9]+$' --dump-dir $EV/route_$NAME > $EV/route_$NAME.log 2>&1
step "rc=$? $(ls $EV/route_$NAME | wc -l) files"
step "EVAL $NAME END"
