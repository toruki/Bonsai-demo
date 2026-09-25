#!/usr/bin/env bash
# Evaluate a Flash-Next variant whose routed experts in some layers were replaced by
# trained/PTQ ternary values (experts_L{N}.npz from q4x_progressive.py).
#   eval_q4x_variant.sh <name> <values-dir> <first> <last>      (name=base: IQ3_XXS reference only)
# Env: LAYER_LIST="4 5 ... 47"  replaces seq <first> <last> (e.g. to leave some layers at IQ3_XXS);
#      MODEL_ROOT=DIR            where model_<name>/ is written (default: EVAL_DIR);
#      a shard file (or symlink to an identical one) already present in model_<name>/ is kept
#      unless REINJECT=1, so shards shared with another variant can be linked in beforehand.
# Writes PPL + KL (32 x 512 wikitext-2 test, vs the IQ3_XXS logits) and routing / MoE dumps
# (8 x 512 tokens, into $RMASS_DIR/rmass_<name>, compared with rmass_base by routing_mass.py).
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
  D=${MODEL_ROOT:-$EV}/model_$NAME; mkdir -p $D
  LAYERS=${LAYER_LIST:-$(seq $FIRST $LAST)}
  N=""; for l in $LAYERS; do N="$N blk.$l.ffn_down_exps.weight blk.$l.ffn_gate_exps.weight blk.$l.ffn_up_exps.weight"; done
  # routed experts of blk 0-17 live in shard 2, blk 18-47 in shard 3 (UD-IQ3_XXS split)
  L2=""; L3=""; for l in $LAYERS; do if [ $l -le 17 ]; then L2="$L2 $l"; else L3="$L3 $l"; fi; done
  : > $EV/inject_$NAME.log
  for sh in 2 3; do
    SL=$([ $sh = 2 ] && echo "$L2" || echo "$L3")
    if [ -e $D/$S-0000$sh-of-00003.gguf ] && [ -z "${REINJECT:-}" ]; then
      step "$NAME: shard $sh present, kept ($(readlink -f $D/$S-0000$sh-of-00003.gguf))"
    elif [ -n "$SL" ]; then
      step "$NAME: inject shard $sh (layers$SL)"
      $PY -u $HERE/../gguf_inject_ternary.py --src $SRC/$S-0000$sh-of-00003.gguf --dst $D/$S-0000$sh-of-00003.gguf \
          --layers $SL --values-dir $VAL >> $EV/inject_$NAME.log 2>&1 || { step "inject FAILED"; tail -5 $EV/inject_$NAME.log; exit 1; }
    else
      ln -sf $SRC/$S-0000$sh-of-00003.gguf $D/$S-0000$sh-of-00003.gguf
    fi
  done
  if [ -e $D/$S-00001-of-00003.gguf ] && [ -z "${REINJECT:-}" ]; then
    step "$NAME: shard 1 present, kept"
  else
    $PY -u $HERE/../gguf_inject_ternary.py --src $SRC/$S-00001-of-00003.gguf --dst $D/$S-00001-of-00003.gguf \
        --kv-only --names $N --widths 640 2560 >> $EV/inject_$NAME.log 2>&1 || { step "kv FAILED"; exit 1; }
  fi
  M=$D/$S-00001-of-00003.gguf
  step "$NAME: PPL + KL"
  $B/llama-perplexity -m $M -f $TXT -c 512 -b 512 --chunks 32 -ngl 12 --kl-divergence-base $KLB --kl-divergence > $EV/ppl_$NAME.log 2>&1
  grep -E "Final estimate|^Mean PPL|Mean *KLD|Same top" $EV/ppl_$NAME.log
fi
step "$NAME: routing / MoE dump"
# top-k, gate weights, full router softmax, routed-expert output and streams (routing_mass.py / moe_decompose.py)
RM=${RMASS_DIR:-/data/eval}/rmass_$NAME; rm -rf $RM
$T -m $M -f $TXT -c 512 -b 512 -ub 512 -ngl 12 --no-logits --n-chunks 8 --chunk-len 512 \
   --dump-filter '^(ffn_moe_probs|ffn_moe_topk|ffn_moe_weights_norm|ffn_moe_out|l_last)-[0-9]+$' --dump-dir $RM > $RM.log 2>&1
step "rc=$? $(ls $RM | wc -l) files"
if [ "$NAME" != base ]; then
  $PY $HERE/routing_mass.py ${RMASS_DIR:-/data/eval}/rmass_base $RM --first $FIRST --last $LAST --json $EV/rmass_$NAME.json | grep -E "^layer|^mean"
fi
step "EVAL $NAME END"
