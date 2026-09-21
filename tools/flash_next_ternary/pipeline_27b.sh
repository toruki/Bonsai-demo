#!/usr/bin/env bash
# End-to-end: base Qwen3.8-27B BF16  ->  {BF16 ref, PTQ-absmean, PTQ-mseopt} GGUFs
# -> wikitext PPL / KL / per-layer hidden comparison against BF16, plus shipped Bonsai 2.
# Everything lands under /data; the repo's models/ and bin/ are only read.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
SP=${SP:-/tmp/claude-1000/-home-sohey-AI-LLM-Bonsai-demo/7b0cca5b-61b5-4cbf-a4e1-60b4d4b8bfaa/scratchpad}
PY=$SP/tq/bin/python
BIN=$ROOT/bin/cuda-e8
export LD_LIBRARY_PATH=$BIN
BASE=/data/models/qwen3.8-27b-bf16
GG=/data/models/gguf; mkdir -p $GG /data/eval
EVAL_DIR=/data/eval; export EVAL_DIR
LOG=$EVAL_DIR/pipeline.log
step() { echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
QUANT_ARGS=(--pure --output-tensor-type ptq1_0 --token-embedding-type ptq1_0
            --tensor-type ssm_alpha=bf16 --tensor-type ssm_beta=bf16 --tensor-type ssm_conv1d=f32)

# 0. wait for the base download
step "waiting for base download"
until grep -q "^DONE" /data/models/qwen3.8-27b-bf16.download3.log 2>/dev/null; do sleep 60; done
step "base download complete: $(du -sh $BASE | cut -f1)"

# 0b. shipped Bonsai 2 in PTQ1_0 packing (content-identical to the local PQ2_0; same kernel family as B)
if [[ ! -f $GG/shipped-bonsai2-27b-PTQ1_0.gguf ]]; then
  step "downloading shipped PTQ1_0"
  $PY -c "from huggingface_hub import hf_hub_download as h; import shutil; p=h('prism-ml/Ternary-Bonsai-2-27B-gguf','Ternary-Bonsai-2-27B-PTQ1_0.gguf'); shutil.copy(p,'$GG/shipped-bonsai2-27b-PTQ1_0.gguf')" >> $LOG 2>&1 || step "shipped PTQ1_0 download FAILED (will use local PQ2_0)"
fi

# 1. BF16 reference GGUF
if [[ ! -f $GG/qwen3.8-27b-bf16.gguf ]]; then
  step "converting base -> BF16 GGUF"
  $PY $ROOT/llama.cpp/convert_hf_to_gguf.py $BASE --outtype bf16 --outfile $GG/qwen3.8-27b-bf16.gguf >> $EVAL_DIR/convert_bf16.log 2>&1 || { step "BF16 convert FAILED"; exit 1; }
fi

# 2. PTQ variants
for RULE in absmean mseopt; do
  OUT=$GG/ptq-bonsai-$RULE-PTQ1_0.gguf
  [[ -f $OUT ]] && continue
  HF=/data/models/ptq-bonsai/$RULE
  step "building PTQ-$RULE checkpoint"
  $PY $ROOT/tools/flash_next_ternary/make_ptq_bonsai.py --src $BASE --dst $HF --rule $RULE --threads 24 >> $EVAL_DIR/make_$RULE.log 2>&1 || { step "make_ptq $RULE FAILED"; exit 1; }
  step "converting PTQ-$RULE -> F16 GGUF"
  $PY $ROOT/llama.cpp/convert_hf_to_gguf.py $HF --outtype f16 --outfile $GG/ptq-bonsai-$RULE-f16.gguf >> $EVAL_DIR/convert_$RULE.log 2>&1 || { step "convert $RULE FAILED"; exit 1; }
  step "quantizing PTQ-$RULE -> PTQ1_0"
  $BIN/llama-quantize "${QUANT_ARGS[@]}" $GG/ptq-bonsai-$RULE-f16.gguf $OUT PTQ1_0 24 >> $EVAL_DIR/quantize_$RULE.log 2>&1 || { step "quantize $RULE FAILED"; exit 1; }
  rm -rf $HF $GG/ptq-bonsai-$RULE-f16.gguf
  step "PTQ-$RULE done: $(du -sh $OUT | cut -f1)"
done

# 3. evaluations (BF16 first: it writes the KL base)
export KL_BASE=$EVAL_DIR/kl_base_bf16.bin
step "eval BF16 reference (partial offload)"
NGL_BF16=${NGL_BF16:-30}
$ROOT/tools/flash_next_ternary/run_eval.sh bf16 $GG/qwen3.8-27b-bf16.gguf $NGL_BF16 >> $LOG 2>&1 || step "bf16 eval FAILED"
for M in "ptq_absmean:$GG/ptq-bonsai-absmean-PTQ1_0.gguf" "ptq_mseopt:$GG/ptq-bonsai-mseopt-PTQ1_0.gguf" "shipped:$GG/shipped-bonsai2-27b-PTQ1_0.gguf" "shipped_pq2:$ROOT/models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf"; do
  N=${M%%:*}; F=${M#*:}; [[ -f $F ]] || continue
  step "eval $N"
  $ROOT/tools/flash_next_ternary/run_eval.sh $N $F 99 >> $LOG 2>&1 || step "$N eval FAILED"
done

# 4. per-layer comparison against BF16
for N in ptq_absmean ptq_mseopt shipped shipped_pq2; do
  [[ -d $EVAL_DIR/hidden_$N ]] || continue
  step "compare hidden: bf16 vs $N"
  $PY $ROOT/tools/flash_next_ternary/compare_hidden.py $EVAL_DIR/hidden_bf16 $EVAL_DIR/hidden_$N --json $EVAL_DIR/compare_$N.json > $EVAL_DIR/compare_$N.txt 2>&1
done
step "PIPELINE DONE"
