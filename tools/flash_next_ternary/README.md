# flash_next_ternary

Qwen3.8-Flash-Next へ Bonsai 2 風の ternary 量子化を適用できるかを調べるための
実験コード。**既存の Bonsai 2 / Qwen3.8-27B 推論環境には一切触れません。**
`llama.cpp/` も `models/` も読むだけです。

| ファイル | 役割 |
|---|---|
| `bonsai_format.py` | PQ2_0 / PTQ1_0 codec と Hadamard 回転 / fold の numpy 転記。出荷 GGUF に対する byte 一致 selftest 付き |
| `fetch_slice.py` | HF safetensors から必要な tensor(行スライス)だけ HTTP Range 取得。キャッシュは `/data/models/flash-next-bf16-slices` |
| `ternary_quant.py` | group-wise ternary 量子化 4 方式(A: absmean / B: MSE scale / C: threshold grid / D: 交互反復) |
| `run_phase3.py` | 単一 tensor の測定ハーネス |

## 使い方

```bash
# 前提: numpy / requests / huggingface-hub / pyyaml が入った venv
python tools/flash_next_ternary/bonsai_format.py \
    models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf      # selftest

python tools/flash_next_ternary/run_phase3.py --part gate --expert 0
python tools/flash_next_ternary/run_phase3.py --part down --expert 0 --blocks 0 128
```

結果と考察は `docs/phase3_ternary_experiment.md`。
