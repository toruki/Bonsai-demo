# QWEN4EXP (Qwen3.8-Flash-Next) runtime を PrismML fork へ移植

方針: 新規実装ではなく **upstream の実装を最小差分で cherry-pick**。ternary / Hadamard はまだ触らない。
ブランチ: `llama.cpp` の `qwen4exp-port`(`e8-kv` から分岐)。

## 1. 切り分け: 「導入前の commit」か「衝突で欠けている」か

**[確認] 前者(単に導入前)。** fork は upstream を 2026-08-25 の
`5ea87ddad`(webgpu: fix ARGSORT/TOP_K)で分岐しており、upstream が qwen4exp を追加したのは
その 2 日後の `6c84c7d5d`(2026-08-27, PR #27742)。`origin/prism` は qwen4exp 追加より前の
merge-base を持つだけで、Prism 側の変更と衝突して落とされたわけではない。

```
merge-base 5ea87ddad (2026-08-25)
  fork     origin/prism  : +110 commits(low-bit 実装が中心、ggml/src 84 ファイル)
  upstream master        : +502 commits(1151 ファイル / +123k 行)
  qwen4exp 追加          : 6c84c7d5d、merge-base の後・fork には無い
```

upstream 全体を merge すると 92 ファイルが両側で変更されており(`ggml-cuda/mmq*`,
`vecdotq.cuh`, `ggml-quants.c`, `conversion/*`, Metal kernel 群など Prism の低 bit 実装の中心)、
**full merge は避けるべき**。qwen4exp に必要な commit だけを取る方が差分が桁違いに小さい。

## 2. 取り込んだ commit(9 本)

| commit | 内容 | 衝突 |
|---|---|---|
| `fac889fb3` | `llama: model_loader: add TENSOR_READ_LAZY` (#27794) | `llama-model.cpp` の初期化子 1 箇所(fork の `dspark_head_source` / `load_mtp` と併存させる) |
| `925e11799` | `llama: add token ID tracking to KV cell` (#27762) | `#include` 1 行 |
| **`6c84c7d5d`** | **`model: add Qwen3.8-Flash-Next (qwen4exp)`** (#27742) | enum 追加 3 箇所(fork の `DSPARK` と併存)、`llama-kv-cache` の `cell_ext` 拡張 |
| `6fe749801` | graph split 削減 (#27880) | なし |
| `09412af38` | indexer heads を slice 単位で合算 (#28023) | なし |
| `0eadefebd` | recurrent state rollback (#28123) | なし |
| `36b101543` | seq_cp / block position keying / mtmd 入力 / cuda abort 修正 + tests (#27941) | `tests/test-save-load-state.cpp`(fork 未変更なので upstream 版を採用) |
| `41abbfd59` | rms_norm + mul fusion (#28896) | なし |
| `37b53fd45` | hc ops(`GGML_OP_DSV4_HC_{PRE,POST,COMB}`)(#28901) | Vulkan の `supports_op` 1 箇所 |

`3cf03257f`(CUDA sparse FA for qwen4)は FlashAttention 側の変更が大きく、fork の
`fattn-*` と衝突しやすいため **保留**(性能最適化であって正当性には不要)。

前提 commit が 2 本だけで済んだのは、`qwen4exp.cpp` が必要とする補助関数
(`build_attn_qsa` / `build_qsa_top_k` / `build_hc_mix` / `build_hc_combine` /
`build_ple` / `build_conv_state_at`)がすべて `qwen4exp.cpp` + `models.h` の中で
自己完結しているため。ggml 側で足りなかったのは hc op 群だけで、それは `37b53fd45` が入れている。

## 3. 衝突の性質

Prism 側の変更と本質的に競合する箇所は **無かった**。衝突はすべて
「同じ行に別の enum / include / 初期化子を追加した」類で、両方を残せば解決する。
`llama-arch.cpp` の tensor-info テーブルだけは upstream がフォーマットを全面変更していたため
upstream 版を採用し、fork 固有のエントリが無いことを確認した(差分ゼロ)。

## 4. 次のステップ

1. ビルド(CUDA、`build-q4x`)
2. **非 ternary の Flash-Next GGUF(unsloth UD-IQ3_XXS、76 GiB)で logits 正当性を確立**
   - 参照: ユーザーの upstream ビルド `~/AI/LLM/Qwen3.8-Flash-Next/runtime/llama.cpp`(cc231cb0d、qwen4exp あり)
   - 比較: 同一 prompt の greedy 生成一致、および `--kl-divergence-base` を上流で書き出して
     移植版で KL を計算(runtime 間 KL ≈ 0 なら一致)
   - recurrent state / MoE routing / PLE / QSA が経路として動いていることを合わせて確認
3. それが通ってから MoE expert tensor のみ PTQ1_0/PQ2_0 を許可
4. 最後に Hadamard(まず全て H128 で動作確認 → 後で gate/up を H512 に拡張。
   そのためには `prism.hadamard.block_size` を tensor 単位 / tensor class 単位へ拡張する)
