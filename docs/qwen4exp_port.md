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

## 4. ビルドと検証(移植版 = `build-q4x`, commit 09a2961cd)

CUDA ビルド成功(724 ターゲット、arch 120a、CUDA 13.3)。

### 4-1. upstream のテスト(移植した commit に含まれるもの)

| テスト | 結果 |
|---|---|
| `test-llama-archs -a qwen4exp`(CUDA) | **OK** (err 8.73e-08)、state save/load も OK |
| `test-llama-archs -a qwen4exp`(CPU) | **OK** (err 0.00e+00)、state save/load も OK |
| `test-backend-ops -o DSV4_HC_{PRE,POST,COMB}` | **3 種とも OK**(CUDA) |
| `test-quantize-fns`(fork の低 bit 型) | `pq2_0` / `ptq1_0` / `q4_0_e8` / `q2_e8` すべて OK — **移植で fork 側は壊れていない** |

補足: `test-llama-archs` を全 arch で回すと fork 固有の `dspark` で停止する
(`key not found: dspark.dspark.block_size`。KV 名が `%s.dspark.block_size` で
arch 名 `dspark` と二重になる fork 既存の問題で、本移植とは無関係)。

### 4-2. 実モデル(unsloth UD-IQ3_XXS、76 GiB)

- ロード・生成とも正常。`-ngl 12` で prompt 5.5 t/s / gen 7.2 t/s。
- wikitext-2 perplexity(8 chunk, c=512): **PPL 2.0245 ± 0.081**。
  27B BF16 の 6.51 よりかなり低いが、この arch は 51 B の n-gram PLE embedding を持ち
  Wikipedia 系テキストを強く記憶するため、低い値自体は不自然ではない **[要確認]**。

### 4-3. 参照 upstream ビルドとの greedy 比較(注意点あり)

ユーザーの `~/AI/LLM/Qwen3.8-Flash-Next/runtime/llama.cpp` は **cc231cb0d(2026-08-30)**。
これは qwen4exp 追加(08-27)の直後の版で、**私が移植した後続修正を含まない**:

| 修正 | 参照ビルドに含まれるか |
|---|---|
| `36b101543` seq_cp / block position keying / mtmd (09-01) | **含まれない** |
| `41abbfd59` rms_norm+mul fusion (09-14) | **含まれない** |
| `37b53fd45` hc ops (09-16) | **含まれない** |
| `3cf03257f` sparse FA (09-20) | 含まれない(移植でも保留) |

greedy(temp 0)32 token 比較の結果:

| prompt | 結果 |
|---|---|
| "1, 1, 2, 3, 5, 8, 13," | **完全一致** |
| "The capital of France is Paris..." | 20 token 目まで一致、以降 1 token の分岐 |
| "def fibonacci(n):" | 14 token 目で分岐 |

分岐はいずれも同じ位置パターン(閉じ引用符の直後で `".` と `"` が拮抗)で起きており、
**参照が古い実装であること**(特に position keying 修正の有無)と
CUDA 13.3 / 12.8 の数値差の両方が原因になり得る。
したがって「この参照との greedy 完全一致」は正当性の判定基準として不適切。
代わりに参照ツリーで `llama-perplexity` をビルドし、同一 8 chunk の PPL を比較中。

## 5. 次のステップ

1. ~~ビルド~~ 完了
2. **非 ternary の Flash-Next GGUF(unsloth UD-IQ3_XXS、76 GiB)で logits 正当性を確立**
   - 参照: ユーザーの upstream ビルド `~/AI/LLM/Qwen3.8-Flash-Next/runtime/llama.cpp`(cc231cb0d、qwen4exp あり)
   - 比較: 同一 prompt の greedy 生成一致、および `--kl-divergence-base` を上流で書き出して
     移植版で KL を計算(runtime 間 KL ≈ 0 なら一致)
   - recurrent state / MoE routing / PLE / QSA が経路として動いていることを合わせて確認
3. それが通ってから MoE expert tensor のみ PTQ1_0/PQ2_0 を許可
4. 最後に Hadamard(まず全て H128 で動作確認 → 後で gate/up を H512 に拡張。
   そのためには `prism.hadamard.block_size` を tensor 単位 / tensor class 単位へ拡張する)
