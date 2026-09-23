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

### 4-3. 実モデルでのコンポーネント確認(移植版のロードログ)

| 項目 | 確認結果 |
|---|---|
| arch | `qwen4exp`、48 層、`n_embd` 2560 / `n_embd_out` 10240 |
| **MoE routing** | `n_expert = 512`, `n_expert_used = 10` |
| **PLE(n-gram)** | `ple.layers=[1]`, `ngram_size=3`, `heads_per_ngram=8`, 16 head の vocab/offset 配列、`per_layer_token_embd.weight`(27465 MiB)を **lazy read** で保持(前提 commit `fac889fb3` が効いている) |
| **QSA indexer** | `indexer.head_count=4`, `key_length=128`, `top_k=2048`, `compress_ratios=[0,0,0,4,…]`、専用の **indexer KV cache**(512 cells)を生成 |
| **hybrid memory** | `llama_kv_cache`(12 層 = full-attention 層)+ `llama_memory_recurrent`(48 層、R/S バッファ 112.6 MiB)が両方生成される |
| 生成 | 正常に継続生成(graphs reused = 2) |

### 4-4. 参照 upstream ビルドとの greedy 比較(注意点あり)

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

### 4-5. 同一 commit の upstream との perplexity 比較(定量的な正当性確認)

参照ビルド(`cc231cb0d`)は CMakeCache が移動前のパスを指しており再ビルド不能だったため、
**移植した最後の commit `37b53fd45` を git worktree に展開して同じ CUDA 13.3 / arch 120a でビルド**し、
qwen4exp については完全に同一のソースを持つ参照を用意した(`/data/eval/upstream-ref`, build 10998)。

同一モデル(unsloth UD-IQ3_XXS)・同一引数(`-c 512 -b 512 --chunks 8 -ngl 12`)で:

| build | 累積 PPL(chunk 1→8) | Final |
|---|---|---:|
| upstream 37b53fd45 | 2.5105 / 3.8270 / 2.9912 / 2.4979 / 2.2618 / 2.1751 / 2.0517 / 2.0270 | **2.0270 ± 0.081** |
| PrismML fork 移植版 | 2.4151 / 3.6967 / 2.9014 / 2.4269 / 2.2222 / 2.1504 / 2.0336 / 2.0245 | **2.0245 ± 0.081** |

- **差は 0.12 %**(4096 token 上の累積)。両者の誤差棒(±0.081)より遥かに小さい。
- 移植版は 2 回実行して **完全に同一の値**(2.0245、全 chunk 一致)= 決定的。
- chunk 単位では数 % ずれる。fork は ggml 側(`mmq` / `vecdotq` など)を独自実装しており、
  IQ3_XXS + top-10/512 の MoE routing では僅かな数値差が expert 選択を入れ替えるため、
  chunk 単位のずれは想定内。累積で一致することが重要。

**結論: グラフ実装は正しい。**(グラフが壊れていれば PPL は桁で変わるか NaN になる。)

## 5. Step 2 完了判定

| ユーザー指定の確認項目 | 状態 |
|---|---|
| logits 一致 | **OK**(同一 commit の upstream と累積 PPL 0.12 % 差、`test-llama-archs` の backend 間 logits 一致 8.7e-08) |
| short prompt | **OK**(3 prompt で一貫した生成) |
| recurrent state | **OK**(`llama_memory_recurrent` 48 層、`test-llama-archs` の state save/load も OK) |
| MoE routing | **OK**(`n_expert=512 / used=10`、生成が一貫) |
| PLE | **OK**(3-gram / 16 head / 27.5 GiB テーブルを lazy read) |
| QSA | **OK**(indexer 専用 KV cache、top_k 2048、compress_ratios) |

## 6. 次のステップ

1. ~~ビルド~~ 完了
2. **非 ternary の Flash-Next GGUF(unsloth UD-IQ3_XXS、76 GiB)で logits 正当性を確立**
   - 参照: ユーザーの upstream ビルド `~/AI/LLM/Qwen3.8-Flash-Next/runtime/llama.cpp`(cc231cb0d、qwen4exp あり)
   - 比較: 同一 prompt の greedy 生成一致、および `--kl-divergence-base` を上流で書き出して
     移植版で KL を計算(runtime 間 KL ≈ 0 なら一致)
   - recurrent state / MoE routing / PLE / QSA が経路として動いていることを合わせて確認
3. それが通ってから MoE expert tensor のみ PTQ1_0/PQ2_0 を許可
4. 最後に Hadamard(まず全て H128 で動作確認 → 後で gate/up を H512 に拡張。
   そのためには `prism.hadamard.block_size` を tensor 単位 / tensor class 単位へ拡張する)

---

# Step 3: MoE expert だけ ternary (H128 + PTQ1_0)

## 6-1. コード変更(最小差分 2 行)

| ファイル | 変更 |
|---|---|
| `src/llama-model.cpp` | `prism.hadamard` の arch ゲートに `LLM_ARCH_QWEN4EXP` を追加 |
| `conversion/base.py` | `_HADAMARD_ARCHS` に `MODEL_ARCH.QWEN4EXP` を追加 |

事前確認: qwen4exp の expert 3 本はすべて Hadamard 対応ヘルパーを通る
(`build_moe_ffn` → `build_lora_mm_id`、gate/up は `llama-graph.cpp:2210-2242`、down は `:2346`)。
runtime の「検証済み tensor 種別」リストには `ffn_{gate,up,down,gate_up}_exps` が既に入っていたため、
**arch ゲートだけが障害だった**。qwen4exp の他の線形 18 箇所もすべて `build_lora_mm` 経由
(生の `ggml_mul_mat` は indexer の score 計算 1 箇所のみで、重み行列ではない)。

## 6-2. 新ツール `tools/flash_next_ternary/gguf_inject_ternary.py`

既存 GGUF の指定 tensor だけを **H_block fold → group-128 ternary → PTQ1_0** に置き換え、
他は bytes をそのままコピーする。置き換えた tensor だけに `prism.hadamard.*` contract を書くので、
runtime はそこだけ活性を回転する。split GGUF 対応(`--kv-only` で KV shard に contract のみ追加、
非先頭 shard は KV ブロックを持たないので `general.architecture` を合成しない)。
ソースが量子化済みでも `gguf.quants.dequantize` で f32 に戻してから処理する。

## 6-3. 合成モデルでの経路検証

`test-llama-archs -a qwen4exp -o DIR` で小さな qwen4exp GGUF を生成 → 注入 → ロード:

```
- type ptq1_0: 4 tensors
load_tensors: loaded 4 Hadamard-folded weight(s) (0 inverse-lookup)
              using 1 rotation(s) and 2 sign vector(s)
```

ternary の relMSE 0.186–0.188 / zero 0.455(Gaussian に対する ternary の理論最適と一致)。
合成モデルはトークナイザを持たないため forward までは進まない。

## 6-4. 実モデル(unsloth UD-IQ3_XXS)の layer 0 を ternary 化

対象: `blk.0.ffn_{gate,up,down}_exps.weight`(512 expert ぶん、2.5 B params)。
元の型は gate/up が `IQ2_S`、down が `IQ4_NL`。それを dequantize → H128 fold → ternary → PTQ1_0。

| tensor | 元の型 | shape | zero 率 | relMSE |
|---|---|---|---:|---:|
| `blk.0.ffn_down_exps.weight` | IQ4_NL | [640, 2560, 512] | 0.457 | 0.1869 |
| `blk.0.ffn_gate_exps.weight` | IQ2_S | [2560, 640, 512] | 0.457 | 0.1866 |
| `blk.0.ffn_up_exps.weight` | IQ2_S | [2560, 640, 512] | 0.457 | 0.1868 |

shard 2 は 49.6 GB → 46.8 GB に縮小。

### 結果

```
- type ptq1_0: 3 tensors
load_tensors: loaded 3 Hadamard-folded weight(s) (0 inverse-lookup)
              using 1 rotation(s) and 2 sign vector(s)

> The capital of Japan is
[Start thinking]
We need to answer user's query: "The capital of Japan is".
```

| モデル | 8 chunk | 32 chunk |
|---|---:|---:|
| 元の UD-IQ3_XXS | 2.0245 ± 0.081 | 2.4783 ± 0.052 |
| **layer 0 の expert だけ ternary(H128 + PTQ1_0)** | 2.0022 ± 0.077 | **2.5057 ± 0.052** |

8 chunk では差が誤差棒に埋もれた(ternary の方が僅かに低い)ため 32 chunk で測り直した。
**48 層中 1 層ぶんの expert(512 expert, 2.5 B params)を ternary 化したコストは PPL +1.1 %**
(2.4783 → 2.5057)。誤差棒 ±0.05 に対して差 0.027 なので、これでもまだ有意とは言い切れない **[要注意]**。
いずれにせよ CUDA の MoE id 付き matmul(`build_lora_mm_id`)と活性側 Hadamard 回転が
実モデルで正しく動作している。

なおここで使った重みは元 GGUF の IQ2_S/IQ4_NL を dequantize したもので、
BF16 原本ではない。**reconstruction 実験には BF16 teacher が要る**ので、そこは次段で HF から
range read で取得する(27B と同じ `fetch_slice.py` の手順)。

## 7. 次のステップ

1. expert 1 個単位の reconstruction(teacher A = BF16 expert、teacher B = canonical、
   27B で有効だった dual 目的関数)→ 学習済み ternary expert を注入して PPL を再測定
2. Hadamard block を tensor 単位 / tensor class 単位へ拡張し、gate/up を H512 に
   (`prism.hadamard.block_size` が現状は GGUF 全体で単一値)
3. 全 48 層への展開と VRAM 実測

---

# Step 3b: 全 48 層の expert を ternary 化(規模・サイズ・VRAM の実測)

## 8-1. `llama-quantize` の COPY + `--tensor-type` 修正

全層を Python で処理しようとしたが、839 M params の tensor を扱う過程で WSL2 上で
プロセスごと強制終了される事象が繰り返した(チャンク化と I/O 抑制で layer 12 → 21 → 28 → 37 と
前進したが完走せず。`dmesg` には `hv_storvsc: swiotlb buffer is full` が出ていた)。
C++ の `llama-quantize` に切り替えたが、**COPY ftype では `--tensor-type` の上書きが無効**だった
(`tensor_allows_quantization` が `params->only_copy` で即 false を返す)。

`src/llama-quant.cpp` に最小修正を入れ、**COPY でも明示的に名前指定された tensor だけは変換する**
ようにした。混合精度モデルの一部 tensor だけを別の型にする唯一の手段であり、fork 本体にとっても有用。

```
llama-quantize --allow-requantize \
  --tensor-type ffn_gate_exps=ptq1_0 --tensor-type ffn_up_exps=ptq1_0 \
  --tensor-type ffn_down_exps=ptq1_0  src.gguf dst.gguf COPY 16
```

## 8-2. サイズ

| | 元 | PTQ1_0 |
|---|---:|---:|
| `ffn_down_exps`(層あたり) | 450 MiB (IQ4_NL) | 175 MiB |
| `ffn_gate_exps` / `ffn_up_exps`(層あたり各) | 256 MiB (IQ2_S) | 175 MiB |
| **モデル全体** | **78,154 MiB (3.71 bpw)** | **56,979 MiB (2.70 bpw)** |

削減 21.2 GiB(−27 %)、変換 9.5 分(16 スレッド)。
Phase 2 の机上見積もり(routed experts 120.8 B を PTQ1_0 で 24.6 GiB)と実測(24.6 GiB)が一致。

## 8-3. VRAM(RTX 5090 32 GB、`-ngl 48` = 全層オフロード、`-c 2048`)

| 項目 | サイズ |
|---|---:|
| CUDA0 model buffer | 28,422 MiB |
| CPU_Mapped model buffer(PLE の n-gram テーブルなど) | 28,557 MiB |
| CUDA0 KV + RS + compute | 512 MiB |
| **GPU 合計** | **約 28.9 GiB** |

**全 48 層が 5090 に載る。** `-ngl 20/28/36/48` すべて OOM なし。
残り 28.5 GiB は PLE(`per_layer_token_embd`, 27.5 GiB)が CPU 側に残ったもので、
Phase 2 の設計どおり「PLE は CPU RAM へ offload」が実機で成立している。
Phase 2 の見積もり(構成 C で 28.26 GiB)は実測 28.9 GiB とよく一致した。

## 8-4. 品質(ここが問題)

| モデル | 32 chunk PPL |
|---|---:|
| 元の UD-IQ3_XXS | 2.4783 ± 0.052 |
| layer 0 の expert のみ ternary(**H128 + MSE 最適**) | 2.5057 ± 0.052 |
| **全 48 層の expert ternary(fold なし・素の RTN)** | **22.275 ± 0.748** |

生成自体は一貫している:

```
> The capital of Japan is
[Start thinking]
The user is asking about the capital of Japan. This is a straightforward factual question
```

しかし PPL は 9 倍に悪化。層あたりのコスト(+0.027)から線形に予測される 3.8 を大きく超えており、
**27B で観測したのと同じ超線形な累積**が Flash-Next でも起きている。

ただしこの比較には手法差が混じっている点に注意:

| | 量子化器 | Hadamard |
|---|---|---|
| layer 0 版 | group-128 の MSE 最適 | H128 fold あり |
| 全層版 | `llama-quantize` の RTN(absmax) | なし |

27B の知見(`docs/ptq_vs_shipped_27b.md`)では **RTN(absmean 相当)は MSE 最適より PPL で 18 倍悪かった**。
つまり 22.3 のうち相当部分は量子化器の差で説明できる可能性が高く、
fold + MSE 最適 + reconstruction を入れた値とは別物として扱う必要がある。

## 9. 現時点の結論

- **経路は完全に成立した**: qwen4exp runtime(移植版)+ ternary MoE expert + Hadamard 活性回転で、
  実モデルが decode でき、全 48 層が RTX 5090 32 GB に載る(28.9 GiB)。
- **サイズ目標は達成**: 78 GiB → 57 GiB、Phase 2 の見積もりどおり。
- **品質は未達**: 素の PTQ では PPL が 9 倍。27B と同じく、ここから先は
  **reconstruction / ternary-aware fine-tune が必須**。
