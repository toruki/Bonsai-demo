# Qwen3.8-27B での 4 者比較: BF16 / PTQ-absmean / PTQ-mseopt / 出荷 Bonsai 2

目的: 「PTQ で 90 % まで一致する code の残り 10 % + 学習」がどれだけ品質を買っているかを、
**同じ arch(qwen35)・同じ PTQ1_0 packing・同じ Hadamard(出荷 sign vector)・同じ推論コード**で測る。

コード: `tools/flash_next_ternary/{make_ptq_bonsai.py, pipeline_27b.sh, run_eval.sh, dump_hidden/, compare_hidden.py}`
成果物: `/data/models/gguf/*.gguf`、ログと dump は `/data/eval/`。リポジトリ内の既存環境は無変更。

## 0. 経路の検証(fold-only 対照)

fold + manifest + converter + runtime の経路にバグがないことを、
**fold だけして量子化しない F16 モデル**(`--rule none`)で確認した:

| モデル | wikitext-2 PPL (c=512) |
|---|---:|
| BF16 base(8 chunk) | 6.62 相当(64 chunk では 6.513) |
| **fold-only F16(8 chunk)** | **6.618** |

→ 回転基底で走らせても BF16 と同じ。以降の差はすべて ternary 化に起因する。

さらに、我々の PTQ GGUF と出荷 GGUF の code を tensor 種別ごとに直接比較すると
全種で **code 一致 90–96 %、非ゼロ符号一致 100 %、dequant 値の相関 0.92–0.97**
(attn_qkv / attn_gate / ssm_out / ffn_* / attn_q,k,v,output / output / token_embd、layer 0・3・63)。
レイアウト・embed の inverse・ssm_out の grouped-V 処理はすべて出荷と一致している。

## 1. 最終結果(wikitext-2 test, c=512, 64 chunk = 32k token)

| | モデル | PPL | PPL / BF16 | Mean KLD† | Median KLD† | top-1 一致† |
|---|---|---:|---:|---:|---:|---:|
| A | BF16 base(参照) | **6.513 ± 0.124** | 1.00 | — | — | — |
| B1 | PTQ-absmean(threshold 0.5·mean\|W'\|、support-mean scale) | 15,169 ± 437 | 2329 | 8.26 | 8.36 | 3.2 % |
| B2 | PTQ-mseopt(group ごとの exact weight-MSE 最適) | 829.6 ± 23 | 127 | 5.10 | 4.74 | 10.9 % |
| C | 出荷 Bonsai 2(PTQ1_0) | **8.615 ± 0.175** | **1.32** | **0.367** | **0.195** | **75.2 %** |
| C' | 出荷 Bonsai 2(PQ2_0、手元ファイル) | 8.616 ± 0.176 | 1.32 | — | — | — |

† `llama-perplexity --kl-divergence`、BF16 の logits を基準、16 chunk(8k token)。
その 16 chunk 上の PPL は BF16 7.38 / B1 26,344 / B2 1,108 / C 10.27。

固定 prompt(1082 token)の logits 比較(`compare_hidden.py`、BF16 基準):

| | mean KL | median KL | top-1 一致 | top-5 overlap | logit cosine |
|---|---:|---:|---:|---:|---:|
| B1 absmean | 7.10 | 7.17 | 8.0 % | 7.1 % | 0.650 |
| B2 mseopt | 4.54 | 4.13 | 15.7 % | 21.3 % | 0.803 |
| C 出荷 | **0.32** | **0.14** | **80.0 %** | **69.7 %** | **0.893** |

## 2. 層別の誤差(residual stream `l_out-N`、BF16 基準、1082 token)

| layer | 出荷 cos / relMSE | mseopt cos / relMSE | absmean cos / relMSE |
|---:|---|---|---|
| 0 | 0.9945 / 0.110 | 0.9835 / 0.072 | 0.9837 / 0.061 |
| 1 | 0.9925 / 0.088 | 0.9824 / 0.071 | 0.9810 / 0.078 |
| 3 | 0.9886 / 0.082 | 0.9586 / 0.109 | 0.9466 / 0.140 |
| 7 | 0.9769 / 0.096 | 0.9230 / 0.203 | 0.9160 / 0.211 |
| 15 | 0.9690 / 0.102 | 0.8974 / 0.295 | 0.8877 / 0.291 |
| 23 | 0.9285 / 0.162 | 0.7679 / 0.457 | 0.7495 / 0.474 |
| 31 | 0.8962 / 0.210 | 0.7024 / 0.526 | 0.6982 / 0.537 |
| 39 | 0.8543 / 0.272 | 0.6142 / 0.625 | 0.6243 / 0.615 |
| 47 | 0.8235 / 0.329 | 0.5875 / 0.670 | 0.5631 / 0.703 |
| 51 | 0.8222 / 0.337 | 0.3854 / 0.934 | 0.3009 / 1.042 |
| 55 | 0.8401 / 0.310 | 0.4041 / 0.907 | 0.2943 / 1.013 |
| 63 | 0.7220 / 0.668 | 0.4186 / 0.956 | 0.2270 / 1.343 |

relMSE 増分が大きい層(mseopt): 50 (+0.124), 34 (+0.081), 51 (+0.081), 2 (+0.078), 0 (+0.072), 19 (+0.067)。
出荷: 63 (+0.291), 0 (+0.110), 58 (+0.038), 62 (+0.032)。

全表は `/data/eval/compare_{shipped,ptq_mseopt,ptq_absmean}.txt`。

## 3. 読み取れること

### 3-1. PTQ は「大きく劣化」側。学習が品質のほぼすべて

- 最良の PTQ(mseopt)でも **PPL 830 / KL 5.1 / top-1 11 %**。出荷は **8.6 / 0.37 / 75 %**。
  ご提示の判定で言えば「BF16 6.0 / PTQ 9.5 / shipped 6.3」型よりさらに極端で、
  **PTQ 単独では使い物にならず、差分(≈ code の 5–10 % と scale・norm の再学習)がほぼ全て**。
- Bonsai 2 自身も wikitext PPL では BF16 比 **+32 %**(KL 0.37)。
  ベンチマーク 98.2 % 保持とは別の顔で、ternary の代償はゼロではない。

### 3-2. 崩れ方は「特定層の破綻」ではなく「全層での累積」

- PTQ は layer 0 で cos 0.984 から始まり、単調に落ちて layer 51 で 0.39。
  増分の大きい層(50, 34, 51, 2, 0, 19)も +0.07〜0.12 で、突出した 1 層は無い。
- → 「その層だけ BF16/FP8 にする」型の mixed precision では救えない。
  全層を少しずつ良くする必要がある(= 学習で全層を動かすのと同じ話)。
- 出荷モデルは layer 0 の relMSE が PTQ より *大きい*(0.110 vs 0.07)が cos は高い(0.9945)。
  方向は合っていてノルムが変わっている = 学習で residual のスケールが再調整された痕跡 **[推測]**。

### 3-3. weight-MSE 最適 ≫ absmean — 「Bonsai の動作点に合わせる」は PTQ では逆効果

| | 出荷 code との一致 | PPL | KL |
|---|---:|---:|---:|
| absmean(Bonsai の動作点、zero 31 %) | **89.9 %** | 15,169 | 8.26 |
| mseopt(zero 46 %) | 85.2 % | **830** | **5.10** |

出荷 code に近い方が **18 倍悪い**。Bonsai の zero率 33 % は「学習中の STE 量子化器の動作点」であって、
後処理の目的関数として良いわけではない。**weight-MSE は PTQ の目的関数として悪くない**
(activation-aware でさらに良くなる余地はあるが、それで 2 桁を埋めるのは考えにくい)。

### 3-4. Phase 3 の解釈の更新

Phase 3 で「単一 tensor で cosine 0.90 / rel-MSE 0.187」を "大きい" と評価したが、
実モデルでは **64 層分の累積で cos 0.4、PPL 2 桁増**になることが確認された。
単一 tensor 指標だけでは危険側に楽観になり得る。

## 4. 判断

ご提示の分岐では **「大きく劣化 → QAT / 蒸留方向へ」** に該当する。ただし重要な但し書き:

- 出荷 Bonsai 2 は我々の PTQ から **code の 5–10 % しか離れていない**(符号反転はほぼ無し)。
  つまり "良い ternary 解" は PTQ 初期値のすぐ近くに存在し、学習で到達している。
  ゼロから QAT する必要はなく、**PTQ-mseopt を初期値にした短時間の ternary-aware fine-tune**
  が現実的な路線。
- 27B は 1 枚の 5090 でフル QAT はできない(latent BF16 54 GB)。
  現実的なのは **層ごとの出力再構成(blockwise reconstruction)**: BF16 teacher の層入出力
  (今回の dump 経路で採取可能)を使い、1 層ずつ ternary code + scale + norm を STE で最適化する。
  1 層あたり ~0.4B params で 32 GB に収まる。
- その前段として **GPTQ 系の誤差補償(Hessian 付き ternary)** を試す価値はある。
  学習なしで PPL 830 → 数十まで下がれば、fine-tune の出発点として大きく有利。

Flash-Next(125B、512 expert)にこれを持ち込むなら、
「expert 1 個の出力再構成が 27B の 1 層と同じ手順で回るか」が次の検証単位になる。

## 5. 副産物

- `/data/models/gguf/qwen3.8-27b-bf16.gguf`(54.7 GB)、`foldonly-f16.gguf`(54.7 GB、対照)、
  `ptq-bonsai-{absmean,mseopt}-PTQ1_0.gguf`(各 6.0 GB)、`shipped-bonsai2-27b-PTQ1_0.gguf`(5.9 GB)
- `/data/eval/hidden_{bf16,shipped,ptq_mseopt,ptq_absmean}/` — 1082 token × 64 層 residual + logits
- `/data/eval/kl_base_bf16.bin` — BF16 logits(16 chunk)、以後の KL 測定に再利用可
