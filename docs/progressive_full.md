# 全 64 層 progressive ternary(Qwen3.8-27B、dual teacher)

pilot(`docs/progressive_pilot.md`)の判定を通過したため、layer 16 から再開して 64 層すべてを
block ごとに学習した。設定は pilot の dual と同じ(anchor 0.2、aux 0.1、1500 step/block、640 seq)。
block ごとに新プロセス(1 プロセスで回し続けると allocator 断片化で 8 倍遅くなった)。
所要: 16 block × ~20 分 ≈ 5.5 時間(RTX 5090 1 枚)。embed / lm_head / final norm は **BF16 のまま**。

driver: `/data/eval/prog_full_dual2.sh`、成果物: `/data/models/gguf/progressive-dual-27b-PTQ1_0.gguf`

## 1. 学生 stream の累積誤差(valid 32 seq、canonical BF16 基準、各 block 出口)

| 出口 | cos | relMSE | 増分 |
|---:|---:|---:|---:|
| 3 | 0.995 | 0.015 | |
| 7 | 0.988 | 0.038 | +0.024 |
| 11 | 0.987 | 0.040 | +0.002 |
| 15 | 0.985 | 0.042 | +0.002 |
| 19 | 0.969 | 0.068 | +0.026 |
| 23 | 0.946 | 0.109 | +0.041 |
| 27 | 0.945 | 0.114 | +0.005 |
| 31 | 0.938 | 0.126 | +0.012 |
| 35 | 0.920 | 0.152 | +0.026 |
| 39 | 0.910 | 0.166 | +0.014 |
| 43 | 0.921 | 0.152 | −0.014 |
| 47 | 0.923 | 0.148 | −0.004 |
| 51 | 0.905 | 0.177 | +0.029 |
| 55 | 0.889 | 0.230 | +0.054 |
| 59 | 0.856 | 0.259 | +0.029 |
| 63 | 0.881 | 0.213 | −0.046 |

増分はおおむね一定〜減少(43–47、63 では anchor が引き戻している)で、超線形発散は最後まで起きなかった。
難しい block は 20–23、32–35、52–55(PTQ の層別解析で増分が大きかった層と一致)。

## 2. 完全モデルの評価(wikitext-2、c=512、64 chunk / KL は 16 chunk)

| モデル | ternary 対象 | PPL | PPL/BF16 | Mean KLD | top-1 一致 |
|---|---|---:|---:|---:|---:|
| BF16 | — | 6.513 | 1.00 | — | — |
| PTQ-mseopt | 64 層 + embed + head | 829.6 | 127 | 5.10 | 10.9 % |
| **progressive dual** | **64 層**(embed / head BF16) | **11.07** | **1.70** | **0.736** | **65.9 %** |
| 出荷 Bonsai 2 | 64 層 + embed + head | 8.62 | 1.32 | 0.367 | 75.2 % |

PTQ1_0 に pack した GGUF(6.6 GB、embed / head は BF16)は fork の runtime でそのまま動き、
F16 版と同じ PPL(§4)。

## 3. 層別 hidden 比較(固定 prompt 1082 token、BF16 基準、cos / relMSE)

| layer | 出荷 Bonsai 2 | PTQ | progressive |
|---:|---|---|---|
| 0 | 0.9945 / 0.110 | 0.9835 / 0.072 | **0.9975 / 0.006** |
| 15 | 0.9690 / 0.102 | 0.8974 / 0.295 | **0.9835 / 0.043** |
| 31 | 0.8962 / 0.210 | 0.7024 / 0.526 | **0.9417 / 0.119** |
| 47 | 0.8235 / 0.329 | 0.5875 / 0.670 | **0.9288 / 0.140** |
| 55 | 0.8401 / 0.310 | 0.4041 / 0.907 | **0.8978 / 0.202** |
| 63 | 0.7220 / 0.668 | 0.4186 / 0.956 | **0.8842 / 0.193** |
| logits | KL 0.319 / top-1 80.0 % / cos 0.893 | KL 4.54 / 15.7 % / 0.803 | KL 0.660 / 71.1 % / **0.957** |

## 4. 読み取れること

1. **PTQ → progressive で PPL 830 → 11.1(75 分の 1)、KL 5.1 → 0.74(7 分の 1)。**
   33 万 token、5.5 時間、end-to-end 学習なしでここまで戻る。
2. **residual stream は全層で出荷 Bonsai 2 より BF16 に近い**(layer 63 で cos 0.884 vs 0.722)。
   logit cosine も高い(0.957 vs 0.893)。**それでも KL は出荷の 2 倍**(0.74 vs 0.37)。
   → 出荷モデルの優位は「最終 residual → logits」の部分(final norm / lm_head、出荷は ternary かつ学習済み)と、
   stream の *方向* より *どの成分* が合っているかにある。我々は final norm / lm_head を一切触っていない。
3. 累積は線形以下に収まったが、後段(52–59)で増分が大きい。この領域に学習を追加投入する余地がある。
4. **最も安い次の一手は「頭の較正」**: 学生の最終 stream を入力に、final norm(+ lm_head)を
   BF16 logits との KL で学習する。パラメータは norm 5k(+ head 1.27B、BF16 のまま or ternary)。
   1 pass で済み、KL の残り半分のどれだけがここにあるかが分かる。
5. その次は、後段 block の再学習(step / データ増、anchor 増)と、2 回目の progressive pass。

## 5. 作った GGUF の中身

`progressive-dual-27b-PTQ1_0.gguf`: 400 tensor が PTQ1_0(fold 済み、出荷と同じ `prism.hadamard.*` contract、
同じ sign vector)、`token_embd` / `output` / `ssm_alpha` / `ssm_beta` が BF16、norm / conv1d が F32。
出荷 Bonsai 2 との差は embed / head の型(BF16 2.5 GB vs ternary 0.55 GB)のみ。
