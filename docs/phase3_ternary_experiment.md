# Phase 3: 単一 tensor の ternary 実験(Flash-Next routed expert)

前提: [`docs/ternary_analysis.md`](ternary_analysis.md) / [`docs/flash_next_quantization_plan.md`](flash_next_quantization_plan.md)

コード: `tools/flash_next_ternary/`(既存の推論環境には一切触れていません)

> **この実装は Bonsai 方式ではありません。** Phase 1 §15 のとおり PrismML の
> threshold / scale 規則は本リポジトリからも出荷 GGUF からも同定できません。
> 以下は **Bonsai-inspired ternary quantization** です。

---

## 1. 実験セットアップ

対象: `Qwen/Qwen3.8-Flash-Next`(BF16 原本)の
`model.language_model.layers.0.mlp.experts.{gate_up_proj, down_proj}` の expert 0〜3、
および `shared_expert.gate_proj.weight`。

131 shard(~350GB)はダウンロードせず、safetensors header を読んで
**必要な expert の行範囲だけ HTTP Range 取得**しています
(`tools/flash_next_ternary/fetch_slice.py`、キャッシュは `/data/models/flash-next-bf16-slices`、
expert 1 本あたり 6.5 MB)。

パイプライン:

```
BF16 W → sign flip (S) → blockwise Hadamard (H) → group-128 ternary → dequantize → PQ2_0/PTQ1_0 packing
```

- `H` は `llama-model.cpp` が合成するものと同一(normalized Sylvester-Walsh、対称直交)。
- fold は `W_folded = (W * s) @ H`、活性側は `R x = H (s ⊙ x)`。
  **各 block size で `W_folded @ (R x) == W @ x` を相対誤差 <2e-5 で検証してから**
  測定しています(`run_phase3.py` の assert)。
- group scale は **fp16 に丸めています**。PQ2_0/PTQ1_0 は fp16 scale しか格納できないため。

scale/threshold は指示どおり 4 方式を並列に実装(`ternary_quant.py`):

| | 方式 |
|---|---|
| A | `s = mean(|W|)`、threshold = `s/2`(BitNet b1.58 相当) |
| B | `s` を MSE 最小化(round-to-nearest 固定、`s = r·mean|W|` の grid) |
| C | threshold を grid search、support 上で `s = mean(|w_i|)`(TWN の閉形式) |
| D | C から開始して `s ← <w,t>/<t,t>` と再割り当てを交互反復 |

---

## 2. 結果: expert 0 / gate(shape [640, 2560])

statistics: `std=0.013546  absmax=0.17773  kurtosis=3.731`

```
 block method              rel_mse    cosine   zero%   max_err  out_rel[iid]  out_rel[outlier]
------------------------------------------------------------------------------------------------
     - A_absmean           0.28945  0.873711   32.22   0.16793       0.29058       0.26434
     - B_mse_scale         0.20793  0.889978   48.78   0.15930       0.21035       0.21288
     - C_threshold_grid    0.20791  0.889990   48.79   0.15928       0.21019       0.21277
     - D_alternating       0.20791  0.889990   48.79   0.15928       0.21019       0.21277

   128 A_absmean           0.26051  0.888225   30.84   0.06103       0.26038       0.28323
   128 B_mse_scale         0.18681  0.901770   45.63   0.05381       0.18750       0.21501
   128 C_threshold_grid    0.18679  0.901783   45.62   0.05382       0.18751       0.21486
   128 D_alternating       0.18679  0.901783   45.62   0.05382       0.18752       0.21498

   512 A_absmean           0.26087  0.888167   30.91   0.06549       0.26144       0.27763
   512 B_mse_scale         0.18662  0.901879   45.73   0.05907       0.18667       0.21516
   512 C_threshold_grid    0.18660  0.901889   45.73   0.05893       0.18665       0.21615
   512 D_alternating       0.18660  0.901890   45.73   0.05893       0.18665       0.21600

packing round-trip @ block 512: PQ2_0 exact=True  PTQ1_0 exact=True
storage: PQ2_0 2.125 bpw, PTQ1_0 1.750 bpw (BF16 reference 16.0)
```

`block = -` は回転なしの対照。`out_rel[*]` は合成 activation
(iid / channel outlier あり、いずれも RMSNorm 相当で正規化)経由の出力相対 MSE。

## 3. 結果: 他の tensor での再現性

block は 2560 幅なら 512、640 幅なら 128。方式 D。

| tensor | kurtosis | rel_MSE (D) | cosine (D) | zero% | rel_MSE (A) |
|---|---:|---:|---:|---:|---:|
| expert0 gate | 3.73 | 0.18660 | 0.901889 | 45.73 | 0.26087 |
| expert1 gate | 4.81 | 0.18655 | 0.901916 | 45.65 | 0.26046 |
| expert2 gate | 3.44 | 0.18721 | 0.901552 | 45.71 | 0.26109 |
| expert3 gate | 3.44 | 0.18694 | 0.901703 | 45.67 | 0.26071 |
| expert0 down | 3.73 | 0.18725 | 0.901525 | 45.70 | 0.26102 |
| expert1 down | 5.42 | 0.18671 | 0.901828 | 45.64 | 0.26049 |
| expert2 down | 3.49 | 0.18719 | 0.901560 | 45.80 | 0.26137 |
| expert3 down | 3.15 | 0.18700 | 0.901664 | 45.73 | 0.26105 |
| shared_expert gate | 118.19 | 0.18299 | 0.903883 | 45.44 | 0.25507 |

expert 間のばらつきは 0.4% 以内。**expert 個別の最適化は不要**です。

---

## 4. 読み取れたこと

### 4-1. この制約下の PTQ ternary は最適点に到達しており、同じ枠内では改善しません

N(0,1) に対する最適 ternary 量子化を解析的に解くと:

```
rel_mse = 0.19017   cosine = 0.8999   threshold = 0.6120 σ   scale = 1.2240 σ   zero率 = 45.95%
```

実測(C / D)は **rel_mse 0.1866–0.1872 / zero率 45.6–45.8%** で、
group-128 の適応 scale のぶんだけ解析値をわずかに下回っています。

つまり **B ≈ C ≈ D は同じ最適点に収束しており、同じ枠内で
「もっと良い threshold / scale optimizer」を探しても意味がありません。**
A(absmean)だけが 0.26 と劣りますが、これは MSE を最適化していないため当然です。

ただしこの「最適」は次の制約をすべて固定した上での話です:
固定 Hadamard(block 128/512)、group 128、group 内共通 scale、
{-1,0,+1}、**目的関数 = weight MSE**。
以下を許せば改善余地は残っています(未検証):
activation-aware な目的関数、GPTQ/Hessian 系の誤差補償、group 間・column 間の補正、
learned rotation / permutation、tensor ごとの Hadamard block、一部 channel の高 bit 化、
ternary 化後の短時間 fine-tune / QAT。特に **weight MSE 最小 ≠ 推論誤差最小** です。

routed expert の重みは kurtosis ≈ 3.5(= ほぼ Gaussian)で、
**Gaussian は回転不変なので Hadamard の効果も小さい**(0.208 → 0.187、約 10% 改善)。
回転の価値が出るのは重い裾を持つ tensor です:

| shared_expert gate_proj | kurtosis | rel_MSE | cosine | max_err |
|---|---:|---:|---:|---:|
| 回転なし | 118.19 | 0.24537 | 0.868685 | 0.27734 |
| H128 | 6.19 | **0.17854** | **0.906341** | 0.08213 |
| H512 | 5.28 | 0.18341 | 0.903652 | 0.07136 |

→ **Hadamard rotation は実装済み・効果確認済み**ですが、
routed expert に限れば効果は 10% 程度で、劇的ではありません。

### 4-2. Bonsai 2 の zero率は MSE 最適点ではありません — QAT の強い傍証

| | zero率 |
|---|---:|
| Bonsai 2 27B 実測(4 tensor) | **32.77 – 32.80 %** |
| PTQ の MSE 最適点(本実験) | 45.6 – 45.8 % |
| absmean 規則(方式 A)の実測 | 30.8 – 30.9 % |
| absmean 規則の解析値(Gaussian) | 31.01 % |

**[推測]** Bonsai 2 の zero率は MSE 最適点から大きく外れており、
むしろ **absmean(BitNet b1.58 型)straight-through 量子化器**の動作点に近い値です。
PTQ でこの点を選ぶ理由は無い(MSE が 40% 悪化する)ので、
**学習時に ternary 制約下で重みが適応した結果**と考えるのが自然です。

### 4-3. したがって、PTQ だけでは Bonsai 2 と同等品質にはなりません

単一 tensor で **cosine 0.90 / 相対 MSE 19%** は、量子化としてはかなり大きな誤差です。
参考までに Q4_K 級は相対 MSE で 1e-3 オーダーです。
Bonsai 2 が FP16 比 98.2% を保持しているのは、この誤差を許容できる重みへ
**学習で持っていったから**であって、後処理で到達できる水準ではありません。

→ **Flash-Next に対する現実的な選択肢は次の3つです。**

1. **routed experts の QAT / 蒸留**(Bonsai と同じ路線)。125B・512 expert では計算資源が要る。
2. **ternary を諦めて 2〜3bit の非 ternary(IQ2_S / IQ3_XXS 等)に留める。**
   現に unsloth の UD-IQ3_XXS が 76.32 GiB で動いており、
   PTQ ternary(cosine 0.90)より確実に良い。
3. **experts の一部だけ ternary、残りを 3-4bit にする mixed。**
   VRAM 目標(Phase 2 §5)に合わせて ternary 比率を決める。

### 4-4. packing 側は完全に再現できています

我々が生成した ternary 重みは `PQ2_0` / `PTQ1_0` へ **byte 完全一致で pack/unpack** できます
(`packing round-trip: exact=True`)。
さらに出荷済み Bonsai 2 27B GGUF の実バイト列に対しても repack が byte 一致します
(`bonsai_format.py` の selftest)。

**足りないのは「良い ternary 重みを作ること」だけで、フォーマット側の障害はありません。**

---

## 5. 測定の限界(正直に)

- **activation は合成です。** 実 activation を取るには qwen4exp の graph を
  動かす必要があり、PrismML fork にはその arch がありません
  (`docs/flash_next_quantization_plan.md` §0-1)。Phase 4 の課題です。
  合成 activation での `out_rel_mse` は weight 側の `rel_mse` とほぼ一致しており
  (0.187 vs 0.187)、追加情報はほとんど持っていません。
- Hadamard の sign vector は **我々が seed 1234 で生成したもの**です。
  Bonsai 2 のものは GGUF から取り出せますが幅が合いません(Phase 1 §15)。
  runtime は任意の ±1 列を受け付けるので実害はありません。
- layer 0 のみを見ています。深い層ほど外れ値が強くなる傾向が一般にあるため、
  layer 24 / 47 でも同じ結論になるかは未確認です。

---

## 6. 次に判断すべきこと

Phase 4(1 layer / 数 expert の forward 検証)に進む前に、方針の選択が必要です。

| 選択肢 | 内容 | Phase 4 の意味 |
|---|---|---|
| (1) PTQ ternary のまま進む | cosine 0.90 の重みで layer forward を測る | 品質は期待できないが「どれだけ壊れるか」の定量値は得られる |
| (2) QAT へ舵を切る | 小規模(1 layer / 数 expert)で ternary-aware fine-tune を試す | 現実的な唯一の ternary 路線。ただし calibration data と学習基盤が要る |
| (3) mixed 2-3bit へ切り替える | ternary をやめ、VRAM 目標だけ満たす | 最短で 5090 に載る。Bonsai とは別物になる |

Phase 3 の結果としては、**(1) をこのまま 125B へ広げることは推奨しません。**
