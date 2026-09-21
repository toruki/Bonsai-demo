# Bonsai 2 27B ↔ Qwen3.8-27B の weight 対応解析(Phase 3.5)

目的: 出荷 Bonsai 2 27B の ternary code / scale を、公開されている base model
`Qwen/Qwen3.8-27B`(BF16)と **要素単位で対応付け**、PrismML の変換工程を結果から逆算する。

コード: `tools/flash_next_ternary/bonsai_reverse.py`
データ: Bonsai 側は `models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf` の実バイト列
(Phase 1 で byte-exact にデコード済み)。base 側は HF から該当 tensor の行範囲だけ Range 取得。

対象: layer 0 の `ffn_gate`(1024 行)/ `ffn_up`(512 行)/ `ffn_down`(256 行、入力幅 17408)、
layer 32 / 63 の `ffn_gate`(512 行)。合計 約 1,300 万要素。

---

## 1. fold 規約は実データで一意に確定した **[確認]**

base weight `W` を 5 通りの規約で回転し、Bonsai の `t·d` との相関を取った結果(layer 0 ffn_gate):

| fold 規約 | corr(W', t·d) | 符号一致率 (t≠0) | AUC (t=0 vs t≠0) |
|---|---:|---:|---:|
| H のみ(sign 無し) | −0.029 | 0.486 | 0.500 |
| **sign → H(runtime と同じ)** | **0.878** | **0.99964** | **0.965** |
| H → sign | 0.002 | 0.501 | 0.500 |
| 回転なし | 0.001 | 0.500 | 0.500 |
| norm を fold してから sign → H | 0.877 | 0.99952 | 0.963 |

- `W_folded = (W · s) @ H_1024` **のみ**が説明力を持ち、他は完全に無相関。
  Phase 1 で runtime から導いた数式が、出荷重みに対しても成立する。
- GGUF に格納された sign vector は **そのまま使われている**(学習された別の S ではない)。
- RMSNorm を weight に fold した痕跡は無い(fold すると僅かに悪化)。
- ffn_up / ffn_down / layer 32 / layer 63 でも同じ(corr 0.877–0.885、符号一致 99.94–99.98%)。

## 2. しかし単一 threshold では説明できない **[確認]**

`u = W' / d`(base weight を Bonsai の scale で正規化)に対し、group ごとに
最良の単一 threshold を置いたときの誤分類率:

| tensor | 誤分類率 | 完全分離できた group | threshold/d の中央値 |
|---|---:|---:|---:|
| L0 ffn_gate | 8.28 % | 0.002 % | 0.371 |
| L0 ffn_up | 8.79 % | 0.000 % | 0.370 |
| L0 ffn_down | 8.00 % | 0.003 % | 0.372 |
| L32 ffn_gate | 8.29 % | 0.03 % | 0.373 |
| L63 ffn_gate | **4.96 %** | 1.9 % | 0.378 |

`u` の分布を code 別に見ると帯が重なっている(layer 0 ffn_gate、bin 幅 0.125、−2〜2):

```
code -1:  1.7 2.3 3.0 3.9 4.8 6.0 7.2 8.6 10.0 11.4 12.5 12.1 9.3 5.1 1.8 0.3 | 0 ...
code  0:  ... 0 | 0.4 2.0 5.8 10.8 14.7 16.4 16.4 14.7 10.7 5.8 2.0 0.4 | 0 ...
code +1:  ... 0 | 0.3 1.8 5.1 9.4 12.1 12.5 11.4 10.0 8.6 7.2 6.0 4.8 3.8 3.0 2.3 1.7
```

- `t=0` の要素の |u| は p90 = 0.43、p99 = 0.61 まで伸びる。
- `t=±1` の要素の |u| は p1 = 0.19、p10 = 0.42 から始まる。
- 符号反転(`t = −sign(W')`)は **0.02–0.06 %** しかない。

→ **Bonsai の code は base weight の決定的関数ではない。** 境界付近の要素が
base とは違う側に落ちており、これは *latent weight が学習で動いた* 痕跡です。
符号反転がほぼ無いのは、学習で 0 を跨いで動いた重みが稀であることと整合します。

さらに **norm weight も base から変わっている**(converter の `+1` 規則を考慮した上で):

| tensor | max\|diff\| | rms diff | 完全一致の割合 |
|---|---:|---:|---:|
| `blk.0.attn_norm.weight` | 0.071 | 0.0089 | 1.2 % |
| `blk.0.post_attention_norm.weight` | 0.037 | 0.0033 | 14.8 % |
| `output_norm.weight` | 0.051 | 0.0081 | 26.2 % |

ternary 化されない norm まで動いているので、**Bonsai 2 は base checkpoint の後処理ではなく、
学習を経た checkpoint** と結論して良いと考えます **[確認に近い推測]**。

## 3. それでも「隠れた PTQ 規則」の輪郭は見える **[確認]**

base weight に各種 threshold 規則を適用し、Bonsai の code との一致率を測定(layer 0 ffn_gate、524 万要素):

| 規則 | 一致率 | zero率 | 0→非0 | 非0→0 |
|---|---:|---:|---:|---:|
| Bonsai の d を使い threshold 0.5·d(通常 RTN) | 87.72 % | 41.9 % | 1.56 % | 10.72 % |
| Bonsai の d を使い threshold 0.37·d | 89.93 % | 31.7 % | 5.58 % | 4.50 % |
| **threshold = 0.5·mean\|W'\|(absmean、BitNet b1.58 型)** | **89.87 %** | 30.9 % | 6.02 % | 4.12 % |
| threshold = 0.52·mean\|W'\| | 89.99 % | 32.1 % | 5.37 % | 4.64 % |
| threshold = 0.41·rms(W') | 89.86 % | 31.7 % | 5.63 % | 4.51 % |
| 方式 C / D(weight-MSE 最適、Phase 3) | 85.15 % | 45.8 % | 0.94 % | 13.91 % |

- **Bonsai の動作点は weight-MSE 最適ではなく、`threshold ≈ 0.5·mean|W_group|` の点**。
  これは BitNet b1.58 の absmean 量子化器の threshold と同じです。
- ただし不一致 10 % は threshold のずれではなく境界付近の "揺れ" です:
  一致要素の |u| は threshold から中央値 0.345 離れているのに対し、
  不一致要素は中央値 0.077 しか離れていない(p90 でも 0.206)。
- 深い層ほど一致が良い(layer 63 で誤分類 5 %)。
  浅い層ほど学習で動いた、という解釈と整合します。

scale 側(layer 0 ffn_gate、Bonsai の d を何が再現するか):

| scale 候補 | d / 候補 の中央値 | CV | 相対誤差 rms |
|---|---:|---:|---:|
| `mean|W'|`(absmean scale) | 1.389 | 0.023 | 0.39 |
| **`mean|W'|` over Bonsai の support (t≠0)** | **1.036** | **0.021** | **0.043** |
| `<W',t>/<t,t>`(code 固定の LSQ 最適) | 1.036 | 0.021 | 0.043 |
| TWN 閉形式(support = \|w\|>0.5·mean) | 1.040 | 0.033 | 0.052 |
| `amax`(packer の RTN 規則) | 0.398 | 0.129 | 0.60 |

- **`d ≈ 1.04 × (support 上の mean|W'|)`**、CV 2 %。scale は "非ゼロ要素の平均絶対値" 型で、
  BitNet の `mean|W|` 全体平均(比 1.39)ではありません。
- Gaussian 解析値でも threshold 0.5·mean|w| → support-mean scale = 1.339·mean|w| となり、
  実測の 1.35–1.39 と一致します。
- どの候補も **完全一致はしない**(最良でも 4 % 前後の残差)。
  d が base ではなく学習後の latent weight から計算されているなら、当然そうなります。
  残差の符号(実 d が base 推定より 3.6 % 大きい)は、学習で非ゼロ重みの大きさが
  僅かに育ったと読めます **[推測]**。

## 4. 結論

| 項目 | 状態 |
|---|---|
| fold 規約(sign→H、block 1024、shipped sign vector) | **確定**。出荷重みで検証済み |
| threshold 規則の動作点 | `≈ 0.5·mean|W'_group|`(absmean 型)。base weight で 90 % の code を再現 |
| scale 規則 | `≈ mean|W'|` over support(TWN / LSQ 型)、比 1.04、CV 2 % |
| 残り 10 % の code と norm の変化 | **学習由来**。単一 threshold・単一 scale では説明不能 |
| PrismML の工程の推定 | absmean 型 STE 量子化器を用いた **ternary-aware training(QAT / 蒸留)**、初期値は Qwen3.8-27B。sign vector は固定 |

つまり **「PTQ で 90 % まで行き、残り 10 % + norm を学習で動かした」** という絵になります。
"隠れた PTQ アルゴリズムだけで Bonsai 2 が作れる" 可能性は、この時点でほぼ否定されます。

## 5. この結果が可能にする次の実験

Bonsai 2 27B の arch(`qwen35`)は **既存 fork runtime がそのまま動かせます**。
したがって、上で同定した規則(sign→H 1024、threshold 0.5·mean|W'|、support-mean scale)を
base Qwen3.8-27B に PTQ として適用し、同じ GGUF contract で pack すれば、

```
Qwen3.8-27B BF16  --(同定した PTQ 規則)-->  "PTQ-Bonsai"  vs  出荷 Bonsai 2 (QAT)
```

を **同じ runtime・同じ perplexity / KL 測定**で比較できます。
これは「QAT がどれだけ品質を買っているか」の直接測定であり、Flash-Next で
PTQ ternary を諦めるべきか、短時間 QAT でどこまで戻るかの見積もりになります。
必要なもの: base 27B BF16(18 shard、約 54 GB、`/data` に置く)、fold + pack の
オフライン処理(CPU で可)、`hadamard_packing.json` の生成、fork の `convert_hf_to_gguf.py`。
