# 頭の較正: stream × head クロス評価と final norm / lm_head の KL 学習

コード: `tools/flash_next_ternary/recon/head_calib.py`(評価・学習)、`student_utils.py` / `regen_stream.py`(学生 stream の再生成)、
`train_joint.py`(blocks 60–63 + head の joint 蒸留)。データ: valid 32 seq × 512 token(評価)、train 640 seq(学習)。
「stream」は layer 63 出口の residual、「head」は final RMSNorm + lm_head。出荷の head は
final norm(F32、学習済み)+ 回転(sign→H₁₀₂₄)+ ternary `output.weight` を PyTorch に載せ直したもの。

## 1. クロス評価(BF16 stream → BF16 head の logits を基準、valid 16k token)

| stream → head | KL | top-1 | PPL(valid) |
|---|---:|---:|---:|
| ① BF16 → BF16 | 0 | 100 % | 6.562 |
| ② progressive → BF16(現状) | 0.550 | 71.8 % | 10.460 |
| ③ 出荷 → 出荷(現状の出荷) | 0.307 | 77.7 % | 8.339 |
| ④ 出荷 → BF16 | 0.474 | 75.2 % | 10.121 |
| ⑤ progressive → 出荷 | 0.859 | 59.8 % | 13.875 |
| ⑥ BF16 → 出荷 | 0.337 | 73.0 % | 8.806 |

読み取れること:

- ④ → ③: 出荷は **stream と head が共適応**しており、head 側で KL を 0.474 → 0.307(−35 %)稼いでいる。
- ただし **BF16 head の下でも出荷 stream(0.474)は progressive stream(0.550)より良い。**
  progressive の stream は residual の cos / relMSE では出荷より BF16 に近い(layer 63: 0.884 / 0.193 vs 0.722 / 0.668)のに、
  logits に効く成分では劣る。**hidden MSE ≠ logit 品質** の直接証拠で、reconstruction の目的関数が
  「head が読む方向」を優先していないことを意味する。
- ⑤ は最悪(0.859): head と stream は対でしか意味を持たない。⑥: 出荷 head は BF16 stream に対しても 0.337 で、
  単体で優れた head ではない。
- したがって「出荷の優位は head にある」は一部(35 %)しか正しくない。残りは stream 自体の中身の差。

## 2. progressive stream を固定した head 較正(teacher = BF16 logits、L = KL + 0.1·CE、400 step)

| step | 学習対象 | params | valid KL | top-1 | PPL |
|---|---|---:|---:|---:|---:|
| — | なし(②) | 0 | 0.550 | 71.8 % | 10.46 |
| 1 | final RMSNorm | 5 k | 0.542 | 71.8 % | 10.33 |
| 2 | + 入力 channel scale + vocab bias | 0.26 M | 0.538 | 71.8 % | 10.25 |
| 3 | + LoRA r=64 on lm_head | 16.5 M | 0.536 | 71.8 % | 10.07 |
| 4 | full lm_head(1.27 B) | — | 未実施(fp32 W + Adam ≈ 25 GB で 32 GB GPU が飽和し実用速度が出ない) |

**head 側だけでは KL は 2.5 % しか下がらない。** top-1 は動かない。出荷が head で 35 % 稼いでいるのは、
head が stream と *同時に* 学習されたからであって、固定 stream に対する head の後付け較正では再現できない。

→ 結論: 残りの差は stream 側にあり、**stream と head を一緒に動かす**必要がある(次節の joint 蒸留)。

## 3. 次: blocks 60–63 + final norm + lm_head(LoRA)の joint logit 蒸留

学生入力 = 再生成した S_60(学生 stream)、teacher = BF16 logits(canonical C_64 → BF16 head)、
loss = KL + 0.1·CE + 0.1·relMSE(stream_64 vs C_64)。最終 4 層(latent / scale / norm)+ final norm + head LoRA を同時に学習。

## 4. joint 蒸留の結果(blocks 62–63 + final norm + head LoRA、600 step、KL + 0.1·CE + 0.1·anchor)

4 層 + head は latent 1.5 B の fp32 + Adam で 32 GB を超え実用速度が出なかったため、最終 2 層に縮小
(S_62 は S_60 から学生 60–61 で再生成)。lm_head の fp32 化は vocab チャンク分割に変更(5 GB × 2 の一時確保を回避)。

| | valid KL | top-1 | valid PPL |
|---|---:|---:|---:|
| 学習前 | 0.550 | 71.8 % | 10.45 |
| head のみ最良(LoRA) | 0.536 | 71.8 % | 10.07 |
| **joint(62–63 + head)** | **0.529** | **72.2 %** | **9.74** |

訓練バッチの KL は step 100 で既に 0.06–0.2(valid 0.53)まで落ちており、**logits 蒸留は 33 万 token で強く過学習**する。
hidden 目標(token あたり 5120 次元)に比べて logits KL は制約が弱く、この段はデータ量が律速。

## 5. 最終 PPL / KL(wikitext-2 test、64 chunk / KL 16 chunk、PTQ1_0 GGUF、GPU 全載せ)

| モデル | ternary 対象 | PPL | Mean KLD | top-1 |
|---|---|---:|---:|---:|
| BF16 | — | 6.513 | — | — |
| PTQ-mseopt | 64 層 + embed + head | 829.6 | 5.10 | 10.9 % |
| progressive dual | 64 層 | 11.07 | 0.736 | 65.9 % |
| **progressive + joint 較正** | 64 層(head は BF16 + LoRA 畳み込み) | **10.41** | **0.710** | **66.5 %** |
| 出荷 Bonsai 2 | 64 層 + embed + head | 8.62 | 0.367 | 75.2 % |

成果物: `/data/models/gguf/progressive-joint-27b-PTQ1_0.gguf`(10.6 GB)。

## 6. 結論

- ① 〜 ⑤ を通しての KL は 0.736 → 0.710。目標としていた 0.4 前後には届かず、**head 側・最終 block 側の後付け較正では
  残りの差はほとんど埋まらない**。クロス評価が示したとおり、差の本体は stream の中身(head が読む成分)にあり、
  出荷はそれを stream と head の同時学習で獲得している。
- 一方で reconstruction 段(progressive)は 33 万 token で PTQ から 75 倍改善しており **データ効率が非常に高い**。
  logits 段だけが data-hungry。
- したがって残る差を詰める現実的な手段は、(a) 最終段の蒸留データを 10 倍以上(数 M token)に増やす、
  (b) progressive の目的関数に「head が読む方向」を入れる(例: block 出口の loss を BF16 head を通した logits KL で
  重み付け、あるいは lm_head の主成分方向で重み付けした hidden MSE)、の 2 つ。(b) は progressive の
  データ効率を保ったまま、クロス評価で見えた「hidden MSE ≠ logit 品質」を直接是正する案で、次に試す価値が最も高い。
- Flash-Next への含意: 「expert 単位の reconstruction(データ効率が高い)+ 出力側の軽い蒸留」の二段構成は
  有効だが、二段目は expert 再構成より桁違いにデータが要る。二段目を省いて一段目の目的関数を logit-aware にする方が
  125 B では現実的。
