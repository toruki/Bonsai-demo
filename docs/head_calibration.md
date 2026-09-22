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
