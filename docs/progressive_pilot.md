# 16 層 progressive pilot(layer 0–15、4 block)

コード: `tools/flash_next_ternary/recon/progressive.py`、driver `/data/eval/prog_pilot.sh`
データ: wiki.train 640 seq × 512 token(学習)、wiki.valid 32 seq(検証)。
canonical BF16 stream は token embedding から PyTorch で BF16 層を順に流して生成し、
llama.cpp の `l_out-3/7/11/15` と **relMSE ≤ 2.3e-6** で一致することを各 block で確認済み。

## 設定

block b(4 層)ごとに:

- 学生入力 `S_b` = 学習済み学生 block 0..b−1 の出力(f16)
- Teacher A = BF16 block b(`S_b`)、その中間層出力も保持
- Teacher B = canonical BF16 stream `C_{b+1}`
- loss = relMSE_delta(y, A_exit) + anchor·relMSE_delta(y, C_exit) + (1 − cos(y, A_exit)) + aux·Σ relMSE_delta(h_j, A_int[j])
- 学習: latent + scale + RMSNorm、AdamW(2e-4 / 1e-4 / 1e-4)、cosine 減衰、1500 step、bs 2、checkpointing、~18 分/block
- 変種: **ptq**(学習なし)、**aonly**(anchor 0, aux 0)、**dual**(anchor 0.2, aux 0.1)

評価: 各 block 出口で学生 stream vs canonical の cos / relMSE(valid)。
16 層終了後、layer 0–15 だけ ternary に差し替えた mixed モデル(残り 48 層 BF16)を
fork converter で F16 GGUF 化し、wikitext-2 PPL(64 chunk)と BF16 基準の KL(16 chunk)。

## 結果

### 累積誤差(学生 stream vs canonical BF16、valid)

| 出口 | PTQ cos / relMSE | A-only cos / relMSE | dual cos / relMSE |
|---:|---|---|---|
| layer 3 | 0.965 / 0.086 | 0.995 / 0.015 | 0.995 / 0.0145 |
| layer 7 | 0.929 / 0.186 | 0.987 / 0.040 | 0.988 / 0.038 |
| layer 11 | 0.923 / 0.229 | 0.986 / 0.043 | 0.987 / 0.040 |
| layer 15 | 0.904 / 0.279 | 0.984 / 0.045 | **0.985 / 0.042** |

### mixed モデル(layer 0–15 ternary、残り BF16)

| 変種 | wikitext PPL | Mean KLD (vs BF16) | top-1 一致 |
|---|---:|---:|---:|
| BF16(参照) | 6.513 | — | — |
| PTQ | 28.13 (+332 %) | 1.568 | 48.0 % |
| A-only progressive | 7.10 (+9.0 %) | 0.198 | 83.4 % |
| **dual progressive** | **7.04 (+8.0 %)** | **0.183** | **83.8 %** |

## 読み取れること

1. **超線形発散は起きない。** PTQ は relMSE 0.086 → 0.279 と伸び続けるが、progressive は
   0.015 → 0.040 → 0.043 → 0.045 で **飽和**する。各 block が上流誤差を吸収している。
2. **KL は PTQ の 8.5 分の 1**(1.57 → 0.18)。判定基準「半分前後」を大きく超えた。
   4 層の時の 0.033 を単純に 4 倍すると 0.13 で、実測 0.18 はそれに近い(ほぼ線形)。
3. **dual は A-only よりわずかに良い**(全 block で一貫)。anchor 0.2 / aux 0.1 は害がなく、
   後段でも同様なら全体で効く。差は小さいので、主効果は progressive そのもの。
4. delta vsC(block 寄与を canonical と比べた値)は cos 0.80–0.91 に留まる一方、stream vsC は 0.985。
   各 block は「canonical の層寄与」を再現しているのではなく、**上流のずれを含めて stream を
   canonical に寄せる**方向に学習している。出荷 Bonsai の層が単体では BF16 層を再現しない
   (`docs/layer32_recon.md` §1)ことと同じ構図。

## 判断

残り 48 層を dual で続行(layer 16 から再開、`/data/eval/prog_full_dual.sh`)。
全 64 層 ternary(embed / lm_head は BF16 のまま)の完全モデルで PPL / KL を測り、
出荷 Bonsai 2(8.62 / 0.367)と比較する。所要 ~4.5 時間 + export/評価 15 分。
