# suffix-aware progressive の pilot(layer 48–63)

目的: block 出力の誤差のうち「最終 logits に効く成分」を直接学習するため、学生 block の出力を
**凍結した BF16 の後段(suffix)+ head** に通し、teacher logits(canonical BF16)との KL を loss に加える。

    L = L_block(A/anchor/cos)  +  λ · KL_top256( teacher || student via BF16 suffix )

実装: `progressive.py --suffix-kl λ --topk 256`。teacher top-256 は C_64 → BF16 head で事前計算。
suffix は bf16 で GPU 常駐(層ごと checkpointing)、勾配は学生 block にのみ流す。

## なぜ layer 48–63 か

layer 0–15 で行うと suffix が 48–60 層(bf16 46 GB)になり GPU に載らない(CPU から毎 step 流すと 10 s/step)。
後段 16 層なら suffix ≤ 15 層(11.5 GB)で収まる。学生 block は 2 層でも 32 GB を超えたため **1 層 block**
(latent 6 GB + Adam)に縮小。比較は同条件の λ=0(base1)と λ=1(suffix1)、入力は共通の学生 stream S_48、
評価は全 64 層モデル(0–47 は dual 版)の wikitext PPL / KL。

## 結果

| 変種(layer 48–63 を 1 層 block × 600 step、bs 1) | 出口 L63 stream cos / relMSE | wikitext PPL | Mean KLD | top-1 |
|---|---|---:|---:|---:|
| base1(λ = 0) | 0.869 / 0.234 | **11.68** | **0.776** | **65.1 %** |
| suffix1(λ = 1、top-256 KL) | 0.854 / 0.253 | 12.24 | 0.837 | 63.8 % |
| 参考: 4 層 dual × 1500 step(元 run) | 0.881 / 0.213 | 11.07 | 0.736 | 65.9 % |
| 参考: 2 層 block × 600 step、λ = 0 | 0.864 / 0.241 | 12.25 | 0.824 | 64.1 % |

suffix1 の valid suffix-KL(top-256)の推移: L48 で 0.333 → **0.305**(1 層目は最終 KL を 8 % 下げた)、
以後 0.309 / 0.320 / 0.336 / 0.351 / 0.363 / 0.381 / 0.405 / 0.428 / 0.446 / 0.464 / 0.489 / 0.506 / 0.519 / 0.542 / **0.629**(L63)。
訓練バッチの suffix-KL は 0.06–0.10 まで落ちており(valid 0.4–0.6)、**logits 項は強く過学習**。

## 判断

- suffix-aware は同条件の baseline より **悪化**(KL 0.776 → 0.837)。判定基準(0.183 → 0.12 相当の明確な改善)には遠く及ばない。
- 原因は joint 蒸留と同じ: logits KL は token あたりの制約が弱く、33 万 token では過学習して汎化しない。
  凍結 suffix を通した勾配は 1 seq/step ではノイズが大きく、hidden 再構成項と競合して stream 精度も落とす(relMSE 0.234 → 0.253)。
- 1 層目(L48)で valid KL が 8 % 下がった事実は「方向は正しい」ことを示すが、
  この効果はデータを桁で増やさないと累積しない。
- **27B をこれ以上詰める費用対効果は低い。Flash-Next へ移る。**

## Flash-Next への持ち込み方(この時点の結論)

1. expert 単位の reconstruction(teacher A = BF16 expert(学生入力)、teacher B = canonical、hidden 目標)が主。
   データ効率が高く、5090 1 枚で回る。125 B でも expert 1 個(2560×640 × 3)は 27B の 1 層より小さい。
2. logits ベースの較正(head / suffix / joint)は 27B で一貫して効果が小さく、データ律速。
   Flash-Next で入れるなら **数 M token 以上**を前提にし、最初は省く。
3. 出荷 Bonsai 2 の優位(KL 0.37 vs 0.74)は stream と head の共適応 + 大規模学習にあり、
   局所再構成では 2 倍の KL 差が残る。この差を許容できる用途(RTX 5090 での 125B 実用化)かどうかが
   Flash-Next 側の設計判断になる。
