# Flash-Next progressive reconstruction pilot(layer 4–11, H128)

Flash-Next(`qwen4exp`)の routed experts に、27B で効いた progressive reconstruction
(dual teacher)を 8 層だけ適用し、同じ 8 層の素の PTQ と比較した記録。
Bonsai の学習パイプラインの再現ではない(**Bonsai-inspired ternary quantization**)。

## 1. teacher mapping の確定(手順 ①)

| 項目 | 結果 |
|---|---|
| teacher | IQ3_XXS GGUF を dequantize した層(BF16 canonical は PLE だけで 102 GB あり 5090 系では扱えない) |
| 対応付け | `recon/q4x_layer.py` が GGUF → transformers `Qwen4ExpTextDecoderLayer` を構築(hc norm の −1、V の tiled→grouped、conv1d の分割、QSA indexer の結合など) |
| 数値照合 | llama.cpp を対象層だけ F32 にしたモデルの `l_last-*` dump と比較。**layer 2 (GDN) delta relMSE 2.1e-6、layer 7 (QSA) 2.5e-6** |
| MoE の勾配 | `TernaryMoE`(fused autograd)を float64 gradcheck で 1e-15 一致 |

量子化重みの llama.cpp matmul は活性を q8 化するため、数値照合は対象層を F32 にしたモデルで行った。

## 2. 設定

- 対象: layer 4–11 の routed experts(gate / up / down、各層 512 expert)。shared expert・attention・router は触らない
- 形式: H128(`make_signs(width, 0)`)+ group 128 ternary、mse-opt scale で初期化、PTQ1_0 で格納
- 学習単位: 1 層。student 入力 → student router の top-k(**毎 step 計算、固定しない**)→ 選ばれた ternary experts → shared expert → block 出力
- 目的関数: `relMSE_delta(y, A) + 0.2·relMSE_delta(y, C) + (1 − cos(y, A))`
  (A = teacher(student 入力)、C = teacher(canonical 入力)、27B と同じ dual)
- データ: wikitext-2 train 256 × 512 token(学習)、valid 32 × 512(評価)、IQ3_XXS の `l_last-3`
- 最適化: latent は per-expert Adam を backward 内で適用(全体勾配 10 GB を作らない、m/v は bf16)、
  scale は AdamW。**lr_w 1e-4、300 step/層、cosine**。hc norm は凍結(GGUF へ書き出さないため)
- 速度: 0.65 s/step、peak 28.6 GiB。8 層で約 55 分

### lr の選定(layer 4、200 step)

最初の 2e-4 / 600 step は重み(|w| 中央値 1.6e-2、閾値 ≈ 0.8e-2)に対して大きすぎ、
code が振動して 200 step 時点で PTQ とほぼ同じ(0.1813)だった。

| lr_w | valid delta relMSE(PTQ 0.1865) |
|---|---:|
| 1e-5 | 0.1384 |
| 3e-5 | 0.1273 |
| **1e-4** | **0.1207** |

学習 loss(0.05–0.08)と valid(0.12)の差が大きく、256 系列では既に過学習気味。

## 3. 層ごとの結果(valid)

| 層 | 種別 | delta vsA: PTQ | delta vsA: trained | 出口 stream vs canonical: PTQ | trained |
|---|---|---:|---:|---:|---:|
| 4 | GDN | 0.1865 | 0.1176 | 0.0143 | 0.0090 |
| 5 | GDN | 0.1592 | 0.0993 | 0.0332 | 0.0202 |
| 6 | GDN | 0.2185 | 0.1402 | 0.0490 | 0.0297 |
| 7 | QSA | 0.1660 | 0.0908 | 0.0600 | 0.0342 |
| 8 | GDN | 0.0728 | 0.0468 | 0.0636 | 0.0353 |
| 9 | GDN | 0.1871 | 0.1317 | 0.0664 | 0.0374 |
| 10 | GDN | 0.1747 | 0.1101 | 0.0696 | 0.0391 |
| 11 | QSA | 0.1617 | 0.1107 | 0.0714 | 0.0404 |

(delta vsA の PTQ 列は progressive 側の入力 stream で測った値なので、PTQ run のログとは 5 層目以降わずかに異なる)
出口 stream の誤差は全層で約 43 % 減。zero 率は 0.457 → 0.449。

## 4. モデル全体の評価(llama.cpp、wikitext-2 test 32 × 512、IQ3_XXS 比)

| | IQ3_XXS | PTQ(4–11) | progressive(4–11) |
|---|---:|---:|---:|
| PPL | 2.4783 | 2.5769 | **2.5252** |
| PPL 増分 | — | +0.0986 ± 0.022 | **+0.0469 ± 0.018**(52 % 減) |
| mean KL | — | 0.1898 | **0.1302**(31 % 減) |
| KL 99 % | — | 2.243 | 1.723 |
| same top-1 | — | 88.17 % | 90.00 % |

### routing 変化率(`ffn_moe_topk`、8 × 512 token、IQ3_XXS との top-10 集合の差)

| 層範囲 | PTQ: 集合変化 | PTQ: top-1 変化 | progressive: 集合変化 | progressive: top-1 変化 |
|---|---:|---:|---:|---:|
| 0–4(入力が同じ) | 0 | 0 | 0 | 0 |
| 5–11(置換層) | 0.121 | 0.194 | 0.084 | 0.129 |
| 12–46(下流) | 0.145 | 0.216 | 0.114 | 0.169 |

layer 0–4 の変化 0 は測定系の健全性確認(layer 4 の router は未変更の stream を見る)。
routing の変化は置換層の後も下流で減衰せず、むしろ増える。

## 5. 判定

基準「8 層 PTQ に対し KL / PPL 増分が半分以下」に対して:

- **PPL 増分: 52 % 減で基準をぎりぎり満たす**(ただし誤差幅 ±0.02 に対して差 0.05 なので余裕は無い)
- **KL: 31 % 減で基準に届かない**

27B の 16 層 pilot(KL が PTQ の 8.5 分の 1)ほどの効果は出ていない。
主な違いとして考えられるのは、(a) Flash-Next の IQ3 teacher 自体が低 bit で、PTQ の初期誤差が
27B(relMSE 0.47)より小さい(0.19)ため伸び代が小さい、(b) 256 系列で既に過学習している、
(c) routing の変化が下流に広がり、層単位の出力再構成では捉えられない、の 3 点。

## 6. 次の候補

1. 学習データを 4 倍(1024 系列、bf16 保存で 10 GB)にして過学習を抑える
2. hc norm と shared expert gate を学習対象に戻し、GGUF 書き出しに対応させる
3. 下流の routing 変化を直接抑える項(次層 router logits の KL)を目的関数に加える

## 7. 追加実験 1: 学習データ 4 倍(1024 系列、loss は同じ)

wikitext-2 train の chunk 0–1023(既存 256 + 新規 768)。他の設定は §2 と同じ(lr 1e-4、300 step/層、
hc norm 凍結)。300 step × batch 2 なので各系列は高々 1 回しか見ない(256 系列では平均 2.3 回)。

| | PTQ | 256 系列 | **1024 系列** | 48 層へ進む目安 |
|---|---:|---:|---:|---:|
| PPL | 2.5769 | 2.5252 | **2.5191** | |
| PPL 増分 | +0.0986 | +0.0469 | **+0.0408** | |
| mean KL | 0.1898 | 0.1302 | **0.1149** | ≤ 0.10–0.11 |
| KL 99 % | 2.243 | 1.723 | 1.455 | |
| same top-1 | 88.17 % | 90.00 % | 90.49 % | |
| routing 集合変化: 置換層 5–11 | 12.1 % | 8.4 % | 7.9 % | < 7 % |
| routing 集合変化: 下流 12–46 | 14.5 % | 11.4 % | 10.9 % | < 8–9 % |
| stream relMSE: layer 4 / 7 / 11 | 0.0143 / 0.0600 / 0.0714 | 0.0090 / 0.0342 / 0.0404 | 0.0086 / 0.0320 / 0.0373 | |

データ増で KL は 12 %、KL 99 % は 16 % 下がった。ただし KL は 0.10 付近に届かず、
下流の routing 変化もほとんど動かない。データ不足は原因の一部に過ぎず、routing の増幅は
層単位の出力再構成では抑えきれていない。→ 次層 router の KL 項を加える(§8)。

途中経過: layer 8 の npz 保存中に `/` が満杯になり、`np.savez` がエラーを出さずにファイルを
切り詰めた。保存後のサイズ検証を追加し、`--resume`(保存済みの層は npz の値でストリームだけ進める)で
layer 8 から再開した。復元した layer 4–7 の stream relMSE は記録値と 4 桁一致。

## 再現

```bash
T=tools/flash_next_ternary/recon
python $T/q4x_progressive.py --first 4 --last 11 --steps 0   --out /data/eval/q4x_pilot_ptq
python $T/q4x_progressive.py --first 4 --last 11 --steps 300 --eval-every 300 --lr-w 1e-4 --lr-norm 0 \
    --out /data/eval/q4x_pilot_prog
bash $T/eval_q4x_variant.sh base
bash $T/eval_q4x_variant.sh ptq  /data/eval/q4x_pilot_ptq  4 11
bash $T/eval_q4x_variant.sh prog /data/eval/q4x_pilot_prog 4 11
python $T/routing_change.py /home/sohey/AI/LLM/q4x_eval/route_base /home/sohey/AI/LLM/q4x_eval/route_prog

# §7: 1024 系列(chunk 256-1023 を追加採取)
dump_hidden_q4x -m <IQ3_XXS> -f wiki.train.raw -c 512 -b 512 -ub 512 -ngl 24 --no-logits \
    --n-chunks 768 --chunk-offset 256 --chunk-len 512 --dump-filter '^l_last-3$' --dump-dir act_train_256_1024
python $T/q4x_progressive.py --first 4 --last 11 --steps 300 --eval-every 300 --lr-w 1e-4 --lr-norm 0 \
    --train-dir /data/eval/q4x_act_train act_train_256_1024 --out pilot_d1024 [--resume]
bash $T/eval_q4x_variant.sh d1024 pilot_d1024 4 11
```

PTQ / 256 系列の npz と評価済みモデルは `/mnt/f/q4x_archive/` に退避してある。
