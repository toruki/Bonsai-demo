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

## 8. 次層 router 項(不採用)と gate 重み付き routing

### 8.1 router 項の sweep(layer 4、1024 系列、300 step)

次層(layer 5)の teacher を router まで通し、teacher 出力 A と学生出力の router logits を比べる項を追加。

| 次層 router 項 | 次層 routing 変化 | 次層 KL | delta vsA |
|---|---:|---:|---:|
| (PTQ) | 8.97 % | 0.0058 | 0.1865 |
| **なし** | **5.94 %** | **0.0026** | **0.1125** |
| KL λ 0.03 / 0.1 / 0.3 | 5.97 / 5.92 / 5.95 % | 0.0026 | 0.1127 |
| KL λ 3 / 10 / 30 | 7.02 / 7.60 / 8.08 % | 0.0038 / 0.0045 / 0.0053 | 0.1206 / 0.1388 / 0.1647 |
| logit 相対 MSE λ 0.3 / 1 | 6.27 / 6.58 % | 0.0028 / 0.0032 | 0.1136 / 0.1182 |

小さい λ は効かず(KL が 0.002 程度で勾配に寄与しない)、大きい λ は routing も再構成も悪化させる。
学習中の train KL 自体が上がるので、汎化ではなく最適化の問題(STE + Adam で、ノイズの多い router 勾配に
全 latent が同じ歩幅で動き code が反転する)。router を直接合わせる方針は採らない。

### 8.2 gate 重み付き routing(1024 系列、8 × 512 token、`routing_mass.py`)

| 層 | 集合変化 | 入れ替わった gate 重み | p90 | 入れ替わり重み > 0.1 のトークン | 512 分布 KL | 入力 stream のずれ | MoE 出力 relMSE / cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4(入力同一) | 0 | 0 | 0 | 0 | 0 | 0 | 0.191 / 0.906 |
| 5–11 | 7.9 % | 4.6 % | 10.3 % | 13 % | 0.005 | 0.023 | 0.251 / 0.856 |
| 12–46 | 10.9 % | 7.0 % | 14.6 % | 23 % | 0.010 | 0.059 | 0.177 / 0.905 |

入れ替わる expert は平均(10 %)より軽いが無視できる重みではない。512 分布の KL は非常に小さく、
上位 10 の重みが平たいために境界の入れ替わりにも相応の重みが乗る。

### 8.3 MoE 出力誤差の分解(下流 10 層、`moe_decompose.py`)

学生の実入力を teacher の expert に通し、学生自身の routing と元モデルの routing(top-k + 重み)を
強制した場合で比べる。Python と llama.cpp の MoE 出力の一致は全層で relMSE 0.0003–0.0017。

| 下流 10 層平均 | 合計 | routing 分 | 入力ずれ分 | routing の割合 |
|---|---:|---:|---:|---:|
| PTQ | 0.288 | 0.093 | 0.207 | 0.32 |
| 1024 系列 progressive | 0.180 | 0.061 | 0.123 | 0.34 |

routing は MoE 出力誤差の約 1/3。その割合は層によらず 0.28–0.37 で、PTQ と progressive でも同じ。
routing の変化は stream のずれに比例して起きる二次的な結果で、独立した壊れ方ではない。
→ 置換層での stream 誤差を下げることが唯一の効く手。λ_router = 0 のまま 16 層へ拡張する(§9)。

## 9. 16 層への拡張(layer 4–19、1024 系列、router 項なし)

layer 4–11 は §7 の npz から `--resume` で復元(stream relMSE は記録値と一致)、12–19 を新たに学習。

| | PTQ(8 層) | 8 層 | **16 層** | 目安 |
|---|---:|---:|---:|---:|
| PPL | 2.5769 | 2.5191 | **2.5932** | |
| PPL 増分 | +0.099(+4.0 %) | +0.041(+1.6 %) | **+0.115(+4.6 %)** | +3–4 % |
| mean KL | 0.190 | 0.115 | **0.180** | 0.2 前後 |
| KL 99 % | 2.24 | 1.45 | 2.29 | |
| same top-1 | 88.2 % | 90.5 % | 88.3 % | |
| 入れ替わった gate 重み: 置換層 / 下流 | — | 4.6 / 7.0 % | 6.6 / 9.6 % | |

層ごとの学習効果(valid delta vsA、PTQ → 学習後):

| 層 | 12 | 13 | 14 | 15 (QSA) | 16 | 17 | 18 | 19 |
|---|---|---|---|---|---|---|---|---|
| 出口 stream relMSE | 0.0412 | 0.0441 | 0.0481 | 0.0560 | 0.0678 | 0.0742 | 0.0792 | 0.0846 |

layer 15 / 16 は下げ幅が 20 / 23 %(他の層は 35–48 %)で、layer 16 は PTQ 誤差 0.278 がこれまでの最大。
後半 8 層の悪化はこの 2 層に偏る。KL は層数 2 倍で 1.57 倍(線形以下)、PPL 増分は 2.8 倍
(ただし差の誤差幅 ±0.022)。

## 10. layer 16 の step 数(300 / 600 / 900)

layer 16 は PTQ 誤差 0.278 と最も難しい層。validation を 50 step ごとに測り、変更しない teacher の
次 2 層に通した stream 誤差(probe)も見た。

| layer 16 | delta vsA | 自層出口 stream | +1 層 | +2 層 |
|---|---:|---:|---:|---:|
| PTQ | 0.2780 | 0.0702 | 0.0724 | 0.0717 |
| 300 step | 0.2126 | 0.0678 | 0.0701 | 0.0694 |
| 600 step | 0.2019 | 0.0673 | 0.0696 | 0.0689 |
| 900 step | 0.1977 | 0.0672 | 0.0695 | 0.0688 |

900 step でも 0.198 止まりで、stream と probe はほぼ動かない。step 数ではなく ternary の表現能力が
ボトルネック。600 / 900 step では lr が高い前半に validation が一度悪化し(step 50: 0.245 →
step 100: 0.263)、改善は lr が下がる後半に来るので、単純な patience の early stop は最良点を取り違える。
→ 48 層は固定 300 step で進めた。

## 11. layer 4–47(44 層)

layer 0–3 は元の IQ3_XXS のまま(PLE 入力の近似や未学習 PTQ を混ぜず、44 層の効果だけを測るため)。

| | IQ3_XXS | 8 層 | 16 層 | **44 層** | 全 48 層 PTQ(§ qwen4exp_port) |
|---|---:|---:|---:|---:|---:|
| PPL | 2.478 | 2.519 | 2.593 | **3.168(+27.8 %)** | 22.3 |
| mean KL | — | 0.115 | 0.180 | **0.488** | — |
| KL 99 % | — | 1.45 | 2.29 | 5.26 | — |
| same top-1 | — | 90.5 % | 88.3 % | 79.9 % | — |
| 入れ替わった gate 重み(置換層平均) | — | 4.6 % | 6.6 % | 10.4 % | — |

出口 stream relMSE は layer 29–32 で急増(1 層 +0.02–0.035)した後、layer 33–44 は 0.27–0.29 で
頭打ち、layer 45 で 0.20 に下がる。発散はしない。学習の削減率は後段でも 33–38 % を保つが、
各層で増える stream 誤差の多くは、入ってきたずれを teacher 層自体が増幅する分(目標 A に含まれ、
その層の expert では消せない)。KL は線形外挿(0.4 前後)より約 2 割悪い。

### 推論時 VRAM(RTX 5090 32 GB、`llama-bench -ngl 99 -fa on`、アイドル 0.8 GB 込みのピーク)

| 構成 | VRAM | pp512 | tg128 |
|---|---:|---:|---:|
| 全部 GPU | 32.0 GB(上限、共有メモリへ溢れる) | 210 t/s | 21 t/s |
| layer 0–3 の expert を CPU(`-ot 'blk\.[0-3]\.ffn_(gate\|up\|down)_exps=CPU'`) | **28.6 GB** | 780 t/s | **76 t/s** |

## 12. 学習時の VRAM(32 GB → 24 GB)

layer 4、300 step、1024 系列での実測。delta relMSE はすべて 0.1125 で結果は変わらない。

| | 修正前 | 16 bit Adam | 8 bit Adam | 8 bit + `--max-vram 24` |
|---|---:|---:|---:|---:|
| 学生構築までの peak(PyTorch) | 28.6 GB | 19.2 GB | 19.2 GB | 19.2 GB |
| 学習中の peak(PyTorch) | 28.6 GB | 21.0 GB | 16.5 GB | 16.5 GB |
| nvidia-smi peak(アイドル 0.8 GB 込み) | 32.0 GB | 32.0 GB | 32.0 GB | **24.1 GB** |
| 1 層の所要時間 | — | 488 s | 543 s | 484 s |

- 学生の構築で、使わない expert の器(fp32 10 GB)を GPU に載せてから差し替えていた → `build_layer(drop_experts=True)`
- 層の終わりの `export()` が latent 全体を一度に fp32 で量子化していた → 32 expert ずつに分割
- Adam の m / √v を int8 / uint8(128 要素ごとの fp32 scale)で保持(`--adam-bits 8`)。
  scale を fp16 にすると勾配 ~1e-7 で underflow し、v だけが 0 になって発散した(delta 1.12)
- 残る差(PyTorch の allocated に対して reserved が大きい)は caching allocator の断片化で、
  `--max-vram` の上限でキャッシュを解放させて抑える。`expandable_segments` は WSL で
  `CUDA driver error: device not ready` になり使えない

## 13. 難しい層だけ IQ3 に戻す(再学習なし)

44 層版の npz から、指定した層だけ ternary を外して組み直した(layer 0–3 は IQ3 のまま)。
VRAM と速度は layer 0–3 の expert を CPU に置いた構成(`-ot`)で測った。

| 構成(IQ3 に戻した層) | PPL | mean KL | KL 99 % | same top-1 | VRAM peak | tg128 |
|---|---:|---:|---:|---:|---:|---:|
| 44 層版(なし) | 3.168 | 0.488 | 5.26 | 79.9 % | 28.6 GB | 76 t/s |
| A(29–32) | 3.042 | 0.437 | 4.77 | 80.7 % | 31.5 GB | 56 t/s |
| B(15–16) | 3.122 | 0.472 | — | 80.2 % | 30.8 GB | 59 t/s |
| C(15–16 + 29–32) | 3.006 | 0.424 | — | 81.0 % | 31.9 GB | 41 t/s |

誤差が急増していた 6 層を戻しても KL は 13 % しか下がらず、VRAM は上限に張り付いて速度が落ちる。
少数の難しい層が後段を壊しているのではなく、層をまたいだ累積が主因。
→ 数層だけの mixed precision は費用対効果が合わない。

## 14. 2 層 block reconstruction(block 4–5 での診断)

`recon/q4x_block2.py`:layer L と L+1 を同時に学習し、2 層通過後の出力を主 loss にする
(+ 0.1 × layer L 出口、+ 0.2 × canonical anchor、router 項なし)。VRAM を 24 GB に収めるため
expert latent は fp16(確率的丸めで更新、layer 4 の 1 層学習で 0.1126 vs fp32 0.1125 と同等)。

| block 4–5(layer 5 出口 stream relMSE、1 層版 0.0193) | 結果 |
|---|---:|
| PTQ | 0.0332 |
| 2 層同時、PTQ から、lr 1e-4 | 0.0277(不安定) |
| 2 層同時、PTQ から、lr 5e-5 | 0.0224(300 step でまだ低下中) |
| 1 層版から warm start、scale 固定 | 0.0193(値が一切変わらない) |
| 1 層版から warm start、scale lr 1e-5 | 0.0204 |
| 1 層版から warm start、scale lr 1e-4 | 0.0384 |

- 2 層同時学習は、この block では 1 層版に勝てない。1 層版の目標 A = teacher(学生入力) が
  すでに上流誤差を織り込んでおり、2 層化で得るのは「L が L+1 を見越せる」分だけ。
  その代わり勾配が 2 層分の STE を通ってノイズが増える。
- npz は code × scale しか持たないので、warm start の latent は code の中心から始まり、
  300 step では境界(0.5 scale)まで動けない。変化は scale の Adam 更新(ノイズ主導)だけで、悪化する。
- 副産物:scale の学習率(1e-4)は、1 層版でもノイズ源になっている可能性がある。

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
