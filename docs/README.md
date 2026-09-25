# Ternary quantization の調査記録(Bonsai 2 27B → Qwen3.8-Flash-Next)

作業ブランチ: `flash-next-ternary-analysis`(このリポジトリ)/ `qwen4exp-port`(`llama.cpp/`)。
既存の Bonsai 2 / Qwen3.8-27B 推論環境には一切変更を加えていない。
実験コードは `tools/flash_next_ternary/`、大きな中間生成物は `/data` と `/home/sohey/AI/LLM/` にある。

## 読む順番

| # | ドキュメント | 内容 | 一行での結論 |
|---|---|---|---|
| 1 | [ternary_analysis.md](ternary_analysis.md) | Bonsai 2 の形式・カーネル・Hadamard・GGUF contract の解析 | **このリポジトリに ternary 生成処理は無い**。あるのは推論 runtime と packer だけ |
| 2 | [flash_next_quantization_plan.md](flash_next_quantization_plan.md) | Flash-Next(`qwen4exp`)の tensor 分類と VRAM 設計 | experts が 68 %、PLE が 29 %。fork に `qwen4exp` が無い / H1024 は使えない |
| 3 | [phase3_ternary_experiment.md](phase3_ternary_experiment.md) | 単一 tensor での ternary 化と 4 方式比較 | 固定 Hadamard + weight-MSE の枠では rel_MSE 0.187 が最適点 |
| 4 | [bonsai_reverse_analysis.md](bonsai_reverse_analysis.md) | 出荷 Bonsai 2 と base Qwen3.8-27B の要素対応 | fold 規約は確定。code は単一 threshold で説明不能 = **学習由来** |
| 5 | [ptq_vs_shipped_27b.md](ptq_vs_shipped_27b.md) | BF16 / PTQ 2 種 / 出荷 Bonsai の 4 者比較 | PTQ は PPL 830、出荷は 8.6。差はほぼ全て学習 |
| 6 | [layer32_recon.md](layer32_recon.md) | 単層・4 層 block の reconstruction | 単層で relMSE 0.47 → 0.20。mixed で KL 半減 |
| 7 | [progressive_pilot.md](progressive_pilot.md) | 16 層 progressive(PTQ / A-only / dual) | 累積が飽和。KL は PTQ の 8.5 分の 1 |
| 8 | [progressive_full.md](progressive_full.md) | 全 64 層 progressive | **PPL 830 → 11.07、KL 5.10 → 0.74**(5.5 時間、33 万 token) |
| 9 | [head_calibration.md](head_calibration.md) | stream × head のクロス評価と較正 | 出荷の優位は stream と head の共適応。後付け較正では埋まらない |
| 10 | [suffix_pilot.md](suffix_pilot.md) | suffix-aware(最終 logits への KL)progressive | baseline より悪化。logits 目的関数はデータ律速 |
| 11 | [qwen4exp_port.md](qwen4exp_port.md) | Flash-Next runtime の移植と ternary expert | **移植完了・全 48 層 ternary が 5090 に載る(28.9 GiB)**。品質は未達 |
| 12 | [flash_next_progressive_pilot.md](flash_next_progressive_pilot.md) | Flash-Next の progressive reconstruction(8 → 16 → 44 層)と改善の探索 | 44 層 ternary PPL 3.17 / KL 0.49。2 層 block・追加学習対象・anchor・2 bit 化はいずれも小改善で探索終了。routing は強く偏り expert cache が有効 |
| 13 | [freetoken_expert_offload_survey.md](freetoken_expert_offload_survey.md) | FreeToken の expert cache を ternary Flash-Next に使えるか(Codex 調査) | FreeToken の LRU expert cache を PrismML fork へ移植する案を推奨。まず routing trace で hit 率を検証 |

## 到達点

### 27B(縮小模型として使い切った)

| モデル | wikitext PPL | KL vs BF16 |
|---|---:|---:|
| BF16 | 6.513 | — |
| 素の PTQ ternary | 829.6 | 5.10 |
| **progressive reconstruction** | **11.07** | **0.736** |
| 出荷 Bonsai 2 | 8.62 | 0.367 |

### Flash-Next(125 B + 51 B PLE)

| 項目 | 実測 |
|---|---|
| runtime | upstream `qwen4exp` を fork へ移植、累積 PPL 差 0.12 % |
| ternary expert | H128 + PTQ1_0 で decode 成立(arch ゲート 2 行の変更) |
| サイズ | 78 GiB → **57 GiB** |
| VRAM | 全 48 層で **28.9 GiB**(RTX 5090 に収まる) |
| 品質 | PPL 2.48 → 22.3(**超線形累積が主因**、量子化器の差ではない) |

## 残っている課題

1. Flash-Next の progressive reconstruction の強化(8 層 pilot は KL 31 % 減に留まる。データ増・routing 項が候補)
2. Hadamard block の tensor 単位化(`prism.hadamard.block_size` が GGUF 全体で単一値。
   gate/up を H512 にするには拡張が必要)
3. 出荷 Bonsai 2 との残り 2 倍の KL 差 — stream と head の同時学習が必要

## ツール

| ファイル | 用途 |
|---|---|
| `bonsai_format.py` | PQ2_0 / PTQ1_0 codec と Hadamard 回転(出荷 GGUF に対し byte 一致を自己検証) |
| `fetch_slice.py` | HF safetensors から必要な tensor だけ HTTP Range 取得 |
| `ternary_quant.py` | group-wise ternary 4 方式(absmean / MSE scale / threshold grid / 交互反復) |
| `gguf_inject_ternary.py` | 既存 GGUF の指定 tensor を fold+ternary+PTQ1_0 に置換、contract を付与 |
| `make_ptq_bonsai.py` | base checkpoint 全体を PTQ-Bonsai 化 |
| `dump_hidden/` | 層ごとの residual stream と logits を GGUF から採取 |
| `compare_hidden.py` | 2 つの dump の層別 cosine / rel-MSE と logit KL |
| `recon/` | reconstruction 一式(`ternary_layer` / `train_recon` / `train_block` / `progressive` / `head_calib` / `train_joint` / `export_mixed`) |
