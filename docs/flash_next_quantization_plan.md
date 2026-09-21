# Phase 2: Qwen3.8-Flash-Next の構造確認と量子化方針(変換は未実施)

前提資料: [`docs/ternary_analysis.md`](ternary_analysis.md)

情報源:
- `/data/models/qwen3.8-flash-next/hf-cache/.../UD-IQ3_XXS/*.gguf`(手元にある実ファイル、3 split)
- HuggingFace `Qwen/Qwen3.8-Flash-Next` の `model.safetensors.index.json`(BF16 原本、131 shard)
- `~/AI/LLM/Qwen3.8-Flash-Next/qwen_startup.log`(upstream llama.cpp `cc231cb0d` によるロードログ)

表記は Phase 1 と同じ(**[確認]** / **[推測]**)。

---

## 0. まず判明した2つの重大な前提

### 0-1. PrismML fork は Flash-Next を読めません

**[確認]** Flash-Next の GGUF arch は **`qwen4exp`**(`general.description = "A Preview of the
Qwen4 Architecture"`)。この arch は **upstream ggml-org/llama.cpp にはあるが、
本リポジトリの PrismML fork (`e8-kv`, `3ce81ff`) には存在しません**
(`src/llama-arch.cpp` の arch 名一覧に `qwen4exp` 無し。grep で 0 件)。

`conversion/base.py:687` の Hadamard 許可 arch も
`LLAMA / QWEN3 / QWEN3MOE / QWEN35 / QWEN35MOE / QWEN3NEXT` のみで `QWEN4EXP` は含まれません。

→ ternary 重みを作れたとしても、**走らせる runtime が現状ありません**。
   最終的には「upstream の `qwen4exp` 実装を fork へ port する」か
   「Prism の Hadamard + ternary を upstream へ port する」かの
   どちらかが必須です。これは Phase 9(runtime 統合)の本体であり、
   Phase 3〜5 の検証は numpy/PyTorch 側で完結させます。

### 0-2. Hadamard block 1024 は Flash-Next では使えません

**[確認]** runtime は `weight->ne[0] % block_size != 0` を拒否し
(`src/llama-model.cpp:1978-1982`)、`block_size` は 2 冪でなければならず
(`:1214`)、しかも **GGUF 全体で 1 つの値**(`prism.hadamard.block_size`)です。

Flash-Next の入力次元と 2 冪約数:

| 入力次元 | 該当 tensor | 最大 2 冪約数 |
|---|---|---|
| 2560 | attn_qkv/gate/q/k/v, ffn_{gate,up}_exps, shexp, router, indexer, token_embd, output | **512** |
| 6144 | attn_output, ssm_out | 2048 |
| 10240 | hc_attn_down, hc_ffn_down | 2048 |
| **640** | **ffn_down_exps**, ffn_down_shexp | **128** |
| 320 | hc_*_up | 64 |
| 160 | per_layer_token_embd | 32 |

→ routed expert を全部(gate/up/down)覆う単一 block は **128** しかありません。
   Bonsai 2 27B の 1024 とは異なるため、**回転の強さ(outlier 平滑化効果)は弱くなります**。

選択肢:
- **(a) block 128 で統一** — 今の GGUF contract のまま通る。回転は弱い。
- **(b) `ffn_down` だけ block 128、他は 512** — fork の metadata schema を
  「tensor ごとの block_size」に拡張する改造が必要。
- **(c) `ffn_down_exps` は回転せず ternary 化、`gate/up` のみ block 512 で回転** — schema 変更不要
  (`weight_names` に入れなければ回転されない)だが、down 側の精度が落ちる。

**Phase 3 では (a) block 128 を既定とし、(b)(c) 相当として block 512 も併せて測定します。**
「まず既存 Bonsai 2 と同一値(1024)を使う」という指示は、2560 と 640 で数学的に
成立しないため、この点だけは前提を修正しました。

---

## 1. モデル諸元(実測)

**[確認]** GGUF metadata より:

```
arch                           qwen4exp        block_count             48
embedding_length               2560            context_length          262144
attention.head_count           24              head_count_kv            2
attention.key_length/value_len 256 / 256       rope.dimension_count    64
expert_count                   512             expert_used_count       10
expert_feed_forward_length     640             expert_shared_ffn_len   640
ssm.conv_kernel 4  ssm.state_size 128  ssm.group_count 16  ssm.time_step_rank 48
ssm.inner_size                 6144            full_attention_interval  4
hyper_connection.count         4               hyper_connection.low_rank 320
attention.indexer.head_count   4               indexer.key_length      128
attention.indexer.top_k        2048            attention.compress_ratios [0,0,0,4]×12
ple.layers [1]  ple.ngram_size 3  ple.heads_per_ngram 8  ple.conv_kernel 4
embedding_length_per_layer_input 160
総パラメータ                     176.944B
```

- 48 層のうち **12 層が full attention (QSA)**、**36 層が Gated DeltaNet**
  (`full_attention_interval = 4`、`compress_ratios` が 4 層ごとに 4)。
- PLE(n-gram embedding)は **layer 1 のみ**、128 shard × 16 head。

---

## 2. パラメータ予算

**[確認]** 手元の GGUF から集計:

| グループ | params | 比率 |
|---|---:|---:|
| routed MoE experts (`ffn_{gate,up,down}_exps`) | 120.796 B | 68.27 % |
| n-gram PLE embedding (`per_layer_token_embd`) | 51.200 B | 28.94 % |
| GDN 大 Linear (`attn_qkv`, `attn_gate`, `ssm_out` ×36) | 2.076 B | 1.17 % |
| embed + lm_head (`token_embd`, `output`) | 1.271 B | 0.72 % |
| hyper-connection (`hc_*_up/down` ×48 + output) | 0.636 B | 0.36 % |
| full-attn 大 Linear (`attn_q/k/v/output` ×12) | 0.598 B | 0.34 % |
| shared expert (`ffn_*_shexp`) | 0.236 B | 0.13 % |
| router (`ffn_gate_inp`, `ffn_gate_inp_shexp`) | 0.063 B | 0.04 % |
| PLE projections (`ple_key`, `ple_value`) | 0.033 B | 0.02 % |
| QSA indexer (`indexer.{q,k}_proj`) | 0.020 B | 0.01 % |
| norm / conv1d / bias / inject | 0.015 B | 0.01 % |
| **合計** | **176.944 B** | |

**重要**: routed experts だけで 68%。ここを ternary 化できれば全体の削減はほぼ達成され、
**FP8 化の寄与は数 GiB 止まり**です(§5 参照)。優先順位の判断材料になります。

---

## 3. tensor 分類表

`ne0` は入力次元(= 回転軸)。「128」「512」は使える最大 Hadamard block。

### ternary 候補(第一優先)

| GGUF tensor | HF 名 | 形状 | 本数 | ne0 | H block | 備考 |
|---|---|---|---|---|---|---|
| `blk.N.ffn_gate_exps.weight` | `mlp.experts.gate_up_proj` の前半 | [2560, 640, 512] | 48 | 2560 | 512 | **Phase 3 の主対象** |
| `blk.N.ffn_up_exps.weight` | 同 後半 | [2560, 640, 512] | 48 | 2560 | 512 | |
| `blk.N.ffn_down_exps.weight` | `mlp.experts.down_proj` | [640, 2560, 512] | 48 | 640 | **128** | block 制約の律速 |

`build_lora_mm_id`(`llama-graph.cpp:1600-1640`)が MoE 経路の活性回転に対応済みで、
`conversion/base.py:701-706` の `_HADAMARD_KINDS` も
`ffn_{gate,up,down}_exps` / `ffn_gate_up_exps` を許可しています。**[確認]**

### ternary 候補(第二優先 / 要検証)

| tensor | 形状 | 本数 | ne0 | 備考 |
|---|---|---|---|---|
| `blk.N.ffn_{gate,up,down}_shexp.weight` | [2560,640] / [640,2560] | 48 ×3 | 2560 / 640 | 全 token が通るので routed より感度が高い可能性。Bonsai 2 は shexp も ternary 化している(`_HADAMARD_KINDS` に含まれる)**[確認]** |
| `token_embd.weight`, `output.weight` | [2560, 248320] | 1+1 | 2560 | Bonsai 2 では両方 ternary。`token_embd` は inverse-after-lookup 扱い **[確認]** |

### FP8 候補(ternary が成立してから)

| tensor | 形状 | 本数 | 備考 |
|---|---|---|---|
| `blk.N.attn_qkv.weight` | [2560, 10240] | 36 | GDN in_proj_qkv。Bonsai 2 では ternary |
| `blk.N.attn_gate.weight` | [2560, 6144] | 36 | GDN z gate |
| `blk.N.ssm_out.weight` | [6144, 2560] | 36 | grouped-V permute が必要(§4) |
| `blk.N.attn_q/k/v/output.weight` | [2560,12288] 他 | 12 ×4 | full-attn |
| `blk.N.hc_{attn,ffn}_{up,down}.weight` | [10240,320] / [320,10240] | 48 ×4 | gated residual (hyper-connection) |
| `blk.N.ple_{key,value}.weight` | [2560,10240] / [2560,2560] | 1 ×2 | |

### BF16 維持

| tensor | 形状 | 理由 |
|---|---|---|
| `blk.N.ffn_gate_inp.weight` (router) | [2560, 512] | top-10/512 の選択に直結。0.063B しかなく削る価値が無い |
| `blk.N.ffn_gate_inp_shexp.weight` | [2560] | 同上 |
| `blk.N.indexer.{q,k}_proj.weight` | [2560,512] / [2560,128] | QSA の top-2048 選択に直結。BF16 のまま出荷されている **[確認]** |
| `blk.N.hc_{attn,ffn}_inject.weight`, `hc_*_norm` | [10240,4] / [10240] | 残差の混合係数。極小 |
| `blk.N.ssm_{alpha,beta}.weight`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `ssm_norm` | 小 | Bonsai 2 でも全て高精度保持(whitepaper Table 2)**[確認]** |
| 全 `*_norm.weight`, `attn_{q,k}_norm`, `indexer.{q,k}_norm` | 小 | |
| `blk.N.ple_{conv1d,norm_*}` | 小 | |

### CPU RAM offload

| tensor | 形状 | params | 備考 |
|---|---|---|---|
| `per_layer_token_embd.weight` | [160, 320001536] | 51.2 B | ne0=160 → 2 冪約数 32 しかなく **回転不可に近い**。lookup table なので ternary 化の意味も薄い。4bit 以下 + CPU 常駐が基本 |

### 対象外

| 項目 | 理由 |
|---|---|
| vision tower (`model.visual.*`, 27 block) | 別 mmproj GGUF。Bonsai 2 も別パッケージ |
| MTP (`mtp.*`: 1 層 + fc_embedding/fc_hidden) | 現 GGUF には含まれていない。speculative decoding 用途で後回し |

---

## 4. Flash-Next 固有で注意すべき点

1. **[確認] `ssm_out` の grouped-V 順序** — Bonsai 2 と同じ問題が起きます。
   Flash-Next は `linear_num_key_heads=16` / `linear_num_value_heads=48`(`ssm.group_count=16`,
   `ssm.time_step_rank=48`)なので rep = 3。fold は grouped 順で計算し、
   `prism.hadamard.gdn_v_grouped = true` を立てる必要があります
   (`llama-model.cpp:2090-2099`, `llama-graph.cpp:1561-1567`)。

2. **[確認] `gate_up_proj` の分割** — HF 側は `gate` と `up` が 1 tensor に結合されています。
   GGUF は `ffn_gate_exps` / `ffn_up_exps` に分離済み。
   **回転と group scale は分離後の tensor ごとに計算しなければなりません**
   (結合したまま回転すると gate/up が混ざります)。

3. **[推測] hyper-connection の扱い** — Bonsai 2 に該当物が無く、
   `_HADAMARD_KINDS` にも `hc_*` は含まれていません。
   4-way 残差の混合行列なので数値的に敏感な可能性があり、まず BF16/FP8 に留めるべきです。

4. **[確認] QSA indexer は既に BF16 で出荷** — unsloth の UD-IQ3_XXS でも
   `indexer.{q,k}_proj` だけ BF16。top-k 選択の感度が高いことの傍証です。

5. **[推測] expert が 512 個 × 6B active** — expert あたりの実効サンプル数が少なく、
   PTQ の calibration が routed expert では効きにくい可能性があります。
   Phase 4 では expert 選択頻度も記録すべきです。

---

## 5. サイズ / VRAM 試算

bpw 前提: `PTQ1_0` = 1.75、`PQ2_0` = 2.125、`FP8 + g128 fp16 scale` = 8.125、`BF16` = 16。

| グループ | params | A: experts のみ ternary | B: + 大Linear FP8 | C: + embed/head ternary | D: C だが PQ2_0 |
|---|---:|---:|---:|---:|---:|
| routed experts | 120.796B | 24.61 G | 24.61 G | 24.61 G | 29.88 G |
| shared expert | 0.236B | 0.44 G | 0.05 G | 0.05 G | 0.06 G |
| GDN 大 Linear | 2.076B | 3.87 G | 1.96 G | 1.96 G | 1.96 G |
| full-attn 大 Linear | 0.598B | 1.11 G | 0.57 G | 0.57 G | 0.57 G |
| embed + lm_head | 1.271B | 2.37 G | 1.20 G | 0.26 G | 0.31 G |
| hyper-connection | 0.636B | 1.18 G | 0.60 G | 0.60 G | 0.60 G |
| router / indexer / PLE proj / norm | 0.131B | 0.25 G | 0.22 G | 0.22 G | 0.22 G |
| **GPU 合計** | | **33.83 GiB** | **29.20 GiB** | **28.26 GiB** | **33.60 GiB** |
| RTX 5090 (実効 ~31.0 GiB) 残余 | | **−2.83** | **+1.80** | **+2.74** | **−2.60** |

CPU RAM 側(PLE n-gram embedding 51.2B):

| 形式 | サイズ |
|---|---|
| BF16 | 95.37 GiB |
| IQ4_NL / Q4 | 25.33 GiB |
| PQ2_0 | 12.67 GiB |
| PTQ1_0 | 10.43 GiB |

**読み取れること:**

- **PQ2_0 では 5090 に載りません**(構成 D)。PTQ1_0 が事実上の必須条件です。
  Bonsai 2 の実測では PTQ1_0 は Blackwell で PQ2_0 より遅い(whitepaper Table 5:
  5090 で 134.4 vs 142.5 tok/s)ので、**速度と搭載可能性のトレードオフ**になります。
- 最良の構成 C でも余剰は **~2.7 GiB**。200K context の KV cache と CUDA workspace には
  全く足りないので、KV は本リポジトリの `q4_0_e8` / `q2_e8`(`KV-CACHE.md`)のような
  低 bit KV と併用する前提になります。
- **FP8 化の寄与は A→B で 4.6 GiB** のうち大半が「shared expert と embed を低 bit にした分」で、
  attention/GDN の FP8 化自体は約 2.5 GiB。ternary experts が成立しなければ意味がありません。
  → 指示どおり **FP8 は ternary 成立後**で正しい順序です。

---

## 6. 次に行うこと(Phase 3 の範囲)

対象を **`model.language_model.layers.0.mlp.experts.gate_up_proj` の expert 0 の gate 側
(BF16, [2560, 640] 相当)** 1 本に絞り、

```
BF16 W → sign flip → blockwise Hadamard → group-128 ternary → dequantize
```

を実装し、weight 側指標(MSE / relative MSE / cosine / max error / zero率 / scale 分布)と
activation 経由の出力指標を測ります。scale 規則は Phase 1 §15 のとおり **不明**なので、
A(mean|W|)/ B(MSE 最小)/ C(threshold grid search)/ D(反復)を並べて比較します。

BF16 原本は `/data`(389 GB 空き)に置きます。全体 131 shard ではなく、
safetensors の header を読んで **必要な expert のスライスだけ HTTP range 取得**します。

**この時点ではモデル全体の変換は行いません。**
