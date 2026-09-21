# Phase 1: Bonsai 2 / Qwen3.8-27B ternary 実装の解析

対象リビジョン:
- `Bonsai-demo` — `fbffef6` (branch `kv-e8-presets` から `flash-next-ternary-analysis` を作成)
- `llama.cpp/` (PrismML-Eng fork) — `3ce81ff`, branch `e8-kv`
- モデル — `models/bonsai2-gguf/27B/Ternary-Bonsai-2-27B-PQ2_0.gguf` (6.87 GiB)

表記:
- **[確認]** … ソースを読んだ / 実データで検証した事実
- **[推測]** … 根拠はあるが本リポジトリからは確認できない推論

---

## 0. 結論(先に重要な事実)

**[確認] このリポジトリには ternary モデルを *生成* する処理は存在しません。**

存在するのは以下の2つだけです。

1. **推論 runtime** — PrismML fork の llama.cpp (ggml 型 `PQ2_0`/`PTQ1_0` の kernel、
   activation 側 Hadamard 変換、GGUF metadata contract)
2. **packing / conversion** — *すでに ternary 化・すでに Hadamard fold 済み* の
   checkpoint を GGUF の `PQ2_0` / `PTQ1_0` レイアウトへ詰め直す処理

欠けているのは「BF16 の dense weight から {-1,0,+1} を決める処理」、すなわち
threshold / scale の決定、Hadamard fold の実行、および(おそらく)その学習です。

根拠:

- `ggml/src/ggml-quants.c:2206` `quantize_row_ptq1_0_ref()` と
  `ggml/src/ggml-quants.c:114` `quantize_row_pq2_0_ref()` は
  **`d = amax(group)` + `round(x/d)` の RTN しか行わない**。
  これは「入力がすでに group-128 の ternary である」場合にのみ可逆で、
  `ggml-common.h:217-221` のコメントも
  *"lossless for checkpoints that are already ternary at group 128"* と明記している。
- `quantize_ptq1_0()` は `quant_weights`(imatrix)を明示的に捨てている
  (`ggml-quants.c:2289`: *"ternary codes come from the weights themselves; an imatrix has no role"*)。
- `llama.cpp/conversion/base.py:635` `add_hadamard_metadata()` は
  `dir_model/hadamard_packing.json` という **外部で生成された manifest を読むだけ**で、
  回転そのものは一切行わない。manifest の `status` は
  `"requires-matching-runtime"` 固定 (`base.py:648`)。
- `scripts/`, `tests/` にも ternary 生成コードは無い(grep 済み)。

したがって以降で我々が作るものは **Bonsai そのものではなく
"Bonsai-inspired ternary quantization"** として扱う必要があります。

---

## 1. ternary weight のフォーマット

**[確認]** `w_i = s_g * t_i`、`t_i ∈ {-1,0,+1}`、`s_g` は fp16、group `g` は
入力軸(row 方向、GGUF の `ne[0]`)に沿った **128 要素**。

実データによる検証(`models/bonsai2-gguf/27B/...PQ2_0.gguf`):

| tensor | blocks | code hist (-1 / 0 / +1 / +2) | zero率 | scale min/med/max |
|---|---|---|---|---|
| `blk.0.ffn_gate.weight` | 696 320 | 29.93M / 29.21M / 29.98M / **0** | 0.3277 | 0.00727 / 0.01111 / 0.02904 |
| `blk.0.attn_qkv.weight` | 409 600 | 17.63M / 17.19M / 17.61M / **0** | 0.3278 | 0.00405 / 0.01642 / 0.07812 |
| `output.weight` | 9 932 800 | 426.7M / 416.9M / 427.8M / **0** | 0.3279 | 0.00364 / 0.01470 / 0.02859 |
| `token_embd.weight` | 9 932 800 | 435.9M / 417.0M / 418.5M / **0** | 0.3280 | 1.10e-5 / 0.01391 / 0.02644 |

- `+2` コード(`PQ2_0` は 2bit なので `{-1,0,+1,+2}` を表現できる)が **一度も使われていない**
  → checkpoint は真に ternary。
- zero率が全 tensor で **0.3277〜0.3280** と極めて一様。
- **[確認]** scale は 128 ごとに独立(period 2/4/8/40 のいずれでも繰り返さないことを検証済み)。
  つまり Hadamard block(1024)と quant group(128)は別物で、1 block = 8 group。

whitepaper (`bonsai-2-27b-whitepaper.pdf` §2.1) も
*"one shared FP16 scale for every group of 128 weights"*、
実効 `log2(3) + 16/128 ≈ 1.71 bpw` と一致。

---

## 2. PTQ1_0 / PQ2_0 の実装

### 型 ID と block レイアウト

**[確認]** `ggml/include/ggml.h:435-436`, `ggml/src/ggml-common.h:209-228`

| 型 | id | group | block バイト列 | bpw |
|---|---|---|---|---|
| `PQ2_0` | 142 | 128 | `fp16 d` + `uint8 qs[32]` = 34 B | 2.125 |
| `PTQ1_0` | 143 | 128 | `uint8 qs[24]` + `uint8 qh[2]` + `fp16 d` = 28 B | 1.75 |
| (参考) upstream `Q2_0` | 42 | 64 | — | mainline 互換 |

`ftype`: `GGML_FTYPE_MOSTLY_PQ2_0 = 128`, `MOSTLY_PTQ1_0 = 129` (`ggml.h:486-487`)。
`gguf-py/gguf/constants.py:5454-5455, 5650-5651` に同じ値。

### PQ2_0 codec

`ggml/src/ggml-quants.c:114 quantize_row_pq2_0_ref`

```
d  = amax(x[0..127])            (fp32 で計算し fp16 で保存)
id = 1/d
q  = clamp(round(x*id) + 1, 0, 3)
qs[j/4] |= q << ((j%4)*2)       (little-endian, 2bit/要素)
```
コード意味: `00=-1, 01=0, 10=+1, 11=+2`。dequant は `(q-1)*d`。

### PTQ1_0 codec

`ggml/src/ggml-quants.c:2206 quantize_row_ptq1_0_ref`

upstream `TQ1_0` (type 34) と同じ **base-3 trit packing** で、1 byte に 5 trit
(`3^5 = 243 ≤ 256`)。`TQ1_0` が block 256 / `qs[48]` で 32→16 の 2 段構成なのに対し、
`PTQ1_0` は block 128 / `qs[24]` なので段が **32/16/8** に一般化されている
(`ggml-quants.c:2204 ptq1_0_stages`)。末尾 8 要素は `qh[2]` に 4 trit/byte
(最上位 trit へ寄せるため `q *= 3` が 1 回余分に入る)。

エンコードは `q = ((uint16)q * 256 + 242) / 243`(ceil 除算)、
デコードは `xi = ((q * pow3[n] & 0xFF) * 3) >> 8` — ここは upstream TQ1_0 と同一。

### 自前実装での検証

`tools/flash_next_ternary/bonsai_format.py` に上記2つを numpy で転記し、
**出荷 GGUF の実バイト列と突き合わせ済み**:

```
hadamard identities                       OK
blk.0.ffn_gate.weight   PQ2_0 repack byte-exact / PTQ1_0 lossless (2560 blocks)
blk.0.attn_qkv.weight   PQ2_0 repack byte-exact / PTQ1_0 lossless (2560 blocks)
blk.0.ssm_out.weight    PQ2_0 repack byte-exact / PTQ1_0 lossless (3072 blocks)
```

すなわち **dequant → 再 quant が byte 完全一致**。packing 側は完全に再現できています。

---

## 3. group size

**[確認]** quant group = **128**(`QK_PQ2_0` / `QK_PTQ1_0`、`ggml-common.h:210,222`)、
入力軸に沿う。Hadamard block = **1024**(§5 参照)。両者は独立。

---

## 4-6. Hadamard rotation の実装

### 数式

whitepaper §2.4 **[確認]**:

```
R = (1/sqrt(n)) H_n S ,  n = 1024
f(x) = W ( R x )
```

`H_n` は Walsh–Hadamard、`S` は固定 ±1 対角。GGUF に格納されている `W` は
**すでに fold 済み** の `W_folded`、runtime は毎 matmul の直前に活性へ `R` を掛けます。

### weight 側(= fold)

**[確認] fold を行うコードは本リポジトリに存在しません。**
`conversion/base.py` は manifest を読んで metadata を書くだけです。

数学的には runtime の式から一意に決まります(`bonsai_format.py::fold_weight` で実装・検証):

```
W_folded = W R^{-1} = W S^{-1} H^{-1} = W S H     (H は対称直交、S は ±1)
```

**順序が重要**: 先に列へ sign を掛け、その後 block 1024 ごとに H を右から掛ける。
(逆順にすると恒等式が壊れます — 実際に selftest で検出しました。)

### 回転行列の構成

**[確認]** `src/llama-model.cpp:2019-2035`

```c
scale = 1/sqrtf(block_size);
parity = popcount(row & col) & 1;
data[row*block_size + col] = parity ? -scale : +scale;
```

→ **normalized Sylvester-Walsh**、対称かつ直交(`H = H^T = H^{-1}`)。
`scipy.linalg.hadamard` の行/列順序とは *たまたま* 一致しますが、
正規化と sign vector を含めた `R` 全体は一致しないので、そのまま使ってはいけません。

### inference 時の activation 変換

**[確認]** `src/llama-graph.cpp:1546-1578` (`build_lora_mm`) および
`1600-1640` (`build_lora_mm_id`、MoE 用)。適用順は:

1. (GDN の `ssm_out` のみ) feature 軸を tiled `[hd, nk, rep]` → grouped `[hd, rep, nk]` へ permute
   (`prism.hadamard.gdn_v_grouped`、`llama-graph.h:23-33` の `perm_hd/perm_nk/perm_rep`)
2. `ggml_mul(cur, t.signs)` — sign vector は入力幅ぶんの 1-D F32、列方向にブロードキャスト
3. `llama_mul_mat_hadamard(ctx, cur, t.rot)` (`src/llama-impl.h:57-75`)
   — `[block, rest]` に reshape して `ggml_mul_mat(rot, res)`、
   `ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD)` を付与
4. `ggml_mul_mat(w, cur_mm)` 本体

同じ活性に複数の fold 済み weight がぶら下がる場合、`hadamard_memo`
(`llama-graph.h:1133`)で変換を 1 回に共有します。

`token_embd.weight` だけは **inverse-after-lookup**: lookup 後に `R` を掛けて
回転基底へ入れ直します(`llama-graph.cpp:2428-2432`)。

---

## 7. scale の保存方法

**[確認]** block ごとの fp16 を block 構造体内に格納。
`PQ2_0` は先頭 2 バイト、`PTQ1_0` は末尾 2 バイト。別 tensor には持ちません。
`d = amax(group)` なので ternary checkpoint では `d` がそのまま `s_g`。

---

## 8. {-1,0,+1} の packing 方法

§2 参照。`PQ2_0` = 2bit/要素の素直な詰め方、`PTQ1_0` = base-3 で 5 trit/byte。
どちらも scale は group 128 共有。`PTQ1_0` ↔ `PQ2_0` は
**同じ trit 列の別表現で相互に無損失**(selftest で確認)。

---

## 9. CUDA kernel

**[確認]** すべて `ggml/src/ggml-cuda/` 配下:

| 役割 | ファイル / シンボル |
|---|---|
| FWHT(活性側回転) | `fwht.cu` / `fwht.cuh` — `ggml_cuda_op_fwht`, `ggml_cuda_op_fwht_signed` |
| FWHT dispatch | `ggml-cuda.cu:1822` (hint 経由), `:3478-3498` (sign fusion パターン検出), `:5217-5220` (supports_op) |
| MMVQ dot | `vecdotq.cuh:809/895 vec_dot_ptq1_0_q8_1(_multi)`, `:955 vec_dot_pq2_0_q8_1` |
| MMQ instance | `template-instances/mmq-instance-ptq1_0.cu`, `mmq-instance-pq2_0.cu` |
| MMQ tile load | `mmq-load-tiles.cuh`, `mmq.cuh`, `mmq.cu` |
| Hopper 専用 | `mmq-hopper-q1.cu` |
| dequant / getrows | `convert.cu`, `dequantize.cuh`, `getrows.cu` |

`fwht.cu` には 3 実装(warp `fwht_cuda`、shared-mem `fwht_cuda_smem`、
block 並列 `fwht_cuda_block`)があり、いずれも `has_signs` テンプレートで
**sign flip を load 経路に融合**しています(whitepaper 付録 A.2 の記述と一致)。
sign は `signs + (r % n_blk) * N` — 1024 要素ブロックごとに専用の sign が割り当たります。

CPU 側: `ggml-cpu/quants.c:285/335` の generic、`arch/x86/quants.c:561` (AVX-VNNI)、
`arch/arm/quants.c:297` (NEON)。Metal: `ggml-metal/kernels/{mul_mv,mul_mm,quantize,dequantize}.metal`。
Vulkan は `dequant_ptq1_0.comp` のみで MMQ 無し(MODEL-FORMATS.md の表と一致)。

---

## 10. GGUF tensor type / metadata との対応

**[確認]** 実ファイルから読み出した metadata:

```
general.architecture          = qwen35
qwen35.block_count            = 64
prism.hadamard.version        = 1
prism.hadamard.block_size     = 1024
prism.hadamard.transform      = normalized-sylvester-walsh-hadamard
prism.hadamard.axis           = input-last-dimension
prism.hadamard.sign_mode      = explicit
prism.hadamard.sign_widths    = [5120, 6144, 17408]
prism.hadamard.sign_values    = 28672 個の ±1
prism.hadamard.weight_names   = 401 個
prism.hadamard.inverse_weight_names = ['token_embd.weight']
prism.hadamard.gdn_v_grouped  = True
```

sign vector は **入力幅ごとに 1 本**(5120 / 6144 / 17408)で、層ごとには持ちません。

fold 対象 401 tensor の内訳:

```
ffn_down/gate/up      64 each   attn_qkv / attn_gate / ssm_out   48 each
attn_q/k/v/output     16 each   output.weight                     1
```

**2-D の `PQ2_0` tensor は `token_embd.weight` 以外すべて fold 済み**
(token_embd は inverse-after-lookup 側)。

tensor 型の分布(27B PQ2_0 ファイル):

| tensor | 型 | 本数 |
|---|---|---|
| `attn_q/k/v/output`, `attn_qkv`, `attn_gate`, `ssm_out`, `ffn_{gate,up,down}`, `output`, `token_embd` | `PQ2_0` | 402 |
| `ssm_alpha.weight`, `ssm_beta.weight` | `BF16` | 96 |
| 各種 norm, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias` | `F32` | 多数 |

whitepaper Table 2(高精度で保持する tensor 一覧)と一致します。

loader 側の検証: `src/llama-model-loader.cpp` 経由で `src/llama-model.cpp:1196-1335`。
`transform` 名、`axis`、`sign_mode`、block size の power-of-two 性、
さらに **「その weight が本当に Hadamard 対応 matmul 経路に乗っているか」** まで
検査して、条件を満たさなければロードを拒否します。
書き出し側にも同じガードが `conversion/base.py:687-741` にあります
(許可 arch は `LLAMA, QWEN3, QWEN3MOE, QWEN35, QWEN35MOE, QWEN3NEXT`)。

---

## 11. upstream llama.cpp との差分

**[確認]** fork 固有の主な追加:

1. **新 ggml 型** `PQ2_0`(142)/`PTQ1_0`(143) — upstream の `GGML_TYPE_COUNT` 超なので
   mainline は安全に拒否する。`Q2_0`(42) は upstream にも存在するため
   **安全に失敗せず gibberish を出す**(`MODEL-FORMATS.md`、`AGENTS.md:36`)。
2. **`GGML_HINT_SRC0_IS_HADAMARD`** (`ggml/include/ggml.h:453`) と
   `ggml_mul_mat_set_hint` — mul_mat に「src0 は Hadamard 行列」というヒントを付け、
   backend が FWHT に差し替えられるようにする仕組み。
3. **FWHT op**(`ggml-cuda/fwht.cu`, Metal `misc.metal`)と sign 融合。
4. **`prism.hadamard.*` GGUF contract** と loader/converter 両側の検証。
5. **rotation / sign tensor の動的生成** — GGUF には入っておらず、
   ロード時に `llama-model.cpp` が block_size × block_size の F32 として合成する。
6. E8 lattice KV cache 型 `q4_0_e8`(id 未記載)/ `q2_e8`
   (`ggml-common.h:237-252`、本ブランチ `e8-kv` の作業)。これは KV 側で ternary とは別系統。

なお本 fork の型 id 42 の意味は prism-v7 で変更されています(group128 → group64)。
旧ファイルは `prism` (= prism-v5) ブランチでしか読めません。

---

## 12. Qwen3.8-27B 固有の処理

**[確認]** arch 名は GGUF 上 `qwen35`(= Qwen3.5 系の hybrid attention)。
64 層のうち 48 層が linear attention (Gated DeltaNet)、16 層が full attention。

固有点:

- `ssm_out.weight` の fold は **grouped V 順**で計算されているため、
  runtime は活性を tiled → grouped に permute してから回転する
  (`prism.hadamard.gdn_v_grouped`、`llama-graph.cpp:1561-1567`、
  head 幾何は `hparams.ssm_dt_rank` / `ssm_n_group` から導出 `llama-model.cpp:2090-2099`)。
- 再帰状態パス(`ssm_alpha`/`ssm_beta` = in_proj_a/b、`conv1d`、`A_log`、`dt_bias`、`ssm_norm`)は
  **ternary 化されず BF16/F32 のまま**。whitepaper Table 2 で 26.2M param / 0.0976%。
- vision tower は別 GGUF(`...mmproj-Q8_0.gguf`、HQQ 4bit を Q8_0 コンテナに格納)。

---

## 13-14. conversion / quantization コードの所在

**[確認]**

| 処理 | 所在 | 状態 |
|---|---|---|
| HF safetensors → GGUF (tensor 名 mapping、GDN の qkvz 分解等) | `llama.cpp/conversion/*.py`, `convert_hf_to_gguf.py` | **あり** |
| Hadamard metadata の GGUF 書き出し | `conversion/base.py:635 add_hadamard_metadata` | **あり(manifest を転記するだけ)** |
| f32 → PQ2_0 / PTQ1_0 packing | `ggml-quants.c`, `src/llama-quant.cpp`, `tools/quantize/` | **あり(RTN のみ)** |
| **BF16 → Hadamard fold** | — | **無し** |
| **dense → {-1,0,+1} の threshold / scale 最適化** | — | **無し** |
| **ternary QAT / 蒸留** | — | **無し** |

`hadamard_packing.json` の schema(base.py が受理する形)は判明しています:

```jsonc
{
  "schema_version": 1 | 2,
  "kind": "hadamard-weight-fold",
  "status": "requires-matching-runtime",
  "transform": { "name": "normalized-signed-sylvester-walsh-hadamard",
                 "block_size": 1024, "sign_mode": "identity" | "explicit" },
  "signs": { "5120": [±1 × 5120], ... },          // sign_mode == "explicit" のとき
  "tensors": [ { "name": "<HF tensor name>", "axis": -1,
                 "role": "fold-before-matmul" | "inverse-after-lookup" }, ... ]
}
```

→ **我々の側で manifest を生成すれば、fork の converter と runtime はそのまま使えます。**
足りないのは fold 処理そのものと ternary 化アルゴリズムだけです。

---

## 15. 再現に必要な処理(未知の部分の切り分け)

```
BF16/FP16 W
  ↓ ① sign vector S の選択            ← [不明] 生成規則不明(乱数 seed か学習か)
  ↓ ② W_folded = (W * s) @ H_1024     ← [確認] 数式は runtime から一意に決まる
  ↓ ③ group-128 の scale s_g 決定      ← [不明] amax でないことは確実
  ↓ ④ {-1,0,+1} 量子化 (threshold)     ← [不明] zero率 ~0.328 になる規則
  ↓ ⑤ packing (PQ2_0 / PTQ1_0)        ← [確認] byte 完全再現済み
```

### ③④ について分かっていること / 分からないこと

**[確認]** `d = amax(group)` の RTN **ではない**。
RTN(`d=amax`, threshold `d/2`)を正規分布に掛けると zero率は ~85% になりますが、
実測は 32.8% です。したがって Prism 側の scale は amax より **かなり小さい**
値が選ばれ、packer 側の `d = amax` は「結果として ternary 値の scale を拾い直している」
だけです。

**[確認]** scale は fp16 に丸められた後、packing 時に再現される(byte 一致で検証済み)。

**[推測]** zero率が 4 tensor すべてで 0.3277–0.3280 に集中しているのは、
tensor 統計に依存する rule(例: `0.7 * mean|W|`)ではなく、
**回転後の分布がほぼ Gaussian である前提の固定 threshold** が使われている可能性を示します。
Gaussian 仮定なら zero率 0.328 は `threshold ≈ 0.4235 σ`、
すなわち `s ≈ 0.847 σ_group` に相当します。ただし **これは検証できません**
— packed ファイルからは `σ` と `s` の比が構成上 1.22 に固定されてしまい、
元の `W` 無しには threshold 規則を同定できないためです。

**[推測]** 98.2% のベンチマーク保持率は、PTQ 単独では説明しにくい水準です。
Prism 側で ternary-aware な学習(QAT / 蒸留)が行われていると考えるのが自然ですが、
whitepaper にも本リポジトリにも記述はありません。
→ **我々が PTQ で作るものは、性能面で Bonsai 2 と同等にはならない前提で進めるべきです。**

**[確認]** sign vector は GGUF に生値で入っているので、
**Bonsai 2 27B が使っている S は取り出せます**(幅 5120/6144/17408 の 3 本)。
ただし Flash-Next の幅(2560/6144/10240/640 など)とは一致しないため、
そのままは流用できません。生成規則が不明なので、我々は自前の seed で作る必要があります
(runtime は manifest 経由で任意の ±1 列を受け付けるため、これは問題になりません)。

---

## 16. 参照ファイル一覧(索引)

```
ggml/include/ggml.h:435                     型 ID 定義
ggml/src/ggml-common.h:209-228              block_pq2_0 / block_ptq1_0
ggml/src/ggml-quants.c:114                  quantize_row_pq2_0_ref
ggml/src/ggml-quants.c:495                  dequantize_row_pq2_0
ggml/src/ggml-quants.c:2204                 ptq1_0_stages
ggml/src/ggml-quants.c:2206                 quantize_row_ptq1_0_ref
ggml/src/ggml-quants.c:2256                 dequantize_row_ptq1_0
ggml/src/ggml-quants.c:2288                 quantize_ptq1_0 (imatrix を破棄)
ggml/src/ggml.c:686-707                     type_traits 登録
ggml/src/ggml-cuda/fwht.cu                  FWHT 3 実装 + sign 融合
ggml/src/ggml-cuda/ggml-cuda.cu:1822,3478   FWHT dispatch / sign fusion 検出
ggml/src/ggml-cuda/vecdotq.cuh:809,895,955  vec_dot
src/llama-impl.h:57                         llama_mul_mat_hadamard
src/llama-graph.h:23                        llama_hadamard_transform
src/llama-graph.cpp:1546,1600,2428          活性側変換の適用点
src/llama-model.cpp:1196-1335               prism.hadamard.* metadata 検証
src/llama-model.cpp:1951-2106               rotation / sign tensor の合成
src/llama-quant.cpp:396,503,824             ftype ↔ ggml type
conversion/base.py:620-755                  manifest 読み込み + metadata 書き出し
conversion/qwen.py:550,622                  GDN の fold 済み latent 取り扱い
tests/test-ptq1_0-element-map.cpp           trit 配置のテスト
tests/test-ptq1_0-cuda-dot.cpp              CUDA dot のテスト
```

## 17. 本 Phase で作成した検証コード

`tools/flash_next_ternary/bonsai_format.py`
— PQ2_0 / PTQ1_0 codec と `R` / fold / unfold の numpy 転記。
`python tools/flash_next_ternary/bonsai_format.py` で出荷 GGUF に対する
byte 一致 selftest が走ります。既存の推論環境には一切触れません。
