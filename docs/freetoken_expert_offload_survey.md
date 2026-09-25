# FreeToken expert offload / cache の Qwen3.8-Flash-Next ternary 版への適用調査

調査日: 2026-09-25  
対象:

- FreeToken: `/home/sohey/AI/LLM/FreeToken`
- PrismML llama.cpp fork: `/home/sohey/AI/LLM/Bonsai-demo/llama.cpp` (`qwen4exp-port`)
- 現在の ternary モデル: layer 0-3 は IQ3_XXS、layer 4-47 は PTQ1_0

この文書はソースと既存ドキュメントの静的調査だけに基づく。ビルド、GPU 実行、モデルロード、ネットワークアクセスは行っていない。したがって、実測でない数値は明記した。

## 結論

推奨は **(b) FreeToken の expert cache の設計を PrismML llama.cpp fork に段階的に移植する** ことである。

理由は次の通り。

1. FreeToken は llama.cpp fork ではなく、Python/PyTorch/Triton と C++/CUDA extension からなる独立 runtime である。現在の GGUF、PTQ1_0、Hadamard 契約をそのまま利用できない。
2. 一方、FreeToken の現在のツリーには Qwen3.8-Flash-Next (`qwen4_exp`) の hyper-connection、GDN、QSA、PLE、shared/routed MoE が既に実装されている。したがって案 (a) の主な不足はモデル本体ではなく、PTQ1_0 の loader/bank/kernel と Hadamard 融合である。
3. PrismML 側には qwen4exp と PTQ1_0 + H128/sign の動作経路が既にあり、実測品質・速度もある。ここへ expert 単位 cache を足す方が、数値一致を壊す範囲を expert の保管・転送・ID remap に限定できる。
4. 32 GiB のうち 3.5 GiB を空ける仮定では、現在の 44 PTQ1_0 層のうち、32k は全 44 層、128k は安全側で約 38 層、256k は約 29 層相当を GPU cache に置ける見込みである。cache fraction はそれぞれ 100%、86%、66%。top-10/512 で十分かは routing trace による検証が必要だが、試す価値のある容量である。
5. 静的 `-ot` / `--n-cpu-moe` は比較基準と即時の退避策として有用だが、長い context ほど CPU を毎 token 通る層が増えるため、最終方針にはしない。

## 1. FreeToken は何か

### 1.1 runtime の構成

FreeToken は README が「edge-native Mixture-of-Experts serving engine」と呼ぶ独立 runtime であり、llama.cpp の fork ではない (`README.md:18-23`)。README が llama.cpp を inspiration の一つとして列挙しているだけである (`README.md:75-81`)。

主な構成は以下。

- Python: engine、scheduler、model graph、KV/cache 管理、weight loader
- PyTorch: tensor と CUDA graph の実行基盤
- Triton: attention、MoE、量子化 expert、copy kernel
- C++/CUDA extension: `python/freetoken/kernel/csrc/`; `setup.py` は `torch.utils.cpp_extension` を使う
- checkpoint: Hugging Face safetensors を直接読む。任意で FTW fast-load 形式へ変換する (`docs/models.md:3-4,53-54`)
- HTTP API: OpenAI / Anthropic / Responses 互換 server

NVIDIA GPU と CUDA/nvcc を前提にする (`docs/install.md:5-16`)。ggml graph や GGUF tensor を中心にした llama.cpp とは、loader、allocator、kernel 呼び出し、CUDA graph の境界が異なる。

### 1.2 対応モデルと量子化形式

公開リストには Qwen3.8-Flash-Next、Qwen3.6/3.5 MoE、GLM-5.x、DeepSeek-V4、Gemma-4、MiniMax、gpt-oss などがある (`docs/models.md:8-21`)。Qwen3.8-Flash-Next は FP8 と NVFP4 checkpoint が明示されている (`docs/models.md:12`)。

MoE cache が認識する bank schema は `python/freetoken/moe/offload_cache.py:32-78` にあり、現在は次を含む。

- BF16
- 128x128 block FP8
- GGUF Q4_0
- NVFP4 の native / Marlin / Blackwell 系 layout
- MXFP4
- DeepSeek 系 FP4

対応形式は cache 本体だけで完結せず、`python/freetoken/layers/quantization/moe/` の `MoEMethod`、`BankSpec`、expert GEMM と組になる (`base.py:67,164-177`)。**PTQ1_0 は schema、quant method、kernel のいずれにも存在しない**。FreeToken 内には DeepSeek-V4 indexer 用の Hadamard 実装があるが (`python/freetoken/kernel/triton/dsv4/hadamard.py:1-44`)、PTQ1_0 expert matmul 用の H128 + per-row sign/FWHT ではない。

## 2. FreeToken の expert cache / offload

### 2.1 実行方式

FreeToken の MoE 実行方針は以下 (`docs/models.md:37-49`)。

| 方式 | expert weight | decode |
|---|---|---|
| fused | GPU 常駐 | GPU |
| offload | pinned host + GPU cache | hit は GPU、miss を H2D 後 GPU |
| cpu | host bank | CPU GEMV |
| hybrid | host + GPU cache | hit/fetch 分を GPU、残りの miss を CPU |
| auto | VRAM と利用可能 backend から選択 | 上記を選択 |

入口は `python/freetoken/engine/engine.py:616` の `_init_offload_moe_cache`、cache 本体は `python/freetoken/moe/offload_cache.py:104` の `OffloadMoeCache`、layer 側は `python/freetoken/layers/moe.py:148` の `OffloadMoELayer` である。

### 2.2 cache の粒度と LRU

cache の最小単位は **ある layer の routed expert 1 個**である。その expert の gate/up/down と scale など、quant method が登録した全 bank の同じ row が一つの slot に対応する。

- ID は概念上 `layer_id * num_experts + expert_id` の global ID。
- `slot_for_id` は全 layer/expert から slot への表、`id_of_slot` は逆表、`usage` は LRU timestamp (`offload_cache.py:103-209`)。
- pool は layer ごとに分割せず、全 layer で共有する global LRU。
- `ensure_experts()` が選択 expert を確保し、routing ID を expert ID から slot ID へ in-place で書き換える (`offload_kernels.py:19-40`; `layers/moe.py:261-296`)。
- 現在の policy は LRU のみ。

同じ quant format の layer は bank の row shape/dtype が同一である必要がある (`OffloadMoeCache.set_bank_sources`, `offload_cache.py:292-365`)。したがって、IQ3_XXS 4 層と PTQ1_0 44 層を一つの pool に混ぜる設計にはできない。最初は layer 0-3 を従来どおり CPU/static にし、同形状の PTQ1_0 layer 4-47 だけを一 pool にするのが安全である。

### 2.3 CPU-GPU 転送

host 側は layer ごとの bank を pinned memory として保持し、CUDA から参照できる device pointer/UVA alias を作る (`offload_cache.py:292-430`)。miss の全 bank row は `copy_missing()` から `fast_index_copy_multi_jit` へ渡され、一回の計画でまとめて転送される (`offload_cache.py:1011-1053`)。

小さい row を個別 `cudaMemcpyBatchAsync` に渡すと遅いという前提がコードにあり、連続 expert を最大 256 KiB の run に coalesce する (`offload_cache.py:17-26`)。量子形式ごとに copy loop を作るのでなく、bank registry により転送機構を共通化している。

### 2.4 decode、prefill、hybrid

decode:

1. router が top-k expert ID を出す。
2. `OffloadMoELayer._decode_routed()` が `ensure_experts()` を呼ぶ。
3. hit は既存 slot、miss は LRU victim を割り当てる。
4. `copy_missing()` が全 bank row を H2D する。
5. expert GEMM は slot cache を読む (`layers/moe.py:261-296`)。

prefill は token ごとの router 選択だけを転送する方式ではなく、**layer 全体 512 experts を stream** する (`layers/moe.py:347-384`)。二つの full-layer buffer を借り、次 layer の H2D と現在 layer の GEMM を overlap できる (`offload_cache.py:606-701`)。そのため overlap 使用時の cache 下限は `2 * num_experts` slot (`engine/cache_budget.py:70-81`)。

さらに `prefill_hit_d2d` は既に resident な expert を cache から double buffer へ D2D gather し、miss だけを host から coalesced H2D する (`offload_cache.py:746-816`)。これは cache が `2 * num_experts` より大きい時だけ有効である。

hybrid decode は miss 全部を転送しない。`ensure_experts_hybrid()` が転送数を固定上限または比率で制限し、残りの route を CPU executor に渡す (`offload_kernels.py:43-55`; `offload_cache.py:855-874`)。CPU GEMV と PCIe fetch + GPU GEMM を重ね、最後に partial result を加算する (`layers/moe.py:298-345`)。fetch 対象は単なる expert ID 順でなく、再登場した miss を recency で優先する (`offload_kernels.py:128-188`)。engine は `ft bench bw` の PCIe/CPU bandwidth profile から overlap が釣り合う fetch fraction を決める (`engine/engine.py:744-773`)。

### 2.5 hit 率と VRAM 利用を改善する仕組み

- 全 layer 共有 LRU: layer 間で未使用枠を融通する。
- router ID を slot ID に GPU 上で remap: decode path を CUDA graph capture 可能に保つ。
- multi-bank fused copy と連続 row の coalescing。
- prefill double buffer と H2D/GEMM overlap。
- prefill hit の D2D 再利用。
- hybrid の recency 優先 fetch と CPU/GPU overlap。
- `--moe-cache-auto`: KV reserve を先に確保し、残りを MoE slot に割り当てる (`engine/cache_budget.py:31-95`)。
- runtime cache resize: host bank は保持したまま GPU slot pool を作り直す。ただし cold start になる (`offload_cache.py:465-534`)。
- 計測: miss rate、layer 別 miss、routing histogram、working set、90% を覆う expert 数、stationary oracle hit、entropy を集計できる (`offload_cache.py:928-1009`)。

この repository に synthetic copy benchmark はある (`benchmarks/bench_offload_cache_copy.py`) が、Qwen3.8 top-10/512 の routing trace に基づく hit 率や tok/s の保存結果は見つからなかった。従って、このモデルで cache がどれだけ効くかを示す既存実測値はない。

## 3. qwen4exp への適用可能性

### 3.1 model architecture

調査開始時の想定と異なり、現在の FreeToken は `python/freetoken/models/qwen4_exp/` を既に持つ。

- model flow: PLE -> hyper-connection -> GDN/QSA -> hyper-connection -> MoE (`model.py:58-127`)
- 36 GDN / 12 QSA の layer group (`config.py:172-193`)
- GDN recurrent + convolution state (`gdn.py:13-120`)
- QSA indexer と sparse attention (`attention.py:76-247`, `attention/qsa_sparse.py:1-27`)
- PLE の pinned-host UVA table と early prefetch (`ple.py:1-65`; `model.py:103-127`)
- shared expert gate + routed expert (`models/qwen4_exp/moe.py:13-29`; `models/qwen3_5_moe/moe.py:56-82`)
- hyper-connection stream 数 4、低 rank 320 も config/test に含む

shipping geometry の test は 48 layer、hidden 2560、512 experts、top-10、expert width 640、GDN 16 K heads / 48 V heads / dim 128 / conv 4、QSA index budget 2048 / ratio 4 を検証する (`tests/models/qwen4_exp/test_config.py:17-66,79-134`)。AOT model registry にも `Qwen4ExpForConditionalGeneration`, top-k 10, width 640, NVFP4/FP8 が登録済み (`kernel/aot_models.py:197-207`)。

したがって案 (a) で architecture をゼロから移植する必要はない。ただし FreeToken の official FP8/NVFP4 path と PrismML fork の GGUF qwen4exp が完全に同じ logits になることは、この静的調査では確認していない。PLE table も FreeToken docs/loader は official checkpoint の 47.7 GiB pinned tableを前提とする一方、現在の GGUF 実測は 27.5 GiB CPU mapped であり、入力 format と table layout の差を吸収する loader が必要である (`docs/models.md:62`; `Bonsai-demo/docs/qwen4exp_port.md:143,286`)。

### 3.2 PTQ1_0 と Hadamard

FreeToken は PTQ1_0 を現状のまま扱えない。PrismML 側では qwen4exp が `build_moe_ffn()` を呼び (`src/models/qwen4exp.cpp:991`)、gate/up/down が `build_lora_mm_id()` に到達する (`src/llama-graph.cpp:1988-2346`)。Hadamard metadata の検証・tensor 生成は `src/llama-model.cpp:1215-1348,1996-2113`、matmul hint は `src/llama-impl.h:71`、PTQ1_0 CUDA MMQ は `ggml/src/ggml-cuda/mmq.cu:21-22,375-432` にある。この一連に相当する機能が FreeToken にはない。必要なものは最低でも以下。

1. GGUF の PTQ1_0 packed trits、group-128 fp16 scale、`prism.hadamard.*` metadata/sign を読む loader、または FTW side-bank converter。
2. gate/up/down の packed row と scale/sign を表す `BankSpec` と per-expert byte 計算。
3. decode と prefill の expert kernel。PTQ1_0 を直接読む GEMV/GEMM と、各 matmul 直前の H128 + sign を fuse する。
4. CPU/hybrid を使うなら同じ contract の CPU executor。これがなければ最初は pure offload GPU path に限定する。
5. correctness test: PrismML と FreeToken で個別 expert 出力、layer 出力、最終 logits を比較する。

Hadamard は gate/up の入力 hidden 2560 を 128 要素 block ごとに、down の入力 640 も同様に回す必要がある。既存の DeepSeek-V4 用 dense `H` matmul (`kernel/triton/dsv4/hadamard.py`) は数値定義の参照にはなるが、PTQ1_0 の符号 metadata と fused expert kernel を代替しない。ggml の `GGML_HINT_SRC0_IS_HADAMARD` / `build_lora_mm_id` も PyTorch/Triton runtime から直接再利用できない。

回避策は以下。

- PTQ1_0 を NVFP4/FP8 に再量子化: FreeToken を早く動かせるが、現在の 525 MiB/layer という容量優位と既存の品質評価を失う。
- load 時に weight 側へ逆回転を畳み、既存 kernel で扱える形式へ展開: activation FWHT は消せるが、再量子化誤差と容量増が生じる。
- PTQ1_0 kernel を FreeToken に新設: 容量は維持できるが、最も大きな実装・検証項目になる。
- PrismML に cache だけ移植: 既存 PTQ1_0/FWHT path を保てるため、本調査の推奨。

## 4. top-10 / 512 experts で cache は効くか

batch 1 decode では、一 layer、一 token あたり最大 10 expert、44 PTQ layer では 440 route 参照になる。top-8 より working set と miss 転送量は大きく、512 experts は routing が一様なら cache に厳しい。

ただし今回の VRAM 予算は少数 slot ではない。後述の安全側案は layer 当たり平均で次の expert 数に相当する。

| context | slot 数 | PTQ 44 層に均した slots/layer | 全 512 に対する割合 |
|---:|---:|---:|---:|
| 32k | 22,528 | 512 | 100% |
| 128k | 19,456 | 442 | 86% |
| 256k | 14,848 | 337 | 66% |

一様独立 routing の定常近似では hit 率の基準線は cache fraction 程度であり、expert popularity の偏りや時間的 locality があれば上回る。逆に prompt/domain 切替や routing の広がりで下回り得る。global LRU は各 layer を順に一回ずつ通るため、単純な layer 内 LRU とは厳密には一致しない。

1 expert slot は約 1.025 MiB である。44 layer x top-10 が全 miss なら 1 token あたり最大約 451 MiB の H2D になり実用的でない。一様近似で 128k/86% hit なら約 63 MiB/token、256k/66% hit なら約 153 MiB/token。PCIe 4.0 x16 を 32 GB/s と仮定した転送時間の理想下限は約 2.0 ms / 4.8 ms だが、これは protocol、copy launch、競合を無視した推測である。実効 bandwidth と実 routing trace が判断を決める。

FreeToken の `decode_routing_stats()` (`offload_cache.py:979-1009`) がまさに必要であり、working set、90% coverage、oracle hit を実 workload で採るまでは「効く」と断定しない。

prefill については、単純な full-layer stream は 44 x 525 MiB = 23.1 GiB を一 prefill pass で読む。decode cache だけを移植しても long prompt の prefill は改善しない。第 2 段階で FreeToken の double buffering と hit-D2D/miss-H2D split を移植する価値がある。

## 5. 統合方針の比較

| 案 | 実装量 | 速度の見込み | 主な利点 | 主なリスク | 評価 |
|---|---|---|---|---|---|
| (a) FreeToken に qwen4exp + PTQ1_0 | 大。ただし qwen4exp 自体は既存で、主作業は GGUF/PTQ1 reader、bank、decode/prefill kernel、Hadamard/sign、必要なら CPU executor | expert cache、compressed QSA index slab、GDN state pool をそのまま利用できれば高い可能性。ただし PTQ kernel 性能は未知 | cache/auto budget/hybrid/prefill が完成済み | 二つ目の PTQ1 実装、数値一致、GGUF/PLE 差、CUDA/Triton kernel、CPU path。現在の PrismML 実測資産を直接使えない | 長期の別 runtime 対応としては有力。今回の最短路ではない |
| **(b) FreeToken の cache 設計を PrismML に移植** | **中-大**。host bank、slot tensor/view、LRU/remap、async copy/event、graph lifetime、prefill buffer が必要 | 32k は全 resident、128k は高 hit が期待できる。256k は routing skew 次第。PTQ1 kernel は現状を維持 | 検証済み qwen4exp/GGUF/PTQ1/H128 を保持。変更範囲を weight residency に限定 | ggml tensor は静的配置前提。dynamic slot と `build_moe_ffn`/MMID、CUDA graph、allocator/event の整合が難所 | **推奨**。decode-only -> prefill -> hybrid の順に限定して進める |
| (c) `-ot` / `--n-cpu-moe` の静的 offload | 極小。既存機能だけ | 現状 layer 0-3 CPU で 76 t/s。context が長くなるほど CPU layer を増やす必要があり、毎 token 固定で遅くなる | 即時利用、予測可能、correctness risk が低い | expert locality を使えない。525 MiB/layer 単位で粗い。CPU layer は hit しても GPU に戻らない | baseline と fallback。最終解にはしない |

PrismML の `-cmoe` は全 MoE expert を CPU、`-ncmoe N` は先頭 N layer の expert を CPU にする (`llama.cpp/common/arg.cpp:2765-2781`)。`-ot` は tensor regex 単位なので、現在の layer 0-3 IQ3 と layer 4-47 PTQ を明示的に分けるにはこちらが適する。

既存実測は重要な baseline である。全 GPU は VRAM 32.0 GB 上限に達して tg128 21 t/s、layer 0-3 expert を CPU にすると 28.6 GB、pp512 780 t/s、tg128 76 t/s (`flash_next_progressive_pilot.md:220-227`)。これは「GPU に置くほど速い」とは限らず、OS/driver の共有 memory spill を避ける余裕が最優先であることを示す。

## 6. VRAM 予算

### 6.1 前提と式

以下は single sequence、KV type F16、nominal 32 GiB、3.5 GiB を空けて使用上限 28.5 GiB とした机上見積もりである。実機の総 VRAM 表示、driver context、CUDA allocator fragmentation で変わる。

#### QSA main KV

full attention は 12 層、K/V とも 2 KV heads x 256、F16 とする (`flash_next_quantization_plan.md:65-73`; `qwen4exp.cpp:26-148`)。

```text
12 layers * (K + V) * 2 heads * 256 * 2 bytes
= 24 KiB/token
```

#### PrismML の QSA indexer cache

`llama-memory-hybrid-idx.cpp:49-64` は `hparams_idx` の KV head 数を 1、K head dim を 128 に変えるが、V head dim は元の 256 のままである。通常 KV cache は MLA でない限り K と V の両方を確保する (`llama-kv-cache.cpp:180-254`)。qwen4exp graph は indexer K だけを `cpy_k()` し、V は読んでいない (`qwen4exp.cpp:551-615`)。

したがって現コードの allocation は:

```text
12 layers * (K 128 + V 256) * 1 head * 2 bytes
= 9 KiB/token
```

これは「必要データ量」ではなく「現実装の確保量」である。K だけなら 3 KiB/token、さらに FreeToken のように ratio 4 の pooled key だけなら 0.75 KiB/token になる。FreeToken の式は `kvcache/base.py:18-35`、QSA compressed pool は `kvcache/qsa_pool.py:1-16` にある。

#### GDN recurrent state

context 長には比例しない。fork の `llama-memory-recurrent.cpp:20-138` と `llama-hparams.cpp:183-250` から、一 sequence について:

```text
conv history = (4 - 1) * (2 * 16 * 128 + 48 * 128)
             = 30,720 float
recurrent    = 48 * 128 * 128
             = 786,432 float
36 layers * (30,720 + 786,432) * 4 bytes
= 112.22 MiB
PLE state ~= 0.35 MiB
合計 ~= 112.57 MiB
```

これは既存ログの 112.6 MiB と一致する (`qwen4exp_port.md:85`)。multi-sequence では sequence/slot 数に比例する。

#### model weight と expert slot

all-PTQ1 の実測 CUDA model buffer 28,422 MiB から 48 x 525 MiB の routed experts を引くと、GPU に置く非 routed weight は約 3,222 MiB = 3.15 GiB と推定できる (`qwen4exp_port.md:269-282`)。

```text
PTQ1 expert 1 slot = 525 MiB / 512 = 1.025 MiB
PTQ1 layer-equivalent = 525 MiB = 0.513 GiB
```

table は layer 0-3 IQ3_XXS を CPU に固定し、layer 4-47 の 44 homogeneous PTQ1 layer だけを cache 対象とする。

#### compute buffer と runtime overhead

`-c 2048` の既存測定は KV + RS + compute 合計 512 MiB (`qwen4exp_port.md:276-282`)。上記 KV は約 66 MiB、RS は約 113 MiB なので、そこでの compute/temporary は差分約 333 MiB である。

long context では QSA の `score` / `expanded` / mask が `n_kv * n_ubatch` に比例する (`qwen4exp.cpp:551-687`)。allocator の lifetime reuse があるためソースだけから exact peak は決まらない。ここでは次の **推測レンジ**を置く。

- 32k: 0.35-0.50 GiB
- 128k: 0.55-0.85 GiB
- 256k: 0.85-1.35 GiB

さらに driver/CUDA allocator/未計上 tensor として 0.4 GiB を仮置きした。これは実測 28.6 GB と component sum の差を丸めたもので、必ず実ログで置換すべき値である。

### 6.2 context 別見積もり

| context | main KV 24 KiB/tok | indexer allocation 9 KiB/tok | GDN+PLE state | compute 推測 | expert 以外の合計\* | 28.5 GiB 内の PTQ cache | 推奨する安全側常駐量 | cache fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32k | 0.75 GiB | 0.281 GiB | 0.110 GiB | 0.35-0.50 GiB | 5.04-5.19 GiB | 23.31 GiB 以上 | **44/44 層 = 22.56 GiB** | 100% |
| 128k | 3.00 GiB | 1.125 GiB | 0.110 GiB | 0.55-0.85 GiB | 8.33-8.63 GiB | 19.87-20.17 GiB | **38/44 層 = 19.48 GiB** | 86% |
| 256k | 6.00 GiB | 2.250 GiB | 0.110 GiB | 0.85-1.35 GiB | 12.76-13.26 GiB | 15.24-15.74 GiB | **29/44 層 = 14.87 GiB** | 66% |

\* 非 routed weight 3.15 GiB + KV/indexer + state + compute + runtime overhead 仮置き 0.4 GiB。PLE 27.5 GiB 本体は CPU mapped なので含めない。

安全側の 38/29 は expert layer 単位で丸めた。cache 自体は expert slot 単位なので、実装では 19,456 / 14,848 slots とし、特定 layer を固定する必要はない。compute peak がレンジ上端を越える、または物理総量が 32 GiB 未満なら 1-2 layer 相当をさらに減らす。

参考として静的 offload だけなら、同じ安全側容量に合わせて 128k は PTQ layer を少なくとも 6 層、256k は少なくとも 15 層 CPU に置くことになる。これらは全 decode token で CPU 実行される。dynamic cache は同じ VRAM でも hot expert を全 44 層から選べる点が違う。

### 6.3 indexer cache 改善の余地

PrismML indexer cache を K-only にすれば 6 KiB/token、さらに ratio-4 compressed key にすれば合計 8.25 KiB/token を削減できる。256k では約 2.06 GiB、PTQ1 約 4 層分に相当する。ただし pooling 時点、rope position、未完 block の state を正しく扱う必要があり、単なる tensor size 変更ではない。

この最適化は expert cache と独立して価値があるが、最初の cache correctness 実験に混ぜない。変更点を一つずつ検証する。

## 7. 推奨実装ステップ

### Phase 0: 変更前の測定

1. 32k / 128k / 256k で起動時の CUDA model/KV/compute buffer 内訳を記録し、上表の compute 推測を実測へ置換する。
2. `-ot` で PTQ layer を 0, 6, 15 層追加 offload した静的 baseline を測る。
3. coding prompt の prefill と 512-2,048 token decode を分け、pp/tg、VRAM、CPU bandwidth を採る。

### Phase 1: routing trace だけで cache を判定

現在の router 出力 top-10 ID を layer/token ごとに記録する。weight の配置や matmul は変えない。offline で FreeToken と同じ global `(layer, expert)` LRU を再生し、22,528 / 19,456 / 14,848 slots について以下を出す。

- 全体と layer 別 hit/miss
- bytes transferred/token (`misses * 1.025 MiB`)
- working set、90% coverage expert 数、entropy
- 最初の cold fill を除いた定常値
- prompt prefill 直後と長い decode 中の変化

**これが最初に行うべき小さな検証実験である。** PTQ kernel、allocator、非同期 copy を一切変えず、方式 (b) に投資する価値を先に判断できる。128k で miss が一様近似の 14% より十分低い、または実効転送見積もりが tg budget 内なら実装へ進む。256k が悪ければ hybrid CPU を先に作るのでなく、slot 増加、QSA indexer 削減、静的 hot expert の組合せを検討する。

### Phase 2: decode-only の最小 cache

1. layer 0-3 IQ3 は既存 `-ot` で CPU 固定。
2. layer 4-47 PTQ1 の gate/up/down を host bank として保持する。
3. PTQ1 の packed bytes/scale/sign を変形せず保持する GPU slot pool を作る。
4. routing 後に global ID -> slot ID を remap し、既存 `build_moe_ffn` / `build_lora_mm_id` / FWHT kernel に slot view を渡す。
5. 最初は同期 copy、batch 1、CUDA graph 無効で correctness を取る。
6. 一 expert、1 layer、全 layer の順に static GPU 結果と logits/hidden を比較する。

この段階では eviction policy を LRU 一つ、decode を GPU-only に限定する。CPU/hybrid と prefill overlap を同時実装しない。

### Phase 3: 非同期化と graph 安定化

- pinned host allocation と UVA/device pointer を導入。
- gate/up/down の同一 expert row を一つの copy plan にまとめる。
- copy stream/event と compute stream の lifetime を明示する。
- remap/copy workspace を固定 shape にして CUDA graph capture を戻す。
- cache stats を常設し、hit 率と実 H2D bytes を検証する。

### Phase 4: prefill

- 2 x 512 expert の full-layer double buffer。
- layer N compute と layer N+1 H2D の overlap。
- cache resident row は D2D、miss row だけ H2D。
- chunked prefill ごとに full 23.1 GiB を再転送しないかを計測する。

### Phase 5: 必要な時だけ hybrid / indexer 最適化

- routing trace が 256k で高 miss の場合、FreeToken 型の「recurring miss を GPU fetch、残り CPU」を追加する。
- CPU PTQ1 GEMV が十分速いかを単独 benchmark してから hybrid に使う。
- indexer V の未使用 allocation 除去、次に ratio-4 compressed index slab を別変更として検討する。

## 8. 実装上の注意点

- **mixed format:** FreeToken の一 cache は同一 bank shape/dtype 前提。最初の 4 IQ3 層は別 pool にせず CPU/static のままにする。
- **slot の意味:** 三つの projection を別々に evict しない。同一 `(layer, expert)` の gate/up/down/scale/sign は atomic に更新する。
- **Hadamard metadata:** slot copy で expert order を変える時、sign/rotation metadata も同じ mapping で追随させる。matmul ごとに input transform が必要。
- **prefill と decode:** decode で良い hit 率でも、prefill の full-layer stream は別問題。ベンチを分離する。
- **CUDA graph:** host が毎 token routing ID を読んで LRU を決める実装は避ける。最終的には device-side lookup/remap と固定 workspace が必要。
- **kernel の expert 次元:** global pool では MMID から見える slot 次元が最大 22,528 になる。既存 PTQ1 kernel の index 幅、grid、tensor stride がこの大きさを扱えるかを最初に確認する。上限がある場合は layer-local quota cache にして correctness を先に取り、後から layer 間の枠融通を足す。
- **OOM 余裕:** 32 GB 上限へ張り付くと既存測定のように速度が大幅低下する。cache size は「割り当て成功」ではなく 3-4 GB headroom と速度で決める。
- **複数 request:** 本表は single sequence。KV と GDN state は並行 sequence 数、scheduler slot 数で増える。server concurrency を上げるなら cache slot をさらに削る必要がある。

## 最終判断

FreeToken の cache は qwen4exp に概念上適用でき、現 FreeToken は official Qwen3.8-Flash-Next architecture まで既に対応している。しかし ternary 版に対する直接の障害は PTQ1_0 と `prism.hadamard.*` contract であり、これを FreeToken に再実装すると、現在動いている PrismML kernel と loader を二重化する。

従って、短中期は PrismML を runtime として維持し、FreeToken の設計から次の順に取り込むのが妥当である。

1. routing trace + offline global LRU simulation
2. PTQ1 44 層だけの decode-only global expert cache
3. pinned/coalesced async H2D と CUDA graph 対応
4. prefill double buffer + hit-D2D
5. 必要なら hybrid CPU と QSA indexer memory 削減

静的 `-ot` は各段階の baseline/fallback として残す。この順なら、最初の小さな trace 実験で cache の成立性を判断でき、成立しない場合も PTQ1 kernel や model graph を壊す前に止められる。


## 付録 A. 実測 VRAM 予算(2026-09-26、fork build-q4x、IQ3_XXS、routed expert を全て CPU、`-fa on`)

`llama-cli -lv 4 -ngl 99 -cmoe` のロードログから。CUDA context のアイドル分(約 0.8 GB)は別。

| context | model(非 expert) | QSA KV | GDN RS | indexer KV | compute | 合計 | 32 GB − 合計 − 3.5 GiB 余裕 = cache | IQ3 slot 数(1.88 MiB)/ 常駐率(48 層) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32k | 3,816 | 768 | 113 | 288 | 1,056 | 6,041 MiB | 約 22.1 GiB | 約 11,800 / 48 % |
| 128k | 3,816 | 3,072 | 113 | 1,152 | 1,201 | 9,354 MiB | 約 18.8 GiB | 約 10,000 / 41 % |
| 256k | 3,816 | 6,144 | 113 | 2,304 | 1,969 | 14,346 MiB | 約 13.9 GiB | 約 7,400 / 30 % |

routing trace(§ 20 of flash_next_progressive_pilot.md、コード corpus)の LRU 再生では常駐率 34 % で hit 91.8 %
(miss 35/token ≈ 66 MiB/token)、51 % で 97.1 %(12.4/token ≈ 23 MiB/token)。256k は帯域次第で decode が
数 ms/token 遅くなる見込み。indexer KV の K-only 化(§6.3、256k で約 2 GiB)は cache 側にそのまま効く。
