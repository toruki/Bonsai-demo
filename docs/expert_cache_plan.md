# Expert cache 実装計画(PrismML llama.cpp fork、IQ3_XXS Flash-Next、256k)

Codex(gpt-6-astra、read-only)による fork の静的調査に基づく計画。2026-09-26。
採用方針: 全 48 層で共有する固定 GPU slot bank 3 本 + slot ID remap、既存 mul_mat_id と量子化 kernel を再利用。
M0(観測)→ M1(host LRU の最小版、256k 動作)→ M2/M3(非同期化・device 化)→ M4(prefill)。

**採用するのは、全 48 層で共有する固定 GPU slot bank 3 本と、独立した slot ID tensor を使う方式です。** 最小版では既存の CPU custom op を routing と expert 計算の間に挿入し、host 側 LRU・miss 転送・ID remap を実施します。既存の CUDA `mul_mat_id` と量子化 kernel は再利用します。

その後、同じ bank 配置のまま device 側 lookup／copy に移行します。最初からスケジューラに汎用的な動的 weight cache を組み込むより、変更範囲と数値不一致の原因を限定できます。

以下はソースの静的調査に基づく計画です。ファイル作成・変更、ビルド、推論・計測は行っていません。

まず、容量の分母を修正する必要があります。

| 項目 | 全 48 層を対象にした値 |
|---|---:|
| 全 expert 数 | 48 × 512 = 24,576 |
| 1 slot | 962 / 512 = 約 1.879 MiB |
| 8,000 slot | 約 14.68 GiB、常駐率 **32.6%** |
| 10,600 slot | 約 19.45 GiB、常駐率 **43.1%** |
| 15 GiB に入る slot | 約 8,175、管理領域・padding を除く |

調査メモの表は旧構成の **44 PTQ 層**が対象で、replay も既定値が `--layers 4 47` です。提示された hit 率は有用ですが、元の IQ3 モデル・全 48 層・実際の slot 数で再集計するまでは、そのまま性能予測に使いません。256k と約 3.5 GiB の余裕を優先し、調整対象は cache 容量と prefill microbatch にします。  
根拠：[survey:243](/home/sohey/AI/LLM/Bonsai-demo/docs/freetoken_expert_offload_survey.md:243)、[lru_replay.py:20](/home/sohey/AI/LLM/Bonsai-demo/tools/flash_next_ternary/recon/lru_replay.py:20)。

**配置は固定し、実行時に変えるのは bank の内容と ID だけにします。**

Qwen4exp は routed expert を `[入力次元, 出力次元, expert数]` で作り、shared expert は別 tensor にしています。したがって ggml の次元順では、次の配置が自然です。

| bank | ggml の shape | 型 |
|---|---|---|
| gate | `[n_embd, n_ff_exp, S]` | IQ2_S |
| up | `[n_embd, n_ff_exp, S]` | IQ2_S |
| down | `[n_ff_exp, n_embd, S]` | IQ4_NL |

`S` は global slot 数です。論理的な `[slot, ...]` の slot は **`ne[2]`** に置きます。3 bank の同じ slot 番号が、一つの `(layer, expert)` を表します。router の expert 数は 512、top-k は 10 のままです。  
根拠：[qwen4exp.cpp:250](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/src/models/qwen4exp.cpp:250)、[ggml.c:3363](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml.c:3363)。

比較すると、次の判断になります。

| 方式 | 利点 | 問題・判断 |
|---|---|---|
| **固定 bank＋slot ID** | 既存 MMID、IQ2_S／IQ4_NL kernel を再利用できる | **採用**。準備処理と lifetime の追加が中心 |
| 元の CPU tensor の `data`／shape を実行中に差し替える | 見かけの変更量は小さい | scheduler の配置判断、graph 再利用、capture 済み pointer と衝突するため不採用 |
| expert pointer table を読む専用 MMID kernel | 大きな連続 bank が不要 | fusion・量子形式ごとの kernel 契約まで変更するため初期段階では不採用 |
| 既存 scheduler の選択 expert 転送を LRU 化する | 部分転送処理を利用できる | 元の expert ID／tensor shape に基づくコピー。global pool と 3 tensor 一体 eviction を入れると allocator／scheduler の変更が広がる |

既存の部分転送は `input->nb[2]` を expert サイズとして元の offset にコピーしています。永続 resident table はありません。  
根拠：[ggml-backend.cpp:1641](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-backend.cpp:1641)。

**routing の挿入点は `build_moe_ffn()` の top-k 確定後です。ただし元 ID は保存します。**

現在の流れは、top-k の `selected_experts` から routing weight を `ggml_get_rows()` で取得し、同じ ID を gate／up／down に渡しています。この ID を in-place で slot 番号にすると routing weight の取得が壊れます。  
根拠：[llama-graph.cpp:2145](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/src/llama-graph.cpp:2145)、[llama-graph.cpp:2162](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/src/llama-graph.cpp:2162)。

実装する流れは次です。

```text
router → top-k original_ids ─────────→ routing weights
                   │
                   └→ cache_prepare(layer, original_ids)
                          │  miss の gate/up/down を転送
                          └→ slot_ids
                                 │
固定 GPU bank ────────────────────┴→ 既存 mul_mat_id
                                       gate/up → SiLU → down
                                                         │
routing weights ─────────────────────────────────────────┴→ 集約
```

`build_moe_ffn()` 内に `expert_compute_ids` を追加し、cache 有効時だけ `slot_ids` にします。gate／up／down の MMID だけがこれを使い、routing と集約順序は保持します。shared expert も従来の計算を使います。  
根拠：[llama-graph.cpp:2228](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/src/llama-graph.cpp:2228)、[qwen4exp.cpp:987](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/src/models/qwen4exp.cpp:987)。

ここには一つ注意があります。`build_lora_mm_id()` は ID を外付け scale の取得と LoRA にも使います。最小版は対象モデルの 3 bank が通常の量子化 block だけで完結することを初期化時に確認し、外付け expert scale、expert LoRA、rotation、merged gate/up は未対応として明示的に拒否します。対応を広げる場合は、計算用 slot ID と付帯情報用 original ID を関数引数でも分離します。  
根拠：[llama-graph.cpp:1605](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/src/llama-graph.cpp:1605)。

**最小版の `cache_prepare` は、既存 `ggml_map_custom1()` の非 in-place、`n_tasks=1` を使います。**

これは入力と同じ型・shape の別 tensor を作り、userdata 付き関数を CPU で呼べます。入力を I32 top-k、出力を I32 slot ID とすれば、新しい ggml opcode はまだ不要です。GPU／CPU 境界は scheduler に処理させます。  
根拠：[ggml.c:6002](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml.c:6002)、[CPU ops.cpp:11544](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-cpu/ops.cpp:11544)。

処理は以下の順序です。

1. GPU の top-k を scheduler が CPU にコピーする。view の可能性があるため、ID は `nb[]` に従って読む。
2. `gid = layer * 512 + expert` を引き、選択された resident slot をすべて保護する。
3. miss ごとに空き slot、なければ保護されていない LRU slot を割り当てる。
4. 各 host bank の `expert * nb[2]` から、GPU bank の `slot * nb[2]` へ packed bytes をコピーする。
5. **3 bank の転送が完了してから** resident として確定し、元の top-k 順で slot ID を返す。
6. scheduler が slot ID を GPU にコピーし、既存 MMID を実行する。

LRU は全層共有ですが、所有者は **一つの `llama_context`** にします。model は読み取り専用 host bank を持ち、context が GPU bank、対応表、LRU、統計を持ちます。初期対象は単一 GPU・単一 sequence・1 token の microbatch です。

全選択 slot を先に保護するのは、同じ top-10 内で、これから利用する hit を別 miss の victim にしないためです。gate／up が終わっても down が読むので、途中で解放しません。

同期点は明確に分けます。

| 段階 | host に routing を読むか | 必要な同期 |
|---|---|---|
| 最小版 | **毎層読む** | top-k D2H 完了、slot 上書き前の既存 reader 完了、3 bank H2D 完了 |
| host LRU＋非同期転送版 | 毎層読む | D2H 待ちは残る。複数 H2D をまとめ、完了待ちを減らす |
| device 版 | **読まない** | 同一 CUDA stream 内で lookup → copy → MMID を順序付ける |

最小版は同期版 `ggml_backend_tensor_set()` を利用できます。CUDA 実装はコピー後に stream を同期するため、遅くても転送完了の契約が明確です。上書き前には対象 CUDA backend の reader 完了も保証します。  
根拠：[ggml-cuda.cu:786](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:786)。

**scheduler の allocator に cache bank を管理させないことが重要です。**

3 bank は context lifetime の専用 backend buffer に事前確保し、毎 token の compute buffer から切り離します。`WEIGHTS` usage と固定 pointer を使い、必要な padding を初期化します。host bank は userdata 経由の登録情報から参照し、decode MMID の source に入れません。これにより、scheduler に元の巨大な expert tensor を GPU へ複製させずに済みます。

また `-cmoe` は CPU 配置 override ですが、loader は CPU 用の追加 buffer type も検討します。cache の転送元には **再配置・repack されていない標準 GGUF layout** が必要なので、対象 routed weight は loader で通常の CPU buffer／mmap に固定します。PLE は cache 対象に含めません。  
根拠：[llama-model-loader.cpp:1193](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/src/llama-model-loader.cpp:1193)、[llama-model.cpp:1871](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/src/llama-model.cpp:1871)。

**CUDA graph は、正しさの確認後に二段階で戻します。**

最小版では CUDA graph と `GGML_CUDA_GRAPH_OPT` を無効にします。llama の graph 再利用とは区別し、後者には cache instance／mode／capacity 世代を再利用条件として追加します。bank の pointer や shape を毎 token 変更しません。現実装も CUDA graph の再利用判定で source pointer・shape・stride を比較しています。  
根拠：[llama-graph.h:892](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/src/llama-graph.h:892)、[ggml-cuda.cu:2593](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:2593)。

まずは CPU prepare を境にした **GPU split ごとの capture** を戻せます。host LRU は CUDA capture の外にあるため、この段階でも毎層の host 往復は残ります。

host 往復まで除く段階では、専用の CUDA prepare op を追加します。

- device 上の resident table、逆引き表、LRU、固定長 miss descriptor を更新する。
- 同じ top-k 順で slot ID を出す。
- mapped pinned host bank から、device の copy kernel が miss の packed bytes を取得する。
- bank pointer、workspace、kernel 起動構成を固定し、miss 数は device データとして扱う。
- copy 完了後に通常の MMID を実行する。

**lookup だけを device 化しても不十分です。** device が決めた可変 source／destination を host の `cudaMemcpyAsync` 発行で処理すると、miss 情報を host に返す同期が残ります。capture 内まで完結させるには device が解釈する copy descriptor と copy kernel が必要です。

既存の host registration は利用候補ですが、device からの参照可否・alias pointer・登録範囲の寿命を別途確認します。全 expert は約 45.1 GiB あり、全 bank の pinning は RAM・登録失敗リスクを伴います。host LRU 版では小さな pinned staging 領域を使う選択肢も残します。  
根拠：[ggml-cuda.cu:4924](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4924)。

bank 更新は graph から見えにくい副作用なので、device 版でも当初は単一 stream を維持します。複数 stream の最適化は、prepare と全 bank reader の依存関係を表現してからです。現 optimizer は source 関係から依存を判断します。  
根拠：[ggml-cuda.cu:4614](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:4614)。

**約 8,000 slot の decode は、既存 MMVQ を使える見込みがあります。ただし境界試験は必須です。**

quantized MMID は小 batch で `ggml_cuda_mul_mat_vec_q()` に入り、single-token kernel は ID から weight channel を直接選びます。選択数に応じた grid であり、8,000 slot 全体の GEMM を実行する構造ではありません。IQ2_S と IQ4_NL の実装もあります。  
根拠：[ggml-cuda.cu:1924](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu:1924)、[mmvq.cu:586](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-cuda/mmvq.cu:586)、[mmvq.cu:1311](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-cuda/mmvq.cu:1311)。

ただし offset は一部 `int` です。判定すべきなのは bank のバイト数が 2 GiB を超えるかではなく、**量子化 block 単位の最大 offset が int に収まるか**です。今回のサイズでは成立する見込みですが、最終 slot、高い slot 番号の gate/up fusion、全 miss を実寸で検証します。  
根拠：[mmvq.cu:668](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-cuda/mmvq.cu:668)、[vecdotq.cuh:1543](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-cuda/vecdotq.cuh:1543)。

**prefill の第一段階は、routed expert を CPU で従来どおり計算します。**

`ubatch.n_tokens == 1` かつ単一 sequence だけ cache 経路を使い、それ以外は元の host weight を使います。1 token の prompt 末尾もこの経路で処理可能です。単に CPU に weight を置くだけでは、scheduler の op offload によって GPU に戻される可能性があるため、prefill の routed MMID は明示的に CPU に割り当てます。  
根拠：[ggml-backend.cpp:950](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-backend.cpp:950)。

この段階の契約は、**256k が処理でき、decode cache が正しく動くこと**です。prefill 高速化は含めません。prefill による cache warm-up も初期版では行わず、cold decode を実測します。

後続の prefill 改善は、まず 512 slot の全 expert layer buffer 一つ、必要なら二つで overlap します。今回の型では一つ約 962 MiB、二つ約 1.879 GiB です。これは別途 VRAM を増やすか、既存 cache の一部を借りる必要があります。後者なら借りた slot の resident entry を必ず無効化します。

さらに、full-layer streaming は通常の layer 順 prefill では **microbatch ごとに最大約 45.1 GiB を再転送**します。256k 全体で一回だけ転送すれば済むとは見積もりません。

変更対象は次の範囲に限定します。新規名は実装案です。

| ファイル／関数 | 変更内容 |
|---|---|
| 新規 `src/llama-expert-cache.{h,cpp}` | host bank 登録、3 GPU bank、global LRU、prepare custom op、統計、解放 |
| `src/llama-context.{h,cpp}`：constructor／destructor、`graph_params()`、`sched_reserve()` | context が cache を所有。256k の memory と pp／tg 両 graph を含めて容量検証 |
| `src/llama-graph.{h,cpp}`：`llm_graph_params::allow_reuse()`、`build_moe_ffn()` | cache を渡す、original／slot ID の分離、decode 分岐、prefill CPU 割当 |
| `src/models/qwen4exp.cpp`：`build_layer_ffn()` | 対象 bank／layer の選択。shared expert は既存処理 |
| `src/llama-model-loader.cpp`：`create_tensor()`、`src/llama-model.cpp`：load 後検査 | routed bank を canonical host layout に固定し、型・shape・stride を検証 |
| `include/llama.h`、`common/common.{h,cpp}`、`common/arg.cpp`、`src/CMakeLists.txt` | opt-in の cache 容量設定、パラメータ受け渡し、追加ソース登録 |
| `tools/flash_next_ternary/dump_hidden/dump_hidden.cpp`、`recon/lru_replay.py` | 実 decode の同期費用測定、全 48 層・実容量の replay |
| `tests/test-backend-ops.cpp` | 既存 `test_mul_mat_id` に remap／高 slot 番号の検証を追加 |
| device 版のみ：`ggml.h`、`ggml.c`、`ggml-cuda.cu`、新規 CUDA cache 実装 | prepare op、device LRU／copy、backend capability と capture 対応 |

**初期版では `ggml-backend.cpp` と MMVQ の変更を予定しません。** 標準の backend 分割と buffer API を利用し、実寸試験で問題が見つかった場合だけ kernel 修正を追加します。既存の reserve は pp と tg の双方を扱っているので、ここに cache 付き topology を反映します。  
根拠：[llama-context.cpp:736](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/src/llama-context.cpp:736)、[test-backend-ops.cpp:4800](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/tests/test-backend-ops.cpp:4800)。

段階と規模の見積もりは、テスト・設定を含む追加／変更行数で次の程度です。

| 段階 | 成果物・完了条件 | 規模目安 | 主なリスク |
|---|---|---:|---|
| M0：観測 | 実 decode の top-k host 読み出し費用、全 48 層 replay、実 tensor layout 一覧 | 100–250 行 | callback の観測費用を本体性能と混同する |
| M1：最小 cache | CPU prepare、同期転送、全 48 層 global LRU、prefill CPU、256k 動作 | 700–1,200 行 | slot 寿命、loader repack、graph 切替、padding |
| M2：host 版改善 | pinned staging／登録、H2D 一括発行、split 単位 capture | 250–500 行 | 転送元の寿命、stream の順序、毎層同期が支配的 |
| M3：device 化 | device LRU、mapped-host copy、固定 workspace、capture 再生 | 900–1,600 行 | pinning、LRU 更新費用、capture と副作用 |
| M4：prefill 改善 | full-layer streaming、必要なら double buffer | 500–900 行 | cache 容量との競合、microbatch ごとの再転送 |

M1 は機能として運用評価できる区切りです。M2 以降の採否は測定から判断します。

**数値検証は「同じ重み・同じ CUDA 演算で配置だけ違う」比較を中心にします。**

- **packed bytes**：転送元 expert と slot の 3 tensor を byte 単位で一致確認。hit、miss、eviction、最終 slot を含める。
- **単層比較**：1 層分約 962 MiB を通常の 512-expert GPU tensor として置き、同じ activation／original ID に対する cache 版と比較する。全モデルを GPU に載せる必要はない。
- **中間値**：gate／up、SiLU、down、routing weights、MoE 出力を比較。kernel・fusion を揃えた配置変更のみの試験では bitwise 一致を第一基準にする。
- **全モデル**：固定 token 列を teacher forcing で与え、元 ID、各層出力、logits を比較する。CPU baseline との差には演算差があるため、単層の配置一致試験と分け、最大絶対誤差・相対 L2・logit 順位を記録する。
- **lifetime**：小容量で強制 eviction、全 hit、全 miss、prefill→decode→prefill、graph 再構築、capture の warm-up／replay を試す。
- **256k**：context の確保成功だけでなく、長い prefix を実際に処理して末尾で decode する。KV／GDN／PLE state を変えず、VRAM peak と余裕を確認する。

性能は callback を外した状態で、prefill と decode を別々に測ります。decode は cold／warm の tok/s、token latency の p50／p95、miss/token、実 H2D bytes、routing 待ち、LRU、転送、MMID、VRAM peak を記録します。

例えば提示値の 35 miss/token が再現すれば、転送 payload は約 **65.8 MiB/token** です。ただし時間には転送だけでなく、48 層の host 往復と計算が加わります。hit 率だけから tok/s は予測しません。

**最初の 1 日は、既存 trace を取り直すだけで終えず、「host 往復の費用」を切り分けます。**

既存 `dump_hidden` は `ffn_moe_topk` の I32 と strided view を取得できます。ただし現状は prompt を batch で渡す構造なので、1 token ずつ進む測定モードを追加する計画にします。  
根拠：[dump_hidden.cpp:28](/home/sohey/AI/LLM/Bonsai-demo/tools/flash_next_ternary/dump_hidden/dump_hidden.cpp:28)、[dump_hidden.cpp:122](/home/sohey/AI/LLM/Bonsai-demo/tools/flash_next_ternary/dump_hidden/dump_hidden.cpp:122)。

同じ token 列・weight 配置で次の四条件を比較します。

| 条件 | 分離する費用 |
|---|---|
| A：callback なし | baseline |
| B：top-k で停止する callback、データ取得なし | graph 分割・同期の費用 |
| C：B＋top-10 ID を host に読む | D2H の増分 |
| D：C＋8,000 slot の CPU LRU、weight 転送なし | LRU の増分と miss 分布 |

scheduler の評価 callback は対象 node まで実行して backend を同期するので、B を置かないと D2H／LRU の費用を過大評価します。測定区間中のファイル書き込みは避け、集計結果を最後に出します。  
根拠：[ggml-backend.cpp:1748](/home/sohey/AI/LLM/Bonsai-demo/llama.cpp/ggml/src/ggml-backend.cpp:1748)。

同日に replay を全 48 層・8,000 slot・実 byte 数へ合わせ、cold 区間と定常区間を分けます。実装予定の「選択中 slot を保護する」LRU も参照モデル化します。初日の成果物は、**正しい容量分母、host 制御の ms/token、実際の miss bytes/token** の三つです。これで M1 の検証基準が定まり、host 版で運用評価まで進めるか、早めに M3 が必要かを判断できます。

## M0 の実測(2026-09-26)

### routing trace(IQ3_XXS、全 48 層、コード corpus 32k token、選択中 expert を保護する LRU)
layer 0 は他より散る(picks の 50 % / 90 % を 82 / 282 expert)。中間層は 31 / 204。

| slot 数 | 常駐率 | hit | miss/token | MiB/token(1.879 MiB/slot) |
|---:|---:|---:|---:|---:|
| 14,848 | 60 % | 98.4 % | 7.9 | 14.8 |
| 11,800 | 48 % | 96.3 % | 17.7 | 33.2 |
| 10,600 | 43 % | 95.1 % | 23.6 | 44.3 |
| 8,175(256k 予算) | 33 % | 91.1 % | 42.8 | 80.5 |

### host 往復の費用(ternary-44 を GPU 常駐、1 token ずつ 128 token、`dump_hidden --decode-timing`)

| 条件 | CUDA graph あり | CUDA graph なし |
|---|---:|---:|
| A: callback なし | 7.53 ms/tok(133 t/s) | 15.82 |
| B: 48 層の top-k で graph を止める(読まない) | 20.06 ms/tok(50 t/s) | 29.44 |
| C: B + top-10 id を host に読む | 19.85 | — |
| D: C + 8,175 slot の LRU(転送なし) | 19.79 | — |

- 層ごとの graph 分割・同期そのものが **約 12.5–13.6 ms/token**(48 分割で 1 分割 ≈ 0.27 ms)。
  id の読み出しと LRU の計算は誤差の範囲。CUDA graph の有無によらず同じ増分なので、graph の再生を失う
  費用ではなく分割・同期の費用。
- 256k 予算(8,175 slot)で miss 80 MiB/token を PCIe で転送すると +3–5 ms/token の見込み。
- したがって host 制御の M1 は約 24 ms/token(約 40 t/s)が上限の目安で、133 t/s 級に戻すには
  M3(device 側 lookup/copy、同期なし)が必須。M1 は正しさと 256k 動作の確認に位置付ける。


## M1 の実装と検証(2026-09-26、fork commit abb11c9e6)

`--expert-cache-slots N`(`--cpu-moe` を暗黙に付ける)。`src/llama-expert-cache.{h,cpp}`(約 230 行)+
`build_moe_ffn` の分岐(n_tokens == 1 のとき、top-k id を `ggml_map_custom1` の CPU op で slot id に写像し、
3 本の bank tensor に対して既存の `mul_mat_id`)。routing weight は元の id のまま。

- **正しさ**: layer 36–47 を bank 経由にした場合と、同じ層を GPU 常駐にした場合(同じ CUDA kernel)で
  4 chunk × 512 token の単 token decode を比較 → **KL 0.000000(最大 4.9e-5)、PPL 同一**。
  PPL は slot 数(2,000 / 4,000 / 6,000 / 8,175)によらず同一 = eviction による不整合なし。
- 参考: CPU expert path(`-cmoe`)と GPU path は数値が異なる(KL 0.042、top-1 一致 94 %)。
  これは kernel の差で cache の問題ではないが、CPU 側の量子化 matmul の精度として要注意。
- unsloth UD-IQ3_XXS は layer 2 だけ gate/up が IQ3_S(他は IQ2_S)なので layer 2 は cache 対象外(CPU で計算)。
- **速度**(8,175 slot、コード corpus、2048 token prompt の後 512 token): **34 ms/token(29 t/s)**、p95 53 ms。
  M0 の見積もり(同期 12.5 ms + 転送)どおりで、pageable memory からの H2D が残りの大半。
- prefill(expert は CPU、op offload が層ごとに重みを GPU へ流す): ub 512 で 158 t/s、**ub 2048 で 569 t/s**。
  256k は約 8–10 分の見込み(attention の増分は別)。
- VRAM: 8,175 slot + c 512 で 21.6 GB。

### 256k end-to-end(M1 の契約)
`dump_hidden --decode-timing`(262,000 token のコードを ub 1024 で prefill → 64 token を decode)、7,000 slot:
- prefill **262k token を 1,535 s(25.6 分、平均 171 t/s)**。4k での 569 t/s(ub 2048)より遅いのは、
  context が伸びるほど QSA attention / indexer の分が増えるためと、ub 1024 のため。
- decode at 256k: **101 ms/token(9.9 t/s)**、p95 120 ms。短い context の 34 ms との差は attention 側の
  費用(GPU 常駐の ternary-44 で同条件を測って切り分け中)。
- VRAM peak **31.8 GB** — 上限に張り付いた。7,000 slot(13.1 GB)+ KV 6 GB + indexer 2.3 GB + ub 1024 の
  compute で余裕がない。256k では slot を 5,500 程度に落とすか、indexer KV の K-only 化(約 2 GB)が要る。

### context 長と decode 速度(IQ3_XXS + cache、1 token ずつ)

| context | slot | VRAM peak | decode | p95 | prefill 平均 |
|---:|---:|---:|---:|---:|---:|
| 2k | 8,175 | 21.6 GB | 34–37 ms(27–29 t/s) | 53–59 | 569 t/s(4k、ub 2048) |
| 128k | 7,000 | 25.3 GB | 70.7 ms(14.2 t/s) | 108 | 205 t/s(ub 1024) |
| 256k | 7,000 | 31.8 GB | 101 ms(9.9 t/s) | 120 | 171 t/s |
| 256k | 4,500 | 26.5 GB | 113.8 ms(8.8 t/s) | 133 | 197 t/s |

context 依存の増分は約 +0.27 ms / 1k token(256k で +67 ms)で、cache 由来(約 25–30 ms)より大きい。
QSA / Lightning Indexer の decode 経路(indexer K cache の走査と top-2048 選択)が疑わしい。
GPU 常駐の ternary-44 で 256k を試すと VRAM が溢れてシステムメモリに退避(prefill 5.3 時間、decode 348 ms)。

host の expert tensor を cudaHostRegister で pin(`LLAMA_EXPERT_CACHE_PIN=1`)しても 36.7 → 35.3 ms と効果なし。

## M2(2026-09-26)と長文 decode の分析

- `LLAMA_EXPERT_CACHE_STAGING=1`: miss を page-locked staging(top-k 分 18.8 MiB)経由で非同期にコピーし、
  token ごとに 1 回だけ同期 → 2k context で **34.1 → 26.7 ms/token(37.5 t/s)**、PPL 3.7755 のまま。
  cudaHostRegister による pin は `GGML_CUDA_REGISTER_HOST` 未設定だと即 false(141 range 失敗)で、未検証。
- CPU スレッド数(1/4/16)は ±2 ms。

Codex(gpt-6-astra)による 256k の decode 増分(+67 ms)の分析(fork の静的調査):
1. **QSA attention が全長 KV を処理している**: `build_attn_qsa()` は選択位置以外を −∞ にしたマスクで通常の
   `build_attn_mha()` を呼ぶ(gather なし)。FA は QK を全長で計算してからマスクを足すので、256k では 6 GiB の
   KV を 12 層ぶん読む。選択した約 2,051 セルなら 48 MiB。→ 選択 KV を gather して FA(15–30 ms/token)。
2. **indexer が毎 decode で全履歴を再 pooling**(gather + cont + 加算 + norm + RoPE を過去の全ブロックで再実行)。
   完成ブロックの結果を保持して増分更新に(GPU 10–25 ms + CPU `set_input_qsa` 5–10 ms)。
3. top-k は CUDA 13.3 + CCCL 3.3 の DeviceTopK を既に使っているので優先度低。
4. VRAM: indexer の未使用 V(1.5 GiB)、全長 score/mask の compute、prefill/decode の compute 領域管理、main KV の Q8_0(2.8 GiB)。
推奨順: 未使用 V 削除 → QSA の gather attention → indexer の増分化 → M3(device 側 lookup、8–15 ms)→ M4。

### warm 状態の context 依存(pin 有効、prefill 後 400 token を除いた 300 token の平均)

| context | slot | prefill(ub 1024) | decode | 最初の 400 token |
|---:|---:|---:|---:|---:|
| 2k | 8,175 | — | 26.4 ms(37.9 t/s) | — |
| 32k | 8,175 | 534 t/s | 37.3 ms(26.8 t/s) | 35.3 ms |
| 128k | 7,000 | **415 t/s**(pin なしは 205 t/s) | 58.6 ms(17.1 t/s) | 57.3 ms |

- context 依存の増分は約 0.25 ms / 1k token → 256k で約 90 ms/token(11 t/s)の見込み。attention / indexer 側。
- `GGML_CUDA_REGISTER_HOST=1` で host の expert を page-lock すると、prefill(op offload が層ごとに重みを
  GPU へ流す)も DMA になり **2 倍**速い。256k の prefill は約 12 分の見込み。
- prefill 直後の cold cache の影響は小さい(最初の 400 token で +2 ms 程度)。

## 長文 decode の改善(2026-09-26)

- **indexer の未使用 V を 1 要素に**(fork `0c9fcf159`): 256k で indexer KV 2,304 → 774 MiB(−1.5 GiB)。
  batch path / 単 token path とも PPL 同一。
- **QSA attention の gather 化**(fork、`LLAMA_QSA_GATHER`、既定 on): 単 token decode では top-k の cell だけを
  `get_rows` で集めて(256 の倍数に −inf の行で padding)flash attention に渡す。同一入力での最初の QSA 層の
  attention 出力は masked 版と relMSE 4.7e-9 で一致(fp16 の丸め)。
- **MoE の数値感度**: 下流の層では MoE の top-k routing が離散的に切り替わるため、わずかな数値差が logits では
  KL 0.02–0.06 に増幅される(batch vs 単 token: 0.057、CPU expert vs GPU: 0.042、gather vs masked: 0.017)。
  cache の「KL 0」は byte 一致の copy だからで、これは実装の正しさの判定には attention 出力の直接比較を使う。
- prefill 中の routing 記録 → 最初の decode で warm-up(`LLAMA_EXPERT_CACHE_WARM=1`)は、最初の 100 token で
  −4 ms/token に対して 15 GB の一括 copy が要るので既定 off。
- gather 化の速度: 32k 37.3 → 38.1 ms(差なし)、128k 58.6 → 55.4 ms(−3 ms)。全長 KV の読み出しは
  支配的ではなかった(FA kernel が mask 済み tile を飛ばしていたと見られる)。context 依存の +30 ms/128k の
  正体は別(indexer の全履歴再 pooling と周辺の O(n_kv) op 群、CPU の `set_input_qsa` が候補)。64k で
  Nsight のプロファイルを取って特定する。

### 同期 1 回の固定費(WSL2、RTX 5090、マイクロベンチ)

| 操作 | 時間 |
|---|---:|
| cudaStreamSynchronize(空の stream) | 0.1 µs |
| kernel 起動 + synchronize | **38 µs** |
| 64 B の H2D / D2H(pinned)+ synchronize | **34–35 µs** |
| 64 B pageable cudaMemcpy | 36 µs |
| 1.88 MiB pinned H2D + synchronize | 84 µs(≈ 22 GB/s) |
| kernel 起動のみ | 1.7 µs |

GPU との往復 1 回 ≈ 35 µs。Nsight のタイムラインでは decode 1 token あたり cudaStreamSynchronize が
305(32k)〜476(64k)回 → **12–17 ms/token** で、GPU の idle 27–29 ms/token の大半を占める。
64k の GPU kernel 時間は 9.8 ms/token(expert の mmvq 4.7、indexer の K gather 1.8、FA 0.3)。
scheduler は CPU op の入出力(top-k id の D2H、slot id の H2D)ごとに数回 synchronize するので、
host 主導の設計ではこの固定費を消せない。→ **M3(device 側の lookup + copy、同期なし)を優先**。


## M3: device 側 lookup + fill(2026-09-26、fork commit は git log 参照)

新 ggml op `GGML_OP_EXPERT_CACHE`(CUDA 実装のみ、`ggml/src/ggml-cuda/expert-cache.cu`):
- host の expert tensor(44.2 GiB)を `cudaHostRegister(Mapped|Portable|ReadOnly)` で device から見える形に登録
  (WSL2 でも成功)。マイクロベンチでは device kernel が mapped host memory を読む帯域 43 GB/s(cudaMemcpyAsync 47 GB/s)。
- lookup kernel(1 block): 層ごとの固定 partition(slots/47 = 173)で LRU。hit を先に保護してから miss ごとに
  stamp 最小の slot を退避、miss list を device に書く。fill kernel(grid k × 32 chunk): miss の 3 tensor を
  host → bank slot にコピー。同じ stream なので後続の mul_mat_id との順序は自動、host の同期はゼロ、
  token 全体が 1 つの CUDA graph に入る。
- global LRU と層別 partition の hit 率差は 0.5 %(replay)。
- 正しさ: masked QSA(決定的)で host LRU と PPL 同一(3.7755、4,000 / 8,175 slot)、常駐 slot の byte 比較
  (`LLAMA_EXPERT_CACHE_VERIFY=300`)で不一致 0。
- 速度(2k context、1024 token、8,175 slot): **34 → 4.15 ms/token(240 t/s)**、hit 88.3 %。

### 単 token decode の非決定性
gather 版 QSA(`LLAMA_QSA_GATHER=1`)は同一設定でも run ごとに PPL が変わる(3.8228 / 3.8171、KL 0.046):
top-k の出力順が安定せず、attention の累積順の丸め差が MoE の routing で増幅される。masked 版は決定的
(3.7755 が再現)。gather は既定 off に戻した(128k で −3 ms の価値しかない)。

### M3 の context 依存(warm、pinned prefill、masked QSA)— 訂正版

**訂正**: 最初に報告した M3 の decode(2k 3.9 / 32k 6.6 / 128k 18 / 256k 34 ms)は計測ツールの誤りで、
`llama_decode` の投入時間しか測っていなかった(同期ゼロの M3 では GPU の完了を待たない)。
`llama_synchronize` を含めて測り直した値:

| context | slot | prefill | decode(訂正後) | p95 | hit | VRAM peak |
|---:|---:|---:|---:|---:|---:|---:|
| 2k | 8,175 | 577 t/s | **14.7 ms(68 t/s)** | 18.8 | 87.6 % | 22.6 GB |
| 2k | 12,000 | — | **13.1 ms(76 t/s)** | 16.3 | 92.6 % | — |
| 32k | 8,175 | 581 t/s | **21.5 ms(46.5 t/s)** | 24.6 | 82.6 % | 23.6 GB |
| 128k | 7,000 | 534 t/s(4.1 分) | **40.1 ms(24.9 t/s)** | 43.8 | 78.7 % | 24.9 GB |
| 256k | 6,500 | 471 t/s(9.3 分) | **62.4 ms(16.0 t/s)** | 66.2 | 75.2 % | 28.7 GB |

llama-server / llama-cli の実測(2k、200 token 生成)も 59–65 t/s で整合。M2 比で 2k 34 → 14.7、32k 37 → 21.5、
128k 59 → 40、256k 101 → 62 ms。context 依存の増分は約 0.19 ms / 1k token で、256k では decode の約 3/4 を占める
(indexer の全履歴再 pooling と O(n_kv) の host 処理が候補)。Nsight(2k、llama-cli): GPU kernel 11.2 ms/token
(expert mmvq 4.9、fill 3.3、quantize 1.0)、idle 5 ms。

### PLE 予測 lookup の O(1) 化(2026-09-26、fork commit 44bafce31)

`llama_kv_cache::get_prev_tokens()`(PLE n-gram の前 2 token を引く)が decode token ごとに使用中 cell を
全走査していた(32k で 3.2 ms、128k で約 13 ms、256k で約 25 ms/token)。`apply_ubatch()` で直近 256 個の
(seq, pos, token) をリングに記録し、追記 decode ではリングで解決、cell を編集する操作(seq_rm/cp/keep/add/div、
clear、state 復元)でリングを捨てて全走査に戻す。PPL 同一(3.7755 / 3.6967)。

| context | slot | prefill(pinned) | decode(M3 訂正版) | **PLE 修正後** | p95 | hit |
|---:|---:|---:|---:|---:|---:|---:|
| 32k | 8,175 | 581 t/s | 21.5 ms | **17.8 ms(56 t/s)** | — | 82.6 % |
| 128k | 7,000 | 475 t/s(4.6 分) | 40.1 ms | **29.8 ms(33.6 t/s)** | 33.5 | 78.6 % |
| 256k | 6,500 | 383 t/s(11.4 分) | 62.4 ms | **38.0 ms(26.3 t/s)** | 44.2 | 75.5 % |

`LLAMA_DECODE_PROFILE=1` の段階別内訳(ms/token、単 token ubatch の平均):

| context | set_inputs | うち qsa | compute launch | wait for device | 合計 |
|---:|---:|---:|---:|---:|---:|
| 128k | 1.15 | 1.0 | 3.3 | 24.0 | 29.8 |
| 256k | 2.5 | 2.1 | 4.3 | 31.8 | 38.0 |

残る context 依存分は GPU 側(2k の約 11 ms → 256k 32 ms、約 0.08 ms / 1k token)で、候補は indexer の全履歴
再 pooling(`build_qsa_top_k`: 全ブロックの K を gather → 平均 → norm → RoPE を毎 token、12 層)と、その周辺の
O(n_kv) op(score の cell 展開、mask 加算、top-k)。CPU の `set_input_qsa`(O(n_kv) の cell 走査 + 約 3.3 MiB の
H2D)は 256k で 2.1 ms。prefill が前回より遅い(471 → 383 t/s)のは同時に走っていた解析ジョブの影響と見られ、未確認。

### 128k decode の Nsight 内訳(PLE 修正後、7,000 slot、284 token、27.0 ms/token)

GPU busy 21.9 ms / idle 5.2 ms。上位カーネル(ms/token):

| 項目 | ms/token | 内容 |
|---|---:|---|
| `ecache_fill_kernel` | 6.15 | expert cache の miss(47 層、hit 78.6 %、約 190 MiB/token を mapped host から読む) |
| `mul_mat_vec_q` | 4.52 | expert / attention の行列ベクトル積 |
| `k_get_rows_float`(grid = n_kv) | 3.16 | **indexer**: 全ブロック member の gather(r·n_blocks 行)+ score の cell 展開(n_kv 行)、12 層 × 2 |
| `flash_attn_ext_f16` | 2.51 | **QSA の全長 masked attention**、12 層 × 0.21 ms |
| DtoD memcpy 16 MiB × 48 | 0.80 | indexer: member slice の `ggml_cont` × 4 / 層 |
| `k_bin_bcast` / `rms_norm` / `scale` / `rope`(grid = n_blocks) | ~1.5 | indexer: pooling の加算・norm・rope、mask 加算 |

context に比例するのは indexer の再 pooling 連鎖(約 5.5 ms @128k → 約 11 ms @256k)と全長 FA(2.5 → 5 ms)で、
合わせて 256k の 38 ms のうち約 16 ms。CPU の `set_input_qsa` は 1.0 / 2.1 ms。完成ブロックの pooled K は不変なので、
永続テンソルに保持して新規ブロックだけ更新する「増分 pooled-K」(recent-token ring と同じ追記高速路 + cell 編集で無効化)で
前者を消せる見込み(256k で約 38 → 28 ms)。設計は gpt-6-astra にレビュー依頼(scratch: codex_pooled_k.md)。

### 増分 pooled-K と決定的 top-k(2026-09-26、fork commits 6c7ff9fc4 / 323accbe4 / 9567b3609)

gpt-6-astra のレビュー(scratch `codex_pooled_k.md`)に沿って実装。要点:
- **永続 pooled K**(`llama_memory_hybrid_idx::pooled`): QSA 層ごとに F32 [128, kv_size/r + 2] を indexer K と同じ
  buffer type に確保(256k で 384 MiB)。完成ブロックの pooled K(norm・rope 後)は member 4 cell と block 位置だけで
  決まるので不変。単 token 追記(fast path)は「その token で完成するブロック」と「末尾が写像される spare ブロック
  (full path と同じく cell 0 ×4・位置 0 から作る)」の 2 行だけを pool → `ggml_set_rows` → 保存行の view に対して
  score。`set_input_qsa` の O(n_kv) 走査も省き、cell_blk は host 側の写しを差分更新、bias は n_bid / spare / -inf。
- **有効条件**(Codex A): stream 1 本、1 sequence が位置 0..n-1 を隙間なく占め、ranked(mrope 画像)でなく、
  full path が最後に記録したレイアウトと indexer cache の `edit_gen` が一致し、追記位置が n_bid·r + tail と一致する
  text token 1 個。full path は毎回レイアウトを検証して記録し(2D/複数 seq/gap があれば無効)、全行を書き戻すので
  prefill 直後の最初の decode から fast path に入る。cell 編集(seq_rm/cp/keep/add/div、clear、state 復元)は
  `recent_clear()` と同じ場所で `edit_gen` を進めるので、1 ubatch だけ full path に戻って復帰する。
- `prepare()` の投機的 `apply_ubatch()` が recent ring に入っていた(PLE 修正の取りこぼし)のも併せて修正。
- **決定的 top-k**(`ggml/src/ggml-cuda/top-k.cu`): CUB DeviceTopK は同値の要素を任意に選ぶ。QSA は block score を
  4 cell に展開するため cutoff(2051 − tail)が 4 回に 3 回同点になり、**同じリクエストでも run ごとに生成が変わって
  いた**(llama-server、temp 0、20k prompt で確認)。score の順序保存ビットと index の補数を 64-bit key に詰めて
  `MaxKeys` で選ぶ(同点は index 小が勝つ)。

検証:
| テスト | 結果 |
|---|---|
| 単 token PPL、4 × 512(cutoff なし)、pooled off を base に on の KL | **KL 0.000000、PPL 同一(2.4676)**、決定的 top-k でも同一 |
| 8k context、decode 6 token、中間テンソル dump | layer 3(最初の QSA 層)の block score は有効行で bit 一致、top-k 集合は 6 token すべて一致。差は -inf でマスクされる未使用行のみ。後段の差は旧 top-k の同点選択によるもの |
| llama-server、20k prompt → 追記 → checkpoint 復元 + 途中編集 → 追記(5 リクエスト、temp 0) | 旧 top-k: off 同士でも run ごとに B/E が変化。**決定的 top-k: on×2 と off がすべて同一** |
| 32k Nsight A/B(同条件・連続) | GPU busy 15.34 → 14.43 ms/token(get_rows −0.41、cont copy −0.19、bcast −0.12、norm −0.10、rope −0.06)、wall 19.28 → 17.87 |

wall time の run 間ばらつきは ±1–2 ms あり(同じコードで 128k が 26.3 と 29.8 ms の日もある)、32k の A/B の初回は
逆に +1.9 ms だった。効果はカーネル時間で判断する。

128k Nsight A/B(7,000 slot、同条件・連続、pooled off → on): **wall 27.12 → 21.56 ms/token(46 t/s)、GPU busy 21.91 → 17.93**。
内訳: get_rows −1.64、DtoD copy −0.78、bcast −0.56、rms_norm −0.39、rope −0.22、scale −0.19、FA −0.27(ばらつき)。
残る context 依存の GPU 時間は全長 masked FA(2.24 ms @128k)と score の cell 展開 + top-k(get_rows 1.6 ms のうち
n_kv 行の展開分)。

### score 展開の flat get_rows と PLE の host gather(2026-09-26、fork commits e9cb6add3 / 5ef0d8422)

gpt-6-astra の次段の順位付け(scratch `codex_next.md`): #1 単要素行の get_rows を flat 化、#2 PLE の gather を
set_input へ、その後は堅牢化・prefill へ、という判断に同意して実装。
- `getrows.cu`: 行長 1 の get_rows(QSA の block score → cell 展開、12 層 × n_kv 行)は 1 行 1 ブロック(256 スレッド中
  1 本だけ働く)だったのを 1 要素 1 スレッドの flat カーネルに。値は同一。
- PLE: 27.5 GiB の host-mapped テーブルからの行 gather を graph 内の CPU op から `llm_graph_input_ple::set_input`
  の host gather + F32 入力に。graph splits(bs=1)は 4 のまま(残りは layer 2 の CPU expert 等)。`LLAMA_PLE_HOST_GATHER=0` で旧経路。
- KL の下限: llama-perplexity の `--kl-divergence-base` は base を量子化して保存するため、**同一構成でも Maximum KLD
  0.000050 が出る**(Mean は 0.000000)。今日の全変更はこの下限内で logits 同一。

**最終計測(2026-09-26 夜、GPU 単独、pooled K + flat get_rows + PLE host gather + 決定的 top-k)**

| context | slot | prefill | decode | p95 | hit | set_inputs / launch / wait(ms) | 朝(M3 訂正版) |
|---:|---:|---:|---:|---:|---:|---|---:|
| 2k | 8,175 | 497 t/s | **13.6 ms(73 t/s)** | 16.9 | 87.4 % | 0.07 / 2.8 / 11.4 | 14.7 |
| 32k | 8,175 | 509 t/s | **16.5 ms(60.5 t/s)** | 19.6 | 82.4 % | 0.07 / 2.8 / 13.0 | 21.5 |
| 128k | 7,000 | 471 t/s(4.6 分) | **18.8 ms(53 t/s)** | 22.4 | 79.2 % | 0.18 / 2.9 / 15.7 | 40.1 |
| 256k | 6,500 | 421 t/s(10.4 分) | **22.3 ms(44.8 t/s)** | 25.6 | 75.0 % | 0.35 / 3.1 / 19.0 | 62.4 |

context 依存の増分は 0.19 → **0.034 ms / 1k token**。残りは expert の miss(fill 約 6 ms、hit 率で決まる)、expert mmvq
(4.5 ms)、全長 masked FA(256k で約 4.5 ms)、CPU の launch(約 3 ms)。VRAM peak(256k)は再計測時に未取得(以前 28.7 GB
+ pooled 384 MiB)。

### 堅牢化(2026-09-26 夜、ユーザー判断: decode 最適化は一区切り、運用準備へ)

- **KV 編集シナリオのテスト** `tools/flash_next_ternary/qsa_edit_test/`(`q4x_eval/edit_test.sh`): hybrid(GDN)の
  状態は `seq_rm` では巻き戻せない(recurrent の部分削除は投機デコード用の rollback 窓 `n_rs_seq` が無いと失敗する)ので、
  llama-server の checkpoint と同じ「snapshot → 続行 → restore → 末尾 truncate → 再生」を主軸に、奇数バッチ(5)、
  `seq_cp` → 2 sequence 同居 → `seq_keep`、7 token 刻みの prefill、256 cell / 1024 token 境界をまたぐ単 token、
  whole-context と per-seq の state blob を網羅。**69 step すべて pooled off/on で logits hash が一致、restore 後の
  再生も一致(fail 0)**。
- **256k の VRAM peak**(最終コード、6,500 slot、prefill + decode 中 0.5 s サンプリング): **28,451 MiB / 32,579**、
  実行中はほぼ一定(compute buffer は予約済み)。余裕 4.1 GB → slot は 1.88 MiB/個なので 256k でも 7,500 程度までは
  可能(+1.9 GB)。ただし他の GPU プロセス(例: reading-resolver)と同居するなら 6,500 のまま。decode 23.3 ms(42.9 t/s)。

### 起動スクリプト(運用)

`tools/flash_next_ternary/start_flash_next_server.sh`: fork の `build-q4x/bin/llama-server` を expert cache 付きで起動
(`FLASH_CTX` 既定 262144、`FLASH_SLOTS` は context に応じて 8175 / 7000 / 6500、`FLASH_HOST` / `FLASH_PORT`、
`GGML_CUDA_REGISTER_HOST=1` と CUDA 13.3 の `LD_LIBRARY_PATH` を設定、`--jinja`、`-np 1`)。追加引数はそのまま渡す
(`--reasoning-budget 2048` など)。smoke(32k): chat は `reasoning_content` に思考が出る thinking モデル、tool 定義付き
リクエストは native `tool_calls`(`finish_reason: tool_calls`)を返すことを確認、生成 60.9 t/s。

### 256k 受け入れテスト(llama-server、起動スクリプト、`q4x_eval/server_256k.py`)

| リクエスト | prefill された token | 時間 | 生成 |
|---|---:|---:|---:|
| A: 221,663 token の prompt | 221,663 | 523 s(424 t/s) | 39.6 t/s |
| B: 追記 | 5 | 0.4 s | (EOS) |
| C: 末尾付近を編集(checkpoint 復元 + 再 prefill) | 246 | 1.7 s | 43.7 t/s |
| D: B の prompt に戻す | 1,092 | 3.9 s | B と同一出力 |

VRAM peak 28,455 MiB(6,500 slot)。checkpoint は 1 個 112.6 MiB(host RAM、既定最大 32 個)。

### prefill 高速化の調査(2026-09-27)

**ubatch サイズ(32k、8,175 slot)**: ub 1024 344 t/s(この回は他プロセスの影響で遅め、通常 ~510)/ **2048: 650 t/s** /
**4096: 760 t/s**。compute buffer 1.2 / 2.0 / 3.9 GiB、VRAM peak 24.7 / 24.4 / 26.3 GB。
**256k では ub 2048 は効かない**: prefill 399 t/s(ub 1024 の 421 と同等)、compute buffer 7.5 GiB、**VRAM peak 32,049 MiB(上限
32,579)**。256k の prefill は重み転送ではなく context 比例の処理(行ごとの top-k、全長 attention)が支配的。ub 2048 は ≤128k 向け。

**prefill の Nsight(32k、ub 1024、15 s 窓)**: H2D copy 244 GiB / 6.8 s(36 GiB/s、1〜5 MiB × 75k 回 = 使用 expert のみの
コピー)、カーネル 4.0 s、**両者は全く重ならない**(union = 和)、残り 4.2 s(28 %)は起動律速(15 s で cudaLaunchKernel 758k 回、
CPU 3.3 s)。カーネル内訳: mul_mat_q 1.43 s、**DeviceTopK 1.25 s(497,664 launch)**: QSA の top-k が行ごとに逐次呼ばれている
(1024 行 × 12 層 × ubatch)。→ 行をまとめて 1 launch にする radix-select を実装(`k_top_k_select`、fork commit は git log 参照、`GGML_CUDA_TOPK_BATCH=0` で旧経路):
1 行 1 thread block、64-bit packed key を float から都度生成して 8 bit × 8 pass の radix select → k 番目以上を収集。選択集合は
per-row 版と同一(batched prefill の KL: mean 0、max は base の量子化下限)。**32k prefill 478 → 666 t/s(ub 1024、+39 %)**。

**slot save/restore(llama-server `--slot-save-path`)**: 20k token の slot は 659 MiB(≈ 32 KiB/token → 220k で約 7 GB)、
save 0.5 s、restore 0.1 s。**末尾で生成した後の保存は、短い prefix からの続きには使えない**(hybrid の状態は末尾時点のもの
だけで checkpoint は保存されない → 全再 prefill)。**prefix だけを prefill(`n_predict: 0`)して保存**すれば、restore 後に
「prefix + 追加テキスト」は prompt_n=12 / 0.5 s で続き、prefix の直後を編集する要求も checkpoint 経由で 0.5 s。
コードベースの長い prefix を一度作って使い回す運用に有効。

### llama-server での prefix 再利用(単一 slot、65k context、expert cache 8,175 slot)

| リクエスト | prefill された token | 時間 |
|---|---:|---:|
| A: 20,664 token の code prompt | 20,664 | 36.8 s(561 t/s)、生成 37.7 t/s |
| B: A + 生成結果 + 追記(append only) | **9** | 0.4 s |
| C: A の末尾 500 token を削って編集 | 533(checkpoint 復元) | 1.7 s |
| D: B の prompt に戻す | 1,064(checkpoint 復元) | 2.5 s |

fork の server は recurrent state の context checkpoint(112.6 MiB / 個、最大 32)を自動で作り、prefix が
一致しない場合は最も近い checkpoint から差分だけ再処理する。コーディング用途(最初に大きな prefill、以後は
差分 + decode)はそのまま成立する。起動例:
`GGML_CUDA_REGISTER_HOST=1 llama-server -m <IQ3_XXS> -c 262144 -b 1024 -ub 1024 -ngl 99 --expert-cache-slots 6500 -fa on -np 1`
