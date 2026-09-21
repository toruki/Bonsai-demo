# Layer 32 の ternary-aware reconstruction training(Qwen3.8-27B)

目的: 単純 PTQ で崩壊した ternary 層を、BF16 teacher の block 出力に合わせて
**ternary 制約下で再学習**したとき、どこまで戻るか。そして出荷 Bonsai 2 の code に近づくか。

コード: `tools/flash_next_ternary/recon/{ternary_layer.py, train_recon.py, train_block.py}`
データ: BF16 GGUF から `dump_hidden --n-chunks` で採取した residual stream(wiki.train 160 seq × 512 token、
valid は wiki.valid 32 seq)。

## 0. セットアップの検証

| 項目 | 結果 |
|---|---|
| transformers の `Qwen3_5DecoderLayer`(fp32)vs llama.cpp BF16 の `l_out-32` | full cos 1.000000、relMSE 3.8e-7 |
| 出荷 Bonsai 2 の layer 32 を同じ層クラスに載せ直す(`TernaryLinear.from_folded`) | refold 誤差 0、code self-agree 1.0000 |

層実装は runtime と一致し、出荷層も誤差ゼロで再現できるので、以下の比較は同じ土俵です。

学習対象: latent weight(382.7M)、group scale(2.99M)、RMSNorm(10k)。
freeze: in_proj_a/b、conv1d、A_log、dt_bias。quantizer は round-to-nearest + LSQ 型 STE、
活性側は runtime と同じ `H(s⊙x)`。loss = block delta の rel-MSE + (1 − cos)。

指標の注意: block 出力は residual を含むので `full cos` は 0.99 台に張り付く。
層の寄与そのものを見る **`delta` = y − x** の cos / relMSE が本質。

## 1. 最初の驚き: 出荷 Bonsai の layer 32 は BF16 層を単体では再現していない

BF16 の residual stream を入力にして BF16 層出力と比べる(valid 32 seq):

| 層 | delta cos | delta relMSE | full cos |
|---|---:|---:|---:|
| PTQ-mseopt(学習前) | 0.727 | 0.467 | 0.9897 |
| **出荷 Bonsai 2 layer 32** | **0.747** | **0.444** | 0.9901 |

出荷層は PTQ 初期値と大差ない。出荷モデルの residual stream 自体が BF16 から
ずれている(層 31 で cos 0.90 / relMSE 0.21、`docs/ptq_vs_shipped_27b.md`)ため、
**出荷モデルは「各層が BF16 層を個別に再現する」形ではなく、内部表現ごと動いた状態で
end-to-end に整合している**。したがって「単層再構成で出荷 code に近づく」は
そもそも起きにくい設定だった。

## 2. 結果(valid、600 step、bs 4 × 512 token、AdamW、cosine 減衰)

| run | 初期値 | lr(w/s/norm) | step | delta cos | delta relMSE | code 変化(対初期値) | 出荷 code 一致 |
|---|---|---|---:|---:|---:|---:|---:|
| 学習前 | PTQ-mseopt | — | 0 | 0.727 | 0.467 | 0 % | 85.5 % |
| **A** | PTQ-mseopt | 2e-4 / 1e-4 / 1e-4 | 600 | **0.882** | **0.225** | 14.5 % | 82.2 % |
| B | 出荷 Bonsai | 同上 | 600 | 0.865 | 0.253 | 9.8 % | 90.2 % |
| C(対照) | PTQ-mseopt、**code 固定** | scale/norm のみ | 600 | 0.837 | 0.300 | 0 % | 85.5 % |
| D | PTQ-mseopt | 5e-4 / 2e-4 / 2e-4 | 1500 | 0.844 | 0.291 | 28.6 % | 71.1 % |
| E | PTQ-mseopt | 2e-4、1500 step | (750 で停止) | 0.841 | 0.296 | 21 % | 77.9 % |

観察:

1. **単層でも delta relMSE 0.467 → 0.225(cos 0.727 → 0.882)まで戻る。** ただし 0.99 には遠い。
2. **code を動かすことが効く**: code 固定(C)は 0.300 で飽和、code 可変(A)は 0.225。
   差 0.075 が「5–10 % の code 変更」の寄与。ただし初期の改善の大半(0.467→0.30)は scale + norm だけで得られる。
3. **出荷 code は BF16 単層再現の意味では良い初期値ではない**(B < A)。B は出荷から 10 % 離れつつ改善する。
4. **出荷 code には近づかない**: A は 85.5 % → 82.2 % と離れる。§1 の理由による。
5. **lr 5e-4 は code を無駄に暴れさせる**(29 % 変化で A より悪い)。2e-4 が妥当。
6. **160 seq(82k token)では過学習**: E は train 0.18 に対し valid 0.30 で停滞、code だけ動き続けた。
   386M param に対して data が足りない。→ 640 seq へ増強中。
7. **zero率は学習で 45.7 % → 39–43 %(A)、33–39 %(D)へ下がる。**
   出力再構成の目的関数は weight-MSE 最適点より少ない zero を好み、
   出荷 Bonsai の 32.8 % に近づく方向。「absmean 型の動作点は学習の結果として現れる」という
   逆解析の推測と整合する。

tensor 別の出荷 code 一致(run A 終了時): in_proj_qkv 0.862 / in_proj_z 0.841 / out_proj 0.821 /
gate 0.807 / up 0.821 / down 0.777。MLP 側、特に down_proj が最も動く。

## 3. 解釈

- 「ternary 解は一意ではない」が確認された: 出荷とは別の code で同程度(むしろ良い)単層再現に到達する。
- 一方、単層 × BF16 teacher の設定は出荷モデルの学習設定と違うので、
  出荷 code との一致率を「再現度」の指標にするのはこの設定では不適切。
  出荷に近づけたいなら、teacher を「出荷モデル自身の residual stream」にするか、
  end-to-end(logits)の蒸留にする必要がある。
- 実用上重要なのは「BF16 からどれだけ戻せるか」であり、その意味で
  **block 単位(誤差を含んだ入力で下流層を学習)** と **データ増強** が次の 2 手。

## 4. 次

- 640 seq で run A を再実行(過学習の切り分け)。
- `train_block.py` で layer 32–35 を block として学習(入力は BF16 l_out-31、目標は BF16 l_out-35、
  中間 exit の補助 loss は任意)。4 層 fp32 + AdamW で ~25 GB。
