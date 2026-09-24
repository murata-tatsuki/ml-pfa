# Cluster 単位の calorimeter energy regression（option 3）

`--cluster-energy-pooling` を付けた multi-head model だけで有効になります。
付けない場合、既存の network・loss・checkpoint の経路を使用します。
track energy head とその loss の構成は維持します。
共通 encoder は従来どおり学習されるので、再学習後の track 出力が固定されるという意味ではありません。

## Network と教師値

calorimeter head の hit feature を cluster ごとに sum / mean / max で集約し、
`log1p(deposit 合計)` と `log1p(calo hit 数)` を加えます。
その後、MLP（128 → 64 → 1、最後に Softplus）で cluster の energy を直接予測します。
energy deposit に掛ける補正係数を予測する方式ではありません。
track hit は calorimeter pooling と deposit 合計から除外します。

`truth` は `data.y[:, 0]` の truth ID を使用します。
`predicted` は clustering head の beta・座標と track flag を使い、
既存の `objectcondensation.get_clustering_np_new` と同じ方法で所属を決めます。
event をまたいで同じ ID を集約することはありません。
所属決定には勾配を流さず、pooling・MLP・共通 encoder には勾配を流します。

予測 cluster C の教師値は以下です。

```text
target(C) = Σ_truth t E_truth(t)
            × Σ_{calo hit i ∈ C ∩ t} deposit(i)
              / Σ_{calo hit i ∈ t} deposit(i)
```

`E_truth` は従来と同じ `sqrt(mass² + px² + py² + pz²)`、
deposit は入力変換後の `data.x` ではなく `data.feat[:, 0]` を使います。
分母は、その event のモデル入力に残っている当該 truth 粒子の全 calo hit です。
predicted cluster に入らない hit も分母に含み、その分を他の cluster へ再配分しません。
truth noise（ID ≤ 0）の寄与は 0 です。deposit 合計が 0 の truth 粒子は hit 数で等分します。
負の deposit は 0 として扱います。
truth 集約では、calo hit を持つ各 truth 粒子の energy がそのまま教師値になります。

## 学習

既存の学習コマンドで、以下を指定してください。

```bash
--use-multihead-model \
--multihead-regression-heads 2 \
--energy-regression --energy-regression-cluster \
--LE-cluster sum_log_perCluster \
--cluster-energy-pooling \
--cluster-energy-source truth
```

beta・座標による集約で学習する場合は、最後の指定を次に置き換えます。

```bash
--cluster-energy-source predicted \
--cluster-energy-tbeta 0.7 \
--cluster-energy-td 0.5
```

- 学習と validation は同じ `--cluster-energy-source` を使います。省略時は `truth` です。
- `--LE-cluster` は `sum`、`sum_log`、`sum_log_perCluster`、`log_ratio_mse`、
  `log_scaled_relative` に対応します。既定値の `distribution` は新方式では使えません。
- `sum` は従来と同じ係数 5 の二乗誤差、`sum_log` は `log1p(abs(pred-target))` です。
  `sum_log_perCluster` と 2 種類の scale-invariant loss は event 内の calo cluster 数で平均し、
  さらに batch 内の event 数で平均します。他の 2 種類は cluster loss の和を event 数で割ります。
  calo cluster のない event の寄与は 0 です。
- 既存の regression 係数、energy loss を有効化する epoch、段階的な重み付けは継続します。
- 同じ構成の既存 multi-head checkpoint は `--model-ckpt` で読み込めます。
  共通 encoder・clustering・track・calo feature 部分を引き継ぎ、新しい cluster MLP を初期化します。
  新方式の checkpoint を再学習する場合も `--cluster-energy-pooling` を指定してください。
- 新方式では通常の単一 device 学習と DDP に対応します。
  `--jit`、`--dp`、`--energy-branch`、`--energy-regression-weight` との併用は明示的にエラーにします。
  新方式を無効にした既存の組み合わせには制限を追加していません。

## 推論・ROOT 出力

新 checkpoint は自動認識します。推論で `--cluster-energy-pooling` を付ける必要はありません。
通常の推論は、学習時の source にかかわらず `predicted` が既定です。
学習時に truth 集約を使っても、通常の推論時に truth ID は使いません。
既存の `--tbeta` / `--td` を pooling と最終 clustering の両方に使います。

```bash
python save_root_reco_w_Cedric_fast.py \
  INPUT.h5 CHECKPOINT.pth.tar OUTPUT.root 0 1 False 5 3 \
  --energy-regression --energy-regression-cluster \
  --cluster-energy-source predicted --tbeta 0.7 --td 0.5
```

truth での性能確認は、上記の source を `truth` にし、さらに `--truth-clustering` を指定します。
`--truth-clustering` を使い source を省略した場合も、pooling は truth になります。
source と最終 clustering を異なる設定にすると、最終 cluster の energy は下記の互換用配分値を
再集計した値になるため、比較時は両者をそろえてください。

既存の ROOT writer は hit ごとの energy を合計するため、cluster 予測 energy を
calo deposit の比率で各 hit に分配して渡します。deposit 合計が 0 なら等分します。
これは出力形式の互換処理であり、hit 単位で energy を予測しているわけではありません。
同じ所属で合計すれば、MLP の cluster energy が復元されます。track と未所属 hit の配分値は 0 です。

`save_root_reco.py`、`save_root_reco_w_Cedric.py`、`save_root_reco_w_Cedric_fast.py` に対応します。
`save_root_reco_scan_fast.py` は単一 threshold のみ対応します。
新 head の予測は所属 cluster に依存するため、1 回の推論を使い回す threshold scan は使用できません。
scan には `save_root_reco_w_Cedric_fast.py --beta-d-scan` を使ってください。
momentum 依存の別 clustering 方式との併用は対応していません。

Python から直接推論する場合は、`model.get_model(..., jit=False, ...)` で読み込み、
`cluster_energy.inference_forward(model, data)` を使います。
新方式では raw deposit が必要なので、`model(data.x, data.batch)` だけでは推論できません。

## 確認範囲

単体テストは `python -m unittest test_cluster_energy -v` で実行できます。
教師値の split / merge、event 分離、微小・ゼロ deposit、空 calo、勾配、
既存 checkpoint、新 checkpoint、情報共有の epoch 設定を確認します。
CPU で truth / predicted の学習・validation・checkpoint 保存、streaming 入力、
既存方式の学習、実 H5 の ROOT 出力、2 プロセス DDP の勾配同期を確認しました。
GPU 実行と、実データでの再学習による精度比較は未実施です。
predicted 集約は既存の CPU clustering を forward 内で呼ぶため、truth 集約より処理時間が増えます。
pretraining の既存集計は truth 単位の指標であり、predicted cluster 単位の最終性能とは区別してください。
