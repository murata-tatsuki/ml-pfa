# 新H5の学習入力とtruth未対応行

最新版 fixed_uds と同じ `pandora-eval-1` 形式の nnqq2M を学習する場合、
既存の `train.py` コマンドに次を追加する。

```bash
--extended-h5-input --exclude-gap-hits
```

truth 集約 head と組み合わせる場合は以下を併用する。

```bash
--use-multihead-model --multihead-regression-heads 2 \
--energy-regression --energy-regression-cluster \
--LE-track alpha_tracker_diff_log_perCluster --LE-cluster sum_log_perCluster \
--cluster-energy-pooling --cluster-energy-source truth \
--extended-h5-input --exclude-gap-hits
```

学習用 `-i` と validation 用 `-ii` の両方に同じ入力方針を適用する。
両方とも新形式である必要がある。H5は読取専用で、元のgap hit・全PFO情報・builderを変更しない。
再生成・結合時には `row_info` と `collections` を含む新形式のgroupを保持すること。

## 「有効な入力」の定義

- `row_info.feature_valid == 1`。
- 保存された13個のraw featureがすべて有限で、float32に変換しても有限。
- 正規化・角度追加など既存の前処理後、実際にモデルへ渡すfloat32 featureがすべて有限。

NaN、Inf、float32で表現できない値は除外する。tanhでInfが有限値に見えてしまう場合も除外する。
track state不足などで生成器がfeature_valid=0とした行も除外する。
これは数値として利用可能という定義であり、あらゆる物理的な異常値を検出する保証ではない。
追加のenergy、pT、角度、Pandora所属によるcutは導入していない。
truthの有無は入力採否に使わない。既存形式の前処理は変更しない。

gapはtruthの有無に関係なく、collection名で識別する。
`EcalBarrelCollectionGapHits` と `EcalEndcapsCollectionGapHits` のcalorimeter行を
特徴量生成・network forwardの前に除外する。GNNの学習・validation・推論で同じ除外を使うこと。
gapを除いても粒子の教師energyを差し引く処理は行わない。

## truth未対応の扱い

有効な未対応hit/trackはencoderに入力する。`truth_valid`をDataに保持する。
既知truthは正のcluster ID、未対応は0で表現するが、未対応をnoise教師として使わない。

OC・track lossの直前で、forward済みの予測と教師値を既知truth行に絞る。
これにより未対応行は引力・反発・beta・track energyなどの教師あり項目の対象にならない。
network入力からは除いていないため、近傍featureや共通encoderを通じて既知行の予測には影響する。
未対応行の内部featureへの間接的な勾配まで止める処理ではない。

truthがないeventは教師ありlossへの寄与を0とし、batch全体のevent数で正規化する。
既知行のないrank/batchでも微分可能な0を返す。

### calorimeter pooling

- `truth`：既知truthのcalo hitだけを各truth clusterに集約する。未対応行の所属は推測しない。
- `predicted`：全有効入力のbeta/座標でclusterを作り、そのcalo hitを集約する。
  **truth未対応のcalo hitを1つでも含むclusterはenergy lossから除外**する。
  未対応部分をenergy=0とした不完全な教師値を使わない。
  未対応trackだけを含むことを理由にはcalo lossを除外しない（trackはcalo poolingに含めないため）。

predictedでloss対象外になったclusterも予測・出力される。
per-cluster平均は教師値が完全なcalo clusterだけで行う。
有効なcalo教師clusterがないeventの寄与は0で、batch内の全event数で平均する。
不完全clusterが多いと学習に使えるcalo教師が減るので、truth集約との比較が必要。
既知truth行だけから作る教師は、利用可能なラベル範囲での教師である。

## loaderと互換性

`--ilc-streaming` ではファイルchunk・shuffle buffer・rank/worker分割の既存設定を利用する。
それ以外は新形式のsharded読込を使い、全H5を一括でメモリに連結しない。
空になるeventはloaderが除外するが、truthがないことを理由にはeventを除外しない。
極少数hitのeventが既存networkの近傍探索/BatchNormの要件を満たさない場合の対応は別途必要。

新形式モードの読み込み時の `--timing-cut` は非対応。事前に下記の `timingcut.py` で学習専用H5を作成できる。
`--mctpe`、`--jit`、`--dp`、`--energy-branch`、`--energy-regression-weight` も併用しない。
通常の単一deviceとDDP、既存のmono/multi-head、cluster poolingに対応する。
新フラグを付けない既存H5の学習経路は維持する。
`--exclude-gap-hits` だけを学習に指定した場合は、識別情報不足を避けるためエラーにする。

新モードのcheckpointには `training_input_config` を保存する。
再学習時の入力方針はCLIで明示する。推論のgap除外は自動切替しないので、下記flagを指定すること。

## 評価

新H5の評価は `save_root_pandora_eval.py ... --exclude-gap-hits` を使用する。
学習と評価は共通のgap識別・特徴量の有効性判定を使う。
Pandoraは保存された全PFOを使い、GNN入力のgap除外ではPFO energyを変更しない。
truth未対応入力を含む評価と、既知truthだけの診断を区別する。

## 検証

```bash
python -m unittest test_extended_training test_cluster_energy test_pandora_eval_analysis -v
```

gap除外・未対応行保持・入力と評価の一致・H5非変更・未知予測値へのloss不変性・
未知行への直接の教師勾配が0・空教師batch・不完全predicted clusterのloss除外を確認する。

実H5の16イベントで学習と評価の入力一致、および実イベントのforward/loss/backwardを確認した。
CPUで2 workerのstreaming、truth/predicted pooling、既存headの5 epoch学習・validation・
checkpoint保存を確認した。2プロセスGloo DDPでは片方のrankがtruthなしの条件で、
beta/energy lossのepoch切替とパラメータ同期を確認した。
GPUでの動作と実データ再学習後の精度向上は未確認。


## timing cut済みの学習専用H5

```bash
python timingcut.py -i /path/to/full.h5 -o /path/to/tc_training.h5 \
  --maximumTime 14 --minimumPt 0.3
```

新形式の出力groupは `feature`、`label`、`row_info`、`collections`、`event` の5つのみ。
`schema_version` と `metadata` を含む元のファイル属性を保持し、
`training_only` と実際のcut条件を記録した `timing_cut` 属性を追加する。
不要なPandora/PFO等のgroupは読み込まず、出力にも保存しない。
削除行・範囲外イベントの内部バッファを詰め直し、gzip level 1で圧縮する。
元H5は変更せず、同一ファイルへの出力は禁止する。

条件は従来と同じ `feature[:,4] < maximumTime`、かつcalo行または
`pT = sqrt(feature[:,7]**2 + feature[:,8]**2) > minimumPt`。
境界値と等しい行は除外する。教師ラベルのenergyや粒子IDは変更しない。
`feature`・`label`・`row_info`を同じ行maskで選別し、
`collections`・`event`はイベント範囲だけを揃える。
`--nstart`は開始イベント、`--nend`は終了イベントindex（含まない、-1は末尾）。
空イベントは保存し、学習ローダー側で除外する。
truth未対応やgapという理由で追加の削除はせず、学習時の既存の処理に任せる。

旧形式は `feature`・`label` と、存在する場合の `event` のみを保存する。
旧形式でも `pandora` は保存しないため、出力は学習専用として使用する。

学習・validationの両方を同じ条件で前処理し、出力H5に対して
`--extended-h5-input --exclude-gap-hits` を指定する。
`train.py` に `--timing-cut` を追加する必要はない。
この軽量H5はPandora比較用評価ファイルには使えない。比較評価には元の全group入りH5を残す。
GNN評価でも同じtime/pT条件を適用する必要があり、現時点で評価側へcut条件が自動反映されるわけではない。

検証: `python -m unittest test_timingcut -v`
