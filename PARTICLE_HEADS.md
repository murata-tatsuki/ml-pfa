# PID headと5種類のenergy head（opt-in）

既存の学習コマンドに新オプションを付けなければ、モデル・loss・checkpoint形式は従来どおり。
`--use-multihead-model` が必要。PIDと5回帰headは独立に切り替えられる。

| 設定 | 追加するオプション |
| --- | --- |
| PIDのみ追加、既存energy headを維持 | `--pid-head` |
| 5回帰head、PIDなし（学習・truth PID診断用） | `--five-particle-energy-heads` |
| PIDと5回帰head、通常推論対応 | `--pid-head --five-particle-energy-heads` |

両方を使う場合、既存の入力・学習率・epochなどの指定に以下を追加する。

```bash
--use-multihead-model \
--energy-regression --energy-regression-cluster \
--LE-track alpha_tracker_diff_log_perCluster --LE-cluster sum_log_perCluster \
--pid-head --five-particle-energy-heads --pid-loss-weight 1.0
```

`--five-particle-energy-heads` は回帰head数を自動的に5にする。
`--output-dimension` はこれまでどおりclustering側の次元で、PIDの5 logitsやenergyの5出力を足さない。
クラス順は **0 charged_hadron / 1 electron / 2 muon / 3 photon / 4 neutral_hadron**。
headの重みは全nodeで共有し、それぞれのnodeに5 energy値と5 PID logitsを出す。
5回帰headはSoftplusによる非負出力、PIDはsoftmax前のlogits。

## 学習する値

| truth粒子種 | energy予測 | PID教師node |
| --- | --- | --- |
| charged hadron / electron / muon | truth object内の最高beta **track**の対応head値 | 同じtrack alpha |
| photon / neutral hadron | truth object内の**calo nodeだけ**の対応head値の和 | trackがあれば最高beta track、なければ最高beta node |

対応するenergy headだけに `log(1 + abs(E_pred - E_truth))` をかける。
非対応headへの0教師は追加しない。PIDは上記alphaで5クラスcross entropy。
alpha選択にはdetachしたbetaを使い、選択操作自体には微分しない。
energy教師は従来と同じ `sqrt(mass² + px² + py² + pz²)`（GeV）。
粒子種はtruthのPDGとchargeで決め、trackの有無でtruth粒子種を変更しない。
electron/muonは反粒子も同じクラス、整数のabs(PDG)>=100をhadronとしてchargeで分類する。
未対応PDG、無効energy、truth不明・noiseのnodeは補助lossの教師から除外する。
荷電truthでtrackがなければenergy lossだけ除外し、PIDは最高beta nodeで学習する。
neutral truthでcaloがなければenergy lossだけ除外する。

lossは、各event内の有効truth object平均を取り、次にbatchの全eventで平均する。
energyとPIDはそれぞれの有効object数を分母にする。教師のないeventの寄与は0。
energyは5クラスをまとめて平均し、クラスごとの平均lossを単純に5つ足す方式ではない。
`--regression-coefficinet`、`--epochs-noLE`、`--LE-gradually` はenergyに引き続き適用する。
PIDには `--pid-loss-weight`（既定1）と `--epochs-noPID`（既定-1、epoch 0から有効）を適用する。
OC/beta lossの設定・開始epochは従来どおり。

ログの `L_E_<species>` は足すと `L_E` になる寄与量。
`L_PID` は重み・開始epoch適用後、`L_PID_unweighted` は適用前。
`PID_accuracy_event_mean` はevent平均正解率で、教師のないeventの寄与は0。
`N_PID` と `N_E_missing_track` はbatch当たりの教師数・trackなし荷電truth数（epoch表示ではbatch平均）。

## 推論と出力

### 現在のclustering + 代表点PIDの基準評価

ROOT出力の `--pid-readout representative` で、学習済みPID headを必須にし、
既存のbeta/座標clusteringの後に代表点PIDを読む。trackがあれば最高betaのtrack、
なければ最高betaのnodeの5クラスargmaxを使用する。hit間のPID不一致でclusterを
分割・結合しない。`save_root_pandora_eval.py` の `auto`（既定）は
PID headがあれば同じ読み出しを使用する。
head構成はcheckpointから復元するため、推論時の `--pid-head` や
`--five-particle-energy-heads` の再指定は不要。5 energy headでPIDなしの通常推論は拒否する。

既存の `save_root_reco_w_Cedric_fast.py` に `--pid-readout representative` を指定して
今回のnnqq学習用H5（`nnqq-2m-eval-1`、training-only）を評価できる。
`single-particle-eval-1` と `pandora-eval-1` の学習用H5も読める。
このコマンドの既定値は `--pid-readout legacy` で、従来の5-tree出力を使用する。
representative指定時は、このファイル内のstructured writerでPID評価ROOTを作成する。

```bash
python3 -B save_root_reco_w_Cedric_fast.py \
  /data/suehara/mldata/pfa/murata/data/tc/tc_nnqq_2M_nobrems/validation/dd_ft0_99.h5 \
  checkpoint/energy_regression/ckpts_gravnet_new02_2026_10_04_162529_outputD5_multihead/ckpt_65_1.pth.tar \
  /home/murata/pid_baseline_epoch65.root \
  0 10 False 7 5 \
  --pid-readout representative --energy-regression --energy-regression-cluster \
  --momentum --momentum-amp --exclude-gap-hits \
  --tbeta 0.9 --td 0.5 --device cpu
```

位置引数は従来どおり `datapath ckpt outfile nstart nend timingCut input_dim output_dim`。
representativeモードの `timingCut` は `False` にする。`nstart` は開始index、
`nend` はそこから読む件数（従来のtimingCut=Falseの意味を維持）、`-1` は末尾まで。
directoryやglobはソートしたファイル全体で連続したindexを使用する。
`--multi-h5` はH5ごとに同じ範囲を評価し、出力directoryに `<H5 stem>.root` を作る。
modelは一度だけロードして再利用する。同名stemの衝突は拒否する。
既存PID出力は `--replace` で明示的に上書きできる。
入力H5の保存済みtime/pT cutを
そのまま使用し、gap除外・feature validity・特徴量変換は学習と共通。
detector one-hotを使った学習なら `--detector-one-hot` も合わせる。
2 energy head + PIDにも対応し、その場合は従来のenergy計算を使用する。
head構成はcheckpointから復元する。`--pandora`、`--truth-clustering`、
`--1tomany-clustering`、`--beta-d-scan`、`--energy-regression-weight`、
`--mctpe`、`--energy-branch`、truth poolingとの併用は拒否する。
`--pid-detail clusters` でhit出力を省略、`--pid-threads` でCPU thread数を指定できる。
代表点PID経路も既存fast版と同じC++ writerを使用し、hitはNumPy列配列から一括Fillする。
`--root-chunk-events` は書き込みをまとめるイベント数（既定10）。ROOTはLZ4圧縮を使用する。
`--pipeline-depth` はGPU推論の先読み件数（既定40、0で無効、CPUでは無効）。
推論結果をCPU配列として保持し、イベント順を維持してCPU clustering・評価・ROOT書き込みと
次イベントのGPU推論を並行処理する。PID・energyの計算式とROOT branchは同じ。

### epoch 57の並列shell

`shell/run_root_parallel_5head.sh` は従来と同じ `INDEX INPUT.h5 [GPU]` を受け取る。
checkpointは `ckpts_gravnet_new02_2026_10_04_162529_outputD5_multihead/ckpt_57_1.pth.tar`。
渡したH5の形式を確認し、そのファイルを直接、既存writerの代表点PIDモードへ渡す。
shell内でtime・pT cutを追加しない。一時H5も作成しない。

```bash
cd /home/murata/master/shell
partType=ss
energy=350
GPU=0
ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_01?.h5 \
  | nl -v0 \
  | xargs -r -n2 -P10 env RUN_GPU="$GPU" FAST_SCAN_WORKERS=3 bash run_root_parallel_5head.sh
```

出力は `output/energy_regression_1to1/skimmed/tc_nnqq_2M/5D/E_regression/5head_pid/2026_10_04_162529/epoch57/<energy>GeV/tbeta090td050/<H5 stem>.root`。
tbeta=0.9、td=0.5は固定でscanは実行しない。`FAST_SCAN_WORKERS` はこのshellでは
PID用CPU thread数の別名（`PID_THREADS` の指定が優先）。GPU指定がなければindexでGPUへ分配する。
`ROOT_NEND=1` で先頭1イベントの確認、`ROOT_OUTPUT_BASE` で出力root directoryの変更、
`ROOT_REPLACE=1` で再生成、`PYTHON_BIN` でPython環境の指定ができる。
`run_root_parallel_fast.sh` と同じく `ROOT_CHUNK_EVENTS=10`、`PIPELINE_DEPTH=40` が既定。
環境変数で変更でき、`PIPELINE_DEPTH=0` なら先読みを止めて同じ結果を逐次処理で確認できる。

提示の旧H5は `--pid-input-format legacy` で読み、従来のfeaturizationと同じく
MC未対応行をモデル入力から除外する。collection情報がないためgap除外・detector one-hotは不可。
新形式H5では `--pid-input-format extended` とgap除外を使用し、MC未対応入力を保持する。
この差をROOT metadataの `input_format`/`input_scope` に、旧形式の除外行数を
`eval_events.n_unlabelled_inputs_excluded` に保存する。旧H5を新schemaへ書き換えない。

- `eval_clusters`：`algorithm=1`、予測`pid`、`pid_seed_input_row`、5クラスの
  `pid_probability_<species>`、`energy`、`energy_head`、`energy_fallback`。
- `eval_hits`：元H5の各行、入力mask・cluster所属、各PID logit、5 species energy。
  除外された行の予測値はNaN。`--pid-detail clusters` では省略する。
- `eval_matches`：truth粒子ごとのbest deposit overlap match、`truth_pid`、予測`pid`、
  `pid_correct`、`efficiency`、`purity`、`response=E_reco/E_truth`。
  未マッチは`matched=0`・`pid=-1`、判定不能は`pid_correct=-1`。
- `pid_confusion`：truth PIDを横軸、予測PIDを縦軸にした生カウント。
  未マッチは専用bin。truth不明・未対応PDGは集計しない。
- `particle_eval_metadata`：読み出し方式、入力cut情報、head構成、checkpoint/code hash。

truth matchは既知モデル入力に対する粒子単位の診断で、一対一対応ではない。
同じreco clusterが複数truth粒子に対応することがあり、confusionもtruth粒子単位。
track-only truthはinput overlapでmatchし、depositの分母が0ならefficiency/purityはNaN。
`eval_events.known_truth_energy` は既知入力に現れたtruth粒子の部分和で、全event energyの
教師ではない。このwriterのschemaは `particle-baseline-root-1` で、Pandora比較マクロや
旧5-treeマクロに渡す形式ではない。

全group入りのPandora評価H5を使う従来の `save_root_pandora_eval.py` にも
`--pid-readout representative` を追加した。そちらのROOT schemaは従来どおり。

`save_root_pandora_eval.py` のstructured readoutを使う。checkpointからhead構成を復元する。
入力形式はこのwriterが従来対応するPandora評価H5。通常の学習H5を直接渡すwriterではない。
PIDは予測clusterのtrack優先alphaで**常に5クラス全部**からargmaxする。
荷電PIDはtrack alphaのenergy、neutral PIDはcalo和を使う。
trackなしで荷電PIDになった場合はPIDを保持し、photon/neutral_hadronのうちPID確率が高いheadのcalo和を使う。
同確率ならphotonを使う。neutral PIDなのにcaloがなければ和は0。
truth情報を通常推論のPID・代用判定には使わない。

`eval_clusters` に `pid`、`energy_head`、`energy_fallback`、`pid_seed_input_row` を保存する。
`pid` と実際に使った `energy_head` は代用時に異なる。Pandoraのこれらの列は-1。
PIDのみ有効の場合、energyの計算は従来どおりで `energy_head=-1`、`energy_fallback=0`。
5回帰headの場合 `energy_alpha` / `energy_any_track` は同じspecies別energy。
`eval_hits` には `energy_<species>`、`pid_logit_<species>` を各optionに応じて追加する。
5回帰headの旧 `tracker_energy` / `calo_energy` 列はNaN（2列への暗黙変換はしない）。
クラス順、代用方式、head構成、interaction epochをROOT metadataにも保存する。
algorithm=2のtruth groupingもPIDは**予測PID**。truth PIDによる診断とは異なる。

5回帰head・PIDなしで通常推論を実行するとエラーにする。
従来の2 energy列を前提としたROOT writerも、5回帰headでは明示的にエラーにする。
truth grouping + truth PIDのenergy診断は以下でCSVに保存できる（PID head不要）。

```bash
python diagnose_particle_heads.py INPUT_H5_OR_DIR new_response.csv \
  --checkpoint checkpoint.pth.tar \
  --thetaphi --momentum --momentum-amp \
  --extended-h5-input --exclude-gap-hits --stop 100
```

特徴量オプションは学習時に合わせる。旧形式H5なら最後の新H5用2オプションを外す。
CSVではtruth headのenergyを読み、trackなし荷電truthなどは `energy_valid=0`、energy/responseはNaN。
新しい出力名が必要。`--pretraining` の応答指標もtruth grouping + truth headを使用する。

## Cluster energy集約・追加粒子score（独立したopt-in）

既存の5-head + PID学習に、次をそれぞれ追加できる。両方とも省略すると、上記の
モデル構成・初期化・出力・loss・checkpoint形式を維持する。

| 目的 | 追加するオプション |
| --- | --- |
| cluster全体からcalo energyを直接回帰 | `--particle-cluster-energy` |
| trackで数えた粒子の断片と追加粒子を区別 | `--particle-cluster-score` |
| 両方を同時学習 | `--particle-cluster-energy --particle-cluster-score` |

別networkの事前学習・backbone固定は行わない。同じGNNのhit embeddingをclusterごとに
sum/mean/max集約し、測定ECAL/HCAL/その他/不明collectionのenergy和、hit数、energy分散、
位置・形状、近傍trackとの距離・運動量を加えたMLPからenergyとscoreを出す。
GNNと追加headは初回から同時に学習する。collection情報がない旧H5は不明channelを使用する。
モデル入力にMC energy・PID・truth所属は追加しない。truth所属は教師用の集約とtarget作成にのみ使う。

既存の入力・学習率などの指定に、例えば次を追加する。

```bash
--use-multihead-model \
--energy-regression --energy-regression-cluster \
--LE-track alpha_tracker_diff_log_perCluster --LE-cluster sum_log_perCluster \
--pid-head --five-particle-energy-heads \
--particle-cluster-energy --particle-cluster-score \
--particle-cluster-source truth --particle-cluster-score-weight 1.0
```

`--particle-cluster-source truth`（既定）は正のtruth IDで集約する。
`predicted` は既存のbeta/virtual coordinate clusteringで集約し、欠損・混入を含む候補を学習する。
その場合の閾値は `--particle-cluster-tbeta 0.9 --particle-cluster-td 0.5`（既定）で指定する。
所属決定は離散処理なので微分しないが、集約とenergy/score lossから共通GNNへの勾配は流れる。
高β点への最近傍割当、低β点のβ優先順割当、単独trackの追加結合の規則を保つ。
CUDA上の`predicted`集約はTorch版を自動使用し、β・virtual coordinate・所属をCPUへ転送しない。
距離計算は一時領域約64 MiBを目安に分割する。CPU実行時は従来のNumPy版を使用する。
GPU版のβ完全同値時は入力hit番号順に固定するため、旧NumPyの不安定sortとは順序が
異なる場合がある。また、`td`付近ではCPU/CUDA間の距離の丸め差があり得る。
追加optionは不要で、truth集約と集約option無効時の学習は従来どおり。

### Energyの教師と採用値

energy出力はcluster当たり1つの非負calo energy。photonとneutral hadronで共有する。
予測clusterには、各truth粒子のMC energyを**その粒子のcalo deposit割合**で配分する。
分母には未cluster hitも含め、分割されたneutralでMC energyを重複計上しない。
対応する入力trackが別clusterにある荷電粒子のcalo断片はenergy教師0とする。
入力trackがない荷電粒子のdepositは残す。対応trackを含むclusterでは荷電粒子のcalo energyも
補助教師として学習する。truth不明hitを含むclusterは、完全な教師を作れないため補助lossから除く。
MC親子関係は現行入力にないので、異なるMC ID間の親子の重複判定には使わない。

`--particle-cluster-energy` 有効時は、従来の荷電3 headのlossを保ち、中性2 headのnode和lossを
集約energyの `log1p(abs(E_pred - E_target))` に置き換える。
荷電項の分母は従来どおり全有効truth object数、集約項の分母は有効calo cluster数とし、
それぞれevent平均して足す。従って、option間でtotal lossの数値を直接性能比較しない。
energy係数・開始epoch・rampは既存の `--regression-coefficinet`、`--epochs-noLE`、
`--LE-gradually` に従う。`L_E_cond` は荷電項、`L_E_cluster` / `L_E_particle_cluster` は集約項。
このoptionでは `L_E_photon` / `L_E_neutral_hadron` は0で、5 species寄与だけを足してもtotalにはならない。
`--pretraining` の集約候補のresponseはPDG=0で記録する。

推論では、truth/reco/scanで**最終所属が決まった後**に同じheadで集約し直す。
荷電PIDかつ1 trackなら測定運動量とPID質量からenergyを計算し、複数trackの荷電PIDは従来のhead値を保つ。
neutral PID、またはtrackなし荷電PIDのneutral fallbackには集約energyを採用する。
5 speciesのraw head値は診断用に保持するが、このoptionの中性raw headは採用energyではなく、
そのnode和の回帰lossも学習しない。

### 追加粒子scoreの教師と選別

scoreは、trackなし候補のdepositのうち「入力trackで既に数えられる荷電粒子の断片ではない」
割合を0〜1で学習する。neutralと入力trackなし荷電は1、入力trackを持つ荷電の断片は0、
混入候補はdeposit割合のsoft targetとし、BCEWithLogitsをevent平均する。
truth所属では断片の負例が現れないため、track付きtruth粒子のcalo hitを物理座標の最大幅方向で
2群に分け、trackを外した候補も作る。元のneutralを重複して数えず、これらの負例をepoch 0から学習する。
追加項は `L_particle_score`、係数は `--particle-cluster-score-weight`（既定1）。
energyの開始epochとは独立にepoch 0から有効。PID・OC/betaのlossは維持する。

推論時はscore付きcheckpointからheadを自動復元する。統合writerの例：

```bash
python3 -B save_root.py INPUT.h5 NEW_CHECKPOINT.pth.tar OUTPUT.root \
  0 -1 False 7 5 \
  --momentum --momentum-amp --pid-readout representative \
  --tbeta 0.9 --td 0.5 --particle-cluster-score-threshold 0.5
```

入力特徴量の指定は学習に合わせる。`--truth-clustering` とbeta/distance scanにも同じ読出しを適用する。
`--particle-cluster-score-threshold` の省略時は、score付きcheckpointに限り0.5。
scoreを学習していないcheckpointへの閾値指定はエラーにする。
閾値未満の**trackなしclusterだけ**採用energyを0にし、track付き候補は保持する。
scoreをenergyへ掛けたり、hitを他clusterへ移したりはしない。
新headのROOTには `cluster_score` / `cluster_accepted` を追加し、棄却候補も所属・raw出力を診断用に残す。
`energy_pid`、採用hit寄与、event energyは選別後の値を使用する。energyのみの場合score列はNaN、採用は1。
閾値はvalidationでJERと平均energy、neutralの採否を合わせて比較する。

通常推論に追加の学習flag指定は不要。旧checkpointから `--model-ckpt` で開始すれば追加headだけ新規初期化し、
共通GNNなどを引き継ぐ。新head付きcheckpointをそのheadの学習flagなしで読み込むことは拒否する。
energyのみ/scoreのみから両方への追加も可能。単一processとDDPに対応し、既存の
`--cluster-energy-pooling`（2-head用）とは別機能で併用しない。

## Checkpointと対応範囲

新構成は `particle_heads_config` をcheckpointに保存し、粒子種順の欠落・不一致を拒否する。
`--model-ckpt` で旧multihead checkpointから開始するときはshared encoderとclusteringを引き継ぐ。
旧track headをcharged/electron/muon、旧calo headをphoton/neutralに複製し、PIDは新規初期化する。
旧checkpointにないheadは新規初期化する。既存の読込機能と同様、optimizerのresumeではない。
2種類に戻して5種類のcheckpointを読み込むことはできない。
新オプションなしで旧checkpointを読む経路は変更しない。

単一processとDDPに対応。`--jit`、`--dp`、`--energy-branch`、
`--energy-regression-weight`、`--cluster-energy-pooling`との併用は未対応として拒否する。
過去の検証コマンド: `python -m unittest test_particle_heads test_cluster_energy test_extended_training test_pandora_eval_analysis -v`。
確認用テストソースは2026-10-17に削除済みで、このコマンドは削除前の記録です。
回帰精度やjet energy resolutionの改善は、この実装だけでは保証されないため比較学習で評価する。
