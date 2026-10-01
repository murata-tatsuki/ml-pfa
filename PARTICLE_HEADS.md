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

## Checkpointと対応範囲

新構成は `particle_heads_config` をcheckpointに保存し、粒子種順の欠落・不一致を拒否する。
`--model-ckpt` で旧multihead checkpointから開始するときはshared encoderとclusteringを引き継ぐ。
旧track headをcharged/electron/muon、旧calo headをphoton/neutralに複製し、PIDは新規初期化する。
旧checkpointにないheadは新規初期化する。既存の読込機能と同様、optimizerのresumeではない。
2種類に戻して5種類のcheckpointを読み込むことはできない。
新オプションなしで旧checkpointを読む経路は変更しない。

単一processとDDPに対応。`--jit`、`--dp`、`--energy-branch`、
`--energy-regression-weight`、`--cluster-energy-pooling`との併用は未対応として拒否する。
検証コマンド: `python -m unittest test_particle_heads test_cluster_energy test_extended_training test_pandora_eval_analysis -v`。
回帰精度やjet energy resolutionの改善は、この実装だけでは保証されないため比較学習で評価する。
