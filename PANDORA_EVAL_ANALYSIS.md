# 新H5のPandora・GNN比較

`pandora-eval-1` H5専用の解析経路。既存H5・元の評価マクロは変更しない。
新H5を旧 `save_root_reco_w_Cedric_fast.py` に渡すと、新コマンドを案内して停止する。
旧ROOTを新評価の結果として再利用しない。

## 実行

リポジトリのディレクトリから実行する。指定されたmulti-head / epoch 27の設定例:

```bash
python -B save_root_pandora_eval.py \
  /home/murata/simulation/validation/pandora_eval/final/dd_350_sample.h5 \
  output/pandora_eval/dd_350.root \
  --checkpoint checkpoint/energy_regression/ckpts_gravnet_new02_2026_08_17_174838_outputD5_multihead/ckpt_27_1.pth.tar \
  --device cpu --input-dim 7 --output-dim 5 \
  --momentum --momentum-amp --calo-head --tbeta 0.9 --td 0.5 \
  --regression-output-activation linear
```

`--device cuda:0`でGPUを使える。既存checkpointを使い、再学習しない。
`input-dim=7`は角度を含む基本特徴量数。momentumの3成分と大きさを加え、実際の入力は11次元。
`output-dim=5`はbetaと座標の出力数。tracker/calo回帰の2列を加える。
`--model-variant auto`でlegacy/multiheadを判定する。cluster-energy-poolingモデルはGNN用に
predicted membership、truth診断用にtruth membershipで別々にforwardする。

Pandoraのみの確認ではcheckpoint不要:

```bash
python -B save_root_pandora_eval.py NEW.h5 pandora_reference.root --pandora-only
```

H5のディレクトリや引用符付きglobも入力可能。同じ衝突エネルギー・設定のファイルを指定する。
`--start 0 --stop 4`は最初の4イベント（stopは排他的）。
旧H5照合の有無によらず、source ID/source indexで追跡する。`legacy_index=-1`で除外しない。
既存ROOTは上書きしない。明示的な再生成には `--replace` が必要。
出力は一時ファイルで作成・検査してから公開する。推論失敗はデフォルトで処理を中止し、
不完全なROOTを公開しない。`--keep-failed-events`指定時は失敗理由とNaNを保存して続行する。

## 評価マクロ

元の `efficiency_purity_check_reco_effpur_contiribution.cxx` をコピーして新スキーマ向けに修正した
`macro/src/efficiency_purity_check_reco_effpur_contiribution_pandora_eval.cxx`を使う。
旧ROOTの枝・ファイル一覧を使った集計は新スキーマでは実行しない。

```bash
root -l -b -q 'macro/src/efficiency_purity_check_reco_effpur_contiribution_pandora_eval.cxx("output/pandora_eval/dd_350.root","output/pandora_eval/dd_350_performance.root")'
```

入力ROOTは同一エネルギーのglobを指定してもよい。source eventの重複と比較設定の混在を検査する。
評価結果の出力先も新しいファイル名を指定する。

- `qPdg=1..3`。中央領域は`thrust<=0.7`、`E_visible + mcEnergyENu`。
- 角度別はupstreamの13区間、**ニュートリノ補正なし**。
- `thrust`はH5のquark energy-weighted |cos(theta)|。event thrustへの置換はしない。
- エネルギースペクトルはupstreamのAnalysePerformanceと同じ100000 bins、0–5000 GeV。
- `sqrt(2)*RMS90/Mean90*100`と誤差は、LCPandoraAnalysisの未変更のAnalysisHelperで計算する。
  `E_reco/E_truth`分布の幅には置換しない。
- 5イベント未満、underflow/overflowがある分布は分解能を有効値にしない。
  `resolution`ツリーの`valid`を確認する。空の分布を分解能0として描画しない。

`macro/src/pandora_reference/`にはcommit
`61b8993d121e84efa07bd74678b2ef687597ba77`の`AnalysisHelper.cc/.h`を未変更で収録した。
元のLICENSEも同ディレクトリに収録している。

## 入力とtruth

`dataset.py`の共通前処理に、`row_info`を受け取る経路を追加した。
旧呼び出しの前処理は維持する。新経路は次の規則を使う。

- feature_validかつ前処理後の特徴量が有限な行をGNN入力にする。
- MC対応の有無、Pandora所属、legacy_model_domainでGNN入力を選別しない。
- timing/pT cutを追加しない。MC track momentumをモデル入力にしない。
- track判定はrow_info.kind、整数IDはrow_info.signed_object_id。
- 並べ替え後も元H5のinput_row/input_rankを保存し、IDをfloat32に丸めない。
- 50行未満という旧yielderの暗黙スキップは行わない。モデルが処理できなければ推論失敗として扱う。
- 全PFO表とリンクはGNN入力のmaskとは独立して保存する。track-only PFOもgetEnergy()を使う。

truthクラスタにはMC対応のある入力だけを入れる。未知のMC ID=-1を一つのtruth粒子にしない。
未知行も含めてforwardした出力から、既知truthのクラスタ単位でエネルギーを集計する。
このtruth結果は既知ラベル範囲の診断であり、全MC targetや完全なperfect PFAではない。

各イベントで保存する比較集合:

|フラグ|意味|
|---|---|
|reference_valid|H5のPFO/MC/quark評価量が有効|
|common_valid|reference_validかつGNN推論成功。Pandora/GNNの共通集合|
|truth_complete|全保存入力を使用でき、すべてにtruth対応がある|
|three_way_valid|common_validかつtruth推定成功かつtruth_complete|

不足するtruthを推測して三者比較に含めない。今回の16イベントには全イベントに未知gap hit等があり、
完全な三者比較の集合は空になる。partial truthの結果は別名で保存する。
全入力に対するtruth比較が必要なら、KEK側で欠落MC対応を回復できるかの追加調査が必要。
入力候補の保存範囲は拡張したが、Pandora内部のhit/track選別や校正まで同じとは主張しない。

## エネルギーの選択

既定の`--energy-policy alpha`では、現在のイベントROOT処理と同様に最高betaの入力を代表点とする。
その点がtrackならtracker回帰値、それ以外ならクラスタ内calo回帰値の和を使う。
代表点は既存処理と同じ`argsort(-beta)`で選ぶ。

`--energy-policy any-track`ではtrackを含むクラスタについて最高betaのtrackの回帰値を使う。
どちらの設定でも両方の値を保存し、評価マクロが使う`energy`を設定で明示する。
GNNとtruthで同じ規則を使う。calo headのないモデルでは`--no-calo-head`を指定し、
neutralも代表点のenergy head値を使う。Pandoraはモデルの規則ではなくPFOのエネルギーを使う。

## ROOTの内容

|Tree/Object|内容|
|---|---|
|eval_events|全入力イベントの識別、選択フラグ、各エネルギー、nu/thrust/qPdg、入力数、失敗理由|
|PfoAnalysisTree|upstream AnalysePerformanceへ渡せるfloat枝とqPdg/I。GNNの成否では除外しない|
|pfo / pfo_links|全PFOと全構成要素リンク|
|eval_hits|全入力行とmodel_inputフラグ、元行番号、整数ID、推論値、GNN/truthクラスタ|
|eval_truth|H5のtruth_particles。全MC collectionやtarget一覧ではない|
|eval_clusters|algorithm=0 Pandora、1 GNN、2 known-truth。クラスタごとのエネルギー等|
|eval_matches|既知truth入力に対するoverlap評価。未対応truth粒子も行を保存|
|pandora_eval_metadata|H5生成metadata、実行引数、checkpoint/code hash、評価範囲|
|comparison_configuration|複数ROOTの比較設定一致を検査する情報|

粒子のefficiency/purityは、共通の**truth対応があるモデル入力**のdepositを分母にする。
truthごとに最大deposit overlapのクラスタを選ぶ（全体の一対一割当ではない）。
未マッチtruthはefficiency=0、purityは未定義。track-onlyなどdeposit分母0はNaN。
未知入力のdepositは別に記録し、既知範囲のpurityを全入力のpurityと解釈しない。
モデルの再構成性能から独立に、未知truthの割合も必ず確認する。

評価ROOTには、共通イベントの中央エネルギー比較canvas、角度別分解能graph/canvas、
labelled efficiency/purity、範囲を自動拡張する粒子エネルギー残差histogramを保存する。
旧マクロのすべての図を同じ名前で再生成する形式ではない。
`resolution.sample`は0=Pandora全有効イベント、1/2=共通Pandora/GNN、
3/4/5=完全truth共通のPandora/GNN/truth、6=partial truth診断。
`angle_bin=-1`は中央領域、それ以外は角度区間番号。

## 検証

```bash
python -B -m unittest test_pandora_eval_analysis test_pandora_eval_reference -v
```

実H5の16イベント・77,502行・872PFOの整合性、未知truth入力1,203行の保持、
旧形式の前処理一致、track-only PFO、大きいID、空/1行イベント、無効特徴量、
未知truthをクラスタ化しないこと、明示したenergy policyを確認する。
単独テストは実H5がなくても実行でき、実H5依存の項目のみskipする。

独立照合テストは180イベントの人工fixtureを作り、元のLCPandoraAnalysisの
AnalysePerformanceを別の実行ファイルとしてビルドする。14分布の全binと、
計算可能な角度区間の分解能・誤差の完全一致を検査する。
LCPandoraAnalysisの場所は環境変数LCPANDORA_ANALYSISで指定できる。
少数の実イベントによる推論確認は動作検証であり、GNNの性能や文献との物理条件の一致の確認ではない。

## 2026-08-17 174838 / epoch 27 の確認

このcheckpointにはmodelのstate_dictのみが保存されており、パラメータを持たない
出力活性化関数の情報はない。gitの旧定義（b2c5c23）では回帰headの末尾がLinear、
現在の定義ではその後にSoftplusがある。小さい出力をSoftplusに通すと約0.693に
変わり、hitごとの値を合計するcalo回帰では大きな余分なエネルギーになる。

この旧定義を使うには、明示的に次のオプションを指定する。

```bash
python -B save_root_pandora_eval.py \
  /home/murata/simulation/validation/pandora_eval/final/dd_350_sample.h5 \
  output/pandora_eval/dd_350_174838_epoch27.root \
  --checkpoint checkpoint/energy_regression/ckpts_gravnet_new02_2026_08_17_174838_outputD5_multihead/ckpt_27_1.pth.tar \
  --device cpu --input-dim 7 --output-dim 5 --momentum --momentum-amp \
  --calo-head --model-variant multihead --tbeta 0.9 --td 0.5 \
  --regression-output-activation linear
```

指定時のみ、ロードしたmultiheadモデルの回帰head末尾のSoftplusを除く。
重みとクラスタリングheadは維持する。既定値currentは現在のモデル定義を使う。
新たにSoftplus付きで学習したモデルにlinearを指定してはいけない。
checkpoint名から自動推定はしない。選択はROOTの実行引数・comparison_configurationに保存する。
model.pyとgravnet_model.pyのhashも記録する。

旧モデル定義をgitからメモリ上に読み込んだモデルと、Softplusだけを除いたモデルで、
350 GeVのsource index 0のbeta・座標による所属・tracker/calo出力が完全一致した。
4イベントではGNN可視エネルギーはsource index 0/1/2/22について
310.969 / 348.663 / 335.950 / 308.863 GeV。Pandoraは
341.634 / 359.359 / 365.501 / 316.038 GeV。いずれもニュートリノ補正前。
4イベントのみなので分解能や全体の性能を確定する統計量ではない。

前回のpretrainedをcurrent定義で実行した結果も、物理性能の判断に使う前に、
そのcheckpointの学習時の出力定義との互換性を確認する必要がある。


## Gap hit除外での全エネルギー評価（2026-09-25）

`--exclude-gap-hits`を指定すると、`EcalBarrelCollectionGapHits` と
`EcalEndcapsCollectionGapHits` のcalorimeter行だけを、特徴量生成・forward前に除く。
MC対応の有無では選別しない。他の追加hit・trackは残し、PFOのエネルギーは変更しない。
`input_row` は除外・前処理後も元H5の行番号に戻す。H5自体を編集しない。
`n_excluded_gap_hits` と `n_invalid_inputs` は重複しない。両者と `n_model_inputs` の和は
`n_inputs` に一致し、`n_model_hits + n_model_tracks == n_model_inputs` になる。
`truth_complete` は従来どおり全保存入力に対する定義で、除外後のtruth coverageを意味しない。

`--detail clusters` は `eval_events`・`PfoAnalysisTree`・`eval_clusters` を保存し、
per-hit/PFO-link/truth-particle/efficiency-purity表は空にする。全量jet energy比較用であり、
この出力でeff/purを評価したとは解釈しない。`--detail events` はクラスタ表も省略する。
従来の詳細出力は既定の `--detail full` で維持する。
比較設定にgap除外とdetailを含め、異なる入力方針のROOTを混ぜない。

全量実行・再開:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B /data/suehara/mldata/pfa/murata/pandora_eval/nogap_all_epoch27/run_bepp_all.py --workers 12
```

3,595 H5の一覧をKEKのLCIO処理一覧と照合し、全749,200イベントを処理する。
GPU 0/1を使い、TF32を無効化。epoch27のlinear回帰・tbeta=.9・td=.5・alpha energyを固定する。
既存の正常出力は再検証して再利用できる。入力stat・コード/checkpoint hashが変われば停止する。
ファイルごとのエラーはauditに残し、1ファイルでも未完了なら全量比較図を出さない。
全量成功後、compare_all.pyが中央/前方/角度別のPandora・GNN・IDR比較をPNG/PDF/ROOT/CSVに保存する。
IDR比較は角度最終binを0.97–0.98とし、ニュートリノ補正なしを主結果、補正ありを別結果にする。
Pandora側の全150分布はKEKのLCIO結果とbin単位で一致することを検証する。

CPU/GPUでは浮動小数点・近傍探索の差によりクラスタリング境界が変わり得るので、
gap除外効果の200イベント試験では同じGPU経路でgap有無を比較する。
全量GNNはGPU経路に統一し、過去のCPU小標本の数値をそのまま差し引かない。
