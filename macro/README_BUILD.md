# ROOT マクロをネイティブ実行ファイルとしてビルドする

インタプリタ（`root hoge.cxx`）は JIT のオーバーヘッドがあり、キャンバス多数のマクロでは表示や処理が重くなりがちです。`g++` + ROOT ライブラリでリンクした実行ファイルにすると、同じ処理でも速く安定しやすいです。

## ディレクトリ構成

- **`src/*.cxx`** … ROOT マクロ本体（インタプリタ実行・ビルドにリンクする `.cxx`）
- **`mains/*.cxx`** … ネイティブ実行用の `main` だけ置く場所（例: `mains/main_efficiency_purity_contribution.cxx`）
- **`root_common_includes.h`** … コンパイル／cling 用の共通 `#include`（`macro/` 直下）
- **`Makefile` / `CMakeLists.txt`** … `master/macro` で `make`（ビルド成果物は `build/`）

## 前提

- CMake 3.16+
- ROOT が `find_package(ROOT CONFIG)` で見つかること（`thisroot.sh` を source したシェルで CMake を実行するのが確実）
- PID版はメタデータ検証に `nlohmann/json.hpp` を使用します（headerのみの依存）。

## ビルド（推奨: `make` のみ）

```bash
cd master/macro
make
```

`make` は内部で `cmake -S . -B build` と `cmake --build build --parallel` を実行します。デバッグビルドは `make BUILD_TYPE=Debug` など。

実行（作業ディレクトリは入力 ROOT の相対パスが通る場所にしてください）:

```bash
./build/efficiency_purity_contribution
```

## PID版のビルドと実行

対象は `src/efficiency_purity_check_reco_effpur_contiribution_pid.cxx`。
入口は `mains/main_efficiency_purity_contribution_pid.cxx` です。

```bash
cd ~/master/macro
make efficiency_purity_contribution_pid

# ソース内の既定入力リスト・出力先で実行
./build/efficiency_purity_contribution_pid

# 入力ROOT・Pandora切替・出力先を指定
./build/efficiency_purity_contribution_pid /path/to/pid.root false figures/pid
./build/efficiency_purity_contribution_pid /path/to/pandora.root true figures/pandora

# 解析せず引数を確認
./build/efficiency_purity_contribution_pid --help
```

引数は `[INPUT_ROOT [PANDORA [OUTPUT_DIRECTORY]]]` の順です。
`PANDORA` は `false/true/0/1`、省略時は `false`。
入力と出力先は省略または空文字列ならソース内の既定設定を使用します。
`qq_energy`・`use_truth_clustering` 等は引き続きPIDソース内の設定で、
入力パスや出力先から自動判定しません。変更後は再度 `make` してください。
相対入力パスを使う場合は従来どおり `macro/` から実行します。

### track 置換あり・なしの保存

PID版は同じクラスタ所属・予測PID・イベントから、次の2方式を保存します。

- `raw_head`：単一trackの荷電clusterも、選択済みGNN energy headを使う。
- `track_replaced`：予測が荷電種でtrackが正確に1本の場合だけ、カロリメータ入口の測定運動量とPID質量から `sqrt(p^2+m_PID^2)` を使う。

中性・trackなし・複数trackの採用値、クラスタ所属、予測PIDは変えません。画像は形式ごとのディレクトリ内で方式を分けます。efficiency/purity、PID・clustering診断、保存された5つの生headの比較図は共通で、各形式の直下に一度だけ保存します。

```text
出力先/
  png/                            pdf/・eps/も同じ構成
    共通図.png                    efficiency/purity・clustering等
    raw_head/                     track置換なしのenergy図
    track_replaced/               track置換ありのenergy図
  pdf/
  eps/
  jet_energy_resolution_summary.root
  jet_energy_resolution_summary.txt
```

各energy・clusteringのデータはROOT 1個とtxt 1個にまとめます。ROOTのトップschemaは `pid-analysis-summary-2` です。以下はファイルシステムのディレクトリではなく、単一ROOTファイル内部のグループです。

```text
jet_energy_resolution_summary.root の内部
  clustering/                 共通のefficiency/purity集計・定義
    diagnostics/              共通の詳細ヒストグラム・canvas
  raw_head/                   置換なしのイベント・JER・定義
    diagnostics/              置換なしの回帰・dijet等の詳細結果
  track_replaced/             置換ありのイベント・JER・定義
    diagnostics/              置換ありの回帰・dijet等の詳細結果
```

txtも1個とし、`[clustering]`・`[raw_head]`・`[track_replaced]` の節で集計を区別します。共通集計は一度だけ保持し、元の詳細ヒストグラムとcanvasもROOT内部に残します。JERだけのhelper実行では、同じ入力・イベント・beta条件の既存統合ROOTから詳細結果を保持します。

通常のPID解析では、元ROOTを1ファイルずつ開き、同じtreeの行から共通診断と両方式のenergy診断を同時に集計します。energyに依存するヒストグラムを2組持ち、共通のefficiency/purity・PID・clustering診断は1回だけ集計します。全入力ROOTをメモリへ保持する方式ではありません。

測定track energyが保存されたROOTは、保存済み値と生head値から両方式を作ります。旧ROOTに測定運動量がない場合は、読み取り済みの必要な行をファイル単位の一時bridgeへ渡し、保存された入力来歴と元H5の `feature[:,7:10]`・`pid_seed_input_row` の対応をPythonで検証して復元します。この復元のために元ROOTを開き直すことはありません。MC運動量で代用せず、対応を確認できなければエラーになります。

解析終了時は `save_pid_energy_readouts.py` で完成済みの共通・方式別診断を統合し、ROOT・txtと画像を反映します。元の推論ROOT/H5は変更せず、bridgeや統合前のROOTは削除します。Python環境には ROOT・uproot・NumPy・h5py・Awkward が必要です。必要なら `PID_READOUT_PYTHON=/path/to/python` で実行環境を指定します。

既存summaryから両版のJER・イベント応答を統合出力する場合、efficiency/purity解析やGNN推論を繰り返す必要はありません。統合ROOT・従来の採用値summary・従来の方式別summaryを入力にできます。

```bash
python3 -B save_pid_energy_readouts.py \
  --summary figures/nnqq2M_fixed_uds_5head_pid/40GeV/reco/jet_energy_resolution_summary.root \
  --draw
```

回帰・dijet等のenergy診断も作成する場合は、上記に `--analysis-executable build/efficiency_purity_contribution_pid` を付けます。実行ファイルの `qq_energy` は入力summaryと一致している必要があります。clustering方式はsummaryから取得します。既存の個別ROOT・txtを一括削除する処理は含みません。統合後の入力を優先して読み込みます。

既存モデルの全energy・両clusteringへ追加する例です。旧ROOTからの測定運動量の復元は元H5も読むため、保存済み値を使う通常の作図より時間がかかります。

```bash
pid_base=figures/nnqq2M_fixed_uds_5head_pid
for pid_energy in 40 91 200 350 500; do
  for pid_method in reco truth_clustering; do
    python3 -B save_pid_energy_readouts.py \
      --summary "$pid_base/${pid_energy}GeV/$pid_method/jet_energy_resolution_summary.root" \
      --draw || exit
  done
done
```

multi-energy比較は、既存の5引数に続く第6引数で方式を指定します。既定値は `"both"` で、全energy・reco/truth_clusteringの両版summaryが必要です。

```bash
root -l -b -q 'plot_jet_energy_resolution_pid.cxx()'
# 第6引数：both / raw_head / track_replaced / saved
# saved は従来の親summaryを使う明示的な互換モード
```

energy比較図22種（event energy ratioの全5energyパネル図をcentral・forward・inclusiveの3種追加）は `comparison/{png,pdf,eps}/{raw_head,track_replaced}/`、共通clustering比較4図は `comparison/{png,pdf,eps}/` に保存します。データは `comparison/jet_energy_resolution_comparison.root` 1個です。ROOT内の `raw_head/`・`track_replaced/` に方式別結果、ROOT直下に共通のclustering比較を保存します。第2引数で出力先を指定した場合も同じ構成です。各energyの統合ROOTを優先し、従来の方式別summary・clustering_summaryも読み込めます。mode定義・beta条件の混在を拒否します。Pandoraは両方式で同じモデル共通参照を使用します。

通常の `make` でも既存版とPID版の両実行ファイルをビルドします。
PID版の実行ファイル、オブジェクト、依存関係などはすべて `build/` 内に生成されます。
PID版は `root ...cxx` やACLiCでは起動せず、上記の実行ファイルを使ってください。
以前のPID用 `src/*_pid_cxx.so`・依存ファイル・ROOT辞書は不要です。
他のマクロのビルド・実行方式は変更していません。

**表示について:** 実行ファイルは `main` 内で `gROOT->SetBatch(kTRUE)` しており、**コンパイル版ではキャンバスは開きません**（サーバや SSH でもそのまま走る）。インタプリタで `root ... .x src/....cxx` したときは従来どおり GUI が出ます。画像保存は `saving_canvas` や `SaveAs` がバッチでも有効です。

### 保存後に `double free or corruption (!prev)` で終了する場合

ビルド時と実行時で異なるROOTを混在させると、同じROOTライブラリが二重にロードされ、保存後の終了処理で二重解放が発生します。確認したケースでは、CMake cacheの `/work/root/build` と、実行環境の `/usr/local` の `libRIO.so` が両方読み込まれていました。

`make` は現在の `root-config --prefix` にある `cmake/ROOTConfig.cmake` を毎回指定し、古い `ROOT_DIR` cacheを更新します。ROOT環境を設定したシェルで、PID版を再ビルドしてください。

```bash
cd ~/master/macro
make efficiency_purity_contribution_pid
./build/efficiency_purity_contribution_pid
```

検証（2026-10-18）：旧実行ファイルで保存後のSIGABRTと `libRIO.so` の二重読込を再現しました。修正後は6イベントで終了コード0、23図のPNG/PDF/EPSと3つのROOTを保存し、全ROOTの再読込も成功しました。event別の保存エネルギーは入力の `energy_pid` 和と厳密一致し、jet分解能・clustering集計値も同じ入力の修正前結果と一致しています。

別のROOTを明示する場合は、対応する環境を読み込んだうえで `make ROOT_CONFIG=/path/to/root/bin/root-config efficiency_purity_contribution_pid` を使います。CMake packageの場所が異なる環境では `ROOT_DIR=/path/to/cmake-directory` を指定できます。`CMAKE_FLAGS` の `-DROOT_DIR=...` 指定も優先されます。CMakeを直接使う場合も、同じROOTの `-DROOT_DIR=...` を指定してください。

### `TClassTable::Add ... already in TClassTable` などの Warning

**原因:** 実行時に **別々のパスから同じ ROOT の `.so` が二重に読み込まれている**ことが多いです（例: `LD_LIBRARY_PATH` に `/home/local/lib` とビルドツリーの `lib` が混在）。

**根本対処（推奨）:**

```bash
# いったんクリアしてから、使う ROOT だけを source
unset LD_LIBRARY_PATH   # または重複パスを手で削る
source /path/to/your/ROOT/bin/thisroot.sh
ldd ./build/efficiency_purity_contribution | head   # libCore.so の実体が1系統か確認
```

ビルドも **同じ `thisroot.sh` を source したシェル**で `cmake` / `make` すると整合しやすいです。

**ログ上の抑止:** `mains/main_efficiency_purity_contribution.cxx` 先頭で `gErrorIgnoreLevel = kError` として **Warning 全体を非表示**にしています（他の Warning も消えるので、環境を直したあとはこの行を外してもよいです）。

CMake を直接使う場合:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

`CMakeLists.txt` を大きく変えたあとに再設定したいときは `make reconfigure`。

## 別のマクロを追加するとき

1. マクロ先頭で `#include "root_common_includes.h"` し、足りないクラスがあれば `root_common_includes.h` にヘッダを追記する。
2. `mains/main_yourtool.cxx` を追加し、`void your_macro();` を宣言して `main` から呼ぶ。
3. `CMakeLists.txt` に次を追加する:

```cmake
macro_root_executable(your_tool
    SOURCES
        mains/main_yourtool.cxx
        src/your_macro.cxx
)
```

`macro_root_executable` は列挙したソースを ROOT にリンクします。インクルードパスは `macro/` 直下も見えるので、`#include "root_common_includes.h"` はそのまま使えます。

## root-config だけでリンクする場合（参考）

CMake を使わない最小例:

```bash
c++ -O2 -std=c++14 -I. -o efficiency_purity_contribution \
  mains/main_efficiency_purity_contribution.cxx \
  src/efficiency_purity_check_reco_effpur_contiribution.cxx \
  $(root-config --cflags --libs)
```
（`macro/` で実行し、`-I.` で `root_common_includes.h` を解決）

複数ターゲットや依存管理には上記の `make`（CMake 経由）の方が向きます。

## インタプリタで動かすとき

`#include "root_common_includes.h"` は **`macro/` をインクルードパスに含める**必要があります（ヘッダは直下にあるため）。

`macro/` にいて `src` 内のマクロを実行する例:

```bash
cd master/macro
root -l -b -q 'gSystem->AddIncludePath("-I'$(pwd)'"); .x src/efficiency_purity_check_reco_effpur_contiribution.cxx'
```


## epochごとのJERを描く

`plot_jer_vs_epoch.cxx` は `epoch_scan` の `metrics.csv` を読み、横軸epoch・縦軸JERの図を作ります。対応する学習ログが見つかれば、lossとの上下段比較・散布図・成分別学習曲線も追加します。推論や元ROOTの再解析は行いません。`valid`・`complete`・`eligible_for_ranking` がすべて真の行を使用し、同じenergy内の入力イベント集合・ファイル数・beta cut・角度範囲が異なる比較や重複行を拒否します。

```bash
cd /data/murata/master/macro
root -l -b -q 'plot_jer_vs_epoch.cxx()'
```

既定入力は `../output/jer_epoch_scan/5head_pid_onehot_2026_10_09_154602/`。直下の `*GeV/metrics.csv` を自動で探します。まだ `--collect` していないenergyはメッセージを出してスキップし、描けるCSVが一つもなければエラーになります。新たに40 GeVなどを集計した後、同じmacroを実行すると追加されたenergyも描きます。

特定のenergyだけ描く場合は、そのdirectoryまたはCSVを指定します。

```bash
root -l -b -q 'plot_jer_vs_epoch.cxx("../output/jer_epoch_scan/5head_pid_onehot_2026_10_09_154602/200GeV")'
```

引数は以下の順です。省略または入力・出力の空文字列は既定値を使います。

| 引数 | 既定値・選択肢 |
| --- | --- |
| 1: input | モデル/run directory、energy directory、または `metrics.csv` |
| 2: outputDirectory | `macro/figures/jer_epoch_scan/<model>_<run>/` |
| 3: readout | `saved` / `raw_head` / `track_replaced`。既定 `saved` |
| 4: method | `both` / `reco` / `truth_clustering`。既定 `both` |
| 5: region | `both` / `central` / `forward`。既定 `both` |
| 6: zoomMax | 拡大図のJER上限 [%]。既定 `15`、`0` は拡大図を作らない |
| 7: trainingLog | 空または `auto` で `shell/tmp/*<run>*/rank0.log` を自動探索。ファイルを明示可能。`none` でloss図を無効化 |
| 8: betaStart | β項が有効になる最初のepoch。`-2`（既定）は既知runのみ自動設定、`-1` は不明 |
| 9: energyStart | energy項が有効になる最初のepoch。既定 `-2`、`-1` は不明 |

例：200 GeVの通常reco・centralだけ、拡大図の上限を10%にする場合：

```bash
root -l -b -q 'plot_jer_vs_epoch.cxx("../output/jer_epoch_scan/5head_pid_onehot_2026_10_09_154602/200GeV","","saved","reco","central",10)'
```

既定ではenergyごとにcentral／forwardの2パネル図を保存し、通常recoを実線、truth clusteringを破線で区別します。CSVの `error_percent` を誤差棒として使い、通常recoのscan内最小点に星を付けてepochとJERを凡例・標準出力に表示します。truthのみ選択した場合はtruthの最小点を表示します。これは観測された最小値の表示であり、最終採用epochの自動決定ではありません。

複数energyがある場合は通常recoのenergy重ね描きも作ります（truthのみ指定時はtruthを重ねます）。全体図を常に出し、さらに指定上限までの拡大図を出します。拡大図には上限を超えた点数を表示します。欠けたepochの間は線をつなぎません。

保存形式はPNG・PDFと、全点の `TGraphErrors`・canvas・入力情報を含むROOTです。既定実行で91／200 GeVのCSVがある場合の例：

```text
macro/figures/jer_epoch_scan/5head_pid_onehot_2026_10_09_154602/
├── 91GeV/
│   ├── jer_vs_epoch_91GeV_saved.png       # PDFも同じ場所
│   ├── jer_vs_epoch_91GeV_saved_zoom.png
│   ├── jer_loss_vs_epoch_91GeV_saved_zoom.png
│   ├── jer_vs_validation_loss_91GeV_central_saved_zoom.png
│   └── jer_vs_energy_loss_91GeV_central_reco_saved_zoom.png
├── 200GeV/
│   └── ...                              # 200 GeVの図
├── jer_vs_epoch_overview_saved.png
├── jer_vs_epoch_overview_saved_zoom.png
├── loss_components_vs_epoch_saved.png
├── energy_loss_components_vs_epoch_saved.png
├── energy_loss_correlation_overview_saved.png
├── energy_loss_correlations_saved.csv
└── jer_vs_epoch_91-200GeV_saved.root
```

energy別のPNG／PDFは、指定した出力directoryの `<energy>GeV/` 以下へ保存します。energy比較図、共通のloss曲線・相関一覧、まとめROOT・CSVは親directoryへ保存します。単一energyを入力に指定した場合も、画像はそのenergyのsubdirectoryへ保存します。ファイル名は変更しません。

同じ出力directoryで再実行すると対応する図・plot ROOTを更新します。入力CSVや推論ROOTは変更しません。ROOTインタプリタで直接実行でき、`make` やACLiCの `+` は不要です。


### lossとの比較

通常の `root -l -b -q 'plot_jer_vs_epoch.cxx()'` で、CSVのrun IDと一致する学習ログを自動探索します。該当ログがなければ既存のJER図だけを作り、複数見つかれば第7引数での明示を求めます。別のrun IDがパスにあるログは拒否します。

特定のログを指定する例です。既存の第1〜6引数はそのまま使用できます。

```bash
root -l -b -q 'plot_jer_vs_epoch.cxx("../output/jer_epoch_scan/5head_pid_onehot_2026_10_09_154602","","saved","both","both",15,"../shell/tmp/tc_nnqq_2M_timingcut_forcealpha_thetaphi_outputD5_2026_10_09_154602_alpha_tracker_diff_log_perCluster_multihead_ranks/rank0.log")'
```

追加する図は次のとおりです。すべてPNG・PDFと同じplot ROOTへ保存します。

- **epochの上下段比較**：energy／角度regionごとに上段はtrain／validationの総loss、下段は通常reco／truth clusteringのJERと誤差。両段でepoch範囲を揃え、validation loss最小epochとJER最小epoch、β／energy項の開始位置を縦線で表示します。全体図とJER拡大図を出します。
- **loss–JER散布図**：energy／regionごとに、横軸をvalidation総loss・`L_E`・`L_V`・`L_beta`・`L_PID`とした5パネルを出します。縦軸はJER（誤差付き）、色はepoch、markerは通常reco／truth clusteringを区別します。全体図とJER拡大図を出します。
- **成分別学習曲線**：`L_E`・`L_V`・`L_beta`・`L_PID`のtrain／validationを別パネルで表示します。負値もあるため軸は線形です。

現在の既知run `5head_pid_onehot / 2026_10_09_154602` は、β項がepoch 2、energy項がepoch 4から有効です。総loss最小の選択と散布図は、全項が有効なepoch 4以降を対象とし、初期の別の目的関数と混同しません。初期epochも上下段・成分図の曲線には残します。別runには2／4を流用せず、開始位置が不明であることを表示します。確認済みの設定がある場合は第8／9引数で指定できます。総loss最小は表示対象JERと対応するepochの範囲で選ぶため、部分scanでは全学習ログの最小epochと異なる場合があります。

総lossはログの `train loss :`／`Returning` を読み、そのまま使用します。成分の印字百分率や和から再構成しません。`Training epoch N` とCSVのepoch Nを結び、batch途中のlossは使いません。総lossが未完了・非finiteのepochや重複したepochは除外して表示し、欠損成分をゼロ補完しません。欠損epochを線でつながず、散布図にはepochが一致する点だけを使用します。

学習ログの成分値は印字精度（例：小数4桁）を保持し、総lossとCSVのJER／誤差は保存された精度で使用します。lossは混合validation sample全体、JERはenergy別の評価であり、散布図の相関は原因を確定するものではありません。推論ROOTや学習・scanのコード／planには変更を加えません。

代表的な追加ファイルは `91GeV/jer_loss_vs_epoch_91GeV_saved_zoom.png`（上下段）、`91GeV/jer_vs_validation_loss_91GeV_central_saved_zoom.png`（散布図）、`loss_components_vs_epoch_saved_active.png`（全項有効化後の成分）です。PDFも同名で保存し、数値graphと散布点ごとのepochは既存plot ROOTへ追加します。

検証：`root -l -b -q 'plot_jer_vs_epoch.cxx()'` の既定経路に相当する自動ログ探索で、91／200 GeVの全172 epochを描画。独立パースしたloss 1,720値、既存JER 1,376点、散布図6,720点のepoch・loss・JER・誤差が厳密一致しました。validation最小印は全項有効化後のepoch14、通常reco JERの最小印は91 GeV centralのepoch28／forwardのepoch63、200 GeV centralのepoch19／forwardのepoch10と一致。PNG20枚・PDF20枚・ROOTを保存し、入力ログ・CSVは不変でした。確認専用ソース・生成物はタスク専用一時領域で扱い、完了後に削除します。


### energy loss内訳とJERの相関

同じ `plot_jer_vs_epoch.cxx()` で、5-headの粒子種別lossを含むログから次の図を追加します。

- **内訳別散布図**：合計`L_E`、荷電`L_E_cond`、中性`L_E_cluster`、`L_E_photon`、`L_E_neutral_hadron`の5パネル。energy・central／forward・通常reco／truth clusteringごとに図を分けます。横軸はvalidation loss、縦軸はJERとCSVの誤差、色はepochです。
- **相関の一覧図**：5成分×energy／角度regionのSpearman相関を、通常recoとtruthで別パネルに表示します。セル内に相関係数と点数を記載し、色範囲は−1〜1で固定します。
- **内訳のepoch推移**：合計・荷電・中性のlossを同じy軸で重ね、trainとvalidationを比較します。

各散布図には全有効epochのSpearman相関係数ρと点数Nを表示し、epoch 60以降に3点以上ある場合は、その範囲の相関も表示します。正のρはlossが大きいepochほどJERも大きい傾向を表し、因果関係や係数変更の効果を保証しません。JER拡大図にも全点から計算した同じρを表示し、表示範囲外の点を相関計算から落としません。通常recoとtruthを混ぜて相関を計算しません。

同順位は平均順位で処理し、有限な組が3点未満、またはloss／JERが定数ならρを未定義とします。未定義値は0へ置換しません。既知runは全項有効になるepoch 4以降を使い、それ以外のrunでは既存のactivation設定に従います。欠けたepochやlossは補完しません。

この5-headモデルの`L_E_cond`はcharged hadron＋electron＋muon、`L_E_cluster`はphoton＋neutral hadronの寄与です。種別の値は共通のparticle分母に対する寄与で、各種別内の平均lossではありません。粒子種別lossが保存されていないログでは、内訳の図をスキップして理由を表示します。

出力は既存の `figures/jer_epoch_scan/<model>_<run>/` です。例：

```text
200GeV/jer_vs_energy_loss_200GeV_central_reco_saved_zoom.png
200GeV/jer_vs_energy_loss_200GeV_central_truth_clustering_saved_zoom.png
energy_loss_correlation_overview_saved.png
```

同名PDFも作成します。ROOTには内訳のloss曲線、散布点、各点のepoch、相関の数値を追加し、`energy_loss_correlations_saved.csv`にはenergy・region・method・成分・epoch範囲・N・ρを保存します。引数でmethod／regionを絞った場合は、その指定を出力名へ付けます。入力CSV、推論ROOT、学習・scanのコードとplanは変更しません。

検証：91／200 GeV・全172 epochについて、内訳loss 1,720値と新散布図6,720点（epoch・loss・JER・誤差）が独立抽出値と一致しました。全有効期間／epoch≥60の相関80件は独立したSciPyの平均順位Spearmanと1e-12以内で一致し、CSV・ROOT TTree・一覧図・full／zoom図の表示を照合しました。既存98 graphの値も不変です。`root -l -b -q 'plot_jer_vs_epoch.cxx()'` の通常経路で既存20種＋新18種のPNG／PDFとROOT・CSVを出力します。確認専用コードと生成物はタスク用一時領域で扱い、完了後に削除します。

保存先変更の確認：通常の描画経路で91／200 GeVそれぞれ32ファイル（PNG16＋PDF16）がenergy subdirectoryへ入り、共通14ファイルは親に保存されることを確認しました。全38 PNGの内容は変更前と一致。既存のenergy別64画像も同じ配置へ移動し、画像内容とまとめROOT・CSVは保持します。
