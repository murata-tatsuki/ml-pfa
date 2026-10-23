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
