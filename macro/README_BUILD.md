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

通常の `make` でも既存版とPID版の両実行ファイルをビルドします。
PID版の実行ファイル、オブジェクト、依存関係などはすべて `build/` 内に生成されます。
PID版は `root ...cxx` やACLiCでは起動せず、上記の実行ファイルを使ってください。
以前のPID用 `src/*_pid_cxx.so`・依存ファイル・ROOT辞書は不要です。
他のマクロのビルド・実行方式は変更していません。

**表示について:** 実行ファイルは `main` 内で `gROOT->SetBatch(kTRUE)` しており、**コンパイル版ではキャンバスは開きません**（サーバや SSH でもそのまま走る）。インタプリタで `root ... .x src/....cxx` したときは従来どおり GUI が出ます。画像保存は `saving_canvas` や `SaveAs` がバッチでも有効です。

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
