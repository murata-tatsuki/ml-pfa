# Track–calorimeter hit の embedding 可視化

checkpoint で推論した **最終クラスタリング座標**をイベントごとに取り出し、
PCA／元座標・metric MDS・t-SNE と、元の空間の距離評価を出力します。
既存の学習・推論コードは変更しません。`extract.py` が既存の `model.py`、
`dataset_ilc_sharded.py`、`cluster_energy.inference_forward` を利用します。

## HTMLで各GravNet層を見る

`display/` と同じPlotlyを使ったHTML出力に対応しています。PNGも同時に保存できます。
`--gravnet-coordinates` で各層の学習座標を保存し、描画時に `--all-coordinates --html` を指定します。

```bash
cd /home/murata/master
python embedding_visualization/extract.py \
  --checkpoint checkpoint/energy_regression/ckpts_gravnet_new02_2026_09_29_120252_outputD5_multihead/ckpt_9_1.pth.tar \
  --input /data/suehara/mldata/pfa/murata/data/tc/tc_nnqq_2M_nobrems/validation/dd_ft0_99.h5 \
  --output embedding_visualization/results/layers_embeddings_new \
  --embedding-dim 4 --num-events 1 \
  --thetaphi --momentum --momentum-amp --extended-h5-input --exclude-gap-hits \
  --gravnet-coordinates

python embedding_visualization/plot.py \
  --input embedding_visualization/results/layers_embeddings_new \
  --output embedding_visualization/results/layers_plots_new \
  --all-coordinates --html --max-points 500 --perplexities 10 30 50 --seeds 42
```

`layers_plots_new/index.html` をブラウザで開きます。
各イベントの `display.html` では上部の選択欄で **GravNet 1〜4層／最終座標**、
散布図の選択欄で **PCA／MDS／各t-SNE設定** を切り替えられます。
点のhoverでtruth ID、track/hit、β、元座標、保存されているhit/MC IDを確認できます。
拡大縮小・粒子ごとの表示切り替え・カメラボタンからのPNG保存もできます。
HTMLにPlotly.jsを埋め込むため、各 `display.html` はオフラインで単独でも開けます。
PNGへのリンクも使う場合は出力フォルダごとコピーしてください。

各層で取り出すのは `gravnet_blocks[i].gravnet_layer.lin_s` の出力です。
これは**その層の中でkNNと距離重み付けに実際に使う学習座標**であり、現在のモデルでは4次元です。
層の最後には新たな座標は出力されず、96次元の特徴量が出力されます。
`--gravnet-features` を追加すると、これも `gravnet_1_features` などのキーで保存できます。
入力・伝播特徴量・モデル出力は変更せず、forward hookで値を読むだけです。

- 座標キー: `gravnet_1_coords` 〜 `gravnet_4_coords`、最終座標は従来の `embedding`。
- `--all-coordinates` は4層の座標と最終座標を同じ描画標本で比較します。
- 個別には `--embedding-key gravnet_2_coords`、特徴量には `--embedding-key gravnet_2_features` を使えます。
- 層の次元削減はそれぞれ独立です。図の向き・位置は揃っていません。
  各層で座標の尺度も異なり、絶対距離が小さいだけで改善とはいえません。
  各層の元空間で計算した近傍純度も併読してください。
- hoverに表示するβは最終出力のβで、各層のβではありません。
- `--html` を省略すると従来どおりPNGと数値ファイルのみを生成します。

既存の描画結果にHTMLだけ追加する場合、推論・次元削減を再実行する必要はありません。

```bash
python embedding_visualization/export_html.py \
  --plots embedding_visualization/results/smoke_multihead_plots
```

この変換は `report.json` が参照する元NPZと、保存済みの投影 `.npy` を読みます。
既存HTMLは上書きしません。HTML出力には追加で `plotly` が必要です（5.13.1で確認）。

## 1. checkpoint から抽出

学習・推論が動く Python 環境で実行してください。
`--embedding-dim` は **β・charge likeness・energy などを除いた座標だけの次元数**です。
現在の `train.py` では通常、学習時の `--output-dimension 5` に対して座標は **4次元**です。
出力テンソル全体の列数や中間特徴量の幅とは異なります。学習設定を確認してください。

```bash
cd /home/murata/master
python embedding_visualization/extract.py \
  --checkpoint /path/to/ckpt.pth.tar \
  --input /path/to/validation.h5 \
  --output embedding_visualization/results/embeddings_run1 \
  --embedding-dim 4 --start 0 --num-events 3 \
  --thetaphi --device cpu
```

`/path/to/...` は実際のパスに置き換えます。入力は単一H5またはH5のあるディレクトリです。
ディレクトリの場合はファイル名順。`--start` は空イベントも含む元H5の通算イベント番号、
`--num-events` は調べるイベント数です。末尾に達した場合は利用できる分だけ処理します。
空イベントは `manifest.json` に理由を記録します。
1ファイルずつ読み込み、イベント全体を前処理・推論した後に保存します。

入力フラグは学習時に合わせます。

| フラグ | 意味 |
|---|---|
| なし | 基本5入力 |
| `--thetaphi` | 角度2入力を追加 |
| `--momentum` | 運動量3入力を追加 |
| `--momentum --momentum-amp` | 運動量3入力と大きさ1入力を追加 |
| `--mctpe` | 既存のMC運動量入力モード。これを使った旧形式学習の場合のみ |
| `--extended-h5-input` | `pandora-eval-1` / `nnqq-2m-eval-1` 形式 |
| `--extended-h5-input --exclude-gap-hits` | 新形式でgap hitを除いた学習に対応 |
| `--device cuda:0` | GPU推論 |

入力幅はcheckpointと照合します。ただし同じ幅でも入力の意味が違う場合があるので、
フラグの意味まで自動推定しません。timing cut は、学習時と同じ処理済みH5を渡してください。
抽出側で追加のtiming cutは行いません。

対応モデルは通常の `GravnetModel` と `GravNetModelMultiHead`、DDPの `module.` prefix、
cluster-energy pooling checkpointです。JIT・独立のenergy-branchモデルには対応しません。
multihead はclustering headの幅と指定次元数からβ・charge列を区別し、既存adapterで推論します。
poolingは既存ローダーのpredicted設定を利用します。
legacyは末尾 `D` 列を座標として抽出するため、**誤ったDが形状だけで検出できない場合があります**。
`manifest.json` に出力幅・座標開始列を保存します。

## 2. 描画と距離評価

```bash
python embedding_visualization/plot.py \
  --input embedding_visualization/results/embeddings_run1 \
  --output embedding_visualization/results/plots_run1 \
  --max-points 1500 --perplexities 10 30 50 --seeds 42 7 123 --k 10
```

描画はCPUで実行し、画面のないサーバーでもPNGを生成します。初回は
`--max-points 500 --perplexities 30 --seeds 42` で軽く確認できます。
MDSは点数に対して二乗のメモリが必要で、t-SNEはperplexityとseedの組ごとに実行します。
`--threads` は既定1です。scikit-learn 1.1.2で動作確認し、新旧のMDS／TSNE引数名にも対応します。

- クラスタは10色と9種類の形の組み合わせで識別します。同じ正解粒子には全層・全手法で同じ色と形を割り当てます。
- trackは小さい中抜きのクラスタ形状に黒い＋を重ねます。calorimeter hitは塗りつぶしたクラスタ形状です。
- noise／truth不明の点はグレー。正解粒子IDはイベント内のIDです。
- 各正解クラスタのcalorimeter hitのうちβ最大の点が描画対象に含まれる場合、小さい黒いリングで強調します。
- 全trackを保持し、残りの描画枠をcalorimeterの正解クラスタごとに均等に近く割り当てます。
  **これは表示用の標本であり、元のhit密度の忠実な表示ではありません。**
- 同じ標本を全手法・全seedで使用します。track数が `--max-points` を超える場合は上限を増やします。
- **距離指標は標本ではなく、保存された全点で計算します。**
- `--focus-truth 12` でID 12だけ強調できます。全点で求めた2D配置を維持して他を薄くします。
- PCA・MDS・t-SNEはいずれもtrackとhitを一緒に処理し、各次元の標準化やL2正規化をしません。
- 元座標が2次元以下ならbaselineは元座標。3次元なら元座標の3D図も追加します。
- perplexityが標本数以上なら、そのt-SNEだけ理由を記録してスキップします。

## 出力

### 粒子ごとの凡例画像

各イベントの `particle_legend.png` に **色と形、truth ID、MC ID、粒子名、PDG ID、MC energy [GeV]** を
表として保存します。描画された点数／保存された全点数も記載します。
元の散布図には表を追加しません。全GravNet層・投影で色と形が共通なので、凡例はイベントに1組です。
凡例画像の上部にcalorimeter hit・track・highest betaの記号説明も付けます。
highest betaは、各truth clusterのcalorimeter hitの中で最終出力βが最大の点です。
描画標本にその点がある場合だけリングを描きます。各GravNet層で別のβを計算するわけではありません。
35粒子を超えた場合は `particle_legend_02.png`、`particle_legend_03.png` と分割します。
数値と色は `particle_legend.csv` / `particle_legend.json` にも保存します。

energyは `display/display_h5.py` と同じ **MC粒子の全エネルギー**
`sqrt(mass^2 + mcpx^2 + mcpy^2 + mcpz^2)` です。hitのdeposit energyや予測energyとは異なります。
同じ粒子に属するhitのMC energyを足し合わせることはしません。
粒子名は [PDGのMC番号規約](https://pdg.lbl.gov/2026/mcdata/mc_particle_id_contents.html) に従い、
未登録のコードはPDG番号そのものを表示します。truth不明の行はUnknown/noise、energyはN/Aです。
同じ粒子のラベルに異なるenergyがある場合は範囲を表示します。

以降は `plot.py` の実行時に自動作成されます。HTML出力時は、凡例の別画像へのリンクも追加します。
新しい `extract.py` はPDGとMC energyをNPZに保存します。
古いNPZでも元H5への参照が残っていれば、MC IDで粒子情報を照合して補完できます。

すでに描画済みの結果に**凡例だけ追加**するコマンド:

```bash
python embedding_visualization/particle_legend.py --plots /path/to/plots
```

複数runを含む親フォルダも指定できます。既存の投影やHTML、NPZは書き換えません。
生成済みの凡例だけ更新する場合は `--overwrite`、1ページの行数変更は `--rows-per-page 40` を指定します。
元H5から補完する場合のみh5pyとAwkwardが必要です。

抽出フォルダ:

- `manifest.json`: checkpointの絶対パス・SHA256、設定、バージョン、元H5のイベント位置、処理状態。
- `event_000000.npz`: 全点のembedding、β、truth ID、truth validity、track mask、hit/MC ID、energyなど。
  配列の行順は共通です。extended形式の `input_row` は前処理側の値を保存します。
  legacy形式の `input_row` は処理後の行番号で、元H5の行番号ではありません。

描画フォルダの各イベント以下:

| ファイル | 内容 |
|---|---|
| `overview.png` | baseline・MDS・最初のt-SNE・距離分布の4パネル |
| `baseline.png`, `mds.png`, `tsne_p*_seed*.png` | 各投影の単独図 |
| `original_3d.png` | 3次元embeddingの場合のみ、元座標の3D図 |
| `distance_heatmap.png` | track × 正解caloクラスタの距離中央値。正解対応セルは赤枠 |
| `per_track.csv` | trackごとの距離中央値・近傍純度・正解hitの最初の順位 |
| `distance_heatmap.csv`, `distances.npz` | 全track・全正解caloクラスタの距離表とヒストグラム |
| `selected_rows.npy`, `*.npy` | 表示に使った元の行番号と2D座標 |
| `report.json` | 数値指標、MDSの距離再現誤差、設定、色対応、skip理由 |

`--all-coordinates` の場合はイベント以下に空間名のサブディレクトリを作り、上記のファイルを保存します。
`--html` の場合は描画先に `index.html`、各イベント以下に `display.html` を追加します。

描画先の `summary.json` は各イベントの指標一覧です。
ヒートマップ画像は先頭40 track × 40クラスタまで表示し、CSV／NPZには全件保存します。
出力先は**新規ディレクトリ**を指定します。既存の結果を上書きしません。

## 指標の読み方

- 距離は元のembedding空間でのユークリッド距離です。
- 同一粒子／別粒子の距離ヒストグラムは、trackごとに確率へ正規化して平均します。
  hit数の多い粒子だけで結果が決まるのを避けます。各側の有効track数も保存します。
- `knn_purity` は **truth既知のcalorimeter hit** から近い順にk個選び、同じtruth IDの割合を計算します。
  候補がk個未満なら全候補を使い、`k_effective` に個数を保存します。
- `nearest_hit_accuracy` は同じ候補集合で、最近傍が正解だったtrackの割合です。
- 対応する正解hitのないtrackはretrieval指標の平均から除外し、件数を別記します。
  truth不明track・hitも教師あり比較から除外しますが、前処理で残った行は推論・散布図には含めます。
- legacy入力は既存ローダーに合わせてMC ID=-1の行を除外します。
  extended入力は既存の有効性・gap方針に従い、truth不明を理由には除外しません。
- t-SNE上の遠近は元の距離そのものではありません。MDSも歪みがあるため元空間の指標と併読します。
  `relative_distance_error` は `sqrt(sum((d_original-d_2d)^2)/sum(d_original^2))` です。
- それぞれのイベント・checkpointで独立に次元削減します。別の図の向きや位置を直接比較しません。

## 中間層を見る

抽出時に `--layer postgn_dense`（通常モデル）、または
`--layer model.head_postgn_dense.0`（multiheadのclustering head）を追加すると、
hookで中間特徴量も同じNPZの `intermediate` に保存します。
層はイベントの行数と一致する2次元テンソルを返す必要があります。

```bash
python embedding_visualization/plot.py \
  --input embedding_visualization/results/embeddings_with_layer \
  --output embedding_visualization/results/intermediate_plots \
  --embedding-key intermediate --max-points 500 --perplexities 30
```

中間層のユークリッド距離には、最終OC座標と同じ学習上の意味があるとは限りません。
まず最終座標を評価し、中間層の図は補助的な診断として使ってください。

## テスト・デモ

```bash
python -m unittest discover -s embedding_visualization -p 'test_*.py' -v
python embedding_visualization/demo.py --output /tmp/embedding_demo_inputs
python embedding_visualization/plot.py \
  --input /tmp/embedding_demo_inputs --output /tmp/embedding_demo_plots \
  --max-points 150 --perplexities 10 30 --seeds 42 7 --iterations 500
```

デモは**合成データ**です。event 0は正しい対応、event 1は2本のtrackを意図的に入れ替えます。
正しいイベントの最近傍正解率は1、入れ替えたイベントは1/3になります。
実checkpointの性能を表すデータではありません。

描画のみの依存はNumPy、SciPy、scikit-learn、Matplotlib、threadpoolctl。
抽出は追加で既存masterのPyTorch、PyG、torch-scatter、torch_cmspepr、Awkward、h5py等が必要です。
`load_digits` やTensorFlowは不要です。

## 色・形・役割マーカーの更新

`visual_style.py` でPNG、HTML、凡例の配色と記号を共有しています。
近い色を並べたHSVから、離れた色と丸／四角／上下左右の三角／菱形／五角形／六角形の組み合わせへ変更しました。
最初の90クラスタは色と形の組が重複しません。それ以上では色の明度も変えます。
trackは小さい中抜きのクラスタ記号＋黒い＋、highest beta calo hitは小さい黒リングです。
従来の大きい星・四角とtrack上のID文字を除き、点の重なりを減らしています。

既存の出力にも、新しい表示を反映できます（PNG・HTML・凡例・表示設定を更新）。

```bash
python embedding_visualization/restyle.py --plots /path/to/plots
```

複数runを含む親フォルダも指定できます。保存済みの2D座標と描画標本を使い、
推論・PCA・MDS・t-SNEをやり直しません。元のNPZ、投影NPY、距離指標は維持します。
粒子凡例は `report.json` の `truth_styles` と `truth_colors` に合わせて更新します。

## 拡大表示と点のサイズ

散布図・overview・粒子凡例はPNGと同名の**ベクターPDF**も保存します。
例：`embedding/tsne_p30_seed42.pdf`、`embedding/overview.pdf`、`particle_legend.pdf`。
PNGをPDFに貼るのではなく、点・線・文字を直接描画するので、拡大してもぼやけません。
距離ヒートマップの独立画像は従来どおりPNGです。
`display.html` から各層のPDFと凡例PDFを開けます。

PNG・PDFの点の直径相当の倍率は `--marker-scale` で指定します。
標準は縮小前の `1.0` 倍です。HTMLは別設定の `--html-marker-scale` を使い、
新規描画時は `0.6` 倍を標準にします。既存結果の更新時はHTMLの現在の倍率を維持します。
trackとhighest betaの記号・線幅も一緒に縮小し、別図の凡例は読みやすいサイズを保ちます。
新規描画の `plot.py` と既存結果更新の `restyle.py` の両方で利用できます。

```bash
# masterディレクトリで実行。PNG/PDFを縮小前のサイズに戻し、HTMLのサイズを維持
python embedding_visualization/restyle.py \
  --plots embedding_visualization/results --marker-scale 1.0
```

PDFのページ拡大では点自体も大きくなります。
点の画面上のサイズを保ちながら密集した場所を調べる場合は、HTMLのグラフ内でドラッグして拡大してください。
いずれも保存済みの描画対象の点を表示します。拡大によってサンプリングで省略したhitが追加されるわけではありません。
