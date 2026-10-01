# 動作確認（2026-09-30）

## 色・形と役割マーカーの改善

- 10色×9形状の割当で最初の90クラスタの組が一意であることを確認。
- trackを中抜きクラスタ形状＋小さい＋、highest-beta calo hitを小さい黒リングへ変更。
  粒子凡例にも同じ色・形と役割のキーを表示。単独散布図内の役割凡例とtrack ID文字は除去。
- 既存testイベントを保存済み投影から再描画し、PNGと凡例を目視確認。
  新規 `plot.py --html` の出力も確認。既存13テスト成功。
- `restyle.py` はPNG、HTML、凡例とreportの表示設定のみを更新し、
  NPZ・投影NPY・描画標本・距離指標を維持する。

## 凡例画像の追加確認

- 合計13テスト成功。MC energyの計算、粒子ごとのenergyをhit数分加算しないこと、
  大きなMC ID、色一致、truth不明、energy不一致の範囲表示、画像分割を確認。
- 既存のtraining 6イベント・test 4イベント・先行確認2イベントについて、
  元H5からPDGとMC energyを取得して凡例のみを生成。全粒子の情報を取得できたことを確認。
  推論・次元削減の再実行や既存の図への描き込みは行っていません。
- `particle_legend.png` を目視確認。色は既存の `report.json` のRGBAを直接使用。

## 各GravNet層とHTML出力の追加確認

- 合計9テスト成功。追加テストでは、実際にforward内で呼ばれる `lin_s` の座標と
  block末尾の特徴量を区別して捕捉し、hookが出力を書き換えないことを確認。
- 下記multihead checkpoint・同一validationイベントで、4層すべての座標 `(2961, 4)` と
  層末尾特徴量 `(2961, 96)` を保存。最終embedding・β・truth ID・track maskは
  hook追加前の保存結果と完全一致。
- 同一500点で4層＋最終座標のPCA/MDS/t-SNE（perplexity 10/30/50、seed 42）を生成。
- HTMLに5空間を収録し、外部JS読み込みなし・PNGリンクの存在・全空間での標本一致を確認。
  Firefox headlessで実際のページ描画も確認。
- 新しい結果は `results/layers_embeddings` と `results/layers_plots`。
  ブラウザで開くファイルは `results/layers_plots/index.html` または
  `results/layers_plots/event_000000/display.html`。
  再現用コマンドはREADMEの「HTMLで各GravNet層を見る」を参照。

## 最初の実装の確認

- `test_diagnostics.py`: 7件成功。正誤対応、回転・平行移動に対する距離不変性、
  truth不明、正解hitなし、track保持サンプリング、出力列判定、小標本・非有限値を確認。
- 合成データ: 正しい対応の最近傍正解率1、trackを2本入れ替えた場合1/3。
- 通常checkpoint `ckpts_gravnet_new02_2025_06_19_152211_outputD5/ckpt_12_1.pth.tar`
  と `master/h5/test.h5`: 810点・track 11本。最終4次元と中間128次元の抽出・描画に成功。
- multihead checkpoint と新形式nnqq validation（下記）: 2,961点・track 19本。
  最終4次元と中間128次元の抽出、500点を使ったPCA/MDS、
  perplexity 10/30/50 × seed 42/7 のt-SNE、全点の距離評価に成功。
- 実行環境: Python 3.9.7、PyTorch 1.12.1+cu113（今回の推論はCPU）、
  NumPy 1.22.3、scikit-learn 1.1.2、Matplotlib 3.5.1。

multihead の確認結果は `results/smoke_multihead_embeddings` と
`results/smoke_multihead_plots` に保存しています。`results` はgit管理対象外です。
図は `results/smoke_multihead_plots/event_000000/overview.png` から見られます。
これは動作確認用の1イベントであり、モデル全体の性能評価ではありません。

再現コマンド（出力先は未使用の名前に変更してください）:

```bash
cd /home/murata/master
python embedding_visualization/extract.py \
  --checkpoint checkpoint/energy_regression/ckpts_gravnet_new02_2026_09_29_120252_outputD5_multihead/ckpt_9_1.pth.tar \
  --input /data/suehara/mldata/pfa/murata/data/tc/tc_nnqq_2M_nobrems/validation/dd_ft0_99.h5 \
  --output embedding_visualization/results/nnqq_embeddings_new \
  --embedding-dim 4 --num-events 1 \
  --thetaphi --momentum --momentum-amp --extended-h5-input --exclude-gap-hits \
  --layer model.head_postgn_dense.0

python embedding_visualization/plot.py \
  --input embedding_visualization/results/nnqq_embeddings_new \
  --output embedding_visualization/results/nnqq_plots_new \
  --max-points 500 --perplexities 10 30 50 --seeds 42 7 --iterations 500
```

中間層を見る場合は同じNPZに対して `plot.py --embedding-key intermediate` を使います。

## 2026-09-30: ベクターPDF・マーカー縮小

- 既存の単体テスト13件が成功。
- 実イベントの保存済みNPZから80点を表示し、`plot.py --marker-scale 0.4 --html`
  で新規描画を確認。PCA、MDS、t-SNE、overview、2ページの粒子凡例にPDFを出力。
- 上記6個のPDFを `pdfimages -list` で検査し、埋め込みラスター画像がないことを確認。
- 既存の5空間の結果を `restyle.py --marker-scale 0.6` で再描画し、
  PDFのレンダリングを目視確認。HTMLの点サイズはcalo 3、track 4.2、最高βリング4.8 px。
  HTML内の各投影PDFと凡例PDFへのリンクを確認。
