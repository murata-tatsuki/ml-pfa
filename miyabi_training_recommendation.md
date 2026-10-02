# Miyabi 学習設定: 2026-09-29 短時間測定の結果

これはGPU時間節約を優先した最初の結果。最新の実時間優先設定は `miyabi_training_fastest.md` と `shell/miyabi_train_energy_fast.sh` を参照。

目的は「精度を維持してGPU時間を削減」。今回はユーザー指定により速度測定のみ、上限1 GPU時間。
全ジョブは終了し、PBS walltime × GPU数の合計は失敗分を含め **26.23 GPU分（0.437 GPU時間）**。
本番学習や精度を比較する試験学習は投入していない。

## 今回の暫定設定

`shell/miyabi_train_energy_4gpu.sh` の既定値に反映した。

```text
GPU数                         4（Miyabi-G 4ノード）
--batch-size                  32 / GPU
global batch                  128
--learning-rate               4e-4
--ddp-lr-scaling               none
--stream-files-per-chunk      8
--stream-shuffle-buffer       256
--num-workers                 2 / GPU
--weight-decay                1e-4
--lr-policy                   cosineReduce
--clip-mode                   value
--clip-value                  10
--epochs-nobeta                1
--epochs-noLE                  3
precision                     FP32
```

元のコードでは `--learning-rate 1e-4` をDDP内でGPU数倍していた。
今回の `4e-4 + scaling=none` は、元の4 GPU時と同じoptimizer learning rateを明示する変更。
LR、weight decay、loss係数、scheduler、warm-upを精度面で最適化したという意味ではない。
batchの比較時もoptimizer LRを4e-4に固定し、速度だけを評価した。

## GPU数の比較

全て chunk/buffer/workers = 8/256/2。train 1,960,000イベント + validation 20,000イベントへの外挿。

| GPU数 | batch/GPU | global batch | 推定時間/epoch | 推定GPU時間/epoch | GPU実使用ピーク |
|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 32 | 12.26 h | 12.26 GPU h | 24.78 GiB |
| 2 | 32 | 64 | 6.21 h | 12.43 GPU h | 24.63 GiB |
| 2 | 64 | 128 | 6.93 h | 13.85 GPU h | 81.90 GiB |
| 4 | 32 | 128 | 3.12 h | 12.49 GPU h | 24.35 GiB |

元のglobal batch=128を保つ比較では **4 GPU × 32** が2 GPU × 64より推定GPU時間で約9.9%少なく、実時間も約55%短い。
1 GPU × 64は途中でCUDA OOM、1 GPU × 128は未実行。batch 64は本番の可変サイズイベントに対する余裕が小さい。

1 GPU × 32の推定GPU時間は4 GPU × 32より約1.8%少ないが、global batchが変わり、同じイベント数に対するoptimizer更新回数も変わる。
精度を維持した節約とは判断できない。現時点でGPU数を減らす根拠としては弱い。
将来1・2 GPUへ減らす場合は、gradient accumulationでglobal batchを維持する比較も候補。ただし今回は未実装・未測定。

## Loaderの比較

4 GPU × batch 32では、32/256/2 → 8/256/2で短区間からの推定epoch時間が3.34 → 3.12 h。
rank 0のtrain warm-up（読み込み＋2 batch）は52.27 → 15.80秒、host tree RSSは54.53 → 16.77 GiB。
4 workersへの増加やbuffer=128では、2 workers・buffer=256より速くならなかった。
CPUメモリの余裕と測定速度から8/256/2を暫定採用する。

## 限界と精度最適化の扱い

- 各条件・phaseはwarm-up 2 batch + 計測10 batchのみ。各条件で初期model・optimizer・Torch RNGをリセット。
- 初期読み込み、後続chunkの読み込み、checkpoint I/O、キュー待ちはepoch外挿に含まれない。データ分割やファイルキャッシュの差もある。
- GPUメモリはPyTorch allocatedの最大値。将来の大きなイベントでの上限を保証しない。
- chunk変更はシャッフル順にも影響する。精度維持はまだ検証していない。
- lossを全て有効にして速度を測った。短い試験のloss値からハイパーパラメータの優劣は選ばない。
- 後続の精度探索候補はoptimizer LR 1e-4/2e-4/4e-4、warm-up 0/2/5 epoch、その後weight decay 1e-5/1e-4/1e-3。今回は未実行。
- 同じ学習イベント数と固定validationでenergy lossとclustering lossを別々に評価し、許容差を決めてから精度面の採用判断を行う。
- optimizer・cosineReduce・epochを含む再開機能を実装済み。中断したepochをやり直す手順は [miyabi_training_resume.md](miyabi_training_resume.md) を参照。

## 起動時に修正した問題

- コンテナに不足していた元のKNNライブラリ `torch_cmspepr` をARM64/Hopper向けにビルドしてユーザー領域へ追加。
  ソース: https://github.com/cms-pepr/pytorch_cmspepr 、commit `e94c49b5326f10a7ac8a634a873090fd42bab10a`。
  再現手順: `shell/miyabi_install_cmspepr.sh`。
- GravNetの同種グラフで `target_to_source` に渡していた片側 `None` の特徴を両端同じ特徴へ修正。
  グラフの向きや近傍探索アルゴリズムは変更せず、直接集約式との一致・逆伝播をCPUテストで確認した。
- DDP起動・固定LR・GravNetのCPUテスト8件、レポート集計テスト3件が通過。
  その後、実GPUで1・2・4 GPUの学習・validation計測に成功した。

詳細: `benchmark_miyabi_scaling.md`、機械可読結果: `benchmark_miyabi_scaling_results.json`、ジョブ一覧と予算: `benchmark_miyabi_scaling_jobs.json`。
