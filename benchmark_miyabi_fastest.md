# Miyabi GPU数・batch size速度比較

更新: 2026-09-29T11:27:54+09:00

目的: 学習完了までの実時間を短縮。今回は速度測定のみ、精度は未評価。
投入枠の上限: 57.234 GPU分 / 60 GPU分。自動再投入なし。

PBSが報告した消費（終了済み失敗分も含む）: 53.50 GPU分。

- 3444017: 4 GPU, state=F, completed cases=4; estimated.start_time: Tue Sep 29 09:45:50 2026; resources_used.walltime: 00:03:26; Exit_status: 0
- 3444167: 1 GPU, state=F, completed cases=2; estimated.start_time: Tue Sep 29 09:49:52 2026; resources_used.walltime: 00:02:10; Exit_status: 1
  rank 0: batch/GPU=64でGPUメモリ不足。後続条件は未測定。
- 3444121: 2 GPU, state=F, completed cases=3; estimated.start_time: Tue Sep 29 09:49:52 2026; resources_used.walltime: 00:02:30; Exit_status: 0
- 3444188: 4 GPU, state=F, completed cases=2; resources_used.walltime: 00:01:15; Exit_status: 0
- 3444199: 8 GPU, state=F, completed cases=2; resources_used.walltime: 00:01:25; Exit_status: 0
- 3444200: 4 GPU, state=F, completed cases=1; resources_used.walltime: 00:00:55; Exit_status: 0
- 3444218: 16 GPU, state=F, completed cases=1; estimated.start_time: Tue Sep 29 11:26:16 2026; resources_used.walltime: 00:00:46; Exit_status: 0

各条件・各phaseはwarm-up 2 batch + 測定10 batch。LR=4e-4を固定。precisionは各行に記載。
Batch/GPUは1 GPUあたりのイベント数。Global batch = GPU数 × Batch/GPU。

| Job | GPU | Batch/GPU | Global batch | Precision | Chunk/buffer/workers | Train events/s | GPU実使用ピーク GiB | 推定epoch時間 h | 推定GPU時間/epoch h |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|
| 3444167 | 1 | 16 | 16 | fp32 | 8/256/2 | 39.72 | 8.06 | 13.821 | 13.821 |
| 3444167 | 1 | 32 | 32 | fp32 | 8/256/2 | 44.81 | 24.78 | 12.264 | 12.264 |
| 3444121 | 2 | 16 | 32 | fp32 | 8/256/2 | 78.63 | 7.81 | 6.982 | 13.963 |
| 3444121 | 2 | 32 | 64 | fp32 | 8/256/2 | 88.42 | 24.63 | 6.214 | 12.429 |
| 3444121 | 2 | 64 | 128 | fp32 | 8/256/2 | 79.29 | 81.90 | 6.927 | 13.853 |
| 3444188 | 4 | 8 | 32 | fp32 | 8/256/2 | 128.95 | 2.94 | 4.258 | 17.031 |
| 3444188 | 4 | 16 | 64 | fp32 | 8/256/2 | 173.63 | 8.39 | 3.170 | 12.681 |
| 3444017 | 4 | 32 | 128 | fp32 | 8/256/2 | 176.32 | 24.35 | 3.122 | 12.487 |
| 3444200 | 4 | 32 | 128 | bf16 | 8/256/2 | 152.61 | 21.21 | 3.605 | 14.419 |
| 3444017 | 4 | 32 | 128 | fp32 | 8/128/4 | 175.20 | 24.63 | 3.166 | 12.663 |
| 3444017 | 4 | 32 | 128 | fp32 | 8/256/4 | 175.54 | 26.38 | 3.160 | 12.639 |
| 3444017 | 4 | 32 | 128 | fp32 | 32/256/2 | 164.75 | 24.98 | 3.339 | 13.355 |
| 3444199 | 8 | 16 | 128 | fp32 | 8/256/2 | 305.17 | 8.36 | 1.815 | 14.517 |
| 3444199 | 8 | 32 | 256 | fp32 | 8/256/2 | 347.21 | 26.38 | 1.597 | 12.777 |
| 3444218 | 16 | 8 | 128 | fp32 | 8/256/1 | 497.04 | 3.00 | 1.112 | 17.791 |

同一global batch=128・precision=fp32・loader条件での暫定実時間最短: 8 GPU × batch 16 (1.815 時間/epoch、14.517 GPU時間/epoch)。精度維持の確認は未実施。

同一global batch=32・precision=fp32・loader条件での暫定実時間最短: 4 GPU × batch 8 (4.258 時間/epoch、17.031 GPU時間/epoch)。精度維持の確認は未実施。

同一global batch=64・precision=fp32・loader条件での暫定実時間最短: 4 GPU × batch 16 (3.170 時間/epoch、12.681 GPU時間/epoch)。精度維持の確認は未実施。

## 測定の限界

- train 1,960,000 / validation 20,000イベントへの短区間からの外挿です。全epochを測っていません。
- rankごとの進捗batch数から各phaseの最大推定時間を採用します。
- 初回読み込み、後続chunk読み込み、checkpoint保存、キュー待ちは外挿に含みません。
- 各条件のデータ分割・イベント数・ファイルキャッシュ状態は異なり、速度差にはその影響も含まれます。
- 大きなbatchではイベントあたりの更新回数が減ります。GPU時間/epochが短くても、目標精度までの総時間が短いとは限りません。
- 今回のbatch比較ではLRを固定し、速度のみ評価しています。loss値からLRやbatchの最適性を選びません。
- GPUメモリはPyTorch allocatorの最大allocated値です。NCCLなどの外部割当は含みません。reserved値もJSONに記録します。
- GPUメモリ不足・時間切れが起きた条件は未完了として扱い、自動再実行しません。

## 後続の精度比較候補（今回実行しない）

- 基準: global batch 128、実際のoptimizer LR 4e-4、AdamW weight decay 1e-4、元のloss設定。
- LR: 1e-4 / 2e-4 / 4e-4、warm-up: 0 / 2 / 5 epochを段階的に比較。
- 有望な条件でweight decay: 1e-5 / 1e-4 / 1e-3を比較。全組合せの一括探索はしない。
- 同じ学習イベント数・固定validationでenergy lossとclustering lossを別々に評価。
- beta/energy lossの有効化時期を揃え、LR scheduler・warm-upを短い試験用に都合よく変更しない。
- 精度評価と許容差を決めてから本番設定を選ぶ。元のloss定義やqminなどは今回変更しない。
