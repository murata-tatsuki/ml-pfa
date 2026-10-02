# 学習完了までの実時間を優先する設定

ユーザーの最新指定により、GPU時間の節約から実時間の短縮へ評価軸を変更。
精度比較の試験学習は行わず、これまでと追加測定の合計を1 GPU時間以内に制限する。

## 実測済み

| 設定 | 推定時間/epoch | 備考 |
|---|---:|---|
| 8 GPU × batch 16、FP32 | 1.81 h | 採用候補。global batch 128を維持し、4 GPUより約42%短縮 |
| 8 GPU × batch 32、FP32 | 1.60 h | 実測済み最速だがglobal batch 256に変わる。精度未確認のため既定値にはしない |
| 4 GPU × batch 32、FP32 | 3.12 h | 元の4 GPU設定。global batch 128 |
| 4 GPU × batch 32、BF16 AMP | 3.61 h | 約15%遅いため採用しない |
| 2 GPU × batch 64、FP32 | 6.93 h | global batch 128。GPUメモリ約82 GiB |

初期読み込み、後続chunk読み込み、checkpoint保存、キュー待ちは含まない短区間からの外挿。
学習が所要精度へ到達するまでの時間を測ったものではない。

## 追加測定

8 GPU × batch 16（global batch 128）とbatch 32（global batch 256）はjob 3444199で正常終了。
PBS割当後の起動処理に約5分を要したが、学習処理自体は1分25秒で終了した。
実消費の累計は失敗分・BF16測定を含め41.23 GPU分。

最後に16 GPU × batch 8、workers=1（global batch 128）をjob 3444218で最大1分測定する。
validationファイルが20個なので、16 GPUでは空workerを作らないようworkers=1にする。
2026-09-29 10:02 JST時点ではdebug-gで開始予測12:26:59。regular-gにも即時実行できる16ノードの空きは確認できなかった。
これを含む累計消費の上限は57.23 GPU分。追加投入は行わない。
16 GPUは未測定なので本番の既定値にはしない。結果は `benchmark_miyabi_fastest.md` に自動集約する。

## 反映した速度優先設定

最新のユーザー指定により、`shell/miyabi_train_energy_fast.sh` と自動継続版の既定値は4 GPU × batch 32へ変更した。上記の8 GPU測定結果は比較用に残す。
誤って長時間学習を始めないよう、既定の動作は10 batchの速度測定。
実効learning rateは4e-4、chunk/buffer/workersは8/256/2、precisionはFP32。
GPU数を増やしてもlearning rateを機械的に倍増しない。
元の4 GPU用スクリプトは比較・フォールバック用に残す。

```text
GPU数: 4
batch/GPU: 32
global batch: 128
learning rate: 4e-4（ddp-lr-scaling=none）
stream-files-per-chunk: 8
stream-shuffle-buffer: 256
num-workers: 2
AMP: off
```

## Epoch数と精度

最新の指定により、自動のearly stoppingは使用せず、ユーザーがログを確認して手動停止する。
500 epochは上限として残す。8 GPU × batch 16で上限まで走る場合、単純外挿では約38日になる。
今回は速度測定のみという予算指定のため、LRなどの精度探索は実施しない。

## ログ確認と手動停止

- メインログ: `RUN_DIR/ranks/rank0.log`。PBS標準出力の `Main training log:` 行にも実際のパスを出力する。
- 学習の進捗とlossは100 batchごと（8 GPU × batch 16の測定値なら学習本体は約42秒ごと、I/O待ちは別）。`RANK_LOG_INTERVAL`で変更可能。
- 各epochの検証完了時に、従来のloss内訳に加えて `EPOCH_RESULT epoch=... train_loss=... validation_loss=... lr=...` を追記する。epoch番号は0始まり。
- `tail -f`で上記ログを確認できる。停止は `qdel JOB_ID`（JOB_IDは実際のジョブIDに置換）。
- 学習・validationまで完了したepochを `RUN_DIR/checkpoints/ckpt_EPOCH_1.pth.tar` に保存する。
  `RUN_DIR/checkpoints/last.pth.tar` からoptimizer・cosineReduceを含めて再開でき、中断したepochを最初からやり直す。速度測定モードではcheckpointを保存しない。
- 自動早期終了は行わないが、設定したepoch上限・PBS walltime到達時や実行エラーでは終了する。

本番学習は未投入。完全再開を実装済み。投入・再開手順は [miyabi_training_resume.md](miyabi_training_resume.md) を参照。
