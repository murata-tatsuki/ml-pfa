# Epoch単位の中断・再開

`train.py --resume CHECKPOINT` とPBSスクリプトの `RESUME_CHECKPOINT` を追加した。
この変更後に作るcheckpointが対象。従来の重みだけのcheckpointからはoptimizerやLRの履歴を復元できないため、`--resume`では明示的にエラーにする。従来の`--model-ckpt`は重みを初期値として使う用途のまま。

## 保存と復元

- 初回学習前に `checkpoints/ckpt_initial.pth.tar` を保存。
- 学習・validation・validation依存scheduler更新が完了したepochごとに、`checkpoints/ckpt_EPOCH_1.pth.tar` を保存。
- `checkpoints/last.pth.tar` は最後に保存を完了したファイルを指す相対symlink。
- 一時ファイルへの保存、flush/fsync、renameの順で更新する。書き込み中の強制終了では前の`last.pth.tar`を使用できる。
- モデルに加えて各rankのAdamW状態、scheduler、AMP scaler（使用時）、Python/NumPy/PyTorch/CUDAの乱数状態、モデルbufferを復元する。lossとLRのログ用履歴も引き継ぐ。
- `cosineReduce`は現在のLRだけでなく、`t_epoch`、拡張後の`restart_period`、`restarts`、`eta_max`、iterationカウンタ、減衰callbackをすべて復元。warmup中の再開にも対応。
- epoch番号は0始まり。例えばepoch 23まで保存済みでepoch 24の途中に停止した場合、epoch 24を最初からやり直す。validation中に停止した場合も、そのepochの学習からやり直す。
- `--epochs 500`は「通算500 epochまで」で、再開後さらに500 epochではない。
- epoch依存のloss有効化やモデルinteractionも、復元したepoch番号に従う。

再開時は同じデータ・GPU数・batch size・loader設定・学習設定を使う。GPU数や学習設定が変わっていたらエラーにし、LRやglobal batchが意図せず変わることを防ぐ。データファイル自体も変更しないこと。epoch上限・ログ出力先・checkpoint出力先・進捗表示頻度は変更できる。

## Truth clusteringでエネルギー回帰を学習する

最新の学習段階には `shell/miyabi_train_energy_truth.sh` を使う。
4 GPU × batch 32（global batch 128）、実効初期LR=4e-4、cosineReduce、FP32、norm clipping上限10。
通常のmultiheadに `--cluster-energy-pooling --cluster-energy-source truth` を加え、学習とvalidationの両方で正解クラスタをpoolingに使う。
クラスタリングheadも従来のlossで学習する。既存のloss有効化条件を保ち、エネルギーlossが最適化に入るのはepoch番号4から（0始まり）。

データは `/work/gw25/w25002/data/tc_nnqq_2M_nobrems/train` と `validation`。
chunk/buffer/workersは8/256/2。過去の非poolingモデルとは異なるモデルなので、新しい保存先を使う。

```bash
cd /home/w25002/ml-pfa
qsub -v RUN_DIR=/work/gw25/w25002/checkpoint/energy_truth_run01 \
  shell/miyabi_train_energy_truth.sh
```

epoch境界の所要時間予測・自動再投入・cosineReduceを含む完全再開が有効。
後続ジョブにもtruth poolingとnorm clippingを明示的に引き継ぐ。
同じ保存先に今回のtruthモデルのcheckpointがあれば続きから再開する。
通算500 epochまたは最大25ジョブまで。手動停止するまで無制限に実行する設定ではない。

```bash
qstat
tail -f /work/gw25/w25002/checkpoint/energy_truth_run01/ranks/rank0.log
```

停止は `touch /work/gw25/w25002/checkpoint/energy_truth_run01/STOP_AUTO`。
再開時は旧ジョブの終了を確認してこのファイルを削除し、上のqsubを再実行する。
predictedへの切り替えは自動では行わない。truthで回帰性能を確認した後、別の学習・評価段階として扱う。

CPU・模擬テスト54件で、truth poolingの勾配、完全再開後の重み更新、起動引数と自動再投入時の設定保持を確認済み。
このpooling版のGPU本番・48時間をまたぐ連続実行は未検証。本番ジョブはこの実装作業では投入していない。

## 4 GPUでの投入例（実行はしていない）

初回本番：

```bash
cd /home/w25002/ml-pfa
qsub -q regular-g -l select=4:mpiprocs=1 -l walltime=48:00:00 \
  -v BENCHMARK_BATCHES=0 \
  shell/miyabi_train_energy_fast.sh
```

PBS出力の `Outputs:` 行に出るRUN_DIRを控える。48時間上限・手動qdel・エラーで停止した後は、同じRUN_DIRを指定して再投入する。次のパスの`JOBID`は初回ジョブの実際の値に置換する：

```bash
cd /home/w25002/ml-pfa
train_run=/work/gw25/w25002/checkpoint/energy_4gpu_JOBID
qsub -q regular-g -l select=4:mpiprocs=1 -l walltime=48:00:00 \
  -v "BENCHMARK_BATCHES=0,RUN_DIR=$train_run,RESUME_CHECKPOINT=$train_run/checkpoints/last.pth.tar" \
  shell/miyabi_train_energy_fast.sh
```

初回に既定値以外の`BATCH_SIZE`などを指定した場合、再投入でも同じ値を指定する。現在の既定値は4 GPU × batch 32（global batch 128）。

`RUN_DIR`を同じにすると `ranks/rank0.log` に追記する。中断したepochの途中ログは残り、`RESUME rank=0 next_epoch=...` の行以後が再実行分。最新保存を確認する行は `CHECKPOINT completed_epoch=... next_epoch=...`。

```bash
tail -f "$train_run/ranks/rank0.log"
```

同じRUN_DIRに複数のジョブを同時に走らせない。上記は手動再投入の手順。自動早期終了は行わない。

## 自動で次の48時間ジョブへ継続する場合

専用の `shell/miyabi_train_energy_auto.sh` を使う。screenやSSH接続の維持は不要。
4 GPU、各GPUのbatch size 32、global batch 128。cosineReduceなどはfast版と同じ設定で、速度測定は無効。

```bash
cd /home/w25002/ml-pfa
qsub -v RUN_DIR=/work/gw25/w25002/checkpoint/energy_run01 \
  shell/miyabi_train_energy_auto.sh
```

保存先に `checkpoints/last.pth.tar` があれば自動的に再開する。なければ新規学習を始める。
各epochの学習・validation・checkpoint保存が完了した境界で、次のepochが残り時間に収まるか判定する。
予測時間は「直近5 epochの最大所要時間 × 1.2」。PBSの48時間上限から終了処理用の10分も差し引く。
予測時間が残り時間以上なら、次のepochを開始せず正常終了し、現在のジョブの終了を待つ依存ジョブを1件投入する。
そのジョブは同じ保存先・同じ学習設定で、最後に完了したepochの次から再開する。通常は途中epochの計算を捨てずに済む。
実測値のない各ジョブの最初のepochは開始する。予測が外れた場合の対策として、上限10分前の強制停止も残す。
その対策が発動した場合のみ、途中のepochを最初からやり直す。
キュー待ちが入るため、次の実行がすぐ始まるとは限らない。

通算500 epoch到達、実行エラー、手動停止、1回の実行中に完了epochが増えなかった場合は継続しない。
初回を含め最大25ジョブまで（`AUTO_MAX_JOBS`で変更可）。自動投入でも通常のGPU時間・トークンを消費する。
実機の48時間をまたぐ連続投入は未検証。ノード障害や再投入処理の失敗時は自動継続せずログにエラーを残す。

手動停止には次のファイルを作る。実行中なら約30秒以内に停止処理を始め、以降の自動再投入を止める。
すでに後続ジョブがキューに入っていた場合も、そのジョブは学習開始前にこのファイルを確認して終了する。

```bash
touch /work/gw25/w25002/checkpoint/energy_run01/STOP_AUTO
```

ジョブID・継続履歴は `RUN_DIR/auto_jobs.jsonl`、直近の後続ジョブIDは `RUN_DIR/next_job_id`。
`ranks/rank0.log` の `EPOCH_TIME` に実測時間、`EPOCH_BUDGET` に予測時間・残り時間・開始/終了の判断を記録する。
状態確認はMiyabiでは `qstat` を使う（この環境のラッパーは `qstat -u` に対応しない）。
手動停止後に再開したいときは、旧ジョブがすべて終了してから `STOP_AUTO` を削除し、上のqsubを再実行する。

## 検証範囲

CPUテストで、25 epochの連続実行と「途中のepochを捨てて再開」の各batchのLR、最終重み、AdamW状態の完全一致を確認。初回epoch、warmup中、cosine restart付近、最大LRの減衰後を検証した。2 rankのCPU DDPでもrankごとの乱数・optimizer・bufferと次epochの結果を確認した。書き込み失敗時の旧checkpoint維持、旧形式・設定変更の拒否、ReduceLROnPlateauの復元も検証済み。

GPU本番での48時間中断試験は未実施。CUDA演算の非決定性による数値差までは保証しない。速度測定ではcheckpointを保存しない。
