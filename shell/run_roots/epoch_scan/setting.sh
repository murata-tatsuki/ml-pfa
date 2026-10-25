#!/usr/bin/env bash
# このserverで実行する担当を設定し、bash setting.sh で起動します。
# cd /data/murata/master/shell/run_roots/epoch_scan

# bash setting.sh --epochs all --workers-per-gpu 20 --gpus 0 --energy 91

# bash setting.sh --model 5head_pid_onehot --epochs all --energy 200 --gpus 0 --workers-per-gpu 20

set -euo pipefail

# scan_config.json の name を指定します。
# 2head / 5head_pid / 5head_pid_onehot / 5head_pid_onehot_truth_gather / 5head_pid_onehot_preicted_gather
model=${MODEL:-5head_pid_onehot}
energy=${ENERGY:-200}       # 40 / 91 / 200 / 350 / 500 GeV
epochs=${EPOCHS:-0:20}     # 0:20 は0〜19。全保存epochは all。
gpus=${GPUS:-0,1}          # このserverで使うGPU番号。複数ならカンマ区切り。
workers_per_gpu=${WORKERS_PER_GPU:-10}  # 各GPUで同時に動かすsave_root数。
worker_mode=${WORKER_MODE:-persistent} # persistent: epoch間でworkerを再利用。subprocess: 従来方式。

# files-per-flavor はenergyから自動選択します。
# 40/91: 3、200/350: 8、500: 17。上書きは --files-per-flavor N。
# 同じ設定で再実行すると同じ計画を再開します。
# all のcheckpoint一覧を更新したい場合などは --tag next を指定します。
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$script_dir/run_scan.sh" \
  --model "$model" --energy "$energy" --epochs "$epochs" --gpus "$gpus" \
  --workers-per-gpu "$workers_per_gpu" --worker-mode "$worker_mode" "$@"
