#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "${script_dir}/.." && pwd)

if (( $# < 2 )); then
  echo "Usage: $0 INDEX INPUT.h5 [GPU]" >&2
  exit 2
fi

idx=$1
file=$2
gpu_override=${3:-${RUN_GPU:-}}

if [[ ! $idx =~ ^[0-9]+$ ]]; then
  echo "INDEX must be a non-negative integer: $idx" >&2
  exit 2
fi
if [[ ! -f $file || $file != *.h5 ]]; then
  echo "Input H5 does not exist: $file" >&2
  exit 2
fi

if [[ -n $gpu_override ]]; then
  if [[ ! $gpu_override =~ ^[0-9]+$ ]]; then
    echo "GPU override must be a non-negative integer: $gpu_override" >&2
    exit 2
  fi
  gpu=$gpu_override
else
  ngpu=$(python -c 'import torch; print(torch.cuda.device_count())')
  if (( ngpu <= 0 )); then
    echo "No CUDA device is available." >&2
    exit 1
  fi
  gpu=$((idx % ngpu))
fi

checkpoint=/data/murata/master/checkpoint/energy_regression/ckpts_gravnet_new02_2026_08_17_174900_outputD5/ckpt_66_1.pth.tar
output_base=${repo_dir}/output/energy_regression_1to1
input_dim=7
output_dim=5

base=$(basename -- "$file" .h5)
input_energy=$(basename -- "$(dirname -- "$file")")
output_directory=${output_base}/skimmed/tc_nnqq_2M/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/${input_energy}/mono-head/truth_clustering
output_file=${output_directory}/${base}.root
mkdir -p -- "$output_directory"

exec env CUDA_VISIBLE_DEVICES="$gpu" python "${repo_dir}/save_root_reco_w_Cedric_fast.py" \
  "$file" \
  "$checkpoint" \
  "$output_file" \
  0 5000000 False "$input_dim" "$output_dim" \
  --energy-regression \
  --momentum \
  --momentum-amp \
  --device cuda:0 \
  --tbeta 0.9 \
  --td 0.5 \
  --truth-clustering \
  --energy-regression-cluster \
  --event-total-energy \
  --root-chunk-events "${ROOT_CHUNK_EVENTS:-10}" \
  --pipeline-depth "${PIPELINE_DEPTH:-40}"
