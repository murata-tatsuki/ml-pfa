#!/usr/bin/env bash
# ls /data/suehara/mldata/pfa/murata/data/tc/tc_fixed_uds_pandoraFixing/40GeV/uu*.h5 | nl -v0 | xargs -r -n2 -P10 env RUN_GPU=0 FAST_SCAN_WORKERS=2 bash /data/murata/master/shell/run_root_parallel_5head_truth.sh
# ... | nl -v0 | xargs -r -n2 -P10 env RUN_GPU=0 FAST_SCAN_WORKERS=3 bash run_root_parallel_5head.sh
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
cd "$repo_dir"

if (( $# < 2 )); then
  echo "Usage: $0 INDEX INPUT.h5 [GPU]" >&2
  exit 2
fi
idx=$1
file=$2
gpu_override=${3:-${RUN_GPU:-}}
python_bin=${PYTHON_BIN:-python3}
pid_threads=${PID_THREADS:-${FAST_SCAN_WORKERS:-3}}
nend=${ROOT_NEND:--1}
device=${ROOT_DEVICE:-cuda:0}
replace=${ROOT_REPLACE:-0}

if [[ ! $idx =~ ^[0-9]+$ ]]; then
  echo "INDEX must be a non-negative integer: $idx" >&2
  exit 2
fi
if [[ ! -f $file || $file != *.h5 ]]; then
  echo "Input H5 does not exist: $file" >&2
  exit 2
fi
file=$(realpath -- "$file")
if [[ ! $pid_threads =~ ^[1-9][0-9]*$ ]]; then
  echo "PID_THREADS / FAST_SCAN_WORKERS must be positive: $pid_threads" >&2
  exit 2
fi
if [[ $nend != -1 && ! $nend =~ ^[1-9][0-9]*$ ]]; then
  echo "ROOT_NEND must be a positive event count or -1: $nend" >&2
  exit 2
fi
if [[ $replace != 0 && $replace != 1 ]]; then
  echo "ROOT_REPLACE must be 0 or 1" >&2
  exit 2
fi
python_bin=$(command -v "$python_bin")

if [[ -n $gpu_override ]]; then
  if [[ ! $gpu_override =~ ^[0-9]+$ ]]; then
    echo "GPU override must be a non-negative integer: $gpu_override" >&2
    exit 2
  fi
  gpu=$gpu_override
elif [[ $device == cpu ]]; then
  gpu=0
else
  ngpu=$("$python_bin" -B -c 'import torch; print(torch.cuda.device_count())')
  if (( ngpu <= 0 )); then
    echo "No CUDA device is available." >&2
    exit 1
  fi
  gpu=$((idx % ngpu))
fi

checkpoint=/data/murata/master/checkpoint/energy_regression/ckpts_gravnet_new02_2026_10_04_162529_outputD5_multihead/ckpt_57_1.pth.tar
if [[ ! -f $checkpoint ]]; then
  echo "Checkpoint does not exist: $checkpoint" >&2
  exit 1
fi

# Select compatibility from the actual supplied H5; never substitute another dataset.
input_format=$("$python_bin" -B - "$file" <<'PY'
import sys
import h5py
from extended_h5 import TRAINING_SCHEMAS
with h5py.File(sys.argv[1], 'r') as f:
    schema = f.attrs.get('schema_version')
    if schema in TRAINING_SCHEMAS:
        print('extended')
    elif schema is None and 'row_info' not in f and 'collections' not in f:
        if 'feature' not in f or 'label' not in f:
            raise SystemExit('Legacy input requires feature and label builders')
        print('legacy')
    else:
        raise SystemExit(f'Unsupported input schema: {schema!r}')
PY
)

base=$(basename -- "$file" .h5)
input_energy=$(basename -- "$(dirname -- "$file")")
output_base=${ROOT_OUTPUT_BASE:-$repo_dir/output/energy_regression_1to1}
# output_directory=$output_base/skimmed/tc_nnqq_2M/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/$input_energy/5head_pid/tbeta090td050
output_directory=$output_base/skimmed/tc_nnqq_2M/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/$input_energy/5head_pid/truth_clustering
output_file=$output_directory/$base.root
# Python validates existing outputs for default resume; ROOT_REPLACE=1 regenerates.
mkdir -p -- "$output_directory"

input_options=(--pid-input-format "$input_format")
if [[ $input_format == extended ]]; then
  input_options+=(--exclude-gap-hits)
else
  echo "[PID] Legacy inputs: known-MC rows; collection-based gap exclusion unavailable."
fi
if [[ $replace == 1 ]]; then
  input_options+=(--replace)
fi

# FAST_SCAN_WORKERS is accepted as a CPU-thread alias here; beta/td are fixed.
echo "[PID] index=$idx GPU=$gpu epoch=57 input=$file output=$output_file"
CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -B "$repo_dir/save_root.py" \
  "$file" "$checkpoint" "$output_file" \
  0 "$nend" False 7 5 \
  --pid "${input_options[@]}" \
  --energy-heads 5 \
  --momentum --momentum-amp --model-variant multihead \
  --device "$device" --pid-threads "$pid_threads" \
  --root-chunk-events "${ROOT_CHUNK_EVENTS:-10}" \
  --pipeline-depth "${PIPELINE_DEPTH:-40}" \
  --tbeta 0.9 --td 0.5 \
  --truth-clustering
