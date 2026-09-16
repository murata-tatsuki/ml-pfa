#!/bin/sh

cd ..

checkpoint_path=/data/murata/master/checkpoint
output_path=output/energy_regression_1to1
outD=5
D=5
input_dim=7


file="$2"
idx="$1"
GPU_OVERRIDE="${3:-${RUN_GPU:-}}"

if [ -n "${GPU_OVERRIDE}" ]; then
  case "${GPU_OVERRIDE}" in
    *[!0-9]*)
      echo "GPU override must be a non-negative integer: ${GPU_OVERRIDE}" >&2
      exit 1
      ;;
  esac
  GPU="${GPU_OVERRIDE}"
else
  NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
  if [ "${NGPU}" -le 0 ]; then
    echo "No CUDA device is available." >&2
    exit 1
  fi
  GPU=$((idx % ${NGPU}))
fi
# GPU=$((idx % 2))
# energy=200
outD=5



train_particle=nnqq_2M
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_06_30_151610_outputD5
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2026_08_17_174838_outputD5_multihead
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2026_08_17_174900_outputD5
epoch=27



base=$(basename "$file" .h5)
inputEnergy=$(basename $(dirname "$file"))

mkdir -p ${output_path}/${outdirectory}

outdirectory=skimmed/tc_${train_particle}/${outD}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/${inputEnergy}/multi-head

CUDA_VISIBLE_DEVICES=$GPU python save_root_reco_scan_once.py \
  "$file" \
  "${checkpoint}/ckpt_${epoch}_1.pth.tar" \
  "${output_path}/${outdirectory}" \
  0 5000 False 7 5 \
  --energy-regression \
  --energy-regression-cluster \
  --momentum \
  --momentum-amp \
  --event-total-energy \
  --model-variant multihead \
  --device cuda:0 \
  --beta-d-scan




