#!/bin/sh

cd ..

checkpoint_path=/home/murata/master/checkpoint
output_path=output/energy_regression_1to1
outD=5
D=5
input_dim=7


## fixed uds
  train_particle=fixed_uds
  energy=40
  qq=dd
  # test_path=/data/suehara/mldata/pfa/murata/data/tc/tc_fixed_uds/code_test
  # outdir=test.root
  # outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/${energy}GeV/tbeta090td050.root
  outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/${energy}GeV/test_dd_001.root
  # outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/${energy}GeV/raw_${qq}.root
  # test_path=/data/suehara/mldata/pfa/murata/data/raw/fixed_uds/${qq}/${energy}GeV
  test_path=/data/suehara/mldata/pfa/murata/data/raw/fixed_uds/${qq}/${energy}GeV/dd_001.h5
  # test_path=/data/suehara/mldata/pfa/murata/data/tc/tc_fixed_uds/${energy}GeV
  # test_path=/data/suehara/mldata/pfa/murata/data/tc/tc_fixed_uds/${energy}GeV/less_samples
  # test_path=/home/murata/data_murata/data/tc/tc_fixed_uds/code_test/40GeV_1file
  checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_06_30_151610_outputD5
  epoch=444
  # python save_root_____edit.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 5000000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:0 --tbeta 0.9 --td 0.5 --energy-regression-cluster
  # python save_root_reco.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 5000000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:0 --tbeta 0.9 --td 0.5 --energy-regression-cluster
##



# ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/${qq}/${energy}GeV/*.h5 \
# | nl -v0 \
# | xargs -n2 -P4 bash -c "

# idx=\$0
# file=\$1
# GPU=\$((idx % 2))

# base=\$(basename \$file .h5)

# outdir=skimmed/tc_${train_particle}/${outD}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/${energy}GeV/perh5file/\${base}.root

# CUDA_VISIBLE_DEVICES=\$GPU python save_root_reco.py \
#   \$file \
#   ${checkpoint}/ckpt_${epoch}_1.pth.tar \
#   ${output_path}/\${outdir} \
#   0 5000000 False ${input_dim} ${outD} \
#   --energy-regression \
#   --momentum \
#   --momentum-amp \
#   --device cuda:\$GPU \
#   --tbeta 0.9 \
#   --td 0.5 \
#   --energy-regression-cluster
# "

file="$2"
idx="$1"
NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
GPU=$((idx % ${NGPU}))
GPU=1
# GPU=$((idx % 2))
# energy=200
train_particle=fixed_uds_brems

base=$(basename "$file" .h5)
inputEnergy=$(basename $(dirname "$file"))

outdir=skimmed/tc_${train_particle}/${outD}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/${inputEnergy}/tbeta090td050/${base}.root
# outdir=skimmed/tc_${train_particle}/${outD}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/${inputEnergy}/truth_clustering/${base}.root

CUDA_VISIBLE_DEVICES=$GPU python save_root_reco.py \
  "$file" \
  "${checkpoint}/ckpt_${epoch}_1.pth.tar" \
  "${output_path}/${outdir}" \
  0 5000000 False ${input_dim} ${outD} \
  --energy-regression \
  --momentum \
  --momentum-amp \
  --device cuda:0 \
  --tbeta 0.9 \
  --td 0.5 \
  --energy-regression-cluster \
  --event-total-energy



  # python save_root_reco.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 5000000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:0 --tbeta 0.9 --td 0.5 --energy-regression-cluster
##
# ls /data/suehara/mldata/pfa/murata/data/tc/tc_fixed_uds/40GeV/*.h5 | nl -v0 | xargs -n2 -P8 bash run_root_parallel.sh
# ls /data/suehara/mldata/pfa/murata/data/tc/tc_fixed_uds_brems/40GeV/*.h5 | nl -v0 | xargs -n2 -P12 bash run_root_parallel.sh