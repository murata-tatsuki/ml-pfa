#!/bin/bash
#PBS -q debug-g
#PBS -W group_list=gw25
#PBS -l select=1
#PBS -l walltime=00:10:00

cd /home/w25002/ml-pfa

mkdir -p /work/gw25/w25002/checkpoint/run_1gpu

alphbeta=alpha_tracker_diff_log_perCluster
sumdis=sum_log_perCluster

# IPv4通信の強制とマスターノード設定
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT="29500"
export GLOO_SOCKET_IFNAME="lo"

module load singularity

singularity exec --nv \
  -B /work:/work \
  -B /home/w25002:/home/w25002 \
  /work/gw25/w25002/singularity/pfa_arm64.sif \
  python train.py \
    -i /work/gw25/w25002/data/tc_nnqq_2M \
    -ii /work/gw25/w25002/data/validation \
    --no-split \
    --thetaphi \
    --epochs 500 \
    --batch-size 32 \
    --output-dimension 5 \
    --ckptdir /work/gw25/w25002/checkpoint/run_1gpu \
    --momentum \
    --momentum-amp \
    --qmin 0.2 \
    --learning-rate 1e-4 \
    --lr-policy cosineReduce \
    --clip-value 10 \
    --energy-regression --energy-regression-cluster --LE-track ${alphbeta} --LE-cluster ${sumdis} \
    --epochs-nobeta 1 --epochs-noLE 3 --use-multihead-model --multihead-regression-heads 2 --multihead-interaction-mode none