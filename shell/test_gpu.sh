#!/bin/bash
#PBS -q debug-g
#PBS -W group_list=gw25
#PBS -l select=1:ncpus=72
#PBS -l walltime=00:05:00

cd ..

module load singularity

singularity exec --nv \
  -B /work:/work \
  -B /home/w25002:/home/w25002 \
  /work/gw25/w25002/singularity/pfa_arm64.sif \
  python3 /home/w25002/ml-pfa/train.py \
    -i /home/w25002/work/data/train \
    -ii /home/w25002/work/data/validation \
    --no-split \
    --thetaphi \
    --cuda cuda:0 \
    --epochs 1 \
    --batch-size 4 \
    --output-dimension 5 \
    --ckptdir /home/w25002/work/checkpoint/test \
    --momentum \
    --momentum-amp \
    --qmin 0.2 \
    --learning-rate 2e-5 \
    --lr-policy cosineReduce \
    --clip-value 10
