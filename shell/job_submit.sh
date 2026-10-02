#!/bin/bash
#PBS -q debug-g
#PBS -W group_list=gw25
#PBS -l select=1:ncpus=72
#PBS -l walltime=00:20:00

cd /home/w25002/ml-pfa

module load singularity

outputD=5
ncuda=1

singularity exec --nv \
  -B /work:/work \
  -B /home/w25002:/home/w25002 \
  /work/gw25/w25002/singularity/pfa_arm64.sif \
  python train.py -i /home/w25002/work/data/train -ii /home/w25002/work/data/validation --no-split --thetaphi --cuda cuda:0 --epochs 500 --beta-track --force-track-alpha --batch-size 32 --output-dimension ${outputD} --ckptdir /home/w25002/work/checkpoint/test --momentum --momentum-amp --qmin 0.2 --learning-rate 2e-5 --lr-policy cosineReduce --clip-value 10 