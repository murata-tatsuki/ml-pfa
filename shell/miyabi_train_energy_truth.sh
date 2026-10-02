#!/bin/bash
#PBS -N pfa-energy-truth
#PBS -q regular-g
#PBS -W group_list=gw25
#PBS -l select=4:mpiprocs=1
#PBS -l walltime=48:00:00
#PBS -j oe

# Truth-cluster pooling regression: 4 GPUs x batch 32, effective LR=4e-4.
# qsub -v RUN_DIR=/work/gw25/w25002/checkpoint/energy_truth_run01 shell/miyabi_train_energy_truth.sh
# Use a new RUN_DIR for this model; old non-pooling checkpoints cannot fully resume it.
set -eo pipefail
export CLUSTER_ENERGY_POOLING=1
export CLUSTER_ENERGY_SOURCE=truth
export CLIP_MODE=${CLIP_MODE:-norm}
export EXPECTED_NODES=${EXPECTED_NODES:-4}
export BATCH_SIZE=${BATCH_SIZE:-32}
export RUN_DIR=${RUN_DIR:-/work/gw25/w25002/checkpoint/energy_truth_${PBS_JOBID:?Submit with qsub}}
exec bash /home/w25002/ml-pfa/shell/miyabi_train_energy_auto.sh
