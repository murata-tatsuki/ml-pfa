#!/bin/bash
#PBS -N pfa-energy-auto
#PBS -q regular-g
#PBS -W group_list=gw25
#PBS -l select=4:mpiprocs=1
#PBS -l walltime=48:00:00
#PBS -j oe

# First submission (4 GPU, batch/GPU=32, cosineReduce):
# qsub -v RUN_DIR=/work/gw25/w25002/checkpoint/energy_run01 shell/miyabi_train_energy_auto.sh
# Existing checkpoints in RUN_DIR are resumed automatically.
# Stop this chain, including a queued successor: touch "$RUN_DIR/STOP_AUTO"
# Default cap: AUTO_MAX_JOBS=25 including this first job; EPOCHS=500 total.
# Stop at an epoch boundary if the next epoch's predicted time will not fit.
# Fallback timeout: 10 minutes before PBS walltime; redo a partial epoch if needed.
set -eo pipefail
cd /home/w25002/ml-pfa
exec python3 -u -B tools/miyabi_auto_resume.py
