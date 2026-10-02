#!/bin/bash
#PBS -N pfa-energy-fast
#PBS -q debug-g
#PBS -W group_list=gw25
#PBS -l select=4:mpiprocs=1
#PBS -l walltime=00:03:00
#PBS -j oe

# Requested profile: 4 GPU x batch 32, global batch=128, optimizer LR=4e-4.
# By default this is a short benchmark, not a 500-epoch production run.
# See miyabi_training_fastest.md for measured results and current status.
set -eo pipefail
export EXPECTED_NODES=${EXPECTED_NODES:-4}
export BATCH_SIZE=${BATCH_SIZE:-32}
export LEARNING_RATE=${LEARNING_RATE:-4e-4}
export DDP_LR_SCALING=none
export STREAM_FILES_PER_CHUNK=${STREAM_FILES_PER_CHUNK:-8}
export STREAM_SHUFFLE_BUFFER=${STREAM_SHUFFLE_BUFFER:-256}
export NUM_WORKERS=${NUM_WORKERS:-2}
export AMP=${AMP:-0}
export RANK_LOG_INTERVAL=${RANK_LOG_INTERVAL:-100}
if [[ -n ${RESUME_CHECKPOINT:-} ]]; then
    export BENCHMARK_BATCHES=${BENCHMARK_BATCHES:-0}
else
    export BENCHMARK_BATCHES=${BENCHMARK_BATCHES:-10}
fi
export BENCHMARK_LOADER_SWEEP=${BENCHMARK_LOADER_SWEEP:-0}
exec bash /home/w25002/ml-pfa/shell/miyabi_train_energy_4gpu.sh
