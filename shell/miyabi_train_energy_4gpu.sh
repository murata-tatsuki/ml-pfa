#!/bin/bash
#PBS -N pfa-energy-4gpu
#PBS -q debug-g
#PBS -W group_list=gw25
#PBS -l select=4:mpiprocs=1
#PBS -l walltime=00:10:00
#PBS -j oe

# Submit from /home/w25002/ml-pfa. Override TRAIN_DIR and VALID_DIR with qsub -v.
# Start with debug-g; for training use: qsub -q regular-g -l walltime=24:00:00 ...
# Timing-based defaults: see miyabi_training_recommendation.md. Accuracy is not tuned.
set -eo pipefail

PROJECT_DIR=/home/w25002/ml-pfa
CONTAINER=/work/gw25/w25002/singularity/pfa_arm64.sif
TRAIN_DIR=${TRAIN_DIR:-/work/gw25/w25002/data/tc_nnqq_2M_nobrems/train}
VALID_DIR=${VALID_DIR:-/work/gw25/w25002/data/tc_nnqq_2M_nobrems/validation}
: "${PBS_NODEFILE:?Submit this script with qsub}"
: "${PBS_JOBID:?Submit this script with qsub}"

for input_path in "$TRAIN_DIR" "$VALID_DIR" "$CONTAINER"; do
    if [[ ! -e "$input_path" ]]; then
        echo "Missing input: $input_path" >&2
        exit 1
    fi
done
NNODES=${EXPECTED_NODES:-4}
if [[ $(sort -u "$PBS_NODEFILE" | wc -l) -ne "$NNODES" ]]; then
    echo "This script requires $NNODES distinct Miyabi-G nodes." >&2
    exit 1
fi

module purge
module load nvidia nv-hpcx singularity
set -u
cd "$PROJECT_DIR"

# One common rendezvous address and output directory for all workers.
MASTER_ADDR=$(head -n 1 "$PBS_NODEFILE")
MASTER_PORT=${MASTER_PORT:-29500}
RUN_DIR=${RUN_DIR:-/work/gw25/w25002/checkpoint/energy_${NNODES}gpu_${PBS_JOBID}}
mkdir -p "$RUN_DIR/ranks"
export MASTER_ADDR MASTER_PORT PROJECT_DIR CONTAINER
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1
# Loopback-only settings from the single-GPU job prevent communication across nodes.
unset GLOO_SOCKET_IFNAME NCCL_SOCKET_IFNAME

args=(
    train.py
    -i "$TRAIN_DIR" -ii "$VALID_DIR"
    --no-split --thetaphi --epochs "${EPOCHS:-500}"
    --beta-track --force-track-alpha
    --batch-size "${BATCH_SIZE:-32}" --output-dimension 5
    --ckptdir "$RUN_DIR/checkpoints"
    --energy-regression --energy-regression-cluster
    --LE-track alpha_tracker_diff_log_perCluster --LE-cluster sum_log_perCluster
    --momentum --momentum-amp --qmin 0.2
    --learning-rate "${LEARNING_RATE:-4e-4}" --ddp-lr-scaling "${DDP_LR_SCALING:-none}"
    --weight-decay "${WEIGHT_DECAY:-1e-4}"
    --lr-policy "${LR_POLICY:-cosineReduce}" --clip-value "${CLIP_VALUE:-10}"
    --clip-mode "${CLIP_MODE:-value}" --seed "${TRAIN_SEED:-1009}"
    --ddp --ddp-log-dir "$RUN_DIR/ranks"
    --progress-rank 0 --progress-mininterval 20 --rank-log-interval "${RANK_LOG_INTERVAL:-100}"
    --ilc-streaming --stream-files-per-chunk "${STREAM_FILES_PER_CHUNK:-8}"
    --stream-shuffle-buffer "${STREAM_SHUFFLE_BUFFER:-256}"
    --num-workers "${NUM_WORKERS:-2}" --epochs-nobeta "${EPOCHS_NOBETA:-1}" --epochs-noLE "${EPOCHS_NOLE:-3}"
    --use-multihead-model --multihead-regression-heads 2
    --multihead-interaction-mode none --extended-h5-input --exclude-gap-hits
)
case ${CLUSTER_ENERGY_POOLING:-0} in
    0) ;;
    1)
        case ${CLUSTER_ENERGY_SOURCE:-truth} in
            truth|predicted) ;;
            *) echo "CLUSTER_ENERGY_SOURCE must be truth or predicted" >&2; exit 1 ;;
        esac
        args+=(--cluster-energy-pooling --cluster-energy-source "${CLUSTER_ENERGY_SOURCE:-truth}")
        ;;
    *) echo "CLUSTER_ENERGY_POOLING must be 0 or 1" >&2; exit 1 ;;
esac
if [[ -n ${AUTO_TRAIN_DEADLINE:-} ]]; then
    args+=(--epoch-budget-deadline "$AUTO_TRAIN_DEADLINE"
           --epoch-budget-stop-file "${AUTO_EPOCH_STOP_FILE:?Missing epoch boundary signal path}")
fi
if [[ -n ${RESUME_CHECKPOINT:-} ]]; then
    if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
        echo "Missing resume checkpoint: $RESUME_CHECKPOINT" >&2
        exit 1
    fi
    if [[ ${BENCHMARK_BATCHES:-0} -gt 0 ]]; then
        echo "RESUME_CHECKPOINT cannot be used with BENCHMARK_BATCHES > 0" >&2
        exit 1
    fi
    args+=(--resume "$RESUME_CHECKPOINT")
fi
if [[ ${BENCHMARK_BATCHES:-0} -gt 0 ]]; then
    # Exercise the full loss used after the initial warm-up epochs.
    args+=(--benchmark-batches "$BENCHMARK_BATCHES" --benchmark-warmup "${BENCHMARK_WARMUP:-2}"
           --epochs-nobeta -1 --epochs-noLE -1 --rank-log-interval 1)
    if [[ ${BENCHMARK_LOADER_SWEEP:-1} == 0 ]]; then
        args+=(--no-benchmark-loader-sweep)
    fi
    if [[ -n ${BENCHMARK_BATCH_SIZES:-} ]]; then
        IFS=: read -r -a bench_sizes <<< "$BENCHMARK_BATCH_SIZES"
        args+=(--benchmark-batch-sizes "${bench_sizes[@]}")
    fi
fi
if [[ ${LR_WARMUP_EPOCHS:-0} -gt 0 ]]; then
    args+=(--lr-warmup --lr-warmup-epochs "$LR_WARMUP_EPOCHS")
fi
if [[ ${AMP:-0} == 1 ]]; then
    args+=(--amp --amp-dtype "${AMP_DTYPE:-bf16}")
fi

echo "Nodes: $(sort -u "$PBS_NODEFILE" | tr '\n' ' ')"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Training: $TRAIN_DIR; validation: $VALID_DIR"
echo "Outputs: $RUN_DIR"
echo "Model: multihead; cluster energy pooling=${CLUSTER_ENERGY_POOLING:-0}; source=${CLUSTER_ENERGY_SOURCE:-truth}; clip-mode=${CLIP_MODE:-value}"
echo "Main training log: $RUN_DIR/ranks/rank0.log"
echo "Resume checkpoint after each complete epoch: $RUN_DIR/checkpoints/last.pth.tar"
date -Is

# MPI starts one torchrun launcher per node; each launcher starts one GPU worker.
# --gpus 0,1,2,3 must not be passed: each Miyabi-G node exposes only GPU 0.
mpiexec -n "$NNODES" --map-by ppr:1:node --bind-to none \
    env PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
    MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" \
    PROJECT_DIR="$PROJECT_DIR" CONTAINER="$CONTAINER" \
    OMP_NUM_THREADS="$OMP_NUM_THREADS" PYTHONUNBUFFERED=1 NNODES="$NNODES" \
    bash -c '
        cd "$PROJECT_DIR"
        exec singularity exec --nv \
            -B /work:/work -B /home/w25002:/home/w25002 \
            "$CONTAINER" \
            python -u -B -m torch.distributed.run \
                --nnodes="$NNODES" --nproc-per-node=1 \
                --node-rank="$OMPI_COMM_WORLD_RANK" \
                --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
                "$@"
    ' bash "${args[@]}"
