#!/bin/bash
# One-time CPU compilation for the existing ARM64 PyTorch container.
# Run before submitting GPU jobs; no GPU allocation is required for compilation.
set -eo pipefail
module load singularity
set -u
source_dir=/home/w25002/pytorch_cmspepr
expected_commit=e94c49b5326f10a7ac8a634a873090fd42bab10a
if [[ ! -d "$source_dir/.git" ]]; then
    git clone https://github.com/cms-pepr/pytorch_cmspepr.git "$source_dir"
    git -C "$source_dir" checkout --detach "$expected_commit"
fi
if [[ $(git -C "$source_dir" rev-parse HEAD) != "$expected_commit" ]]; then
    echo "Unexpected torch_cmspepr source revision; inspect before building." >&2
    exit 1
fi
singularity exec -B /work:/work \
    /work/gw25/w25002/singularity/pfa_arm64.sif \
    env CC=/usr/bin/gcc CXX=/usr/bin/g++ CUDA_HOME=/usr/local/cuda \
        FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=2 \
        python -m pip install --user --no-deps --no-build-isolation "$source_dir"
