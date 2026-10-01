#!/bin/sh

cd /home/murata/master

checkpoint_path=/home/murata/master/checkpoint
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2026_08_17_174838_outputD5_multihead
epoch=27

input_dir=/home/murata/data_murata/data/tc/tc_nnqq_2M
num_events=3

# 再実行時に既存の結果と重ならない出力先
run_tag=$(date +%Y%m%d_%H%M%S)
outbase=embedding_visualization/results/epoch${epoch}_${run_tag}

for sample in uu dd; do
    python embedding_visualization/extract.py \
        --checkpoint "${checkpoint}/ckpt_${epoch}_1.pth.tar" \
        --input "${input_dir}/nnqq_0_${sample}_ft0.h5" \
        --output "${outbase}/${sample}/embeddings" \
        --embedding-dim 4 \
        --start 0 --num-events "${num_events}" \
        --thetaphi --momentum --momentum-amp \
        --gravnet-coordinates \
        --device cpu || break

    python embedding_visualization/plot.py \
        --input "${outbase}/${sample}/embeddings" \
        --output "${outbase}/${sample}/plots" \
        --all-coordinates --html \
        --max-points 500 \
        --perplexities 30 --seeds 42 || break

    echo "HTML: ${PWD}/${outbase}/${sample}/plots/index.html"
done




cd /home/murata/master

checkpoint_path=/home/murata/master/checkpoint
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2026_08_17_174838_outputD5_multihead
epoch=27

input_dir=/home/murata/data_murata/data/tc/tc_nnqq/test
run_tag=$(date +%Y%m%d_%H%M%S)
outbase=embedding_visualization/results/test_epoch${epoch}_${run_tag}

samples=(
    nnqq_109_dd_eL_pR
    nnqq_309_dd_eR_pL
    nnqq_509_uu_eL_pR
    nnqq_709_uu_eR_pL
)

for sample in "${samples[@]}"; do
    python embedding_visualization/extract.py \
        --checkpoint "${checkpoint}/ckpt_${epoch}_1.pth.tar" \
        --input "${input_dir}/${sample}.h5" \
        --output "${outbase}/${sample}/embeddings" \
        --embedding-dim 4 \
        --start 0 --num-events 1 \
        --thetaphi --momentum --momentum-amp \
        --gravnet-coordinates \
        --device cpu || break

    python embedding_visualization/plot.py \
        --input "${outbase}/${sample}/embeddings" \
        --output "${outbase}/${sample}/plots" \
        --all-coordinates --html \
        --max-points 500 \
        --perplexities 30 --seeds 42 || break

    echo "HTML: ${PWD}/${outbase}/${sample}/plots/index.html"
done