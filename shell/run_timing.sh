#!/bin/bash
python train.py \
  -i /data/suehara/mldata/pfa/murata/ntau_lessSample/train \
  -ii /data/suehara/mldata/pfa/murata/ntau_lessSample/validation \
  --no-split \
  --cuda cuda:0 \
  --epochs 50 \
  --batch-size 8 \
  --ilc-sharded \
  --output-dimension 2 \
  --ckptdir checkpoint_arpad/test_timing \
  --timing \
  > log_Arpad/test_timing.log 2>&1