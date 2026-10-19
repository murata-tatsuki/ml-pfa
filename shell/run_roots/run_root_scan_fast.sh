#!/bin/sh

# 40 GeV	500	各flavour 7ファイル（計10,500 event）
# 91 GeV	500	各flavour 7ファイル（計10,500 event）
# 200 GeV	200	各flavour 17ファイル（計10,200 event）
# 350 GeV	200	各flavour 17ファイル（計10,200 event）
# 500 GeV	100	各flavour 34ファイル（計10,200 event）

partType=ss
energy=350
GPU=0


ls \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_01?.h5 \
| nl -v0 \
| xargs -r -n2 -P10 env RUN_GPU=$GPU FAST_SCAN_WORKERS=3 bash run_root_parallel_scan_fast.sh