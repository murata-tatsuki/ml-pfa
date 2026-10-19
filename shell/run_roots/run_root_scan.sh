#!/bin/sh

# 40 GeV	500	各flavour 7ファイル（計10,500 event）
# 91 GeV	500	各flavour 7ファイル（計10,500 event）
# 200 GeV	200	各flavour 17ファイル（計10,200 event）
# 350 GeV	200	各flavour 17ファイル（計10,200 event）
# 500 GeV	100	各flavour 34ファイル（計10,200 event）

partType=dd
energy=40
GPU=0


ls \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_001.h5 \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_002.h5 \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_003.h5 \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_004.h5 \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_005.h5 \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_006.h5 \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_007.h5 \
| nl -v0 \
| xargs -r -n2 -P7 env RUN_GPU=$GPU bash run_root_parallel_scan.sh