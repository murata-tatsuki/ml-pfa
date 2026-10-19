#!/bin/sh

# Resume-capable counterpart of run_root_scan_fast.sh.
# Adjust partType, energy, GPU, and -P in the same way as the original script.

# 40 GeV   500 events/file: 7 files/flavour (10,500 events total)
# 91 GeV   500 events/file: 7 files/flavour (10,500 events total)
# 200 GeV  200 events/file: 17 files/flavour (10,200 events total)
# 350 GeV  200 events/file: 17 files/flavour (10,200 events total)
# 500 GeV  100 events/file: 34 files/flavour (10,200 events total)

partType=dd
energy=350
GPU=0

ls \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_001.h5 \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_002.h5 \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_003.h5 \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_004.h5 \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_005.h5 \
  /data/suehara/mldata/pfa/murata/data/raw/fixed_uds_pandoraFixed_nobrems/${partType}/${energy}GeV/*_006.h5 \
| nl -v0 \
| xargs -r -n2 -P10 env RUN_GPU=$GPU FAST_SCAN_WORKERS=2 bash run_root_parallel_scan_fast_resume.sh
