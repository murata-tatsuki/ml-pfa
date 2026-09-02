#!/bin/sh


ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/uu/40GeV/*.h5 | nl -v0 | xargs -n2 -P30 env RUN_GPU=1 bash run_root_parallel_truth.sh
ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/dd/40GeV/*.h5 | nl -v0 | xargs -n2 -P30 env RUN_GPU=1 bash run_root_parallel_truth.sh
ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/ss/40GeV/*.h5 | nl -v0 | xargs -n2 -P30 env RUN_GPU=1 bash run_root_parallel_truth.sh

ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/uu/91GeV/*.h5 | nl -v0 | xargs -n2 -P20 env RUN_GPU=0 bash run_root_parallel_truth.sh
ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/dd/91GeV/*.h5 | nl -v0 | xargs -n2 -P20 env RUN_GPU=0 bash run_root_parallel_truth.sh
ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/ss/91GeV/*.h5 | nl -v0 | xargs -n2 -P20 env RUN_GPU=0 bash run_root_parallel_truth.sh

ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/uu/200GeV/*.h5 | nl -v0 | xargs -n2 -P20 env RUN_GPU=2 bash run_root_parallel_truth.sh
ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/dd/200GeV/*.h5 | nl -v0 | xargs -n2 -P20 env RUN_GPU=2 bash run_root_parallel_truth.sh
ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/ss/200GeV/*.h5 | nl -v0 | xargs -n2 -P20 env RUN_GPU=2 bash run_root_parallel_truth.sh

ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/uu/500GeV/*.h5 | nl -v0 | xargs -n2 -P6 env RUN_GPU=3 bash run_root_parallel_truth.sh
ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/dd/500GeV/*.h5 | nl -v0 | xargs -n2 -P6 env RUN_GPU=3 bash run_root_parallel_truth.sh
ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/ss/500GeV/*.h5 | nl -v0 | xargs -n2 -P6 env RUN_GPU=3 bash run_root_parallel_truth.sh

ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/uu/350GeV/*.h5 | nl -v0 | xargs -n2 -P8 env RUN_GPU=0 bash run_root_parallel_truth.sh
ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/dd/350GeV/*.h5 | nl -v0 | xargs -n2 -P12 env RUN_GPU=1 bash run_root_parallel_truth.sh && ls /data/suehara/mldata/pfa/murata/data/raw/fixed_uds/ss/350GeV/*.h5 | nl -v0 | xargs -n2 -P12 env RUN_GPU=1 bash run_root_parallel_truth.sh
