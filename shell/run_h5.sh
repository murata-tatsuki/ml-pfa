#!/bin/sh

cd ..
# python save_pred.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test checkpoint/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar h5/outfile_display_edit.h5 0 10 0 7 3 0 False
python save_pred_pandora.py h5/ntau_10GeV_10_pandora/test checkpoint/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar h5/outfile_display_pandora_ntau.h5 0 1 0 7 3 0 True > log/h5_test.log
# python save_pred_pandora.py ../data_murata/uds_91_pandora/test checkpoint/ckpts_gravnet_new02_May13_1920/ckpt_19_1.pth.tar h5/outfile_display_pandora_uds.h5 0 1 0 7 3 0 True
