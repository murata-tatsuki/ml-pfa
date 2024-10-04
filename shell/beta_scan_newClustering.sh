#!/bin/sh

cd ..

# DATE=`date '+%Y_%m_%d_%H%M%S'`
# echo ${DATE}

## scan only tbeta (0-99%, 1%)
# python save_root_beta_scan.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test /home/murata/master/checkpoint/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5/ckpt_49_1.pth.tar output/hyper_parameter/beta_threshold/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10 0 500000 False 7 5 False

## scan tbeta (0.0-0.9, 0.1 increment) and td (0.1-0.9, 0.1 increment) D=5
  # ntau train ntau test
# python save_root_tbeta_td_scan.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test /home/murata/master/checkpoint/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5/ckpt_49_1.pth.tar output/new_clustering/hyper_parameter/tbeta_td/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10/tbeta010td010 0 500000 False 7 5 False
  # ntau train jet test
# python save_root_tbeta_td_scan.py /data/suehara/gravnet_ilc/data/uds91/test /home/murata/master/checkpoint/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5/ckpt_49_1.pth.tar output/hyper_parameter/tbeta_td/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_uds91/tbeta010td010 0 500000 False 7 5 False
  # jet train jet test
# python save_root_tbeta_td_scan.py /data/suehara/gravnet_ilc/data/uds91/test /home/murata/master/checkpoint/output_dimensions/ckpts_gravnet_new02_2024_06_30_065937_outputD5/ckpt_49_1.pth.tar output/new_clustering/hyper_parameter/tbeta_td/tc_uds91_timingcut_forcealpha_thetaphi_5D_49_uds91/tbeta010td010 0 500000 False 7 5 False
python save_root_tbeta_td_scan_reversed.py /data/suehara/gravnet_ilc/data/uds91/test /home/murata/master/checkpoint/output_dimensions/ckpts_gravnet_new02_2024_06_30_065937_outputD5/ckpt_49_1.pth.tar output/new_clustering/hyper_parameter/tbeta_td/tc_uds91_timingcut_forcealpha_thetaphi_5D_49_uds91/tbeta010td010_reversed 0 500000 False 7 5 False
# python save_root.py /data/suehara/gravnet_ilc/data/uds91/test ${checkpoint_path}/ckpts_gravnet_new02_May13_1920/ckpt_19_1.pth.tar output/uds_to_uds/gravnet/tc_usd91_timingcut_forcealpha_thetaphi_19_usd.root 0 10000 False 7 3 False

## scan tbeta (0.01-0.09, 0.01 increment) and td (0.1-0.9, 0.1 increment)
  # ntau train ntau test
# python save_root_tbeta_below01_td_scan.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test /home/murata/master/checkpoint/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5/ckpt_49_1.pth.tar output/hyper_parameter/tbeta_td/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10/tbeta001td010 0 500000 False 7 5 False
  # ntau train jet test
# python save_root_tbeta_below01_td_scan.py /data/suehara/gravnet_ilc/data/uds91/test /home/murata/master/checkpoint/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5/ckpt_49_1.pth.tar output/hyper_parameter/tbeta_td/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_uds91/tbeta001td010 0 500000 False 7 5 False
  # jet train jet test
# python save_root_tbeta_below01_td_scan.py /data/suehara/gravnet_ilc/data/uds91/test /home/murata/master/checkpoint/output_dimensions/ckpts_gravnet_new02_2024_06_30_065937_outputD5/ckpt_49_1.pth.tar output/hyper_parameter/tbeta_td/tc_uds91_timingcut_forcealpha_thetaphi_5D_49_uds91/tbeta001td010 0 500000 False 7 5 False


# skimmed
# python save_root_tbeta_td_scan.py /data/suehara/mldata/pfa/murata/skimmed/ntau_10GeV_10/test /home/murata/master/checkpoint/output_dimensions/ckpts_gravnet_new02_2024_08_16_165355_outputD5_skimmed/ckpt_49_1.pth.tar output/new_clustering/hyper_parameter/tbeta_td/skimmed/ntau/tbeta010td010 0 500000 False 7 5 False
# python save_root_tbeta_td_scan.py /data/suehara/mldata/pfa/murata/skimmed/ntau_10GeV_10/test /home/murata/master/checkpoint/output_dimensions/ckpts_gravnet_new02_2024_08_16_165355_outputD5_skimmed/ckpt_49_1.pth.tar output/skimmed/new_clustering/hyper_parameter/tbeta_td/ntau/tbeta010td010 0 500000 False 7 5 False


## scan tbeta (0) and td (0.1-0.9, 0.1 increment)
# python save_root_tbeta_below01_td_scan.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test /home/murata/master/checkpoint/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5/ckpt_49_1.pth.tar output/hyper_parameter/tbeta_td/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10/tbeta001td010_ 0 500000 False 7 5 False

## scan tbeta (0.1-0.9, 0.1 increment) and td increases until energy equals momentum (if no track, use td=0.5)
# python save_root_tbeta_scan_td_momentum.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test /home/murata/master/checkpoint/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5/ckpt_49_1.pth.tar output/hyper_parameter/tbeta_td/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10/td_momentum 0 500000 False 7 5 False
