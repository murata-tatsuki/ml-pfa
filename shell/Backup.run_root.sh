#!/bin/sh

cd ..


train_particle=ntau_10GeV_10
test_particle=uds91         # ntau_10GeV_10      uds91



test_path=/data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test
if [ ${test_particle} = "uds91" ]; then
  test_path=/data/suehara/mldata/pfa/uds91/test
fi
checkpoint_path=/home/murata/master/checkpoint


# python save_root.py /data/suehara/ntau/test_tc_ntau_10GeV_10 /data/suehara/gravnet_ilc/checkpoint/ntau_new_tc2_forcealpha_nolbetanoise_230901_1/ckpt_19_1.pth.tar output.root 0 100 False 5 3

# small dataset
#python save_root.py mydata/ntau_one_validate /data/suehara/gravnet_ilc/checkpoint/ntau_new_tc2_forcealpha_nolbetanoise_230901_1/ckpt_19_1.pth.tar output2.root 0 1 False 5 3

#for mc=-1 debug
#python save_root.py /data/suehara/ntau/test_tc_ntau_10GeV_10 /data/suehara/gravnet_ilc/checkpoint/ntau_new_tc2_forcealpha_nolbetanoise_230901_1/ckpt_19_1.pth.tar output_debug.root 96 96 False 5 3
#python save_root.py /data/suehara/ntau/test_tc_ntau_10GeV_10 /data/suehara/gravnet_ilc/checkpoint/ntau_new_tc2_forcealpha_nolbetanoise_230901_1/ckpt_19_1.pth.tar output_debug.root 96 97 False 5 3

#python save_root.py /data/suehara/ntau/test_tc_ntau_10GeV_10 /data/suehara/gravnet_ilc/checkpoint/ntau_new_tc2_forcealpha_nolbetanoise_230901_1/ckpt_19_1.pth.tar output.root 0 100 False 5 3
#python save_root.py mydata/ntau_one_validate /data/suehara/gravnet_ilc/checkpoint/ntau_new_tc2_forcealpha_nolbetanoise_230901_1/ckpt_19_1.pth.tar output2.root 0 1 False 5 3

#python save_root.py ${test_path} ${checkpoint_path}/ckpts_gravnet_new02_Apr22_1626_tochu/ckpt_16_1.pth.tar output.root 0 100 False 7 3


### これが本来のもの
# python save_root.py ${test_path} ${checkpoint_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49.root 0 5000 False 7 3 False

#python save_root.py h5/ntau_10GeV_10_pandora/ntau_10GeV_10 ${checkpoint_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/ntau_to_ntau/pandora/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49.root 0 10000 False 7 3 True
# python save_root.py h5/ntau_10GeV_10_pandora/ntau_10GeV_10 ${checkpoint_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/ntau_to_ntau/pandora/test.root 0 1 False 7 3 True
#python save_root.py h5/ntau_10GeV_10_pandora/ntau_10GeV_10 ${checkpoint_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/gravnet/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49.root 0 10000 False 7 3 False
# python save_root.py ${test_path}/ ${checkpoint_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49_usd.root 0 50000 False 7 3 False

# python save_root.py /data/suehara/gravnet_ilc/data/uds91/test ${checkpoint_path}/ckpts_gravnet_new02_May13_1920/ckpt_19_1.pth.tar output/uds_to_uds/gravnet/tc_usd91_timingcut_forcealpha_thetaphi_19_usd.root 0 10000 False 7 3 False
# python save_root.py ../data_murata/uds_91_pandora/test/ ${checkpoint_path}/ckpts_gravnet_new02_May13_1920/ckpt_19_1.pth.tar output/uds_to_uds/pandora/tc_usd91_timingcut_forcealpha_thetaphi_19_usd.root 0 100000 False 7 3 True


##### hyperparameter tuning 
### output dimension
D=9

cp_path=${checkpoint_path}/
if [ ${D} -eq 3 ]; then
  cp_path=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_Apr23_1328
if [ ${D} -eq 4 ]; then
  cp_path=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_Jun19_1825_D4
elif [ ${D} -eq 5 ]; then
  cp_path=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5
elif [ ${D} -eq 9 ]; then
  cp_path=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_Jun27_1419_D9
elif [ ${D} -eq 17 ]; then
  cp_path=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_Jun27_1426_D17
elif [ ${D} -eq 5 ]; then
  cp_path=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5
fi
output_path=output/hyper_parameter/dimensions
## D=3
python save_root.py ${test_path} ${cp_path}/ckpt_49_1.pth.tar ${output_path}/tc_${train_particle}_timingcut_forcealpha_thetaphi_3D_49_${test_particle}.root 0 500000 False 7 ${D} False
# python save_root.py ${test_path} ${cp_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar ${output_path}/tc_${train_particle}_timingcut_forcealpha_thetaphi_3D_49_uds.root 0 500000 False 7 3 False
## D=4
# python save_root.py ${test_path} ${cp_path}/output_dimensions/ckpts_gravnet_new02_Jun19_1825/ckpt_49_1.pth.tar ${output_path}/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_4D_49_ntau_10GeV_10.root 0 500000 False 7 4 False
# python save_root.py ${test_path} ${cp_path}/output_dimensions/ckpts_gravnet_new02_Jun19_1825/ckpt_49_1.pth.tar ${output_path}/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_4D_49_uds.root 0 500000 False 7 4 False
## D=5
# python save_root.py ${test_path} ${cp_path}/output_dimensions/ckpts_gravnet_new02_Jun22_0937/ckpt_49_1.pth.tar ${output_path}/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10.root 0 500000 False 7 5 False
# python save_root.py ${test_path} ${cp_path}/output_dimensions/ckpts_gravnet_new02_Jun22_0937/ckpt_49_1.pth.tar ${output_path}/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_uds.root 0 500000 False 7 5 False
## D=9
# python save_root.py ${test_path} ${cp_path}/ckpt_49_1.pth.tar ${output_path}/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_9D_49_ntau_10GeV_10.root 0 500000 False 7 9 False
# python save_root.py ${test_path} ${cp_path}/output_dimensions/ckpts_gravnet_new02_Jun27_1419_D9/ckpt_49_1.pth.tar ${output_path}/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_9D_49_uds.root 0 500000 False 7 9 False
    ## second train (first one had very low eff. and pur. might be due to random parameter effect or low statistics)
python save_root.py ${test_path} ${cp_path}/output_dimensions/ckpts_gravnet_new02_2024_07_04_110317_outputD9/ckpt_49_1.pth.tar ${output_path}/tc_${train_particle}_timingcut_forcealpha_thetaphi_3D_49_${test_particle}.root 0 500000 False 7 ${D} False
## D=17
# python save_root.py ${test_path} ${cp_path}/output_dimensions/ckpts_gravnet_new02_Jun27_1426_D17/ckpt_49_1.pth.tar ${output_path}/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_17D_49_ntau_10GeV_10.root 0 500000 False 7 17 False
# python save_root.py ${test_path} ${cp_path}/output_dimensions/ckpts_gravnet_new02_Jun27_1426_D17/ckpt_49_1.pth.tar ${output_path}/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_17D_49_uds.root 0 500000 False 7 17 False

## uds　train
## D=5
# python save_root.py ${test_path} ${cp_path}/output_dimensions/ckpts_gravnet_new02_2024_06_30_065937_outputD5/ckpt_49_1.pth.tar ${output_path}/uds_to_uds/tc_usd91_timingcut_forcealpha_thetaphi_5D_49_usd.root 0 500000 False 7 5 False


##### fine tuning 
## epoch=20
# python save_root.py ${test_path} ${checkpoint_path}/fune_tuning/ckpts_gravnet_new02_Jun19_1813/ckpt_39_1.pth.tar output/fine_tuning/tc_ntau_10GeV_10_uds_timingcut_forcealpha_thetaphi_39_ntau_10GeV_10.root 0 500000 False 7 3 False
# python save_root.py ${test_path} ${checkpoint_path}/fune_tuning/ckpts_gravnet_new02_Jun19_1813/ckpt_39_1.pth.tar output/fine_tuning/tc_ntau_10GeV_10_uds_timingcut_forcealpha_thetaphi_39_uds.root 0 500000 False 7 3 False
## epoch=25
# python save_root.py ${test_path} ${checkpoint_path}/fune_tuning/ckpts_gravnet_new02_Jun22_0946/ckpt_49_1.pth.tar output/fine_tuning/tc_ntau_10GeV_10_uds_timingcut_forcealpha_thetaphi_49_ntau_10GeV_10.root 0 500000 False 7 3 False
# python save_root.py ${test_path} ${checkpoint_path}/fune_tuning/ckpts_gravnet_new02_Jun22_0946/ckpt_49_1.pth.tar output/fine_tuning/tc_ntau_10GeV_10_uds_timingcut_forcealpha_thetaphi_49_uds.root 0 500000 False 7 3 False


## test
#python save_root.py test_ ${checkpoint_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/test_test.root 0 5000 False 7 3 True
#python save_root.py test_100 ${checkpoint_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/test/test_100_.root 0 5000 False 7 3 False
#python save_root.py test_ ${checkpoint_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/test/test_pandora.root 0 5000 False 7 3 True


#python save_root.py /data/suehara/mldata/pfa/qq91sub/tc/test ${checkpoint_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/ntau_to_qq/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49.root 0 100 False 7 3
