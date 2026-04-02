#!/bin/sh

cd ..

train_sample=ntau_10GeV_10    # ntau_10GeV_10   uds91   mix   ntau_10to100GeV_10
test_sample=ntau_10GeV_10     # ntau_10GeV_10   uds91   mix   ntau_10to100GeV_10
# sample=uds91      # ntau_10GeV_10   uds91   mix   ntau_10to100GeV_10
# ncuda=0     #1  # 0 or 1

outputD=5   # output coordinate dimension """""""+1"""""""
skimmed=false

energy=false
beta=false      ## loss term w/ beta * MSE
alpha=false     ## loss term w/ only alpha MSE
coef=10         ## coefficient of energy loss term


pandora="False"                       # True or False
cl_method="new_clustering/"            # "" or new_clustering/


# energy_regression=true
# energy_regression_betaMSE=false

learning_rate=false



test_path=/data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/test
if [ ${sample} = "uds91" ]; then
  test_path=/data/suehara/mldata/pfa/uds91/test
elif [ ${sample} = "mix" ]; then
  test_path=/data/suehara/mldata/pfa/murata/mix_samples/test
elif [ ${sample} = "ntau_10to100GeV_10" ]; then
  test_path=/data/suehara/gravnet_ilc/data/ntau_10to100GeV_10/test
fi
suf=""
if "${skimmed}"; then
  test_path=/data/suehara/mldata/pfa/murata/skimmed/${test_sample}/test
  if [ ${sample} = "mix" -o ${sample} = "ntau_10to100GeV_10" ]; then
    test_path=""
  fi
  suf=_skimmed
fi


ckp_path=/home/murata/master/checkpoint
if "${energy}"; then
  ckp_path=/home/murata/master/checkpoint/energy_regression
fi


output_path=output/${cl_method}hyper_parameter/dimensions


cp_path[3]=${ckp_path}/ckpts_gravnet_new02_Apr23_1328
cp_path[4]=${ckp_path}/output_dimensions/ckpts_gravnet_new02_Jun19_1825_D4
cp_path[5]=${ckp_path}/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5
# cp_dir[9]=${ckp_path}/output_dimensions/ckpts_gravnet_new02_Jun27_1419_D9
cp_path[9]=${ckp_path}/output_dimensions/ckpts_gravnet_new02_2024_07_04_110317_outputD9   ## second train (first one had very low eff. and pur. might be due to random parameter effect or low statistics)
cp_path[17]=${ckp_path}/output_dimensions/ckpts_gravnet_new02_Jun27_1426_D17

if [ ${train_sample} = "uds91" ]; then
  cp_path[5]=${ckp_path}/output_dimensions/ckpts_gravnet_new02_2024_06_30_065937_outputD5 
  output_path=${output_path}/uds_to_uds
fi

if "${skimmed}"; then
  # cp_path[5]=${ckp_path}/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5
fi


# if [ "${energy_regression}" ]; then
#   output_path=output/energy_regression/new_clustering
#   cp_path[$D]=""
#   cp_path[5]=${ckp_path}/checkpoint/energy_regression/ckpts_gravnet_new02_2024_08_21_134147_outputD5
#   if [ "${energy_regression_betaMSE}" ]; then
#     cp_path[5]=${ckp_path}/checkpoint/energy_regression/ckpts_gravnet_new02_2024_08_22_123039_outputD5
#   fi
# fi
# python save_root.py ${test_path} ${cp_path[${D}]}/ckpt_49_1.pth.tar ${output_path}/tc_${train_sample}_${D}D_49_${test_sample}.root 0 500000 False 8 ${D} False
# python save_root.py ${test_path} ${cp_path[${D}]}/ckpt_49_1.pth.tar ${output_path}/tc_${train_sample}_betaMSE_${D}D_49_${test_sample}.root 0 500000 False 8 ${D} False



if "${learning_rate}"; then
  lr=1e-4                           #    1e-4   2e-5    5e-5    9e-6 (default) 
  output_path=output/hyper_parameter/learning_rate

  cp_path_lr=${ckp_path}/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5
  if [ ${lr} = "2e-5" ]; then
    cp_path_lr=${ckp_path}/output_dimensions/ckpts_gravnet_new02_2024_07_07_100401_outputD5 
  elif [ ${lr} = "5e-5" ]; then
    cp_path_lr=${ckp_path}/output_dimensions/ckpts_gravnet_new02_2024_07_10_070602_outputD5 
  elif [ ${lr} = "1e-4" ]; then
    cp_path_lr=${ckp_path}/output_dimensions/ckpts_gravnet_new02_2024_07_13_052149_outputD5 
  fi
  python save_root.py ${test_path} ${cp_path_lr}/ckpt_49_1.pth.tar ${output_path}/tc_${train_sample}_timingcut_forcealpha_thetaphi_${D}D_49_${test_sample}_lr${lr}.root 0 500000 False 7 ${D} False
fi

# python save_root.py ${test_path} ${cp_path_lr}/ckpt_49_1.pth.tar ${output_path}/tc_${train_sample}_timingcut_forcealpha_thetaphi_${D}D_49_${test_sample}_lr${lr}.root 0 500000 False 7 ${D} False




python save_root.py ${test_path} ${cp_path[${outputD}]}/ckpt_49_1.pth.tar ${output_path}/tc_${train_sample}_timingcut_forcealpha_thetaphi_${outputD}D_49_${test_sample}.root 0 500000 False 7 ${D} ${pandora}
# python save_root.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test ${ckp_path}/output_dimensions/ckpts_gravnet_new02_2024_08_16_165355_outputD5_skimmed/ckpt_49_1.pth.tar output/test.root 0 500000 False 7 ${D} False
# python save_root.py ${test_path} ${cp_path_lr}/ckpt_49_1.pth.tar ${output_path}/tc_${train_sample}_timingcut_forcealpha_thetaphi_${D}D_49_${test_sample}_lr${lr}.root 0 500000 False 7 ${D} False

















# python save_root.py /data/suehara/ntau/test_tc_ntau_10GeV_10 /data/suehara/gravnet_ilc/checkpoint/ntau_new_tc2_forcealpha_nolbetanoise_230901_1/ckpt_19_1.pth.tar output.root 0 100 False 5 3

# small dataset
#python save_root.py mydata/ntau_one_validate /data/suehara/gravnet_ilc/checkpoint/ntau_new_tc2_forcealpha_nolbetanoise_230901_1/ckpt_19_1.pth.tar output2.root 0 1 False 5 3

#for mc=-1 debug
#python save_root.py /data/suehara/ntau/test_tc_ntau_10GeV_10 /data/suehara/gravnet_ilc/checkpoint/ntau_new_tc2_forcealpha_nolbetanoise_230901_1/ckpt_19_1.pth.tar output_debug.root 96 96 False 5 3
#python save_root.py /data/suehara/ntau/test_tc_ntau_10GeV_10 /data/suehara/gravnet_ilc/checkpoint/ntau_new_tc2_forcealpha_nolbetanoise_230901_1/ckpt_19_1.pth.tar output_debug.root 96 97 False 5 3

#python save_root.py /data/suehara/ntau/test_tc_ntau_10GeV_10 /data/suehara/gravnet_ilc/checkpoint/ntau_new_tc2_forcealpha_nolbetanoise_230901_1/ckpt_19_1.pth.tar output.root 0 100 False 5 3
#python save_root.py mydata/ntau_one_validate /data/suehara/gravnet_ilc/checkpoint/ntau_new_tc2_forcealpha_nolbetanoise_230901_1/ckpt_19_1.pth.tar output2.root 0 1 False 5 3

#python save_root.py ${test_path} ${ckp_path}/ckpts_gravnet_new02_Apr22_1626_tochu/ckpt_16_1.pth.tar output.root 0 100 False 7 3


### これが本来のもの
# python save_root.py ${test_path} ${ckp_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49.root 0 5000 False 7 3 False

#python save_root.py h5/ntau_10GeV_10_pandora/ntau_10GeV_10 ${ckp_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/ntau_to_ntau/pandora/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49.root 0 10000 False 7 3 True
# python save_root.py h5/ntau_10GeV_10_pandora/ntau_10GeV_10 ${ckp_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/ntau_to_ntau/pandora/test.root 0 1 False 7 3 True
#python save_root.py h5/ntau_10GeV_10_pandora/ntau_10GeV_10 ${ckp_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/gravnet/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49.root 0 10000 False 7 3 False
# python save_root.py ${test_path}/ ${ckp_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49_usd.root 0 50000 False 7 3 False

# python save_root.py /data/suehara/gravnet_ilc/data/uds91/test ${ckp_path}/ckpts_gravnet_new02_May13_1920/ckpt_19_1.pth.tar output/uds_to_uds/gravnet/tc_usd91_timingcut_forcealpha_thetaphi_19_usd.root 0 10000 False 7 3 False
# python save_root.py ../data_murata/uds_91_pandora/test/ ${ckp_path}/ckpts_gravnet_new02_May13_1920/ckpt_19_1.pth.tar output/uds_to_uds/pandora/tc_usd91_timingcut_forcealpha_thetaphi_19_usd.root 0 100000 False 7 3 True


output_path=output/new_clustering/hyper_parameter/dimensions
# python save_root.py ${test_path} ${cp_path[${D}]}/ckpt_49_1.pth.tar ${output_path}/tc_${train_sample}_timingcut_forcealpha_thetaphi_${D}D_49_${test_sample}.root 0 500000 False 7 ${D} False


# if [ "${energy_regression}" ]; then
#   output_path=output/energy_regression/new_clustering
#   cp_path[$D]=""
#   cp_path[5]=${ckp_path}/checkpoint/energy_regression/ckpts_gravnet_new02_2024_08_21_134147_outputD5
#   if [ "${energy_regression_betaMSE}" ]; then
#     cp_path[5]=${ckp_path}/checkpoint/energy_regression/ckpts_gravnet_new02_2024_08_22_123039_outputD5
#   fi
# fi
# python save_root.py ${test_path} ${cp_path[${D}]}/ckpt_49_1.pth.tar ${output_path}/tc_${train_sample}_${D}D_49_${test_sample}.root 0 500000 False 8 ${D} False
# python save_root.py ${test_path} ${cp_path[${D}]}/ckpt_49_1.pth.tar ${output_path}/tc_${train_sample}_betaMSE_${D}D_49_${test_sample}.root 0 500000 False 8 ${D} False


### learning rate
lr=1e-4      #    1e-4   2e-5    5e-5    9e-6 (default) 
D=5
output_path=output/hyper_parameter/learning_rate

cp_path_lr=${ckp_path}/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5
if [ ${lr} = "2e-5" ]; then
  cp_path_lr=${ckp_path}/output_dimensions/ckpts_gravnet_new02_2024_07_07_100401_outputD5 
elif [ ${lr} = "5e-5" ]; then
  cp_path_lr=${ckp_path}/output_dimensions/ckpts_gravnet_new02_2024_07_10_070602_outputD5 
elif [ ${lr} = "1e-4" ]; then
  cp_path_lr=${ckp_path}/output_dimensions/ckpts_gravnet_new02_2024_07_13_052149_outputD5 
fi

# python save_root.py ${test_path} ${cp_path_lr}/ckpt_49_1.pth.tar ${output_path}/tc_${train_sample}_timingcut_forcealpha_thetaphi_${D}D_49_${test_sample}_lr${lr}.root 0 500000 False 7 ${D} False



## skimmed
D=5
# python save_root.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test ${ckp_path}/output_dimensions/ckpts_gravnet_new02_2024_08_16_165355_outputD5_skimmed/ckpt_49_1.pth.tar output/test.root 0 500000 False 7 ${D} False
# python save_root.py /data/suehara/mldata/pfa/murata/skimmed/ntau_10GeV_10/test ${ckp_path}/output_dimensions/ckpts_gravnet_new02_2024_08_16_165355_outputD5_skimmed/ckpt_49_1.pth.tar output/skimmed/ntau_to_ntau/tc_${train_sample}_${D}D_49_${test_sample}.root 0 500000 False 7 ${D} False

##### fine tuning 
## epoch=20
# python save_root.py ${test_path} ${ckp_path}/fune_tuning/ckpts_gravnet_new02_Jun19_1813/ckpt_39_1.pth.tar output/fine_tuning/tc_ntau_10GeV_10_uds_timingcut_forcealpha_thetaphi_39_ntau_10GeV_10.root 0 500000 False 7 3 False
# python save_root.py ${test_path} ${ckp_path}/fune_tuning/ckpts_gravnet_new02_Jun19_1813/ckpt_39_1.pth.tar output/fine_tuning/tc_ntau_10GeV_10_uds_timingcut_forcealpha_thetaphi_39_uds.root 0 500000 False 7 3 False
## epoch=25
# python save_root.py ${test_path} ${ckp_path}/fune_tuning/ckpts_gravnet_new02_Jun22_0946/ckpt_49_1.pth.tar output/fine_tuning/tc_ntau_10GeV_10_uds_timingcut_forcealpha_thetaphi_49_ntau_10GeV_10.root 0 500000 False 7 3 False
# python save_root.py ${test_path} ${ckp_path}/fune_tuning/ckpts_gravnet_new02_Jun22_0946/ckpt_49_1.pth.tar output/fine_tuning/tc_ntau_10GeV_10_uds_timingcut_forcealpha_thetaphi_49_uds.root 0 500000 False 7 3 False


## test
#python save_root.py test_ ${ckp_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/test_test.root 0 5000 False 7 3 True
#python save_root.py test_100 ${ckp_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/test/test_100_.root 0 5000 False 7 3 False
#python save_root.py test_ ${ckp_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/test/test_pandora.root 0 5000 False 7 3 True


#python save_root.py /data/suehara/mldata/pfa/qq91sub/tc/test ${ckp_path}/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar output/ntau_to_qq/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49.root 0 100 False 7 3
