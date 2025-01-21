#!/bin/sh

cd ..


train_particle=ntau_10GeV_10    # ntau_10GeV_10      uds91    ntau_10to100GeV_10
test_particle=${train_particle}     # ntau_10GeV_10      uds91    ntau_10to100GeV_10

energy_regression=true
energy_regression_betaMSE=false
testSuffix=""



test_path=/data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test
if [ ${test_particle} = "uds91" ]; then
  test_path=/data/suehara/mldata/pfa/uds91/test
elif [ ${test_particle} = "ntau_10to100GeV_10" ]; then
  test_path=/data/suehara/mldata/pfa/murata/ntau_10to100GeV_10_lessSample/test
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
D=5
output_path=output/hyper_parameter/dimensions

cp_path[3]=${checkpoint_path}/ckpts_gravnet_new02_Apr23_1328
cp_path[4]=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_Jun19_1825_D4
cp_path[5]=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5
# cp_dir[9]=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_Jun27_1419_D9
cp_path[9]=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_2024_07_04_110317_outputD9   ## second train (first one had very low eff. and pur. might be due to random parameter effect or low statistics)
cp_path[17]=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_Jun27_1426_D17

if [ ${train_particle} = "uds91" ]; then
  cp_path[5]=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_2024_06_30_065937_outputD5 
  output_path=${output_path}/uds_to_uds
fi

# python save_root.py ${test_path} ${cp_path[${D}]}/ckpt_49_1.pth.tar ${output_path}/tc_${train_particle}_timingcut_forcealpha_thetaphi_${D}D_49_${test_particle}.root 0 500000 False 7 ${D} False

output_path=output/new_clustering/hyper_parameter/dimensions
# python save_root.py ${test_path} ${cp_path[${D}]}/ckpt_49_1.pth.tar ${output_path}/tc_${train_particle}_timingcut_forcealpha_thetaphi_${D}D_49_${test_particle}.root 0 500000 False 7 ${D} False


if "${energy_regression}"; then
  output_path=output/energy_regression/new_clustering
  cp_path[$D]=""
  edited="_edit"

  outfile=tc_${train_particle}_${D}D_49_${test_particle}${edited}_energyTree.root
  cp_path[5]=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_08_21_134147_outputD5
  if "${energy_regression_betaMSE}"; then
    cp_path[5]=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_08_22_123039_outputD5
    outfile=tc_${train_particle}_${D}D_49_${test_particle}_MSE${edited}_energyTree.root
  fi

  outD=$((D+1))
fi

# echo ${outfile}
        ### test
        # python save_root_energyRegression.py ${test_path} ${cp_path[${D}]}/ckpt_49_1.pth.tar test/test.root 0 100 False 7 ${outD} False True
# python save_root_energyRegression.py ${test_path} ${cp_path[${D}]}/ckpt_49_1.pth.tar ${output_path}/${outfile} 0 500000 False 7 ${outD} False True
# python save_root.py ${test_path} ${cp_path[${D}]}/ckpt_49_1.pth.tar ${output_path}/tc_${train_particle}_betaMSE_${D}D_49_${test_particle}.root 0 500000 False 8 ${D} False

# epoch=14
epoch=49
input_dim=7
momentum=False
momentumAmp=False
MCTpe=False
# outfile=tc_${train_particle}_${D}D_49_${test_particle}_betaMSE_coef1.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_08_22_123039_outputD5

# outfile=tc_${train_particle}_${D}D_49_${test_particle}_betaMSE_coef6.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_10_01_173144_outputD5

# outfile=tc_${train_particle}_${D}D_49_${test_particle}_betaE_positive.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_10_15_183822_outputD5

# outfile=tc_${train_particle}_${D}D_49_${test_particle}_betaE_fixloss.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_10_22_173630_outputD5

# outfile=tc_${train_particle}_${D}D_49_${test_particle}_alphaMSE.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_10_16_112733_outputD5

# train_particle=uds91
# test_particle=uds91
# outfile=tc_${train_particle}_${D}D_49_${test_particle}_alphaMSE.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_10_26_122011_outputD5

# train_particle=ntau_10to100GeV_10
# test_particle=ntau_10to100GeV_10
# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alphaMSE.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_10_23_164606_outputD5

## momentum
momentum=True
# outfile=tc_${train_particle}_${D}D_49_${test_particle}_alpha_momentum.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_10_23_120737_outputD5

# outfile=tc_${train_particle}_${D}D_49_${test_particle}_alphaMSE_momentum.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_11_02_080848_outputD5

# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alphaMSE_momentum_restartPeriod50_condbeta_tbeta060.root
outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alphaMSE_momentum_restartPeriod50_condbeta_tbeta030.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_11_12_163532_outputD5

# outfile=tc_${train_particle}_${D}D_49_${test_particle}_alphaMSE_momentum_restartPeriod30.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_11_13_165313_outputD5

# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alphaMSE_momentum_restartPeriod50_coef001.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_11_19_151106_outputD5

# epoch=59
# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alphaMSE_momentum_restartPeriod30_coef05.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_11_22_144756_outputD5

# epoch=59
# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alphaMSE_momentum_restartPeriod30_coef01.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_11_22_144847_outputD5

epoch=59
momentumAmp=True
# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alphaTrack_momentum_restartPeriod30_trueloss.root
# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alphaTrack_momentum_restartPeriod30_trueloss_truemomentum.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_11_26_174035_outputD5

# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alpha_momentum_restartPeriod30_trueloss_detectecprediction.root
# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alpha_momentum_restartPeriod30_trueloss.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_12_03_113656_outputD5

# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alphaTrack_momentum_restartPeriod30_detectedloss.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_12_04_131221_outputD5

# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alpha_momentum_restartPeriod30_detectedloss.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_12_03_115021_outputD5

mkdir -p ${output_path}/energyTree/restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum_virtualhitTrueMomentum
MCTpe=False
if [ ${MCTpe} = "True" ]; then
  testSuffix=_testMCTruth
elif [ ${MCTpe} = "False" ]; then
  testSuffix=_testDetected
fi
# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}/trash/alphaTrackModifying_momentum_virtualhitTrueMomentum.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_12_25_191117_outputD5

outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum_virtualhitTrueMomentum/alphaTrackModifyingCharge0${testSuffix}.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_05_105638_outputD5

# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum_virtualhitTrueMomentum/alphaTrackModifying${testSuffix}.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_05_110056_outputD5

# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum_virtualhitTrueMomentum/alphaTrackModifyingAll0${testSuffix}.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_05_004402_outputD5

outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum_virtualhitTrueMomentum/alphaTrackModifyingAll0${testSuffix}.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_13_122347_outputD5





# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alphaTrackModifying${testSuffix}.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_08_185604_outputD5

# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alphaTrackModifying_LE16_beta06d05${testSuffix}.root
# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alphaTrackModifying_LE16_beta06d05${testSuffix}_test.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_13_122347_outputD5

outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alphaModifying_LE16_beta03d05${testSuffix}.root
outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alphaModifying_LE16_beta06d05${testSuffix}.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_14_174251_outputD5


outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alphaModifying_EBranch_LE16_testDetected_17.root
epoch=17
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_20_151235_outputD5


# train_particle=uds91
# test_particle=uds91
# outfile=tc_${train_particle}_${D}D_49_${test_particle}_alphaMSE_momentum.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_11_02_081028_outputD5


## momentum amplitude 
# momentum=True
# momentumAmp=True
# outfile=tc_${train_particle}_${D}D_49_${test_particle}_alphaMSE_momentum_momentumAmp.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_11_06_184121_outputD5



##### 500 epoch tau task
# epoch=499
# momentumAmp=False
# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alphaTrack_momentum_restartPeriod30.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_11_22_144511_outputD5



# outfile=test.root
# python save_root_energyRegression.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} False True ${momentum} ${momentumAmp} ${MCTpe}

python save_root_energyRegression.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp




## pandora energy prediction
# test_path=/data/suehara/mldata/pfa/murata/skimmed/pandora/ntau_10GeV_10/test
# output_path=output/energy_regression/new_clustering/energyTree/pandora
# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_pandora.root

# python save_root_energyRegression.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outfile} 0 50000 False ${input_dim} ${outD} True True ${momentum} ${momentumAmp}






outfile=test.root
# python save_root_energyRegression.py /data/suehara/mldata/pfa/murata/code_test_ntau/test ${checkpoint}/ckpt_49_1.pth.tar test/${outfile} 0 1 False 7 ${outD} False True ${momentum} ${momentumAmp}
# python save_root_energyRegression.py ${test_path} ${checkpoint}/ckpt_49_1.pth.tar test/${outfile} 0 1 False 7 ${outD} False True ${momentum} ${momentumAmp}




