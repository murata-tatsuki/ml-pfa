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
checkpoint_path_old=/home/murata/master/checkpoint_old
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
  # output_path=output/energy_regression/new_clustering
  output_path=output/energy_regression_1to1
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
        # python save_root_energyRegression_1to1.py ${test_path} ${cp_path[${D}]}/ckpt_49_1.pth.tar test/test.root 0 100 False 7 ${outD} False True
# python save_root_energyRegression_1to1.py ${test_path} ${cp_path[${D}]}/ckpt_49_1.pth.tar ${output_path}/${outfile} 0 500000 False 7 ${outD} False True
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

# mkdir -p ${output_path}/energyTree/restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum_virtualhitTrueMomentum
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


outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/energy_branch/alphaModifying_EBranch_LE16_testDetected.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_20_151235_outputD5

# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/energy_branch/alphaModifying_EBranch_LE16_005_testDetected.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_22_124616_outputD5

# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/energy_branch/alphaModifying_EBranch_LE16_010_testDetected.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_22_123626_outputD5

outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alphaModifying_LE16_010_testDetected.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_28_183009_outputD5

outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alphaModifying_LE16_010_testDetected_notjit_fortest.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_29_093011_outputD5

outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alphaModifying_EBranch_LE16_005_testDetected_jit.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_29_175616_outputD5

outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alpha_LE16_010_gradually.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_04_173905_outputD5


# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:0



##### cluster energy
# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/cluster_energy_regression/alpha_LE16_010_testDetected_ERcluster.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_30_210728_outputD5

# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/cluster_energy_regression/alpha_LE16_010_testDetected_ERcluster_2025_01_31_155919.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_31_155919_outputD5

# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/cluster_energy_regression/alpha_LE16_010_testDetected_ERcluster_2025_02_02_015711.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_02_015711_outputD5

# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/cluster_energy_regression/alpha_LE16_010_testDetected_ERcluster_2025_02_02_162020.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_02_162020_outputD5

# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/cluster_energy_regression/alphaSqrtdiv_LE16_010_ERcluster_2025_02_04_174502.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_04_174502_outputD5

# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/cluster_energy_regression/alpha_LE16_010_ERcluster_2025_02_05_143651.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_05_143651_outputD5

iepoch=499
outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/cluster_energy_regression/alphaTracker_LE16_010_ERcluster_2025_02_05_143327/alphaTracker_LE16_010_ERcluster_2025_02_05_143327_${iepoch}.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_05_143327_outputD5
outdir=tc_${train_particle}_${D}D/cluster_energy_regression/alphaTracker_LE16_010_22025_02_05_143327/tbeta_td_scan
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:1

outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/cluster_energy_regression/alpha_LE16_010_ERcluster_2025_02_07_150149/alpha_LE16_010_ERcluster_2025_02_07_150149_${iepoch}.root
outfile=tc_${train_particle}_${D}D/cluster_energy_regression/alpha_LE16_010_ERcluster_2025_02_07_150149/alpha_LE16_010_ERcluster_2025_02_07_150149_${iepoch}.root
outdir=tc_${train_particle}_${D}D/cluster_energy_regression/alpha_LE16_010_ERcluster_2025_02_07_150149/tbeta_td_scan
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_07_150149_outputD5
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:1
outfile=tc_${train_particle}_${D}D/cluster_energy_regression/alpha_LE16_010_ERcluster_2025_02_07_150149/test.root
# mkdir -p ${output_path}/${outdir}
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/${outfile} 0 10 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:1 --tbeta 0.9 --td 0.4
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:0 --beta-d-scan
# # # # # # outfile=tc_${train_particle}_${D}D/cluster_energy_regression/alpha_LE16_010_ERcluster_2025_02_07_150149/alpha_LE16_010_ERcluster_2025_02_07_150149_${iepoch}_perfectClustering.root
# # # # # # python save_root_energyRegression_perfectClustering.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:1 --tbeta 0.9 --td 0.4

outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/cluster_energy_regression/alphaTracker_LE16_010_ERcluster_sum_2025_02_10_170311/alphaTracker_LE16_010_ERcluster_sum_2025_02_10_170311_${iepoch}.root
outfile=tc_${train_particle}_${D}D/cluster_energy_regression/alphaTracker_LE16_010_ERcluster_sum_2025_02_10_170311/alphaTracker_LE16_010_ERcluster_sum_2025_02_10_170311_${iepoch}.root
outdir=tc_${train_particle}_${D}D/cluster_energy_regression/alphaTracker_LE16_010_ERcluster_sum_2025_02_10_170311/tbeta_td_scan
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_10_170311_outputD5
outfile=tc_${train_particle}_${D}D/cluster_energy_regression/alphaTracker_LE16_010_ERcluster_sum_2025_02_10_170311/test.root
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:1 --tbeta 0.9 --td 1
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:0 --beta-d-scan

# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/cluster_energy_regression/alpha_LE16_010_ERcluster_sum_2025_02_14_101644/alpha_LE16_010_ERcluster_sum_2025_02_14_101644_${iepoch}.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_14_101644_outputD5

# # outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/cluster_energy_regression/alpha_LE16_010_ERcluster_sum_2025_02_20_165510/alpha_LE16_010_ERcluster_sum_2025_02_20_165510_${iepoch}.root
# # checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_20_165510_outputD5
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:0

# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:1



##### cluster energy, but LE weight 0
outdir=tc_${train_particle}_${D}D/LEweight0/alpha_0_2025_02_19_1537484/tbeta_td_scan
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_19_153748_outputD5
# mkdir -p ${output_path}/${outdir}
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:0 --beta-d-scan



## 500 epochs
# iepoch=499
# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alpha_EBranch_LE16_010_testDetected_${iepoch}.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_30_160628_outputD5
# outdir=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alpha_EBranch_LE16_010_testDetected_${iepoch}_tbeta_td_scan
# mkdir -p ${output_path}/energyTree/${outdir}
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --energy-branch --device cuda:1
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --energy-branch --device cuda:1 --beta-d-scan

# outfile=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alpha_LE16_010_testDetected_${iepoch}.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_30_161918_outputD5
# outdir=restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/alpha_LE16_010_testDetected_${iepoch}_tbeta_td_scan
# mkdir -p ${output_path}/energyTree/${outdir}
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:1
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/energyTree/${outdir} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:1 --beta-d-scan



train_particle=uds91
test_particle=uds91
# outfile=tc_${train_particle}_${D}D_49_${test_particle}_alphaMSE_momentum.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_11_02_081028_outputD5

# iepoch=119
# outfile=restartPeriod30/tc_${train_particle}_${D}D_momentum/alpha_LE16_010_${iepoch}.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_30_164401_outputD5

outfile=restartPeriod30/tc_${train_particle}_${D}D_momentum/alpha_LE16_010_retrain_59.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_04_174146_outputD5

iepoch=470
outfile=tc_${train_particle}_${D}D/fine_tuning_2025_03_04_162442/fine_tuning_2025_03_04_162442_epoch${iepoch}.root
outdir=tc_${train_particle}_${D}D/fine_tuning_2025_03_04_162442/tbeta_td_scan
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_03_04_162442_outputD5
# python save_root_energyRegression_1to1.py /data/suehara/mldata/pfa/murata/tc_uds_91/test ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:1 --beta-d-scan


## cluster energy
iepoch=180
outfile=tc_${train_particle}_${D}D/fine_tuning_2025_03_01_085652/fine_tuning_2025_03_12_141129_epoch${iepoch}.root
outdir=tc_${train_particle}_${D}D/cluster_energy_regression/fine_tuning_2025_03_12_141129/tbeta_td_scan
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_03_12_141129_outputD5

# mkdir -p ${output_path}/${outdir}
# python save_root_energyRegression_1to1.py /data/suehara/mldata/pfa/murata/tc_uds_91/test ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:1 --beta-d-scan

# python save_root_energyRegression_1to1.py /data/suehara/mldata/pfa/murata/tc_uds_91/test ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:1
# outfile=tc_${train_particle}_${D}D/cluster_energy_regression/fine_tuning_2025_03_12_141129/fine_tuning_2025_03_12_141129_epoch${iepoch}_perfectClsutering.root
# python save_root_energyRegression_perfectClustering.py /data/suehara/mldata/pfa/murata/tc_uds_91/test ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:1 --tbeta 0.9 --td 0.4
# outfile=tc_${train_particle}_${D}D/cluster_energy_regression/fine_tuning_2025_03_12_141129/fine_tuning_2025_03_12_141129_epoch${iepoch}_1toMany.root
# python save_root_energyRegression.py /data/suehara/mldata/pfa/murata/tc_uds_91/test ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp --device cuda:1 --tbeta 0.9 --td 0.4



## ntau_10to100GeV
train_particle=ntau_10to100GeV_10
test_particle=ntau_10to100GeV_10
# outfile=restartPeriod30/tc_${train_particle}_${D}D_momentum/alpha_LE16_010.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_31_185432_outputD5

outfile=restartPeriod30/tc_${train_particle}_${D}D_momentum/alpha_sqrtdiv_LE16_010_epoch30.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_04_184337_outputD5

outfile=restartPeriod30/tc_${train_particle}_${D}D_momentum/alpha_sqrtdiv_LE16_010_lr1e-4.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_02_07_182522_outputD5

# python save_root_energyRegression_1to1.py /data/suehara/mldata/pfa/murata/tc_ntau_10to100GeV_10/reduce/test ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:1



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
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} False True ${momentum} ${momentumAmp} ${MCTpe}

# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${iepoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --energy-branch




## energy regression
# # # # # train_particle=ntau_10GeV_10
# # # # # test_path=/data/suehara/mldata/pfa/murata/data/tc/tc_ntau_10GeV_10/test
# # # # # outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster
# # # # # outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_tbeta09td05_truthClustering.root
# # # # # checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_06_23_172406_outputD5
  train_particle=nnqq
  test_path=/data/suehara/mldata/pfa/murata/data/tc/tc_nnqq/test
  # checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_09_10_154302_outputD5
  checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_09_30_105247_outputD5
  outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_Ecoef1_betaSuppress.root
  outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_Ecoef1_100betaSuppress_moreStats.root
  # # # # # # test_path=/data/suehara/mldata/pfa/murata/data/raw/nnqq/test
  # # # # # # # outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_Ecoef1_tbeta09td05_pandora.root
  # # # # # # outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_Ecoef1_tbeta09td05_eventTotalEnergy__.root
  test_path=/data/suehara/mldata/pfa/murata/data/raw/nnqq/new/concat/test
  # # # # # # outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_Ecoef1_tbeta09td05_eventTotalEnergy_moreStats_truthClustering.root
  epoch=485
  output_path=output/energy_regression_1to1
  # outdir=tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/default
  echo ${output_path}/${outdir}
  outD=5
  # outdir=tc_${train_particle}/${D}D/E_regression/test.root
  # python save_root.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster  --momentum --momentum-amp --device cuda:0 --beta-d-scan
  #python save_root_____.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster  --momentum --momentum-amp --device cuda:0 --tbeta 0.9 --td 0.5 --event-total-energy --e-weight
  # python save_root_____.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster  --momentum --momentum-amp --event-total-energy --device cuda:0 --tbeta 0.9 --td 0.5 --truth-clustering
  # python save_root.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --device cuda:0 --beta-d-scan
##

## energy regression weight energy loss
  # train_particle=nnqq
  # test_path=/data/suehara/mldata/pfa/murata/data/tc/tc_nnqq/test
  # test_path=/data/suehara/mldata/pfa/murata/data/raw/nnqq/new/concat/test
  train_particle=ntau_10GeV_10
  # test_path=/data/suehara/mldata/pfa/murata/data/raw/ntau_10GeV_10/test
  test_path=/data/suehara/mldata/pfa/murata/data/tc_ntau_10GeV_10/test
  checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_11_02_095108_outputD5
  outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/weight/test_cond_weight.root
  outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/weight/test_truthcl_cond_weight.root
  epoch=394
  output_path=output/energy_regression_1to1
  echo ${output_path}/${outdir}
  outD=5
  # python save_root_____edit.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:0 --tbeta 0.9 --td 0.5 --energy-regression-weight --truth-clustering
##

## energy regression of neutron events
  train_particle=neutron_1to100GeV
  test_path=/data/suehara/mldata/pfa/murata/neutron_1to100GeV/test
  # checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_10_08_144911_outputD5
  # outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/weight/test_truthcl_cond_weight.root
  checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_11_23_091328_outputD5
  # outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/test_truthcl_cond_cluster.root
  outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/test_cond_cluster.root
  echo ${output_path}/${outdir}
  epoch=124
  # python save_root_____edit.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:0 --tbeta 0.9 --td 0.5 --energy-regression-cluster #--truth-clustering
  # python save_root_____edit_no_n_had.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${outD} --momentum --momentum-amp --device cuda:0 --tbeta 0.9 --td 0.5 --energy-regression-weight --truth-clustering
##

## energy regression of mixed events  nnqq, neutron, kaon, gamma
  train_particle=nnqq
  outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/mixed_finetuning_rawTest.root
  # test_path=/data/suehara/mldata/pfa/murata/data/tc/tc_nnqq/test
  # test_path=/data/suehara/mldata/pfa/murata/data/tc/mixed/train/less_sample/eventCut/eventCut
  # test_path=/data/suehara/mldata/pfa/murata/data/tc/mixed/test
  test_path=/data/suehara/mldata/pfa/murata/data/raw/mix/test
  checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_12_01_081855_outputD5
  epoch=499
  # python save_root_____edit.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 5000000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:0 --tbeta 0.9 --td 0.5 --energy-regression-cluster
##

## energy regression of merged nnqq
  train_particle=nnqq_brems
  outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4
  # test_path=/data/suehara/mldata/pfa/murata/data/tc/eventCut/tc_nnqq_brems/test
  test_path=/data/suehara/mldata/pfa/murata/data/raw/nnqq_brems/test
  checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2026_01_23_164640_outputD5
  epoch=195
  # python save_root_____edit.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 5000000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:1 --energy-regression-cluster --beta-d-scan --event-total-energy --tbeta 0.1
##

## fixed uds
  train_particle=fixed_uds
  energy=500
  outdir=skimmed/tc_${train_particle}/${D}D/E_regression/tbeta_td_scan/qmin02_lr5e-4/${energy}GeV/tbeta090td050.root
  # test_path=/data/suehara/mldata/pfa/murata/data/raw/fixed_uds/uu/${energy}GeV
  test_path=/data/suehara/mldata/pfa/murata/data/tc/tc_fixed_uds/${energy}GeV/less_samples
  checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_06_30_151610_outputD5
  epoch=444
  python save_root_____edit.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 5000000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:0 --tbeta 0.9 --td 0.5 --energy-regression-cluster
##


## pandora energy prediction
  ###### train_particle=ntau_10GeV_10
  ###### test_particle=ntau_10GeV_10
  ###### test_path=/data/suehara/mldata/pfa/murata/skimmed/pandora/ntau_10GeV_10/test
  ###### output_path=output/energy_regression_1to1/pandora
  ###### outfile=tc_${train_particle}_${D}D_pandora_20250317_test.root
  ###### checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_08_22_123039_outputD5
  # train_particle=ntau_10GeV_10
  # test_particle=ntau_10GeV_10
  # test_path=/data/suehara/mldata/pfa/murata/data/ntau_10GeV_10/test
  # output_path=output/energy_regression_1to1/pandora
  # # outfile=tc_${train_particle}_pandora_20250422_1to1.root
  # outfile=tc_${train_particle}_pandora_20250422_1tomany.root
  # checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_08_22_123039_outputD5

  ###### train_particle=uds91
  ###### test_particle=uds91
  ###### test_path=/data/suehara/mldata/pfa/murata/skimmed/uds91/test
  ###### test_path=/data/suehara/mldata/pfa/murata/uds_91_pandora/test
  ###### output_path=output/energy_regression_1to1/pandora
  ###### outfile=tc_${train_particle}_${D}D_pandora_20250317_1tomany.root
  ###### checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_08_22_123039_outputD5
  train_particle=uds91
  test_particle=uds91
  test_path=/data/suehara/mldata/pfa/murata/data/uds91/test
  output_path=output/energy_regression_1to1/pandora
  outfile=tc_${train_particle}_pandora_20250422_1to1.root
  # outfile=tc_${train_particle}_pandora_20250422_1tomany.root
  checkpoint=${checkpoint_path}_old/energy_regression/ckpts_gravnet_new02_2024_08_22_123039_outputD5

  test_path=/data/suehara/mldata/pfa/murata/uds_91_pandora/test
  outfile=tc_${train_particle}_pandora_20250502_1to1_notSkimmed.root
  

  epoch=49
  # mkdir -p ${output_path}
  # python save_root.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outfile} 0 50000 False ${input_dim} ${outD} --pandora
  # python save_root_1tomany.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outfile} 0 50000 False ${input_dim} ${outD} --pandora --device cuda:0
##


## w/o energy prediction
  train_particle=ntau_10GeV_10
  test_particle=ntau_10GeV_10
  # test_path=/data/suehara/mldata/pfa/murata/skimmed/pandora/ntau_10GeV_10/test
  test_path=/data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test
  # outfile=tc_${train_particle}_${D}D_pandora_20250317.root
  # checkpoint=${checkpoint_path_old}/ckpts_gravnet_new02_2025_04_16_115502_outputD5
  checkpoint=${checkpoint_path}/no_energy_regression/ckpts_gravnet_new02_2025_05_21_134024_outputD5
  D=5
  output_path=output/energy_regression_1to1
  outdir=tc_${train_particle}/${D}D/no_E_regression/tbeta_td_scan/qmin01
  # test_path=/data/suehara/mldata/pfa/murata/data/ntau_10GeV_10/test

  # train_particle=uds91
  # test_particle=uds91
  # momentum=False
  # momentumAmp=False
  # # test_path=/data/suehara/mldata/pfa/murata/skimmed/uds91/test
  # test_path=/data/suehara/mldata/pfa/uds91/test
  # checkpoint=${checkpoint_path_old}/output_dimensions/ckpts_gravnet_new02_2024_06_30_065937_outputD5 
  # # outfile=tc_${train_particle}_${D}D/fine_tuning_2025_03_04_162442/fine_tuning_2025_03_04_162442_epoch${iepoch}.root
  # outdir=tc_${train_particle}_${D}D/no_E_regression/tbeta_td_scan

  epoch=499
  # python save_root.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${D} --device cuda:1 --beta-d-scan
  # python save_root.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} ${D} --device cuda:1 --td 0.5 --tbeta 0.9
##



outfile=test.root
# python save_root.py /data/suehara/mldata/pfa/murata/code_test_ntau/test ${checkpoint}/ckpt_49_1.pth.tar test/${outfile} 0 1 False 7 5 --device cuda:1
# python -m cProfile -o shell/prof/batchsize1.prof save_root.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test ${checkpoint}/ckpt_49_1.pth.tar test/${outfile} 0 50000 False 7 9 --device cuda:1
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_49_1.pth.tar test/${outfile} 0 1 False 7 ${outD} False True ${momentum} ${momentumAmp}
# python save_root_energyRegression_1to1.py /data/suehara/mldata/pfa/murata/code_test_ntau/test ${checkpoint}/ckpt_${epoch}_1.pth.tar test/test.root 0 100 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:1
# python -m cProfile -o shell/making_root_gpu_batch20.prof save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar test/test.root 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:1




