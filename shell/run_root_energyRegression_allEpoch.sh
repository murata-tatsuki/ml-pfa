#!/bin/sh

cd ..


train_particle=ntau_10GeV_10    # ntau_10GeV_10      uds91    ntau_10to100GeV_10
test_particle=${train_particle}     # ntau_10GeV_10      uds91    ntau_10to100GeV_10

energy_regression=true
energy_regression_betaMSE=false
reduced_samples=true 
testSuffix=""
testPrefix=""



test_path=/data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test
if [ ${test_particle} = "uds91" ]; then
  test_path=/data/suehara/mldata/pfa/uds91/test
elif [ ${test_particle} = "ntau_10to100GeV_10" ]; then
  test_path=/data/suehara/mldata/pfa/murata/ntau_10to100GeV_10_lessSample/test
elif "${reduced_samples}"; then
  test_path=/data/suehara/mldata/pfa/murata/ntau_lessSample/test
fi
checkpoint_path=/home/murata/master/checkpoint


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

output_path=output/new_clustering/hyper_parameter/dimensions


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

epoch=49
input_dim=7
momentum=False
momentumAmp=False
MCTpe=False


## momentum
momentum=True
momentumAmp=True
epoch=59

MCTpe=False

# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_05_105638_outputD5
# outDir=${output_path}/energyTree/restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum_virtualhitTrueMomentum/epochs/alphaTrackModifyingCharge0

# checkpoint=${checkpoint_path}/energy_regression/reduced_samples/ckpts_gravnet_new02_2025_01_08_192429_outputD5
# outDir=${output_path}/energyTree/restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum_virtualhitTrueMomentum/epochs/reduced/alphaTrackModifying

# checkpoint=${checkpoint_path}/energy_regression/reduced_samples/ckpts_gravnet_new02_2025_01_09_173804_outputD5
# outDir=${output_path}/energyTree/restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum_virtualhitTrueMomentum/epochs/reduced/alphaTrackModifying_coef005

# checkpoint=${checkpoint_path}/energy_regression/reduced_samples/ckpts_gravnet_new02_2025_01_10_170318_outputD5
# outDir=${output_path}/energyTree/restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum_virtualhitTrueMomentum/epochs/reduced/alphaTrackModifying_coef005_LE16

## なんかこれ間違えてそう
# # # # # checkpoint=${checkpoint_path}/energy_regression/reduced_samples/ckpts_gravnet_new02_2025_01_08_185604_outputD5
# # # # # outDir=${output_path}/energyTree/restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/epochs/reduced/alphaTrackModifying_coef005_LE16

# checkpoint=${checkpoint_path}/energy_regression/reduced_samples/ckpts_gravnet_new02_2025_01_13_123114_outputD5
# outDir=${output_path}/energyTree/restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/epochs/reduced/alphaTrackModifying_LE16

# checkpoint=${checkpoint_path}/energy_regression/reduced_samples/ckpts_gravnet_new02_2025_01_13_123247_outputD5
# outDir=${output_path}/energyTree/restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/epochs/reduced/alphaTrackModifying

checkpoint=${checkpoint_path}/energy_regression/reduced_samples/ckpts_gravnet_new02_2025_01_14_212704_outputD5
outDir=${output_path}/energyTree/restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/epochs/reduced/alphaModifying


## full sample
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_01_08_185604_outputD5
# outDir=${output_path}/energyTree/restartPeriod30/tc_${train_particle}_${D}D_${epoch}_${test_particle}_momentum/epochs/alphaTrackModifying



mkdir -p ${outDir}
if [ ${MCTpe} = "True" ]; then
  testPrefix=testMCTruth
  testSuffix=_testMCTruth
elif [ ${MCTpe} = "False" ]; then
  testPrefix=testDetected
  testSuffix=_testDetected
fi

epoch=0
# while [ ${epoch} -lt 60 ]
for i in `seq 1 6`;
do
  epoch=$((i*10-1))

  outfile=${testPrefix}_epoch${epoch}.root
  python save_root_energyRegression.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${outDir}/${outfile} 0 50000 False ${input_dim} ${outD} False True ${momentum} ${momentumAmp} ${MCTpe}

done

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
# epoch=423
# momentumAmp=False
# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_alphaTrack_momentum_restartPeriod30.root
# checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_11_22_144511_outputD5



# outfile=test.root
# python save_root_energyRegression.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/energyTree/${outfile} 0 50000 False ${input_dim} ${outD} False True ${momentum} ${momentumAmp} ${MCTpe}




## pandora energy prediction
# test_path=/data/suehara/mldata/pfa/murata/skimmed/pandora/ntau_10GeV_10/test
# output_path=output/energy_regression/new_clustering/energyTree/pandora
# outfile=tc_${train_particle}_${D}D_${epoch}_${test_particle}_pandora.root

# python save_root_energyRegression.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outfile} 0 50000 False ${input_dim} ${outD} True True ${momentum} ${momentumAmp}






outfile=test.root
# python save_root_energyRegression.py /data/suehara/mldata/pfa/murata/code_test_ntau/test ${checkpoint}/ckpt_49_1.pth.tar test/${outfile} 0 1 False 7 ${outD} False True ${momentum} ${momentumAmp}
# python save_root_energyRegression.py ${test_path} ${checkpoint}/ckpt_49_1.pth.tar test/${outfile} 0 1 False 7 ${outD} False True ${momentum} ${momentumAmp}




