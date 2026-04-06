#!/bin/sh

cd ..


train_particle=ntau_10GeV_10    # ntau_10GeV_10      uds91    ntau_10to100GeV_10
test_particle=${train_particle}     # ntau_10GeV_10      uds91    ntau_10to100GeV_10
epoch=49
input_dim=7
output_dim=5
momentum=False
momentumAmp=False
MCTpe=False

test_path=/data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test
if [ ${test_particle} = "uds91" ]; then
  test_path=/data/suehara/mldata/pfa/uds91/test
elif [ ${test_particle} = "ntau_10to100GeV_10" ]; then
  test_path=/data/suehara/mldata/pfa/murata/ntau_10to100GeV_10_lessSample/test
fi
checkpoint_path=/home/murata/master/checkpoint
output_path=output/regression


epoch=59
outfile=tc_ntau_10GeV_10_5D/no_E_regression/2025_03_24_143709_outputD5_epoch${epoch}.root
checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2025_03_24_143709_outputD5

outfile=tc_ntau_10GeV_10_5D/no_E_regression/2025_03_25_141559_outputD5_epoch${epoch}_ReduceLROnPlateau.root
checkpoint=${checkpoint_path}/ckpts_gravnet_new02_2025_03_25_141559_outputD5

outfile=tc_ntau_10GeV_10_5D/no_E_regression/2025_03_25_141710_outputD9_epoch${epoch}_ReduceLROnPlateau.root
checkpoint=${checkpoint_path}/ckpts_gravnet_new02_2025_03_25_141710_outputD9
output_dim=9


python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outfile} 0 50000 False ${input_dim} ${output_dim}





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




## pandora energy prediction
  # train_particle=ntau_10GeV_10
  # test_particle=ntau_10GeV_10
  # test_path=/data/suehara/mldata/pfa/murata/skimmed/pandora/ntau_10GeV_10/test
  # output_path=output/energy_regression_1to1/pandora
  # outfile=tc_${train_particle}_${D}D_pandora_20250317_test.root
  # checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_08_22_123039_outputD5

  train_particle=uds91
  test_particle=uds91
  test_path=/data/suehara/mldata/pfa/murata/skimmed/uds91/test
  output_path=output/energy_regression_1to1/pandora
  outfile=tc_${train_particle}_${D}D_pandora_20250317_test.root
  checkpoint=${checkpoint_path}/energy_regression/ckpts_gravnet_new02_2024_08_22_123039_outputD5

  epoch=49
  # mkdir -p ${output_path}
  # python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outfile} 0 50000 False ${input_dim} ${outD} --pandora --device cuda:0
##



## w/o energy prediction
  train_particle=ntau_10GeV_10
  test_particle=ntau_10GeV_10
  # test_path=/data/suehara/mldata/pfa/murata/skimmed/pandora/ntau_10GeV_10/test
  test_path=/data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test
  # outfile=tc_${train_particle}_${D}D_pandora_20250317.root
  checkpoint=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_Jun22_0937_D5
  outdir=tc_${train_particle}_${D}D/no_E_regression/tbeta_td_scan

  # train_particle=uds91
  # test_particle=uds91
  # momentum=False
  # momentumAmp=False
  # # test_path=/data/suehara/mldata/pfa/murata/skimmed/uds91/test
  # test_path=/data/suehara/mldata/pfa/uds91/test
  # checkpoint=${checkpoint_path}/output_dimensions/ckpts_gravnet_new02_2024_06_30_065937_outputD5 
  # # outfile=tc_${train_particle}_${D}D/fine_tuning_2025_03_04_162442/fine_tuning_2025_03_04_162442_epoch${iepoch}.root
  # outdir=tc_${train_particle}_${D}D/no_E_regression/tbeta_td_scan

  epoch=49
  # python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar ${output_path}/${outdir} 0 50000 False ${input_dim} 5 --device cuda:0 --beta-d-scan
##



outfile=test.root
# python save_root_energyRegression_1to1.py /data/suehara/mldata/pfa/murata/code_test_ntau/test ${checkpoint}/ckpt_49_1.pth.tar test/${outfile} 0 1 False 7 ${outD} False True ${momentum} ${momentumAmp}
# python save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_49_1.pth.tar test/${outfile} 0 1 False 7 ${outD} False True ${momentum} ${momentumAmp}
# python save_root_energyRegression_1to1.py /data/suehara/mldata/pfa/murata/code_test_ntau/test ${checkpoint}/ckpt_${epoch}_1.pth.tar test/test.root 0 100 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:1
# python -m cProfile -o shell/making_root_gpu_batch20.prof save_root_energyRegression_1to1.py ${test_path} ${checkpoint}/ckpt_${epoch}_1.pth.tar test/test.root 0 50000 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp --device cuda:1




