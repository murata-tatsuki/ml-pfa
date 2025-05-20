#!/bin/sh

### usage
# python save_pred.py ${test sample path} ${checkpoint} ${path of h5 file} ${first event number} ${last event number} ${timingCut} ${input_dim} ${outD} 


cd ..
# python save_pred.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test checkpoint/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar h5/outfile_display_edit.h5 0 10 0 7 3 0 False
# python save_pred_pandora.py h5/ntau_10GeV_10_pandora/test checkpoint/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar h5/outfile_display_pandora_ntau.h5 0 1 0 7 3 0 True > log/h5_test.log
# python save_pred_pandora.py ../data_murata/uds_91_pandora/test checkpoint/ckpts_gravnet_new02_May13_1920/ckpt_19_1.pth.tar h5/outfile_display_pandora_uds__.h5 0 1 0 7 3 0 True
# python save_pred_pandora.py /data/suehara/mldata/pfa/murata/skimmed/uds91/test checkpoint/ckpts_gravnet_new02_May13_1920/ckpt_19_1.pth.tar h5/outfile_display_pandora_uds__.h5 0 1 0 7 3 0 True



# python save_pred.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test checkpoint/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar h5/outfile_display_pandora_ntau.h5 0 1 0 7 3 0 False > log/h5/h5_test.log
# python save_pred.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test checkpoint/energy_regression/ckpts_gravnet_new02_2025_01_30_161918_outputD5/ckpt_499_1.pth.tar h5/test.h5 0 1 0 11 6 0 False #> log/h5/h5_test.log


D=5
outD=$((D+1))
input_dim=7
momentum=True
momentumAmp=True
# python save_pred.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test checkpoint/energy_regression/ckpts_gravnet_new02_2025_01_30_161918_outputD5/ckpt_499_1.pth.tar h5/test_2025_01_30_161918.h5 0 1 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp #> log/h5/test_2025_01_30_161918.log
# python save_pred.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test checkpoint/energy_regression/ckpts_gravnet_new02_2025_01_30_161918_outputD5/ckpt_499_1.pth.tar h5/test.h5 0 1 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp #> log/h5/test_2025_01_30_161918.log
# python save_pred.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test checkpoint/energy_regression/ckpts_gravnet_new02_2025_02_07_150149_outputD5/ckpt_499_1.pth.tar h5/test.h5 0 1 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp #> log/h5/test_2025_01_30_161918.log



# python save_pred.py /data/suehara/mldata/pfa/murata/tc_uds_91/test checkpoint/energy_regression/ckpts_gravnet_new02_2025_03_04_162442_outputD5/ckpt_470_1.pth.tar h5/test.h5 0 1 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp #> log/h5/test_2025_01_30_161918.log
# python save_pred.py /data/suehara/gravnet_ilc/data/uds91/test checkpoint/energy_regression/ckpts_gravnet_new02_2025_03_12_141129_outputD5/ckpt_180_1.pth.tar h5/test.h5 0 1 False ${input_dim} ${outD} --energy-regression --energy-regression-cluster --momentum --momentum-amp #> log/h5/test_2025_01_30_161918.log


# python save_pred.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test checkpoint/energy_regression/ckpts_gravnet_new02_2025_02_10_170311_outputD5/ckpt_499_1.pth.tar h5/test.h5 0 1 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp #> log/h5/test_2025_01_30_161918.log


outD=$((D))
input_dim=7
# python save_pred.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test checkpoint_old/ckpts_gravnet_new02_2025_04_16_115502_outputD5/ckpt_499_1.pth.tar h5/default.h5 0 1 False ${input_dim} ${outD} #> log/h5/test_2025_01_30_161918.log
# python save_pred.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test checkpoint/no_energy_regression/ckpts_gravnet_new02_2025_05_05_122251_outputD5/ckpt_499_1.pth.tar h5/lr3e-5_2.5e-6.h5 0 1 False ${input_dim} ${outD} #> log/h5/test_2025_01_30_161918.log


# python save_pred.py /data/suehara/mldata/pfa/murata/data/tc_ntau_10GeV_10/test checkpoint/no_energy_regression/ckpts_gravnet_new02_2025_05_11_112102_outputD5/ckpt_109_1.pth.tar h5/tc_skimmed_lr3e-5_2.5e-6.h5 0 10 False ${input_dim} ${outD} #> log/h5/test_2025_01_30_161918.log
# python save_pred.py /data/suehara/mldata/pfa/murata/data/tc_ntau_10GeV_10/test checkpoint/no_energy_regression/ckpts_gravnet_new02_2025_05_11_112102_outputD5/ckpt_109_1.pth.tar h5/test.h5 0 1 False ${input_dim} ${outD} #> log/h5/test_2025_01_30_161918.log
python save_pred.py /data/suehara/mldata/pfa/murata/code_test_ntau/test/tc checkpoint/no_energy_regression/ckpts_gravnet_new02_2025_05_11_112102_outputD5/ckpt_109_1.pth.tar h5/test.h5 0 1 False ${input_dim} ${outD} #> log/h5/test_2025_01_30_161918.log