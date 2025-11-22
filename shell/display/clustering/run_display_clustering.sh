#!/bin/sh

cd ../../../display


### usage
# python display_h5.py ${path to h5 file} ${path to html} ${first event number} ${last event number} ${virtual space coordinate dim}  (options )


# python display_h5_240219_cluster.py ../h5/outfile_display_edit.h5 html/out_cluster.html 0 5
# python display_h5_240219_cluster_pandora.py ../h5/outfile_display_pandora_uds.h5 html/out_cluster_pandora.html 0 1
# python display_h5_240219_cluster_pandora.py ../h5/outfile_display_pandora_ntau.h5 html/out_cluster_ntau_pandora.html 0 1 > test.log

#/data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test  /home/murata/master/checkpoint/ckpts_gravnet_new02_Apr23_1328/ckpt_49_1.pth.tar


D=5
outD=$((D+1))
input_dim=7
virtual_space_coordinate_dim=4
momentum=True
momentumAmp=True
# python save_pred.py /data/suehara/mldata/pfa/ntau/tc_ntau_10GeV_10/test checkpoint/energy_regression/ckpts_gravnet_new02_2025_01_30_161918_outputD5/ckpt_499_1.pth.tar h5/test.h5 0 1 False ${input_dim} ${outD} --energy-regression --momentum --momentum-amp
# python display_h5.py ../h5/test_2025_01_30_161918.h5 html/test_2025_01_30_161918.html 0 1 ${virtual_space_coordinate_dim} --energy-regression --momentum --momentum-amp # > test.log
# python display_h5.py ../h5/test.h5 html/test_2025_01_30_161918.html 0 1 ${virtual_space_coordinate_dim} --energy-regression --momentum --momentum-amp # > test.log
# python display_h5.py ../h5/test.h5 html/test_2025_02_07_150149.html 0 1 ${virtual_space_coordinate_dim} --energy-regression --energy-regression-cluster --momentum --momentum-amp # > test.log

# python display_h5.py ../h5/test.h5 html/test_2025_03_04_162442.html 0 1 ${virtual_space_coordinate_dim} --energy-regression --momentum --momentum-amp # > test.log


# python display_h5.py ../h5/test.h5 html/test_2025_03_04_162442.html 0 1 ${virtual_space_coordinate_dim} --energy-regression --momentum --momentum-amp # > test.log


# python display_h5_240219_cluster_pandora.py ../h5/outfile_display_pandora_uds__.h5 html/out_cluster_pandora_use__.html 0 1










# python display_h5_transformer.py ../h5/display/clustering/test.h5 html/transformer_clustering/test.html 0 10 ${virtual_space_coordinate_dim} # > test.log





## neutron
python display_h5_transformer.py ../h5/display/clustering/test_neutron.h5 html/transformer_clustering/test_neutron.html 0 10 ${virtual_space_coordinate_dim} # > test.log