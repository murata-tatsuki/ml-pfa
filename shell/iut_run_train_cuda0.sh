#!/bin/sh

cd ..

DATE=`date '+%Y_%m_%d_%H%M%S'`


#python train.py -i mydata/ntau_one --epochs=1
#python train.py --no-split -i mydata/ntau_one -ii mydata/ntau_one_validate --epochs=1 --batch-size=50
#python train.py -i mydata2/tau_5GeV_100k --timing-cut --use-charged-cluster-loss --output-dimension 4


#python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --thetaphi --cuda cuda:1 --epochs 50 --beta-track --force-track-alpha --batch-size 5 | tee log/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi.log &



### output dimention tuning
output_coordinate=8   # output coordinate dimension """""""+1"""""""
# outputD=$((output_coordinate+1))
outputD=5
ncuda=0     #1  # 0 or 1
sample=ntau_10GeV_10      # ntau_10GeV_10   uds91   mix   ntau_10to100GeV_10
# python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --thetaphi --cuda cuda:0 --epochs 50 --beta-track --force-track-alpha --batch-size 5 --output-dimension 4 > log/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD4.log
# python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --thetaphi --cuda cuda:0 --epochs 50 --beta-track --force-track-alpha --batch-size 5 --output-dimension 5 > log/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD5.log
## ntau
# python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --no-split --thetaphi --epochs 50 --beta-track --force-track-alpha --batch-size 5 --output-dimension ${outputD} --ckptdir checkpoint/iut/output_dimensions/ckpts_gravnet_new02_${DATE}_outputD${outputD} > log/iut/output_dimension/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD${outputD}_${DATE}.log
## ntau skimmed
# python train.py -i /data/suehara/mldata/pfa/murata/skimmed/ntau_10GeV_10/train -ii /data/suehara/mldata/pfa/murata/skimmed/ntau_10GeV_10/validation --thetaphi --cuda cuda:${ncuda} --epochs 50 --beta-track --force-track-alpha --batch-size 5 --output-dimension ${outputD} --ckptdir checkpoint/output_dimensions/ckpts_gravnet_new02_${DATE}_outputD${outputD}_skimmed > log/output_dimension/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD${outputD}_${DATE}_skimmed.log
# python train.py -i /data/suehara/mldata/pfa/murata/skimmed/ntau_10GeV_10/train -ii /data/suehara/mldata/pfa/murata/skimmed/ntau_10GeV_10/validation --no-split --thetaphi --cuda cuda:${ncuda} --epochs 50 --beta-track --force-track-alpha --batch-size 5 --output-dimension ${outputD} --ckptdir checkpoint/output_dimensions/ckpts_gravnet_new02_${DATE}_outputD${outputD}_skimmed > log/output_dimension/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD${outputD}_${DATE}_skimmed.log
## uds
# python train.py -i /data/suehara/gravnet_ilc/data/uds91/train -ii /data/suehara/gravnet_ilc/data/uds91/validation --no-split --thetaphi --cuda cuda:${ncuda} --epochs 50 --beta-track --force-track-alpha --batch-size 1 --output-dimension ${outputD} --ckptdir checkpoint/iut/output_dimensions/ckpts_gravnet_new02_${DATE}_outputD${outputD} > log/iut/output_dimension/tc_uds91_timingcut_forcealpha_thetaphi_outputD${outputD}_${DATE}.log
## mix sample
# python train.py -i /data/suehara/mldata/pfa/murata/mix_samples/train -ii /data/suehara/mldata/pfa/murata/mix_samples/validation --thetaphi --cuda cuda:${ncuda} --epochs 50 --beta-track --force-track-alpha --batch-size 5 --output-dimension ${outputD} --ckptdir checkpoint/output_dimensions/ckpts_gravnet_new02_${DATE}_outputD${outputD} > log/output_dimension/tc_${sample}_timingcut_forcealpha_thetaphi_outputD${outputD}_${DATE}.log

### learning rate w/ output dimensions
lr=2e-5
lr=5e-5
lr=1e-4
lr=5e-4
# python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --thetaphi --cuda cuda:${ncuda} --epochs 50 --beta-track --force-track-alpha --batch-size 10 --output-dimension ${outputD} --ckptdir checkpoint/output_dimensions/ckpts_gravnet_new02_${DATE}_outputD${outputD} --learning-rate ${lr} > log/output_dimension/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD${outputD}_lr${lr}_${DATE}.log



# python train.py -i /data/suehara/gravnet_ilc/data/uds91/train -ii /data/suehara/gravnet_ilc/data/uds91/validation --thetaphi --cuda cuda:1 --epochs 20 --beta-track --force-track-alpha --batch-size 5 > log/uds91_timingcut_forcealpha_thetaphi.log
python train.py -i /data/suehara/gravnet_ilc/data/ntau_10to100GeV_10/train -ii /data/suehara/data/ntau_10to100GeV_10/validation --no-split --thetaphi --cuda cuda:0 --epochs 50 --beta-track --force-track-alpha --batch-size 1 --output-dimension ${outputD} --ckptdir checkpoint/iut/output_dimensions/ckpts_gravnet_new02_${DATE}_outputD${outputD}_ntau10to100GeV_10 > log/iut/output_dimension/tc_ntau10to100GeV_10_outputD${outputD}_${DATE}.log
# python train.py -i /data/suehara/gravnet_ilc/data/ntau_10to100GeV_10/train -ii /data/suehara/data/ntau_10to100GeV_10/validation --thetaphi --cuda cuda:1 --epochs 20 --beta-track --force-track-alpha --batch-size 1


## test
# python train.py -i /data/suehara/gravnet_ilc/data/uds91/train -ii /data/suehara/gravnet_ilc/data/uds91/validation --thetaphi --cuda cuda:1 --epochs 1 --beta-track --force-track-alpha --batch-size 5 --ckptdir checkpoint/test
