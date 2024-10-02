#!/bin/sh

cd /data/suehara/mldata/pfa/murata/mix_samples

DATE=`date '+%Y_%m_%d_%H%M%S'`


mix_path=/data/suehara/mldata/pfa/murata/mix_samples
ntau_path=/data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10
uds_path=/data/suehara/gravnet_ilc/data/uds91

# for file in $( ls ${ntau_path}/train ); do
#   file_=${file##*_}
#   file__=${file_%.*}
#   # num_origin=$(printf "%03d" $file__)
#   num_origin=${file__}
# 
#   quotient=$(( num_origin / 5 ))
#   remain=$(( num_origin % 5 ))
#   num=$(( quotient * 6 + remain ))
#   num_fill=$(printf "%03d" $num)
#   echo ${num_fill} ${num} ${quotient} ${remain}
#   # echo ${num}
#   cp ${ntau_path}/train/${file} ${mix_path}/train/mix_${num_fill}.h5
# done

i=0
for file in $( ls ${uds_path}/train ); do
  file_=${file##*_}
  file__=${file_%.*}
  # num_origin=$(printf "%03d" $file__)
  num_origin=${file__}

  if [ ${num_origin} -gt 200 ]; then
    continue
  fi

  num=$(( num_origin * 6 - 1 ))
  num_fill=$(printf "%03d" $num)
  echo ${num_fill} ${num} ${num_origin}
  # echo ${num}
  cp ${uds_path}/train/${file} ${mix_path}/train/mix_${num_fill}.h5
  # rm ${mix_path}/train/mix_${num_fill}.h5
done


#python train.py -i mydata/ntau_one --epochs=1
#python train.py --no-split -i mydata/ntau_one -ii mydata/ntau_one_validate --epochs=1 --batch-size=50
#python train.py -i mydata2/tau_5GeV_100k --timing-cut --use-charged-cluster-loss --output-dimension 4


#python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --thetaphi --cuda cuda:1 --epochs 50 --beta-track --force-track-alpha --batch-size 5 | tee log/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi.log &

## output dimention tuning
# outputD=5  #9
# ncuda=0     #1  # 0 or 1
# python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --thetaphi --cuda cuda:0 --epochs 50 --beta-track --force-track-alpha --batch-size 5 --output-dimension 4 > log/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD4.log
# python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --thetaphi --cuda cuda:0 --epochs 50 --beta-track --force-track-alpha --batch-size 5 --output-dimension 5 > log/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD5.log
# python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --thetaphi --cuda cuda:${ncuda} --epochs 50 --beta-track --force-track-alpha --batch-size 5 --output-dimension ${outputD} --ckptdir checkpoint/output_dimensions/ckpts_gravnet_new02_${DATE}_outputD${outputD} > log/output_dimension/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD${outputD}.log

# python train.py -i /data/suehara/gravnet_ilc/data/uds91/train -ii /data/suehara/gravnet_ilc/data/uds91/validation --thetaphi --cuda cuda:${ncuda} --epochs 50 --beta-track --force-track-alpha --batch-size 5 --output-dimension ${outputD} --ckptdir checkpoint/output_dimensions/ckpts_gravnet_new02_${DATE}_outputD${outputD} > log/output_dimension/tc_uds91_timingcut_forcealpha_thetaphi_outputD${outputD}.log

# python train.py -i /data/suehara/gravnet_ilc/data/uds91/train -ii /data/suehara/gravnet_ilc/data/uds91/validation --thetaphi --cuda cuda:1 --epochs 20 --beta-track --force-track-alpha --batch-size 5 > log/uds91_timingcut_forcealpha_thetaphi.log
# python train.py -i /data/suehara/gravnet_ilc/data/ntau_10to100GeV_10/train -ii /data/suehara/data/ntau_10to100GeV_10/validation --thetaphi --cuda cuda:1 --epochs 20 --beta-track --force-track-alpha --batch-size 1 > log/ntau_10to100_timingcut_forcealpha_thetaphi.log
# python train.py -i /data/suehara/gravnet_ilc/data/ntau_10to100GeV_10/train -ii /data/suehara/data/ntau_10to100GeV_10/validation --thetaphi --cuda cuda:1 --epochs 20 --beta-track --force-track-alpha --batch-size 1


## test
# python train.py -i /data/suehara/gravnet_ilc/data/uds91/train -ii /data/suehara/gravnet_ilc/data/uds91/validation --thetaphi --cuda cuda:1 --epochs 20 --beta-track --force-track-alpha --batch-size 5 > log/test_uds91_timingcut_forcealpha_thetaphi.log
