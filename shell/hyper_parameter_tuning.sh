#!/bin/sh

cd ..
#python train.py -i mydata/ntau_one --epochs=1
#python train.py --no-split -i mydata/ntau_one -ii mydata/ntau_one_validate --epochs=1 --batch-size=50
#python train.py -i mydata2/tau_5GeV_100k --timing-cut --use-charged-cluster-loss --output-dimension 4


# lr 9.0e-6   -8 ~ -2
# wd 1e-4     -6 ~ -2

i=0
while [ $i -ne 10 ]
do
  lr=$(($RANDOM % 7 -7))
  wd=$(($RANDOM % 5 -5))
  num_lr=$(bc <<< "scale=3; $RANDOM/32767")
  num_wd=$(bc <<< "scale=3; $RANDOM/32767")
  RESULT=$(bc <<< "scale=1; 1 / 10.0")
  # if [ ${num_wd} -lt ${RESULT} ]; then
  echo ${num_lr} ${lr} ${num_wd} ${wd} ${RESULT}
  if [ "$(echo "${num_wd} < ${RESULT}" | bc)" -eq 1  -o "$(echo "${num_lr} < ${RESULT}" | bc)" -eq 1 ]; then
    # echo ${num_wd} or ${num_lr} is less than 0.1
    continue
  fi

  python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --thetaphi --cuda cuda:0 --epochs 20 --beta-track --force-track-alpha --batch-size 5 --learning-rate ${num_lr}e${lr} --weight-decay ${num_wd}e${wd} > log/learning_rate/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_${num_lr}e${lr}_${num_wd}e${wd}.log
  # python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --thetaphi --cuda cuda:1 --epochs 20 --beta-track --force-track-alpha --batch-size 5 --learning-rate ${num_lr}e${lr} --weight-decay ${num_wd}e${wd} > log/learning_rate/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_${num_lr}e${lr}_${num_wd}e${wd}.log
  # python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --thetaphi --cuda cuda:1 --epochs 20 --beta-track --force-track-alpha --batch-size 5 --learning-rate ${num_lr}e${lr} --weight-decay ${num_wd}e${wd}
  # bsub -q s -o job/output.%J -e job/errors.%J "python train.py -i /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/train -ii /data/suehara/gravnet_ilc/data/ntau/tc_ntau_10GeV_10/validation --thetaphi --cuda cuda:1 --epochs 20 --beta-track --force-track-alpha --batch-size 5 --learning-rate ${num_lr}e${lr} --weight-decay ${num_wd}e${wd} > log/learning_rate/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_${num_lr}e${lr}_${num_wd}e${wd}.log"
  let i++
done

# python train.py -i /data/suehara/gravnet_ilc/data/uds91/train -ii /data/suehara/gravnet_ilc/data/uds91/validation --thetaphi --cuda cuda:1 --epochs 20 --beta-track --force-track-alpha --batch-size 5 > log/uds91_timingcut_forcealpha_thetaphi.log
# python train.py -i /data/suehara/gravnet_ilc/data/ntau_10to100GeV_10/train -ii /data/suehara/data/ntau_10to100GeV_10/validation --thetaphi --cuda cuda:1 --epochs 20 --beta-track --force-track-alpha --batch-size 1 > log/ntau_10to100_timingcut_forcealpha_thetaphi.log


## test
# python train.py -i /data/suehara/gravnet_ilc/data/uds91/train -ii /data/suehara/gravnet_ilc/data/uds91/validation --thetaphi --cuda cuda:1 --epochs 20 --beta-track --force-track-alpha --batch-size 5 > log/test_uds91_timingcut_forcealpha_thetaphi.log
