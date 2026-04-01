import sys

import objectcondensation as oc
from cli import parse_train_args
from training.run import launch_ddp_training, run_training_single_gpu

import torch

torch.autograd.set_detect_anomaly(False)


def main():
    print(sys.argv)
    args = parse_train_args()
    if args.verbose:
        oc.DEBUG = True
    if args.ddp:
        launch_ddp_training(args)
        return
    run_training_single_gpu(args)


if __name__ == "__main__":
    main()
