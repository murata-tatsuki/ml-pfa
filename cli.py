"""CLI argument parser for train.py."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence


def build_train_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dry', action='store_true', help='Turn off checkpoint saving and run limited number of events')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more output')
    parser.add_argument('--settings-Sep01', action='store_true', help='Use 21Sep01 settings')
    parser.add_argument('--reduce-noise', action='store_true', help='Randomly kills 95%% of noise')
    parser.add_argument('--timing-cut', action='store_true', help='Eliminate hits outside timing window (4-14 nsec)')
    parser.add_argument('--thetaphi', action='store_true', help='Input theta and phi made from px, py, pz')
    parser.add_argument('--use-charged-cluster-loss', action='store_true', help='Turn on loss function for charged cluster matching')
    parser.add_argument('--ckptdir', type=str)
    parser.add_argument('--cuda', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--epochs-nobeta', type=int, default=7)
    parser.add_argument('--epochs-noLE', type=int, default=15)
    parser.add_argument('--beta-track', action='store_true', help='Include L_beta_track term')
    parser.add_argument('--beta-track-beginning', action='store_true', help='L_beta_track term from epoch 1')
    parser.add_argument('--force-track-alpha', action='store_true', help='Force track as alpha (condensation point)')
    parser.add_argument('--force-innermost-alpha', action='store_true', help='Force innermost hit as alpha (condensation point) for clusters without track')
    parser.add_argument('--output-dimension', type=int, default=3, help='Specify total output dimension (note that 1 dim each is used for beta and charged cluster loss)')
    parser.add_argument('-i', '--inputdir', type=str, required=True, help='Specify input directory for training (required)')
    parser.add_argument('--no-split', action='store_true', help='Do not split sample into training/validating')
    parser.add_argument('-ii', '--inputdir-validate', type=str, help='Specify input directory for validating')
    parser.add_argument('--ilc-sharded', action='store_true', help='Load HDF5 per file without concatenating (lower RAM). Use with many .h5 under -i / -ii.')
    parser.add_argument('--ilc-file-cache', type=int, default=2, help='LRU number of HDF5 files to keep decoded per worker (--ilc-sharded only)')
    parser.add_argument('--learning-rate', type=float, default=9.0e-6)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--energy-regression', action='store_true', help='Turn on energy regression term on loss function and output')
    parser.add_argument('--energy-regression-weight', action='store_true', help='Turn on energy regression term on loss function and output (weighted edep)')
    parser.add_argument('--energy-regression-cluster', action='store_true', help='Turn on energy regression term on loss function and output (cluster energy for neutral particles)')
    parser.add_argument('--regression-coefficinet', type=float, default=1)                       ### energy regression scaling factor
    parser.add_argument('--LE-track', type=str, default='alpha', help='Specify L_E_track loss term')
    parser.add_argument('--LE-cluster', type=str, default='distribution', help='Specify L_E_cluster loss term')
    parser.add_argument('--LE-gradually', action='store_true', help='energy loss term is gradually increases for 10 epochs (LE = LE * (x/10)^2 )')
    parser.add_argument('--momentum', action='store_true', help='Add momentum to GNN input')
    parser.add_argument('--momentum-amp', action='store_true', help='Add absoute momentum to GNN input')
    parser.add_argument('--mctpe', action='store_true', help='Use MC truth momentum and energy for virtual hits')                       ## not using now
    parser.add_argument('--energy-branch', action='store_true', help='Change GNN model to bypass energy')
    parser.add_argument('--restart-period', type=int, default=30)
    parser.add_argument('--jit', action='store_true', help='Use compiled python program')                                               ## not using now
    parser.add_argument('--model-ckpt', type=str, default='', help='Use trained model parameters')
    parser.add_argument('--ReduceLROnPlateau', action='store_true', help='Use ReduceLROnPlateau scheduler')
    parser.add_argument('--qmin', type=float, default=1., help='')
    parser.add_argument('--min-lr', type=float, default=1e-7, help='')
    parser.add_argument('--dp', action='store_true', help='Use dataparallel')
    parser.add_argument('--ddp', action='store_true', help='Use distributed dataparallel')
    parser.add_argument('--gpus', type=str, default=None, help="Comma-separated list of GPU ids to use with --ddp (e.g., '0,1,2'). If not specified, all available GPUs are used.")
    parser.add_argument('--lr-policy', type=str, default='cosine', help='Specify lraning rate policy at lrscheduler.py')
    parser.add_argument('--nrestart-cosreduce', type=int, default=3, help='number of restart without reducing the maximum learning rate')
    parser.add_argument('--clip-value', type=int, default=100, help='threshold of gradient clipping')
    parser.add_argument('--no-clipping', action='store_true', help='do not clip the gradients')           
    parser.add_argument('--l-beta-suppression', action='store_true', help='add to decrease beta of non-condensation point')           
    parser.add_argument('--amp', action='store_true', help='Enable CUDA mixed precision (torch.cuda.amp.autocast). Off: same as before.')
    parser.add_argument('--amp-dtype', type=str, default='bf16', choices=['bf16', 'fp16'], help='AMP compute dtype: bf16 (A100+), fp16 (uses GradScaler). Ignored unless --amp.')
    return parser


def parse_train_args(argv: Optional[Sequence[str]] = None):
    """
    Parse training CLI arguments.

    Parameters
    ----------
    argv :
        If None, uses ``sys.argv[1:]`` (default argparse behavior).
    """
    return build_train_argument_parser().parse_args(argv)
