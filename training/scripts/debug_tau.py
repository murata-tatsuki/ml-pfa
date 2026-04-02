"""
Quick sanity check on TauDataset + GravnetModel + object condensation loss.

Run from repository root::

    python -m training.scripts.debug_tau
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python training/scripts/debug_tau.py`` from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
from torch_geometric.loader import DataLoader

import objectcondensation as oc
from dataset import TauDataset
from gravnet_model import GravnetModel


def debug():
    oc.DEBUG = True
    dataset = TauDataset("data/taus")
    dataset.npzs = []
    for data in DataLoader(dataset, batch_size=len(dataset), shuffle=False):
        break
    print(data.y.sum())
    model = GravnetModel(input_dim=9, output_dim=4)
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.batch)
    pred_betas = torch.sigmoid(out[:, 0])
    pred_cluster_space_coords = out[:, 1:4]
    oc.calc_LV_Lbeta_Eregression(
        pred_betas,
        pred_cluster_space_coords,
        data.y.long(),
        data.batch.long(),
    )


if __name__ == "__main__":
    debug()
