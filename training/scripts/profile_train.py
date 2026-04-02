"""
Minimal CPU profiler over a tiny TauDataset loop (development use).

Run from repository root::

    python -m training.scripts.profile_train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.nn.utils as utils
import tqdm
from torch_geometric.loader import DataLoader
from torch.profiler import ProfilerActivity, profile, record_function

from gravnet_model import GravnetModel
from dataset import TauDataset
from training.loss import loss_fn


def run_profile():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device", device)

    qmin = 1.0
    loss_offset = 1.0
    args = argparse.Namespace(
        energy_regression_weight=False,
        energy_regression=False,
        energy_regression_cluster=False,
        epochs_noLE=15,
        regression_coefficinet=1.0,
        LE_gradually=False,
        beta_track=False,
        beta_track_beginning=False,
        force_track_alpha=False,
        LE_track="alpha",
        LE_cluster="distribution",
        l_beta_suppression=False,
        epochs_nobeta=7,
        jit=False,
        no_clipping=False,
        clip_value=100,
    )

    batch_size = 2
    n_batches = 2
    shuffle = True
    dataset = TauDataset("data/taus")
    dataset.npzs = dataset.npzs[: batch_size * n_batches]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(
        f"Running profiling for {len(dataset)} events, batch_size={batch_size}, {len(loader)} batches"
    )

    model = GravnetModel(input_dim=9, output_dim=8).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=1e-4)

    print("Start limited training loop")
    model.train()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            pbar = tqdm.tqdm(loader, total=len(loader))
            pbar.set_postfix({"loss": "?"})
            for _, data in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()
                result = model(data.x, data.batch)
                if args.jit:
                    raise NotImplementedError("--jit")
                loss, _ = loss_fn(
                    result,
                    data,
                    args,
                    qmin,
                    loss_offset,
                    use_charge_track_likeness=False,
                )
                print(f"loss={float(loss)}")
                loss.backward()
                if not args.no_clipping:
                    utils.clip_grad_value_(
                        model.parameters(), clip_value=args.clip_value
                    )
                optimizer.step()
                pbar.set_postfix({"loss": float(loss)})
    print(prof.key_averages().table(sort_by="cpu_time", row_limit=10))


if __name__ == "__main__":
    run_profile()
