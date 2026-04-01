import argparse
import sys
import matplotlib.pylab as plt
import numpy as np
import torch
from torch_geometric.loader import DataLoader
import tqdm

import objectcondensation as oc
from cli import parse_train_args
from gravnet_model import GravnetModel
from training.loss import loss_fn
from training.run import launch_ddp_training, run_training_single_gpu

import torch.nn.utils as utils

# torch.manual_seed(1009)
torch.autograd.set_detect_anomaly(False)


def main():
    print(sys.argv)
    args = parse_train_args()
    if args.verbose: oc.DEBUG = True
    if args.ddp:
        launch_ddp_training(args)
        return
    run_training_single_gpu(args)


def colorlabel(y, label):
    unique_label = np.unique(label)
    if y == unique_label[0]:
        return "b"
    elif y == unique_label[1]:
        return "g"


def check_plots(coords_list, data_y_list):
    coords_list = np.array(coords_list[0])
    label = np.array(data_y_list[0][0:4000])
    fig, ax = plt.subplots(figsize=(8, 6))
    for x1, y1, label1 in zip(
        coords_list[0:4000, 0], coords_list[0:4000, 1], label
    ):
        ax.scatter(x1, y1, c=colorlabel(label1, label))
    plt.show()


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


def plot_history(train_loss_history, test_loss_history):
    if isinstance(test_loss_history, list):
        plt.figure(figsize=(8, 6))
        plt.plot(test_loss_history, label="test_loss", lw=3, c="b")
        plt.plot(train_loss_history, label="train_loss", lw=3, c="green")
        plt.title("loss function")
        plt.legend(fontsize=14)
        plt.show()


def plot_acc_history(train_acc_history, test_acc_history):
    if isinstance(test_acc_history, list):
        plt.figure(figsize=(8, 6))
        plt.plot(test_acc_history, label="test_acc", lw=3, c="b")
        plt.plot(train_acc_history, label="train_acc", lw=3, c="green")
        plt.title("accuracy")
        plt.legend(fontsize=14)
        plt.show()


def run_profile():
    from torch.profiler import ProfilerActivity, profile, record_function

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
            for i, data in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()
                result = model(data.x, data.batch)
                if args.jit:
                    raise NotImplementedError
                else:
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
                    utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)
                optimizer.step()
                pbar.set_postfix({"loss": float(loss)})
    print(prof.key_averages().table(sort_by="cpu_time", row_limit=10))


if __name__ == "__main__":
    main()
