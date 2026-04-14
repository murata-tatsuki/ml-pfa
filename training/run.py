"""Single-GPU and DDP training entry points."""

from __future__ import annotations

import os
import os.path as osp
import sys
from time import strftime

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import tqdm
from torch_geometric.loader import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler

import objectcondensation as oc
from data_loading import prepare_train_val_datasets
from gravnet_model import GravNetModelBranch, GravnetModel
from lrscheduler import CyclicLRWithRestarts
from model import get_model, get_model_branch
from training.loops import (
    amp_grad_scaler,
    backward_with_optimizer_step,
    ddp_all_reduce_loss_totals,
    eval_batch_loss_components,
    forward_training_loss,
    training_batch_step,
)


def run_requirements(args):
    if args.no_split and args.inputdir_validate is None:
        print("If --no-split is specified, it is required to set --inputdir-validate")
        raise
    if args.regression_coefficinet and args.energy_regression is None:
        print("If --regression-coefficinet is specified, it is required to set --energy-regression")
        raise
    if args.energy_regression_cluster and args.energy_regression is None:
        print("If --cluster-energy is specified, it is required to set --energy-regression")
        raise


def index_setup(args):
    output_dimension = args.output_dimension
    index_pred_tracker_energy = 0
    index_pred_cluster_energy = 0
    index_pred_cluster_space_coords = 0
    if args.energy_regression_weight:
        output_dimension += 5
        index_pred_cluster_energy += 5
    else:
        if args.energy_regression:
            output_dimension += 1
            index_pred_tracker_energy += 1
            if args.energy_regression_cluster:
                output_dimension += 1
                index_pred_cluster_energy += 2
    if args.use_charged_cluster_loss:
        output_dimension += 1
        index_pred_tracker_energy = index_pred_tracker_energy + 1 if index_pred_tracker_energy != 0 else 0
        index_pred_cluster_energy = index_pred_cluster_energy + 1 if index_pred_cluster_energy != 0 else 0

    additional_input_dimension = 0
    if args.momentum:
        additional_input_dimension += 3
        if args.momentum_amp:
            additional_input_dimension += 1

    return (
        output_dimension,
        index_pred_tracker_energy,
        index_pred_cluster_energy,
        index_pred_cluster_space_coords,
        additional_input_dimension,
    )


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_ddp():
    dist.destroy_process_group()


def run_training_ddp(rank, world_size, args):
    setup_ddp(rank, world_size)
    torch.cuda.set_device(rank)

    device = torch.device(f"cuda:{rank}")
    print(device)
    run_requirements(args)
    n_epochs = args.epochs
    batch_size = args.batch_size
    output_dimension = args.output_dimension
    lr_input = args.learning_rate
    weight_decay_input = args.weight_decay
    er_coef = args.regression_coefficinet
    qmin = args.qmin
    min_lr = args.min_lr

    batch_size = batch_size * world_size
    lr_input = lr_input * world_size

    train_dataset, test_dataset, batch_size = prepare_train_val_datasets(args, batch_size)

    (
        output_dimension,
        index_pred_tracker_energy,
        index_pred_cluster_energy,
        index_pred_cluster_space_coords,
        additional_input_dimension,
    ) = index_setup(args)

    print(f"Training dataset size:  {len(train_dataset)}")
    print(f"Validating dataset size:  {len(test_dataset)}")
    print(f"Batch size:  {batch_size}")

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)

    # Model setup
    if args.model_ckpt == "":
        if not args.energy_branch:
            print("Loading GravnetModel")
            model = GravnetModel(input_dim = (6 if args.timing else 5) + args.thetaphi * 2 + additional_input_dimension, output_dim=output_dimension)
        else:
            print("Loading GravnetModel with energy branch")
            model = GravNetModelBranch(input_dim = (6 if args.timing else 5) + args.thetaphi * 2 + additional_input_dimension, output_dim=output_dimension, b_energy_branch=True)
            print(model)
    else:
        print(f"Loading model from checkpoint {args.model_ckpt}")
        if args.energy_branch:
            model = get_model_branch(args.model_ckpt, jit=False, input_dim = (6 if args.timing else 5) + args.thetaphi * 2 + additional_input_dimension, output_dim=output_dimension, ddp=args.ddp)
        else:
            model = get_model(args.model_ckpt, jit=False, input_dim = (6 if args.timing else 5) + args.thetaphi * 2 + additional_input_dimension, output_dim=output_dimension, ddp=args.ddp)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_input, weight_decay=weight_decay_input)
    scaler = amp_grad_scaler(args)
    if getattr(args, "amp", False) and rank == 0:
        print(f"AMP enabled: dtype={args.amp_dtype}, GradScaler={'on' if scaler is not None else 'off'}")
    scheduler = None
    nepoch_factor = None
    if not args.settings_Sep01:
        if args.ReduceLROnPlateau:
            print("use ReduceLROnPlateau scheduler")
            scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, threshold=0.01)
            nepoch_factor = args.epochs_nobeta if not args.energy_regression else max([args.epochs_noLE, args.epochs_nobeta])
            print("epochs to calculate patience ", nepoch_factor)
        else:
            print("restart period : ", args.restart_period)
            scheduler = CyclicLRWithRestarts( optimizer, batch_size, epoch_size, restart_period=args.restart_period, t_mult=1.1, policy=args.lr_policy, min_lr=min_lr, nrestart_cosreduce=args.nrestart_cosreduce)
    loss_offset = 1.0 # To prevent a negative loss from ever occuring

    def train(epoch):
        model.train()
        N_train = len(train_loader)
        loss_components = {}
        gradients = []

        def update(components):
            for key, value in components.items():
                if key not in loss_components:
                    loss_components[key] = value.detach().clone()
                else:
                    loss_components[key] += value.detach()

        if scheduler is not None and not args.settings_Sep01:
            if not args.ReduceLROnPlateau:
                scheduler.step()
        try:
            pbar = tqdm.tqdm(train_loader, total=len(train_loader))
            pbar.set_postfix({"loss": "?"})
            for i, data in enumerate(pbar):
                loss, components = training_batch_step(
                    model,
                    data,
                    device,
                    optimizer,
                    scaler,
                    args,
                    qmin,
                    loss_offset,
                    epoch,
                    scheduler,
                )
                update(components)
                pbar.set_postfix({"loss": float(loss)})
                gradients.append([p.grad.norm().item() for p in model.parameters()])
            layer_grads = np.mean(np.array(gradients), axis=0)
            ddp_all_reduce_loss_totals(loss_components, N_train, device)
            return_loss = None
            if rank == 0:
                print("Training epoch", epoch)
                print(layer_grads)
                print(oc.formatted_loss_components_string_train(loss_components))
                return_loss = loss_components["L_V"] + loss_offset
                if args.LE_track == "alpha_tracker_modifing_charged0":
                    if epoch > args.epochs_nobeta:
                        return_loss += loss_components["L_beta"]
                    if epoch > 15:
                        return_loss += loss_components["L_E"]
                    else:
                        return_loss += loss_components["L_E_charge"]
                else:
                    if epoch > args.epochs_nobeta:
                        return_loss += loss_components["L_beta"]
                    if epoch > args.epochs_noLE:
                        return_loss += loss_components["L_E"]
            train_loss = return_loss.item() if rank == 0 else loss.item()
            return train_loss, None, None, None, None
        except Exception:
            print("Exception encountered:", data, "i:", i)
            raise

    def test(epoch):
        N_test = len(test_loader)
        loss_components = {}

        def update(components):
            for key, value in components.items():
                if key not in loss_components:
                    loss_components[key] = value.detach().clone()
                else:
                    loss_components[key] += value.detach()

        with torch.no_grad():
            model.eval()
            for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                update(
                    eval_batch_loss_components(
                        model,
                        data,
                        device,
                        args,
                        qmin,
                        loss_offset,
                        epoch,
                    )
                )
        ddp_all_reduce_loss_totals(loss_components, N_test, device)
        test_loss = (
            loss_offset
            + loss_components["L_V"]
            + loss_components["L_beta"]
            + loss_components["L_E"]
            if "L_E" in loss_components
            else loss_offset
            + loss_components["L_V"]
            + loss_components["L_beta"]
        )
        if rank == 0:
            print("test " + oc.formatted_loss_components_string(loss_components))
            print(f"Returning {test_loss}")
        return test_loss.item()

    ckpt_dir = strftime("checkpoint/ckpts_gravnet_new02_%b%d_%H%M") if args.ckptdir is None else args.ckptdir

    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = "ckpt_best.pth.tar" if best else "ckpt_{0}_1.pth.tar".format(checkpoint_number)
        ckpt = osp.join(ckpt_dir, ckpt)
        if best:
            print("Saving epoch {0} as new best".format(checkpoint_number))
        if not args.dry:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(dict(model=model.module.state_dict()), ckpt)

    min_loss = 1e9
    train_loss_history = []
    test_loss_history = []
    learning_rates = []

    for i_epoch in range(n_epochs):
        train_sampler.set_epoch(i_epoch)
        test_sampler.set_epoch(i_epoch)
        train_loss, _, _, _, _ = train(i_epoch)
        if rank == 0:
            train_loss_history.append(train_loss)
            learning_rates.append(optimizer.param_groups[0]["lr"])
            print("learning rate : ", learning_rates)
            print("train loss : ", train_loss)
            write_checkpoint(i_epoch)

        test_loss = test(i_epoch)
        if args.ReduceLROnPlateau and nepoch_factor is not None:
            if i_epoch > nepoch_factor:
                scheduler.step(test_loss)
        test_loss_history.append(test_loss)
        if test_loss < min_loss:
            min_loss = test_loss

    cleanup_ddp()


def run_training_single_gpu(args):
    n_epochs = args.epochs
    batch_size = args.batch_size
    output_dimension = args.output_dimension
    lr_input = args.learning_rate
    weight_decay_input = args.weight_decay
    er_coef = args.regression_coefficinet
    qmin = args.qmin
    min_lr = args.min_lr

    device = torch.device(args.cuda) if not args.dp else "cuda"
    print("Using device: ", device)
    if not args.dp:
        torch.cuda.set_device(device)
    if args.dp:
        print("available number of cuda ", torch.cuda.device_count())
        batch_size = batch_size * 2
        lr_input = lr_input * 2
    print("learning rate :", lr_input, ",  weght decay :", weight_decay_input, ", regression coefficient :", er_coef)
    if args.mctpe:
        print("momentum and energy of virtual hits are MC truth")
    else:
        print("momentum and energy of virtual hits are NOT MC truth")
        print("using detected values")

    shuffle = True

    train_dataset, test_dataset, batch_size = prepare_train_val_datasets(args, batch_size)

    (
        output_dimension,
        index_pred_tracker_energy,
        index_pred_cluster_energy,
        index_pred_cluster_space_coords,
        additional_input_dimension,
    ) = index_setup(args)

    print(f"Training dataset size:  {len(train_dataset)}")
    print(f"Validating dataset size:  {len(test_dataset)}")
    print(f"Batch size:  {batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

    if args.model_ckpt == "":
        if not args.energy_branch:
            print("Loading GravnetModel")
            model = GravnetModel(input_dim = (6 if args.timing else 5) + args.thetaphi * 2 + additional_input_dimension, output_dim=output_dimension,)
        else:
            print("Loading GravnetModel with energy branch")
            model = GravNetModelBranch(input_dim = (6 if args.timing else 5) + args.thetaphi * 2 + additional_input_dimension, output_dim=output_dimension, b_energy_branch=True,)
            print(model)
    else:
        print(f"Loading model from checkpoint {args.model_ckpt}")
        if args.energy_branch:
            model = get_model_branch(args.model_ckpt, jit=False, input_dim = (6 if args.timing else 5) + args.thetaphi * 2 + additional_input_dimension, output_dim=output_dimension)
        else:
            model = get_model(args.model_ckpt, jit=False, input_dim = (6 if args.timing else 5) + args.thetaphi * 2 + additional_input_dimension, output_dim=output_dimension)
    if not args.dp:
        model.to(device)
    else:
        model.cuda()
        model = nn.DataParallel(model)
        torch.backends.cudnn.benchmark = True

    epoch_size = len(train_loader.dataset)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_input, weight_decay=weight_decay_input)
    scaler = amp_grad_scaler(args)
    if getattr(args, "amp", False):
        print(f"AMP enabled: dtype={args.amp_dtype}, GradScaler={'on' if scaler is not None else 'off'}")

    scheduler = None
    nepoch_factor = None
    if not args.settings_Sep01:
        if args.ReduceLROnPlateau:
            print("use ReduceLROnPlateau scheduler")
            scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, threshold=0.01)
            nepoch_factor = args.epochs_nobeta if not args.energy_regression else max([args.epochs_noLE, args.epochs_nobeta])
            print("epochs to calculate patience ", nepoch_factor)
        else:
            print("restart period : ", args.restart_period)
            scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=args.restart_period, t_mult=1.1, policy=args.lr_policy, min_lr=min_lr, nrestart_cosreduce=args.nrestart_cosreduce)

    loss_offset = 1.0

    def check_coords(out, data):
        learning_para = {}
        pred_cluster_space_coords = out[:, 1:]
        learning_para["pred_cluster_space_coords"] = pred_cluster_space_coords
        learning_para["data.y.long"] = data.y.long()
        learning_para["data.batch"] = data.batch
        return learning_para

    def check_data(data):
        data_para = {}
        data_para["data.y.long"] = data.y.long()
        data_para["data.x"] = data.x
        return data_para

    def train(epoch):
        print("Training epoch", epoch)
        cluster_space_coords_list = []
        data_y_list = []
        model.train()
        N_train = len(train_loader)
        loss_components = {}
        gradients = []

        def update(components):
            for key, value in components.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value

        if scheduler is not None and not args.settings_Sep01:
            if not args.ReduceLROnPlateau:
                scheduler.step()
        try:
            pbar = tqdm.tqdm(train_loader, total=len(train_loader))
            pbar.set_postfix({"loss": "?"})
            for i, data in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()
                if i == 0:
                    first_para = check_data(data)
                loss, components, result = forward_training_loss(model, data, device, args, qmin, loss_offset, epoch)
                learning_para = check_coords(result, data)
                update(components)
                backward_with_optimizer_step(loss, model, optimizer, scaler, args, scheduler)
                pbar.set_postfix({"loss": float(loss)})
                cluster_space_coords_list.append(learning_para["pred_cluster_space_coords"].tolist())
                data_y_list.append(learning_para["data.y.long"].tolist())
                gradients.append([p.grad.norm().item() for p in model.parameters()])
            layer_grads = np.mean(np.array(gradients), axis=0)
            print(layer_grads)
            for key in loss_components:
                loss_components[key] /= N_train
            print(oc.formatted_loss_components_string_train(loss_components))
            return loss.item(), cluster_space_coords_list, data_y_list, data, first_para
        except Exception:
            print("Exception encountered:", data, "i:", i)
            raise

    def test(epoch):
        N_test = len(test_loader)
        loss_components = {}

        def update(components):
            for key, value in components.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value

        with torch.no_grad():
            model.eval()
            for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                update(
                    eval_batch_loss_components(model, data, device, args, qmin, loss_offset, epoch)
                )
        for key in loss_components:
            loss_components[key] /= N_test
        print("test " + oc.formatted_loss_components_string(loss_components))
        test_loss = (
            loss_offset
            + loss_components["L_V"]
            + loss_components["L_beta"]
            + loss_components["L_E"]
            if "L_E" in loss_components
            else loss_offset
            + loss_components["L_V"]
            + loss_components["L_beta"]
        )
        print(f"Returning {test_loss}")
        return test_loss.item()

    ckpt_dir = strftime("checkpoint/ckpts_gravnet_new02_%b%d_%H%M") if args.ckptdir is None else args.ckptdir

    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = ("ckpt_best.pth.tar" if best else "ckpt_{0}_1.pth.tar".format(checkpoint_number))
        ckpt = osp.join(ckpt_dir, ckpt)
        if best:
            print("Saving epoch {0} as new best".format(checkpoint_number))
        if not args.dry:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(dict(model=model.state_dict()), ckpt)

    min_loss = 1e9
    train_loss_history = []
    test_loss_history = []
    learning_rates = []

    for i_epoch in range(n_epochs):
        train_loss, cluster_space_para, data_y, data, first_para = train(i_epoch)
        learning_rates.append(optimizer.param_groups[0]["lr"])
        print("learning rate : ", learning_rates)
        train_loss_history.append(train_loss)
        print("train loss : ", train_loss)
        write_checkpoint(i_epoch)

        test_loss = test(i_epoch)
        if args.ReduceLROnPlateau and nepoch_factor is not None:
            if i_epoch > nepoch_factor:
                scheduler.step(test_loss)
        test_loss_history.append(test_loss)
        if test_loss < min_loss:
            min_loss = test_loss


def launch_ddp_training(args):
    """Validate GPUs, set ``CUDA_VISIBLE_DEVICES`` if needed, and spawn DDP workers."""
    if args.gpus is not None:
        visible_gpus = [int(x.strip()) for x in args.gpus.split(",") if x.strip()]
        if not visible_gpus:
            print("Error: --gpus must specify at least one GPU (e.g., --gpus 0,1)")
            sys.exit(1)
        n_gpus = torch.cuda.device_count()
        invalid = [g for g in visible_gpus if g < 0 or g >= n_gpus]
        if invalid:
            print(f"Error: Invalid GPU id(s) {invalid}. Available GPUs: 0-{n_gpus - 1}")
            sys.exit(1)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"DDP: Using selected GPUs: {visible_gpus} (CUDA_VISIBLE_DEVICES={args.gpus})")
    else:
        visible_gpus = list(range(torch.cuda.device_count()))
        if not visible_gpus:
            print("Error: No CUDA GPUs available")
            sys.exit(1)
        print(f"DDP: Using all available GPUs: {visible_gpus}")

    world_size = len(visible_gpus)
    mp.spawn(run_training_ddp, args=(world_size, args), nprocs=world_size, join=True)
