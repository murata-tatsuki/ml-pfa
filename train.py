import os, os.path as osp
from contextlib import nullcontext
from time import strftime
import tqdm
import torch
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import DataLoader
import argparse
import matplotlib.pylab as plt
import numpy as np

#from sklearn.metrics import accuracy_score
#import torch_cmspepr.objectcondensation as oc
# import objectcondensation as oc
import objectcondensation as oc
#import torch.nn.functional as f

from gravnet_model import GravnetModel,GravNetModelBranch,GravnetModelWithNoiseFilter,GravNetModelMultiHead
from dataset import ILCDataset
from dataset_ilc_sharded import ILCDatasetSharded


def make_ilc_dataset(args, inputdir):
    """ILCDataset と ILCDatasetSharded を引数で切り替え（デフォルトは従来どおり ILCDataset）。"""
    common = dict(
        timingCut=args.timing_cut,
        thetaphi=args.thetaphi,
        test_mode=True,
        momentum=args.momentum,
        momentumAmp=args.momentum_amp,
        mctpe=args.mctpe,
    )
    if getattr(args, "ilc_sharded", False):
        print(
            "Using ILCDatasetSharded (per-file load, no concatenate). "
            f"file_cache_size={getattr(args, 'ilc_file_cache', 2)}"
        )
        return ILCDatasetSharded(
            inputdir,
            **common,
            file_cache_size=getattr(args, "ilc_file_cache", 2),
        )
    return ILCDataset(inputdir, **common)
from lrscheduler import CyclicLRWithRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from sklearn.manifold import TSNE
from model import get_model, get_model_branch

#from ReadText import ReadText
import sys

# for distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# clipping
import torch.nn.utils as utils

#torch.manual_seed(1009)
torch.autograd.set_detect_anomaly(False)


def amp_autocast(args):
    """AMP オフ時は空のコンテキスト（従来と同じ挙動）。"""
    if not getattr(args, "amp", False):
        return nullcontext()
    dt = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return autocast(enabled=True, dtype=dt)


def amp_grad_scaler(args):
    """fp16 のときのみ GradScaler を使う。AMP オフ時は None。"""
    if not getattr(args, "amp", False):
        return None
    if args.amp_dtype == "fp16":
        return GradScaler()
    return None

def run_requirements(args):
    if (args.no_split and args.inputdir_validate is None):
        print("If --no-split is specified, it is required to set --inputdir-validate")
        raise
    if (args.regression_coefficinet and args.energy_regression is None):
        print("If --regression-coefficinet is specified, it is required to set --energy-regression")
        raise
    if (args.energy_regression_cluster and args.energy_regression is None):
        print("If --cluster-energy is specified, it is required to set --energy-regression")
        raise
    if args.use_multihead_model and args.energy_regression_weight:
        print("--use-multihead-model does not support --energy-regression-weight yet")
        raise
    if args.use_multihead_model and args.energy_regression_cluster and args.multihead_regression_heads < 2:
        print("--energy-regression-cluster with --use-multihead-model requires --multihead-regression-heads >= 2")
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
        if (args.energy_regression):
            output_dimension += 1   # adding track energy to model output
            index_pred_tracker_energy += 1
            if (args.energy_regression_cluster):
                output_dimension += 1   # adding cluster energy to model output
                index_pred_cluster_energy += 2
    if (args.use_charged_cluster_loss):
        output_dimension += 1   # adding output dimension
        index_pred_tracker_energy = index_pred_tracker_energy + 1 if index_pred_tracker_energy!=0 else 0
        index_pred_cluster_energy = index_pred_cluster_energy + 1 if index_pred_cluster_energy!=0 else 0

    additional_input_dimension = 0
    if (args.momentum):
        additional_input_dimension += 3   # adding momentum to model input
        if (args.momentum_amp):
            additional_input_dimension += 1     # adding momentum amplitude to model input
    
    return output_dimension, index_pred_tracker_energy, index_pred_cluster_energy, index_pred_cluster_space_coords, additional_input_dimension


def load_checkpoint_state(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    from collections import OrderedDict
    cleaned_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = key.replace("module.", "") if key.startswith("module.") else key
        cleaned_state_dict[name] = value
    model.load_state_dict(cleaned_state_dict, strict=False)
    return model


def get_model_outputs(result, args):
    if args.use_multihead_model:
        if not isinstance(result, dict):
            raise ValueError("Expected dict output from GravNetModelMultiHead")
        out = result["clustering"]
        regression_heads = result.get("regressions", [])
        return out, regression_heads
    return result, None


def build_model(args, input_dim, output_dimension, ddp=False):
    if args.use_multihead_model:
        clustering_output_dim = args.output_dimension + (1 if args.use_charged_cluster_loss else 0)
        n_reg_heads = args.multihead_regression_heads if args.energy_regression else 0
        n_heads = 1 + n_reg_heads
        regression_dims = [1] * n_reg_heads
        model = GravNetModelMultiHead(
            input_dim=input_dim,
            output_dim=clustering_output_dim,
            n_heads=n_heads,
            regression_output_dims=regression_dims if n_reg_heads > 0 else 1,
            interaction_start_epoch=args.multihead_interaction_start_epoch,
            interaction_mode=args.multihead_interaction_mode,
        )
        if args.model_ckpt != "":
            print(f"Loading multi-head model from checkpoint {args.model_ckpt}")
            model = load_checkpoint_state(model, args.model_ckpt)
        else:
            print(
                "Loading GravNetModelMultiHead "
                f"(heads={n_heads}, reg_heads={n_reg_heads}, interaction={args.multihead_interaction_mode})"
            )
        return model

    if args.model_ckpt == "":
        if not args.energy_branch:
            print(f"Loading GravnetModel")
            model = GravnetModel(input_dim=input_dim, output_dim=output_dimension)
        else:
            print(f"Loading GravnetModel with energy branch")
            model = GravNetModelBranch(input_dim=input_dim, output_dim=output_dimension, b_energy_branch=True)
            print(model)
    else:
        print(f"Loading model from checkpoint {args.model_ckpt}")
        if args.energy_branch:
            model = get_model_branch(args.model_ckpt, jit=False, input_dim=input_dim, output_dim=output_dimension, ddp=ddp)
        else:
            model = get_model(args.model_ckpt, jit=False, input_dim=input_dim, output_dim=output_dimension, ddp=ddp)
    return model


def setup_ddp(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def reduce_loss_components(loss_components, n_batches, device, distributed):
    total_batches = n_batches
    if distributed:
        nbatches = torch.tensor([n_batches], device=device, dtype=torch.long)
        dist.all_reduce(nbatches, op=dist.ReduceOp.SUM)
        total_batches = nbatches.item()
        for key in loss_components:
            dist.all_reduce(loss_components[key], op=dist.ReduceOp.SUM)
    for key in loss_components:
        loss_components[key] /= total_batches
    return total_batches


def compose_return_loss(loss_components, args, epoch, loss_offset):
    return_loss = loss_components["L_V"] + loss_offset
    if args.LE_track == 'alpha_tracker_modifing_charged0':
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
    return return_loss


def run_training(rank, world_size, args):
    distributed = args.ddp and world_size > 1
    is_main_process = (not distributed) or rank == 0

    def log(*items, **kwargs):
        if is_main_process:
            print(*items, **kwargs)

    if distributed:
        setup_ddp(rank, world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        print(device)
    else:
        device = torch.device(args.cuda) if not args.dp else torch.device("cuda")
        print('Using device: ', device)
        if not args.dp and device.type == "cuda":
            torch.cuda.set_device(device)

    try:
        run_requirements(args)
        batch_size = args.batch_size
        lr_input = args.learning_rate
        weight_decay_input = args.weight_decay
        qmin = args.qmin
        min_lr = args.min_lr
        if distributed:
            batch_size *= world_size
            lr_input *= world_size
        elif args.dp:
            print("available number of cuda ", torch.cuda.device_count())
            batch_size *= 2
            lr_input *= 2

        log("learning rate :", lr_input, ",  weght decay :", weight_decay_input, ", regression coefficient :", args.regression_coefficinet)
        if args.mctpe:
            log("momentum and energy of virtual hits are MC truth")
        else:
            log("momentum and energy of virtual hits are NOT MC truth")
            log("using detected values")

        log(f'thetaphi at main: {args.thetaphi}')
        log("Loading dataset...")
        dataset = make_ilc_dataset(args, args.inputdir)
        if args.reduce_noise:
            dataset.reduce_noise = .70
            multiply_batch_size = 1
            log(f'Throwing away {dataset.reduce_noise*100:.0f}% of noise (good for testing ideas, not for final results)')
            log(f'Batch size: {batch_size} --> {multiply_batch_size*batch_size}')
            batch_size *= multiply_batch_size
        if args.dry:
            keep = .005
            log(f'Keeping only {100.*keep:.1f}% of events for debugging')
            dataset, _ = dataset.split(keep)

        if args.no_split:
            train_dataset = dataset
            test_dataset = make_ilc_dataset(args, args.inputdir_validate)
        else:
            train_dataset, test_dataset = dataset.split(.8)

        output_dimension, _, _, _, additional_input_dimension = index_setup(args)
        log(f"Training dataset size:  {len(train_dataset)}")
        log(f"Validating dataset size:  {len(test_dataset)}")
        log(f"Batch size:  {batch_size}")

        train_sampler = None
        test_sampler = None
        if distributed:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=8, pin_memory=True, persistent_workers=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)

        model = build_model(
            args=args,
            input_dim=5 + args.thetaphi * 2 + additional_input_dimension,
            output_dimension=output_dimension,
            ddp=args.ddp,
        )
        if distributed:
            model.to(rank)
            model = DDP(model, device_ids=[rank])
        elif args.dp:
            model.cuda()
            model = torch.nn.DataParallel(model)
            torch.backends.cudnn.benchmark = True
        else:
            model.to(device)

        epoch_size = len(train_loader.dataset)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_input, weight_decay=weight_decay_input)
        scaler = amp_grad_scaler(args)
        if getattr(args, "amp", False):
            log(f"AMP enabled: dtype={args.amp_dtype}, GradScaler={'on' if scaler is not None else 'off'}")

        scheduler = None
        nepoch_factor = None
        if not args.settings_Sep01:
            if args.ReduceLROnPlateau:
                log("use ReduceLROnPlateau scheduler")
                scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, threshold=0.01)
                nepoch_factor = args.epochs_nobeta if not args.energy_regression else max([args.epochs_noLE, args.epochs_nobeta])
                log("epochs to calculate patience ", nepoch_factor)
            else:
                log("restart period : ", args.restart_period)
                scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=args.restart_period, t_mult=1.1, policy=args.lr_policy, min_lr=min_lr, nrestart_cosreduce=args.nrestart_cosreduce)

        loss_offset = 1.

        def loss_fn(out, data, i_epoch=None, return_components=False, use_charge_track_likeness=False, regression_heads=None):
            device = out.device
            pred_betas = torch.sigmoid(out[:,0])
            pred_charge_track_likeness = None
            pred_tracker_energy = None
            pred_cluster_energy = None
            weight_photon = None
            weight_charged_hadron = None
            weight_neutral_hadron = None
            weight_muon = None
            weight_electron = None
            if args.use_multihead_model:
                if use_charge_track_likeness:
                    pred_charge_track_likeness = torch.sigmoid(out[:,1])
                    pred_cluster_space_coords = out[:,2:]
                    assert(pred_charge_track_likeness.device == device)
                else:
                    pred_cluster_space_coords = out[:,1:]
                if args.energy_regression and regression_heads is not None and len(regression_heads) > 0:
                    pred_tracker_energy = regression_heads[0].squeeze(-1)
                if args.energy_regression and args.energy_regression_cluster and regression_heads is not None and len(regression_heads) > 1:
                    pred_cluster_energy = regression_heads[1].squeeze(-1)
            elif args.energy_regression_weight:
                if args.energy_regression and not args.energy_regression_cluster:
                    pred_tracker_energy = out[:,1]
                    weight_photon = out[:,2]
                    weight_charged_hadron = out[:,3]
                    weight_neutral_hadron = out[:,4]
                    weight_muon = out[:,5]
                    weight_electron = out[:,6]
                    pred_cluster_space_coords = out[:,7:]
                elif args.energy_regression and args.energy_regression_cluster:
                    pred_tracker_energy = out[:,1]
                    pred_cluster_energy = out[:,2]
                    weight_photon = out[:,3]
                    weight_charged_hadron = out[:,4]
                    weight_neutral_hadron = out[:,5]
                    weight_muon = out[:,6]
                    weight_electron = out[:,7]
                    pred_cluster_space_coords = out[:,8:]
                elif not args.energy_regression:
                    weight_photon = out[:,1]
                    weight_charged_hadron = out[:,2]
                    weight_neutral_hadron = out[:,3]
                    weight_muon = out[:,4]
                    weight_electron = out[:,5]
                    pred_cluster_space_coords = out[:,6:]
            else:
                if args.energy_regression:
                    if not args.energy_regression_cluster:
                        if use_charge_track_likeness:
                            pred_charge_track_likeness = torch.sigmoid(out[:,1])
                            pred_tracker_energy = out[:,2]
                            pred_cluster_space_coords = out[:,3:]
                            assert(pred_charge_track_likeness.device == device)
                        else:
                            pred_tracker_energy = out[:,1]
                            pred_cluster_space_coords = out[:,2:]
                    else:
                        if use_charge_track_likeness:
                            pred_charge_track_likeness = torch.sigmoid(out[:,1])
                            pred_tracker_energy = out[:,2]
                            pred_cluster_energy = out[:,3]
                            pred_cluster_space_coords = out[:,4:]
                            assert(pred_charge_track_likeness.device == device)
                        else:
                            pred_tracker_energy = out[:,1]
                            pred_cluster_energy = out[:,2]
                            pred_cluster_space_coords = out[:,3:]
                else:
                    if use_charge_track_likeness:
                        pred_charge_track_likeness = torch.sigmoid(out[:,1])
                        pred_cluster_space_coords = out[:,2:]
                        assert(pred_charge_track_likeness.device == device)
                    else:
                        pred_cluster_space_coords = out[:,1:]

            cluster_track_index = data.y[:,1]
            assert all(t.device == device for t in [pred_betas, pred_cluster_space_coords, data.y, data.batch])
            true_energy = torch.sqrt(torch.sum(torch.square(data.label[:,4:8]), 1))
            detected_energy = data.feat[:,0]
            LE_weight = 0 if (i_epoch <= args.epochs_noLE) else (1 if (i_epoch > args.epochs_noLE + 10) else pow((i_epoch - args.epochs_noLE), 2) / 100.0)
            er_coef = args.regression_coefficinet * LE_weight if args.LE_gradually else args.regression_coefficinet
            mcpdg = data.label[:,2]
            mccharge = data.label[:,3]

            LV, Lbeta, LE, LE_charge, out_oc = oc.calc_LV_Lbeta(
                pred_betas,
                pred_cluster_space_coords,
                pred_charge_track_likeness,
                data.y[:,0].long(),
                true_energy,
                data.batch,
                return_components=return_components,
                beta_term_option='short-range-potential',
                beta_track_term=args.beta_track,
                beta_track_term_beginning=args.beta_track_beginning,
                force_track_alpha=args.force_track_alpha,
                cluster_track_index=cluster_track_index,
                qmin=qmin,
                tracker_energy=pred_tracker_energy,
                detected_energy=detected_energy,
                er_coef=er_coef,
                LE_track=args.LE_track,
                LE_cluster=args.LE_cluster,
                Ecl_regression=args.energy_regression_cluster,
                weight_regression=args.energy_regression_weight,
                pred_cluster_energy=pred_cluster_energy,
                l_beta_suppression=args.l_beta_suppression,
                epoch=i_epoch,
                mcpdg=mcpdg,
                mccharge=mccharge,
                weight_photon=weight_photon,
                weight_charged_hadron=weight_charged_hadron,
                weight_neutral_hadron=weight_neutral_hadron,
                weight_muon=weight_muon,
                weight_electron=weight_electron,
            )

            if return_components:
                return out_oc
            loss_terms = {"L_V": LV, "L_beta": Lbeta, "L_E": LE, "L_E_charge": LE_charge}
            return compose_return_loss(loss_terms, args, i_epoch, loss_offset), out_oc

        def update_loss_dict(loss_components, components):
            for key, value in components.items():
                if key not in loss_components:
                    loss_components[key] = value.detach().clone()
                else:
                    loss_components[key] += value.detach()

        def train(epoch):
            log('Training epoch', epoch)
            model.train()
            n_train = len(train_loader)
            loss_components = {}
            gradients = []
            if scheduler is not None and not args.ReduceLROnPlateau:
                scheduler.step()
            try:
                pbar = tqdm.tqdm(train_loader, total=len(train_loader))
                pbar.set_postfix({'loss': '?'})
                for i, data in enumerate(pbar):
                    data = data.to(device)
                    optimizer.zero_grad()
                    with amp_autocast(args):
                        if args.use_multihead_model:
                            result = model(data.x, data.batch, epoch=epoch, return_dict=True)
                        else:
                            result = model(data.x, data.batch)
                        out, regression_heads = get_model_outputs(result, args)
                        if args.jit:
                            raise
                        loss, components = loss_fn(out, data, i_epoch=epoch, use_charge_track_likeness=args.use_charged_cluster_loss, regression_heads=regression_heads)
                        update_loss_dict(loss_components, components)
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        if not args.no_clipping:
                            scaler.unscale_(optimizer)
                            utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        if not args.no_clipping:
                            utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)
                        optimizer.step()
                    if scheduler is not None and not args.ReduceLROnPlateau:
                        scheduler.batch_step()
                    pbar.set_postfix({'loss': float(loss)})
                    gradients.append([p.grad.norm().item() for p in model.parameters() if p.grad is not None])

                layer_grads = np.mean(np.array(gradients), axis=0) if gradients else np.array([])
                reduce_loss_components(loss_components, n_train, device, distributed)
                if is_main_process:
                    print(layer_grads)
                    print(oc.formatted_loss_components_string_train(loss_components))
                return compose_return_loss(loss_components, args, epoch, loss_offset).item() if distributed else loss.item()
            except Exception:
                print('Exception encountered:', data, 'i:', i)
                raise

        def test(epoch):
            n_test = len(test_loader)
            loss_components = {}
            with torch.no_grad():
                model.eval()
                for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                    data = data.to(device)
                    with amp_autocast(args):
                        if args.use_multihead_model:
                            result = model(data.x, data.batch, epoch=epoch, return_dict=True)
                        else:
                            result = model(data.x, data.batch)
                        out, regression_heads = get_model_outputs(result, args)
                        if args.jit:
                            raise
                        update_loss_dict(loss_components, loss_fn(out, data, i_epoch=epoch, return_components=True, use_charge_track_likeness=args.use_charged_cluster_loss, regression_heads=regression_heads))
            reduce_loss_components(loss_components, n_test, device, distributed)
            test_loss = loss_offset + loss_components['L_V'] + loss_components['L_beta'] + loss_components['L_E'] if 'L_E' in loss_components else loss_offset + loss_components['L_V'] + loss_components['L_beta']
            if is_main_process:
                print('test ' + oc.formatted_loss_components_string(loss_components))
                print(f'Returning {test_loss}')
            return test_loss.item()

        ckpt_dir = strftime('checkpoint/ckpts_gravnet_new02_%b%d_%H%M') if args.ckptdir is None else args.ckptdir

        def write_checkpoint(checkpoint_number=None, best=False):
            ckpt = 'ckpt_best.pth.tar' if best else 'ckpt_{0}_1.pth.tar'.format(checkpoint_number)
            ckpt = osp.join(ckpt_dir, ckpt)
            if best:
                print('Saving epoch {0} as new best'.format(checkpoint_number))
            if not args.dry:
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(dict(model=unwrap_model(model).state_dict()), ckpt)

        min_loss = 1e9
        train_loss_history = []
        test_loss_history = []
        learning_rates = []
        for i_epoch in range(args.epochs):
            if distributed:
                train_sampler.set_epoch(i_epoch)
                test_sampler.set_epoch(i_epoch)
            train_loss = train(i_epoch)
            if is_main_process:
                learning_rates.append(optimizer.param_groups[0]["lr"])
                print("learning rate : ", learning_rates)
                train_loss_history.append(train_loss)
                print("train loss : ", train_loss)
                write_checkpoint(i_epoch)
            test_loss = test(i_epoch)
            if args.ReduceLROnPlateau and i_epoch > nepoch_factor:
                scheduler.step(test_loss)
            test_loss_history.append(test_loss)
            if test_loss < min_loss:
                min_loss = test_loss
    finally:
        if distributed:
            cleanup()

def main():
    print(sys.argv)

    #print("Parsing arguments")
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
    parser.add_argument('-i-tune', '--inputdir-tune', type=str, help='Specify input directory for training (option)')                   ## not using now
    parser.add_argument('-ii-tune', '--inputdir-validate-tune', type=str, help='Specify input directory for validating')                ## not using now
    parser.add_argument('--learning-rate', type=float, default=9.0e-6)                                                                  ## not using now
    parser.add_argument('--weight-decay', type=float, default=1e-4)                                                                     ## not using now
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
    parser.add_argument('--use-multihead-model', action='store_true', help='Use GravNetModelMultiHead instead of legacy GravNet models')
    parser.add_argument('--multihead-regression-heads', type=int, default=1, help='Number of regression heads for multi-head model (head-0 is clustering)')
    parser.add_argument('--multihead-interaction-start-epoch', type=int, default=5, help='Epoch to enable clustering->regression interaction in multi-head model')
    parser.add_argument('--multihead-interaction-mode', type=str, default='concat', choices=['none', 'concat', 'add', 'gate'], help='Interaction mode for multi-head model')
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

    args = parser.parse_args()
    if args.verbose:
        oc.DEBUG = True


    if args.ddp:
        # GPU選択: --gpus で指定されたGPUのみを使用。未指定の場合は全GPUを使用
        if args.gpus is not None:
            visible_gpus = [int(x.strip()) for x in args.gpus.split(',') if x.strip()]
            if not visible_gpus:
                print("Error: --gpus must specify at least one GPU (e.g., --gpus 0,1)")
                sys.exit(1)
            # 指定されたGPUが存在するか検証
            n_gpus = torch.cuda.device_count()
            invalid = [g for g in visible_gpus if g < 0 or g >= n_gpus]
            if invalid:
                print(f"Error: Invalid GPU id(s) {invalid}. Available GPUs: 0-{n_gpus-1}")
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
        mp.spawn(run_training, args=(world_size, args), nprocs=world_size, join=True)

        sys.exit()
    
    run_training(0, 1, args)
    return


def colorlabel(y,label):
    unique_label=np.unique(label)
    if y == unique_label[0] :
        return "b"
    elif y == unique_label[1] :return "g"

def check_plots(coords_list,data_y_list):
    #coords_lists has the diferent numbers of elements for each row, so it cannot be converted to numpy!!!!!!!!!!!!!!!!!!!!!
    coords_list=np.array(coords_list[0])
    label=np.array(data_y_list[0][0:4000])
    fig,ax = plt.subplots(figsize = (8,6))
    l = 0
    for x1,y1,label1 in zip(coords_list[0:4000,0],coords_list[0:4000,1],label):
        ax.scatter(x1, y1,c=colorlabel(label1,label))
    plt.show()

# def coord_tsne(Coords,Tag):
#     tsne = TSNE(n_components=2,random_state=41,learning_rate='auto')
#     Coord_reduced = tsne.fit_transform(Coords)

#     plt.figure(figsize=(13,7))
#     plt.scatter(Coord_reduced[0:4000,0],Coord_reduced[0:4000,1],c=Tag,cmap='jet',s=15,alpha=0.5)
#     #plt.axis('off')
#     plt.colorbar()
#     plt.show()

def debug():
    oc.DEBUG = True
    dataset = TauDataset('data/taus')
    dataset.npzs = [
        # 'data/taus/49_nanoML_84.npz',
        # 'data/taus/37_nanoML_4.npz',
        #'data/taus/26_nanoML_93.npz',
        # 'data/taus/142_nanoML_75.npz',
        ]
    for data in DataLoader(dataset, batch_size=len(dataset), shuffle=False): break
    print(data.y.sum())
    model = GravnetModel(input_dim=9, output_dim=4)
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.batch)
    pred_betas = torch.sigmoid(out[:,0])
    pred_cluster_space_coords = out[:,1:4]
    out_oc = oc.calc_LV_Lbeta_Eregression(
        pred_betas,
        pred_cluster_space_coords,
        data.y.long(),
        data.batch.long()
    )

def plot_history(train_loss_history,test_loss_history):
    loss_type = type(test_loss_history)
    if(loss_type is list):
        plt.figure(figsize=(8,6))
        plt.plot(test_loss_history,label='test_loss', lw=3, c='b')
        plt.plot(train_loss_history,label='train_loss',lw=3,c='green')
        plt.title('loss function')
        plt.legend(fontsize=14)
        plt.show()

def plot_acc_history(train_acc_history,test_acc_history):
    loss_type = type(test_acc_history)
    if(loss_type is list):
        plt.figure(figsize=(8,6))
        plt.plot(test_acc_history,label='test_acc', lw=3, c='b')
        plt.plot(train_acc_history,label='train_acc',lw=3,c='green')
        plt.title('accuracy')
        plt.legend(fontsize=14)
        plt.show()

def run_profile():
    from torch.profiler import profile, record_function, ProfilerActivity
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device', device)

    batch_size = 2
    n_batches = 2
    shuffle = True
    dataset = TauDataset('data/taus')
    dataset.npzs = dataset.npzs[:batch_size*n_batches]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    print(f'Running profiling for {len(dataset)} events, batch_size={batch_size}, {len(loader)} batches')

    model = GravnetModel(input_dim=9, output_dim=8).to(device)
    epoch_size = len(loader.dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7, weight_decay=1e-4)

    print('Start limited training loop')
    model.train()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("model_inference"):
            pbar = tqdm.tqdm(loader, total=len(loader))
            pbar.set_postfix({'loss': '?'})
            for i, data in enumerate(pbar):
                data = data.to(device)
                optimizer.zero_grad()
                result = model(data.x, data.batch)
                if args.jit:
                    # loss = loss_fn_jit(result, data, use_charge_track_likeness=args.use_charged_cluster_loss)
                    raise
                else:
                    loss = loss_fn(result, data, use_charge_track_likeness=args.use_charged_cluster_loss)
                print(f'loss={float(loss)}')
                loss.backward()
                if not args.no_clipping:
                    utils.clip_grad_value_(model.parameters(), clip_value=args.clip_value)
                optimizer.step()
                pbar.set_postfix({'loss': float(loss)})
    print(prof.key_averages().table(sort_by="cpu_time", row_limit=10))
    # Other valid keys:
    # cpu_time, cuda_time, cpu_time_total, cuda_time_total, cpu_memory_usage,
    # cuda_memory_usage, self_cpu_memory_usage, self_cuda_memory_usage, count

if __name__ == '__main__':
    pass
    main()
    # debug()
    # run_profile()
