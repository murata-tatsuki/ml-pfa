import os, os.path as osp
from time import strftime
import tqdm
import torch
from torch_geometric.loader import DataLoader
import argparse
import matplotlib.pylab as plt
import numpy as np

#from sklearn.metrics import accuracy_score
#import torch_cmspepr.objectcondensation as oc
# import objectcondensation as oc
import objectcondensation as oc
#import torch.nn.functional as f

from gravnet_model import GravnetModel,GravNetModelBranch,GravnetModelWithNoiseFilter
from dataset import ILCDataset
from lrscheduler import CyclicLRWithRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau
#from sklearn.manifold import TSNE
from model import get_model, get_model_branch
from lcr_module import LCR, LCR_withPID, LCR_withClass, hungarian_set_loss, hungarian_set_loss_bbox_only, hungarian_set_loss_new, hungarian_set_loss_new_sub

#from ReadText import ReadText
import sys

# for distributed data parallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# clipping
import torch.nn.utils as utils

from torch.nn.utils.rnn import pad_sequence

#torch.manual_seed(1009)
torch.autograd.set_detect_anomaly(True)



def index_setup(args):
    output_dimension = args.output_dimension
    index_pred_tracker_energy = 0
    index_pred_cluster_energy = 0
    index_pred_cluster_space_coords = 0
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


def pad_and_mask_batch(tensor, batch):
    """
    Args:
        tensor: (sum BM, D) - GNNの出力（embed, featなど）
        batch:  (sum BM,)   - 各行が属するイベントID
    Returns:
        padded_tensor: (B, N_max, D)
        mask:          (B, N_max) - 1 for valid, 0 for padded
    """
    device = tensor.device
    batch_ids = batch.unique(sorted=True)

    # イベント単位にテンソルを分割（List[(M_i, D)])
    split_tensors = [tensor[batch == b_id] for b_id in batch_ids]
    
    # パディングして (B, N_max, D)
    padded_tensor = pad_sequence(split_tensors, batch_first=True)  # padding = 0 by default

    # mask 生成: True where data is valid
    lengths = [t.size(0) for t in split_tensors]
    max_len = padded_tensor.size(1)
    mask = torch.zeros(len(lengths), max_len, dtype=torch.bool, device=device)
    for i, l in enumerate(lengths):
        mask[i, :l] = True

    return padded_tensor, mask

def split_by_batch(tensor, batch):
    """
    Args:
        tensor: (N, D) - GNNの出力（embed, featなど）
        batch:  (N,)   - 各行が属するイベントID
    Returns:
        split_tensors: List[Tensor] - イベントごとのテンソル [(M_0, D), (M_1, D), ..., (M_B-1, D)]
    """
    batch_ids = batch.unique(sorted=True)
    split_tensors = [tensor[batch == b_id] for b_id in batch_ids]
    return split_tensors

def pdg_id_to_class(pdg_ids):
    """
    pdg_ids: torch.Tensor or np.ndarray, shape (N,)
    Return: torch.Tensor, shape (N,) – class indices (0~9)
    """
    abs_ids = torch.abs(pdg_ids)
    cls = torch.full_like(pdg_ids, fill_value=9)  # default: other neutral hadron (ID=10)

    # Explicit mappings
    cls[pdg_ids == 22] = 0      # photon
    cls[abs_ids == 11] = 1      # electron / positron
    cls[abs_ids == 13] = 2      # muon
    cls[abs_ids == 211] = 3     # charged pion
    cls[abs_ids == 111] = 4     # neutral pion
    cls[abs_ids == 321] = 5     # charged kaon
    cls[abs_ids == 130] = 6     # K_L0
    cls[abs_ids == 310] = 6     # K_S0
    cls[abs_ids == 2212] = 7    # proton
    cls[abs_ids == 2112] = 8    # neutron

    return cls


def formatted_loss_components_string(components: dict, valid: str="") -> str:
    """
    Formats the components returned by calc_LV_Lbeta
    """
    total_loss = components['loss_E']+components['loss_Mag']+components['loss_Dir']
    fractions = { k : v/total_loss for k, v in components.items() }
    fkey = lambda key: f'{components[key]:+.4f} ({100.*fractions[key]:.1f}%)'
    s = (
        '  {valid} loss_E                   = {loss_E}'
        '\n  {valid} loss_Mag                 = {loss_Mag}'
        '\n  {valid} loss_Dir                 = {loss_Dir}'
        .format(L=total_loss,**{k : fkey(k) for k in components})
        )
    return s




def main():
    print(sys.argv)

    #print("Parsing arguments")
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dry', action='store_true', help='Turn off checkpoint saving and run limited number of events')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print more output')
    parser.add_argument('--settings-Sep01', action='store_true', help='Use 21Sep01 settings')
    parser.add_argument('--reduce-noise', action='store_true', help='Randomly kills 95% of noise')
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
    parser.add_argument('--output-dimension', type=int, default=3, help='Specify total output dimension (note that 1 dim each is used for beta and charged cluster loss)')
    parser.add_argument('-i', '--inputdir', type=str, required=True, help='Specify input directory for training (required)')
    parser.add_argument('--no-split', action='store_true', help='Do not split sample into training/validating')
    parser.add_argument('-ii', '--inputdir-validate', type=str, help='Specify input directory for validating')
    parser.add_argument('-i-tune', '--inputdir-tune', type=str, help='Specify input directory for training (option)')                   ## not using now
    parser.add_argument('-ii-tune', '--inputdir-validate-tune', type=str, help='Specify input directory for validating')                ## not using now
    parser.add_argument('--learning-rate', type=float, default=9.0e-6)                                                                  ## not using now
    parser.add_argument('--weight-decay', type=float, default=1e-4)                                                                     ## not using now
    parser.add_argument('--energy-regression', action='store_true', help='Turn on energy regression term on loss function and output')
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
    parser.add_argument('--gnn-model-ckpt', type=str, default='', help='Specify trained gnn model parameters')
    parser.add_argument('--ReduceLROnPlateau', action='store_true', help='Use ReduceLROnPlateau scheduler')
    parser.add_argument('--qmin', type=float, default=1., help='')
    parser.add_argument('--min-lr', type=float, default=1e-7, help='')
    parser.add_argument('--dp', action='store_true', help='Use dataparallel')
    parser.add_argument('--ddp', action='store_true', help='Use distributed dataparallel')
    parser.add_argument('--gpus', type=str, default=None, help="Comma-separated list of GPU ids to use (e.g., '0,1')")
    parser.add_argument('--lr-policy', type=str, default='cosine', help='Specify lraning rate policy at lrscheduler.py')
    parser.add_argument('--nrestart-cosreduce', type=int, default=3, help='number of restart without reducing the maximum learning rate')
    parser.add_argument('--clip-value', type=int, default=100, help='threshold of gradient clipping')
    parser.add_argument('--no-clipping', action='store_true', help='do not clip the gradients')           
    parser.add_argument('--classification', action='store_true', help='turn on claasification in LCR')           
    parser.add_argument('--pid', action='store_true', help='turn on pid in LCR')           
    parser.add_argument('--loss-specify', action='store_true', help='turn on pid in LCR')           

    args = parser.parse_args()
    if args.verbose: oc.DEBUG = True
    reduce_noise = args.reduce_noise
    n_epochs = args.epochs
    batch_size = args.batch_size
    output_dimension = args.output_dimension
    output_dim = output_dimension
    lr_input = args.learning_rate
    weight_decay_input = args.weight_decay
    qmin = args.qmin
    min_lr=args.min_lr

    out_classes = 10 if args.pid else 7

    device = torch.device(args.cuda) if not args.dp else 'cuda'
    print('Using device: ', device)
    if not args.dp: torch.cuda.set_device(device)
    if args.dp:
        print("available number of cuda ", torch.cuda.device_count())
        # batch_size = batch_size * torch.cuda.device_count()
        # lr_input = lr_input * torch.cuda.device_count()
        batch_size = batch_size * 2
        lr_input = lr_input * 2
    print("learning rate :", lr_input, ",  weght decay :", weight_decay_input)






    debug = False
    # event_energy=args.event_total_energy
    energyRegression=args.energy_regression
    energyRegressionCluster=args.energy_regression_cluster
    momentum=args.momentum
    momentumAmp=args.momentum_amp
    mctpe=args.mctpe
    energy_branch=args.energy_branch
    

    # thetaphi = True if input_dim == 7 else False
    output_dimension, index_pred_tracker_energy, index_pred_cluster_energy, index_pred_cluster_space_coords, additional_input_dimension = index_setup(args)

    input_dim = 5+args.thetaphi*2+additional_input_dimension
    print(f"Loading model from checkpoint {args.gnn_model_ckpt}")
    gnn_model = get_model(args.gnn_model_ckpt, jit=False, input_dim=input_dim,output_dim=output_dimension).to(device)
    gnn_model.eval()
    for param in gnn_model.parameters():
        param.requires_grad = False


    shuffle = True

    print(f'thetaphi at main: {args.thetaphi}')
    print("Loading dataset...")
    train_dataset = ILCDataset(args.inputdir,timingCut=args.timing_cut,thetaphi=args.thetaphi,test_mode=True,momentum=args.momentum,momentumAmp=args.momentum_amp,mctpe=args.mctpe)
    test_dataset = ILCDataset(args.inputdir_validate,timingCut=args.timing_cut,thetaphi=args.thetaphi,test_mode=True,momentum=args.momentum,momentumAmp=args.momentum_amp,mctpe=args.mctpe)
    
    print(f"Training dataset size:  {len(train_dataset)}")
    print(f"Validating dataset size:  {len(test_dataset)}")
    print(f"Batch size:  {batch_size}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=16, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=16, pin_memory=True)

    """
    ak_feat: edep, x, y, z, time, track, charge, px, py, pz (atcalo)
        --> save edep, drop others
    ak_label: hitid, mcid, pdg, charge, mass, px, py, pz (of mcp), status
        --> save all labels
    """
    






    print(f"Loading lcr model")
    # lcr_model = LCR(embed_dim_=4,embed_dim=128, num_heads=8, K=256, n_classes=7, feat_dim=5).to(device)
    if args.classification: lcr_model = LCR_withClass(embed_dim_=4,embed_dim=128, num_heads=8, K=256, n_classes=out_classes, feat_dim=4).to(device)
    if args.pid: lcr_model = LCR_withPID(embed_dim_=4,embed_dim=128, num_heads=8, K=256, n_classes=out_classes, feat_dim=4).to(device)
    else : lcr_model = LCR(embed_dim_=4,embed_dim=128, num_heads=8, K=256, feat_dim=4).to(device)
    # lcr_model = LCR(embed_dim=128, num_heads=8, K=256, n_classes=7, feat_dim=5).to(device)
    
    # optimizer = torch.optim.AdamW(lcr_model.parameters(), lr=2e-4, weight_decay=1e-2)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=4e-4, total_steps=N_steps)
    epoch_size = len(train_loader.dataset)
    optimizer = torch.optim.AdamW(lcr_model.parameters(), lr=lr_input, weight_decay=weight_decay_input)
    scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=args.restart_period, t_mult=1.1, policy=args.lr_policy, min_lr=min_lr, nrestart_cosreduce=args.nrestart_cosreduce)


    def get_gnn_output(batched_data):
        with torch.no_grad():
            gnn_outputs: torch.Tensor = gnn_model(batched_data.x, batched_data.batch)
            
            pred_betas = torch.sigmoid(gnn_outputs[:,0])
            if args.energy_regression:
                if not args.energy_regression_cluster:
                    if args.use_charged_cluster_loss:
                        pred_charge_track_likeness = torch.sigmoid(gnn_outputs[:,1])
                        pred_tracker_energy = gnn_outputs[:,2]
                        pred_cluster_energy = None
                        pred_cluster_space_coords = gnn_outputs[:,3:]
                        assert(pred_charge_track_likeness.device == device)
                    else:
                        pred_charge_track_likeness = None
                        pred_tracker_energy = gnn_outputs[:,1]
                        pred_cluster_energy = None
                        pred_cluster_space_coords = gnn_outputs[:,2:]
                else:
                    if args.use_charged_cluster_loss:
                        pred_charge_track_likeness = torch.sigmoid(gnn_outputs[:,1])
                        pred_tracker_energy = gnn_outputs[:,2]
                        pred_cluster_energy = gnn_outputs[:,3]
                        pred_cluster_space_coords = gnn_outputs[:,4:]
                        assert(pred_charge_track_likeness.device == device)
                    else:
                        pred_charge_track_likeness = None
                        pred_tracker_energy = gnn_outputs[:,1]
                        pred_cluster_energy = gnn_outputs[:,2]
                        pred_cluster_space_coords = gnn_outputs[:,3:]
            else:
                pred_tracker_energy = None
                pred_cluster_energy = None
                if args.use_charged_cluster_loss:
                    pred_charge_track_likeness = torch.sigmoid(gnn_outputs[:,1])
                    pred_cluster_space_coords = gnn_outputs[:,2:]
                    assert(pred_charge_track_likeness.device == device)
                else:
                    pred_charge_track_likeness = None
                    pred_cluster_space_coords = gnn_outputs[:,1:]
            cluster_track_index = batched_data.y[:,1]

            assert all(t.device == device for t in [pred_betas, pred_cluster_space_coords, batched_data.y, batched_data.batch,])
            true_energy = torch.sqrt(torch.sum(torch.square(batched_data.label[:,4:8]), 1))
            detected_energy = batched_data.feat[:,0]

        return pred_cluster_space_coords, pred_betas
    
    def train(epoch):
        print('Training epoch', epoch)
        lcr_model.train()
        N_train = len(train_loader)
        loss_components={}
        train_loss = 0.0
        gradients=[]
        def update(components):
            for key, value in components.items():
                if not key in loss_components: loss_components[key] = 0.
                loss_components[key] += value
        if not args.ReduceLROnPlateau: scheduler.step()
        try:
            pbar = tqdm.tqdm(train_loader, total=len(train_loader))
            pbar.set_postfix({'loss': '?'})
            for i, data in enumerate(pbar):
                # print(i, data.x.shape, data.y.shape)
                data = data.to(device)
                optimizer.zero_grad()
                pred_cluster_space_coords, pred_betas = get_gnn_output(data)
                hit_embed, hit_mask = pad_and_mask_batch(pred_cluster_space_coords, data.batch)
                hit_beta, _         = pad_and_mask_batch(pred_betas.unsqueeze(-1), data.batch)
                hit_beta            = hit_beta.squeeze(-1)  # 元のshapeに戻す
                hit_feat, _         = pad_and_mask_batch(data.feat, data.batch)

                if args.classification or args.pid: pred_fourvec, pred_cls, attn_w = lcr_model(hit_embed, hit_beta, hit_feat, hit_mask=hit_mask)
                else: pred_fourvec, attn_w = lcr_model(hit_embed, hit_beta, hit_feat, hit_mask=hit_mask)
                if args.classification:
                    unique_label = split_by_batch(data.label, data.batch)
                    unique_label = [torch.unique(t[:,1], dim=0) for t in unique_label]
                if args.pid:
                    unique_label = split_by_batch(data.label, data.batch)
                    unique_label = [torch.unique(t[:,1:3], dim=0) for t in unique_label]
                    unique_label = [pdg_id_to_class(t[:,1]) for t in unique_label]
                true_energy = torch.sqrt(torch.sum(torch.square(data.label[:,4:8]), 1))
                # truth_four_vector = torch.cat((data.label[:,5:8], true_energy.reshape(true_energy.shape[0],1)), 1)
                truth_four_vector = torch.cat((true_energy.reshape(true_energy.shape[0],1), data.label[:,5:8]), 1)
                truth_four_vector = split_by_batch(truth_four_vector, data.batch)
                truth_four_vector = [torch.unique(t, dim=0) for t in truth_four_vector]

                # print(pred_fourvec, pred_cls, truth_four_vector, unique_label)
                # print(pred_cls.shape)

                if args.classification or args.pid: loss = hungarian_set_loss(pred_fourvec, pred_cls, truth_four_vector, unique_label)
                elif args.loss_specify: 
                    loss, components = hungarian_set_loss_new_sub(pred_fourvec, truth_four_vector)
                    update(components)
                else: loss = hungarian_set_loss_new(pred_fourvec, truth_four_vector)
                # update(components)
                train_loss += loss
                loss.backward()

                if not args.no_clipping:
                    utils.clip_grad_value_(lcr_model.parameters(), clip_value=args.clip_value)
                optimizer.step()
                if not args.ReduceLROnPlateau: scheduler.batch_step()
                pbar.set_postfix({'loss': float(loss)})
                gradients.append([p.grad.norm().item() for p in lcr_model.parameters()])
            # Divide by number of entries
            layer_grads = np.mean(np.array(gradients), axis=0)
            print(layer_grads)
            if args.loss_specify: 
                for key in loss_components:
                    loss_components[key] /= N_train
            train_loss = train_loss.item() / N_train
            print('train                     = ', train_loss)
            if args.loss_specify: print(formatted_loss_components_string(loss_components))
            return train_loss
        except Exception:
            print('Exception encountered:', data, 'i:', i)
            raise

    def test(epoch):
        N_test = len(test_loader)
        loss_components = {}
        test_acc=0.
        test_loss=0.
        def update(components):
            for key, value in components.items():
                if not key in loss_components: loss_components[key] = 0.
                loss_components[key] += value
        with torch.no_grad():

            lcr_model.eval()
            for data in tqdm.tqdm(test_loader, total=len(test_loader)):
                data = data.to(device)
                pred_cluster_space_coords, pred_betas = get_gnn_output(data)
                hit_embed, hit_mask = pad_and_mask_batch(pred_cluster_space_coords, data.batch)
                hit_beta, _         = pad_and_mask_batch(pred_betas.unsqueeze(-1), data.batch)
                hit_beta            = hit_beta.squeeze(-1)  # 元のshapeに戻す
                hit_feat, _         = pad_and_mask_batch(data.feat, data.batch)
                if args.classification or args.pid: pred_fourvec, pred_cls, attn_w = lcr_model(hit_embed, hit_beta, hit_feat, hit_mask=hit_mask)
                else: pred_fourvec, attn_w = lcr_model(hit_embed, hit_beta, hit_feat, hit_mask=hit_mask)
                if args.classification:
                    unique_label = split_by_batch(data.label, data.batch)
                    unique_label = [torch.unique(t[:,1], dim=0) for t in unique_label]
                if args.pid:
                    unique_label = split_by_batch(data.label, data.batch)
                    unique_label = [torch.unique(t[:,1:3], dim=0) for t in unique_label]
                    unique_label = [pdg_id_to_class(t[:,1]) for t in unique_label]
                true_energy = torch.sqrt(torch.sum(torch.square(data.label[:,4:8]), 1))
                truth_four_vector = torch.cat((data.label[:,5:8], true_energy.reshape(true_energy.shape[0],1)), 1)
                truth_four_vector = split_by_batch(truth_four_vector, data.batch)
                truth_four_vector = [torch.unique(t, dim=0) for t in truth_four_vector]
                if args.classification or args.pid: loss = hungarian_set_loss(pred_fourvec, pred_cls, truth_four_vector, unique_label)
                elif args.loss_specify: 
                    loss, components = hungarian_set_loss_new_sub(pred_fourvec, truth_four_vector)
                    update(components)
                else: loss = hungarian_set_loss_new(pred_fourvec, truth_four_vector)
                test_loss += loss
                # update(loss_fn(result, data, i_epoch=epoch, return_components=True, use_charge_track_likeness=args.use_charged_cluster_loss))
        # Divide by number of entries
        if args.loss_specify: 
            for key in loss_components:
                loss_components[key] /= N_test
        # Compute total loss and do printout
        if args.loss_specify: print(formatted_loss_components_string(loss_components, valid="test"))
        # # test_loss = loss_offset + loss_components['L_V']+loss_components['L_beta']
        # test_loss = loss_offset + loss_components['L_V']+loss_components['L_beta']+loss_components['L_E'] if 'L_E' in loss_components else loss_offset + loss_components['L_V']+loss_components['L_beta']
        test_loss = test_loss.item() / N_test
        print(f'Returning {test_loss}')
        return test_loss

    ckpt_dir = strftime('checkpoint/ckpts_gravnet_new02_%b%d_%H%M') if args.ckptdir is None else args.ckptdir
    def write_checkpoint(checkpoint_number=None, best=False):
        ckpt = 'ckpt_best.pth.tar' if best else 'ckpt_{0}_1.pth.tar'.format(checkpoint_number)
        ckpt = osp.join(ckpt_dir, ckpt)
        if best: print('Saving epoch {0} as new best'.format(checkpoint_number))
        if not args.dry:
            os.makedirs(ckpt_dir, exist_ok=True)
            # m = torch.jit.script(model)
            #torch.jit.save(m,ckpt)
            torch.save(dict(model=lcr_model.state_dict()), ckpt)

    min_loss = 1e9
    train_loss_history=[]
    test_loss_history=[]
    epoch_history=[]
    train_acc_history=[]
    test_acc_history=[]
    learning_rates=[]

    for i_epoch in range(n_epochs):
        train_loss=train(i_epoch)
        learning_rates.append(optimizer.param_groups[0]["lr"])
        print("learning rate : ", learning_rates)
        train_loss_history.append(train_loss)
        print("train loss : ", train_loss)
        write_checkpoint(i_epoch)

        test_loss= test(i_epoch)
        # if args.ReduceLROnPlateau:
        #     if i_epoch > nepoch_factor: scheduler.step(test_loss)
        #test_loss/=len(test_loader)
        # test_loss_history.append(test_loss)
        if test_loss < min_loss:
            min_loss = test_loss
            #write_checkpoint(i_epoch, best=True)

        #if i_epoch==0 or i_epoch==30 : check_plots(cluster_space_para,data_y)
        #if i_epoch==30 : check_plots(cluster_space_para,data_y)






if __name__ == '__main__':
    pass
    main()
    # debug()
    # run_profile()
