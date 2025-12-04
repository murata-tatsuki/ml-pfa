import sys
import numpy as np
from distutils.util import strtobool
#import evaluation_noNoise as ev
#import event_view_plot as plt_3d
import awkward as ak
import tools.load_awkward as la
from dataset import ILCDataset
from test_yielder import TestYielder, TestYielder_transformer_Like_Clustering
from model import get_model, get_clustering_model
from matching import get_energy_ABCD,get_mask_charged_neutral,matching_1to1
import argparse
from torch_scatter import scatter_max, scatter_add, scatter_mean
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from train_clustering_ddp import feat_format, query_construction, pad_and_mask_batch


def calc_energy_prediction(prediction, pattern_cluster, pandora=False, energyRegression=False, energyRegressionCluster=False):
    if not pandora:
        predicted_beta = prediction.pred_betas[pattern_cluster]
        predicted_energy = prediction.pred_tracker_energy[pattern_cluster] if energyRegression else -np.zeros(1)
        predicted_energy = predicted_energy[np.argsort(-predicted_beta)] if energyRegression else -np.zeros(1)
        predicted_energy_cluster = prediction.pred_cluster_energy[pattern_cluster] if energyRegressionCluster else -np.ones(1)
        # print(pattern_cluster, )
        # match_track = match_track[pattern_cluster]
        # cond_tracknesses = match_track[np.argsort(-predicted_beta)]
        # cond_trackness = cond_tracknesses[0]
        cond_trackness = 0
        predicted_beta = -np.sort(-predicted_beta)
        # print(predicted_beta[0], cond_trackness)
    else:
        predicted_energy = prediction.pred_tracker_energy[pattern_cluster]
        cond_trackness = 0
        predicted_beta = np.zeros(1)
    
    pred_edep = predicted_energy[0]                                  ## alpha
    pred_edep_cluster = np.sum(predicted_energy_cluster) if not pandora else -1

    return pred_edep, pred_edep_cluster


def assign_hits_to_queries(query, hits, attn_mask=None):
    """
    Cross Attentionを用いて、各hitがどのqueryに最も強く関連しているかを特定する関数。
    
    Args:
        query: Tensor, shape = (B, N_query, D)
        hits: Tensor, shape = (B, N_hit, D)
        attn_mask: Optional mask tensor, shape = (B, N_query, N_hit)
                   (例: padding部などを除外する際に利用)
                   
    Returns:
        hit_to_query: Tensor, shape = (B, N_hit)
                      各hitが最も強く対応するqueryのindexを格納
        attn_weights: Tensor, shape = (B, N_query, N_hit)
                      softmax後のattention重み
    """
    B, N_query, D = query.shape
    _, N_hit, _ = hits.shape

    # --- Step 1: Attentionスコアの計算 (queryとhit間の類似度)
    # QK^T / sqrt(d)
    scale = D ** 0.5
    attn_scores = torch.bmm(query, hits.transpose(1, 2)) / scale  # (B, N_query, N_hit)
    
    # --- Step 2: マスクの適用（必要なら）
    if attn_mask is not None:
        attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
    
    # --- Step 3: softmaxで正規化（各queryごとにhit方向で確率化）
    attn_weights = F.softmax(attn_scores, dim=-1)  # (B, N_query, N_hit)
    
    # --- Step 4: 転置して、各hitに対して「どのqueryから最も重視されたか」を求める
    # (B, N_hit, N_query) にして argmax で最大のquery indexを得る
    attn_hit_view = attn_weights.transpose(1, 2)  # (B, N_hit, N_query)
    hit_to_query = torch.argmax(attn_hit_view, dim=-1)  # (B, N_hit)
    
    return hit_to_query, attn_weights

def match_hits_to_queries(attn_weights):
    """
    すでに計算済みの attention weight (B, N_query, N_hit) を用いて、
    各 hit が最も強く結びついている query の index を返す関数。

    Args:
        attn_weights: Tensor, shape = (B, N_query, N_hit)
                      query→hit の attention 確率（softmax後）

    Returns:
        hit_to_query: Tensor, shape = (B, N_hit)
                      各 hit の最大 attention を持つ query index
    """
    # --- Step 1 ---
    # attention map を転置して、(B, N_hit, N_query にする)
    # こうすることで hit ごとの query 比較が簡単にできる
    attn_hit_view = attn_weights.transpose(1, 2)  # (B, N_hit, N_query)

    # --- Step 2 ---
    # 各 hit について attention が最大の query を選ぶ
    # dim=-1 は query 次元に沿って argmax を取る
    hit_to_query = torch.argmax(attn_hit_view, dim=-1)  # (B, N_hit)

    return hit_to_query



# def save_pred(datapath, ckpt, outfile, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, use_charge_track_likeness=False, pandora=False):
def save_pred(datapath, ckpt_gnn, ckpt_clustering, outfile, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, args={}):
    pandora=args.pandora
    energyRegression=args.energy_regression
    energyRegressionCluster=args.energy_regression_cluster
    energyRegressionWeight=args.energy_regression_weight
    momentum=args.momentum
    momentumAmp=args.momentum_amp
    mctpe=args.mctpe
    energy_branch=args.energy_branch
    device=args.device
    use_charge_track_likeness=False
    if 'cuda' in device: torch.cuda.set_device(device)
    assert(not (args.beta_d_scan and (".root" in outfile)))

    thetaphi = True if input_dim == 7 else False
    if momentum:
        input_dim += 3 
        if momentumAmp:
            input_dim += 1
    if energyRegression:
        output_dim += 1
        if energyRegressionCluster:
            output_dim += 1
    if energyRegressionWeight:
            output_dim += 5
    print(f"Loading GravNet model from checkpoint {ckpt_gnn}")
    if energy_branch:
        model_gnn = get_model_branch(ckpt_gnn, jit=False, input_dim=input_dim,output_dim=output_dim).to(device)
    else:
        model_gnn = get_model(ckpt_gnn, jit=False, input_dim=input_dim,output_dim=output_dim).to(device)
    model_clustering = get_clustering_model(ckpt_clustering, jit=False, input_dim=input_dim,output_dim=output_dim, lcr_block=args.lcr_block, pid=args.pid, ddp=args.ddp)
    print(f"Loading data from {datapath} with {nstart=}, {nend=}, {timingCut=}")
    dataset = ILCDataset(datapath, timingCut=timingCut, thetaphi=thetaphi, test_mode=True, nstart=nstart, nend=nend, pandora=pandora,momentum=momentum,momentumAmp=momentumAmp)
    yielder = TestYielder_transformer_Like_Clustering(model=model_gnn, dataset=dataset, device=device, pandora=pandora)
    # dataset = ILCDataset(datapath, timingCut=timingCut, thetaphi=thetaphi, test_mode=True, nstart=nstart, nend=nend, pandora=pandora)
    # yielder = TestYielder(model=model, dataset=dataset)

    nmax = None if nend==-1 else nend-nstart+1

    x = []
    y = []
    feats = []
    labels = []
    preds = []
    trans_preds = []
    pands = []
    energy = []

    def get_gnn_output_allFeat(batched_data):
        with torch.no_grad():
            gnn_outputs: torch.Tensor = model_gnn(batched_data.x, batched_data.batch)
        return gnn_outputs, torch.sigmoid(gnn_outputs[:,0])

    #for i, (event, prediction) in enumerate(yielder.iter_pred(nmax)):
    # for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta=0.2, td=0.5, nmax=nmax, pandora=pandora)):
    for i, (event, data, prediction, clustering, matches, condensation_points) in enumerate(yielder.iter_matches(tbeta=0.9, td=0.5, nmax=nmax, energyRegression=energyRegression, energyRegressionCluster=energyRegressionCluster)):

        if i == nmax: break

        if i < 10 or i%100 == 0:
            print("Event", i, "processing...")
        
        matches12, matches21 = matches
        all_truth_ids = list(set(np.unique(event.y[:,0])))
        all_mcid = list(set(np.unique(event.label[:,1]).astype(np.int32)))
        all_cluster_ids = list(set(np.unique(clustering)))
        assert( len(all_truth_ids) == len(all_mcid) )
        assert( len(event.y[:,0]) == len(event.label[:,1]) )
        n_hits = len(event.label[:,1])

        matched_reco_clusterIds = matching_1to1(event, clustering, matches12)
        print(matches12)
        print(matched_reco_clusterIds)
        dict_energy = {}
        for id in all_truth_ids:
            pattern_mcid = (event.y[:,0]==id)
            match_label = event.label[pattern_mcid] 
            
            if (id in matched_reco_clusterIds.keys()):
                reco_match = matched_reco_clusterIds[id]
                for rid in reco_match:
                    pattern_cluster = (clustering==rid)
                    
                    pred_edep, pred_edep_cluster = calc_energy_prediction(prediction, pattern_cluster, pandora,energyRegression,energyRegressionCluster)

                    my_label = match_label[0]
                    mcen = np.sqrt(my_label[4]**2 + my_label[5]**2 + my_label[6]**2 + my_label[7]**2)
                    dict_energy[rid] = {"truth_cluster_id":id, "reco_cluster_id":rid, "pred_edep":pred_edep, "pred_edep_cluster":pred_edep_cluster, "mcen":mcen}
            # for MC particle, take any element from the match because they should be the same

        for rid in all_cluster_ids:
            if (rid not in dict_energy.keys()):
                pattern_cluster = (clustering==rid)
                pred_edep, pred_edep_cluster = calc_energy_prediction(prediction, pattern_cluster, pandora,energyRegression,energyRegressionCluster)
                dict_energy[rid] = {"truth_cluster_id":-1, "reco_cluster_id":rid, "pred_edep":pred_edep, "pred_edep_cluster":pred_edep_cluster, "mcen":-1}

        np_energy = np.zeros((event.y.shape[0],3))
        truth_four_vec = event.label[:,4:8]
        truth_four_vec[:,0] = torch.norm(truth_four_vec, dim=-1)
        for i, reco_clusterId in enumerate(clustering):
            if (reco_clusterId in dict_energy.keys()):
                np_energy[i] = np.array([dict_energy[reco_clusterId]["pred_edep"], dict_energy[reco_clusterId]["pred_edep_cluster"], truth_four_vec[i,0]])
            else:
                np_energy[i] = np.array([-1, -1, -1])
        # label_mcen = np.sqrt(event.label[:,4]**2 + event.label[:,5]**2 + event.label[:,6]**2 + event.label[:,7]**2)
        # truth_ids = event.y[:,0]
        # print(np.array(label_mcen).shape, np.array(truth_ids).shape)
        # out = scatter_mean(np.array(label_mcen), np.array(truth_ids))
        # print(out)
        print(np.unique(np_energy[:,-1]))
        # np_energy[:,-1] = label_mcen
        
        ak_x = ak.from_numpy(event.x)
        ak_y = ak.from_numpy(event.y)
        ak_feat = ak.from_numpy(event.feat.numpy())
        ak_label = ak.from_numpy(event.label.numpy())

        beta = np.expand_dims(prediction.pred_betas, axis=1)

        clustering_reshape = clustering.reshape(-1,1)

        if use_charge_track_likeness:
            charge_track_likeness = np.expand_dims(prediction.pred_charge_track_likeness, axis=1)
            pred = np.concatenate((beta,clustering_reshape,prediction.pred_cluster_space_coords,charge_track_likeness,np_energy), axis=1)
        else:
            pred = np.concatenate((beta,clustering_reshape,prediction.pred_cluster_space_coords,np_energy), axis=1)
            ###### pandora_clusterについてここと同じような処理をする
            
        ak_pred = ak.from_numpy(pred)
        ak_pand = ak.from_numpy(clustering)

        true_charged_mask, pred_charged_mask = get_mask_charged_neutral(event, clustering, matches)
        eA, eB, eC, eD = get_energy_ABCD(event, true_charged_mask, pred_charged_mask)
        ak_energy = ak.Array([eA,eB,eC,eD])




        ## transformer-like clustering calculation

        # gnn_outputs = model_gnn(data.x.to(device), data.batch.to(device)).to('cpu')
        # hit_features = torch.cat((gnn_outputs, event.feat[:,:-3]), dim=-1)
        # pred_fourvec, particle_prob, particle_cls_logits, attn_w, seed_padding_mask = model_clustering(hit_features.unsqueeze(0))

        gnn_outputs, pred_betas = get_gnn_output_allFeat(data)
        hit_features = feat_format(gnn_outputs, data.feat[:,:-3])
        hit_embed, hit_mask     = pad_and_mask_batch(hit_features, data.batch)
        hit_beta, _             = pad_and_mask_batch(pred_betas.unsqueeze(-1), data.batch)
        hit_beta                = hit_beta.squeeze(-1)  # 元のshapeに戻す
        hit_feat, _             = pad_and_mask_batch(data.feat, data.batch)
        query, seed_padding_mask, query_indices_in_key, seed_track_mask = query_construction(hit_embed, hit_mask=hit_mask)
        if args.lcr_block and args.pid: 
            pred_fourvec, particle_prob, particle_cls_logits, attn_w = model_clustering(hit_embed, query, hit_mask=hit_mask)

        attn_weight = match_hits_to_queries(attn_w[-1]).to('cpu').detach().numpy().copy()
        pred_fourvec_np = pred_fourvec.to('cpu').detach().numpy().copy()

        trans_clustering = attn_weight[0].reshape(-1,1)
        np_trans_energy = np.zeros((event.y.shape[0],6))        # predicted and truth four momentum
        pred_four_vec = pred_fourvec_np[0][attn_weight[0]]

        print(truth_four_vec.shape)
        print(pred_four_vec.shape)

        trans_pred = np.concatenate((trans_clustering, truth_four_vec, pred_four_vec), axis=1)
        print(trans_pred.shape)
        ak_trans_pred = ak.from_numpy(trans_pred)









        x.append(ak_x)
        y.append(ak_y)
        feats.append(ak_feat)
        labels.append(ak_label)
        preds.append(ak_pred)
        trans_preds.append(ak_trans_pred)
        pands.append(ak_pand)
        energy.append(ak_energy)
        
        instx = ak_feat[:,1]
        insty = ak_feat[:,2]
        instz = ak_feat[:,3]
        # for test_p, test_x, test_y, test_z in zip(ak_pand, instx, insty, instz):
        #     print(test_p, test_x, test_y, test_z)

    ak_x = ak.Array(x)
    ak_y = ak.Array(y)
    ak_feats = ak.Array(feats)
    ak_labels = ak.Array(labels)
    ak_preds = ak.Array(preds)
    ak_trans_preds = ak.Array(trans_preds)
    ak_pands = ak.Array(pands)
    ak_energy = ak.Array(energy)
    
    print(f"Saving to {outfile}")
    la.save_awkward(outfile, ak_feats, ak_labels, ak_preds, ak_energy, ak_x, ak_y, ak_pands, ak_trans_preds=ak_trans_preds)
    
def main():
    
    if (len(sys.argv) < 9):
        # print("Usage: save_pred.py datapath ckpt outfile nstart nend timingCut input_dim output_dim use_track_likeness pandora")
        print("Usage: save_pred.py datapath ckpt outfile nstart nend timingCut input_dim output_dim pandora energyRegression momentum momentumAmp")
        return
    

    parser = argparse.ArgumentParser()
    parser.add_argument('datapath')
    parser.add_argument('ckpt_gnn')
    parser.add_argument('ckpt_clustering')
    parser.add_argument('outfile')
    parser.add_argument('nstart', type=int)
    parser.add_argument('nend', type=int)
    parser.add_argument('timingCut')
    parser.add_argument('input_dim', type=int)
    parser.add_argument('output_dim', type=int)
    parser.add_argument('--pandora', action='store_true', help='Use PandoraPFA result')
    parser.add_argument('--energy-regression', action='store_true', help='Turn on energy regression term on loss function and output')
    parser.add_argument('--energy-regression-cluster', action='store_true', help='Turn on energy regression term on loss function and output (regression for neutral particle)')
    parser.add_argument('--energy-regression-weight', action='store_true', help='Turn on weighted edep regression term on loss function and output (regression for neutral particle)')
    parser.add_argument('-e','--momentum', action='store_true', help='Add momentum to GNN input')
    parser.add_argument('-ea','--momentum-amp', action='store_true', help='Add absoute momentum to GNN input')
    parser.add_argument('--mctpe', action='store_true', help='Use MC truth momentum and energy for virtual hits')
    parser.add_argument('-eb','--energy-branch', action='store_true', help='Change GNN model to bypass energy')
    parser.add_argument('--beta-d-scan', action='store_true', help='Turn on beta and diameter scan')
    parser.add_argument('--tbeta', type=float, default=0.6)
    parser.add_argument('--td', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu', help='Specify calculation device')
    parser.add_argument('--lcr-block', action='store_true', help='Use LCR block')
    parser.add_argument('--ddp', action='store_true', help='Use ddp for training')
    parser.add_argument('--classification', action='store_true', help='turn on claasification in LCR')
    parser.add_argument('--pid', action='store_true', help='turn on pid in LCR')

    args = parser.parse_args()
    
    # save_root(sys.argv[1],sys.argv[2],sys.argv[3],nstart=int(sys.argv[4]),nend=int(sys.argv[5]),timingCut=strtobool(sys.argv[6]),input_dim=int(sys.argv[7]), output_dim=int(sys.argv[8]), args=args)
    # save_pred(sys.argv[1],sys.argv[2],sys.argv[3],nstart=int(sys.argv[4]),nend=int(sys.argv[5]),timingCut=strtobool(sys.argv[6]),input_dim=int(sys.argv[7]), output_dim=int(sys.argv[8]), use_charge_track_likeness=int(sys.argv[9]), pandora=strtobool(sys.argv[10]))
    save_pred(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],nstart=int(sys.argv[5]),nend=int(sys.argv[6]),timingCut=strtobool(sys.argv[7]),input_dim=int(sys.argv[8]), output_dim=int(sys.argv[9]), args=args)

if __name__=='__main__':
    main()
    
