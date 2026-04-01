import sys
import numpy as np
from distutils.util import strtobool
#import evaluation_noNoise as ev
#import event_view_plot as plt_3d
import awkward as ak
import tools.load_awkward as la
from dataset import ILCDataset
from test_yielder_clustering import TestYielder, TestYielderWithMLClustering, TestYielderGNNTransformer
from model import get_model, get_model_branch, get_clustering_model
from matching import get_energy_ABCD,get_mask_charged_neutral,matching_1to1
import argparse
from torch_scatter import scatter_max, scatter_add, scatter_mean


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


def get_hit_cluster_assignment_single_batch(attn_weights, hit_mask=None):
    """
    1バッチ分のヒットがどのクラスタ(seed)に最も寄与しているかを計算
    
    Args:
        attn_weights: Tensor of shape (K, N_hit)
        hit_mask: Optional mask (N_hit,) for valid hits (1=valid,0=padding)
    
    Returns:
        hit_to_cluster: Tensor of shape (N_hit,), 各ヒットに対応するクラスタ index (-1は無効)
        max_contrib: Tensor of shape (N_hit,), ヒットごとの寄与度
    """
    # ヒットごとに最大のクラスタを取得
    max_contrib, hit_to_cluster = attn_weights.max(dim=0)  # (N_hit,)

    if hit_mask is not None:
        mask = hit_mask.bool()
        hit_to_cluster = hit_to_cluster.masked_fill(~mask, -1)
        max_contrib = max_contrib.masked_fill(~mask, 0.0)

    return hit_to_cluster, max_contrib




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
    model_clustering = get_clustering_model(ckpt_clustering, jit=False, input_dim=input_dim,output_dim=output_dim, lcr_block=args.lcr_block, ddp=args.ddp).to(device)
    print(f"Loading data from {datapath} with {nstart=}, {nend=}, {timingCut=}")
    dataset = ILCDataset(datapath, timingCut=timingCut, thetaphi=thetaphi, test_mode=True, nstart=nstart, nend=nend, pandora=pandora,momentum=momentum,momentumAmp=momentumAmp)
    yielder = TestYielderGNNTransformer(model=model_gnn, model_clustering=model_clustering, dataset=dataset, device=device, pandora=pandora, classification=args.classification, pid=args.pid)
    # dataset = ILCDataset(datapath, timingCut=timingCut, thetaphi=thetaphi, test_mode=True, nstart=nstart, nend=nend, pandora=pandora,momentum=momentum,momentumAmp=momentumAmp, mctpe=mctpe)
    # yielder = TestYielder(model=model, dataset=dataset, device=device, pandora=pandora)
    # dataset = ILCDataset(datapath, timingCut=timingCut, thetaphi=thetaphi, test_mode=True, nstart=nstart, nend=nend, pandora=pandora)
    # yielder = TestYielder(model=model, dataset=dataset)

    nmax = None if nend==-1 else nend-nstart+1

    x = []
    y = []
    feats = []
    labels = []
    preds = []
    pands = []
    energy = []

    #for i, (event, prediction) in enumerate(yielder.iter_pred(nmax)):
    # for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta=0.2, td=0.5, nmax=nmax, pandora=pandora)):
    # for i, (event, prediction, clustering, matches, condensation_points) in enumerate(yielder.iter_matches(tbeta=0.9, td=0.5, nmax=nmax, energyRegression=energyRegression, energyRegressionCluster=energyRegressionCluster)):
    for i, (event_num, event, prediction, pred_fourvec, truth_fourvec, hit_mask, attn_w, fourvec_match) in enumerate(yielder._iter_data(nmax=nmax)):
        print(i, event_num, event, prediction, pred_fourvec, truth_fourvec, hit_mask, attn_w, fourvec_match)
        if i == nmax: break

        if i < 10 or i%100 == 0:
            print("Event", i, "processing...")
        
        """
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
        for i, reco_clusterId in enumerate(clustering):
            if (reco_clusterId in dict_energy.keys()):
                np_energy[i] = np.array([dict_energy[reco_clusterId]["pred_edep"], dict_energy[reco_clusterId]["pred_edep_cluster"], dict_energy[reco_clusterId]["mcen"]])
            else:
                np_energy[i] = np.array([-1, -1, -1])
        # label_mcen = np.sqrt(event.label[:,4]**2 + event.label[:,5]**2 + event.label[:,6]**2 + event.label[:,7]**2)
        # truth_ids = event.y[:,0]
        # print(np.array(label_mcen).shape, np.array(truth_ids).shape)
        # out = scatter_mean(np.array(label_mcen), np.array(truth_ids))
        # print(out)
        print(np.unique(np_energy[:,-1]))
        # np_energy[:,-1] = label_mcen
        """

        hit_to_cluster, hit_contrib = get_hit_cluster_assignment_single_batch(attn_w, hit_mask)

        # ヒットiの寄与クラスタ
        print(hit_contrib)
        for i, hit in enumerate(event):
            print(f"ヒット{i} → クラスタ {hit_to_cluster[i].item()}, 寄与度 {hit_contrib[i].item():.3f}")


        clustering = np.ones(event.y[:,0].shape)
        np_energy = np.zeros((event.y.shape[0],4))
        
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
            
        ak_pred = ak.from_numpy(pred)
        ak_pand = ak.from_numpy(clustering)

        # true_charged_mask, pred_charged_mask = get_mask_charged_neutral(event, clustering, matches)
        # eA, eB, eC, eD = get_energy_ABCD(event, true_charged_mask, pred_charged_mask)
        # ak_energy = ak.Array([eA,eB,eC,eD])
        ak_energy = None

        x.append(ak_x)
        y.append(ak_y)
        feats.append(ak_feat)
        labels.append(ak_label)
        preds.append(ak_pred)
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
    ak_pands = ak.Array(pands)
    ak_energy = ak.Array(energy)
    
    print(f"Saving to {outfile}")
    la.save_awkward(outfile, ak_feats, ak_labels, ak_preds, ak_energy, ak_x, ak_y, ak_pands)






def evaluate_particles(true_particles, pred_particles, cost_fn=None, eps_E=0.1, eps_theta=0.05, eps_hit=0.5):
    e_ind = 0
    """
    true_particles, pred_particles: list of dict
        dict keys: 'E', 'p', 'hits' (optional)
    cost_fn: function(t, p) -> float
        マッチング用コスト関数。NoneならΔEでマッチング
    eps_E, eps_theta, eps_hit: 正しいマッチの閾値
    """
    N_true = len(true_particles)
    N_pred = len(pred_particles)
    torch.set_printoptions(edgeitems=1000)
    print(true_particles, pred_particles)
    
    # Hungarianマッチング
    row_ind, col_ind = matching_hungarian_set_bbox_only_(pred_particles, true_particles)
    
    matched_correctly = 0
    delta_E_list = []
    delta_theta_list = []
    
    for i, j in zip(row_ind, col_ind):
        t = true_particles[j]
        p = pred_particles[i]
        print(t, p)
        
        delta_E = abs(p[e_ind] - t[e_ind])
        # delta_E = abs(p[e_ind] - t[e_ind]) / (t[e_ind] + 1e-8)
        delta_theta = angle_between(p[1:], t[1:])
        # if 'hits' in t and 'hits' in p:
        #     jaccard = jaccard_index(t['hits'], p['hits'])
        # else:
            # jaccard = 1.0  # hits情報なしならスキップ
        jaccard = 1.0  # hits情報なしならスキップ

        print(delta_E, delta_theta)
        
        if delta_E < eps_E and delta_theta < eps_theta and jaccard > eps_hit:
            matched_correctly += 1
            delta_E_list.append(delta_E)
            delta_theta_list.append(delta_theta)
    
    # Efficiency / Purity
    efficiency = matched_correctly / N_true if N_true > 0 else 0
    purity = matched_correctly / N_pred if N_pred > 0 else 0
    
    return {
        'Efficiency': efficiency,
        'Purity': purity,
        'MatchedCount': matched_correctly,
        'DeltaE': np.array(delta_E_list),
        'DeltaTheta': np.array(delta_theta_list)
    }

def evaluate_particles_(true_particles, pred_particles, cost_fn=None, eps_E=0.1, eps_theta=0.05, eps_hit=0.5):
    e_ind = 0
    """
    true_particles, pred_particles: list of dict
        dict keys: 'E', 'p', 'hits' (optional)
    cost_fn: function(t, p) -> float
        マッチング用コスト関数。NoneならΔEでマッチング
    eps_E, eps_theta, eps_hit: 正しいマッチの閾値
    """
    N_true = len(true_particles)
    N_pred = len(pred_particles)
    torch.set_printoptions(edgeitems=1000)
    # print(true_particles, pred_particles)
    
    # Hungarianマッチング
    row_ind, col_ind = matching_hungarian_set_bbox_only_(pred_particles, true_particles)
    
    matched_correctly = 0
    true_vec_list = []
    pred_vec_list = []
    delta_E_list = []
    delta_theta_list = []
    
    for i, j in zip(row_ind, col_ind):
        t = true_particles[j]
        p = pred_particles[i]
        # print(t, p)
        
        delta_E = (p[e_ind] - t[e_ind])
        # delta_E = abs(p[e_ind] - t[e_ind]) / (t[e_ind] + 1e-8)
        delta_theta = angle_between(p[1:], t[1:])
        # if 'hits' in t and 'hits' in p:
        #     jaccard = jaccard_index(t['hits'], p['hits'])
        # else:
            # jaccard = 1.0  # hits情報なしならスキップ
        jaccard = 1.0  # hits情報なしならスキップ

        # print(delta_E, delta_theta)
        
        # if delta_E < eps_E and delta_theta < eps_theta and jaccard > eps_hit:
        #     matched_correctly += 1
        #     delta_E_list.append(delta_E)
        #     delta_theta_list.append(delta_theta)
        true_vec_list.append(t)
        pred_vec_list.append(p)
        delta_E_list.append(delta_E)
        delta_theta_list.append(delta_theta)
        
    
    # Efficiency / Purity
    efficiency = matched_correctly / N_true if N_true > 0 else 0
    purity = matched_correctly / N_pred if N_pred > 0 else 0
    
    return {
        'TrueVec': true_vec_list,
        'PredVec': pred_vec_list,
        'DeltaE': np.array(delta_E_list),
        'DeltaTheta': np.array(delta_theta_list)
    }


def angle_between(p1, p2):
    """ベクトルp1, p2の間の角度（ラジアン）"""
    cos_theta = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2) + 1e-8)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def jaccard_index(hit_true, hit_pred):
    """Hit集合のJaccard指数"""
    set_true = set(hit_true)
    set_pred = set(hit_pred)
    intersection = len(set_true & set_pred)
    union = len(set_true | set_pred) + 1e-8
    return intersection / union


def get_hit_cluster_assignment(attn_weights, hit_mask=None):
    """
    各ヒットがどのクラスタ(seed)に最も寄与しているかを計算
    
    Args:
        attn_weights: Tensor of shape (B, K, N_hit)
        hit_mask: Optional mask (B, N_hit) for valid hits (1=valid,0=padding)
    
    Returns:
        hit_to_cluster: Tensor of shape (B, N_hit), 
                        各ヒットに対応するクラスタの index (-1 は無効)
        max_contrib: Tensor of shape (B, N_hit), 寄与度
    """
    B, K, N = attn_weights.shape
    device = attn_weights.device

    # 各ヒットが最も寄与するクラスタを取得
    # w_max_idx : (B,N), max_contrib: (B,N)
    max_contrib, w_max_idx = attn_weights.max(dim=1)  

    if hit_mask is not None:
        w_max_idx = w_max_idx.masked_fill(~hit_mask.bool(), -1)
        max_contrib = max_contrib.masked_fill(~hit_mask.bool(), 0.0)

    return w_max_idx, max_contrib

def get_hit_cluster_assignment_single_batch(attn_weights, hit_mask=None):
    """
    1バッチ分のヒットがどのクラスタ(seed)に最も寄与しているかを計算
    
    Args:
        attn_weights: Tensor of shape (K, N_hit)
        hit_mask: Optional mask (N_hit,) for valid hits (1=valid,0=padding)
    
    Returns:
        hit_to_cluster: Tensor of shape (N_hit,), 各ヒットに対応するクラスタ index (-1は無効)
        max_contrib: Tensor of shape (N_hit,), ヒットごとの寄与度
    """
    # ヒットごとに最大のクラスタを取得
    max_contrib, hit_to_cluster = attn_weights.max(dim=0)  # (N_hit,)

    if hit_mask is not None:
        mask = hit_mask.bool()
        hit_to_cluster = hit_to_cluster.masked_fill(~mask, -1)
        max_contrib = max_contrib.masked_fill(~mask, 0.0)

    return hit_to_cluster, max_contrib





    
def main():
    
    if (len(sys.argv) < 9):
        # print("Usage: save_pred.py datapath ckpt outfile nstart nend timingCut input_dim output_dim use_track_likeness pandora")
        print("Usage: save_pred.py datapath ckpt_gnn ckpt_clustering outfile nstart nend timingCut input_dim output_dim pandora energyRegression momentum momentumAmp")
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
    
