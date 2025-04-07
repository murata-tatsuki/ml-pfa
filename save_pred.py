import sys
import numpy as np
from distutils.util import strtobool
#import evaluation_noNoise as ev
#import event_view_plot as plt_3d
import awkward as ak
import tools.load_awkward as la
from dataset import ILCDataset
from test_yielder import TestYielder
from model import get_model
from matching import get_energy_ABCD,get_mask_charged_neutral
import argparse

# def save_pred(datapath, ckpt, outfile, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, use_charge_track_likeness=False, pandora=False):
def save_pred(datapath, ckpt, outfile, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, args={}):
    pandora=args.pandora
    energyRegression=args.energy_regression
    energyRegressionCluster=args.energy_regression_cluster
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
    if energyRegressionCluster:
        output_dim += 1
    print(f"Loading model from checkpoint {ckpt}")
    model = get_model(ckpt, jit=False, input_dim=input_dim,output_dim=output_dim)
    print(f"Loading data from {datapath} with {nstart=}, {nend=}, {timingCut=}")
    dataset = ILCDataset(datapath, timingCut=timingCut, thetaphi=thetaphi, test_mode=True, nstart=nstart, nend=nend, pandora=pandora,momentum=momentum,momentumAmp=momentumAmp, mctpe=mctpe)
    yielder = TestYielder(model=model, dataset=dataset, device=device)
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
    for i, (event, prediction, clustering, matches, condensation_points) in enumerate(yielder.iter_matches(tbeta=0.9, td=1, nmax=nmax, pandora=pandora, energyRegression=energyRegression, energyRegressionCluster=energyRegressionCluster)):

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

        matches12, matches21 = matches
        dict_energy = {}
        for id in all_truth_ids:
            pattern_mcid = (event.y[:,0]==id)
            match_label = event.label[pattern_mcid] 
            
            if (id in matches12.keys()):
                reco_match = matches12[id]
                for rid in reco_match:
                    pattern_cluster = (clustering==rid)
                    
                    if not pandora:
                        predicted_beta = prediction.pred_betas[pattern_cluster]
                        predicted_energy = prediction.pred_tracker_energy[pattern_cluster]
                        predicted_energy = predicted_energy[np.argsort(-predicted_beta)]
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
                    my_label = match_label[0]
                    mcen = np.sqrt(my_label[4]**2 + my_label[5]**2 + my_label[6]**2 + my_label[7]**2)
                    dict_energy[rid] = {"truth_cluster_id":id, "reco_cluster_id":rid, "pred_edep":pred_edep, "pred_edep_cluster":pred_edep_cluster, "mcen":mcen}
            # for MC particle, take any element from the match because they should be the same

        np_energy = np.zeros((event.y.shape[0],3))
        for i, reco_clusterId in enumerate(clustering):
            np_energy[i] = np.array([dict_energy[reco_clusterId]["pred_edep"], dict_energy[reco_clusterId]["pred_edep_cluster"], dict_energy[reco_clusterId]["mcen"]])

        
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
    
def main():
    
    if (len(sys.argv) < 9):
        # print("Usage: save_pred.py datapath ckpt outfile nstart nend timingCut input_dim output_dim use_track_likeness pandora")
        print("Usage: save_pred.py datapath ckpt outfile nstart nend timingCut input_dim output_dim pandora energyRegression momentum momentumAmp")
        return
    

    parser = argparse.ArgumentParser()
    parser.add_argument('datapath')
    parser.add_argument('ckpt')
    parser.add_argument('outfile')
    parser.add_argument('nstart', type=int)
    parser.add_argument('nend', type=int)
    parser.add_argument('timingCut')
    parser.add_argument('input_dim', type=int)
    parser.add_argument('output_dim', type=int)
    parser.add_argument('--pandora', action='store_true', help='Use PandoraPFA result')
    parser.add_argument('--energy-regression', action='store_true', help='Turn on energy regression term on loss function and output')
    parser.add_argument('--energy-regression-cluster', action='store_true', help='Turn on energy regression term on loss function and output (regression for neutral particle)')
    parser.add_argument('-e','--momentum', action='store_true', help='Add momentum to GNN input')
    parser.add_argument('-ea','--momentum-amp', action='store_true', help='Add absoute momentum to GNN input')
    parser.add_argument('--mctpe', action='store_true', help='Use MC truth momentum and energy for virtual hits')
    parser.add_argument('-eb','--energy-branch', action='store_true', help='Change GNN model to bypass energy')
    parser.add_argument('--beta-d-scan', action='store_true', help='Turn on beta and diameter scan')
    parser.add_argument('--tbeta', type=float, default=0.6)
    parser.add_argument('--td', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu', help='Specify calculation device')

    args = parser.parse_args()
    
    # save_root(sys.argv[1],sys.argv[2],sys.argv[3],nstart=int(sys.argv[4]),nend=int(sys.argv[5]),timingCut=strtobool(sys.argv[6]),input_dim=int(sys.argv[7]), output_dim=int(sys.argv[8]), args=args)
    # save_pred(sys.argv[1],sys.argv[2],sys.argv[3],nstart=int(sys.argv[4]),nend=int(sys.argv[5]),timingCut=strtobool(sys.argv[6]),input_dim=int(sys.argv[7]), output_dim=int(sys.argv[8]), use_charge_track_likeness=int(sys.argv[9]), pandora=strtobool(sys.argv[10]))
    save_pred(sys.argv[1],sys.argv[2],sys.argv[3],nstart=int(sys.argv[4]),nend=int(sys.argv[5]),timingCut=strtobool(sys.argv[6]),input_dim=int(sys.argv[7]), output_dim=int(sys.argv[8]), args=args)

if __name__=='__main__':
    main()
    
