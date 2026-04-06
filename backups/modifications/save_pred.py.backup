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

def save_pred(datapath, ckpt, outfile, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, use_charge_track_likeness=False):
    thetaphi = True if input_dim == 7 else False
    print(f"Loading model from checkpoint {ckpt}")
    model = get_model(ckpt, jit=False, input_dim=input_dim,output_dim=output_dim)
    print(f"Loading data from {datapath} with {nstart=}, {nend=}, {timingCut=}")
    dataset = ILCDataset(datapath, timingCut=timingCut, thetaphi=thetaphi, test_mode=True, nstart=nstart, nend=nend)
    yielder = TestYielder(model=model, dataset=dataset, use_charge_track_likeness=use_charge_track_likeness)

    nmax = None if nend==-1 else nend-nstart+1

    x = []
    y = []
    feats = []
    labels = []
    preds = []
    energy = []

    #for i, (event, prediction) in enumerate(yielder.iter_pred(nmax)):
    for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta=0.2, td=0.5, nmax=nmax)):

        if i == nmax: break

        if i < 10 or i%100 == 0:
            print("Event", i, "processing...")
        
        ak_x = ak.from_numpy(event.x)
        ak_y = ak.from_numpy(event.y)
        ak_feat = ak.from_numpy(event.feat.numpy())
        ak_label = ak.from_numpy(event.label.numpy())

        beta = np.expand_dims(prediction.pred_betas, axis=1)

        clustering_reshape = clustering.reshape(-1,1)

        if use_charge_track_likeness:
            charge_track_likeness = np.expand_dims(prediction.pred_charge_track_likeness, axis=1)
            pred = np.concatenate((beta,clustering_reshape,prediction.pred_cluster_space_coords,charge_track_likeness), axis=1)
        else:
            pred = np.concatenate((beta,clustering_reshape,prediction.pred_cluster_space_coords), axis=1)
            
        ak_pred = ak.from_numpy(pred)

        true_charged_mask, pred_charged_mask = get_mask_charged_neutral(event, clustering, matches)
        eA, eB, eC, eD = get_energy_ABCD(event, true_charged_mask, pred_charged_mask)
        ak_energy = ak.Array([eA,eB,eC,eD])

        x.append(ak_x)
        y.append(ak_y)
        feats.append(ak_feat)
        labels.append(ak_label)
        preds.append(ak_pred)
        energy.append(ak_energy)

    ak_x = ak.Array(x)
    ak_y = ak.Array(y)
    ak_feats = ak.Array(feats)
    ak_labels = ak.Array(labels)
    ak_preds = ak.Array(preds)
    ak_energy = ak.Array(energy)
    
    print(f"Saving to {outfile}")
    la.save_awkward(outfile, ak_feats, ak_labels, ak_preds, ak_energy, ak_x, ak_y)
    
def main():
    
    if (len(sys.argv) != 10):
        print("Usage: save_pred.py datapath ckpt outfile nstart nend timingCut input_dim output_dim use_track_likeness")
        return
    
    save_pred(sys.argv[1],sys.argv[2],sys.argv[3],nstart=int(sys.argv[4]),nend=int(sys.argv[5]),timingCut=strtobool(sys.argv[6]),input_dim=int(sys.argv[7]), output_dim=int(sys.argv[8]), use_charge_track_likeness=int(sys.argv[9]))

if __name__=='__main__':
    main()
    
