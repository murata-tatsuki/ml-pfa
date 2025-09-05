import sys
import numpy as np
from distutils.util import strtobool
#import evaluation_noNoise as ev
import awkward as ak
from model import get_model, get_model_branch, get_clustering_model
from dataset import ILCDataset
from test_yielder import TestYielder, TestYielderWithMLClustering
from ROOT import TFile, TTree
import argparse
import torch
from sed import minimum_enclosing_sphere
from matching import matching_1to1, matching_hungarian_set_bbox_only, matching_hungarian_set_bbox_only_

## 1 to 1 match to reco-cluster and true cluster
## the largest edep_match reco-cluster is chosen

class Data:
    ''' TTree data for MCParticle
        to be used for evaluating the efficiency
    '''
    event = np.array([0], dtype=np.int32)
    hitid = np.array([0], dtype=np.int32)
    mcid = np.array([0], dtype=np.int32)
    truthid = np.array([0], dtype=np.int32)
    mcpdg = np.array([0], dtype=np.int32)
    mccharge = np.array([0], dtype=np.int32)
    mcmass = np.array([0], dtype=np.float64)
    mcpx = np.array([0], dtype=np.float64)
    mcpy = np.array([0], dtype=np.float64)
    mcpz = np.array([0], dtype=np.float64)
    mcen = np.array([0], dtype=np.float64)
    mcstatus = np.array([0], dtype=np.int32)
    edep = np.array([0], dtype=np.float64)
    edep_reco = np.array([0], dtype=np.float64)
    edep_match = np.array([0], dtype=np.float64)
    ncluster = np.array([0], dtype=np.int32)
    matched_ncluster = np.array([0], dtype=np.int32)
    matched_cluster = np.array([0], dtype=np.int32)
    pred_edep = np.array([0], dtype=np.float64)
    pred_edep_cluster = np.array([0], dtype=np.float64)
    cond_beta = np.array([0], dtype=np.float64)
    cond_track = np.array([0], dtype=np.int32)
    sed_radius = np.array([0], dtype=np.float64)    # smallest enclosing disk radius


    def setup_branch(this,t):
        t.Branch("event",this.event,"event/I")
        t.Branch("hitid",this.hitid,"hitid/I")
        t.Branch("mcid",this.mcid,"mcid/I")
        t.Branch("truthid",this.truthid,"truthid/I")
        t.Branch("mcpdg",this.mcpdg,"mcpdg/I")
        t.Branch("mccharge",this.mccharge,"mccharge/I")
        t.Branch("mcmass",this.mcmass,"mcmass/D")
        t.Branch("mcpx",this.mcpx,"mcpx/D")
        t.Branch("mcpy",this.mcpy,"mcpy/D")
        t.Branch("mcpz",this.mcpz,"mcpz/D")
        t.Branch("mcen",this.mcen,"mcen/D")
        t.Branch("mcstatus",this.mcstatus,"mcstatus/I")
        t.Branch("edep",this.edep,"edep/D")
        t.Branch("edep_reco",this.edep_reco,"edep_reco/D")
        t.Branch("edep_match",this.edep_match,"edep_match/D")
        t.Branch("ncluster",this.matched_ncluster,"ncluster/I")
        t.Branch("matched_ncluster",this.matched_ncluster,"matched_ncluster/I")
        t.Branch("matched_cluster",this.matched_cluster,"matched_cluster/I")
        t.Branch("pred_edep",this.pred_edep,"pred_edep/D")
        t.Branch("pred_edep_cluster",this.pred_edep_cluster,"pred_edep_cluster/D")
        t.Branch("cond_beta",this.cond_beta,"cond_beta/D")
        t.Branch("cond_track",this.cond_track,"cond_track/I")
        t.Branch("sed_radius",this.sed_radius,"sed_radius/D")

#   ak_feat: edep, x, y, z, time, track, charge, px, py, pz (atcalo)
#   ak_label: hitid, mcid, pdg, charge, mass, px, py, pz (of mcp), status

class RecoData:
    ''' TTree data for reconstructed cluster
        to be used for evaluating the purity
    '''
    event = np.array([0], dtype=np.int32)
    cluster = np.array([0], dtype=np.int32)
    nhits = np.array([0], dtype=np.int32)
    mcid = np.array([0], dtype=np.int32)
    mcpdg = np.array([0], dtype=np.int32)
    mccharge = np.array([0], dtype=np.int32)
    mcmass = np.array([0], dtype=np.float64)
    mcpx = np.array([0], dtype=np.float64)
    mcpy = np.array([0], dtype=np.float64)
    mcpz = np.array([0], dtype=np.float64)
    mcen = np.array([0], dtype=np.float64)
    mcstatus = np.array([0], dtype=np.int32)
    edep_reco = np.array([0], dtype=np.float64)
    edep_mc = np.array([0], dtype=np.float64)
    edep_match = np.array([0], dtype=np.float64)
    pred_edep = np.array([0], dtype=np.float64)
    pred_edep_cluster = np.array([0], dtype=np.float64)

    def setup_branch(this,t):
        t.Branch("event",this.event,"event/I")
        t.Branch("cluster",this.cluster,"cluster/I")
        t.Branch("nhits",this.nhits,"nhits/I")
        t.Branch("mcid",this.mcid,"mcid/I")
        t.Branch("mcpdg",this.mcpdg,"mcpdg/I")
        t.Branch("mccharge",this.mccharge,"mccharge/I")
        t.Branch("mcmass",this.mcmass,"mcmass/D")
        t.Branch("mcpx",this.mcpx,"mcpx/D")
        t.Branch("mcpy",this.mcpy,"mcpy/D")
        t.Branch("mcpz",this.mcpz,"mcpz/D")
        t.Branch("mcen",this.mcen,"mcen/D")
        t.Branch("mcstatus",this.mcstatus,"mcstatus/I")
        t.Branch("edep_reco",this.edep_reco,"edep_reco/D")
        t.Branch("edep_mc",this.edep_mc,"edep_mc/D")
        t.Branch("edep_match",this.edep_match,"edep_match/D")
        t.Branch("pred_edep",this.pred_edep,"pred_edep/D")
        t.Branch("pred_edep_cluster",this.pred_edep_cluster,"pred_edep_cluster/D")

class PredData:
    ''' TTree data for predicted (output of GravNet)
        to be used for evaluating the purity
    '''
    event = np.array([0], dtype=np.int32)
    hitid = np.array([0], dtype=np.int32)
    mcid = np.array([0], dtype=np.int32)
    truthid = np.array([0], dtype=np.int32)
    mcpdg = np.array([0], dtype=np.int32)
    mccharge = np.array([0], dtype=np.int32)
    mcmass = np.array([0], dtype=np.float64)
    mcpx = np.array([0], dtype=np.float64)
    mcpy = np.array([0], dtype=np.float64)
    mcpz = np.array([0], dtype=np.float64)
    mcen = np.array([0], dtype=np.float64)
    mcstatus = np.array([0], dtype=np.int32)
    # edep_mc = np.array([0], dtype=np.float64)
    pred_edep = np.array([0], dtype=np.float64)
    pred_edep_cluster = np.array([0], dtype=np.float64)
    pred_beta = np.array([0], dtype=np.float64)
    pred_alpha = np.array([0], dtype=np.int32)
    trackness = np.array([0], dtype=np.int32)

    def setup_branch(this,t):
        t.Branch("event",this.event,"event/I")
        t.Branch("hitid",this.mcid,"hitid/I")
        t.Branch("mcid",this.mcid,"mcid/I")
        t.Branch("truthid",this.mcid,"truthid/I")
        t.Branch("mcpdg",this.mcpdg,"mcpdg/I")
        t.Branch("mccharge",this.mccharge,"mccharge/I")
        t.Branch("mcmass",this.mcmass,"mcmass/D")
        t.Branch("mcpx",this.mcpx,"mcpx/D")
        t.Branch("mcpy",this.mcpy,"mcpy/D")
        t.Branch("mcpz",this.mcpz,"mcpz/D")
        t.Branch("mcen",this.mcen,"mcen/D")
        t.Branch("mcstatus",this.mcstatus,"mcstatus/I")
        # t.Branch("edep_mc",this.edep_mc,"edep_mc/D")
        t.Branch("pred_edep",this.pred_edep,"pred_edep/D")
        t.Branch("pred_edep_cluster",this.pred_edep,"pred_edep_cluster/D")
        t.Branch("pred_beta",this.pred_beta,"pred_beta/D")
        t.Branch("pred_alpha",this.pred_alpha,"pred_alpha/I")
        t.Branch("trackness",this.trackness,"trackness/I")


# def save_root(datapath, ckpt, outfile, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, pandora=False, energyRegression=False, momentum=False, momentumAmp=False, mctpe=False):
def save_root(datapath, ckpt_gnn, ckpt_clustering, outfile, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, args={}):
    debug = False
    pandora=args.pandora
    energyRegression=args.energy_regression
    energyRegressionCluster=args.energy_regression_cluster
    momentum=args.momentum
    momentumAmp=args.momentum_amp
    energy_branch=args.energy_branch
    device=args.device
    if 'cuda' in device: torch.cuda.set_device(device)

    thetaphi = True if input_dim == 7 else False
    if momentum:
        input_dim += 3 
        if momentumAmp:
            input_dim += 1
    if energyRegression:
        output_dim += 1
        if energyRegressionCluster:
            output_dim += 1
    print(f"Loading gnn model from checkpoint {ckpt_gnn}")
    if energy_branch:
        model_gnn = get_model_branch(ckpt_gnn, jit=False, input_dim=input_dim,output_dim=output_dim).to(device)
    else:
        model_gnn = get_model(ckpt_gnn, jit=False, input_dim=input_dim,output_dim=output_dim).to(device)
    model_clustering = get_clustering_model(ckpt_clustering, jit=False, input_dim=input_dim,output_dim=output_dim).to(device)
    print(f"Loading data from {datapath} with {nstart=}, {nend=}, {timingCut=}")
    dataset = ILCDataset(datapath, timingCut=timingCut, thetaphi=thetaphi, test_mode=True, nstart=nstart, nend=nend, pandora=pandora,momentum=momentum,momentumAmp=momentumAmp)
    yielder = TestYielderWithMLClustering(model=model_gnn, model_clustering=model_clustering, dataset=dataset, device=device, pandora=pandora)

    nmax = None if nend==-1 else nend-nstart+1
    print("number of entry : ", nmax)

    """
    ak_feat: edep, x, y, z, time, track, charge, px, py, pz (atcalo)
        --> save edep, drop others
    ak_label: hitid, mcid, pdg, charge, mass, px, py, pz (of mcp), status
        --> save all labels
    """
    
    outfileDir = outfile


            # outfile = outfileDir + '/tbeta' + format(tbeta_now, '02') + '0td' + format(td_now, '02') + '0.root'

    print("")
    print(f"save_root()...  {outfile}")
    print("")
    file = TFile(outfile,"recreate")
    
    t = TTree("t","tree for MCParticle")
    d = Data()
    d.setup_branch(t)
    t2 = TTree("reco","tree for reconstructed clusters")
    d2 = RecoData()
    d2.setup_branch(t2)
    t3 = TTree("prediction","tree for model output")
    d3 = PredData()
    d3.setup_branch(t3)
    
        # for i, (event, prediction, clustering, matches, condensation_points) in enumerate(yielder.iter_matches(tbeta=0.6, td=0.5, nmax=nmax, pandora=pandora, energyRegression=energyRegression)):
    for i, (event_num, event_data, pred_fourvec, truth_fourvec) in enumerate(yielder._iter_data(nmax=nmax)):
        if i == nmax: break
        if i < 10 or i%100 == 0:
            print("Event", i, "processing...")

            result = evaluate_particles(truth_fourvec, pred_fourvec)

            print("     Efficiency:", result['Efficiency'])
            print("     Purity:", result['Purity'])
            print("     Matched particles:", result['MatchedCount'])
            print("     Delta E:", result['DeltaE'])
            print("     Delta theta (rad):", result['DeltaTheta'])






    print(f"Saving to {outfile}")
    file.Write()


# def evaluate_particles(true_particles, pred_particles, cost_fn=None, eps_E=0.1, eps_theta=0.05, eps_hit=0.5):
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


def main():
    print(sys.argv)
    if (len(sys.argv) < 9):
        print("Usage: save_root.py datapath ckpt outfile nstart nend timingCut input_dim output_dim pandora energyRegression momentum momentumAmp MCTpe")
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
    parser.add_argument('-e','--momentum', action='store_true', help='Add momentum to GNN input')
    parser.add_argument('-ea','--momentum-amp', action='store_true', help='Add absoute momentum to GNN input')
    # parser.add_argument('--mctpe', action='store_true', help='Use MC truth momentum and energy for virtual hits')
    parser.add_argument('-eb','--energy-branch', action='store_true', help='Change GNN model to bypass energy')
    # parser.add_argument('--beta-d-scan', action='store_true', help='Turn on beta and diameter scan')
    # parser.add_argument('--tbeta', type=float, default=0.9)
    # parser.add_argument('--td', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu', help='Specify calculation device')
    # parser.add_argument('--truth-clustering', action='store_true', help='Turn on MC truth clustering')
    # parser.add_argument('--1tomany-clustering', action='store_true', help='Turn on combining reco-clusters')

    args = parser.parse_args()
    
    save_root(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],nstart=int(sys.argv[5]),nend=int(sys.argv[6]),timingCut=strtobool(sys.argv[7]),input_dim=int(sys.argv[8]), output_dim=int(sys.argv[9]), args=args)

if __name__=='__main__':
    main()
    
