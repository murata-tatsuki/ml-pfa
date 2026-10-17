import sys
import numpy as np
from distutils.util import strtobool
#import evaluation_noNoise as ev
import awkward as ak
from model import get_model, get_model_branch, get_clustering_model
from dataset import ILCDataset
from test_yielder import TestYielder, TestYielderWithMLClustering, TestYielderWithMLClustering_trackQuery, TestYielder_transformer_Like_Clustering
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
    pred_edep_weight = np.array([0], dtype=np.float64)
    cond_beta = np.array([0], dtype=np.float64)
    cond_track = np.array([0], dtype=np.int32)
    sed_radius = np.array([0], dtype=np.float64)    # smallest enclosing disk radius

    pred_photon_energy = np.array([0], dtype=np.float64)
    pred_charged_hadron_energy = np.array([0], dtype=np.float64)
    pred_neutral_hadron_energy = np.array([0], dtype=np.float64)
    pred_muon_energy = np.array([0], dtype=np.float64)
    pred_electron_energy = np.array([0], dtype=np.float64)

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
        t.Branch("pred_edep_weight",this.pred_edep_weight,"pred_edep_weight/D")
        t.Branch("cond_beta",this.cond_beta,"cond_beta/D")
        t.Branch("cond_track",this.cond_track,"cond_track/I")
        t.Branch("sed_radius",this.sed_radius,"sed_radius/D")

        t.Branch("pred_photon_energy",this.pred_photon_energy,"pred_photon_energy/D")
        t.Branch("pred_charged_hadron_energy",this.pred_charged_hadron_energy,"pred_charged_hadron_energy/D")
        t.Branch("pred_neutral_hadron_energy",this.pred_neutral_hadron_energy,"pred_neutral_hadron_energy/D")
        t.Branch("pred_muon_energy",this.pred_muon_energy,"pred_muon_energy/D")
        t.Branch("pred_electron_energy",this.pred_electron_energy,"pred_electron_energy/D")

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
    edep_mc = np.array([0], dtype=np.float64)
    pred_edep = np.array([0], dtype=np.float64)
    pred_edep_cluster = np.array([0], dtype=np.float64)
    pred_beta = np.array([0], dtype=np.float64)
    pred_alpha = np.array([0], dtype=np.int32)
    trackness = np.array([0], dtype=np.int32)
    weight_photon = np.array([0], dtype=np.float64)
    weight_charged_hadron = np.array([0], dtype=np.float64)
    weight_neutral_hadron = np.array([0], dtype=np.float64)
    weight_muon = np.array([0], dtype=np.float64)
    weight_electron = np.array([0], dtype=np.float64)

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
        t.Branch("edep_mc",this.edep_mc,"edep_mc/D")
        t.Branch("pred_edep",this.pred_edep,"pred_edep/D")
        t.Branch("pred_edep_cluster",this.pred_edep,"pred_edep_cluster/D")
        t.Branch("pred_beta",this.pred_beta,"pred_beta/D")
        t.Branch("pred_alpha",this.pred_alpha,"pred_alpha/I")
        t.Branch("trackness",this.trackness,"trackness/I")
        t.Branch("weight_photon",this.weight_photon,"weight_photon/D")
        t.Branch("weight_charged_hadron",this.weight_charged_hadron,"weight_charged_hadron/D")
        t.Branch("weight_neutral_hadron",this.weight_neutral_hadron,"weight_neutral_hadron/D")
        t.Branch("weight_muon",this.weight_muon,"weight_muon/D")
        t.Branch("weight_electron",this.weight_electron,"weight_electron/D")

class EventData:
    ''' TTree data for MCParticle
        to be used for evaluating the efficiency
    '''
    event = np.array([0], dtype=np.int32)
    ncluster = np.array([0], dtype=np.int32)
    MC_dijet_energy = np.array([0], dtype=np.float64)
    total_MC_energy_truth = np.array([0], dtype=np.float64)
    total_MC_energy_pred = np.array([0], dtype=np.float64)
    total_predicted_energy_truth = np.array([0], dtype=np.float64)
    total_predicted_energy_pred = np.array([0], dtype=np.float64)


    def setup_branch(this,t):
        t.Branch("event",this.event,"event/I")
        t.Branch("ncluster",this.ncluster,"ncluster/I")
        t.Branch("MC_dijet_energy",this.MC_dijet_energy,"MC_dijet_energy/D")
        t.Branch("total_MC_energy_truth",this.total_MC_energy_truth,"total_MC_energy_truth/D")
        t.Branch("total_MC_energy_pred",this.total_MC_energy_pred,"total_MC_energy_pred/D")
        t.Branch("total_predicted_energy_truth",this.total_predicted_energy_truth,"total_predicted_energy_truth/D")
        t.Branch("total_predicted_energy_pred",this.total_predicted_energy_pred,"total_predicted_energy_pred/D")

class JetData:
    ''' TTree data for MCParticle
        to be used for evaluating the efficiency
    '''
    event = np.array([0], dtype=np.int32)
    # ncluster = np.array([0], dtype=np.int32)
    MC_jet_energy = np.array([0], dtype=np.float64)
    total_predicted_energy_truthBase = np.array([0], dtype=np.float64)
    total_predicted_energy_predBase = np.array([0], dtype=np.float64)


    def setup_branch(this,t):
        t.Branch("event",this.event,"event/I")
        # t.Branch("ncluster",this.ncluster,"ncluster/I")
        t.Branch("MC_jet_energy",this.MC_jet_energy,"MC_jet_energy/D")
        t.Branch("total_predicted_energy_truthBase",this.total_predicted_energy_truthBase,"total_predicted_energy_truthBase/D")
        t.Branch("total_predicted_energy_predBase",this.total_predicted_energy_predBase,"total_predicted_energy_predBase/D")


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


# def save_root(datapath, ckpt, outfile, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, pandora=False, energyRegression=False, momentum=False, momentumAmp=False, mctpe=False):
def save_root(datapath, ckpt_gnn, ckpt_clustering, outfile, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, args={}):
    debug = False
    pandora=args.pandora
    event_energy=args.event_total_energy
    energyRegression=args.energy_regression
    energyRegressionCluster=args.energy_regression_cluster
    energyRegressionWeight=args.energy_regression_weight
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
    if energyRegressionWeight:
            output_dim += 5
    print(f"Loading gnn model from checkpoint {ckpt_gnn}")
    if energy_branch:
        model_gnn = get_model_branch(ckpt_gnn, jit=False, input_dim=input_dim,output_dim=output_dim).to(device)
    else:
        model_gnn = get_model(ckpt_gnn, jit=False, input_dim=input_dim,output_dim=output_dim).to(device)
    model_clustering = get_clustering_model(ckpt_clustering, jit=False, input_dim=input_dim,output_dim=output_dim, lcr_block=args.lcr_block, ddp=args.ddp, pid=args.pid, score_raw=args.score_raw)
    print(f"Loading data from {datapath} with {nstart=}, {nend=}, {timingCut=}")
    dataset = ILCDataset(datapath, timingCut=timingCut, thetaphi=thetaphi, test_mode=True, nstart=nstart, nend=nend, pandora=pandora,momentum=momentum,momentumAmp=momentumAmp)
    yielder = TestYielderWithMLClustering_trackQuery(model=model_gnn, model_clustering=model_clustering, energyRegression=energyRegression, energyRegressionCluster=energyRegressionCluster, classification=args.classification, pid=args.pid, dataset=dataset, device=device, pandora=pandora, score_raw=args.score_raw)

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
    # for i, (event, data, prediction, clustering, matches, condensation_points) in enumerate(yielder.iter_matches(tbeta=0.9, td=0.5, nmax=nmax, energyRegression=energyRegression, energyRegressionCluster=energyRegressionCluster)):

    for i, (event_num, event, prediction, pred_fourvec, truth_four_vector, mask, pcl_prob, cls_logits, attn, clustering, matches) in enumerate(yielder.iter_matches(nmax=nmax)):
        if i == nmax: break
        if i < 10 or i%100 == 0:
            print("Event", i, "processing...")

        matches12, matches21 = matches
        # print("matches12", matches12)
        # print("matches21", matches21)
        # print(clustering)
        if (debug):
            print(f"=== reco --> mc ===")
            for k,v in matches12.items():
                print(f"{k}-->{v}")
            print(f"=== mc --> reco ===")
            for k,v in matches21.items():
                print(f"{k}-->{v}")

        all_truth_ids = list(set(np.unique(event.y[:,0])))
        #all_hitid = list(set(np.unique(event.label[:,0]).astype(np.int32)))
        all_mcid = list(set(np.unique(event.label[:,1]).astype(np.int32)))
        all_cluster_ids = list(set(np.unique(clustering)))
        assert( len(all_truth_ids) == len(all_mcid) )
        assert( len(event.y[:,0]) == len(event.label[:,1]) )
        n_hits = len(event.label[:,1])

        if (debug):
            print(f"{all_truth_ids=}")
            #print(f"{all_mcid=}")
            print(f"{all_cluster_ids=}")

        # matched_reco_clusterIds = matching_1to1(event, clustering, matches12)
        # print(matched_reco_clusterIds)

        total_MC_energy = 0.
        total_predicted_energy = 0.
        total_MC_energy_ = 0.
        total_predicted_energy_ = 0.
        MC_dijet_energy = event.event[1] if event_energy else 0
        MC_jet_energies = event.jet[:,0] if event_energy else np.zeros(2)
        jet_momentum = event.jet[:,1:] if event_energy else np.zeros(1)
        reco_momentum_truthBase = []
        pred_energy_truthBase = []
        reco_momentum_predBase = []
        pred_energy_predBase = []


        # iterate over all mcid
        for id in all_truth_ids:
        
            ''' Get energy in three different ways.
                - edep:       sum the hits that come from the MC particle (Perfect PFA)
                - edep_reco:  find the matching cluster and sum all the hits
                                (including those that do and do not come from the MC particle)
                - edep_match: find the matching cluster and sum those that come from the MC particle
            '''
            ncluster = 0
            matched_ncluster = 0
            matched_cluster = -1

            pattern_mcid = (event.y[:,0]==id)
            match_label = event.label[pattern_mcid]
            match_feat = event.feat[pattern_mcid]
            match_edep = match_feat[:,0].detach().numpy().astype(np.float64)
            # match_track = match_feat[:,5].detach().numpy().astype(np.int32)
            # match_pdg = match_label[:,2].detach().numpy().astype(np.int32)
            # print("pdg", match_pdg)
            # print("track", match_track)
            edep_sum = np.sum(match_edep)
            ncluster = len(pattern_mcid)
            
            edep_reco = 0
            edep_match = 0
            cluster_match = []

            if (id in matches12.keys()):
                reco_match = matches12[id]
                for rid in reco_match:
                    edep_reco = 0
                    edep_match = 0
                    # if (matched_cluster == -1):
                    matched_cluster = rid
                    matched_ncluster += 1
                    pattern_cluster = (clustering==rid)
                    pattern_cluster_feat = event.feat[pattern_cluster]
                    pattern_cluster_edep = pattern_cluster_feat[:,0].detach().numpy().astype(np.float64)
                    edep_reco += np.sum(pattern_cluster_edep)
                    pattern_mcid_cluster = np.logical_and(pattern_mcid, pattern_cluster)
                    edep_mcid_cluster = event.feat[pattern_mcid_cluster][:,0].detach().numpy().astype(np.float64)
                    edep_match += np.sum(edep_mcid_cluster)
                    cluster_match.append([edep_reco, edep_match, rid])
            
            cluster_match = np.array(cluster_match)
            if cluster_match.shape[0]==0:
                continue
            cluster_match_ = cluster_match[np.argsort(cluster_match[:, 1])]
            # print(cluster_match, cluster_match_)
            edep_reco = np.sum(match_edep) if args.truth_clustering else cluster_match_[-1,0]
            edep_match = np.sum(match_edep) if args.truth_clustering else cluster_match_[-1,1]
            pattern_cluster = pattern_mcid if args.truth_clustering else (clustering==cluster_match_[-1,2])
            match_track = event.feat[pattern_cluster]
            match_track = match_track[:,5].detach().numpy().astype(np.float64)


            if not pandora:
                predicted_beta = prediction.pred_betas[pattern_cluster]
                edeps = event.feat[pattern_cluster][:,0].detach().numpy().astype(np.float64)
                predicted_energy = np.zeros(1)
                predicted_energy_cluster = np.zeros(1)
                predicted_energy_weight = np.zeros(1)
                pred_photon_energy = np.zeros(1)
                pred_charged_hadron_energy = np.zeros(1)
                pred_neutral_hadron_energy = np.zeros(1)
                pred_muon_energy = np.zeros(1)
                pred_electron_energy = np.zeros(1)
                if energyRegressionWeight and not energyRegression:
                    pred_weight_photon = prediction.pred_weight_photon[pattern_cluster]
                    pred_weight_charged_hadron = prediction.pred_weight_charged_hadron[pattern_cluster]
                    pred_weight_neutral_hadron = prediction.pred_weight_neutral_hadron[pattern_cluster]
                    pred_weight_muon = prediction.pred_weight_muon[pattern_cluster]
                    pred_weight_electron = prediction.pred_weight_electron[pattern_cluster]
                    pred_weights = pred_weight_photon + pred_weight_charged_hadron + pred_weight_neutral_hadron + pred_weight_muon + pred_weight_electron
                    predicted_energy_weight = np.array([np.sum(edeps * pred_weights)])
                    pred_photon_energy = np.array([np.sum(edeps * pred_weight_photon)])
                    pred_charged_hadron_energy = np.array([np.sum(edeps * pred_weight_charged_hadron)])
                    pred_neutral_hadron_energy = np.array([np.sum(edeps * pred_weight_neutral_hadron)])
                    pred_muon_energy = np.array([np.sum(edeps * pred_weight_muon)])
                    pred_electron_energy = np.array([np.sum(edeps * pred_weight_electron)])
                    predicted_energy = predicted_energy_weight
                if energyRegression and not energyRegressionWeight:
                    predicted_energy = prediction.pred_tracker_energy[pattern_cluster]
                    predicted_energy = predicted_energy[np.argsort(-predicted_beta)]
                    predicted_energy_cluster = prediction.pred_cluster_energy[pattern_cluster] if energyRegressionCluster else -np.ones(1)
                if energyRegression and energyRegressionWeight:
                    predicted_energy = prediction.pred_tracker_energy[pattern_cluster]
                    predicted_energy = predicted_energy[np.argsort(-predicted_beta)]
                    predicted_energy_cluster = prediction.pred_cluster_energy[pattern_cluster] if energyRegressionCluster else -np.ones(1)
                    pred_weight_photon = prediction.pred_weight_photon[pattern_cluster]
                    pred_weight_charged_hadron = prediction.pred_weight_charged_hadron[pattern_cluster]
                    pred_weight_neutral_hadron = prediction.pred_weight_neutral_hadron[pattern_cluster]
                    pred_weight_muon = prediction.pred_weight_muon[pattern_cluster]
                    pred_weight_electron = prediction.pred_weight_electron[pattern_cluster]
                    pred_weights = pred_weight_photon + pred_weight_charged_hadron + pred_weight_neutral_hadron + pred_weight_muon + pred_weight_electron
                    predicted_energy_weight = np.array([np.sum(edeps * pred_weights)])
                    pred_photon_energy = np.array([np.sum(edeps * pred_weight_photon)])
                    pred_charged_hadron_energy = np.array([np.sum(edeps * pred_weight_charged_hadron)])
                    pred_neutral_hadron_energy = np.array([np.sum(edeps * pred_weight_neutral_hadron)])
                    pred_muon_energy = np.array([np.sum(edeps * pred_weight_muon)])
                    pred_electron_energy = np.array([np.sum(edeps * pred_weight_electron)])
                cond_tracknesses = match_track[np.argsort(-predicted_beta)]
                cond_trackness = cond_tracknesses[0]
                predicted_beta = -np.sort(-predicted_beta)
                # print(predicted_beta[0], cond_trackness)
            else:
                predicted_energy = prediction.pred_tracker_energy[pattern_cluster]
                cond_trackness = 0
                predicted_beta = np.zeros(1)

            # for MC particle, take any element from the match because they should be the same
            my_label = match_label[0]
            pred_edep = predicted_energy[0]                                  ## alpha
            pred_edep_cluster = np.sum(predicted_energy_cluster) if not pandora else 0
            # pred_edep = np.sum(predicted_energy) / np.sum(predicted_beta)      ## betaE
            sed_radius=0
            # pred_cood = prediction.pred_cluster_spsace_coords[pattern_mcid]
            # print(type(pred_cood))
            # sed_center, sed_radius = minimum_enclosing_sphere(prediction.pred_cluster_space_coords[pattern_mcid])
            # print(prediction.pred_cluster_space_coords[pattern_mcid].shape, sed_center, _sed_radius, type(_sed_radius), _sed_radius.shape)

            # Set values for TTree and fill
            d.event[0] = i
            d.hitid[0] = my_label[0]
            d.mcid[0] = my_label[1]
            d.truthid[0] = id
            d.mcpdg[0] = my_label[2]
            d.mccharge[0] = my_label[3]
            d.mcmass[0] = my_label[4]
            d.mcpx[0] = my_label[5]
            d.mcpy[0] = my_label[6]
            d.mcpz[0] = my_label[7]
            d.mcen[0] = np.sqrt(d.mcmass[0]**2 + d.mcpx[0]**2 + d.mcpy[0]**2 + d.mcpz[0]**2)
            d.mcstatus[0] = my_label[8]
            d.edep[0] = edep_sum
            d.edep_reco[0] = edep_reco
            d.edep_match[0] = edep_match
            d.ncluster[0] = ncluster
            d.matched_ncluster[0] = matched_ncluster
            d.matched_cluster[0] = matched_cluster
            d.pred_edep[0] = pred_edep
            d.pred_edep_cluster[0] = pred_edep_cluster
            d.cond_beta[0] = predicted_beta[0]
            d.cond_track[0] = cond_trackness
            d.sed_radius[0] = sed_radius
            d.pred_photon_energy[0] = pred_photon_energy
            d.pred_charged_hadron_energy[0] = pred_charged_hadron_energy
            d.pred_neutral_hadron_energy[0] = pred_neutral_hadron_energy
            d.pred_muon_energy[0] = pred_muon_energy
            d.pred_electron_energy[0] = pred_electron_energy

            # if(sed_radius>3):
            #     print(i,sed_center, sed_radius, my_label[2],my_label[5],my_label[6],my_label[7],np.sqrt(d.mcmass[0]**2 + d.mcpx[0]**2 + d.mcpy[0]**2 + d.mcpz[0]**2))

            if (not d.mcid[0] == -1): # skip if track does not have hit
                t.Fill()

            total_MC_energy_ += d.mcen[0]
            total_predicted_energy_ += pred_edep if cond_trackness!=0 else pred_edep_cluster

            reco_momentum_truthBase.append([my_label[5], my_label[6], my_label[7]])
            predenergy = pred_edep if cond_trackness!=0 else pred_edep_cluster
            pred_energy_truthBase.append(predenergy)

        # Iterate over reconstructed clusters
        for cl in all_cluster_ids:
            pattern_cluster = (clustering==cl)
            match_label_ = event.label[pattern_cluster]
            match_feat_ = event.feat[pattern_cluster]
            match_track_ = event.feat[pattern_cluster]
            match_track_ = match_track_[:,5].detach().numpy().astype(np.int32)
            if not pandora:
                predicted_beta_ = prediction.pred_betas[pattern_cluster]
                edeps = event.feat[pattern_cluster][:,0].detach().numpy().astype(np.float64)
                if energyRegression:
                    if not energyRegressionWeight:
                        predicted_energy_ = prediction.pred_tracker_energy[pattern_cluster]
                        predicted_energy_ = predicted_energy_[np.argsort(-predicted_beta_)]
                        predicted_energy_cluster_ = prediction.pred_cluster_energy[pattern_cluster] if energyRegressionCluster else -np.ones(1)
                    else:
                        pred_weights = (prediction.pred_weight_photon + prediction.pred_weight_hadron + prediction.pred_weight_muon + prediction.pred_weight_electron)[pattern_cluster]
                        predicted_energy_ = np.array([np.sum(edeps * pred_weights)])
                        predicted_energy_cluster_ = np.zeros(1)
                    match_label_ = match_label_[np.argsort(-predicted_beta_)]
                    match_feat_ = match_feat_[np.argsort(-predicted_beta_)]
                else:
                    predicted_energy_ = np.zeros(1)
                    predicted_energy_cluster_ = np.zeros(1)
                cond_tracknesses_ = match_track_[np.argsort(-predicted_beta_)]
                cond_trackness_ = cond_tracknesses_[0]
                predicted_beta_ = -np.sort(-predicted_beta_)
                # print(predicted_beta[0], cond_trackness)
            else:
                predicted_energy_ = prediction.pred_tracker_energy[pattern_cluster]
                cond_trackness_ = 0
                predicted_beta_ = np.zeros(1)

            # for MC particle, take any element from the match because they should be the same
            pred_edep_ = predicted_energy_[0]                                  ## alpha
            pred_edep_cluster_ = np.sum(predicted_energy_cluster_) if not pandora else 0
            # print(pred_edep_, pred_edep_cluster_, predicted_beta_, cond_tracknesses_)
            oc_label = match_label_[0]
            total_MC_energy += np.sqrt(oc_label[4]**2 + oc_label[5]**2 + oc_label[6]**2 + oc_label[7]**2)
            total_predicted_energy += pred_edep_ if cond_trackness_!=0 else pred_edep_cluster_

            # reco_momentum_predBase.append([oc_label[5], oc_label[6], oc_label[7]])
            oc_feat = match_feat_[0]
            reco_momentum_predBase.append([oc_feat[1], oc_feat[2], oc_feat[3]])
            predenergy = pred_edep_ if cond_trackness_!=0 else pred_edep_cluster_
            pred_energy_predBase.append(predenergy)
            
            continue
            # break
            pattern_cluster = (clustering==cl)
            pattern_cluster_feat = event.feat[pattern_cluster]
            pattern_cluster_edep = pattern_cluster_feat[:,0].detach().numpy().astype(np.float64)
            edep_reco = np.sum(pattern_cluster_edep)

            # Get MC particle matching the cluster
            matched_mcp_found = False
            matched_mcp = -1
            edep_mcp = -1
            for m in matches21:
                if (m[0] == cl):
                    matched_mcp_found = True
                    matched_mcp = m[0][0]

            edep_mcp = -1
            if (matched_mcp_found):
                pattern_mcp = (event.y[:,0]==matched_mcp)
                pattern_mcp_feat = event.feat[pattern_mcp]
                pattern_mcp_edep = pattern_mcp_feat[:,0].detach().numpy().astype(np.float64)
                edep_mcp = np.sum(pattern_mcp_edep)

                pattern_mcp_match_label = event.label[pattern_mcp].detach().numpy().astype(np.float64)
                mcp_label = pattern_mcp_match_label[0]

            d2.event[0] = i
            d2.cluster[0] = cl

            d2.mcid[0] = -1
            d2.mcpdg[0] = -1
            d2.mccharge[0] = -1
            d2.mcmass[0] = -1
            d2.mcpx[0] = -1
            d2.mcpy[0] = -1
            d2.mcpz[0] = -1
            d2.mcen[0] = -1
            d2.mcstatus[0] = -1
            d2.pred_edep[0] = 0
            d2.pred_edep_cluster[0] = 0

            if (matched_mcp_found):
                d2.mcid[0] = matched_mcp
                d2.mcpdg[0] = mcp_label[2]
                d2.mccharge[0] = mcp_label[3]
                d2.mcmass[0] = mcp_label[4]
                d2.mcpx[0] = mcp_label[5].astype(np.float64)
                d2.mcpy[0] = mcp_label[6]
                d2.mcpz[0] = mcp_label[7]
                d2.mcen[0] = np.sqrt(d2.mcmass[0]**2 + d2.mcpx[0]**2 + d2.mcpy[0]**2 + d2.mcpz[0]**2)
                d2.mcstatus[0] = mcp_label[8]
                d2.pred_edep[0] = pred_edep
                d2.pred_edep_cluster[0] = pred_edep_cluster

            d2.edep_reco[0] = edep_reco
            d2.edep_mc[0] = edep_mcp
            t2.Fill()


        # making predicted trees
        if not pandora:
            for ihit in range(n_hits):
            
                '''
                '''

                # pattern_mcid = (event.y[:,0]==id)
                # match_label = event.label[pattern_mcid]
                # match_feat = event.feat[pattern_mcid]
                # match_edep = match_feat[:,0].detach().numpy().astype(np.float64)
                # edep_sum = np.sum(match_edep)
                # ncluster = len(match_label)

                # for MC particle, take any element from the match because they should be the same
                my_label = event.label[ihit]
                my_feat = event.feat[ihit]
                pred_edep = 0
                pred_beta = 0

                # Set values for TTree and fill
                d3.event[0] = i
                d3.hitid[0] = my_label[0]
                d3.mcid[0] = my_label[1]
                d3.truthid[0] = id
                d3.mcpdg[0] = my_label[2]
                d3.mccharge[0] = my_label[3]
                d3.mcmass[0] = my_label[4]
                d3.mcpx[0] = my_label[5]
                d3.mcpy[0] = my_label[6]
                d3.mcpz[0] = my_label[7]
                d3.mcen[0] = np.sqrt(d3.mcmass[0]**2 + d3.mcpx[0]**2 + d3.mcpy[0]**2 + d3.mcpz[0]**2)
                d3.mcstatus[0] = my_label[8]
                d3.edep_mc[0] = my_feat[0]
                d3.pred_edep[0] = -1 if not energyRegression else (-1 if energyRegression else prediction.pred_tracker_energy[ihit])
                d3.pred_edep_cluster[0] = prediction.pred_cluster_energy[ihit] if energyRegression and energyRegressionCluster else -1
                d3.pred_beta[0] = prediction.pred_betas[ihit]
                d3.pred_alpha[0] = 0 # condensation_points[ihit]
                d3.trackness[0] = my_feat[5]
                d3.weight_photon[0] = prediction.pred_weight_photon[ihit] if energyRegressionWeight else 0
                d3.weight_charged_hadron[0] = prediction.pred_weight_charged_hadron[ihit] if energyRegressionWeight else 0
                d3.weight_neutral_hadron[0] = prediction.pred_weight_neutral_hadron[ihit] if energyRegressionWeight else 0
                d3.weight_muon[0] = prediction.pred_weight_muon[ihit] if energyRegressionWeight else 0
                d3.weight_electron[0] = prediction.pred_weight_electron[ihit] if energyRegressionWeight else 0

                if (not d3.mcid[0] == -1): # skip if track does not have hit
                    t3.Fill()









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
    print(sys.argv)
    if (len(sys.argv) < 9):
        print("Usage: save_root.py datapath ckpt_gnn ckpt_clustering outfile nstart nend timingCut input_dim output_dim pandora energyRegression momentum momentumAmp MCTpe")
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
    parser.add_argument('--event-total-energy', action='store_true', help='Use event visible energy')
    parser.add_argument('--energy-regression', action='store_true', help='Turn on energy regression term on loss function and output')
    parser.add_argument('--energy-regression-cluster', action='store_true', help='Turn on energy regression term on loss function and output (regression for neutral particle)')
    parser.add_argument('--energy-regression-weight', action='store_true', help='Turn on energy regression term on loss function and output (weighted edep)')
    parser.add_argument('-e','--momentum', action='store_true', help='Add momentum to GNN input')
    parser.add_argument('-ea','--momentum-amp', action='store_true', help='Add absoute momentum to GNN input')
    # parser.add_argument('--mctpe', action='store_true', help='Use MC truth momentum and energy for virtual hits')
    parser.add_argument('-eb','--energy-branch', action='store_true', help='Change GNN model to bypass energy')
    # parser.add_argument('--beta-d-scan', action='store_true', help='Turn on beta and diameter scan')
    # parser.add_argument('--tbeta', type=float, default=0.9)
    # parser.add_argument('--td', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu', help='Specify calculation device')
    parser.add_argument('--lcr-block', action='store_true', help='Use LCR block')
    parser.add_argument('--classification', action='store_true', help='turn on claasification in LCR')
    parser.add_argument('--pid', action='store_true', help='Use pid')
    parser.add_argument('--score-raw', action='store_true', help='Use raw score for cross attention')
    parser.add_argument('--ddp', action='store_true', help='Use ddp for training')
    parser.add_argument('--truth-clustering', action='store_true', help='Turn on MC truth clustering')
    # parser.add_argument('--1tomany-clustering', action='store_true', help='Turn on combining reco-clusters')

    args = parser.parse_args()
    
    save_root(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4],nstart=int(sys.argv[5]),nend=int(sys.argv[6]),timingCut=strtobool(sys.argv[7]),input_dim=int(sys.argv[8]), output_dim=int(sys.argv[9]), args=args)

if __name__=='__main__':
    main()
    
