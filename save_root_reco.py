import os
import sys
import numpy as np
from distutils.util import strtobool
#import evaluation_noNoise as ev
import awkward as ak
from model import get_model, get_model_branch
from dataset import ILCDataset
from test_yielder_edit import TestYielder
from ROOT import TFile, TTree
import argparse
import torch
from sed import minimum_enclosing_sphere
from matching import matching_1to1

## 1 to 1 match to reco-cluster and true cluster
## the largest edep_match reco-cluster is chosen

def to_numpy(x):
    """Convert torch/numpy-like arrays to numpy safely."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)

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
    ntrack_hits = np.array([0], dtype=np.int32)
    cond_beta = np.array([0], dtype=np.float64)
    cond_is_track = np.array([0], dtype=np.int32)
    matched_truth_pdgid = np.array([0], dtype=np.int32)
    matched_truth_hit_frac = np.array([0], dtype=np.float64)
    matched_truth_edep_frac = np.array([0], dtype=np.float64)
    npdg_comp = np.array([0], dtype=np.int32)
    pdg_comp_ids = np.zeros(64, dtype=np.int32)
    pdg_comp_hits = np.zeros(64, dtype=np.int32)
    pdg_comp_hit_frac = np.zeros(64, dtype=np.float64)
    pdg_comp_edep_frac = np.zeros(64, dtype=np.float64)
    pdg_comp_edep = np.zeros(64, dtype=np.float64)
    pdg_comp_truth_edep = np.zeros(64, dtype=np.float64)
    pdg_comp_track_hits = np.zeros(64, dtype=np.int32)

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
        t.Branch("ntrack_hits",this.ntrack_hits,"ntrack_hits/I")
        t.Branch("cond_beta",this.cond_beta,"cond_beta/D")
        t.Branch("cond_is_track",this.cond_is_track,"cond_is_track/I")
        t.Branch("matched_truth_pdgid",this.matched_truth_pdgid,"matched_truth_pdgid/I")
        t.Branch("matched_truth_hit_frac",this.matched_truth_hit_frac,"matched_truth_hit_frac/D")
        t.Branch("matched_truth_edep_frac",this.matched_truth_edep_frac,"matched_truth_edep_frac/D")
        t.Branch("npdg_comp",this.npdg_comp,"npdg_comp/I")
        t.Branch("pdg_comp_ids",this.pdg_comp_ids,"pdg_comp_ids[npdg_comp]/I")
        t.Branch("pdg_comp_hits",this.pdg_comp_hits,"pdg_comp_hits[npdg_comp]/I")
        t.Branch("pdg_comp_hit_frac",this.pdg_comp_hit_frac,"pdg_comp_hit_frac[npdg_comp]/D")
        t.Branch("pdg_comp_edep_frac",this.pdg_comp_edep_frac,"pdg_comp_edep_frac[npdg_comp]/D")
        t.Branch("pdg_comp_edep",this.pdg_comp_edep,"pdg_comp_edep[npdg_comp]/D")
        t.Branch("pdg_comp_truth_edep",this.pdg_comp_truth_edep,"pdg_comp_truth_edep[npdg_comp]/D")
        t.Branch("pdg_comp_track_hits",this.pdg_comp_track_hits,"pdg_comp_track_hits[npdg_comp]/I")

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
    # jet_p4[MAX_JETS][4]: 行=ジェット、(E, px, py, pz)。実ジェット数は n_jets（以降の行は 0 埋め）
    MAX_JETS = 2

    event = np.array([0], dtype=np.int32)
    n_jets = np.array([0], dtype=np.int32)
    jet_p4 = np.zeros((MAX_JETS, 4), dtype=np.float64)

    def setup_branch(this,t):
        t.Branch("event",this.event,"event/I")
        t.Branch("n_jets",this.n_jets,"n_jets/I")
        t.Branch("jet_p4",this.jet_p4,f"jet_p4[{JetData.MAX_JETS}][4]/D")


def calc_origin_quark(quarks: np.array, clusters: np.array):    # calcurated from closest angles
    quarks = quarks / np.linalg.norm(quarks, axis=1, keepdims=True)
    clusters = clusters / np.linalg.norm(clusters, axis=1, keepdims=True)
    cos_angles = clusters @ quarks.T
    nearest_indices = np.argmax(cos_angles, axis=1)
    return nearest_indices

def calc_pred_jet_energy(energies: np.array, nearest_indices: np.array):    # calcurated jet energy from closest angle
    assert(not (energies.shape != nearest_indices.shape))
    return np.array([np.sum(energies[nearest_indices==0]), np.sum(energies[nearest_indices==1])])

# def save_root(datapath, ckpt, outfile, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, pandora=False, energyRegression=False, momentum=False, momentumAmp=False, mctpe=False):
def save_root(datapath, ckpt, outfile, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, args={}):
    debug = False
    pandora=args.pandora
    event_energy=args.event_total_energy
    energyRegression=args.energy_regression
    energyRegressionCluster=args.energy_regression_cluster
    energyRegressionWeight=args.energy_regression_weight
    momentum=args.momentum
    momentumAmp=args.momentum_amp
    mctpe=args.mctpe
    energy_branch=args.energy_branch
    device=args.device
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
    print(f"Loading model from checkpoint {ckpt}")
    if energy_branch:
        model = get_model_branch(ckpt, jit=False, input_dim=input_dim,output_dim=output_dim).to(device)
    else:
        model = get_model(ckpt, jit=False, input_dim=input_dim, output_dim=output_dim, energy_regression=energyRegression, energy_regression_cluster=energyRegressionCluster, energy_regression_weight=energyRegressionWeight, model_variant=args.model_variant).to(device)
    print(f"Loading data from {datapath} with {nstart=}, {nend=}, {timingCut=}")
    dataset = ILCDataset(datapath, timingCut=timingCut, thetaphi=thetaphi, test_mode=True, nstart=nstart, nend=nend, pandora=pandora,momentum=momentum,momentumAmp=momentumAmp, mctpe=mctpe,event_energy=event_energy)
    yielder = TestYielder(model=model, dataset=dataset, device=device, pandora=pandora, event_energy=event_energy)

    nmax = None if nend==-1 else nend-nstart+1
    print("number of entry : ", nmax)

    """
    ak_feat: edep, x, y, z, time, track, charge, px, py, pz (atcalo)
        --> save edep, drop others
    ak_label: hitid, mcid, pdg, charge, mass, px, py, pz (of mcp), status
        --> save all labels
    """
    
    outfileDir = outfile
    tbeta_list = [args.tbeta]
    td_list = [args.td]

    if args.beta_d_scan:
        tbeta_list = [i/10.0 for i in range(9,0,-1)]
        td_list = [i/10.0 for i in range(9,0,-1)]
        # tbeta_list = [0.9+i/100.0 for i in range(9,0,-1)]
        # td_list = [i/10.0 for i in range(9,0,-1)]

    print(tbeta_list)
    print(td_list)


    for tbeta in tbeta_list:
        for td in td_list:
            tbeta_now = round(tbeta * 100)
            td_now = round(td * 100)
            # outfile = outfileDir + '/tbeta' + format(tbeta_now, '02') + '0td' + format(td_now, '02') + '0.root'
            outfile = outfile if not args.beta_d_scan else outfileDir + '/tbeta' + format(tbeta_now, '03') + 'td' + format(td_now, '03') + '.root'

            print("")
            print(f"save_root()...  {outfile}")
            print("")
            outdir = os.path.dirname(outfile)
            if outdir:
                os.makedirs(outdir, exist_ok=True)
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

            t4 = TTree("event","tree for event")
            d4 = EventData()
            d4.setup_branch(t4)

            t5 = TTree("jet","tree for jets")
            d5 = JetData()
            d5.setup_branch(t5)

            # for i, (event, prediction, clustering, matches, condensation_points) in enumerate(yielder.iter_matches(tbeta=0.6, td=0.5, nmax=nmax, pandora=pandora, energyRegression=energyRegression)):
            for i, (event, prediction, clustering, matches, condensation_points) in enumerate(yielder.iter_matches(tbeta=tbeta, td=td, nmax=nmax, energyRegression=energyRegression, energyRegressionCluster=energyRegressionCluster, energyRegressionWeight=energyRegressionWeight)):
                if i == nmax: break
                if i < 10 or i%100 == 0:
                    print("Event", i, "processing...")

                matches12, matches21 = matches
                # print("matches12", matches12)
                # print("matches21", matches21)
                # print(clustering)
                if args.truth_clustering: clustering = event.y[:,0]
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
                    d.pred_edep_weight[0] = predicted_energy_weight
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
                event_truth_ids_all = to_numpy(event.y[:,0]).astype(np.int32)
                event_edeps_all = to_numpy(event.feat[:,0]).astype(np.float64)
                truth_total_hits = {}
                truth_total_edep = {}
                for tid_all in np.unique(event_truth_ids_all):
                    mask_all = (event_truth_ids_all == tid_all)
                    tid_key = int(tid_all)
                    truth_total_hits[tid_key] = int(np.count_nonzero(mask_all))
                    truth_total_edep[tid_key] = float(np.sum(event_edeps_all[mask_all]))

                for cl in all_cluster_ids:
                    pattern_cluster = (clustering==cl)
                    match_label_ = event.label[pattern_cluster]
                    match_feat_ = event.feat[pattern_cluster]
                    match_track_ = event.feat[pattern_cluster]
                    match_track_ = to_numpy(match_track_[:,5]).astype(np.int32)
                    cluster_truth_ids = to_numpy(event.y[pattern_cluster][:,0]).astype(np.int32)
                    cluster_edeps = to_numpy(match_feat_[:,0]).astype(np.float64)
                    nhits_cluster = len(match_label_)
                    ntrack_hits_cluster = int(np.count_nonzero(match_track_))
                    edep_reco_cluster = float(np.sum(cluster_edeps))
                    if not pandora:
                        predicted_beta_ = prediction.pred_betas[pattern_cluster]
                        edeps = cluster_edeps
                        if energyRegression:
                            if not energyRegressionWeight:
                                predicted_energy_ = prediction.pred_tracker_energy[pattern_cluster]
                                predicted_energy_ = predicted_energy_[np.argsort(-predicted_beta_)]
                                predicted_energy_cluster_ = prediction.pred_cluster_energy[pattern_cluster] if energyRegressionCluster else -np.ones(1)
                            else:
                                predicted_energy_ = prediction.pred_tracker_energy[pattern_cluster]
                                predicted_energy_ = predicted_energy_[np.argsort(-predicted_beta_)]
                                pred_weights = (prediction.pred_weight_photon + prediction.pred_weight_charged_hadron + prediction.pred_weight_neutral_hadron + prediction.pred_weight_muon + prediction.pred_weight_electron)[pattern_cluster]
                                predicted_energy_weight = np.array([np.sum(edeps * pred_weights)])
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
                    if energyRegressionWeight:
                        predenergy = pred_edep_ if cond_trackness_!=0 else predicted_energy_weight
                    else:
                        predenergy = pred_edep_ if cond_trackness_!=0 else pred_edep_cluster_
                    pred_energy_predBase.append(predenergy)

                    # Compose truth contributions in this reconstructed cluster
                    unique_truth = np.unique(cluster_truth_ids)
                    contrib = []
                    for tid in unique_truth:
                        mask_tid = (cluster_truth_ids == tid)
                        if not np.any(mask_tid):
                            continue
                        hit_count_tid = int(np.count_nonzero(mask_tid))
                        edep_tid = float(np.sum(cluster_edeps[mask_tid]))
                        track_hit_count_tid = int(np.count_nonzero(match_track_[mask_tid]))
                        total_hit_tid = int(truth_total_hits.get(int(tid), 0))
                        total_edep_tid = float(truth_total_edep.get(int(tid), 0.0))
                        tid_labels = match_label_[mask_tid]
                        pdg_tid = int(tid_labels[0][2]) if len(tid_labels) > 0 else -1
                        contrib.append((int(tid), pdg_tid, hit_count_tid, edep_tid, track_hit_count_tid, total_hit_tid, total_edep_tid))

                    dominant_truth_id = -1
                    dominant_truth_pdg = -1
                    dominant_hit_frac = 0.0
                    dominant_edep_frac = 0.0
                    edep_match_cluster = 0.0
                    mcp_label = None
                    edep_mc = -1.0

                    if len(contrib) > 0:
                        # "割合が一番多い truth" is defined by hit fraction (tie-break by edep fraction)
                        contrib_sorted = sorted(contrib, key=lambda x: (x[2], x[3]), reverse=True)
                        dominant_truth_id, dominant_truth_pdg, dominant_hit_count, dominant_edep, _, _, _ = contrib_sorted[0]
                        dominant_hit_frac = float(dominant_hit_count) / float(max(nhits_cluster, 1))
                        dominant_edep_frac = float(dominant_edep) / float(max(edep_reco_cluster, 1e-12))
                        edep_match_cluster = dominant_edep

                        pattern_mcp = (event.y[:,0]==dominant_truth_id)
                        pattern_mcp_feat = event.feat[pattern_mcp]
                        pattern_mcp_edep = pattern_mcp_feat[:,0].detach().numpy().astype(np.float64)
                        edep_mc = float(np.sum(pattern_mcp_edep))

                        pattern_mcp_match_label = event.label[pattern_mcp].detach().numpy().astype(np.float64)
                        if len(pattern_mcp_match_label) > 0:
                            mcp_label = pattern_mcp_match_label[0]

                    # Fill composition arrays per truth PDG id (aligned index among 3 arrays)
                    contrib_sorted_all = sorted(contrib, key=lambda x: (x[2], x[3]), reverse=True)
                    ncomp = min(len(contrib_sorted_all), len(d2.pdg_comp_ids))
                    d2.pdg_comp_ids[:] = 0
                    d2.pdg_comp_hits[:] = 0
                    d2.pdg_comp_hit_frac[:] = 0.0
                    d2.pdg_comp_edep_frac[:] = 0.0
                    d2.pdg_comp_edep[:] = 0.0
                    d2.pdg_comp_truth_edep[:] = 0.0
                    d2.pdg_comp_track_hits[:] = 0
                    for ip in range(ncomp):
                        _, pdg_tid, hit_count_tid, edep_tid, track_hit_count_tid, total_hit_tid, total_edep_tid = contrib_sorted_all[ip]
                        d2.pdg_comp_ids[ip] = int(pdg_tid)
                        d2.pdg_comp_hits[ip] = int(hit_count_tid)
                        d2.pdg_comp_hit_frac[ip] = float(hit_count_tid) / float(max(total_hit_tid, 1))
                        d2.pdg_comp_edep_frac[ip] = float(edep_tid) / float(max(total_edep_tid, 1e-12))
                        d2.pdg_comp_edep[ip] = float(edep_tid)
                        d2.pdg_comp_truth_edep[ip] = float(total_edep_tid)
                        d2.pdg_comp_track_hits[ip] = int(track_hit_count_tid)

                    d2.event[0] = i
                    d2.cluster[0] = cl
                    d2.nhits[0] = nhits_cluster
                    d2.ntrack_hits[0] = ntrack_hits_cluster
                    d2.cond_beta[0] = float(predicted_beta_[0]) if len(predicted_beta_) > 0 else 0.0
                    d2.cond_is_track[0] = 1 if int(cond_trackness_) != 0 else 0
                    d2.matched_truth_pdgid[0] = dominant_truth_pdg
                    d2.matched_truth_hit_frac[0] = dominant_hit_frac
                    d2.matched_truth_edep_frac[0] = dominant_edep_frac
                    d2.npdg_comp[0] = ncomp

                    d2.mcid[0] = dominant_truth_id
                    d2.mcpdg[0] = -1
                    d2.mccharge[0] = -1
                    d2.mcmass[0] = -1
                    d2.mcpx[0] = -1
                    d2.mcpy[0] = -1
                    d2.mcpz[0] = -1
                    d2.mcen[0] = -1
                    d2.mcstatus[0] = -1
                    d2.pred_edep[0] = pred_edep_
                    d2.pred_edep_cluster[0] = pred_edep_cluster_

                    if mcp_label is not None:
                        d2.mcpdg[0] = int(mcp_label[2])
                        d2.mccharge[0] = int(mcp_label[3])
                        d2.mcmass[0] = float(mcp_label[4])
                        d2.mcpx[0] = float(mcp_label[5])
                        d2.mcpy[0] = float(mcp_label[6])
                        d2.mcpz[0] = float(mcp_label[7])
                        d2.mcen[0] = np.sqrt(d2.mcmass[0]**2 + d2.mcpx[0]**2 + d2.mcpy[0]**2 + d2.mcpz[0]**2)
                        d2.mcstatus[0] = int(mcp_label[8])

                    d2.edep_reco[0] = edep_reco_cluster
                    d2.edep_mc[0] = edep_mc
                    d2.edep_match[0] = edep_match_cluster
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
                        d3.pred_edep[0] = -1 if not energyRegression else (-1 if energyRegressionWeight else prediction.pred_tracker_energy[ihit])
                        d3.pred_edep_cluster[0] = prediction.pred_cluster_energy[ihit] if energyRegression and energyRegressionCluster else -1
                        d3.pred_beta[0] = prediction.pred_betas[ihit]
                        d3.pred_alpha[0] = condensation_points[ihit]
                        d3.trackness[0] = my_feat[5]
                        d3.weight_photon[0] = prediction.pred_weight_photon[ihit] if energyRegressionWeight else 0
                        d3.weight_charged_hadron[0] = prediction.pred_weight_charged_hadron[ihit] if energyRegressionWeight else 0
                        d3.weight_neutral_hadron[0] = prediction.pred_weight_neutral_hadron[ihit] if energyRegressionWeight else 0
                        d3.weight_muon[0] = prediction.pred_weight_muon[ihit] if energyRegressionWeight else 0
                        d3.weight_electron[0] = prediction.pred_weight_electron[ihit] if energyRegressionWeight else 0

                        if (not d3.mcid[0] == -1): # skip if track does not have hit
                            t3.Fill()

                # for cl in all_cluster_ids:
                #     pattern_cluster = (clustering==cl)
                #     pattern_cluster_feat = event.feat[pattern_cluster]
                #     pattern_cluster_edep = pattern_cluster_feat[:,0].detach().numpy().astype(np.float64)
                #     edep_reco = np.sum(pattern_cluster_edep)
                #     if not pandora:
                #         predicted_beta = prediction.pred_betas[pattern_cluster]
                #         if energyRegression:
                #             predicted_energy = prediction.pred_tracker_energy[pattern_cluster]
                #             predicted_energy = predicted_energy[np.argsort(-predicted_beta)]
                #             predicted_energy_cluster = prediction.pred_cluster_energy[pattern_cluster] if energyRegressionCluster else -np.ones(1)
                #         else:
                #             predicted_energy = np.zeros(1)
                #             predicted_energy_cluster = np.zeros(1)
                #         predicted_beta = -np.sort(-predicted_beta)
                #         # print(predicted_beta[0], cond_trackness)
                #     else:
                #         predicted_energy = prediction.pred_tracker_energy[pattern_cluster]
                #         cond_trackness = 0
                #         predicted_beta = np.zeros(1)
                #     pred_edep = predicted_energy[0]                                  ## alpha
                #     pred_edep_cluster = np.sum(predicted_energy_cluster) if not pandora else 0

                d4.event[0] = i
                d4.ncluster[0] = len(all_cluster_ids)
                d4.MC_dijet_energy[0] = MC_dijet_energy
                d4.total_MC_energy_truth[0] = total_MC_energy_
                d4.total_MC_energy_pred[0] = total_MC_energy
                d4.total_predicted_energy_truth[0] = total_predicted_energy_
                d4.total_predicted_energy_pred[0] = total_predicted_energy
                t4.Fill()

                if event.jet is not None:
                    # q_en_truthBase = calc_pred_jet_energy(np.array(pred_energy_truthBase) , calc_origin_quark(np.array(jet_momentum), np.array(reco_momentum_truthBase)))
                    # q_en_predBase = calc_pred_jet_energy(np.array(pred_energy_predBase) , calc_origin_quark(np.array(jet_momentum), np.array(reco_momentum_predBase)))
                    # print(q_en_truthBase, q_en_predBase, MC_jet_energies)
                    jet_np = to_numpy(event.jet)
                    nj = int(min(jet_np.shape[0], JetData.MAX_JETS))
                    d5.n_jets[0] = nj
                    d5.jet_p4.fill(0.0)
                    if nj > 0:
                        n4 = min(4, jet_np.shape[1])
                        d5.jet_p4[:nj, :n4] = jet_np[:nj, :n4]
                    d5.event[0] = i
                    # d5.MC_jet_energy[0] = MC_jet_energies[0]
                    # d5.total_predicted_energy_truthBase[0] = q_en_truthBase[0]
                    # d5.total_predicted_energy_predBase[0] = q_en_predBase[0]
                    t5.Fill()
                # d5.event[0] = i
                # d5.MC_jet_energy[0] = MC_jet_energies[1]
                # d5.total_predicted_energy_truthBase[0] = q_en_truthBase[1]
                # d5.total_predicted_energy_predBase[0] = q_en_predBase[1]
                # t5.Fill()

            print(f"Saving to {outfile}")
            file.Write()

def main():
    print(sys.argv)
    if (len(sys.argv) < 9):
        print("Usage: save_root.py datapath ckpt outfile nstart nend timingCut input_dim output_dim pandora energyRegression momentum momentumAmp MCTpe")
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
    parser.add_argument('--event-total-energy', action='store_true', help='Use event visible energy')
    parser.add_argument('--energy-regression', action='store_true', help='Turn on energy regression term on loss function and output')
    parser.add_argument('--energy-regression-cluster', action='store_true', help='Turn on energy regression term on loss function and output (regression for neutral particle)')
    parser.add_argument('--energy-regression-weight', action='store_true', help='Turn on energy regression term on loss function and output (enegy weight loss)')
    parser.add_argument('-e','--momentum', action='store_true', help='Add momentum to GNN input')
    parser.add_argument('-ea','--momentum-amp', action='store_true', help='Add absoute momentum to GNN input')
    parser.add_argument('--mctpe', action='store_true', help='Use MC truth momentum and energy for virtual hits')
    parser.add_argument('-eb','--energy-branch', action='store_true', help='Change GNN model to bypass energy')
    parser.add_argument('--beta-d-scan', action='store_true', help='Turn on beta and diameter scan')
    parser.add_argument('--tbeta', type=float, default=0.9)
    parser.add_argument('--td', type=float, default=0.5)
    parser.add_argument('--device', type=str, default='cpu', help='Specify calculation device')
    parser.add_argument('--model-variant', type=str, default='auto', choices=['auto', 'legacy', 'multihead'], help='Select model loader: auto-detect, force legacy GravnetModel, or force multihead loader')
    parser.add_argument('--truth-clustering', action='store_true', help='Turn on MC truth clustering')
    parser.add_argument('--1tomany-clustering', action='store_true', help='Turn on combining reco-clusters')

    args = parser.parse_args()
    
    save_root(sys.argv[1],sys.argv[2],sys.argv[3],nstart=int(sys.argv[4]),nend=int(sys.argv[5]),timingCut=strtobool(sys.argv[6]),input_dim=int(sys.argv[7]), output_dim=int(sys.argv[8]), args=args)

if __name__=='__main__':
    main()
    
