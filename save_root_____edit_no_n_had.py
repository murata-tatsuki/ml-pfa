import sys
import numpy as np
from distutils.util import strtobool
#import evaluation_noNoise as ev
import awkward as ak
from model import get_model, get_model_branch
from dataset import ILCDataset
from test_yielder_edit_no_n_had import TestYielder
from ROOT import TFile, TTree
import argparse
import torch
from sed import minimum_enclosing_sphere
from matching import matching_1to1

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
            output_dim += 4
    print(f"Loading model from checkpoint {ckpt}")
    if energy_branch:
        model = get_model_branch(ckpt, jit=False, input_dim=input_dim,output_dim=output_dim).to(device)
    else:
        model = get_model(ckpt, jit=False, input_dim=input_dim,output_dim=output_dim).to(device)
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
                            pred_weight_neutral_hadron = 0
                            pred_weight_muon = prediction.pred_weight_muon[pattern_cluster]
                            pred_weight_electron = prediction.pred_weight_electron[pattern_cluster]
                            pred_weights = pred_weight_photon + pred_weight_charged_hadron + pred_weight_neutral_hadron + pred_weight_muon + pred_weight_electron
                            predicted_energy_weight = np.array([np.sum(edeps * pred_weights)])
                            pred_photon_energy = np.array([np.sum(edeps * pred_weight_photon)])
                            pred_charged_hadron_energy = np.array([np.sum(edeps * pred_weight_charged_hadron)])
                            pred_neutral_hadron_energy = np.zeros(1)
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
                        d3.pred_edep[0] = -1 if not energyRegression else (-1 if energyRegressionWeight else prediction.pred_tracker_energy[ihit])
                        d3.pred_edep_cluster[0] = prediction.pred_cluster_energy[ihit] if energyRegression and energyRegressionCluster else -1
                        d3.pred_beta[0] = prediction.pred_betas[ihit]
                        d3.pred_alpha[0] = condensation_points[ihit]
                        d3.trackness[0] = my_feat[5]
                        d3.weight_photon[0] = prediction.pred_weight_photon[ihit] if energyRegressionWeight else 0
                        d3.weight_charged_hadron[0] = prediction.pred_weight_charged_hadron[ihit] if energyRegressionWeight else 0
                        d3.weight_neutral_hadron[0] = 0
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

                if event_energy:
                    q_en_truthBase = calc_pred_jet_energy(np.array(pred_energy_truthBase) , calc_origin_quark(np.array(jet_momentum), np.array(reco_momentum_truthBase)))
                    q_en_predBase = calc_pred_jet_energy(np.array(pred_energy_predBase) , calc_origin_quark(np.array(jet_momentum), np.array(reco_momentum_predBase)))
                    # print(q_en_truthBase, q_en_predBase, MC_jet_energies)
                    d5.event[0] = i
                    d5.MC_jet_energy[0] = MC_jet_energies[0]
                    d5.total_predicted_energy_truthBase[0] = q_en_truthBase[0]
                    d5.total_predicted_energy_predBase[0] = q_en_predBase[0]
                    t5.Fill()
                    d5.event[0] = i
                    d5.MC_jet_energy[0] = MC_jet_energies[1]
                    d5.total_predicted_energy_truthBase[0] = q_en_truthBase[1]
                    d5.total_predicted_energy_predBase[0] = q_en_predBase[1]
                    t5.Fill()

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
    parser.add_argument('--truth-clustering', action='store_true', help='Turn on MC truth clustering')
    parser.add_argument('--1tomany-clustering', action='store_true', help='Turn on combining reco-clusters')

    args = parser.parse_args()
    
    save_root(sys.argv[1],sys.argv[2],sys.argv[3],nstart=int(sys.argv[4]),nend=int(sys.argv[5]),timingCut=strtobool(sys.argv[6]),input_dim=int(sys.argv[7]), output_dim=int(sys.argv[8]), args=args)

if __name__=='__main__':
    main()
    
