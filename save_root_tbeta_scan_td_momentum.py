import sys
import numpy as np
from distutils.util import strtobool
#import evaluation_noNoise as ev
import awkward as ak
from model import get_model
from dataset import ILCDataset
from test_yielder import TestYielder
from ROOT import TFile, TTree

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

def save_root(datapath, ckpt, outfileDir, nstart=0, nend=-1, timingCut=False, input_dim=5, output_dim=3, pandora=False):
    debug = False

    thetaphi = True if input_dim == 7 else False
    print(f"Loading model from checkpoint {ckpt}")
    model = get_model(ckpt, jit=False, input_dim=input_dim,output_dim=output_dim)
    print(f"Loading data from {datapath} with {nstart=}, {nend=}, {timingCut=}")
    dataset = ILCDataset(datapath, timingCut=timingCut, thetaphi=thetaphi, test_mode=True, nstart=nstart, nend=nend, pandora=pandora)
    yielder = TestYielder(model=model, dataset=dataset)

    nmax = None if nend==-1 else nend-nstart+1

    """
    ak_feat: edep, x, y, z, time, track, charge, px, py, pz (atcalo)
        --> save edep, drop others
    ak_label: hitid, mcid, pdg, charge, mass, px, py, pz (of mcp), status
        --> save all labels
    """
    
    for tbeta_now in range(0,10):
        for td_now in range(1,10):
            tbeta = tbeta_now / 10
            td = td_now / 10
            outfile = outfileDir + '/tbeta' + format(tbeta_now, '02') + '0.root'


            print("")
            print(f"save_root()...{outfile}")
            print("")
            file = TFile(outfile,"recreate")

            t = TTree("t","tree for MCParticle")
            d = Data()
            d.setup_branch(t)

            t2 = TTree("reco","tree for reconstructed clusters")
            d2 = RecoData()
            d2.setup_branch(t2)


            #for i, (event, prediction) in enumerate(yielder.iter_pred(nmax)):
            # for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta=0.2, td=0.5, nmax=nmax, pandora=pandora)):
            for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta=tbeta, td=td, nmax=nmax, pandora=pandora, True)):
            # for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta=0.2, td=0.5, nmax=nmax)):     ## これをpandoraについてもできるようにする
            #for i, (event, prediction, clustering, matches) in enumerate(yielder.iter_matches(tbeta=0.7, td=0.5, nmax=nmax)):

                if i == nmax: break

                if i < 10 or i%100 == 0:
                    print("Event", i, "processing...")

                #ak_feat = ak.from_numpy(event.feat.numpy())
                #ak_label = ak.from_numpy(event.label.numpy())
                #beta = np.expand_dims(prediction.pred_betas, axis=1)
                #charge_track_likeness = np.expand_dims(prediction.pred_charge_track_likeness, axis=1)
                #pred = np.concatenate((beta,prediction.pred_cluster_space_coords,charge_track_likeness), axis=1)
                #ak_pred = ak.from_numpy(pred)

                matches12, matches21 = matches
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

                if (debug):
                    print(f"{all_truth_ids=}")
                    #print(f"{all_mcid=}")
                    print(f"{all_cluster_ids=}")

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
                    edep_sum = np.sum(match_edep)
                    ncluster = len(match_label)

                    edep_reco = 0
                    edep_match = 0

                    if (id in matches12.keys()):
                        reco_match = matches12[id]
                        for rid in reco_match:
                            if (matched_cluster == -1):
                                matched_cluster = rid
                            matched_ncluster += 1
                            pattern_cluster = (clustering==rid)
                            pattern_cluster_feat = event.feat[pattern_cluster]
                            pattern_cluster_edep = pattern_cluster_feat[:,0].detach().numpy().astype(np.float64)
                            edep_reco += np.sum(pattern_cluster_edep)
                            pattern_mcid_cluster = np.logical_and(pattern_mcid, pattern_cluster)
                            edep_mcid_cluster = event.feat[pattern_mcid_cluster][:,0].detach().numpy().astype(np.float64)
                            edep_match += np.sum(edep_mcid_cluster)

                    # for MC particle, take any element from the match because they should be the same
                    my_label = match_label[0]

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

                    if (not d.mcid[0] == -1): # skip if track does not have hit
                        t.Fill()

                # Iterate over reconstructed clusters
                for cl in all_cluster_ids:
                    break
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

                    d2.edep_reco[0] = edep_reco
                    d2.edep_mc[0] = edep_mcp
                    t2.Fill()
    
            print(f"Saving to {outfile}")
            file.Write()
    
def main():
    if (len(sys.argv) != 10):
        print("Usage: save_root.py datapath ckpt outfileDir nstart nend timingCut input_dim output_dim pandora")
        return
    
    save_root(sys.argv[1],sys.argv[2],sys.argv[3],nstart=int(sys.argv[4]),nend=int(sys.argv[5]),timingCut=strtobool(sys.argv[6]),input_dim=int(sys.argv[7]), output_dim=int(sys.argv[8]), pandora=strtobool(sys.argv[9]))

if __name__=='__main__':
    main()
    
