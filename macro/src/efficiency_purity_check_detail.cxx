// macro to save or display efficiency and purity of the root file
// if the energy regression is conducted, truth vs predicted is also displayed
// 
// usage
// change the fileName, train_particle_type
// change to saving_canvas = true, if you want to save figures
// execute "root efficiency_purity_check.cxx" 
// 
// definition of efficiency and purity
//   double pur = edep_match / edep_reco;
//   double eff = edep_match / edep;

using namespace std;

// conditions
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_nnqq/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_Ecoef1/tbeta090td050.root");
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_nnqq/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_Ecoef1_tbeta09td05_eventTotalEnergy_moreStats.root");
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_ntau_10GeV_10/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster/tbeta090td050.root");
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_nnqq/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_Ecoef1_tbeta09td05_pandora.root");
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_ntau_10GeV_10/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_Ecoef1/tbeta090td050.root");

// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_ntau_10GeV_10/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_tbeta09td05_truthClustering.root");
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_nnqq/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_Ecoef1_tbeta09td05_eventTotalEnergy_moreStats_truthClustering.root");

// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_nnqq/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/alpha_tracker_diff_log_perCluster__sum_log_perCluster_Ecoef1_100betaSuppress_moreStats.root");

// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_nnqq/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/weight/test_truthcl.root");
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_nnqq/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/weight/test_.root");
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_neutron_1to100GeV/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/test_cond_cluster.root");
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_neutron_1to100GeV/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/weight/test_cond_weight.root");

// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_nnqq/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/neutron_finetuning.root");
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_nnqq/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/mixed_finetuning_rawTest.root");

// const string fileName = Form("/data/murata/test/test_1130.root");
// const string fileName = Form("/data/murata/test/test_1220.root");

// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_nnqq/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/test_merged.root");
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_nnqq_brems/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/tbeta090td050.root");
const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_fixed_uds/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/200GeV/tbeta090td050.root");

// const string fileName = Form("../test/test_1130.root");

const bool saving_canvas = false;
const string train_particle_type = "uds91";         // ntau_10GeV_10    uds91   ntau_10to100GeV_10
const string test_particle_type = train_particle_type;      // ntau_10GeV_10    uds91   ntau_10to100GeV_10
const bool kaon_neutron = true;
const bool jet_regression = false;
const bool pandora = false;
const bool ECluster = true;
double beta_threshold = 0;

const bool edep_weight_ = true;


const int energyMax = test_particle_type == "ntau_10GeV_10" ? 12 : (test_particle_type == "uds91" ? 50 : 100 );
const int energyMaximum = test_particle_type == "ntau_10GeV_10" ? 10 : (test_particle_type == "uds91" ? 40 : 100 );
const string test_particle_types = {"ntau_10GeV_10", "uds"};

void efficiency_purity_check_detail(){ 
    int rawfilenum = 1;

    // if(hyper_parameter && fine_tuning){ // condition check
    //     cout << "something wrong with setting boolian " << endl;
    //     abort();
    // }

    TFile *filein[rawfilenum];
    TTree *tree[rawfilenum];
    TTree *tree_pred[rawfilenum];
    TTree *tree_event[rawfilenum];
    TTree *tree_jet[rawfilenum];
    int entry_max[rawfilenum];
    int total_entry_max=0;
    string picDirectory = ".";
    
    filein[0] = new TFile(Form("%s",fileName.c_str()));
    cout << fileName << endl;
    cout << picDirectory << endl;

    for(int i=0; i<rawfilenum; i++){
        tree[i] = (TTree*) filein[i]->Get("t");
        entry_max[i] = tree[i]->GetEntries();
        tree_pred[i] = (TTree*) filein[i]->Get("prediction");
        tree_event[i] = (TTree*) filein[i]->Get("event");
        if(jet_regression) tree_jet[i] = (TTree*) filein[i]->Get("jet");
    }


    int event, hitid, mcid, truthid, mcpdg, mccharge, mcstatus, ncluster, matched_ncluster, matched_cluster, cond_track;
    double mcmass, mcpx, mcpy, mcpz, mcen, edep, edep_reco, edep_match, pred_edep, pred_edep_cluster, cond_beta, pred_alpha;

    int _event, _hitid, _mcid, _truthid, _mcpdg, _mccharge, _mcstatus, _ncluster, _matched_ncluster, _matched_cluster, _pred_alpha;
    double _mcmass, _mcpx, _mcpy, _mcpz, _mcen, _edep, _edep_reco, _edep_match, _pred_edep, _pred_edep_cluster, _pred_beta;

    double _MC_jet_energy, _total_predicted_energy_truthBase, _total_predicted_energy_predBase;



    // TFile fileout("result_test_pandora.root","RECREATE");
    TFile fileout("result/result_test.root","RECREATE");


    const int nParticle = kaon_neutron ? 5 : 3;
    string particleNames_base[5] = {"electron", "pion", "photon", "neutron", "K0"};
    vector<int> particledgValues_base = {11,-11, 211,-211, 22, 2112, 130};
    vector<int> particledgValues_itr_base = {0,0, 1,1, 2, 3, 4};
    vector<int> particledgValues_base_ = {11,-11, 211,-211, 22};
    vector<int> particledgValues_itr_base_ = {0,0, 1,1, 2};

    string particleNames[nParticle];
    vector<int> particledgValues = kaon_neutron ? particledgValues_base : particledgValues_base_;
    vector<int> particledgValues_itr = kaon_neutron ? particledgValues_itr_base : particledgValues_itr_base_;
    for(int ip=0;ip<nParticle;ip++) particleNames[ip] = particleNames_base[ip];

    // const int nParticle = 4;
    // string particleNames[4] = {"electron", "pion", "photon", "neutral particle"};
    // vector<int> particledgValues = {11,-11, 211,-211, 22, 2112, 130};
    // vector<int> particledgValues_itr = {0,0, 1,1, 2, 3, 3};

    double Eres_range = pandora ? 0.05 : 1.5;
    // double Eres_range = 1.5;
    int Eres_nbin = 1200;
    double E_res_binWidth = Eres_range*2/Eres_nbin;
    double Eres_fit_range = 0.5;
    int Eres_fitbin_lower = Eres_range>Eres_fit_range ? (Eres_range-Eres_fit_range)/E_res_binWidth : 0;
    int Eres_fitbin_upper = Eres_range>Eres_fit_range ? Eres_nbin - (Eres_range-Eres_fit_range)/E_res_binWidth : -1;
    int rebin_factor = 0.025 / E_res_binWidth;

    const int nEnergy_jet = 12;
    
    const int nEnergy = 10;
    const double energy_interval = energyMaximum / nEnergy;
    TH1F *purity[nParticle];
    TH1F *purity_energy[nParticle][nEnergy];
    TH1F *purity_energy_normalize[nParticle][nEnergy];
    TH2F *purity2d[nParticle];
    TH1F *efficiency[nParticle];
    TH1F *efficiency_energy[nParticle][nEnergy];
    TH1F *efficiency_energy_normalize[nParticle][nEnergy];
    TH2F *efficiency2d[nParticle];
    TH1F *energy[nParticle];
    TH1F *MCtruth_energy[nParticle];
    TH1F *energy_diff[nParticle];
    TH1F *energy_diff_per_energy[nParticle][nEnergy];
    TH2F *energy2d[nParticle];
    TH2F *clusterenergy2d[nParticle];
    TH1F *energy_resolution[nParticle];
    TH1F *energy_resolution_per_energy[nParticle][nEnergy];
    TH1F *energy_resolution_per_energy_cluster[nParticle][nEnergy];
    TH2F *eff_vs_Ediff[nParticle];
    TH2F *pur_vs_Ediff[nParticle];
    TH2F *condbeta_vs_eff[nParticle];
    TH2F *condbeta_vs_pur[nParticle];
    TH2F *condbeta_vs_Ediff[nParticle];
    TH2F *condbeta_vs_mcen[nParticle];
    TH2F *condbeta_vs_Ediff_track[nParticle];
    TH2F *condbeta_vs_mcen_track[nParticle];
    TH2F *condbeta_vs_Ediff_nottrack[nParticle];
    TH2F *condbeta_vs_mcen_nottrack[nParticle];
    for(int ip=0;ip<nParticle;ip++){
        purity[ip] = new TH1F(Form("purity_%d",ip), Form("%s purity (MC energy>1 GeV)",particleNames[ip].c_str()), 101,0,1.01);
        purity[ip]->SetXTitle("purity (edep_match/edep_reco)");
        purity2d[ip] = new TH2F(Form("purity2d_%d",ip), Form("%s purity",particleNames[ip].c_str()), 101,0,1.01, 24,0,12);
        purity2d[ip]->SetXTitle("purity (edep_match/edep_reco)");
        purity2d[ip]->SetYTitle("MC energy (edep)");
        efficiency[ip] = new TH1F(Form("efficiency_%d",ip), Form("%s efficiency (MC energy>1 GeV)",particleNames[ip].c_str()), 101,0,1.01);
        efficiency[ip]->SetXTitle("efficiency (edep_match/edep)");
        efficiency2d[ip] = new TH2F(Form("efficiency2d_%d",ip), Form("%s efficiency",particleNames[ip].c_str()), 101,0,1.01, 24,0,12);
        efficiency2d[ip]->SetXTitle("efficiency (edep_match/edep)");
        efficiency2d[ip]->SetYTitle("MC energy (edep)");
        energy[ip] = new TH1F(Form("energy_%d",ip), Form("%s predicted energy (MC energy>1 GeV)",particleNames[ip].c_str()), energyMax*10,0,energyMax);
        MCtruth_energy[ip] = new TH1F(Form("MCtruth_energy_%d",ip), Form("%s MCtruth energy",particleNames[ip].c_str()), energyMax*10,0,energyMax);
        energy_diff[ip] = new TH1F(Form("energy_diff_%d",ip), Form("%s (MC energy>1 GeV);predicted energy - MC truth energy",particleNames[ip].c_str()), energyMax*10,-energyMax/2,energyMax/2);
        energy2d[ip] = new TH2F(Form("energy2d_%d",ip), Form("%s energy",particleNames[ip].c_str()), energyMax*10,0,energyMax, energyMax*10,0,energyMax);
        energy2d[ip]->SetXTitle("MC truth energy");
        energy2d[ip]->SetYTitle("predicted energy");
        clusterenergy2d[ip] = new TH2F(Form("clusterenergy2d_%d",ip), Form("%s cluster energy",particleNames[ip].c_str()), energyMax*10,0,energyMax, energyMax*10,0,energyMax);
        clusterenergy2d[ip]->SetXTitle("MC truth energy");
        clusterenergy2d[ip]->SetYTitle("predicted cluster energy");
        eff_vs_Ediff[ip] = new TH2F(Form("eff_vs_Ediff_%d",ip), Form("%s",particleNames[ip].c_str()), 100,0,1, energyMax*20,-energyMax,energyMax);
        eff_vs_Ediff[ip]->SetXTitle("efficiency");
        eff_vs_Ediff[ip]->SetYTitle("predicted cluster energy - true energy");
        pur_vs_Ediff[ip] = new TH2F(Form("pur_vs_Ediff%d",ip), Form("%s",particleNames[ip].c_str()), 100,0,1, energyMax*20,-energyMax,energyMax);
        pur_vs_Ediff[ip]->SetXTitle("purity");
        pur_vs_Ediff[ip]->SetYTitle("predicted cluster energy - true energy");
        condbeta_vs_eff[ip] = new TH2F(Form("condbeta_vs_eff_%d",ip), Form("%s",particleNames[ip].c_str()), 100,0,1, 100,0,1);
        condbeta_vs_eff[ip]->SetXTitle("condensation beta");
        condbeta_vs_eff[ip]->SetYTitle("efficiency");
        condbeta_vs_pur[ip] = new TH2F(Form("condbeta_vs_pur_%d",ip), Form("%s",particleNames[ip].c_str()), 100,0,1, 100,0,1);
        condbeta_vs_pur[ip]->SetXTitle("condensation beta");
        condbeta_vs_pur[ip]->SetYTitle("purity");
        condbeta_vs_Ediff[ip] = new TH2F(Form("condbeta_vs_Ediff_%d",ip), Form("%s",particleNames[ip].c_str()), 100,0,1, energyMax*20,-energyMax,energyMax);
        condbeta_vs_Ediff[ip]->SetXTitle("condensation beta");
        condbeta_vs_Ediff[ip]->SetYTitle("predicted energy - true energy");
        condbeta_vs_mcen[ip] = new TH2F(Form("condbeta_vs_mcen_%d",ip), Form("%s",particleNames[ip].c_str()), 100,0,1, energyMax*10,0,energyMax);
        condbeta_vs_mcen[ip]->SetXTitle("condensation beta");
        condbeta_vs_mcen[ip]->SetYTitle("true energy");
        condbeta_vs_Ediff_track[ip] = new TH2F(Form("condbeta_vs_Ediff_track_%d",ip), Form("%s",particleNames[ip].c_str()), 100,0,1, energyMax*20,-energyMax,energyMax);
        condbeta_vs_Ediff_track[ip]->SetXTitle("condensation beta");
        condbeta_vs_Ediff_track[ip]->SetYTitle("predicted energy - true energy");
        condbeta_vs_mcen_track[ip] = new TH2F(Form("condbeta_vs_mcen_track_%d",ip), Form("%s",particleNames[ip].c_str()), 100,0,1, energyMax*10,0,energyMax);
        condbeta_vs_mcen_track[ip]->SetXTitle("condensation beta");
        condbeta_vs_mcen_track[ip]->SetYTitle("true energy");
        condbeta_vs_Ediff_nottrack[ip] = new TH2F(Form("condbeta_vs_Ediff_nottrack_%d",ip), Form("%s",particleNames[ip].c_str()), 100,0,1, energyMax*20,-energyMax,energyMax);
        condbeta_vs_Ediff_nottrack[ip]->SetXTitle("condensation beta");
        condbeta_vs_Ediff_nottrack[ip]->SetYTitle("predicted energy - true energy");
        condbeta_vs_mcen_nottrack[ip] = new TH2F(Form("condbeta_vs_mcen_nottrack_%d",ip), Form("%s",particleNames[ip].c_str()), 100,0,1, energyMax*10,0,energyMax);
        condbeta_vs_mcen_nottrack[ip]->SetXTitle("condensation beta");
        condbeta_vs_mcen_nottrack[ip]->SetYTitle("true energy");
        energy_resolution[ip] = new TH1F(Form("energy_resolution_%d",ip), Form("%s (truth energy>1 GeV);(predicted - truth) / truth",particleNames[ip].c_str()), energyMax*10,-energyMax/2,energyMax/2);
        for(int ie=0;ie<nEnergy;ie++){
            string title = ie==0 ? Form("%s purity",particleNames[ip].c_str()) : Form("%s purity %d-%d GeV",particleNames[ip].c_str(),ie,ie+1);
            purity_energy[ip][ie] = new TH1F(Form("purity_%d_%d",ip,ie), title.c_str(), 101,0,1.01);
            purity_energy[ip][ie]->SetXTitle("purity (edep_match/edep_reco)");
            purity_energy_normalize[ip][ie] = new TH1F(Form("purity_norm_%d_%d",ip,ie), title.c_str(), 101,0,1.01);
            purity_energy_normalize[ip][ie]->SetXTitle("purity (edep_match/edep_reco)");
            title = ie==0 ? Form("%s efficiency",particleNames[ip].c_str()) : Form("%s efficiency %d-%d GeV",particleNames[ip].c_str(),ie,ie+1);
            efficiency_energy[ip][ie] = new TH1F(Form("efficiency_%d_%d",ip,ie), title.c_str(), 101,0,1.01);
            efficiency_energy[ip][ie]->SetXTitle("efficiency (edep_match/edep)");
            efficiency_energy_normalize[ip][ie] = new TH1F(Form("efficiency_normalize_%d_%d",ip,ie), title.c_str(), 101,0,1.01);
            efficiency_energy_normalize[ip][ie]->SetXTitle("efficiency (edep_match/edep)");

            title = ie==0 ? Form("%s ;predicted energy - MC truth energy",particleNames[ip].c_str()) : Form("%s (%d-%d GeV);predicted energy - MC truth energy",particleNames[ip].c_str(),(int)(ie*energy_interval),(int)((ie+1)*energy_interval));
            // energy_diff_per_energy[ip][ie] = new TH1F(Form("energy_diff_per_energy%d_%d",ip,ie), title.c_str(), 120,-6,6);
            energy_diff_per_energy[ip][ie] = new TH1F(Form("energy_diff_per_energy%d_%d",ip,ie), title.c_str(), Eres_nbin,-Eres_range,Eres_range);
            title = ie==0 ? Form("%s ;(predicted - truth) / truth",particleNames[ip].c_str()) : Form("%s (%d-%d GeV);(predicted - truth) / truth",particleNames[ip].c_str(),(int)(ie*energy_interval),(int)((ie+1)*energy_interval));
            // energy_resolution_per_energy[ip][ie] = new TH1F(Form("energy_resolution_per_energy_%d_%d",ip,ie), title.c_str(), 120,-6,6);
            energy_resolution_per_energy[ip][ie] = new TH1F(Form("energy_resolution_per_energy_%d_%d",ip,ie), title.c_str(), Eres_nbin,-Eres_range,Eres_range);
            energy_resolution_per_energy_cluster[ip][ie] = new TH1F(Form("energy_resolution_per_energy_cluster_%d_%d",ip,ie), title.c_str(), Eres_nbin,-Eres_range,Eres_range);
        }
    }
    TH2F *jet_energy2d = new TH2F(Form("jet_energy2d"), Form(";MC jet energy;predicted jet energy"), 1200,0,120,1200,0,120);
    TH1F *jet_energy_resolution_per_energy[nEnergy_jet];
    for(int ie=0;ie<nEnergy_jet;ie++){
        string title = ie==0 ? Form("jet;(predicted - truth) / truth") : Form("jet (%d-%d GeV);(predicted - truth) / truth",(int)(ie*energy_interval),(int)((ie+1)*energy_interval));
        jet_energy_resolution_per_energy[ie] = new TH1F(Form("energy_resolution_per_energy_%d",ie), title.c_str(), 400,-0.5,0.5);
    }


    // data をとってきてる
    for(int irawfile=0; irawfile<rawfilenum; irawfile++){
        if(rawfilenum>1) cout << irawfile << "/" << rawfilenum << endl;

        tree[irawfile]->SetBranchAddress("event", &event);
        tree[irawfile]->SetBranchAddress("hitid", &hitid);
        tree[irawfile]->SetBranchAddress("mcid", &mcid);
        tree[irawfile]->SetBranchAddress("mcpdg", &mcpdg);
        tree[irawfile]->SetBranchAddress("mcen", &mcen);
        tree[irawfile]->SetBranchAddress("edep", &edep);
        tree[irawfile]->SetBranchAddress("edep_reco", &edep_reco);
        tree[irawfile]->SetBranchAddress("edep_match", &edep_match);
        tree[irawfile]->SetBranchAddress("pred_edep", &pred_edep);
        tree[irawfile]->SetBranchAddress("pred_edep_cluster", &pred_edep_cluster);
        tree[irawfile]->SetBranchAddress("cond_beta", &cond_beta);
        tree[irawfile]->SetBranchAddress("cond_track", &cond_track);

        for(int ientry=0; ientry<entry_max[irawfile]; ientry++){
            tree[irawfile]->GetEntry(ientry);

            if(edep<=0 || edep_reco<=0 || edep_match<0) continue;
            if(cond_beta<beta_threshold) continue;
            auto result = find(particledgValues.begin(), particledgValues.end(), mcpdg);
            if(result == particledgValues.end()) continue;
            int itr = particledgValues_itr[distance(particledgValues.begin(), result)];

            double pur = edep_match / edep_reco;
            double eff = edep_match / edep;

            if(edep>1) purity[itr]->Fill(pur);
            purity2d[itr]->Fill(pur,edep);
            if(edep>1) efficiency[itr]->Fill(eff);
            efficiency2d[itr]->Fill(eff,edep);
            int energy_itr = mcen / energy_interval;
            if(energy_itr<nEnergy){
                purity_energy[itr][energy_itr]->Fill(pur);
                purity_energy_normalize[itr][energy_itr]->Fill(pur);
                efficiency_energy[itr][energy_itr]->Fill(eff);
                efficiency_energy_normalize[itr][energy_itr]->Fill(eff);
            }

            energy[itr]->Fill(pred_edep);
            MCtruth_energy[itr]->Fill(mcen);
            energy_diff[itr]->Fill(pred_edep - mcen);
            energy2d[itr]->Fill(mcen,pred_edep);
            clusterenergy2d[itr]->Fill(mcen,pred_edep_cluster);
            energy_resolution[itr]->Fill( (pred_edep - mcen) / mcen );

            int itr_energy = mcen / energy_interval;
            // cout << itr_energy << ", " << mcen << ", " << energy_interval << endl;
            if(itr_energy<nEnergy && pred_edep>0.1){
                energy_diff_per_energy[itr][itr_energy]->Fill(pred_edep - mcen);
                energy_resolution_per_energy[itr][itr_energy]->Fill( (pred_edep - mcen) / mcen );
            }
            if(itr_energy<nEnergy && pred_edep_cluster>0.1){
                energy_resolution_per_energy_cluster[itr][itr_energy]->Fill( (pred_edep_cluster - mcen) / mcen );
            }

            if(edep>1) eff_vs_Ediff[itr]->Fill(eff,pred_edep_cluster-mcen);
            if(edep>1) pur_vs_Ediff[itr]->Fill(pur,pred_edep_cluster-mcen);
            if(edep>1) condbeta_vs_eff[itr]->Fill(cond_beta,eff);
            if(edep>1) condbeta_vs_pur[itr]->Fill(cond_beta,pur);
            condbeta_vs_Ediff[itr]->Fill(cond_beta,pred_edep-mcen);
            if(cond_track==1) condbeta_vs_Ediff_track[itr]->Fill(cond_beta,pred_edep-mcen);
            else condbeta_vs_Ediff_nottrack[itr]->Fill(cond_beta,pred_edep-mcen);
            condbeta_vs_mcen[itr]->Fill(cond_beta,mcen);
            if(cond_track==1) condbeta_vs_mcen_track[itr]->Fill(cond_beta,mcen);
            else condbeta_vs_mcen_nottrack[itr]->Fill(cond_beta,mcen);
        }

      
        // tree_pred[irawfile]->SetBranchAddress("event", &_event);
        // tree_pred[irawfile]->SetBranchAddress("hitid", &_hitid);
        // tree_pred[irawfile]->SetBranchAddress("mcid", &_mcid);
        tree_pred[irawfile]->SetBranchAddress("mcpdg", &_mcpdg);
        tree_pred[irawfile]->SetBranchAddress("mcen", &_mcen);
        // tree_pred[irawfile]->SetBranchAddress("edep", &_edep);
        tree_pred[irawfile]->SetBranchAddress("pred_edep", &_pred_edep);
        tree_pred[irawfile]->SetBranchAddress("pred_beta", &_pred_beta);
        tree_pred[irawfile]->SetBranchAddress("pred_alpha", &_pred_alpha);
        for(int ientry=0; ientry<tree_pred[irawfile]->GetEntries(); ientry++){
            tree_pred[irawfile]->GetEntry(ientry);
            if(_pred_alpha==1) continue;

            auto result = find(particledgValues.begin(), particledgValues.end(), _mcpdg);
            if(result == particledgValues.end()) continue;
            int itr = particledgValues_itr[distance(particledgValues.begin(), result)];
            // if(_pred_alpha==1) condbeta_vs_Ediff[itr]->Fill(_pred_beta,_pred_edep-_mcen);
            // if(_pred_alpha==1) condbeta_vs_mcen[itr]->Fill(_pred_beta,_mcen);
        }


        if(jet_regression){
            tree_jet[irawfile]->SetBranchAddress("MC_jet_energy", &_MC_jet_energy);
            tree_jet[irawfile]->SetBranchAddress("total_predicted_energy_truthBase", &_total_predicted_energy_truthBase);
            tree_jet[irawfile]->SetBranchAddress("total_predicted_energy_predBase", &_total_predicted_energy_predBase);
            for(int ientry=0; ientry<tree_jet[irawfile]->GetEntries(); ientry++){
                tree_jet[irawfile]->GetEntry(ientry);

                jet_energy2d->Fill(_MC_jet_energy, _total_predicted_energy_predBase);
                int itr_energy = _MC_jet_energy/10;
                jet_energy_resolution_per_energy[itr_energy]->Fill( (_total_predicted_energy_predBase - _MC_jet_energy) / _MC_jet_energy );
            }
        }

    }
    

    
    // gStyle->SetStatX(0.35);
    gStyle->SetOptStat("rme");
    gStyle->SetStatX(0.55);
    gStyle->SetStatY(0.9);
    gStyle->SetStatH(0.3);
    gStyle->SetStatW(0.4);
    // legends をもう少し大きくする

    TCanvas *compare = new TCanvas("compare","compare",1);
    compare->Divide(nParticle,2);
    for(int ip=0;ip<nParticle*2;ip++){
        // TPaveStats *s = (TPaveStats*) gPad->GetPrimitive("stats"); s->SetTextSize(0.1); s->SetX1NDC(0.5); s->SetY1NDC(0.5);
        compare->cd(ip+1);
        gPad->SetLogy();
        if(ip<nParticle) efficiency[ip]->Draw();
        else purity[ip-nParticle]->Draw();
    }

    TCanvas *compare_energy = new TCanvas("compare_energy","compare_energy",1);
    compare_energy->Divide(nParticle,2);
    TLegend *legend_comp_e[nParticle][2];
    double eff_mean[nParticle][nEnergy];
    double eff_mean_error[nParticle][nEnergy];
    double pur_mean[nParticle][nEnergy];
    double pur_mean_error[nParticle][nEnergy];
    for(int ip=0;ip<nParticle*2;ip++){
        compare_energy->cd(ip+1);
        string par_type = ip/nParticle==0 ? " efficiency" : " purity";
        cout << particleNames[ip%nParticle] << par_type.c_str() << endl;
        gPad->SetLogy();
        legend_comp_e[ip%nParticle][ip/nParticle] = new TLegend( 0.4, 0.6, 0.8, 0.9);
        for(int ie=0;ie<nEnergy;ie++){
            string drawOption = ie==0 ? "HIST" : "same HIST";
            int colorId = ie<9 ? ie+1 : ie+2;
            string led_com = "";
            if(ip<nParticle){
                // efficiency_energy[ip][ie]->Rebin(2);
                efficiency_energy_normalize[ip][ie]->SetLineColor(colorId);
                efficiency_energy_normalize[ip][ie]->SetMarkerColor(colorId);
                int scaling = efficiency_energy_normalize[ip][ie]->Integral() == 0 ? 1 : efficiency_energy_normalize[ip][ie]->Integral();
                efficiency_energy_normalize[ip][ie]->Scale(1.0/scaling);
                // efficiency_energy_normalize[ip][ie]->SetMaximum(yaxis_height[ip]*2);
                efficiency_energy_normalize[ip][ie]->Draw(drawOption.c_str());
                eff_mean[ip][ie] = efficiency_energy[ip][ie]->GetMean();
                eff_mean_error[ip][ie] = efficiency_energy[ip][ie]->GetMeanError();
                led_com = Form("%d-%d GeV mean:%f StdErr:%f",(int)(ie*energy_interval),(int)((ie+1)*energy_interval), efficiency_energy[ip][ie]->GetMean(), efficiency_energy[ip][ie]->GetMeanError());
                // led_com = Form("%d-%d GeV mean:%f StdDev:%f",(int)(ie*energy_interval),(int)((ie+1)*energy_interval), efficiency_energy_normalize[ip][ie]->GetMean(), efficiency_energy_normalize[ip][ie]->GetStdDev());
                legend_comp_e[ip%nParticle][ip/nParticle]->AddEntry(efficiency_energy_normalize[ip][ie], led_com.c_str() , "l");
                // legend_comp_e[ip%nParticle][ip/nParticle]->AddEntry(efficiency_energy_normalize[ip][ie], Form("%d-%d GeV",(int)(ie*energy_interval),(int)((ie+1)*energy_interval)), "l");
                legend_comp_e[ip%nParticle][ip/nParticle]->Draw("same");
                cout << "   " << led_com.c_str() << endl;
            } else {
                // purity_energy[ip-nParticle][ie]->Rebin(2);
                purity_energy_normalize[ip-nParticle][ie]->SetLineColor(colorId);
                purity_energy_normalize[ip-nParticle][ie]->SetMarkerColor(colorId);
                int scaling = purity_energy_normalize[ip-nParticle][ie]->Integral() == 0 ? 1 : purity_energy_normalize[ip-nParticle][ie]->Integral();
                purity_energy_normalize[ip-nParticle][ie]->Scale(1.0/scaling);
                // purity_energy_normalize[ip-nParticle][ie]->SetMaximum(yaxis_height[ip]*2);
                purity_energy_normalize[ip-nParticle][ie]->Draw(drawOption.c_str());
                // compare_energy->cd(ip+1)->BuildLegend();
                pur_mean[ip-nParticle][ie] = purity_energy[ip-nParticle][ie]->GetMean();
                pur_mean_error[ip-nParticle][ie] = purity_energy[ip-nParticle][ie]->GetMeanError();
                led_com = Form("%d-%d GeV mean:%f StdErr:%f",(int)(ie*energy_interval),(int)((ie+1)*energy_interval), pur_mean[ip-nParticle][ie], pur_mean_error[ip-nParticle][ie]);
                // led_com = Form("%d-%d GeV mean:%f StdDev:%f",(int)(ie*energy_interval),(int)((ie+1)*energy_interval), purity_energy_normalize[ip-nParticle][ie]->GetMean(), purity_energy_normalize[ip-nParticle][ie]->GetStdDev());
                legend_comp_e[ip%nParticle][ip/nParticle]->AddEntry(purity_energy_normalize[ip-nParticle][ie], led_com.c_str() , "l");
                // legend_comp_e[ip%nParticle][ip/nParticle]->AddEntry(purity_energy_normalize[ip-nParticle][ie], Form("%d-%d GeV",(int)(ie*energy_interval),(int)((ie+1)*energy_interval)), "l");
                legend_comp_e[ip%nParticle][ip/nParticle]->Draw("same");
                cout << "   " << led_com.c_str() << endl;
            }
        }
    }

    TCanvas *compare_per_energy = new TCanvas("compare_per_energy","compare_per_energy",1);
    compare_per_energy->Divide(2,1);
    compare_per_energy->cd();
    TGraphErrors *eff_per_energy[nParticle];
    TGraphErrors *pur_per_energy[nParticle];
    TLegend *legend_comp_per_e = new TLegend( 0.4, 0.1, 0.8, 0.4);;
    for(int ip=0;ip<nParticle;ip++){
        eff_per_energy[ip] = new TGraphErrors();
        pur_per_energy[ip] = new TGraphErrors();
        for(int ie=0;ie<nEnergy;ie++){
            eff_per_energy[ip]->SetPoint(ie, (ie+0.5)*energy_interval, eff_mean[ip][ie]);
            eff_per_energy[ip]->SetPointError(ie, (0.5)*energy_interval, eff_mean_error[ip][ie]);
            pur_per_energy[ip]->SetPoint(ie, (ie+0.5)*energy_interval, pur_mean[ip][ie]);
            pur_per_energy[ip]->SetPointError(ie, (0.5)*energy_interval, pur_mean_error[ip][ie]);
        }
        eff_per_energy[ip]->SetLineColor(ip+1);
        eff_per_energy[ip]->SetMarkerColor(ip+1);
        pur_per_energy[ip]->SetLineColor(ip+1);
        pur_per_energy[ip]->SetMarkerColor(ip+1);
        legend_comp_per_e->AddEntry(eff_per_energy[ip], Form("%s", particleNames[ip].c_str()) , "l");

        string drawOption = ip==0 ? "AP" : "P";
        compare_per_energy->cd(1);
        eff_per_energy[ip]->SetMaximum(1);
        eff_per_energy[ip]->SetMinimum(0);
        eff_per_energy[ip]->Draw(drawOption.c_str());
        compare_per_energy->cd(2);
        pur_per_energy[ip]->SetMaximum(1);
        pur_per_energy[ip]->SetMinimum(0);
        pur_per_energy[ip]->Draw(drawOption.c_str());
    }
    compare_per_energy->cd(1);
    legend_comp_per_e->Draw("same");
    compare_per_energy->cd(2);
    legend_comp_per_e->Draw("same");

    /*
    TCanvas *canvas_energy = new TCanvas("canvas_energy","canvas_energy",1);
    canvas_energy->Divide(nParticle,2);
    canvas_energy->cd();
    for(int ip=0;ip<nParticle*2;ip++){
        canvas_energy->cd(ip+1);
        // gPad->SetLogy();
        // if(ip<nParticle) energy[ip]->Draw();
        if(ip<nParticle){
            // gStyle->SetOptStat(0);
            energy2d[ip]->SetStats(0);
            energy2d[ip]->Draw("colz");
        }
        else {
            // gPad->SetLogy();
            // energy_diff[ip-nParticle]->SetStats(0);
            // energy_diff[ip-nParticle]->Draw();
            clusterenergy2d[ip-nParticle]->SetStats(0);
            clusterenergy2d[ip-nParticle]->Draw("colz");
        }
        // else MCtruth_energy[ip-nParticle]->Draw();
    }
    */

    /*
    TCanvas *canvas_cluster_energy = new TCanvas("canvas_cluster_energy","canvas_cluster_energy",1);
    canvas_cluster_energy->Divide(nParticle,2);
    canvas_cluster_energy->cd();
    for(int ip=0;ip<nParticle*2;ip++){
        canvas_cluster_energy->cd(ip+1);
        gPad->SetGrid(1,1);
        if(ip<nParticle){
            eff_vs_Ediff[ip]->SetStats(0);
            eff_vs_Ediff[ip]->Draw("colz");
        }
        else {
            pur_vs_Ediff[ip-nParticle]->SetStats(0);
            pur_vs_Ediff[ip-nParticle]->Draw("colz");
        }
    }
    */


    TLegend *legend[nParticle];
    TLegend *legend_resolution[nParticle];
    int Ymaximum[nParticle];
    int Ymaximum_resolution[nParticle];
    for(int ip=0;ip<nParticle;ip++){
        Ymaximum[ip] = 0;    
        Ymaximum_resolution[ip] = 0;
        for(int ie=0;ie<nEnergy;ie++){
            Ymaximum[ip] = energy_diff_per_energy[ip][ie]->GetMaximum() > Ymaximum[ip] ? energy_diff_per_energy[ip][ie]->GetMaximum() : Ymaximum[ip];
            Ymaximum_resolution[ip] = energy_resolution_per_energy[ip][ie]->GetMaximum() > Ymaximum_resolution[ip] ? energy_resolution_per_energy[ip][ie]->GetMaximum() : Ymaximum_resolution[ip];
        }
    }
    TCanvas *canvas_energy_scan = new TCanvas("canvas_energy_scan","canvas_energy_scan",1);
    canvas_energy_scan->Divide(nParticle,2);
    canvas_energy_scan->cd();
    for(int ip=0;ip<nParticle;ip++){
        legend[ip] = new TLegend( 0.1, 0.6, 0.5, 0.9) ;
        canvas_energy_scan->cd(ip+1);
        // gPad->SetLogy();
        for(int ie=0;ie<nEnergy;ie++){
            // string drawOption = ie==0 ? "" : "same";
            legend[ip]->AddEntry(energy_diff_per_energy[ip][ie], Form("%d-%d GeV mean:%f StdDev:%f",(int)(ie*energy_interval),(int)((ie+1)*energy_interval), energy_diff_per_energy[ip][ie]->GetMean(), energy_diff_per_energy[ip][ie]->GetStdDev()) , "l");
            string drawOption = ie==0 ? "HIST" : "same HIST";
            int colorId = ie<9 ? ie+1 : ie+2;
            energy_diff_per_energy[ip][ie]->SetLineColor(colorId);
            energy_diff_per_energy[ip][ie]->SetMarkerColor(colorId);
            energy_diff_per_energy[ip][ie]->SetMaximum(Ymaximum[ip]*1.3);
            // energy_diff_per_energy[ip][ie]->Scale(1./efficiency_energy[ip][ie]->GetEntries());
            energy_diff_per_energy[ip][ie]->SetStats(0);
            energy_diff_per_energy[ip][ie]->Draw(drawOption.c_str());
        }
        legend[ip]->SetTextSize(0.03);
        legend[ip]->SetFillStyle(0);
        legend[ip]->Draw("same");

        legend_resolution[ip] = new TLegend( 0.1, 0.6, 0.5, 0.9) ;
        canvas_energy_scan->cd(ip+nParticle+1);
        for(int ie=0;ie<nEnergy;ie++){
            // string drawOption_legend = ie==0 ? "" : "same";
            legend_resolution[ip]->AddEntry(energy_resolution_per_energy[ip][ie], Form("%d-%d GeV mean:%f StdDev:%f",(int)(ie*energy_interval),(int)((ie+1)*energy_interval), energy_resolution_per_energy[ip][ie]->GetMean(), energy_resolution_per_energy[ip][ie]->GetStdDev()) , "l");
            // double res = energy_resolution_per_energy[ip][ie]->GetStdDev() / energy_resolution_per_energy[ip][ie]->GetMean();

            string drawOption = ie==0 ? "HIST" : "same HIST";
            int colorId = ie<9 ? ie+1 : ie+2;
            energy_resolution_per_energy[ip][ie]->SetLineColor(colorId);
            energy_resolution_per_energy[ip][ie]->SetMarkerColor(colorId);
            energy_resolution_per_energy[ip][ie]->SetMaximum(Ymaximum_resolution[ip]*1.3);
            // energy_resolution_per_energy[ip][ie]->Scale(1./efficiency_energy[ip][ie]->GetEntries());
            energy_resolution_per_energy[ip][ie]->SetStats(0);
            energy_resolution_per_energy[ip][ie]->Draw(drawOption.c_str());
        }
        legend_resolution[ip]->SetTextSize(0.03);
        legend_resolution[ip]->SetFillStyle(0);
        legend_resolution[ip]->Draw("same");
    }


    TCanvas *canvas_energy_resolution_scan = new TCanvas("canvas_energy_resolution_scan","canvas_energy_resolution_scan",1400,500);
    canvas_energy_resolution_scan->Divide(nParticle,1);
    // TCanvas *canvas_energy_resolution_scan = new TCanvas("canvas_energy_resolution_scan","canvas_energy_resolution_scan",1);
    canvas_energy_resolution_scan->cd();
    double resolution_rms[nParticle][nEnergy];
    double resolution_sigma[nParticle][nEnergy];
    double resolution_sigma_error[nParticle][nEnergy];
    TF1 *gaus = new TF1("gaus", "gaus", -3,3);
    TGraphErrors *energy_resolution_rms[nParticle];
    TGraphErrors *energy_resolution_sigma[nParticle];
    TLegend *legend_res = new TLegend( 0.5, 0.6, 0.9, 0.9);
    // legend_res->AddEntry(energy_resolution_rms[0], Form("rms") , "l");
    // legend_res->AddEntry(energy_resolution_sigma[0], Form("gaussian sigma") , "l");
    for(int ip=0;ip<nParticle;ip++){
        cout << particleNames[ip] << endl;
        energy_resolution_rms[ip] = new TGraphErrors();
        energy_resolution_rms[ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
        energy_resolution_rms[ip]->GetXaxis()->SetTitle("GeV");
        energy_resolution_rms[ip]->GetYaxis()->SetTitle("simga of (pred-truth)/truth");
        energy_resolution_rms[ip]->SetLineColor(kRed);
        energy_resolution_rms[ip]->SetMarkerColor(kRed);
        energy_resolution_rms[ip]->SetMaximum(0.5);
        energy_resolution_rms[ip]->SetMinimum(0);
        energy_resolution_sigma[ip] = new TGraphErrors();
        energy_resolution_sigma[ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
        energy_resolution_sigma[ip]->SetLineColor(kBlue);
        energy_resolution_sigma[ip]->SetMarkerColor(kBlue);
        energy_resolution_sigma[ip]->SetMaximum(0.5);
        energy_resolution_sigma[ip]->SetMinimum(0);
        if(ip==0) { 
            legend_res->AddEntry(energy_resolution_rms[0], Form("rms") , "l");
            legend_res->AddEntry(energy_resolution_sigma[0], Form("gaussian sigma") , "l");
        }
        canvas_energy_resolution_scan->cd(ip+1);
        for(int ie=0;ie<nEnergy;ie++){
            // cout << energy_resolution_per_energy[ip][ie]->GetStdDev() << "    ";
            energy_resolution_per_energy[ip][ie]->SetAxisRange(-Eres_fit_range,Eres_fit_range);
            // cout << energy_resolution_per_energy[ip][ie]->GetStdDev() << endl;
            resolution_rms[ip][ie] = energy_resolution_per_energy[ip][ie]->GetStdDev();
            // // // // // if(energy_resolution_per_energy[ip][ie]->Integral(Eres_fitbin_lower,Eres_fitbin_upper)>30){
            // // // // //     // TH1F *histo_clone = (TH1F*) energy_resolution_per_energy[ip][ie]->Clone();
            // // // // //     // int clonemaxbin = histo_clone->GetBinContent(histo_clone->GetMaximumBin());
            // // // // //     // double clonebinwidth = histo_clone->GetBinWidth(1);
            // // // // //     // while (clonemaxbin<30 || clonebinwidth>0.2) {
            // // // // //     //     histo_clone->Rebin(30);
            // // // // //     //     clonemaxbin = histo_clone->GetBinContent(histo_clone->GetMaximumBin());
            // // // // //     //     clonebinwidth = histo_clone->GetBinWidth(1);
            // // // // //     // }
            // // // // //     // histo_clone->Fit("gaus","NQ","",-Eres_fit_range,Eres_fit_range);
            // // // // //     if(ip==0) energy_resolution_per_energy[ip][ie]->Fit("gaus","NQ","",-0.2,0.2);
            // // // // //     else energy_resolution_per_energy[ip][ie]->Fit("gaus","NQ","",-Eres_fit_range,Eres_fit_range);
            // // // // //     resolution_sigma[ip][ie] = gaus->GetParameter(2);
            // // // // //     resolution_sigma_error[ip][ie] = gaus->GetParError(2);
            // // // // // } else {
            // // // // //     resolution_sigma[ip][ie] = -1;
            // // // // //     resolution_sigma_error[ip][ie] = 0;
            // // // // // }
            if(energy_resolution_per_energy[ip][ie]->Integral(Eres_fitbin_lower,Eres_fitbin_upper)>30){
                // TH1F *histo_clone = (TH1F*) energy_resolution_per_energy[ip][ie]->Clone();
                // int clonemaxbin = histo_clone->GetBinContent(histo_clone->GetMaximumBin());
                // double clonebinwidth = histo_clone->GetBinWidth(1);
                // while (clonemaxbin<30 || clonebinwidth>0.2) {
                //     histo_clone->Rebin(30);
                //     clonemaxbin = histo_clone->GetBinContent(histo_clone->GetMaximumBin());
                //     clonebinwidth = histo_clone->GetBinWidth(1);
                // }
                // histo_clone->Fit("gaus","NQ","",-Eres_fit_range,Eres_fit_range);
                if(ip==0) energy_resolution_per_energy[ip][ie]->Fit("gaus","NQ","",-0.2,0.2);
                // else energy_resolution_per_energy[ip][ie]->Fit("gaus","NQ","",-Eres_fit_range,Eres_fit_range);
                else energy_resolution_per_energy[ip][ie]->Fit("gaus","NQ","",-0.1,0.1);
                resolution_sigma[ip][ie] = gaus->GetParameter(2);
                resolution_sigma_error[ip][ie] = gaus->GetParError(2);
            } else {
                energy_resolution_per_energy[ip][ie]->Rebin(4);
                energy_resolution_per_energy[ip][ie]->Fit("gaus","NQ","",-0.1,0.1);
                resolution_sigma[ip][ie] = gaus->GetParameter(2);
                resolution_sigma_error[ip][ie] = gaus->GetParError(2);
                if(gaus->GetParameter(2)>0.2){
                    resolution_sigma[ip][ie] = -1;
                    resolution_sigma_error[ip][ie] = 0;
                }
            }

            energy_resolution_rms[ip]->SetPoint(ie, (ie+0.5)*energy_interval, resolution_rms[ip][ie]);
            energy_resolution_rms[ip]->SetPointError(ie, (0.5)*energy_interval, 0);
            energy_resolution_sigma[ip]->SetPoint(ie, (ie+0.5)*energy_interval, resolution_sigma[ip][ie]);
            energy_resolution_sigma[ip]->SetPointError(ie, (0.5)*energy_interval, resolution_sigma_error[ip][ie]);

            // energy_resolution_per_energy[ip][ie]->SetAxisRange(-Eres_range,Eres_range);
            // cout << "   " << resolution_rms[ip][ie] << ", " << resolution_sigma[ip][ie] << endl;
            if( (ip<2 && ie>6) || (ip==2 && (ie>1 && ie<5)) ){
                string energy_range = Form("%d-%d GeV",(int)(ie*energy_interval),(int)((ie+1)*energy_interval));
                cout << "  " << energy_range << " : " << resolution_sigma[ip][ie] << endl;
            }
            // energy_resolution_per_energy[ip][ie]->Rebin(rebin_factor);
            // energy_diff_per_energy[ip][ie]->Rebin(rebin_factor);
        }
        energy_resolution_rms[ip]->Draw("AP");
        energy_resolution_sigma[ip]->Draw("P");
        // legend_res->SetTextSize(0.03);
        // legend_res->SetFillStyle(0);
        legend_res->Draw("same");
    }

    // TCanvas *c_energy_scan = new TCanvas("c_energy_scan","c_energy_scan",1);
    // c_energy_scan->cd();
    // energy_resolution_sigma[0]->SetLineColor(kBlue);
    // energy_resolution_sigma[0]->SetMarkerColor(kBlue);
    // energy_resolution_sigma[1]->SetLineColor(kRed);
    // energy_resolution_sigma[1]->SetMarkerColor(kRed);
    // energy_resolution_sigma[2]->SetLineColor(kGreen);
    // energy_resolution_sigma[2]->SetMarkerColor(kGreen);
    // energy_resolution_sigma[0]->Draw("AP");
    // for(int ip=1;ip<nParticle;ip++){
    //     energy_resolution_sigma[ip]->Draw("P");
    // }

    /*
    TCanvas *canvas_beta_clustering = new TCanvas("canvas_beta_clustering","canvas_beta_clustering",1);
    canvas_beta_clustering->Divide(nParticle,2);
    canvas_beta_clustering->cd();
    for(int ip=0;ip<nParticle*2;ip++){
        canvas_beta_clustering->cd(ip+1);
        // gPad->SetLogy();
        // if(ip<nParticle) energy[ip]->Draw();
        if(ip<nParticle){
            // gStyle->SetOptStat(0);
            condbeta_vs_eff[ip]->SetStats(0);
            condbeta_vs_eff[ip]->Draw("colz");
        }
        else {
            // gPad->SetLogy();
            condbeta_vs_pur[ip-nParticle]->SetStats(0);
            condbeta_vs_pur[ip-nParticle]->Draw("colz");
        }
        // else MCtruth_energy[ip-nParticle]->Draw();
    }
    */


    /*
    TCanvas *canvas_beta_energy = new TCanvas("canvas_beta_energy","canvas_beta_energy",1);
    canvas_beta_energy->Divide(nParticle,2);
    canvas_beta_energy->cd();
    for(int ip=0;ip<nParticle*2;ip++){
        canvas_beta_energy->cd(ip+1);
        // gPad->SetLogy();
        // if(ip<nParticle) energy[ip]->Draw();
        if(ip<nParticle){
            // gStyle->SetOptStat(0);
            condbeta_vs_Ediff[ip]->SetStats(0);
            condbeta_vs_Ediff[ip]->Draw("colz");
        }
        else {
            // gPad->SetLogy();
            condbeta_vs_mcen[ip-nParticle]->SetStats(0);
            condbeta_vs_mcen[ip-nParticle]->Draw("colz");
        }
        // else MCtruth_energy[ip-nParticle]->Draw();
    }
    */
    /*
    TCanvas *canvas_beta_ediff = new TCanvas("canvas_beta_ediff","canvas_beta_ediff",1);
    canvas_beta_ediff->Divide(nParticle,2);
    canvas_beta_ediff->cd();
    for(int ip=0;ip<nParticle*2;ip++){
        canvas_beta_ediff->cd(ip+1);
        // gPad->SetLogy();
        // if(ip<nParticle) energy[ip]->Draw();
        if(ip<nParticle){
            // gStyle->SetOptStat(0);
            condbeta_vs_Ediff_track[ip]->SetStats(0);
            condbeta_vs_Ediff_track[ip]->Draw("colz");
        }
        else {
            // gPad->SetLogy();
            condbeta_vs_Ediff_nottrack[ip-nParticle]->SetStats(0);
            condbeta_vs_Ediff_nottrack[ip-nParticle]->Draw("colz");
        }
        // else MCtruth_energy[ip-nParticle]->Draw();
    }
    */
    /*
    TCanvas *canvas_beta_mcen = new TCanvas("canvas_beta_mcen","canvas_beta_mcen",1);
    canvas_beta_mcen->Divide(nParticle,2);
    canvas_beta_mcen->cd();
    for(int ip=0;ip<nParticle*2;ip++){
        canvas_beta_mcen->cd(ip+1);
        // gPad->SetLogy();
        // if(ip<nParticle) energy[ip]->Draw();
        if(ip<nParticle){
            // gStyle->SetOptStat(0);
            condbeta_vs_mcen_track[ip]->SetStats(0);
            condbeta_vs_mcen_track[ip]->Draw("colz");
        }
        else {
            // gPad->SetLogy();
            condbeta_vs_mcen_nottrack[ip-nParticle]->SetStats(0);
            condbeta_vs_mcen_nottrack[ip-nParticle]->Draw("colz");
        }
        // else MCtruth_energy[ip-nParticle]->Draw();
    }
    */



    TLegend *legend_resolution_cluster[nParticle];
    int Ymaximum_resolution_cluster[nParticle];
    for(int ip=0;ip<nParticle;ip++){
        Ymaximum_resolution_cluster[ip] = 0;
        for(int ie=0;ie<nEnergy;ie++){
            Ymaximum_resolution_cluster[ip] = energy_resolution_per_energy_cluster[ip][ie]->GetMaximum() > Ymaximum_resolution_cluster[ip] ? energy_resolution_per_energy_cluster[ip][ie]->GetMaximum() : Ymaximum_resolution_cluster[ip];
        }
    }
    double cluster_resolution_rms[nParticle][nEnergy];
    double cluster_resolution_sigma[nParticle][nEnergy];
    double cluster_resolution_sigma_error[nParticle][nEnergy];
    TGraphErrors *cluster_energy_resolution_rms[nParticle];
    TGraphErrors *cluster_energy_resolution_sigma[nParticle];
    TCanvas *canvas_cluster_energy_resolution_scan = new TCanvas("canvas_cluster_energy_resolution_scan","canvas_cluster_energy_resolution_scan",1);
    canvas_cluster_energy_resolution_scan->Divide(nParticle,2);
    canvas_cluster_energy_resolution_scan->cd();
    if(ECluster){
        for(int ip=0;ip<nParticle;ip++){
            legend_resolution_cluster[ip] = new TLegend( 0.1, 0.6, 0.5, 0.9) ;
            canvas_cluster_energy_resolution_scan->cd(ip+1);
            for(int ie=0;ie<nEnergy;ie++){
                // string drawOption_legend = ie==0 ? "" : "same";
                legend_resolution_cluster[ip]->AddEntry(energy_resolution_per_energy_cluster[ip][ie], Form("%d-%d GeV mean:%f StdDev:%f",(int)(ie*energy_interval),(int)((ie+1)*energy_interval), energy_resolution_per_energy_cluster[ip][ie]->GetMean(), energy_resolution_per_energy_cluster[ip][ie]->GetStdDev()) , "l");
                // double res = energy_resolution_per_energy_cluster[ip][ie]->GetStdDev() / energy_resolution_per_energy_cluster[ip][ie]->GetMean();
    
                string drawOption = ie==0 ? "HIST" : "same HIST";
                int colorId = ie<9 ? ie+1 : ie+2;
                energy_resolution_per_energy_cluster[ip][ie]->SetLineColor(colorId);
                energy_resolution_per_energy_cluster[ip][ie]->SetMarkerColor(colorId);
                energy_resolution_per_energy_cluster[ip][ie]->SetMaximum(Ymaximum_resolution_cluster[ip]*1.3);
                // energy_resolution_per_energy_cluster[ip][ie]->Scale(1./efficiency_energy[ip][ie]->GetEntries());
                energy_resolution_per_energy_cluster[ip][ie]->SetStats(0);
                energy_resolution_per_energy_cluster[ip][ie]->Draw(drawOption.c_str());
            }
            legend_resolution_cluster[ip]->SetTextSize(0.03);
            legend_resolution_cluster[ip]->SetFillStyle(0);
            legend_resolution_cluster[ip]->Draw("same");
    
    
            canvas_cluster_energy_resolution_scan->cd(ip+nParticle+1);
            cout << particleNames[ip] << " cluster energy " << endl;
            cluster_energy_resolution_rms[ip] = new TGraphErrors();
            cluster_energy_resolution_rms[ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
            cluster_energy_resolution_rms[ip]->GetXaxis()->SetTitle("GeV");
            cluster_energy_resolution_rms[ip]->GetYaxis()->SetTitle("simga of (pred-truth)/truth");
            cluster_energy_resolution_rms[ip]->SetLineColor(kRed);
            cluster_energy_resolution_rms[ip]->SetMarkerColor(kRed);
            cluster_energy_resolution_rms[ip]->SetMaximum(0.5);
            cluster_energy_resolution_rms[ip]->SetMinimum(0);
            cluster_energy_resolution_sigma[ip] = new TGraphErrors();
            cluster_energy_resolution_sigma[ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
            cluster_energy_resolution_sigma[ip]->SetLineColor(kBlue);
            cluster_energy_resolution_sigma[ip]->SetMarkerColor(kBlue);
            cluster_energy_resolution_sigma[ip]->SetMaximum(0.5);
            cluster_energy_resolution_sigma[ip]->SetMinimum(0);
            for(int ie=0;ie<nEnergy;ie++){
                // cout << energy_resolution_per_energy[ip][ie]->GetStdDev() << "    ";
                energy_resolution_per_energy_cluster[ip][ie]->SetAxisRange(-Eres_fit_range,Eres_fit_range);
                // cout << energy_resolution_per_energy[ip][ie]->GetStdDev() << endl;
                cluster_resolution_rms[ip][ie] = energy_resolution_per_energy_cluster[ip][ie]->GetStdDev();
                if(energy_resolution_per_energy_cluster[ip][ie]->Integral(Eres_fitbin_lower,Eres_fitbin_upper)>30){
                    // TH1F *histo_clone = (TH1F*) energy_resolution_per_energy[ip][ie]->Clone();
                    // int clonemaxbin = histo_clone->GetBinContent(histo_clone->GetMaximumBin());
                    // double clonebinwidth = histo_clone->GetBinWidth(1);
                    // while (clonemaxbin<30 || clonebinwidth>0.2) {
                    //     histo_clone->Rebin(30);
                    //     clonemaxbin = histo_clone->GetBinContent(histo_clone->GetMaximumBin());
                    //     clonebinwidth = histo_clone->GetBinWidth(1);
                    // }
                    // histo_clone->Fit("gaus","NQ","",-Eres_fit_range,Eres_fit_range);
                    energy_resolution_per_energy_cluster[ip][ie]->Fit("gaus","NQ","",-Eres_fit_range,Eres_fit_range);
                    cluster_resolution_sigma[ip][ie] = gaus->GetParameter(2);
                    cluster_resolution_sigma_error[ip][ie] = gaus->GetParError(2);
                } else {
                    cluster_resolution_sigma[ip][ie] = -1;
                    cluster_resolution_sigma_error[ip][ie] = 0;
                }
    
                cluster_energy_resolution_rms[ip]->SetPoint(ie, (ie+0.5)*energy_interval, cluster_resolution_rms[ip][ie]);
                cluster_energy_resolution_rms[ip]->SetPointError(ie, (0.5)*energy_interval, 0);
                cluster_energy_resolution_sigma[ip]->SetPoint(ie, (ie+0.5)*energy_interval, cluster_resolution_sigma[ip][ie]);
                cluster_energy_resolution_sigma[ip]->SetPointError(ie, (0.5)*energy_interval, cluster_resolution_sigma_error[ip][ie]);
    
                // energy_resolution_per_energy[ip][ie]->SetAxisRange(-Eres_range,Eres_range);
                // cout << "   " << resolution_rms[ip][ie] << ", " << resolution_sigma[ip][ie] << endl;
                if( (ip<2 && ie>6) || (ip>=2 && (ie>1 && ie<5)) ){
                    string energy_range = Form("%d-%d GeV",(int)(ie*energy_interval),(int)((ie+1)*energy_interval));
                    cout << "  " << energy_range << " : " << cluster_resolution_sigma[ip][ie] << endl;
                }
                // energy_resolution_per_energy[ip][ie]->Rebin(rebin_factor);
                // energy_diff_per_energy[ip][ie]->Rebin(rebin_factor);
           }
            cluster_energy_resolution_rms[ip]->Draw("AP");
            cluster_energy_resolution_sigma[ip]->Draw("P");
            // legend_res->SetTextSize(0.03);
            // legend_res->SetFillStyle(0);
            legend_res->Draw("same");
        }
    }


    TCanvas *canvas_energy_regression_result = new TCanvas("canvas_energy_regression_result","canvas_energy_regression_result",1);
    canvas_energy_regression_result->Divide(nParticle,2);
    for(int ip=0;ip<nParticle;ip++){
        canvas_energy_regression_result->cd(ip+1);
        energy2d[ip]->SetStats(0);
        clusterenergy2d[ip]->SetStats(0);
        if(ip<2) energy2d[ip]->Draw("colz");
        else clusterenergy2d[ip]->Draw("colz");

        canvas_energy_regression_result->cd(ip+1+nParticle);
        if(ip<2){
            energy_resolution_rms[ip]->Draw("AP");
            energy_resolution_sigma[ip]->Draw("P");
            // legend_res->SetTextSize(0.03);
            // legend_res->SetFillStyle(0);
            legend_res->Draw("same");
        } else {
            cluster_energy_resolution_rms[ip]->Draw("AP");
            cluster_energy_resolution_sigma[ip]->Draw("P");
            legend_res->Draw("same");
        }
    }


    TCanvas *canvas_jet_energy = new TCanvas("canvas_jet_energy","canvas_jet_energy",1400,500);
    canvas_jet_energy->Divide(3,1);
    canvas_jet_energy->cd();
    canvas_jet_energy->cd(1);
    jet_energy2d->SetStats(0);
    jet_energy2d->Draw("colz");

    canvas_jet_energy->cd(2);
    TLegend *legend_jet_resolution;
    int Ymaximum_jet_resolution = 0;
        for(int ie=0;ie<nEnergy_jet;ie++){
            Ymaximum_jet_resolution = jet_energy_resolution_per_energy[ie]->GetMaximum() > Ymaximum_jet_resolution ? jet_energy_resolution_per_energy[ie]->GetMaximum() : Ymaximum_jet_resolution;
        }
        legend_jet_resolution = new TLegend( 0.1, 0.6, 0.5, 0.9) ;
        for(int ie=0;ie<nEnergy_jet;ie++){
            legend_jet_resolution->AddEntry(jet_energy_resolution_per_energy[ie], Form("%d-%d GeV mean:%f StdDev:%f",(int)(ie*10),(int)((ie+1)*10), jet_energy_resolution_per_energy[ie]->GetMean(), jet_energy_resolution_per_energy[ie]->GetStdDev()) , "l");
            string drawOption = ie==0 ? "HIST" : "same HIST";
            int colorId = ie<9 ? ie+1 : ie+2;
            jet_energy_resolution_per_energy[ie]->SetLineColor(colorId);
            jet_energy_resolution_per_energy[ie]->SetMarkerColor(colorId);
            jet_energy_resolution_per_energy[ie]->SetMaximum(Ymaximum_jet_resolution*3);
            jet_energy_resolution_per_energy[ie]->SetStats(0);
            jet_energy_resolution_per_energy[ie]->Draw(drawOption.c_str());
        }
        legend_jet_resolution->SetTextSize(0.03);
        legend_jet_resolution->SetFillStyle(0);
        legend_jet_resolution->Draw("same");

    canvas_jet_energy->cd(3);
    double jet_resolution_rms[nEnergy_jet];
    double jet_resolution_sigma[nEnergy_jet];
    double jet_resolution_sigma_error[nEnergy_jet];
    TGraphErrors *jet_energy_resolution_rms;
    TGraphErrors *jet_energy_resolution_sigma;
    TLegend *legend_jet_res = new TLegend( 0.5, 0.75, 0.9, 0.9);
    cout << "jet energy resolution" << endl;
    jet_energy_resolution_rms = new TGraphErrors();
    jet_energy_resolution_rms->SetTitle(Form("jet"));
    jet_energy_resolution_rms->GetXaxis()->SetTitle("GeV");
    jet_energy_resolution_rms->GetYaxis()->SetTitle("simga of (pred-truth)/truth (%)");
    jet_energy_resolution_rms->SetLineColor(kRed);
    jet_energy_resolution_rms->SetMarkerColor(kRed);
    jet_energy_resolution_rms->SetMaximum(15);
    jet_energy_resolution_rms->SetMinimum(0);
    jet_energy_resolution_sigma = new TGraphErrors();
    jet_energy_resolution_sigma->SetTitle(Form("jet"));
    jet_energy_resolution_sigma->SetLineColor(kBlue);
    jet_energy_resolution_sigma->SetMarkerColor(kBlue);
    jet_energy_resolution_sigma->SetMaximum(15);
    jet_energy_resolution_sigma->SetMinimum(0);
    legend_jet_res->AddEntry(jet_energy_resolution_rms, Form("rms") , "l");
    legend_jet_res->AddEntry(jet_energy_resolution_sigma, Form("gaussian sigma") , "l");
    for(int ie=0;ie<nEnergy_jet;ie++){
        jet_energy_resolution_per_energy[ie]->SetAxisRange(-Eres_fit_range,Eres_fit_range);
        jet_resolution_rms[ie] = jet_energy_resolution_per_energy[ie]->GetStdDev();
        if(jet_energy_resolution_per_energy[ie]->Integral(Eres_fitbin_lower,Eres_fitbin_upper)>30){
            jet_energy_resolution_per_energy[ie]->Fit("gaus","NQ","",-0.1,0.1);
            jet_resolution_sigma[ie] = gaus->GetParameter(2);
            jet_resolution_sigma_error[ie] = gaus->GetParError(2);
        } else {
            jet_energy_resolution_per_energy[ie]->Rebin(4);
            jet_energy_resolution_per_energy[ie]->Fit("gaus","NQ","",-0.1,0.1);
            jet_resolution_sigma[ie] = gaus->GetParameter(2);
            jet_resolution_sigma_error[ie] = gaus->GetParError(2);
            if(gaus->GetParameter(2)>0.2){
                jet_resolution_sigma[ie] = -1;
                jet_resolution_sigma_error[ie] = 0;
            }
        }
        jet_energy_resolution_rms->SetPoint(ie, (ie+0.5)*10, jet_resolution_rms[ie]*100);
        jet_energy_resolution_rms->SetPointError(ie, (0.5)*10, 0);
        jet_energy_resolution_sigma->SetPoint(ie, (ie+0.5)*10, jet_resolution_sigma[ie]*100);
        jet_energy_resolution_sigma->SetPointError(ie, (0.5)*10, jet_resolution_sigma_error[ie]*100);
        string energy_range = Form("%d-%d GeV",(int)(ie*10),(int)((ie+1)*10));
        cout << "  " << energy_range << " : " << jet_resolution_sigma[ie] << endl;
    }
    jet_energy_resolution_rms->Draw("AP");
    jet_energy_resolution_sigma->Draw("P");
    legend_jet_res->Draw("same");


    
    if(saving_canvas){  // saving canvases
        
        compare->SaveAs(Form("%s/efficiency_purity.pdf",picDirectory.c_str()));
        // compare2d->SaveAs(Form("%s/efficiency_purity_vs_energy%s.pdf",picDirectory.c_str(),suffix.c_str()));
        // compare_energy->SaveAs(Form("%s/per_energy%s.pdf",picDirectory.c_str(),suffix.c_str()));
        // compare_energy_normalized->SaveAs(Form("%s/per_energy_norm%s.pdf",picDirectory.c_str(),suffix.c_str()));
        // canvas_energy->SaveAs(Form("%s/energy_truth_vs_pred.pdf",picDirectory.c_str()));
        canvas_energy_scan->SaveAs(Form("%s/energy_scan.pdf",picDirectory.c_str()));
        canvas_energy_resolution_scan->SaveAs(Form("%s/energy_resolution_scan.pdf",picDirectory.c_str()));
        // canvas_beta_energy->SaveAs(Form("%s/beta_vs_energy.pdf",picDirectory.c_str()));
        // canvas_beta_ediff->SaveAs(Form("%s/beta_vs_energy_ediff.pdf",picDirectory.c_str()));
        // canvas_beta_mcen->SaveAs(Form("%s/beta_vs_energy_mcen.pdf",picDirectory.c_str()));
        canvas_energy_regression_result->SaveAs(Form("%s/energy_regression.pdf",picDirectory.c_str()));
    }
    

}
