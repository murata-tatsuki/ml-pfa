using namespace std;

// efficiency, purityを表示して保存するマクロ

const string test_particle_types = {"ntau_10GeV_10", "uds"};

// conditions
const bool saving_canvas = false;
const string train_particle_type = "uds91";      // ntau_10GeV_10    uds91   ntau_10to100GeV_10
const string test_particle_type = train_particle_type;      // ntau_10GeV_10    uds91   ntau_10to100GeV_10
const bool pandora = true;

const bool momentum_input = true;


const string ab = "alpha";                                // alpha       betaE
const bool modi = true;
const string lossF = "alphaTrackModifying_LE16";               // alphaTrackModifying       alphaTrackModifyingCharge0     alphaTrackModifyingAll0       alphaModifying       alphaTrackModifying_LE16
const string testMCdetect = "testDetected";                       // testMCTruth       testDetected 
const bool trainMCdetect = false;                            // momenta of virtual hit are MC truth or detected
const bool reduced_samples = true;
const bool epochs = false;
const string lossF_epochs = "alphaTrackModifying_LE16";               // alphaTrackModifying       alphaTrackModifying_coef005_LE16     alphaTrackModifying_coef005     alphaTrackModifying_LE16        alphaModifying
const int nepoch = 59;

const bool energyByPass = false;
const bool ECluster = true;

const bool filepath_ = true;





const int dimension = 5;    // output dimensions  (one for beta, others are for coordinates)
const bool hyper_parameter = false;

// const bool fine_tuning = false;
// const int epoch = 25;       // 20   25
// const int train_epoch = epoch*2-1;

const int energyMax = test_particle_type == "ntau_10GeV_10" ? 12 : 100;
const int energyMaximum = test_particle_type == "ntau_10GeV_10" ? 10 : (test_particle_type == "uds91" ? 40 : 100 );


void energy_regression_pandora_check(){ 
    int rawfilenum = 1;

    // if(hyper_parameter && fine_tuning){ // condition check
    //     cout << "something wrong with setting boolian " << endl;
    //     abort();
    // }

    TFile *filein[rawfilenum];
    TTree *tree[rawfilenum];
    TTree *tree_pred[rawfilenum];
    int entry_max[rawfilenum];
    int total_entry_max=0;
    string picDirectory = "../pic/energy_regression";
    string fileName = "";
    
    // energy regression
    if(!pandora){
        if(!momentum_input){
            // if(ab=="betaE") fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10_betaE.root");
            // if(ab=="alpha") fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_%s_5D_49_%s_alphaMSE.root",train_particle_type.c_str(),test_particle_type.c_str());
            // if(ab=="betaE") fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10_betaE_fixloss.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_uds91_5D_49_uds91_alphaMSE.root");

            fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_%s_5D_49_%s_alphaMSE.root",train_particle_type.c_str(),test_particle_type.c_str());
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_uds91_5D_49_uds91_alphaMSE.root");
        } else {
            if(!modi){
            // cout << Form("../output/energy_regression/new_clustering/energyTree/tc_%s_5D_49_%s_alphaMSE_momentum.root",train_particle_type.c_str(),test_particle_type.c_str()) << endl;
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_%s_5D_49_%s_alphaMSE_momentum.root",train_particle_type.c_str(),test_particle_type.c_str());
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10_alphaMSE_momentum.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10_alphaMSE_momentum_momentumAmp.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_%s_5D_49_%s_alphaMSE_momentum_restartPeriod50.root",train_particle_type.c_str(),test_particle_type.c_str());
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10_alphaMSE_momentum_restartPeriod50_coef001.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_17_ntau_10GeV_10_alphaMSE_momentum_restartPeriod50_coef001.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10_alphaMSE_momentum_restartPeriod30_coef01.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10_alphaMSE_momentum_restartPeriod30_coef05.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_alphaTrack_momentum_restartPeriod30.root");
            
            // long task
            fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_499_ntau_10GeV_10_alphaTrack_momentum_restartPeriod30.root");


                // adjustment 
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_alphaTrack_momentum_restartPeriod30_trueloss_detectecprediction.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_alpha_momentum_restartPeriod30_trueloss_detectecprediction.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_alphaTrack_momentum_restartPeriod30_trueloss_backup.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_alpha_momentum_restartPeriod30_trueloss.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_alphaTrack_momentum_restartPeriod30_detectedloss.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_alpha_momentum_restartPeriod30_detectedloss.root");
            // fileName = Form("../output/energy_regression/new_clustering/energyTree/tc_ntau_10GeV_10_5D_14_ntau_10GeV_10_alphaMSE_momentum_restartPeriod50_condbeta_tbeta060.root");

            // modifing
            } else {
                string trainOption = trainMCdetect ? "_virtualhitTrueMomentum" : "";
                string trainOptionPic = trainMCdetect ? "virtualhitTrueMomentum/" : "";
                picDirectory = Form("../pic/energy_regression/restartPeriod/30/modifying/%s%s",trainOptionPic.c_str(), lossF.c_str());
                fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum%s/%s_%s.root",trainOption.c_str(),lossF.c_str(),testMCdetect.c_str());

                if(epochs){
                    string reducedOption = reduced_samples ? "reduced/" : "";
                    picDirectory = Form("../pic/energy_regression/restartPeriod/30/modifying/%s%s%s/%d",trainOptionPic.c_str(),reducedOption.c_str(),lossF_epochs.c_str(),nepoch);
                    fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum%s/epochs/%s%s/%s_epoch%d.root",trainOption.c_str(),reducedOption.c_str(),lossF_epochs.c_str(),testMCdetect.c_str(),nepoch);
                }
            }

            if(energyByPass){
                // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/alphaTrackModifyingCharge0_EBranch_testDetected.root");
                fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/energy_branch/alphaModifying_EBranch_LE16_005_testDetected.root");
                // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/energy_branch/alphaModifying_EBranch_LE16_010_testDetected.root");
                // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/energy_branch/alphaModifying_EBranch_LE16_testDetected.root");
            }
            if(ECluster){
                // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/cluster_energy_regression/alpha_LE16_010_testDetected_ERcluster.root");
                fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/cluster_energy_regression/alpha_LE16_010_testDetected_ERcluster_2025_01_31_155919.root");
                // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/cluster_energy_regression/alpha_LE16_010_testDetected_ERcluster_2025_02_02_015711.root");
                // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/cluster_energy_regression/alpha_LE16_010_testDetected_ERcluster_2025_02_02_162020.root");
            }
        }
    } else {
        fileName = Form("../output/energy_regression/new_clustering/energyTree/pandora/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10_pandora.root");
        fileName = Form("../output/energy_regression/new_clustering/tc_ntau_10to100GeV_10_5D_59_ntau_10to100GeV_10_pandora_20250220.root");
        fileName = Form("../output/energy_regression/new_clustering/tc_ntau_10GeV_10_5D_59_ntau_10to100GeV_10_pandora_20250304.root");
    }
    if(filepath_){
        // // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/alpha_LE16_010_testDetected_179.root");     // なんか変 pred_e が1/2にってる
        // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/alpha_LE16_010_testDetected_499.root");     // なんか変 pred_e が1/2にってる
        // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/alpha_LE16_010_testDetected_499_tbeta_td_scan/tbeta090td070.root");
        // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/cluster_energy_regression/alpha_LE16_010_testDetected_ERcluster_2025_02_02_015711.root");
        // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/cluster_energy_regression/alphaSqrtdiv_LE16_010_ERcluster_2025_02_04_174502.root");
        // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/alpha_LE16_010_testDetected_499_tbeta_td_scan/tbeta090td030.root");
        fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/cluster_energy_regression/alpha_LE16_010_ERcluster_sum_2025_02_14_101644/alpha_LE16_010_ERcluster_sum_2025_02_14_101644_494.root");
        // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/alpha_LE16_010_gradually.root");
        // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_uds91_5D_momentum/alpha_LE16_010_retrain_59.root");
        // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10to100GeV_10_5D_momentum/alpha_sqrtdiv_LE16_010_lr1e-4.root");
        // fileName = Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10to100GeV_10_5D_momentum/alpha_LE16_010.root");
        // fileName = Form("../output/energy_regression_1to1/pandora/tc_ntau_10GeV_10_5D_pandora_20250317.root");
        fileName = Form("../output/energy_regression_1to1/pandora/tc_uds91_5D_pandora_20250317.root");
        fileName = Form("../output/energy_regression_1to1/pandora/tc_uds91_5D_pandora_20250317_1tomany.root");
    }
    filein[0] = new TFile(Form("%s",fileName.c_str()));
    cout << fileName << endl;
    cout << picDirectory << endl;

    for(int i=0; i<rawfilenum; i++){
        tree[i] = (TTree*) filein[i]->Get("t");
        entry_max[i] = tree[i]->GetEntries();
        tree_pred[i] = (TTree*) filein[i]->Get("prediction");
    }


    int event, hitid, mcid, truthid, mcpdg, mccharge, mcstatus, ncluster, matched_ncluster, matched_cluster, cond_track;
    double mcmass, mcpx, mcpy, mcpz, mcen, edep, edep_reco, edep_match, pred_edep, pred_edep_cluster, cond_beta, pred_alpha;

    int _event, _hitid, _mcid, _truthid, _mcpdg, _mccharge, _mcstatus, _ncluster, _matched_ncluster, _matched_cluster, _pred_alpha;
    double _mcmass, _mcpx, _mcpy, _mcpz, _mcen, _edep, _edep_reco, _edep_match, _pred_edep, _pred_edep_cluster, _pred_beta;



    // TFile fileout("result_test_pandora.root","RECREATE");
    TFile fileout("result/result_test.root","RECREATE");


    // string particleNames[3] = {"electron", "pion", "gamma"};
    string particleNames[3] = {"electron", "pion", "photon"};
    vector<int> particledgValues = {11,-11, 211,-211, 22};

    double Eres_range = pandora ? 0.1 : 1.5;
    // double Eres_range = 1.5;
    // double Eres_range = 1.5;
    int Eres_nbin = 1200;
    double E_res_binWidth = Eres_range*2/Eres_nbin;
    double Eres_fit_range = 0.5;
    int Eres_fitbin_lower = Eres_range>Eres_fit_range ? (Eres_range-Eres_fit_range)/E_res_binWidth : 0;
    int Eres_fitbin_upper = Eres_range>Eres_fit_range ? Eres_nbin - (Eres_range-Eres_fit_range)/E_res_binWidth : -1;
    int rebin_factor = 0.025 / E_res_binWidth;
    
    const int nParticle = 3;
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
            purity_energy[ip][ie] = new TH1F(Form("purity_%d_%d",ip,ie), title.c_str(), 51,0,1.02);
            purity_energy[ip][ie]->SetXTitle("purity (edep_match/edep_reco)");
            purity_energy_normalize[ip][ie] = new TH1F(Form("purity_norm_%d_%d",ip,ie), title.c_str(), 51,0,1.02);
            purity_energy_normalize[ip][ie]->SetXTitle("purity (edep_match/edep_reco)");
            title = ie==0 ? Form("%s efficiency",particleNames[ip].c_str()) : Form("%s efficiency %d-%d GeV",particleNames[ip].c_str(),ie,ie+1);
            efficiency_energy[ip][ie] = new TH1F(Form("efficiency_%d_%d",ip,ie), title.c_str(), 51,0,1.02);
            efficiency_energy[ip][ie]->SetXTitle("efficiency (edep_match/edep)");
            efficiency_energy_normalize[ip][ie] = new TH1F(Form("efficiency_normalize_%d_%d",ip,ie), title.c_str(), 51,0,1.02);
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

        // bool bool_cout = false;
        // int ievent = -1;
        for(int ientry=0; ientry<entry_max[irawfile]; ientry++){
            tree[irawfile]->GetEntry(ientry);

            if(edep<=0 || edep_reco<=0 || edep_match<0) continue;
            auto result = find(particledgValues.begin(), particledgValues.end(), mcpdg);
            if(result == particledgValues.end()) continue;
            int itr = distance(particledgValues.begin(), result) / 2;

            double pur = edep_match / edep_reco;
            double eff = edep_match / edep;

            if(edep>1) purity[itr]->Fill(pur);
            purity2d[itr]->Fill(pur,edep);
            if(edep<10){
                purity_energy[itr][(int)edep]->Fill(pur);
                purity_energy_normalize[itr][(int)edep]->Fill(pur);
                efficiency_energy[itr][(int)edep]->Fill(eff);
                efficiency_energy_normalize[itr][(int)edep]->Fill(eff);
            }
            if(edep>1) efficiency[itr]->Fill(eff);
            efficiency2d[itr]->Fill(eff,edep);

            energy[itr]->Fill(pred_edep);
            MCtruth_energy[itr]->Fill(mcen);
            energy_diff[itr]->Fill(pred_edep - mcen);
            energy2d[itr]->Fill(mcen,pred_edep);
            clusterenergy2d[itr]->Fill(mcen,pred_edep_cluster);
            energy_resolution[itr]->Fill( (pred_edep - mcen) / mcen );

            if(mcen>1 && pred_edep<0.01){
                // bool_cout = ievent!=event;
                // if(bool_cout && ievent!=event) cout << ientry << "/" << entry_max[irawfile] << endl; ievent=event;
                cout << "event:" << event << " hitid:" << hitid << " mcid:" << mcid << " mcpdg:" << mcpdg << " mcen:" << mcen << " pred_edep:" << pred_edep << endl;
            }

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
            int itr = distance(particledgValues.begin(), result) / 2;
            // if(_pred_alpha==1) condbeta_vs_Ediff[itr]->Fill(_pred_beta,_pred_edep-_mcen);
            // if(_pred_alpha==1) condbeta_vs_mcen[itr]->Fill(_pred_beta,_mcen);
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

  /*
    TCanvas *compare2d = new TCanvas("compare2d","compare2d",1);
    compare2d->Divide(nParticle,2);
    for(int ip=0;ip<nParticle*2;ip++){
        compare2d->cd(ip+1);
        gPad->SetLogz();
        if(ip<nParticle) efficiency2d[ip]->Draw("colz");
        else purity2d[ip-nParticle]->Draw("colz");
    }


    int yaxis_height[nParticle*2];
    for(int ip=0;ip<nParticle*2;ip++){
        yaxis_height[ip] = 0;
        for(int ie=0;ie<nEnergy;ie++){
            if(ip<nParticle) yaxis_height[ip] = yaxis_height[ip]<efficiency_energy[ip][ie]->GetMaximum() ? efficiency_energy[ip][ie]->GetMaximum() : yaxis_height[ip];
            else yaxis_height[ip] = yaxis_height[ip]<purity_energy[ip-nParticle][ie]->GetMaximum() ? purity_energy[ip-nParticle][ie]->GetMaximum() : yaxis_height[ip];
        }
    }
    TCanvas *compare_energy = new TCanvas("compare_energy","compare_energy",1);
    compare_energy->Divide(nParticle,2);
    TLegend *legend[nParticle][2];

    for(int ip=0;ip<nParticle*2;ip++){
        compare_energy->cd(ip+1);
        gPad->SetLogy();

        legend[ip%nParticle][ip/nParticle] = new TLegend( 0.4, 0.6, 0.8, 0.9) ;

        for(int ie=0;ie<nEnergy;ie++){
            string drawOption = ie==0 ? "" : "same";
            int colorId = ie<9 ? ie+1 : ie+2;

            if(ip<nParticle){
                // efficiency_energy[ip][ie]->Rebin(2);
                efficiency_energy[ip][ie]->SetLineColor(colorId);
                efficiency_energy[ip][ie]->SetMarkerColor(colorId);
                efficiency_energy[ip][ie]->SetMaximum(yaxis_height[ip]*2);
                efficiency_energy[ip][ie]->Draw(drawOption.c_str());
                legend[ip%nParticle][ip/nParticle]->AddEntry(efficiency_energy[ip][ie], Form("%d-%d GeV",ie,ie+1) , "l");
                legend[ip%nParticle][ip/nParticle]->Draw();
            } else {
                // purity_energy[ip-nParticle][ie]->Rebin(2);
                purity_energy[ip-nParticle][ie]->SetLineColor(colorId);
                purity_energy[ip-nParticle][ie]->SetMarkerColor(colorId);
                purity_energy[ip-nParticle][ie]->SetMaximum(yaxis_height[ip]*2);
                purity_energy[ip-nParticle][ie]->Draw(drawOption.c_str());
                // compare_energy->cd(ip+1)->BuildLegend();
                legend[ip%nParticle][ip/nParticle]->AddEntry(purity_energy[ip-nParticle][ie], Form("%d-%d GeV",ie,ie+1) , "l");
                legend[ip%nParticle][ip/nParticle]->Draw();
            }
        }
    }


    TCanvas *compare_energy_normalized = new TCanvas("compare_energy_normalized","compare_energy_normalized",1);
    compare_energy_normalized->Divide(nParticle,2);
    for(int ip=0;ip<nParticle*2;ip++){
        compare_energy_normalized->cd(ip+1);
        gPad->SetLogy();

        for(int ie=0;ie<nEnergy;ie++){
            string drawOption = ie==0 ? "HIST" : "same HIST";
            int colorId = ie<9 ? ie+1 : ie+2;

            if(ip<nParticle){
                efficiency_energy_normalize[ip][ie]->SetLineColor(colorId);
                efficiency_energy_normalize[ip][ie]->SetMarkerColor(colorId);
                // efficiency_energy[ip][ie]->SetMaximum(yaxis_height[ip]*2);
                efficiency_energy_normalize[ip][ie]->Scale(1./efficiency_energy[ip][ie]->GetEntries());
                efficiency_energy_normalize[ip][ie]->Draw(drawOption.c_str());
                // legend[ip%nParticle][ip/nParticle]->AddEntry(efficiency_energy[ip][ie], Form("%d-%d GeV",ie,ie+1) , "l");
                // legend[ip%nParticle][ip/nParticle]->Draw();
            } else {
                purity_energy_normalize[ip-nParticle][ie]->SetLineColor(colorId);
                purity_energy_normalize[ip-nParticle][ie]->SetMarkerColor(colorId);
                // purity_energy[ip-nParticle][ie]->SetMaximum(yaxis_height[ip]*2);
                purity_energy_normalize[ip-nParticle][ie]->Scale(1./purity_energy[ip-nParticle][ie]->GetEntries());
                purity_energy_normalize[ip-nParticle][ie]->Draw(drawOption.c_str());
                // legend[ip%nParticle][ip/nParticle]->AddEntry(purity_energy[ip-nParticle][ie], Form("%d-%d GeV",ie,ie+1) , "l");
                // legend[ip%nParticle][ip/nParticle]->Draw();
            }
        }
    }
  */

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
                energy_resolution_per_energy[ip][ie]->Fit("gaus","NQ","",-Eres_fit_range,Eres_fit_range);
                resolution_sigma[ip][ie] = gaus->GetParameter(2);
                resolution_sigma_error[ip][ie] = gaus->GetParError(2);
            } else {
                resolution_sigma[ip][ie] = -1;
                resolution_sigma_error[ip][ie] = 0;
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
                if( (ip<2 && ie>6) || (ip==2 && (ie>1 && ie<5)) ){
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




    
    if(saving_canvas){  // saving canvases
        string suffix = "";
        string absuf = ab;
        if(hyper_parameter) suffix = Form("_%s",test_particle_type.c_str());
        // if(fine_tuning)     suffix = Form("_epoch%d_%s",epoch,test_particle_type.c_str());
        if(momentum_input)     suffix = Form("_momentum");
        if(modi){
            absuf = "";
            suffix = testMCdetect;
        }
        
        compare->SaveAs(Form("%s/efficiency_purity_%s%s.pdf",picDirectory.c_str(),absuf.c_str(),suffix.c_str()));
        // compare2d->SaveAs(Form("%s/efficiency_purity_vs_energy%s.pdf",picDirectory.c_str(),suffix.c_str()));
        // compare_energy->SaveAs(Form("%s/per_energy%s.pdf",picDirectory.c_str(),suffix.c_str()));
        // compare_energy_normalized->SaveAs(Form("%s/per_energy_norm%s.pdf",picDirectory.c_str(),suffix.c_str()));
        canvas_energy->SaveAs(Form("%s/energy_truth_vs_pred_%s%s.pdf",picDirectory.c_str(),absuf.c_str(),suffix.c_str()));
        canvas_energy_scan->SaveAs(Form("%s/energy_scan_%s%s.pdf",picDirectory.c_str(),absuf.c_str(),suffix.c_str()));
        canvas_energy_resolution_scan->SaveAs(Form("%s/energy_resolution_scan_%s%s.pdf",picDirectory.c_str(),absuf.c_str(),suffix.c_str()));
        canvas_beta_energy->SaveAs(Form("%s/beta_vs_energy_%s%s.pdf",picDirectory.c_str(),absuf.c_str(),suffix.c_str()));
        canvas_beta_ediff->SaveAs(Form("%s/beta_vs_energy_ediff_%s%s.pdf",picDirectory.c_str(),absuf.c_str(),suffix.c_str()));
        canvas_beta_mcen->SaveAs(Form("%s/beta_vs_energy_mcen_%s%s.pdf",picDirectory.c_str(),absuf.c_str(),suffix.c_str()));
    }
    

}
