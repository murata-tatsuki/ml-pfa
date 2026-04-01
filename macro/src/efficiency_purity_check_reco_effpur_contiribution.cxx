// macro to save or display efficiency and purity of the root file
// if the energy regression is conducted, truth vs predicted is also displayed
// 
// usage
// change the fileName, train_particle_type
// change to saving_canvas = true, if you want to save figures
// execute from macro/: root -l 'gSystem->AddIncludePath("-I."); .x src/efficiency_purity_check_reco_effpur_contiribution.cxx'
// or build: see README_BUILD.md — make in macro/ → ./build/efficiency_purity_contribution 
// 
// definition of efficiency and purity
//   double pur = edep_match / edep_reco;
//   double eff = edep_match / edep;

#include "root_common_includes.h"

#include <cmath>

using namespace std;

const int qq_energy = 350;
const double c_event_clean_threshold = 0.1;
const bool use_truth_clustering = true;


const double reco_truth_energy_min =
    qq_energy == 40  ? 0  :
    qq_energy == 91  ? 60  :
    qq_energy == 200 ? 150 :
    qq_energy == 350 ? 300 :
    qq_energy == 500 ? 400 : -1.0;

const double reco_truth_energy_max =
    qq_energy == 40  ? 60  :
    qq_energy == 91  ? 150  :
    qq_energy == 200 ? 300 :
    qq_energy == 350 ? 400 :
    qq_energy == 500 ? 600 : -1.0;

const double histo_energy_range_max =
    qq_energy == 40  ? 60  :
    qq_energy == 91  ? 150  :
    qq_energy == 200 ? 300 :
    qq_energy == 350 ? 450 :
    qq_energy == 500 ? 600 : 600;
const double histo_energy_range_max_y =
    qq_energy == 40  ? 100  :
    qq_energy == 91  ? 200  :
    qq_energy == 200 ? 300 :
    qq_energy == 350 ? 450 :
    qq_energy == 500 ? 600 : 600;

const int num_file =
    qq_energy == 40  ? 300 :
    qq_energy == 91  ? 300 :
    qq_energy == 200 ? 748  :
    qq_energy == 350 ? 749 :
    qq_energy == 500 ? 1498 : 1;
const int nfile =
    qq_energy == 40  ? 100 :
    qq_energy == 91  ? 100 :
    qq_energy == 200 ? 250 :
    qq_energy == 350 ? 250 :
    qq_energy == 500 ? 500 : 1;

const int total_number_of_event = 
    qq_energy == 40  ? num_file* 500 :
    qq_energy == 91  ? num_file* 500 :
    qq_energy == 200 ? num_file* 200 :
    qq_energy == 350 ? num_file* 200 :
    qq_energy == 500 ? num_file* 100 : 1;




// conditions
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_fixed_uds/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/tbeta090td050.root", qq_energy);
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_fixed_uds/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/raw_dd.root", qq_energy);
const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_fixed_uds/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/truth_clustering/dd_001.root", qq_energy);
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_fixed_uds/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/tbeta090td050.root", qq_energy);
// const string fileName = Form("../output/energy_regression_1to1/skimmed/tc_fixed_uds/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/truth_clustering/test_dd_001.root", qq_energy);

const bool saving_canvas = true;
// const string train_particle_type = "ntau_10GeV_10";         // ntau_10GeV_10    uds91   ntau_10to100GeV_10
const string train_particle_type =  qq_energy == 40  ? "ntau_10GeV_10" :
                                    qq_energy == 91  ? "uds91" : 
                                    qq_energy == 200 ? "uds91" :
                                    qq_energy == 350 ? "ntau_10to100GeV_10" :
                                    qq_energy == 500 ? "ntau_10to100GeV_10" : "ntau_10to100GeV_10";
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

// MC truth energy (GeV) bins for per-bin resolution / efficiency / purity vs energy
static const Double_t kTruthEnergyBinEdges[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
    40, 50, 60, 70, 80, 90, 100,
    125, 150, 175, 200, 225, 250
};
static const int kNTruthEnergyBins = sizeof(kTruthEnergyBinEdges) / sizeof(Double_t) - 1;

/** Bin index 0 .. kNTruthEnergyBins-1, or -1 if outside [edges[0], edges[nb]]. */
inline int truthEnergyBinFromMcen(double mcen) {
    const int nb = kNTruthEnergyBins;
    if (mcen < kTruthEnergyBinEdges[0] || mcen > kTruthEnergyBinEdges[nb])
        return -1;
    for (int i = 0; i < nb - 1; ++i) {
        if (mcen >= kTruthEnergyBinEdges[i] && mcen < kTruthEnergyBinEdges[i + 1])
            return i;
    }
    if (mcen >= kTruthEnergyBinEdges[nb - 1] && mcen <= kTruthEnergyBinEdges[nb])
        return nb - 1;
    return -1;
}

inline double truthEnergyBinCenter(int ie) {
    return 0.5 * (kTruthEnergyBinEdges[ie] + kTruthEnergyBinEdges[ie + 1]);
}
inline double truthEnergyBinHalfWidth(int ie) {
    return 0.5 * (kTruthEnergyBinEdges[ie + 1] - kTruthEnergyBinEdges[ie]);
}

/** Truth MC による分類: 0=荷電, 1=光子, 2=中性ハドロン（荷電・22 以外） */
inline int truthPhysicsCategoryFromMc(int mcpdg, int mccharge) {
    if (mccharge != 0) return 0;
    if (mcpdg == 22) return 1;
    return 2;
}

static const int kNTruthPhysCat = 3;
static const char* kTruthPhysCatName[kNTruthPhysCat] = {"charged", "photon", "neutral_hadron"};
static const char* kTruthPhysSigmaLabel[kNTruthPhysCat] = {"charged", "photon", "neutral"};

// fileNames[qq][i] はループで i=1..nfile を使う。500 GeV で nfile=500 のとき i=500 まで必要なので [501]。
string fileNames[3][501] = {};
string fileNames_raw[num_file] = {};
const string qqNames[3] = {"dd", "ss", "uu"};

bool exist_file(int energy, string particle_type, int num){
    if(energy==500){
        if(particle_type=="dd"&&num==335) return true;
        if(particle_type=="ss"&&num==38) return true;
    }
    if(energy==200){
        if(particle_type=="dd"&&num==16) return true;
        if(particle_type=="uu"&&num==223) return true;
    }
    if(energy==350){
        if(particle_type=="uu"&&num==218) return true;
    }

    return false;
}

double calculateRMS90(TH1F* h1, TF1* f1) {
    if (!h1 || h1->GetEntries() == 0) return 0;

    int nBins = h1->GetNbinsX();
    double totalEntries = h1->Integral();
    double targetEntries = 0.9 * totalEntries;

    double minWidth = 1e30;
    double bestRMS90 = 0;
    double bestLow = 0, bestHigh = 0;
    double bestMean = 0;

    // 1. 最小幅の90%区間をスキャン
    for (int i = 1; i <= nBins; ++i) {
        double sum = 0;
        for (int j = i; j <= nBins; ++j) {
            sum += h1->GetBinContent(j);
            if (sum >= targetEntries) {
                double lowEdge = h1->GetBinLowEdge(i);
                double highEdge = h1->GetBinLowEdge(j + 1);
                double width = highEdge - lowEdge;

                if (width < minWidth) {
                    minWidth = width;
                    bestLow = lowEdge;
                    bestHigh = highEdge;

                    // 区間内での統計量を計算
                    double sumX = 0, sumX2 = 0, nSub = 0;
                    for (int k = i; k <= j; ++k) {
                        double content = h1->GetBinContent(k);
                        double center = h1->GetBinCenter(k);
                        sumX  += content * center;
                        sumX2 += content * center * center;
                        nSub  += content;
                    }
                    bestMean = sumX / nSub;
                    const double var = sumX2 / nSub - bestMean * bestMean;
                    // 浮動小数点誤差で var<0 になり sqrt が NaN になるのを防ぐ
                    bestRMS90 = (var > 0) ? TMath::Sqrt(var) : 0.0;
                }
                break;
            }
        }
    }
    // 2. TF1が渡されている場合、その関数に計算結果を反映させる
    if (f1) {
        f1->SetRange(bestLow, bestHigh); // 関数の描画範囲を90%区間に限定
        f1->SetParameter(0, h1->GetBinContent(h1->GetMaximumBin())); // Constant
        f1->SetParameter(1, bestMean);                               // Mean
        f1->SetParameter(2, bestRMS90);                              // Sigma
        // 4. 90%区間内だけでフィットを実行
        // "R"オプションはSetRangeで指定した範囲でフィットするという意味
        // "Q"はQuiet（ログを出さない）、"L"はLikelihood（最尤法）などお好みで
        h1->Fit(f1, "RQ"); 
        
        // 5. 比較用にRMS90の値を何らかの形で保持
        // 例えば、関数のタイトルや別の変数に格納しておく
    }
    return bestRMS90;
}
double calculateRMS90(TH1F* h1, double* mean90_out) {
    if (!h1 || h1->GetEntries() == 0) {
        if (mean90_out) *mean90_out = 0;
        return 0;
    }

    int nBins = h1->GetNbinsX();
    double totalEntries = h1->Integral();
    double targetEntries = 0.9 * totalEntries;

    double minWidth = 1e30; // 最小幅を保持
    double bestRMS90 = 0;
    double bestMean = 0;
    
    // 窓（Window）をスライドさせて最小幅の90%区間を探す
    for (int i = 1; i <= nBins; ++i) {
        double sum = 0;
        for (int j = i; j <= nBins; ++j) {
            sum += h1->GetBinContent(j);
            
            if (sum >= targetEntries) {
                // 現在の窓の幅を計算
                double width = h1->GetBinLowEdge(j + 1) - h1->GetBinLowEdge(i);
                
                if (width < minWidth) {
                    minWidth = width;
                    
                    // この区間内でのRMSを計算
                    double sumX = 0;
                    double sumX2 = 0;
                    double nSub = 0;
                    for (int k = i; k <= j; ++k) {
                        double content = h1->GetBinContent(k);
                        double center = h1->GetBinCenter(k);
                        sumX  += content * center;
                        sumX2 += content * center * center;
                        nSub  += content;
                    }
                    bestMean = sumX / nSub;
                    const double var = sumX2 / nSub - bestMean * bestMean;
                    bestRMS90 = (var > 0) ? TMath::Sqrt(var) : 0.0;
                }
                break; // ターゲットに達したらjのループを抜けて次の開始bin(i)へ
            }
        }
    }
    if (mean90_out) *mean90_out = bestMean;
    return bestRMS90;
}
double calculateRMS90(TH1F* h1) {
    double dummy;
    return calculateRMS90(h1, &dummy);
}

void efficiency_purity_check_reco_effpur_contiribution(){ 
    int rawfilenum = num_file;
    int irawfilenum = 0;

    // if(hyper_parameter && fine_tuning){ // condition check
    //     cout << "something wrong with setting boolian " << endl;
    //     abort();
    // }

    TFile *filein[rawfilenum];
    TTree *tree[rawfilenum];
    TTree *tree_pred[rawfilenum];
    TTree *tree_reco[rawfilenum];
    TTree *tree_event[rawfilenum];
    TTree *tree_jet[rawfilenum];
    int entry_max[rawfilenum];
    int total_entry_max=0;
    string path_to_file = use_truth_clustering ? "truth_clustering" : "reco";
    string picDirectory = Form("figures/fixed_uds/%dGeV/%s",qq_energy, path_to_file.c_str());
    
    if(rawfilenum == 1) fileNames_raw[0] = Form("%s",fileName.c_str());
    else {
        for(int qq=0; qq<3; qq++){
            for(int i=1; i<=nfile; i++){
                if(exist_file(qq_energy, qqNames[qq], i)) continue;
                path_to_file = use_truth_clustering ? "truth_clustering" : "tbeta090td050";
                // fileNames[qq][i] = Form("../output/energy_regression_1to1/skimmed/tc_fixed_uds/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/%s/%s_%03d.root", qq_energy, path_to_file.c_str(), qqNames[qq].c_str(), i);
                // fileNames[qq][i] = Form("../output/energy_regression_1to1/skimmed/tc_fixed_uds/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/perh5file/%s_%03d.root", qq_energy, qqNames[qq].c_str(), i);
                fileNames[qq][i] = Form("../output/energy_regression_1to1/skimmed/tc_fixed_uds_brems/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/%s/%s_%03d.root", qq_energy, path_to_file.c_str(), qqNames[qq].c_str(), i);
                fileNames_raw[irawfilenum] = Form("%s",fileNames[qq][i].c_str());
                irawfilenum++;
            }
        }
    }
    cout << fileName << endl;
    cout << picDirectory << endl;
    

    int event, hitid, mcid, truthid, mcpdg, mccharge, mcstatus, ncluster, matched_ncluster, matched_cluster, cond_track;
    double mcmass, mcpx, mcpy, mcpz, mcen, edep, edep_reco, edep_match, pred_edep, pred_edep_cluster, cond_beta, pred_alpha;

    int _event, _hitid, _mcid, _truthid, _mcpdg, _mccharge, _mcstatus, _ncluster, _matched_ncluster, _matched_cluster, _pred_alpha;
    double _mcmass, _mcpx, _mcpy, _mcpz, _mcen, _edep, _edep_reco, _edep_match, _pred_edep, _pred_edep_cluster, _pred_beta;

    double _MC_jet_energy, _total_predicted_energy_truthBase, _total_predicted_energy_predBase;

    int reco_event, reco_cluster, reco_nhits, reco_mcid, reco_mcpdg, reco_mccharge, reco_mcstatus, reco_ntrack_hits, reco_cond_is_track, reco_matched_truth_pdgid, reco_npdg_comp;
    int reco_pdg_comp_ids[64];
    double reco_mcmass, reco_mcpx, reco_mcpy, reco_mcpz, reco_mcen, reco_edep_reco, reco_edep_mc, reco_edep_match, reco_pred_edep, reco_pred_edep_cluster, reco_cond_beta, reco_matched_truth_edep_frac;

    const int n_beta_thresholds = 10;  // 0, 0.1, 0.2, ..., 0.9
    struct EventRecoEnergySummary {
        double pred_energy_reco_track = 0.0;
        double pred_energy_cond_track = 0.0;
        double pred_energy_reco_track_beta[10] = {0};   // beta>=0, 0.1, ..., 0.9
        double pred_energy_cond_track_beta[10] = {0};
    };
    struct EventConfusionSummary {
        double reco_energy_sum = 0.0;
        double confusion_weighted_sum = 0.0;
    };
    struct EventTrackCategoryEnergySummary {
        double pred_energy_sum[6] = {0,0,0,0,0,0};
    };



    // TFile fileout("result_test_pandora.root","RECREATE");
    const string output_path = Form("../output/energy_regression_1to1/skimmed/results/tc_fixed_uds_brems/5D/%dGeV/%s.root", qq_energy, path_to_file.c_str());
    TFile fileout(output_path.c_str(),"RECREATE");


    const int nParticle = kaon_neutron ? 6 : 4;
    string particleNames_base[6] = {"electron", "pion", "photon", "neutron", "K0", "muon"};
    vector<int> particledgValues_base = {11,-11, 211,-211, 22, 2112, 130, 13,-13};
    vector<int> particledgValues_itr_base = {0,0, 1,1, 2, 3, 4, 5,5};
    vector<int> particledgValues_base_ = {11,-11, 211,-211, 22, 13,-13};
    vector<int> particledgValues_itr_base_ = {0,0, 1,1, 2, 3,3};

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
    const int nEnergy = kNTruthEnergyBins;
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
    TH1F *event_energy_diff_per_energy_reco_track[nEnergy_jet];
    TH1F *event_energy_diff_per_energy_cond_track[nEnergy_jet];
    TH1F *event_pred_energy_sum_by_track_category[6];
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
    TH2F *energy2d_truth_phys[kNTruthPhysCat];
    TH1F *energy_resolution_per_energy_truth_phys[kNTruthPhysCat][nEnergy];
    for(int ic=0; ic<kNTruthPhysCat; ic++){
        energy2d_truth_phys[ic] = new TH2F(Form("energy2d_truth_phys_%s", kTruthPhysCatName[ic]), Form("%s (truth);MC truth energy (GeV);predicted energy (GeV)", kTruthPhysCatName[ic]), energyMax * 10, 0, energyMax, energyMax * 10, 0, energyMax);
        energy2d_truth_phys[ic]->SetXTitle("MC truth energy (GeV)");
        energy2d_truth_phys[ic]->SetYTitle("predicted energy (GeV)");
        for(int ie=0; ie<nEnergy; ie++){
            string title = Form("%s truth (%.0f-%.0f GeV);(predicted - truth) / truth", kTruthPhysCatName[ic], kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]);
            energy_resolution_per_energy_truth_phys[ic][ie] = new TH1F(Form("energy_resolution_truth_phys_%s_%d", kTruthPhysCatName[ic], ie),title.c_str(), Eres_nbin, -Eres_range, Eres_range);
        }
    }
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
            string title = ie==0 ? Form("%s purity",particleNames[ip].c_str()) : Form("%s purity %.0f-%.0f GeV", particleNames[ip].c_str(), kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]);
            purity_energy[ip][ie] = new TH1F(Form("purity_%d_%d",ip,ie), title.c_str(), 101,0,1.01);
            purity_energy[ip][ie]->SetXTitle("purity (edep_match/edep_reco)");
            purity_energy_normalize[ip][ie] = new TH1F(Form("purity_norm_%d_%d",ip,ie), title.c_str(), 101,0,1.01);
            purity_energy_normalize[ip][ie]->SetXTitle("purity (edep_match/edep_reco)");
            title = ie==0 ? Form("%s efficiency",particleNames[ip].c_str()) : Form("%s efficiency %.0f-%.0f GeV", particleNames[ip].c_str(), kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]);
            efficiency_energy[ip][ie] = new TH1F(Form("efficiency_%d_%d",ip,ie), title.c_str(), 101,0,1.01);
            efficiency_energy[ip][ie]->SetXTitle("efficiency (edep_match/edep)");
            efficiency_energy_normalize[ip][ie] = new TH1F(Form("efficiency_normalize_%d_%d",ip,ie), title.c_str(), 101,0,1.01);
            efficiency_energy_normalize[ip][ie]->SetXTitle("efficiency (edep_match/edep)");

            title = ie==0 ? Form("%s ;predicted energy - MC truth energy",particleNames[ip].c_str()) : Form("%s (%.0f-%.0f GeV);predicted energy - MC truth energy",particleNames[ip].c_str(), kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]);
            // energy_diff_per_energy[ip][ie] = new TH1F(Form("energy_diff_per_energy%d_%d",ip,ie), title.c_str(), 120,-6,6);
            energy_diff_per_energy[ip][ie] = new TH1F(Form("energy_diff_per_energy%d_%d",ip,ie), title.c_str(), Eres_nbin,-Eres_range,Eres_range);
            title = ie==0 ? Form("%s ;(predicted - truth) / truth",particleNames[ip].c_str()) : Form("%s (%.0f-%.0f GeV);(predicted - truth) / truth",particleNames[ip].c_str(), kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]);
            // energy_resolution_per_energy[ip][ie] = new TH1F(Form("energy_resolution_per_energy_%d_%d",ip,ie), title.c_str(), 120,-6,6);
            energy_resolution_per_energy[ip][ie] = new TH1F(Form("energy_resolution_per_energy_%d_%d",ip,ie), title.c_str(), Eres_nbin,-Eres_range,Eres_range);
            energy_resolution_per_energy_cluster[ip][ie] = new TH1F(Form("energy_resolution_per_energy_cluster_%d_%d",ip,ie), title.c_str(), Eres_nbin,-Eres_range,Eres_range);
        }
    }
    TH2F *jet_energy2d = new TH2F(Form("jet_energy2d"), Form(";MC jet energy;predicted jet energy"), 1200,0,120,1200,0,120);
    TH2F *event_energy2d_reco_track = new TH2F(Form("event_energy2d_reco_track"), Form("Event energy (reco cluster track-based);MC event energy;predicted event energy"), 600,0,600,600,0,600);
    TH2F *event_energy2d_cond_track = new TH2F(Form("event_energy2d_cond_track"), Form("Event energy (condensation track-based);MC event energy;predicted event energy"), 600,0,600,600,0,600);
    TH1F *event_energy_sum_reco_track_distribution = new TH1F(Form("event_energy_sum_reco_track_distribution"), Form("Event reconstructed energy sum (reco cluster track-based);reconstructed event energy sum;entries"), 600,0,600);
    TH1F *event_energy_sum_cond_track_distribution = new TH1F(Form("event_energy_sum_cond_track_distribution"), Form("Event reconstructed energy sum (condensation track-based);reconstructed event energy sum;entries"), 600,0,600);
    TH1F *event_energy_sum_reco_track_distribution_beta[10];
    TH1F *event_energy_sum_cond_track_distribution_beta[10];
    for(int ib=0; ib<n_beta_thresholds; ib++){
        double beta_min = 0.1 * ib;
        event_energy_sum_reco_track_distribution_beta[ib] = new TH1F(Form("event_energy_sum_reco_track_distribution_beta_%.0f", beta_min*10), Form("Event energy sum (reco track, #beta>=%.1f);reconstructed event energy sum;entries", beta_min), 600,0,600);
        event_energy_sum_cond_track_distribution_beta[ib] = new TH1F(Form("event_energy_sum_cond_track_distribution_beta_%.0f", beta_min*10), Form("Event energy sum (cond track, #beta>=%.1f);reconstructed event energy sum;entries", beta_min), 600,0,600);
    }
    TH1F *clustering_correct_ratio_by_particle = new TH1F(Form("clustering_correct_ratio_by_particle"), Form("Reco clustering correctness by particle;particle;correct clustering ratio"), nParticle, 0, nParticle);
    TH2F *clustering_composition_matrix = new TH2F(Form("clustering_composition_matrix"), Form("Reco clustering composition matrix;true particle;component particle in cluster"), nParticle, 0, nParticle, nParticle, 0, nParticle);
    for(int ip=0; ip<nParticle; ip++){
        clustering_correct_ratio_by_particle->GetXaxis()->SetBinLabel(ip+1, particleNames[ip].c_str());
        clustering_composition_matrix->GetXaxis()->SetBinLabel(ip+1, particleNames[ip].c_str());
        clustering_composition_matrix->GetYaxis()->SetBinLabel(ip+1, particleNames[ip].c_str());
    }
    TH1F *track_pdg_charge_relation_ratio = new TH1F(Form("track_pdg_charge_relation_ratio"), Form("Track presence vs MC charge ratio;category;ratio"), 6, 0, 6);
    track_pdg_charge_relation_ratio->GetXaxis()->SetBinLabel(1, "track&charged&condTrack");
    track_pdg_charge_relation_ratio->GetXaxis()->SetBinLabel(2, "track&charged&!condTrack");
    track_pdg_charge_relation_ratio->GetXaxis()->SetBinLabel(3, "track&neutral&condTrack");
    track_pdg_charge_relation_ratio->GetXaxis()->SetBinLabel(4, "track&neutral&!condTrack");
    track_pdg_charge_relation_ratio->GetXaxis()->SetBinLabel(5, "no track & charged");
    track_pdg_charge_relation_ratio->GetXaxis()->SetBinLabel(6, "no track & neutral");
    TH2F *track_pdg_charge_vs_condbeta = new TH2F(Form("track_pdg_charge_vs_condbeta"), Form("Track/charge category vs condensation beta;track/charge category;condensation point beta"), 6, 0, 6, 100, 0, 1);
    track_pdg_charge_vs_condbeta->GetXaxis()->SetBinLabel(1, "track&charged&condTrack");
    track_pdg_charge_vs_condbeta->GetXaxis()->SetBinLabel(2, "track&charged&!condTrack");
    track_pdg_charge_vs_condbeta->GetXaxis()->SetBinLabel(3, "track&neutral&condTrack");
    track_pdg_charge_vs_condbeta->GetXaxis()->SetBinLabel(4, "track&neutral&!condTrack");
    track_pdg_charge_vs_condbeta->GetXaxis()->SetBinLabel(5, "no track & charged");
    track_pdg_charge_vs_condbeta->GetXaxis()->SetBinLabel(6, "no track & neutral");
    const char* track_category_labels[6] = {
        "track&charged&condTrack",
        "track&charged&!condTrack",
        "track&neutral&condTrack",
        "track&neutral&!condTrack",
        "no track & charged",
        "no track & neutral"
    };
    TH2F *track_charge_vs_truth_particle = new TH2F(Form("track_charge_vs_truth_particle"), Form("Track/charge category vs truth particle;track/charge category;truth particle"), 6, 0, 6, nParticle, 0, nParticle);
    for(int ic=0; ic<6; ic++){
        track_charge_vs_truth_particle->GetXaxis()->SetBinLabel(ic+1, track_category_labels[ic]);
    }
    for(int ip=0; ip<nParticle; ip++){
        track_charge_vs_truth_particle->GetYaxis()->SetBinLabel(ip+1, particleNames[ip].c_str());
    }
    for(int ic=0; ic<6; ic++){
        event_pred_energy_sum_by_track_category[ic] = new TH1F(Form("event_pred_energy_sum_by_track_category_%d",ic+1),Form("%s;event pred energy sum;entries", track_category_labels[ic]),6000, 0, 600);
    }
    TH1F *c_event_distribution = new TH1F(Form("c_event_distribution"), Form("Event confusion index C_{event};C_{event};entries"), 100, 0, 1);
    TH2F *c_event_vs_truth_energy = new TH2F(Form("c_event_vs_truth_energy"), Form("Event confusion index vs truth energy;truth event energy;C_{event}"), 600, 0, 600, 100, 0, 1);
    TH1F *event_residual_all_reco_track = new TH1F(Form("event_residual_all_reco_track"), Form("Reco-track based;(pred-truth)/truth;entries"), Eres_nbin, -Eres_range, Eres_range );
    TH1F *event_residual_clean_reco_track = new TH1F(Form("event_residual_clean_reco_track"), Form("Reco-track based clean;(pred-truth)/truth;entries"), Eres_nbin, -Eres_range, Eres_range);
    TH1F *event_residual_all_cond_track = new TH1F(Form("event_residual_all_cond_track"), Form("Cond-track based;(pred-truth)/truth;entries"), Eres_nbin, -Eres_range, Eres_range);
    TH1F *event_residual_clean_cond_track = new TH1F(Form("event_residual_clean_cond_track"), Form("Cond-track based clean;(pred-truth)/truth;entries"), Eres_nbin, -Eres_range, Eres_range);
    TH1F *jet_energy_resolution_per_energy[nEnergy_jet];
    for(int ie=0;ie<nEnergy_jet;ie++){
        string title = ie==0 ? Form("jet;(predicted - truth) / truth") : Form("jet (%d-%d GeV);(predicted - truth) / truth",(int)(ie*10),(int)((ie+1)*10));
        jet_energy_resolution_per_energy[ie] = new TH1F(Form("energy_resolution_per_energy_%d",ie), title.c_str(), 400,-0.5,0.5);

        string diff_title = Form("Event energy (%d-%d GeV);predicted energy - truth energy",(int)(ie*10),(int)((ie+1)*10));
        string res_title = Form("Event energy (%d-%d GeV);(predicted - truth) / truth",(int)(ie*10),(int)((ie+1)*10));
        event_energy_diff_per_energy_reco_track[ie] = new TH1F(Form("event_energy_diff_per_energy_reco_track_%d",ie), diff_title.c_str(), Eres_nbin,-Eres_range,Eres_range);
        event_energy_diff_per_energy_cond_track[ie] = new TH1F(Form("event_energy_diff_per_energy_cond_track_%d",ie), diff_title.c_str(), Eres_nbin,-Eres_range,Eres_range);
    }


    // data をとってきてる
    int clustering_total_by_particle[nParticle] = {0};
    int clustering_correct_by_particle[nParticle] = {0};
    int track_charge_total = 0;
    int n_track_cond_charged = 0;
    int n_track_notcond_charged = 0;
    int n_track_cond_neutral = 0;
    int n_track_notcond_neutral = 0;
    int n_notrack_charged = 0;
    int n_notrack_neutral = 0;
    for(int irawfile=0; irawfile<rawfilenum; irawfile++){
        if(rawfilenum>1) cout << irawfile << "/" << rawfilenum << "  ";// << endl;
        map<int, double> event_truth_energy_sum_ttree;

        cout << fileNames_raw[irawfile] << endl;
        filein[irawfile] = new TFile(Form("%s",fileNames_raw[irawfile].c_str()));
        tree[irawfile] = (TTree*) filein[irawfile]->Get("t");
        if(!tree[irawfile]){
            cout << "   error in root file conversion!!!!!" << endl;
            continue;
        }
        entry_max[irawfile] = tree[irawfile]->GetEntries();
        tree_pred[irawfile] = (TTree*) filein[irawfile]->Get("prediction");
        tree_reco[irawfile] = (TTree*) filein[irawfile]->Get("reco");
        tree_event[irawfile] = (TTree*) filein[irawfile]->Get("event");
        if(jet_regression) tree_jet[irawfile] = (TTree*) filein[irawfile]->Get("jet");

        tree[irawfile]->SetBranchAddress("event", &event);
        tree[irawfile]->SetBranchAddress("hitid", &hitid);
        tree[irawfile]->SetBranchAddress("mcid", &mcid);
        tree[irawfile]->SetBranchAddress("mcpdg", &mcpdg);
        tree[irawfile]->SetBranchAddress("mccharge", &mccharge);
        tree[irawfile]->SetBranchAddress("mcen", &mcen);
        tree[irawfile]->SetBranchAddress("edep", &edep);
        tree[irawfile]->SetBranchAddress("edep_reco", &edep_reco);
        tree[irawfile]->SetBranchAddress("edep_match", &edep_match);
        tree[irawfile]->SetBranchAddress("pred_edep", &pred_edep);
        tree[irawfile]->SetBranchAddress("pred_edep_cluster", &pred_edep_cluster);
        tree[irawfile]->SetBranchAddress("cond_beta", &cond_beta);
        tree[irawfile]->SetBranchAddress("cond_track", &cond_track);

        for(int ientry=0; ientry<entry_max[irawfile]; ientry++){
            if(rawfilenum==1 && entry_max[irawfile]>10000 && ientry%100000==0) cout << "calcualting truth total energy : event " << ientry << "/" << entry_max[irawfile] << endl;
            tree[irawfile]->GetEntry(ientry);
            if(mcen>0){
                event_truth_energy_sum_ttree[event] += mcen;
            }
        }
        for(int ientry=0; ientry<entry_max[irawfile]; ientry++){
            if(rawfilenum==1 && entry_max[irawfile]>10000 && ientry%100000==0) cout << "t event " << ientry << "/" << entry_max[irawfile] << endl;
            tree[irawfile]->GetEntry(ientry);
            
            auto truth_it_reco = event_truth_energy_sum_ttree.find(event);
            if(truth_it_reco == event_truth_energy_sum_ttree.end()) continue;
            const double truth_energy_reco = truth_it_reco->second;
            // if(truth_energy_reco<reco_truth_energy_min || truth_energy_reco>reco_truth_energy_max) continue;


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
            int energy_itr = truthEnergyBinFromMcen(mcen);
            if(energy_itr >= 0){
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

            const int ipc = truthPhysicsCategoryFromMc(mcpdg, mccharge);
            const double ipc_energy = ipc == 0 ? pred_edep : pred_edep_cluster;
            energy2d_truth_phys[ipc]->Fill(mcen, ipc_energy);

            // cout << itr_energy << ", " << mcen << endl;
            if(energy_itr >= 0 && pred_edep>0.1){
                energy_diff_per_energy[itr][energy_itr]->Fill(pred_edep - mcen);
                energy_resolution_per_energy[itr][energy_itr]->Fill( (pred_edep - mcen) / mcen );
            }
            if(energy_itr >= 0 && pred_edep_cluster>0.1){
                energy_resolution_per_energy_cluster[itr][energy_itr]->Fill( (pred_edep_cluster - mcen) / mcen );
            }
            if(energy_itr >= 0){
                energy_resolution_per_energy_truth_phys[ipc][energy_itr]->Fill((ipc_energy - mcen) / mcen);
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

        
        if(0){
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
            if(rawfilenum==1 && tree_pred[irawfile]->GetEntries()>1000 && ientry%10000000==0) cout << "predicted hit  " << ientry << "/" << tree_pred[irawfile]->GetEntries() << endl;
            tree_pred[irawfile]->GetEntry(ientry);
            if(_pred_alpha==1) continue;

            auto result = find(particledgValues.begin(), particledgValues.end(), _mcpdg);
            if(result == particledgValues.end()) continue;
            int itr = particledgValues_itr[distance(particledgValues.begin(), result)];
            // if(_pred_alpha==1) condbeta_vs_Ediff[itr]->Fill(_pred_beta,_pred_edep-_mcen);
            // if(_pred_alpha==1) condbeta_vs_mcen[itr]->Fill(_pred_beta,_mcen);
        }
        }


        if(jet_regression){
            tree_jet[irawfile]->SetBranchAddress("MC_jet_energy", &_MC_jet_energy);
            tree_jet[irawfile]->SetBranchAddress("total_predicted_energy_truthBase", &_total_predicted_energy_truthBase);
            tree_jet[irawfile]->SetBranchAddress("total_predicted_energy_predBase", &_total_predicted_energy_predBase);
            for(int ientry=0; ientry<tree_jet[irawfile]->GetEntries(); ientry++){
                if(rawfilenum==1 && tree_jet[irawfile]->GetEntries()>1000 && ientry%100000==0) cout << "jet event  " << ientry << "/" << tree_jet[irawfile]->GetEntries() << endl;
                tree_jet[irawfile]->GetEntry(ientry);

                jet_energy2d->Fill(_MC_jet_energy, _total_predicted_energy_predBase);
                int itr_energy = _MC_jet_energy/10;
                jet_energy_resolution_per_energy[itr_energy]->Fill( (_total_predicted_energy_predBase - _MC_jet_energy) / _MC_jet_energy );
            }
        }

        if(tree_reco[irawfile]){
            map<int, EventRecoEnergySummary> event_energy_summary;
            map<int, EventConfusionSummary> event_confusion_summary;
            map<int, EventTrackCategoryEnergySummary> event_track_category_energy_summary;
            tree_reco[irawfile]->SetBranchAddress("event", &reco_event);
            tree_reco[irawfile]->SetBranchAddress("cluster", &reco_cluster);
            tree_reco[irawfile]->SetBranchAddress("nhits", &reco_nhits);
            tree_reco[irawfile]->SetBranchAddress("mcid", &reco_mcid);
            tree_reco[irawfile]->SetBranchAddress("mcpdg", &reco_mcpdg);
            tree_reco[irawfile]->SetBranchAddress("mccharge", &reco_mccharge);
            tree_reco[irawfile]->SetBranchAddress("mcmass", &reco_mcmass);
            tree_reco[irawfile]->SetBranchAddress("mcpx", &reco_mcpx);
            tree_reco[irawfile]->SetBranchAddress("mcpy", &reco_mcpy);
            tree_reco[irawfile]->SetBranchAddress("mcpz", &reco_mcpz);
            tree_reco[irawfile]->SetBranchAddress("mcen", &reco_mcen);
            tree_reco[irawfile]->SetBranchAddress("mcstatus", &reco_mcstatus);
            tree_reco[irawfile]->SetBranchAddress("edep_reco", &reco_edep_reco);
            tree_reco[irawfile]->SetBranchAddress("edep_mc", &reco_edep_mc);
            tree_reco[irawfile]->SetBranchAddress("edep_match", &reco_edep_match);
            tree_reco[irawfile]->SetBranchAddress("pred_edep", &reco_pred_edep);
            tree_reco[irawfile]->SetBranchAddress("pred_edep_cluster", &reco_pred_edep_cluster);
            tree_reco[irawfile]->SetBranchAddress("ntrack_hits", &reco_ntrack_hits);
            tree_reco[irawfile]->SetBranchAddress("cond_beta", &reco_cond_beta);
            tree_reco[irawfile]->SetBranchAddress("cond_is_track", &reco_cond_is_track);
            tree_reco[irawfile]->SetBranchAddress("matched_truth_pdgid", &reco_matched_truth_pdgid);
            tree_reco[irawfile]->SetBranchAddress("matched_truth_edep_frac", &reco_matched_truth_edep_frac);
            tree_reco[irawfile]->SetBranchAddress("npdg_comp", &reco_npdg_comp);
            tree_reco[irawfile]->SetBranchAddress("pdg_comp_ids", reco_pdg_comp_ids);

            for(int ientry=0; ientry<tree_reco[irawfile]->GetEntries(); ientry++){
                if(rawfilenum==1 && tree_reco[irawfile]->GetEntries()>1000 && ientry%100000==0) cout << "reco event  " << ientry << "/" << tree_reco[irawfile]->GetEntries() << endl;
                tree_reco[irawfile]->GetEntry(ientry);
                if(reco_cond_beta<beta_threshold) continue;
                auto truth_it_reco = event_truth_energy_sum_ttree.find(reco_event);
                if(truth_it_reco == event_truth_energy_sum_ttree.end()) continue;
                const double truth_energy_reco = truth_it_reco->second;
                // if(truth_energy_reco<reco_truth_energy_min || truth_energy_reco>reco_truth_energy_max) continue;

                auto &sum = event_energy_summary[reco_event];
                const double pred_reco = (reco_ntrack_hits>0 ? reco_pred_edep : reco_pred_edep_cluster);
                const double pred_cond = (reco_cond_is_track!=0 ? reco_pred_edep : reco_pred_edep_cluster);
                sum.pred_energy_reco_track += pred_reco;
                sum.pred_energy_cond_track += pred_cond;
                for(int ib=0; ib<n_beta_thresholds; ib++){
                    double beta_min = 0.1 * ib;
                    if(reco_cond_beta >= beta_min){
                        sum.pred_energy_reco_track_beta[ib] += pred_reco;
                        sum.pred_energy_cond_track_beta[ib] += pred_cond;
                    }
                }
                auto &conf_sum = event_confusion_summary[reco_event];
                const double f_main = reco_matched_truth_edep_frac < 0 ? 0.0 : (reco_matched_truth_edep_frac > 1.0 ? 1.0 : reco_matched_truth_edep_frac);
                conf_sum.reco_energy_sum += reco_edep_reco;
                conf_sum.confusion_weighted_sum += reco_edep_reco * (1.0 - f_main);

                const bool has_track = (reco_ntrack_hits>0);
                const bool is_charged = (reco_mccharge!=0);
                const bool cond_is_track_flag = (reco_cond_is_track!=0);
                track_charge_total += 1;
                int track_charge_category = -1;
                if(has_track && is_charged && cond_is_track_flag){
                    n_track_cond_charged += 1;
                    track_charge_category = 1;
                }
                else if(has_track && is_charged && !cond_is_track_flag){
                    n_track_notcond_charged += 1;
                    track_charge_category = 2;
                }
                else if(has_track && !is_charged && cond_is_track_flag){
                    n_track_cond_neutral += 1;
                    track_charge_category = 3;
                }
                else if(has_track && !is_charged && !cond_is_track_flag){
                    n_track_notcond_neutral += 1;
                    track_charge_category = 4;
                }
                else if(!has_track && is_charged){
                    n_notrack_charged += 1;
                    track_charge_category = 5;
                }
                else {
                    n_notrack_neutral += 1;
                    track_charge_category = 6;
                }
                track_pdg_charge_vs_condbeta->Fill(track_charge_category - 0.5, reco_cond_beta);
                const double pred_energy_for_category = (reco_ntrack_hits>0 ? reco_pred_edep : reco_pred_edep_cluster);
                event_track_category_energy_summary[reco_event].pred_energy_sum[track_charge_category-1] += pred_energy_for_category;

                auto true_itr = find(particledgValues.begin(), particledgValues.end(), reco_matched_truth_pdgid);
                if(true_itr != particledgValues.end()){
                    int true_pid = particledgValues_itr[distance(particledgValues.begin(), true_itr)];
                    track_charge_vs_truth_particle->Fill(track_charge_category - 0.5, true_pid + 0.5);
                    clustering_total_by_particle[true_pid] += 1;
                    bool is_correct_cluster = true;
                    const int ncomp = reco_npdg_comp < 64 ? reco_npdg_comp : 64;
                    for(int ic=0; ic<ncomp; ic++){
                        auto comp_itr = find(particledgValues.begin(), particledgValues.end(), reco_pdg_comp_ids[ic]);
                        if(comp_itr == particledgValues.end()) continue;
                        int comp_pid = particledgValues_itr[distance(particledgValues.begin(), comp_itr)];
                        clustering_composition_matrix->Fill(true_pid + 0.5, comp_pid + 0.5);
                        if(comp_pid != true_pid) is_correct_cluster = false;
                    }
                    if(is_correct_cluster) clustering_correct_by_particle[true_pid] += 1;
                }
            }

            for(const auto &entry : event_energy_summary){
                const EventRecoEnergySummary &sum = entry.second;
                event_energy_sum_reco_track_distribution->Fill(sum.pred_energy_reco_track);
                event_energy_sum_cond_track_distribution->Fill(sum.pred_energy_cond_track);
                for(int ib=0; ib<n_beta_thresholds; ib++){
                    event_energy_sum_reco_track_distribution_beta[ib]->Fill(sum.pred_energy_reco_track_beta[ib]);
                    event_energy_sum_cond_track_distribution_beta[ib]->Fill(sum.pred_energy_cond_track_beta[ib]);
                }

                double c_event = -1.0;
                auto conf_it = event_confusion_summary.find(entry.first);
                if(conf_it != event_confusion_summary.end() && conf_it->second.reco_energy_sum>0){
                    c_event = conf_it->second.confusion_weighted_sum / conf_it->second.reco_energy_sum;
                    c_event_distribution->Fill(c_event);
                }

                auto truth_it = event_truth_energy_sum_ttree.find(entry.first);
                if(truth_it == event_truth_energy_sum_ttree.end()) continue;
                const double truth_energy = truth_it->second;
                if(c_event>=0){
                    c_event_vs_truth_energy->Fill(truth_energy, c_event);
                }
                if(truth_energy<=0) continue;
                const double residual_reco_track = (sum.pred_energy_reco_track - truth_energy) / truth_energy;
                const double residual_cond_track = (sum.pred_energy_cond_track - truth_energy) / truth_energy;
                event_residual_all_reco_track->Fill(residual_reco_track);
                event_residual_all_cond_track->Fill(residual_cond_track);
                if(c_event>=0 && c_event<c_event_clean_threshold){
                    event_residual_clean_reco_track->Fill(residual_reco_track);
                    event_residual_clean_cond_track->Fill(residual_cond_track);
                }
                // if(truth_energy<=0) continue;
                // int itr_energy = truth_energy / 10.0;
                // if(itr_energy<0 || itr_energy>=nEnergy_jet) continue;

                // event_energy_diff_per_energy_reco_track[itr_energy]->Fill(sum.pred_energy_reco_track - truth_energy);
                // event_energy_diff_per_energy_cond_track[itr_energy]->Fill(sum.pred_energy_cond_track - truth_energy);
                event_energy2d_reco_track->Fill(truth_energy, sum.pred_energy_reco_track);
                event_energy2d_cond_track->Fill(truth_energy, sum.pred_energy_cond_track);
            }
            for(const auto &entry : event_track_category_energy_summary){
                auto truth_it = event_truth_energy_sum_ttree.find(entry.first);
                if(truth_it == event_truth_energy_sum_ttree.end()) continue;
                const double truth_energy_reco = truth_it->second;
                // if(truth_energy_reco<reco_truth_energy_min || truth_energy_reco>reco_truth_energy_max) continue;
                const EventTrackCategoryEnergySummary &cat_sum = entry.second;
                for(int ic=0; ic<6; ic++){
                    event_pred_energy_sum_by_track_category[ic]->Fill(cat_sum.pred_energy_sum[ic]);
                }
            }
        }

        filein[irawfile]->Close();
    }

    for(int ip=0; ip<nParticle; ip++){
        double ratio = clustering_total_by_particle[ip] > 0 ? (double)clustering_correct_by_particle[ip] / clustering_total_by_particle[ip] : 0.0;
        clustering_correct_ratio_by_particle->SetBinContent(ip+1, ratio);
        double err = clustering_total_by_particle[ip] > 0 ? sqrt(ratio * (1.0-ratio) / clustering_total_by_particle[ip]) : 0.0;
        clustering_correct_ratio_by_particle->SetBinError(ip+1, err);
        cout << "clustering correctness " << particleNames[ip]
             << " : " << clustering_correct_by_particle[ip] << "/" << clustering_total_by_particle[ip]
             << " = " << ratio << endl;
    }
    if(track_charge_total>0){
        track_pdg_charge_relation_ratio->SetBinContent(1, (double)n_track_cond_charged / track_charge_total);
        track_pdg_charge_relation_ratio->SetBinContent(2, (double)n_track_notcond_charged / track_charge_total);
        track_pdg_charge_relation_ratio->SetBinContent(3, (double)n_track_cond_neutral / track_charge_total);
        track_pdg_charge_relation_ratio->SetBinContent(4, (double)n_track_notcond_neutral / track_charge_total);
        track_pdg_charge_relation_ratio->SetBinContent(5, (double)n_notrack_charged / track_charge_total);
        track_pdg_charge_relation_ratio->SetBinContent(6, (double)n_notrack_neutral / track_charge_total);
    }
    cout << "track/charge relation ratios (total-normalized)" << endl;
    cout << "  track & charged & condTrack : " << n_track_cond_charged << "/" << track_charge_total
         << " = " << (track_charge_total>0 ? (double)n_track_cond_charged/track_charge_total : 0) << endl;
    cout << "  track & charged & !condTrack : " << n_track_notcond_charged << "/" << track_charge_total
         << " = " << (track_charge_total>0 ? (double)n_track_notcond_charged/track_charge_total : 0) << endl;
    cout << "  track & neutral & condTrack : " << n_track_cond_neutral << "/" << track_charge_total
         << " = " << (track_charge_total>0 ? (double)n_track_cond_neutral/track_charge_total : 0) << endl;
    cout << "  track & neutral & !condTrack : " << n_track_notcond_neutral << "/" << track_charge_total
         << " = " << (track_charge_total>0 ? (double)n_track_notcond_neutral/track_charge_total : 0) << endl;
    cout << "  no track & charged : " << n_notrack_charged << "/" << track_charge_total
         << " = " << (track_charge_total>0 ? (double)n_notrack_charged/track_charge_total : 0) << endl;
    cout << "  no track & neutral : " << n_notrack_neutral << "/" << track_charge_total
         << " = " << (track_charge_total>0 ? (double)n_notrack_neutral/track_charge_total : 0) << endl;
    

    
    // gStyle->SetStatX(0.35);
    gStyle->SetOptStat("rme");
    gStyle->SetStatX(0.55);
    gStyle->SetStatY(0.9);
    gStyle->SetStatH(0.3);
    gStyle->SetStatW(0.4);
    // legends をもう少し大きくする

    
    TCanvas *compare = new TCanvas("compare","compare",2560,1440);
    compare->Divide(nParticle,2);
    for(int ip=0;ip<nParticle*2;ip++){
        compare->cd(ip+1);
        gPad->SetLogy();
        if(ip<nParticle) efficiency[ip]->Draw();
        else purity[ip-nParticle]->Draw();
        gPad->Update();  // stats box を生成
        TPaveStats *ps = (TPaveStats*) gPad->GetPrimitive("stats");
        if(ps){
            double y2 = ps->GetY2NDC();
            double y1 = ps->GetY1NDC();
            ps->SetY1NDC(y2 - (y2 - y1));
            gPad->Modified();
            gPad->Update();
        }
    }

    TCanvas *compare_energy = new TCanvas("compare_energy","compare_energy",2560,1440);
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
                led_com = Form("%.0f-%.0f GeV mean:%f StdErr:%f", kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1], efficiency_energy[ip][ie]->GetMean(), efficiency_energy[ip][ie]->GetMeanError());
                // led_com = Form("%.0f-%.0f GeV mean:%f StdDev:%f", kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1], efficiency_energy_normalize[ip][ie]->GetMean(), efficiency_energy_normalize[ip][ie]->GetStdDev());
                legend_comp_e[ip%nParticle][ip/nParticle]->AddEntry(efficiency_energy_normalize[ip][ie], led_com.c_str() , "l");
                // legend_comp_e[ip%nParticle][ip/nParticle]->AddEntry(efficiency_energy_normalize[ip][ie], Form("%.0f-%.0f GeV", kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]), "l");
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
                led_com = Form("%.0f-%.0f GeV mean:%f StdErr:%f", kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1], pur_mean[ip-nParticle][ie], pur_mean_error[ip-nParticle][ie]);
                // led_com = Form("%.0f-%.0f GeV mean:%f StdDev:%f", kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1], purity_energy_normalize[ip-nParticle][ie]->GetMean(), purity_energy_normalize[ip-nParticle][ie]->GetStdDev());
                legend_comp_e[ip%nParticle][ip/nParticle]->AddEntry(purity_energy_normalize[ip-nParticle][ie], led_com.c_str() , "l");
                // legend_comp_e[ip%nParticle][ip/nParticle]->AddEntry(purity_energy_normalize[ip-nParticle][ie], Form("%.0f-%.0f GeV", kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]), "l");
                legend_comp_e[ip%nParticle][ip/nParticle]->Draw("same");
                cout << "   " << led_com.c_str() << endl;
            }
        }
    }

    TCanvas *compare_per_energy = new TCanvas("compare_per_energy","compare_per_energy",2560,1440);
    compare_per_energy->Divide(2,1);
    compare_per_energy->cd();
    TGraphErrors *eff_per_energy[nParticle];
    TGraphErrors *pur_per_energy[nParticle];
    TLegend *legend_comp_per_e = new TLegend( 0.4, 0.1, 0.8, 0.4);;
    for(int ip=0;ip<nParticle;ip++){
        eff_per_energy[ip] = new TGraphErrors();
        eff_per_energy[ip]->GetXaxis()->SetTitle("Energy (GeV)");
        eff_per_energy[ip]->GetYaxis()->SetTitle("efficiency");
        pur_per_energy[ip] = new TGraphErrors();
        pur_per_energy[ip]->GetXaxis()->SetTitle("Energy (GeV)");
        pur_per_energy[ip]->GetYaxis()->SetTitle("purity");
        for(int ie=0;ie<nEnergy;ie++){
            eff_per_energy[ip]->SetPoint(ie, truthEnergyBinCenter(ie), eff_mean[ip][ie]);
            eff_per_energy[ip]->SetPointError(ie, truthEnergyBinHalfWidth(ie), eff_mean_error[ip][ie]);
            pur_per_energy[ip]->SetPoint(ie, truthEnergyBinCenter(ie), pur_mean[ip][ie]);
            pur_per_energy[ip]->SetPointError(ie, truthEnergyBinHalfWidth(ie), pur_mean_error[ip][ie]);
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

    
    TCanvas *canvas_energy = new TCanvas("canvas_energy","canvas_energy",2560,1440);
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
    TCanvas *canvas_energy_scan = new TCanvas("canvas_energy_scan","canvas_energy_scan",2560,1440);
    canvas_energy_scan->Divide(nParticle,2);
    canvas_energy_scan->cd();
    for(int ip=0;ip<nParticle;ip++){
        legend[ip] = new TLegend( 0.1, 0.6, 0.5, 0.9) ;
        canvas_energy_scan->cd(ip+1);
        // gPad->SetLogy();
        for(int ie=0;ie<nEnergy;ie++){
            // string drawOption = ie==0 ? "" : "same";
            legend[ip]->AddEntry(energy_diff_per_energy[ip][ie], Form("%.0f-%.0f GeV mean:%f StdDev:%f", kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1], energy_diff_per_energy[ip][ie]->GetMean(), energy_diff_per_energy[ip][ie]->GetStdDev()) , "l");
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
            legend_resolution[ip]->AddEntry(energy_resolution_per_energy[ip][ie], Form("%.0f-%.0f GeV mean:%f StdDev:%f", kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1], energy_resolution_per_energy[ip][ie]->GetMean(), energy_resolution_per_energy[ip][ie]->GetStdDev()) , "l");
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
    double resolution_rms90[nParticle][nEnergy];
    double resolution_sigma[nParticle][nEnergy];
    double resolution_sigma_error[nParticle][nEnergy];
    TF1 *gaus = new TF1("gaus", "gaus", -3,3);
    TGraphErrors *energy_resolution_rms[nParticle];
    TGraphErrors *energy_resolution_rms90[nParticle];
    TGraphErrors *energy_resolution_sigma[nParticle];
    TLegend *legend_res = new TLegend( 0.6, 0.76, 0.9, 0.9);
    legend_res->SetFillStyle(0);
    legend_res->SetBorderSize(1);
    for(int ip=0;ip<nParticle;ip++){
        cout << particleNames[ip] << endl;
        energy_resolution_rms[ip] = new TGraphErrors();
        energy_resolution_rms[ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
        energy_resolution_rms[ip]->GetXaxis()->SetTitle("GeV");
        energy_resolution_rms[ip]->GetYaxis()->SetTitle("#sigma of (pred-truth)/truth");
        energy_resolution_rms[ip]->SetLineColor(kRed);
        energy_resolution_rms[ip]->SetMarkerColor(kRed);
        energy_resolution_rms[ip]->SetMaximum(0.5);
        energy_resolution_rms[ip]->SetMinimum(0);
        energy_resolution_rms90[ip] = new TGraphErrors();
        energy_resolution_rms90[ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
        energy_resolution_rms90[ip]->GetXaxis()->SetTitle("GeV");
        energy_resolution_rms90[ip]->GetYaxis()->SetTitle("RMS_{90} of (pred-truth)/truth");
        energy_resolution_rms90[ip]->SetLineColor(kGreen + 1);
        energy_resolution_rms90[ip]->SetMarkerColor(kGreen + 1);
        energy_resolution_rms90[ip]->SetMaximum(0.5);
        energy_resolution_rms90[ip]->SetMinimum(0);
        energy_resolution_sigma[ip] = new TGraphErrors();
        energy_resolution_sigma[ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
        energy_resolution_sigma[ip]->SetLineColor(kBlue);
        energy_resolution_sigma[ip]->SetMarkerColor(kBlue);
        energy_resolution_sigma[ip]->SetMaximum(0.5);
        energy_resolution_sigma[ip]->SetMinimum(0);
        if(ip==0) { 
            legend_res->AddEntry(energy_resolution_rms[0], Form("rms") , "l");
            legend_res->AddEntry(energy_resolution_sigma[0], Form("gaussian #sigma") , "l");
            legend_res->SetTextSize(0.045);
        }
        canvas_energy_resolution_scan->cd(ip+1);
        gPad->SetLeftMargin(0.15);
        for(int ie=0;ie<nEnergy;ie++){
            // cout << energy_resolution_per_energy[ip][ie]->GetStdDev() << "    ";
            energy_resolution_per_energy[ip][ie]->SetAxisRange(-Eres_fit_range,Eres_fit_range);
            // cout << energy_resolution_per_energy[ip][ie]->GetStdDev() << endl;
            resolution_rms[ip][ie] = energy_resolution_per_energy[ip][ie]->GetStdDev();
            resolution_rms90[ip][ie] = calculateRMS90(energy_resolution_per_energy[ip][ie]);
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

            energy_resolution_rms[ip]->SetPoint(ie, truthEnergyBinCenter(ie), resolution_rms[ip][ie]);
            energy_resolution_rms[ip]->SetPointError(ie, truthEnergyBinHalfWidth(ie), 0);
            energy_resolution_rms90[ip]->SetPoint(ie, truthEnergyBinCenter(ie), resolution_rms90[ip][ie]);
            energy_resolution_rms90[ip]->SetPointError(ie, truthEnergyBinHalfWidth(ie), 0);
            energy_resolution_sigma[ip]->SetPoint(ie, truthEnergyBinCenter(ie), resolution_sigma[ip][ie]);
            energy_resolution_sigma[ip]->SetPointError(ie, truthEnergyBinHalfWidth(ie), resolution_sigma_error[ip][ie]);

            // energy_resolution_per_energy[ip][ie]->SetAxisRange(-Eres_range,Eres_range);
            // cout << "   " << resolution_rms[ip][ie] << ", " << resolution_sigma[ip][ie] << endl;
            // if( (ip<2 && ie>6) || (ip==2 && (ie>1 && ie<5)) ){
            //    string energy_range = Form("%d-%d GeV",(int)(ie*energy_interval),(int)((ie+1)*energy_interval));
            if( (ip<2 && kTruthEnergyBinEdges[ie] >= 28) || (ip==2 && kTruthEnergyBinEdges[ie] >= 8 && kTruthEnergyBinEdges[ie + 1] <= 20) ){
                string energy_range = Form("%.0f-%.0f GeV", kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]);
                cout << "  " << energy_range << " : " << resolution_sigma[ip][ie] << endl;
            }
            // energy_resolution_per_energy[ip][ie]->Rebin(rebin_factor);
            // energy_diff_per_energy[ip][ie]->Rebin(rebin_factor);
        }
        energy_resolution_rms[ip]->Draw("AP");
        energy_resolution_sigma[ip]->Draw("P");
        legend_res->Draw("same");
    }

    double phys_resolution_rms[kNTruthPhysCat][nEnergy];
    double phys_resolution_rms90[kNTruthPhysCat][nEnergy];
    double phys_resolution_sigma[kNTruthPhysCat][nEnergy];
    double phys_resolution_sigma_error[kNTruthPhysCat][nEnergy];
    /** Sum over truth-energy bins: entries_ie * (resolution_ie)^2, per category and estimator. */
    double phys_sum_N_res2_rms[kNTruthPhysCat] = {};
    double phys_sum_N_res2_rms90[kNTruthPhysCat] = {};
    double phys_sum_N_res2_sigma[kNTruthPhysCat] = {};
    TGraphErrors *phys_energy_resolution_rms[kNTruthPhysCat];
    TGraphErrors *phys_energy_resolution_rms90[kNTruthPhysCat];
    TGraphErrors *phys_energy_resolution_sigma[kNTruthPhysCat];
    for(int ic=0; ic<kNTruthPhysCat; ic++){
        phys_energy_resolution_rms[ic] = new TGraphErrors();
        phys_energy_resolution_rms[ic]->SetTitle(Form("%s (truth category)", kTruthPhysCatName[ic]));
        phys_energy_resolution_rms[ic]->GetXaxis()->SetTitle("truth energy (GeV)");
        phys_energy_resolution_rms[ic]->GetYaxis()->SetTitle("#sigma of (pred-truth)/truth");
        phys_energy_resolution_rms[ic]->SetLineColor(kRed);
        phys_energy_resolution_rms[ic]->SetMarkerColor(kRed);
        phys_energy_resolution_rms[ic]->SetMaximum(0.5);
        phys_energy_resolution_rms[ic]->SetMinimum(0);
        phys_energy_resolution_rms90[ic] = new TGraphErrors();
        phys_energy_resolution_rms90[ic]->SetTitle(Form("%s (truth category)", kTruthPhysCatName[ic]));
        phys_energy_resolution_rms90[ic]->GetXaxis()->SetTitle("truth energy (GeV)");
        phys_energy_resolution_rms90[ic]->GetYaxis()->SetTitle("RMS_{90} of (pred-truth)/truth");
        phys_energy_resolution_rms90[ic]->SetLineColor(kGreen + 1);
        phys_energy_resolution_rms90[ic]->SetMarkerColor(kGreen + 1);
        phys_energy_resolution_rms90[ic]->SetMaximum(0.5);
        phys_energy_resolution_rms90[ic]->SetMinimum(0);
        phys_energy_resolution_sigma[ic] = new TGraphErrors();
        phys_energy_resolution_sigma[ic]->SetTitle(Form("%s (truth category)", kTruthPhysCatName[ic]));
        phys_energy_resolution_sigma[ic]->SetLineColor(kBlue);
        phys_energy_resolution_sigma[ic]->SetMarkerColor(kBlue);
        phys_energy_resolution_sigma[ic]->SetMaximum(0.5);
        phys_energy_resolution_sigma[ic]->SetMinimum(0);
        for(int ie=0; ie<nEnergy; ie++){
            const double n_entries_bin = energy_resolution_per_energy_truth_phys[ic][ie]->GetEntries();
            energy_resolution_per_energy_truth_phys[ic][ie]->SetAxisRange(-Eres_fit_range, Eres_fit_range);
            phys_resolution_rms[ic][ie] = energy_resolution_per_energy_truth_phys[ic][ie]->GetStdDev();
            phys_resolution_rms90[ic][ie] = calculateRMS90(energy_resolution_per_energy_truth_phys[ic][ie]);
            if(energy_resolution_per_energy_truth_phys[ic][ie]->Integral(Eres_fitbin_lower, Eres_fitbin_upper) > 30){
                if(ic == 0)
                    energy_resolution_per_energy_truth_phys[ic][ie]->Fit("gaus", "NQ", "", -0.2, 0.2);
                else
                    energy_resolution_per_energy_truth_phys[ic][ie]->Fit("gaus", "NQ", "", -0.1, 0.1);
                phys_resolution_sigma[ic][ie] = gaus->GetParameter(2);
                phys_resolution_sigma_error[ic][ie] = gaus->GetParError(2);
            } else {
                energy_resolution_per_energy_truth_phys[ic][ie]->Rebin(4);
                energy_resolution_per_energy_truth_phys[ic][ie]->Fit("gaus", "NQ", "", -0.1, 0.1);
                phys_resolution_sigma[ic][ie] = gaus->GetParameter(2);
                phys_resolution_sigma_error[ic][ie] = gaus->GetParError(2);
                if(gaus->GetParameter(2) > 0.2){
                    phys_resolution_sigma[ic][ie] = -1;
                    phys_resolution_sigma_error[ic][ie] = 0;
                }
            }
            phys_energy_resolution_rms[ic]->SetPoint(ie, truthEnergyBinCenter(ie), phys_resolution_rms[ic][ie]);
            phys_energy_resolution_rms[ic]->SetPointError(ie, truthEnergyBinHalfWidth(ie), 0);
            phys_energy_resolution_rms90[ic]->SetPoint(ie, truthEnergyBinCenter(ie), phys_resolution_rms90[ic][ie]);
            phys_energy_resolution_rms90[ic]->SetPointError(ie, truthEnergyBinHalfWidth(ie), 0);
            phys_energy_resolution_sigma[ic]->SetPoint(ie, truthEnergyBinCenter(ie), phys_resolution_sigma[ic][ie]);
            phys_energy_resolution_sigma[ic]->SetPointError(ie, truthEnergyBinHalfWidth(ie), phys_resolution_sigma_error[ic][ie]);

            phys_sum_N_res2_rms[ic] += n_entries_bin * phys_resolution_rms[ic][ie] * phys_resolution_rms[ic][ie] * truthEnergyBinCenter(ie) * truthEnergyBinCenter(ie);
            phys_sum_N_res2_rms90[ic] += n_entries_bin * phys_resolution_rms90[ic][ie] * phys_resolution_rms90[ic][ie] * truthEnergyBinCenter(ie) * truthEnergyBinCenter(ie);
            if(phys_resolution_sigma[ic][ie] >= 0)
                phys_sum_N_res2_sigma[ic] += n_entries_bin * phys_resolution_sigma[ic][ie] * phys_resolution_sigma[ic][ie] * truthEnergyBinCenter(ie) * truthEnergyBinCenter(ie);
        }
    }

    double phys_sqrt_sum_res2_over_nev_rms[kNTruthPhysCat];
    double phys_sqrt_sum_res2_over_nev_rms90[kNTruthPhysCat];
    double phys_sqrt_sum_res2_over_nev_sigma[kNTruthPhysCat];
    for(int ic=0; ic<kNTruthPhysCat; ic++){
        phys_sqrt_sum_res2_over_nev_rms[ic] = TMath::Sqrt(phys_sum_N_res2_rms[ic] / total_number_of_event);
        phys_sqrt_sum_res2_over_nev_rms90[ic] = TMath::Sqrt(phys_sum_N_res2_rms90[ic] / total_number_of_event);
        phys_sqrt_sum_res2_over_nev_sigma[ic] = TMath::Sqrt(phys_sum_N_res2_sigma[ic] / total_number_of_event);
    }

    cout << "truth physics category: sqrt( sum_{truth-E bins} N_entries * (resolution)^2 / N_events )" << endl;
    cout << "  N_events (total MC events in skim, same for all categories) = " << total_number_of_event << endl;
    for(int ic=0; ic<kNTruthPhysCat; ic++){
        cout << "  " << kTruthPhysCatName[ic] << endl;
        cout << "    RMS:    sqrt(sum N*sigma^2 / N_evt) = " << phys_sqrt_sum_res2_over_nev_rms[ic] << endl;
        cout << "    RMS90:  sqrt(sum N*sigma^2 / N_evt) = " << phys_sqrt_sum_res2_over_nev_rms90[ic] << endl;
        cout << "    Gauss:  sqrt(sum N*sigma^2 / N_evt) = " << phys_sqrt_sum_res2_over_nev_sigma[ic] << endl;
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
    double cluster_resolution_rms90[nParticle][nEnergy];
    double cluster_resolution_sigma[nParticle][nEnergy];
    double cluster_resolution_sigma_error[nParticle][nEnergy];
    TGraphErrors *cluster_energy_resolution_rms[nParticle];
    TGraphErrors *cluster_energy_resolution_rms90[nParticle];
    TGraphErrors *cluster_energy_resolution_sigma[nParticle];
    TCanvas *canvas_cluster_energy_resolution_scan = new TCanvas("canvas_cluster_energy_resolution_scan","canvas_cluster_energy_resolution_scan",2560,1440);
    canvas_cluster_energy_resolution_scan->Divide(nParticle,2);
    canvas_cluster_energy_resolution_scan->cd();
    if(ECluster){
        for(int ip=0;ip<nParticle;ip++){
            legend_resolution_cluster[ip] = new TLegend( 0.1, 0.6, 0.5, 0.9) ;
            canvas_cluster_energy_resolution_scan->cd(ip+1);
            for(int ie=0;ie<nEnergy;ie++){
                // string drawOption_legend = ie==0 ? "" : "same";
                legend_resolution_cluster[ip]->AddEntry(energy_resolution_per_energy_cluster[ip][ie], Form("%.0f-%.0f GeV mean:%f StdDev:%f", kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1], energy_resolution_per_energy_cluster[ip][ie]->GetMean(), energy_resolution_per_energy_cluster[ip][ie]->GetStdDev()) , "l");
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
            gPad->SetLeftMargin(0.15);
            cout << particleNames[ip] << " cluster energy " << endl;
            cluster_energy_resolution_rms[ip] = new TGraphErrors();
            cluster_energy_resolution_rms[ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
            cluster_energy_resolution_rms[ip]->GetXaxis()->SetTitle("GeV");
            cluster_energy_resolution_rms[ip]->GetYaxis()->SetTitle("#sigma of (pred-truth)/truth");
            cluster_energy_resolution_rms[ip]->SetLineColor(kRed);
            cluster_energy_resolution_rms[ip]->SetMarkerColor(kRed);
            cluster_energy_resolution_rms[ip]->SetMaximum(0.5);
            cluster_energy_resolution_rms[ip]->SetMinimum(0);
            cluster_energy_resolution_rms90[ip] = new TGraphErrors();
            cluster_energy_resolution_rms90[ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
            cluster_energy_resolution_rms90[ip]->GetXaxis()->SetTitle("GeV");
            cluster_energy_resolution_rms90[ip]->GetYaxis()->SetTitle("RMS_{90} of (pred-truth)/truth");
            cluster_energy_resolution_rms90[ip]->SetLineColor(kGreen + 1);
            cluster_energy_resolution_rms90[ip]->SetMarkerColor(kGreen + 1);
            cluster_energy_resolution_rms90[ip]->SetMaximum(0.5);
            cluster_energy_resolution_rms90[ip]->SetMinimum(0);
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
                cluster_resolution_rms90[ip][ie] = calculateRMS90(energy_resolution_per_energy_cluster[ip][ie]);
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
    
                cluster_energy_resolution_rms[ip]->SetPoint(ie, truthEnergyBinCenter(ie), cluster_resolution_rms[ip][ie]);
                cluster_energy_resolution_rms[ip]->SetPointError(ie, truthEnergyBinHalfWidth(ie), 0);
                cluster_energy_resolution_rms90[ip]->SetPoint(ie, truthEnergyBinCenter(ie), cluster_resolution_rms90[ip][ie]);
                cluster_energy_resolution_rms90[ip]->SetPointError(ie, truthEnergyBinHalfWidth(ie), 0);
                cluster_energy_resolution_sigma[ip]->SetPoint(ie, truthEnergyBinCenter(ie), cluster_resolution_sigma[ip][ie]);
                cluster_energy_resolution_sigma[ip]->SetPointError(ie, truthEnergyBinHalfWidth(ie), cluster_resolution_sigma_error[ip][ie]);

                // energy_resolution_per_energy[ip][ie]->SetAxisRange(-Eres_range,Eres_range);
                // cout << "   " << resolution_rms[ip][ie] << ", " << resolution_sigma[ip][ie] << endl;
                string title = ie==0 ? Form("jet;(predicted - truth) / truth") : Form("jet (%.0f-%.0f GeV);(predicted - truth) / truth",kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]);
                // if( (ip<2 && ie>6) || (ip>=2 && (ie>1 && ie<5)) ){
                //     string energy_range = Form("%d-%d GeV",(int)(ie*energy_interval),(int)((ie+1)*energy_interval));
                if( (ip<2 && kTruthEnergyBinEdges[ie] >= 28) || (ip>=2 && kTruthEnergyBinEdges[ie] >= 8 && kTruthEnergyBinEdges[ie + 1] <= 20) ){
                    string energy_range = Form("%.0f-%.0f GeV", kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]);
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

    TLegend *legend_res_regression = new TLegend(0.55, 0.72, 0.9, 0.9);
    legend_res_regression->SetFillStyle(0);
    legend_res_regression->SetBorderSize(1);
    legend_res_regression->AddEntry(energy_resolution_rms[0], "rms", "l");
    legend_res_regression->AddEntry(energy_resolution_rms90[0], "rms_{90}", "l");
    legend_res_regression->AddEntry(energy_resolution_sigma[0], "gaussian #sigma", "l");
    legend_res_regression->SetTextSize(0.04);

    TCanvas *canvas_truth_phys_energy_regression = new TCanvas("canvas_truth_phys_energy_regression", "truth physics category: regression & resolution", 2560, 1440);
    canvas_truth_phys_energy_regression->Divide(kNTruthPhysCat, 2);
    for(int ic=0; ic<kNTruthPhysCat; ic++){
        canvas_truth_phys_energy_regression->cd(ic + 1);
        energy2d_truth_phys[ic]->SetStats(0);
        energy2d_truth_phys[ic]->Draw("colz");
        const double xrg_min = energy2d_truth_phys[ic]->GetXaxis()->GetXmin();
        const double xrg_max = energy2d_truth_phys[ic]->GetXaxis()->GetXmax();
        canvas_truth_phys_energy_regression->cd(ic + 1 + kNTruthPhysCat);
        gPad->SetLeftMargin(0.15);
        phys_energy_resolution_rms[ic]->Draw("AP");
        TH1 *hframe_phys = phys_energy_resolution_rms[ic]->GetHistogram();
        if(hframe_phys)
            hframe_phys->GetXaxis()->SetRangeUser(xrg_min, xrg_max);
        phys_energy_resolution_rms90[ic]->Draw("P");
        phys_energy_resolution_sigma[ic]->Draw("P");
        TLegend *leg_phys = new TLegend(0.45, 0.65, 0.9, 0.9);
        // leg_phys->SetFillStyle(0);
        leg_phys->SetBorderSize(1);
        leg_phys->AddEntry(phys_energy_resolution_rms[ic], Form("rms (#sigma_{%s} = %.4g)", kTruthPhysSigmaLabel[ic], phys_sqrt_sum_res2_over_nev_rms[ic]), "l");
        leg_phys->AddEntry(phys_energy_resolution_rms90[ic], Form("rms_{90} (#sigma_{%s} = %.4g)", kTruthPhysSigmaLabel[ic], phys_sqrt_sum_res2_over_nev_rms90[ic]), "l");
        leg_phys->AddEntry(phys_energy_resolution_sigma[ic], Form("#sigma (#sigma_{%s} = %.4g)", kTruthPhysSigmaLabel[ic], phys_sqrt_sum_res2_over_nev_sigma[ic]), "l");
        leg_phys->SetTextSize(0.04);
        leg_phys->Draw("same");

        gPad->Modified();
        gPad->Update();
    }

    TCanvas *canvas_energy_regression_result = new TCanvas("canvas_energy_regression_result","canvas_energy_regression_result",2560,1440);
    canvas_energy_regression_result->Divide(nParticle,2);
    for(int ip=0;ip<nParticle;ip++){
        // electron / charged pion と同じ reco track ベースの energy regression（pred_edep vs MC）
        const bool use_track_energy_regression = (ip < 2 || particleNames[ip] == "muon");
        canvas_energy_regression_result->cd(ip+1);
        energy2d[ip]->SetStats(0);
        clusterenergy2d[ip]->SetStats(0);
        TH2F *e2d_top = use_track_energy_regression ? energy2d[ip] : clusterenergy2d[ip];
        e2d_top->Draw("colz");
        const double xrg_min = e2d_top->GetXaxis()->GetXmin();
        const double xrg_max = e2d_top->GetXaxis()->GetXmax();

        canvas_energy_regression_result->cd(ip+1+nParticle);
        gPad->SetLeftMargin(0.15);
        if(use_track_energy_regression){
            energy_resolution_rms[ip]->Draw("AP");
            TH1 *hframe = energy_resolution_rms[ip]->GetHistogram();
            if(hframe)
                hframe->GetXaxis()->SetRangeUser(xrg_min, xrg_max);
            energy_resolution_rms90[ip]->Draw("P");
            energy_resolution_sigma[ip]->Draw("P");
            legend_res_regression->Draw("same");
        } else {
            cluster_energy_resolution_rms[ip]->Draw("AP");
            TH1 *hframe = cluster_energy_resolution_rms[ip]->GetHistogram();
            if(hframe)
                hframe->GetXaxis()->SetRangeUser(xrg_min, xrg_max);
            cluster_energy_resolution_rms90[ip]->Draw("P");
            cluster_energy_resolution_sigma[ip]->Draw("P");
            legend_res_regression->Draw("same");
        }
        gPad->Modified();
        gPad->Update();
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


    TCanvas *canvas_event_energy_resolution = new TCanvas("canvas_event_energy_resolution","canvas_event_energy_resolution",1400,500);
    canvas_event_energy_resolution->Divide(2,1);
    TF1 *gaus_event_reco_track = new TF1("gaus_event_reco_track","gaus",0,600);
    TF1 *gaus_event_cond_track = new TF1("gaus_event_cond_track","gaus",0,600);
    gaus_event_reco_track->SetLineColor(kRed);
    gaus_event_cond_track->SetLineColor(kRed);
    TF1 *gaus_event_reco_track_rms90 = new TF1("gaus_event_reco_track_rms90","gaus",0,600);
    TF1 *gaus_event_cond_track_rms90 = new TF1("gaus_event_cond_track_rms90","gaus",0,600);
    gaus_event_reco_track_rms90->SetLineColor(kBlue);
    gaus_event_cond_track_rms90->SetLineColor(kBlue);
    double reco_track_rms90=0, cond_track_rms90=0;
    // canvas_event_energy_resolution のみ optFit 表示（描画時に一時的に gStyle を変更し、描画後に TPaveStats を直接設定して他キャンバスへの影響を防ぐ）
    {
        gStyle->SetOptFit(1111);
        // gStyle->SetStatX(0.9);
        // gStyle->SetStatY(0.9);
        // gStyle->SetStatW(0.2);
        // gStyle->SetStatH(0.14);
        gStyle->SetStatX(0.475);
        gStyle->SetStatY(0.9);
        gStyle->SetStatW(0.2);
        gStyle->SetStatH(0.14);
    }
    canvas_event_energy_resolution->cd(1);
    event_energy_sum_reco_track_distribution->SetStats(1);
    event_energy_sum_reco_track_distribution->SetLineColor(kBlack);
    event_energy_sum_reco_track_distribution->SetAxisRange(0,histo_energy_range_max, "X");
    event_energy_sum_reco_track_distribution->Draw("HIST");
    if(event_energy_sum_reco_track_distribution->GetEntries()>30){
        gaus_event_reco_track->SetParameters(
            event_energy_sum_reco_track_distribution->GetMaximum(),
            event_energy_sum_reco_track_distribution->GetMean(),
            event_energy_sum_reco_track_distribution->GetStdDev()>1e-6 ? event_energy_sum_reco_track_distribution->GetStdDev() : 10.0
        );
        reco_track_rms90 = calculateRMS90(event_energy_sum_reco_track_distribution, gaus_event_reco_track_rms90);
        gaus_event_reco_track_rms90->Draw("same");
        event_energy_sum_reco_track_distribution->Fit("gaus_event_reco_track","Q","",0,qq_energy*1.5);
        gaus_event_reco_track->Draw("same");

        TPaveText *pt_reco = new TPaveText(0.1, 0.4, 0.4, 0.55, "NDC");
        pt_reco->SetFillColor(0); // 背景を白に
        pt_reco->SetBorderSize(1); // 枠線を細く
        pt_reco->SetTextAlign(12); // 左揃え・上下中央
        pt_reco->SetTextFont(42);  // 標準的なサンセリフ体
        pt_reco->AddText(Form("RMS_{90} = %.3f", reco_track_rms90));
        pt_reco->AddText(Form("Fit #sigma_{90} = %.3f", gaus_event_reco_track_rms90->GetParameter(2)));
        pt_reco->Draw();
    }
    canvas_event_energy_resolution->cd(2);
    event_energy_sum_cond_track_distribution->SetStats(1);
    event_energy_sum_cond_track_distribution->SetLineColor(kBlack);
    event_energy_sum_cond_track_distribution->SetAxisRange(0,histo_energy_range_max, "X");
    event_energy_sum_cond_track_distribution->Draw("HIST");
    if(event_energy_sum_cond_track_distribution->GetEntries()>30){
        gaus_event_cond_track->SetParameters(
            event_energy_sum_cond_track_distribution->GetMaximum(),
            event_energy_sum_cond_track_distribution->GetMean(),
            event_energy_sum_cond_track_distribution->GetStdDev()>1e-6 ? event_energy_sum_cond_track_distribution->GetStdDev() : 10.0
        );
        cond_track_rms90 = calculateRMS90(event_energy_sum_cond_track_distribution, gaus_event_cond_track_rms90);
        gaus_event_cond_track_rms90->Draw("same");
        event_energy_sum_cond_track_distribution->Fit("gaus_event_cond_track","Q","",0,qq_energy*1.5);
        gaus_event_cond_track->Draw("same");

        TPaveText *pt_cond = new TPaveText(0.1, 0.4, 0.4, 0.55, "NDC");
        pt_cond->SetFillColor(0);
        pt_cond->SetBorderSize(1);
        pt_cond->SetTextAlign(12);
        pt_cond->SetTextFont(42);
        
        pt_cond->AddText(Form("RMS_{90} = %.3f", cond_track_rms90));
        pt_cond->AddText(Form("Fit #sigma_{90} = %.3f", gaus_event_cond_track_rms90->GetParameter(2)));
        pt_cond->Draw();
    }
    cout << "event reconstructed energy sum gaussian fit" << endl;
    if(event_energy_sum_reco_track_distribution->GetEntries()>30){
        cout << "  reco cluster track-based : mean=" << gaus_event_reco_track->GetParameter(1) << ", sigma=" << gaus_event_reco_track->GetParameter(2) << endl;
        cout << "                     rms90 : mean=" << gaus_event_reco_track_rms90->GetParameter(1) << ", sigma=" << gaus_event_reco_track_rms90->GetParameter(2) << ", RMS90=" << reco_track_rms90 << endl;
    }
    if(event_energy_sum_cond_track_distribution->GetEntries()>30){
        cout << "  condensation track-based : mean=" << gaus_event_cond_track->GetParameter(1) << ", sigma=" << gaus_event_cond_track->GetParameter(2) << endl;
        cout << "                     rms90 : mean=" << gaus_event_cond_track_rms90->GetParameter(1) << ", sigma=" << gaus_event_cond_track_rms90->GetParameter(2) << ", RMS90=" << cond_track_rms90 << endl;
    }
    // canvas_event_energy_resolution の TPaveStats に optFit と位置を直接設定（gStyle に依存しないようにする）
    for(int ipad=1; ipad<=2; ipad++){
        canvas_event_energy_resolution->cd(ipad);
        TPaveStats *ps = (TPaveStats*) gPad->GetPrimitive("stats");
        if(ps){
            ps->SetOptFit(1111);
            ps->SetX1NDC(0.7);
            ps->SetY1NDC(0.76);
            ps->SetX2NDC(0.9);
            ps->SetY2NDC(0.9);
        }
    }
    // gStyle を optStat に戻す（他キャンバス用）
    gStyle->SetOptStat("rme");
    gStyle->SetStatX(0.55);
    gStyle->SetStatY(0.9);
    gStyle->SetStatH(0.3);
    gStyle->SetStatW(0.4);

    TCanvas *canvas_event_energy_sum_by_beta = new TCanvas("canvas_event_energy_sum_by_beta","canvas_event_energy_sum_by_beta",1800,1000);
    canvas_event_energy_sum_by_beta->Divide(2,1);
    TLegend *legend_beta_reco = new TLegend(0.5, 0.5, 0.95, 0.95);
    TLegend *legend_beta_cond = new TLegend(0.5, 0.5, 0.95, 0.95);
    canvas_event_energy_sum_by_beta->cd(1);
    gPad->SetLogy();
    for(int ib=0; ib<n_beta_thresholds; ib++){
        double beta_min = 0.1 * ib;
        int colorId = ib<9 ? ib+1 : ib+2;
        event_energy_sum_reco_track_distribution_beta[ib]->SetLineColor(colorId);
        event_energy_sum_reco_track_distribution_beta[ib]->SetAxisRange(0,histo_energy_range_max, "X");
        event_energy_sum_reco_track_distribution_beta[ib]->SetStats(0);
        string drawOpt = ib==0 ? "HIST" : "HIST same";
        event_energy_sum_reco_track_distribution_beta[ib]->Draw(drawOpt.c_str());
        double mean90, rms90;
        rms90 = calculateRMS90(event_energy_sum_reco_track_distribution_beta[ib], &mean90);
        legend_beta_reco->AddEntry(event_energy_sum_reco_track_distribution_beta[ib], Form("#beta>=%.1f: mean_{90}=%.2f RMS_{90}=%.2f", beta_min, mean90, rms90), "l");
    }
    legend_beta_reco->SetFillStyle(0);
    legend_beta_reco->Draw("same");
    canvas_event_energy_sum_by_beta->cd(2);
    gPad->SetLogy();
    for(int ib=0; ib<n_beta_thresholds; ib++){
        double beta_min = 0.1 * ib;
        int colorId = ib<9 ? ib+1 : ib+2;
        event_energy_sum_cond_track_distribution_beta[ib]->SetLineColor(colorId);
        event_energy_sum_cond_track_distribution_beta[ib]->SetAxisRange(0,histo_energy_range_max, "X");
        event_energy_sum_cond_track_distribution_beta[ib]->SetStats(0);
        string drawOpt = ib==0 ? "HIST" : "HIST same";
        event_energy_sum_cond_track_distribution_beta[ib]->Draw(drawOpt.c_str());
        double mean90, rms90;
        rms90 = calculateRMS90(event_energy_sum_cond_track_distribution_beta[ib], &mean90);
        legend_beta_cond->AddEntry(event_energy_sum_cond_track_distribution_beta[ib], Form("#beta>=%.1f: mean_{90}=%.2f RMS_{90}=%.2f", beta_min, mean90, rms90), "l");
    }
    legend_beta_cond->SetFillStyle(0);
    legend_beta_cond->Draw("same");

    TCanvas *canvas_event_energy_2d = new TCanvas("canvas_event_energy_2d","canvas_event_energy_2d",1400,500);
    canvas_event_energy_2d->Divide(2,1);
    canvas_event_energy_2d->cd(1);
    event_energy2d_reco_track->SetAxisRange(0,histo_energy_range_max, "X");
    event_energy2d_reco_track->SetAxisRange(0,histo_energy_range_max_y, "Y");
    event_energy2d_reco_track->SetStats(0);
    event_energy2d_reco_track->Draw("colz");
    canvas_event_energy_2d->cd(2);
    event_energy2d_cond_track->SetStats(0);
    event_energy2d_cond_track->SetAxisRange(0,histo_energy_range_max, "X");
    event_energy2d_cond_track->SetAxisRange(0,histo_energy_range_max_y, "Y");
    event_energy2d_cond_track->Draw("colz");

    TCanvas *canvas_clustering_quality = new TCanvas("canvas_clustering_quality","canvas_clustering_quality",1800,1800);
    canvas_clustering_quality->Divide(2,2);
    canvas_clustering_quality->cd(1);
    clustering_correct_ratio_by_particle->SetMinimum(0);
    clustering_correct_ratio_by_particle->SetMaximum(1.05);
    clustering_correct_ratio_by_particle->SetStats(0);
    clustering_correct_ratio_by_particle->Draw("E1");
    canvas_clustering_quality->cd(2);
    clustering_composition_matrix->SetStats(0);
    gPad->SetLogz();
    clustering_composition_matrix->SetMarkerSize(2);
    clustering_composition_matrix->Draw("colz text");
    canvas_clustering_quality->cd(3);
    track_pdg_charge_relation_ratio->SetStats(0);
    track_pdg_charge_relation_ratio->SetMinimum(0);
    track_pdg_charge_relation_ratio->SetMaximum(1.05);
    track_pdg_charge_relation_ratio->Draw("HIST");
    canvas_clustering_quality->cd(4);
    track_pdg_charge_vs_condbeta->SetStats(0);
    track_pdg_charge_vs_condbeta->Draw("colz");

    TCanvas *canvas_track_charge_vs_truth_particle = new TCanvas("canvas_track_charge_vs_truth_particle","canvas_track_charge_vs_truth_particle",1200,800);
    canvas_track_charge_vs_truth_particle->cd();
    track_charge_vs_truth_particle->SetStats(0);
    gPad->SetLogz();
    track_charge_vs_truth_particle->SetMarkerSize(2);
    track_charge_vs_truth_particle->Draw("colz text");

    TCanvas *canvas_c_event = new TCanvas("canvas_c_event","canvas_c_event",1400,500);
    canvas_c_event->Divide(2,1);
    canvas_c_event->cd(1);
    c_event_distribution->SetStats(1);
    c_event_distribution->Draw("HIST");
    canvas_c_event->cd(2);
    c_event_vs_truth_energy->SetStats(0);
    c_event_vs_truth_energy->Draw("colz");

    TCanvas *canvas_event_pred_energy_by_track_category = new TCanvas("canvas_event_pred_energy_by_track_category","canvas_event_pred_energy_by_track_category",1800,900);
    canvas_event_pred_energy_by_track_category->Divide(3,2);
    for(int ic=0; ic<6; ic++){
        canvas_event_pred_energy_by_track_category->cd(ic+1);
        gPad->SetLogy();
        event_pred_energy_sum_by_track_category[ic]->SetAxisRange(0,histo_energy_range_max, "X");
        event_pred_energy_sum_by_track_category[ic]->SetLineColor(kBlack);
        event_pred_energy_sum_by_track_category[ic]->SetStats(1);
        event_pred_energy_sum_by_track_category[ic]->Draw("HIST");
        gPad->Update();
        TPaveStats *ps = (TPaveStats*) gPad->GetPrimitive("stats");
        if(ps){
            ps->SetOptStat(1110);  // entry, mean, std dev (rme)、ヒストグラム名なし
            ps->SetX1NDC(0.62);     // StatX=0.9, StatW=0.2 -> x1=0.7
            ps->SetY1NDC(0.7);    // StatY=0.9, StatH=0.14 -> y1=0.76
            ps->SetX2NDC(0.9);
            ps->SetY2NDC(0.9);
            gPad->Modified();
            gPad->Update();
        }
    }

    TCanvas *canvas_confusion_eval = new TCanvas("canvas_confusion_eval","canvas_confusion_eval",1400,500);
    canvas_confusion_eval->Divide(2,1);
    TF1 *gaus_conf_reco_all = new TF1("gaus_conf_reco_all","gaus",-Eres_fit_range,Eres_fit_range);
    TF1 *gaus_conf_reco_clean = new TF1("gaus_conf_reco_clean","gaus",-Eres_fit_range,Eres_fit_range);
    TF1 *gaus_conf_cond_all = new TF1("gaus_conf_cond_all","gaus",-Eres_fit_range,Eres_fit_range);
    TF1 *gaus_conf_cond_clean = new TF1("gaus_conf_cond_clean","gaus",-Eres_fit_range,Eres_fit_range);

    const double sigma_all_reco = event_residual_all_reco_track->GetStdDev();
    const double sigma_clean_reco = event_residual_clean_reco_track->GetStdDev();
    const double sigma_conf_reco = sqrt(max(0.0, sigma_all_reco*sigma_all_reco - sigma_clean_reco*sigma_clean_reco));
    const double sigma_all_cond = event_residual_all_cond_track->GetStdDev();
    const double sigma_clean_cond = event_residual_clean_cond_track->GetStdDev();
    const double sigma_conf_cond = sqrt(max(0.0, sigma_all_cond*sigma_all_cond - sigma_clean_cond*sigma_clean_cond));

    canvas_confusion_eval->cd(1);
    event_residual_all_reco_track->SetLineColor(kBlack);
    event_residual_clean_reco_track->SetLineColor(kBlue);
    event_residual_all_reco_track->SetStats(0);
    event_residual_clean_reco_track->SetStats(0);
    event_residual_all_reco_track->Draw("HIST");
    event_residual_clean_reco_track->Draw("HIST SAME");
    if(event_residual_all_reco_track->GetEntries()>30) event_residual_all_reco_track->Fit("gaus_conf_reco_all","NQ","",-Eres_fit_range,Eres_fit_range);
    if(event_residual_clean_reco_track->GetEntries()>30) event_residual_clean_reco_track->Fit("gaus_conf_reco_clean","NQ","",-Eres_fit_range,Eres_fit_range);
    gaus_conf_reco_all->SetLineColor(kRed);
    gaus_conf_reco_clean->SetLineColor(kMagenta+1);
    if(event_residual_all_reco_track->GetEntries()>30) gaus_conf_reco_all->Draw("same");
    if(event_residual_clean_reco_track->GetEntries()>30) gaus_conf_reco_clean->Draw("same");
    TLegend *legend_conf_reco = new TLegend(0.45,0.62,0.88,0.88);
    legend_conf_reco->AddEntry(event_residual_all_reco_track, Form("all sigma=%.4f", sigma_all_reco), "l");
    legend_conf_reco->AddEntry(event_residual_clean_reco_track, Form("clean (C_{event}<%.2f) sigma=%.4f", c_event_clean_threshold, sigma_clean_reco), "l");
    legend_conf_reco->AddEntry((TObject*)0, Form("confusion term=%.4f", sigma_conf_reco), "");
    legend_conf_reco->SetFillStyle(0);
    legend_conf_reco->Draw("same");

    canvas_confusion_eval->cd(2);
    event_residual_all_cond_track->SetLineColor(kBlack);
    event_residual_clean_cond_track->SetLineColor(kBlue);
    event_residual_all_cond_track->SetStats(0);
    event_residual_clean_cond_track->SetStats(0);
    event_residual_all_cond_track->Draw("HIST");
    event_residual_clean_cond_track->Draw("HIST SAME");
    if(event_residual_all_cond_track->GetEntries()>30) event_residual_all_cond_track->Fit("gaus_conf_cond_all","NQ","",-Eres_fit_range,Eres_fit_range);
    if(event_residual_clean_cond_track->GetEntries()>30) event_residual_clean_cond_track->Fit("gaus_conf_cond_clean","NQ","",-Eres_fit_range,Eres_fit_range);
    gaus_conf_cond_all->SetLineColor(kRed);
    gaus_conf_cond_clean->SetLineColor(kMagenta+1);
    if(event_residual_all_cond_track->GetEntries()>30) gaus_conf_cond_all->Draw("same");
    if(event_residual_clean_cond_track->GetEntries()>30) gaus_conf_cond_clean->Draw("same");
    TLegend *legend_conf_cond = new TLegend(0.45,0.62,0.88,0.88);
    legend_conf_cond->AddEntry(event_residual_all_cond_track, Form("all sigma=%.4f", sigma_all_cond), "l");
    legend_conf_cond->AddEntry(event_residual_clean_cond_track, Form("clean (C_{event}<%.2f) sigma=%.4f", c_event_clean_threshold, sigma_clean_cond), "l");
    legend_conf_cond->AddEntry((TObject*)0, Form("confusion term=%.4f", sigma_conf_cond), "");
    legend_conf_cond->SetFillStyle(0);
    legend_conf_cond->Draw("same");

    cout << "confusion term evaluation (fixed-energy, no energy bin split)" << endl;
    cout << "  reco-track based: sigma_all=" << sigma_all_reco
         << ", sigma_clean=" << sigma_clean_reco
         << ", sigma_conf=" << sigma_conf_reco << endl;
    cout << "  cond-track based: sigma_all=" << sigma_all_cond
         << ", sigma_clean=" << sigma_clean_cond
         << ", sigma_conf=" << sigma_conf_cond << endl;


    
    if(saving_canvas){  // saving canvases
        gSystem->mkdir(picDirectory.c_str(), true);

        compare->SaveAs(Form("%s/efficiency_purity.png",picDirectory.c_str()));
        // compare2d->SaveAs(Form("%s/efficiency_purity_vs_energy%s.png",picDirectory.c_str(),suffix.c_str()));
        // compare_energy->SaveAs(Form("%s/per_energy%s.png",picDirectory.c_str(),suffix.c_str()));
        // compare_energy_normalized->SaveAs(Form("%s/per_energy_norm%s.png",picDirectory.c_str(),suffix.c_str()));
        canvas_energy->SaveAs(Form("%s/regression_cond_cluster.png",picDirectory.c_str()));
        canvas_energy_scan->SaveAs(Form("%s/energy_scan.png",picDirectory.c_str()));
        canvas_energy_resolution_scan->SaveAs(Form("%s/energy_resolution_scan.png",picDirectory.c_str()));
        // canvas_beta_energy->SaveAs(Form("%s/beta_vs_energy.png",picDirectory.c_str()));
        // canvas_beta_ediff->SaveAs(Form("%s/beta_vs_energy_ediff.png",picDirectory.c_str()));
        // canvas_beta_mcen->SaveAs(Form("%s/beta_vs_energy_mcen.png",picDirectory.c_str()));
        canvas_truth_phys_energy_regression->SaveAs(Form("%s/pfa_category_energy_regression.png", picDirectory.c_str()));
        canvas_energy_regression_result->SaveAs(Form("%s/energy_regression.png",picDirectory.c_str()));
        canvas_event_energy_resolution->SaveAs(Form("%s/dijet_energy_resolution.png",picDirectory.c_str()));
        canvas_event_energy_sum_by_beta->SaveAs(Form("%s/dijet_energy_resolution_beta.png",picDirectory.c_str()));
        canvas_event_energy_2d->SaveAs(Form("%s/dijet_scatter.png",picDirectory.c_str()));
        canvas_clustering_quality->SaveAs(Form("%s/category.png",picDirectory.c_str()));
        // canvas_track_charge_vs_truth_particle->SaveAs(Form("%s/track_charge_vs_truth_particle.png",picDirectory.c_str()));
        // canvas_c_event->SaveAs(Form("%s/c_event_distribution.png",picDirectory.c_str()));
        canvas_event_pred_energy_by_track_category->SaveAs(Form("%s/category_energy.png",picDirectory.c_str()));
        // canvas_confusion_eval->SaveAs(Form("%s/confusion_term_evaluation.png",picDirectory.c_str()));
    }
    if(0){
        compare->Write(Form("efficiency_purity"));
        // compare2d->Write(Form("%s/efficiency_purity_vs_energy%s",suffix.c_str()));
        // compare_energy->Write(Form("%s/per_energy%s",suffix.c_str()));
        // compare_energy_normalized->Write(Form("%s/per_energy_norm%s",suffix.c_str()));
        canvas_energy->Write(Form("energy_truth_vs_pred"));
        canvas_energy_scan->Write(Form("energy_scan"));
        canvas_energy_resolution_scan->Write(Form("energy_resolution_scan"));
        // canvas_beta_energy->Write(Form("beta_vs_energy"));
        // canvas_beta_ediff->Write(Form("beta_vs_energy_ediff"));
        // canvas_beta_mcen->Write(Form("beta_vs_energy_mcen"));
        canvas_truth_phys_energy_regression->Write(Form("truth_phys_category_energy_regression"));
        canvas_energy_regression_result->Write(Form("energy_regression"));
        canvas_event_energy_resolution->Write(Form("event_energy_resolution_scan"));
        canvas_event_energy_sum_by_beta->Write(Form("event_energy_sum_by_beta"));
        canvas_event_energy_2d->Write(Form("event_energy_truth_vs_pred_2d"));
        canvas_clustering_quality->Write(Form("reco_clustering_quality"));
        canvas_track_charge_vs_truth_particle->Write(Form("track_charge_vs_truth_particle"));
        canvas_c_event->Write(Form("c_event_distribution"));
        canvas_event_pred_energy_by_track_category->Write(Form("event_pred_energy_sum_by_track_category"));
        canvas_confusion_eval->Write(Form("confusion_term_evaluation"));
    }
    

}
