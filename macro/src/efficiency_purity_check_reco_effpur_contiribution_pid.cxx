// macro to save or display efficiency and purity of the root file
// if the energy regression is conducted, truth vs predicted is also displayed
// 
// usage
// Default configuration (change fileName/train_particle_type below):
//   make efficiency_purity_contribution_pid
//   ./build/efficiency_purity_contribution_pid
// A single PID ROOT file made by save_root_reco_pid.py --pid-readout representative:
//   ./build/efficiency_purity_contribution_pid /path/to/pid.root false figures/pid
// The same three arguments and the default file-list construction are retained.
// PID branches are detected per file. Common clustering/PID diagnostics and
// all five raw energy-head candidate grids are saved in the parent directory.
// Selected-energy images use {png,pdf,eps}/{raw_head,track_replaced}/;
// all ROOT data are consolidated in jet_energy_resolution_summary.root,
// with readout directories and a common clustering/ directory.
// Both modes keep frozen memberships and predicted PID, with
// only single-track charged energy changed. Without PID branches, the original
// two-head/Pandora readout and layout are retained. Existing track-based plot
// names remain for compatibility; their energies coincide within each PID mode.
// MC categories and Cedric W(P,C) always use truth labels and deposited energy,
// independently of predicted PID and the selected energy readout.
// PID confusion counts below are per reco cluster, against the stored dominant
// truth (largest hit fraction, then deposited energy), not a bijective match.
// The third argument is optional.  If omitted, figures are saved under the
// default picDirectory configured below.
// Pandora input still uses the second argument true.
// Event truth energy uses event/MC_dijet_energy, never the sum of t/mcen.
// regression_heads_pid.png and regression_heads_truth.png show all five saved
// energy candidates for each matched t row, using the same matching as
// regression_cond_cluster.png. Rows are charged_hadron/electron/muon/photon/
// neutral_hadron. PID columns are those five classes plus All; truth columns
// are electron/pion/photon/neutron/K0/muon. Charged candidates are evaluated
// at the highest-beta track; neutral candidates sum calorimeter nodes.
// Unavailable candidate energies (NaN for charged heads without a track) are
// omitted per head. PID columns use pid, including when energy_head differs.
// Additional diagnostics: pid_confusion_normalized (per reco cluster),
// clustering_effpur_pid_original and pid_clustering_success_original (per
// input truth object, including unmatched and undefined cases), and a limited
// set of truth_reco_overlap_original_file*_event* matrices. "original" uses
// saved t overlap definitions, independently of Cedric plots and regression.
// Each PID energy/method directory receives one jet_energy_resolution_summary.root
// and .txt. Its raw_head/ and track_replaced/ directories retain complete JER
// summaries and diagnostics/ ROOT objects. Non-PID inputs keep the original
// single-readout summary. The JER metric uses Pandora's
// upstream thrust reconstructed from both saved q/qbar four-vectors, raw event
// reco energy (no neutrino correction), and upstream minimum-RMS RMS90/Mean90.
// qq_energy/use_truth_clustering and the existing input arguments remain the
// configuration for beam energy/method; central/forward spectra are pooled
// directly from events, never averaged from angular-bin resolutions.
// The consolidated ROOT's common clustering/ directory retains original means
// and Cedric energy sums for the multi-energy efficiency/purity bar plots.
// save_pid_energy_readouts.py validates frozen source predictions and recovers
// measured track momentum from the identified original H5 when older ROOTs
// lack a saved measured-track policy. A temporary per-cluster energy cache
// drives two native subprocesses, which fill all dependent plots and ROOT
// histograms. Their data are assembled in one ROOT/text pair after both passes
// succeed. Temporary common/energy ROOTs, summaries and cache are removed. Deposit-based
// efficiency/purity and raw energy-head diagnostics remain common.
// 
// Cedric definition: build W(P,C) from the hit-level prediction tree, use
// track-anchored matching for charged particles and relative-majority ownership
// for neutral particles, then take ratios of global energy sums.

// Native entry point: mains/main_efficiency_purity_contribution_pid.cxx

#include "../root_common_includes.h"

#include <cmath>
#include <limits>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include <cstdlib>
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include "TLine.h"
#include "TLatex.h"
#include "THStack.h"
#include "TNamed.h"
#include "TObjString.h"
#include "pid_jet_resolution_summary.h"
#include "pid_clustering_summary.h"
#include "pid_single_pass_energy.h"

using namespace std;

const string model_condition = "5head_pid_onehot_preicted_gather";       // 2head     5head_pid    5head_pid_onehot     5head_pid_onehot_preicted_gather     5head_pid_onehot_truth_gather
const int qq_energy = 500;
const double c_event_clean_threshold = 0.1;
const bool use_truth_clustering = true;
const int cedric_n_min = 2;
const double cedric_e_min = 0.0;
const int master_unclustered_cluster_id = 0;

// Additional PID/clustering diagnostics; input ROOT files are read unchanged.
const double pid_cluster_success_efficiency = 0.9;
const double pid_cluster_success_purity = 0.9;
const int pid_overlap_event_limit = 3;  // Across all input files; 0 disables matrices.
const int pid_overlap_max_truth = 20;  // Remaining rows/columns are summed as Other.
const int pid_overlap_max_reco = 20;


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
const bool jet_regression = true;
const bool default_pandora = false;
const bool ECluster = true;
double beta_threshold = 0;

const bool edep_weight_ = true;


const int energyMax = test_particle_type == "ntau_10GeV_10" ? 12 : (test_particle_type == "uds91" ? 50 : 100 );
const int energyMaximum = test_particle_type == "ntau_10GeV_10" ? 10 : (test_particle_type == "uds91" ? 40 : 100 );

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

/** jet tree の leading jet の |cos#theta| ビン数（0–1 を等分割） */
static const int kNAbsCosThetaBins = 10;
static const double kAbsCosThetaEdges[kNAbsCosThetaBins + 1] = {
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
};

/**
 * 一度に 1 つの MC truth エネルギービンだけを見る（truthEnergyBinFromMcen の返す index）。
 * 例: 21 なら kTruthEnergyBinEdges[21]–[22] の区間（既定では約 40–50 GeV）。
 * 図のエネルギーを変えるときはここだけ変更する。
 */
const int kRms90VsAbsCosThetaTruthEnergyBin = 21;
static_assert(kRms90VsAbsCosThetaTruthEnergyBin >= 0 && kRms90VsAbsCosThetaTruthEnergyBin < kNTruthEnergyBins,
              "kRms90VsAbsCosThetaTruthEnergyBin must be a valid truth energy bin index");

inline int absCosThetaBinIndex(double absCosTheta) {
    if (absCosTheta < 0.0 || absCosTheta > 1.0 + 1e-12)
        return -1;
    for (int i = 0; i < kNAbsCosThetaBins; ++i) {
        const double lo = kAbsCosThetaEdges[i];
        const double hi = kAbsCosThetaEdges[i + 1];
        const bool upperInclusive = (i == kNAbsCosThetaBins - 1);
        if (absCosTheta >= lo && (upperInclusive ? (absCosTheta <= hi + 1e-12) : (absCosTheta < hi)))
            return i;
    }
    return -1;
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

// Species order matches particle_heads.py; unsupported/unknown truth is -1.
static const int kNPidSpecies = 5;
static const char* kPidSpecies[kNPidSpecies] = {"charged_hadron", "electron", "muon", "photon", "neutral_hadron"};
static const int kNHeadRegressionColumns = 6;
static const char* kHeadRegressionLabels[kNPidSpecies] = {"ch. hadron (alpha)", "electron (alpha)", "muon (alpha)", "photon (sum)", "neutral hadron (sum)"};
static const char* kPredictedRegressionLabels[kNHeadRegressionColumns] = {"charged hadron", "electron", "muon", "photon", "neutral hadron", "All"};
static const char* kTruthRegressionLabels[kNHeadRegressionColumns] = {"electron", "pion", "photon", "neutron", "K0", "muon"};

// Same six truth categories as the original regression_cond_cluster plot.
inline int headRegressionTruthColumn(int pdg) {
    switch(pdg){
        case 11: case -11: return 0;
        case 211: case -211: return 1;
        case 22: return 2;
        case 2112: return 3;
        case 130: return 4;
        case 13: case -13: return 5;
        default: return -1;
    }
}

struct PidHeadEnergyReadout {
    bool enabled = false;
    double energy[kNPidSpecies] = {};

    void bind(TTree* tree) {
        if(!tree) return;
        int present = 0;
        for(int ih=0; ih<kNPidSpecies; ++ih)
            if(tree->GetBranch(Form("energy_%s", kPidSpecies[ih]))) ++present;
        if(present == 0) return;  // Older ROOT files still support existing plots.
        if(present != kNPidSpecies)
            throw std::runtime_error(string("Incomplete energy head branches in ") + tree->GetName());
        for(int ih=0; ih<kNPidSpecies; ++ih){
            energy[ih] = std::numeric_limits<double>::quiet_NaN();
            tree->SetBranchStatus(Form("energy_%s", kPidSpecies[ih]), 1);
            if(tree->SetBranchAddress(Form("energy_%s", kPidSpecies[ih]), &energy[ih]) < 0)
                throw std::runtime_error(string("Energy head branch type mismatch in ") + tree->GetName());
        }
        enabled = true;
    }
};

inline int pidSpeciesFromMc(int pdg, int charge) {
    const int p = std::abs(pdg);
    if(p == 11) return 1;
    if(p == 13) return 2;
    if(p == 22) return 3;
    if(p >= 100) return charge != 0 ? 0 : 4;
    return -1;
}

struct PidEnergyReadout {
    bool enabled = false;
    int pid = -1;
    int energy_head = -1;  // -1 denotes a two-head checkpoint with diagnostic PID.
    int energy_fallback = 0;
    double energy_pid = 0.0;

    void bind(TTree* tree) {
        if(!tree) return;
        const char* fields[] = {"energy_pid", "pid", "energy_head", "energy_fallback"};
        for(const char* field : fields) enabled = enabled || tree->GetBranch(field);
        if(!enabled) return;
        for(const char* field : fields){
            if(!tree->GetBranch(field))
                throw std::runtime_error(string("Incomplete PID ROOT: ") + tree->GetName() + "/" + field);
        }
        for(const char* field : fields) tree->SetBranchStatus(field, 1);
        if(tree->SetBranchAddress("energy_pid", &energy_pid) < 0 ||
           tree->SetBranchAddress("pid", &pid) < 0 ||
           tree->SetBranchAddress("energy_head", &energy_head) < 0 ||
           tree->SetBranchAddress("energy_fallback", &energy_fallback) < 0)
            throw std::runtime_error(string("PID branch type mismatch in ") + tree->GetName());
    }

    double select(double legacy_energy) const {
        if(!enabled) return legacy_energy;
        if(!std::isfinite(energy_pid))
            throw std::runtime_error("Non-finite energy_pid in input ROOT");
        // In particular, do not select by truth PDG, track availability, or
        // pid again: energy_head may differ from pid for the saved fallback.
        return energy_pid;
    }
};

static std::string pidShellQuote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) quoted += c == '\'' ? "'\\''" : std::string(1, c);
    return quoted + "'";
}

static void savePairedPidEnergyReadouts(const std::string& summary_directory,
                                      const std::string& output_directory, bool draw) {
    const std::string helper = std::string(gSystem->DirName(__FILE__)) + "/../save_pid_energy_readouts.py";
    if (gSystem->AccessPathName(helper.c_str()))
        throw std::runtime_error("Missing paired PID readout helper: " + helper);
    const char* configured_python = gSystem->Getenv("PID_READOUT_PYTHON");
    const std::string python = configured_python && configured_python[0] ? configured_python : "python3";
    const std::string command = pidShellQuote(python) + " -B " + pidShellQuote(helper)
        + " --compact-only --staged-directory " + pidShellQuote(summary_directory)
        + " --output-directory " + pidShellQuote(output_directory)
        + " --common-directory " + pidShellQuote(summary_directory + "/common");
    if (gSystem->Exec(command.c_str()) != 0)
        throw std::runtime_error("Failed to save raw_head and track_replaced PID results in " + output_directory);
}

static void recoverMeasuredPidEnergy(const std::string& bridge, const std::string& output) {
    const std::string helper = std::string(gSystem->DirName(__FILE__)) + "/../save_pid_energy_readouts.py";
    const char* configured_python = gSystem->Getenv("PID_READOUT_PYTHON");
    const std::string python = configured_python && configured_python[0] ? configured_python : "python3";
    const std::string command = pidShellQuote(python) + " -B " + pidShellQuote(helper)
        + " --recover-measured " + pidShellQuote(bridge)
        + " --recovery-output " + pidShellQuote(output);
    if (gSystem->Exec(command.c_str()) != 0)
        throw std::runtime_error("Cannot recover original measured PID track momentum");
}

// The helper owns this short-lived cache. Each subprocess loads only the
// current input's cluster energies; the large hit-level prediction is not read.
class PidEnergyReadoutCache {
public:
    struct Input { int file_index; std::string path, metadata; };
    struct Energy { double saved, selected; };
    std::unique_ptr<TFile> file;
    std::vector<Input> inputs;
    std::map<std::pair<int, int>, Energy> clusters;
    std::string mode, method;
    double beta = 0.;

    PidEnergyReadoutCache(const char* path, const char* readout) : mode(readout ? readout : "") {
        if (mode != "raw_head" && mode != "track_replaced")
            throw std::runtime_error("Energy readout must be raw_head or track_replaced");
        TDirectory::TContext context;
        file.reset(TFile::Open(path, "READ"));
        if (!file || file->IsZombie()) throw std::runtime_error("Cannot read paired PID energy cache");
        auto* schema = dynamic_cast<TNamed*>(file->Get("energy_readout_cache_schema"));
        auto* energy = dynamic_cast<TParameter<int>*>(file->Get("beam_energy_GeV"));
        auto* threshold = dynamic_cast<TParameter<double>*>(file->Get("beta_threshold"));
        auto* method_named = dynamic_cast<TNamed*>(file->Get("method"));
        if (!schema || std::string(schema->GetTitle()) != "pid-energy-readout-cache-1" ||
            !energy || energy->GetVal() != qq_energy || !threshold ||
            !std::isfinite(threshold->GetVal()) || !method_named)
            throw std::runtime_error("Paired PID cache schema/energy/method does not match the configured analysis");
        method = method_named->GetTitle();
        if (method != "reco" && method != "truth_clustering")
            throw std::runtime_error("Paired PID cache has an unsupported clustering method");
        beta = threshold->GetVal();
        auto* rows = dynamic_cast<TTree*>(file->Get("inputs"));
        if (!rows) throw std::runtime_error("Paired PID cache has no input list");
        int index = -1;
        std::string* source = nullptr;
        std::string* metadata = nullptr;
        if (rows->SetBranchAddress("file_index", &index) < 0 ||
            rows->SetBranchAddress("path", &source) < 0 ||
            rows->SetBranchAddress("metadata", &metadata) < 0)
            throw std::runtime_error("Invalid paired PID cache input branch types");
        std::map<int, bool> seen;
        for (Long64_t entry = 0; entry < rows->GetEntries(); ++entry) {
            rows->GetEntry(entry);
            if (!source || !metadata || index < 0 || seen[index])
                throw std::runtime_error("Invalid or duplicate paired PID cache input");
            seen[index] = true;
            inputs.push_back({index, *source, *metadata});
        }
        rows->ResetBranchAddresses();
        if (inputs.empty()) throw std::runtime_error("Paired PID cache has no source files");
    }

    void load(int index, const std::string& metadata) {
        TDirectory::TContext context;
        const Input& input = inputs.at(index);
        if (input.metadata != metadata)
            throw std::runtime_error("Source PID metadata changed after preparing readout cache: " + input.path);
        auto* rows = dynamic_cast<TTree*>(file->Get(Form("readouts_file_%d", input.file_index)));
        if (!rows) throw std::runtime_error("Missing cluster rows in paired PID cache: " + input.path);
        int event = -1, cluster = -1;
        double saved = 0., selected = 0.;
        if (rows->SetBranchAddress("event", &event) < 0 || rows->SetBranchAddress("cluster", &cluster) < 0 ||
            rows->SetBranchAddress("saved", &saved) < 0 || rows->SetBranchAddress(mode.c_str(), &selected) < 0)
            throw std::runtime_error("Invalid cluster branches in paired PID cache");
        clusters.clear();
        for (Long64_t entry = 0; entry < rows->GetEntries(); ++entry) {
            rows->GetEntry(entry);
            if (cluster <= 0 || !std::isfinite(saved) || !std::isfinite(selected) ||
                !clusters.emplace(std::make_pair(event, cluster), Energy{saved, selected}).second)
                throw std::runtime_error("Invalid or duplicate cluster in paired PID cache");
        }
        rows->ResetBranchAddresses();
    }

    double select(int event, int cluster, double saved) const {
        if (cluster <= 0) {
            if (saved != 0.) throw std::runtime_error("Unmatched truth row has nonzero saved PID energy");
            return saved;
        }
        const auto found = clusters.find(std::make_pair(event, cluster));
        if (found == clusters.end() || found->second.saved != saved)
            throw std::runtime_error("PID energy cache does not match source event/cluster energy");
        return found->second.selected;
    }
};

struct PidSummaryStaging {
    std::string path;
    PidSummaryStaging() {
        char pattern[] = "/tmp/pid-analysis-readouts-XXXXXX";
        char* created = mkdtemp(pattern);
        if (!created) throw std::runtime_error("Cannot create temporary PID summary directory");
        path = created;
    }
    ~PidSummaryStaging() {
        removeTree(path);
    }
    static void removeTree(const std::string& path) {
        struct stat info;
        if (::lstat(path.c_str(), &info) != 0) return;
        if (!S_ISDIR(info.st_mode)) { ::unlink(path.c_str()); return; }
        DIR* directory = ::opendir(path.c_str());
        if (!directory) return;
        while (auto* item = ::readdir(directory)) {
            const std::string name = item->d_name;
            if (name != "." && name != "..") removeTree(path + "/" + name);
        }
        ::closedir(directory);
        ::rmdir(path.c_str());
    }
};

static bool pidReadoutDependentHistogram(const std::string& name) {
    // All fills that use the selected energy are covered here. MC truth,
    // deposited-energy overlaps, efficiency/purity and raw five-head grids
    // are independent of replacing a reconstructed single-track energy.
    for (const char* prefix : {"energy_", "energy2d_", "clusterenergy2d_", "event_energy",
                              "event_pred_energy_", "event_residual_", "eff_vs_Ediff",
                              "pur_vs_Ediff", "condbeta_vs_Ediff", "hframe_rms90_"})
        if (name.compare(0, std::char_traits<char>::length(prefix), prefix) == 0) return true;
    return false;
}

struct CedricHitRecord {
    int truthid = 0;
    int cluster = master_unclustered_cluster_id;
    int pdg = 0;
    int charge = 0;
    int is_track = 0;
    double mcen = 0.0;
    double edep = 0.0;
    bool truth_valid = true;
};

// Keep these diagnostics independent of the Cedric ownership/weighting rules.
// Efficiency/purity use saved t matching and its edep_match/edep(_reco).
// The success denominator is all known truth objects present in prediction,
// including objects absent from t. Counts are per truth object, not energy weighted.
struct PidClusteringDiagnostics {
    static const int n_status = 6;
    static const int n_energy_bins = 10;
    struct Truth {
        int species = -1;
        double energy = 0.0;
        int status = 4;  // Unmatched, unless a saved t match passes the beta cut.
        double efficiency = -1.0, purity = -1.0;
        bool has_match = false;
    };
    struct Overlap {
        int file = 0, event = 0;
        TH2F *raw = nullptr, *truth_percent = nullptr, *reco_percent = nullptr;
    };
    map<pair<int, int>, Truth> truth_objects;
    vector<Overlap> overlaps;
    TH2F *effpur[2];  // PID correct, PID wrong.
    TH2F *success_species, *success_energy;
    Long64_t unsupported_truth = 0, unknown_hits = 0, missing_truth_matches = 0;

    static const char* statusLabel(int status) {
        static const char* labels[n_status] = {
            "Cluster OK / PID OK", "Cluster OK / PID wrong",
            "Cluster poor / PID OK", "Cluster poor / PID wrong",
            "Unmatched / below beta cut", "Undefined overlap / PID"
        };
        return labels[status];
    }

    PidClusteringDiagnostics() {
        if(pid_cluster_success_efficiency < 0 || pid_cluster_success_efficiency > 1 ||
           pid_cluster_success_purity < 0 || pid_cluster_success_purity > 1 ||
           pid_overlap_event_limit < 0 || pid_overlap_max_truth < 1 || pid_overlap_max_reco < 1)
            throw std::runtime_error("Invalid PID/clustering diagnostic configuration");
        for(int correct=0; correct<2; ++correct)
            effpur[correct] = new TH2F(Form("clustering_effpur_pid_%s_original", correct == 0 ? "correct" : "wrong"),
                Form("PID %s (original overlap);Efficiency: E_{match}/E_{truth deposit};Purity: E_{match}/E_{cluster deposit}",
                     correct == 0 ? "correct" : "wrong"), 50, 0, 1.000001, 50, 0, 1.000001);
        success_species = new TH2F("pid_clustering_success_species_counts_original",
            "Original overlap: all known input truth objects;Truth species;Outcome",
            6, 0, 6, n_status, 0, n_status);
        success_energy = new TH2F("pid_clustering_success_energy_counts_original",
            "Original overlap: all known input truth objects;MC energy (GeV);Outcome",
            n_energy_bins, 0, n_energy_bins, n_status, 0, n_status);
        static const char* energy_labels[n_energy_bins] = {
            "0-1", "1-2", "2-5", "5-10", "10-20", "20-50", "50-100", "100-250", ">=250", "invalid E"
        };
        for(int ip=0; ip<6; ++ip)
            success_species->GetXaxis()->SetBinLabel(ip+1, kPredictedRegressionLabels[ip]);
        for(int ie=0; ie<n_energy_bins; ++ie)
            success_energy->GetXaxis()->SetBinLabel(ie+1, energy_labels[ie]);
        for(int status=0; status<n_status; ++status){
            success_species->GetYaxis()->SetBinLabel(status+1, statusLabel(status));
            success_energy->GetYaxis()->SetBinLabel(status+1, statusLabel(status));
        }
    }

    static int energyBin(double energy) {
        if(!std::isfinite(energy) || energy <= 0.0) return n_energy_bins-1;
        static const double upper[] = {1, 2, 5, 10, 20, 50, 100, 250};
        for(int i=0; i<8; ++i) if(energy < upper[i]) return i;
        return 8;
    }

    void beginFile() { truth_objects.clear(); }

    void observeEvent(int file, int event, const vector<CedricHitRecord>& hits) {
        for(const auto& hit : hits){
            if(!hit.truth_valid || hit.truthid <= 0){ ++unknown_hits; continue; }
            const auto key = make_pair(event, hit.truthid);
            if(truth_objects.count(key)) continue;
            Truth particle;
            particle.species = pidSpeciesFromMc(hit.pdg, hit.charge);
            particle.energy = hit.mcen;
            truth_objects.emplace(key, particle);
        }
        if(static_cast<int>(overlaps.size()) >= pid_overlap_event_limit) return;

        // All deposits participate, including unknown truth and unclustered hits.
        map<int, double> truth_energy, reco_energy;
        map<int, int> truth_pdg;
        map<pair<int, int>, double> shared;
        for(const auto& hit : hits){
            const int tid = hit.truth_valid && hit.truthid > 0 ? hit.truthid : 0;
            const int cid = hit.cluster > master_unclustered_cluster_id ? hit.cluster : 0;
            if(!std::isfinite(hit.edep) || hit.edep < 0.0) continue;
            truth_energy[tid] += hit.edep;
            reco_energy[cid] += hit.edep;
            shared[make_pair(tid, cid)] += hit.edep;
            if(tid > 0) truth_pdg[tid] = hit.pdg;
        }
        if(shared.empty()) return;
        auto select_ids = [](const map<int, double>& energies, int maximum) {
            vector<int> ids;
            for(const auto& item : energies) if(item.first > 0) ids.push_back(item.first);
            std::sort(ids.begin(), ids.end(), [&](int a, int b) {
                return energies.at(a) != energies.at(b) ? energies.at(a) > energies.at(b) : a < b;
            });
            const bool has_other = static_cast<int>(ids.size()) > maximum;
            if(has_other) ids.resize(maximum);
            if(has_other) ids.push_back(-1);  // Aggregate every undisplayed ID.
            ids.push_back(0);  // Unknown truth / unclustered hits, including empty bins.
            return ids;
        };
        const auto tids = select_ids(truth_energy, pid_overlap_max_truth);
        const auto cids = select_ids(reco_energy, pid_overlap_max_reco);
        map<int, int> tbin, cbin;
        for(size_t i=0; i<tids.size(); ++i) tbin[tids[i]] = i+1;
        for(size_t i=0; i<cids.size(); ++i) cbin[cids[i]] = i+1;
        Overlap matrix;
        matrix.file = file; matrix.event = event;
        matrix.raw = new TH2F(Form("truth_reco_overlap_original_file%03d_event%06d", file, event),
            "Shared deposited energy;Reco cluster ID;Truth ID (PDG);Shared deposit (GeV)",
            cids.size(), 0, cids.size(), tids.size(), 0, tids.size());
        matrix.raw->SetDirectory(nullptr);  // Never attach diagnostics to an input file.
        for(size_t i=0; i<tids.size(); ++i){
            const string label = tids[i] == 0 ? "unknown truth" : tids[i] == -1 ? "other truths" :
                Form("%d (%d)", tids[i], truth_pdg[tids[i]]);
            matrix.raw->GetYaxis()->SetBinLabel(i+1, label.c_str());
        }
        for(size_t i=0; i<cids.size(); ++i){
            const string label = cids[i] == 0 ? "unclustered" : cids[i] == -1 ? "other clusters" : Form("%d", cids[i]);
            matrix.raw->GetXaxis()->SetBinLabel(i+1, label.c_str());
        }
        for(const auto& cell : shared){
            const int iy = tbin.count(cell.first.first) ? tbin.at(cell.first.first) : tbin.at(-1);
            const int ix = cbin.count(cell.first.second) ? cbin.at(cell.first.second) : cbin.at(-1);
            matrix.raw->SetBinContent(ix, iy, matrix.raw->GetBinContent(ix, iy) + cell.second);
        }
        matrix.truth_percent = static_cast<TH2F*>(matrix.raw->Clone(Form("%s_truth_percent", matrix.raw->GetName())));
        matrix.reco_percent = static_cast<TH2F*>(matrix.raw->Clone(Form("%s_reco_percent", matrix.raw->GetName())));
        matrix.truth_percent->SetDirectory(nullptr);
        matrix.reco_percent->SetDirectory(nullptr);
        matrix.truth_percent->SetTitle(Form("File %d, event %d: each truth row = 100%%;Reco cluster ID;Truth ID (PDG);Shared truth deposit (%%)", file, event));
        matrix.reco_percent->SetTitle(Form("File %d, event %d: each reco column = 100%%;Reco cluster ID;Truth ID (PDG);Cluster composition (%%)", file, event));
        for(int iy=1; iy<=matrix.raw->GetNbinsY(); ++iy){
            double total = 0.0;
            for(int ix=1; ix<=matrix.raw->GetNbinsX(); ++ix) total += matrix.raw->GetBinContent(ix, iy);
            for(int ix=1; ix<=matrix.raw->GetNbinsX(); ++ix)
                matrix.truth_percent->SetBinContent(ix, iy, total > 0 ? 100.0 * matrix.raw->GetBinContent(ix, iy) / total : 0.0);
        }
        for(int ix=1; ix<=matrix.raw->GetNbinsX(); ++ix){
            double total = 0.0;
            for(int iy=1; iy<=matrix.raw->GetNbinsY(); ++iy) total += matrix.raw->GetBinContent(ix, iy);
            for(int iy=1; iy<=matrix.raw->GetNbinsY(); ++iy)
                matrix.reco_percent->SetBinContent(ix, iy, total > 0 ? 100.0 * matrix.raw->GetBinContent(ix, iy) / total : 0.0);
        }
        overlaps.push_back(matrix);
    }

    void matchTruth(int event, int truthid, int cluster, int pid, double beta,
                    double truth_deposit, double reco_deposit, double shared_deposit) {
        auto it = truth_objects.find(make_pair(event, truthid));
        if(it == truth_objects.end()){ ++missing_truth_matches; return; }
        Truth& particle = it->second;
        if(cluster <= master_unclustered_cluster_id || !std::isfinite(beta) || beta < beta_threshold) return;
        if(particle.has_match) throw std::runtime_error("Duplicate t row for an event/truthid in PID diagnostics");
        particle.has_match = true;
        particle.status = 5;
        if(particle.species < 0 || pid < 0 || pid >= kNPidSpecies ||
           !std::isfinite(truth_deposit) || !std::isfinite(reco_deposit) || !std::isfinite(shared_deposit) ||
           truth_deposit <= 0.0 || reco_deposit <= 0.0 || shared_deposit < 0.0) return;
        double eff = shared_deposit / truth_deposit, pur = shared_deposit / reco_deposit;
        if(eff > 1.0 + 1e-6 || pur > 1.0 + 1e-6) return;  // Do not disguise invalid overlap.
        particle.efficiency = std::min(eff, 1.0);
        particle.purity = std::min(pur, 1.0);
        const bool good_cluster = particle.efficiency >= pid_cluster_success_efficiency &&
                                  particle.purity >= pid_cluster_success_purity;
        particle.status = (good_cluster ? 0 : 2) + (pid == particle.species ? 0 : 1);
    }

    void finishFile() {
        for(const auto& item : truth_objects){
            const Truth& particle = item.second;
            if(particle.species < 0){ ++unsupported_truth; continue; }
            success_species->Fill(particle.species + 0.5, particle.status + 0.5);
            success_species->Fill(5.5, particle.status + 0.5);
            success_energy->Fill(energyBin(particle.energy) + 0.5, particle.status + 0.5);
            if(particle.status < 4)
                effpur[particle.status % 2]->Fill(particle.efficiency, particle.purity);
        }
    }

    static void drawPercentMatrix(TH2F* histogram, bool text=true) {
        const bool crowded = histogram->GetNbinsX() > 8;
        gPad->SetLeftMargin(0.24); gPad->SetRightMargin(0.16);
        gPad->SetBottomMargin(crowded ? 0.25 : 0.20); gPad->SetTopMargin(0.10);
        histogram->SetStats(0); histogram->SetMinimum(0); histogram->SetMaximum(100);
        histogram->GetXaxis()->SetLabelSize(crowded ? 0.027 : 0.035);
        histogram->GetYaxis()->SetLabelSize(crowded ? 0.027 : 0.035);
        if(crowded) histogram->GetXaxis()->LabelsOption("v");
        // Fixed title positions avoid overlaps with categorical bin labels.
        const string x_title = histogram->GetXaxis()->GetTitle();
        const string y_title = histogram->GetYaxis()->GetTitle();
        histogram->GetXaxis()->SetTitleSize(0);
        histogram->GetYaxis()->SetTitleSize(0);
        histogram->Draw("COLZ");
        TLatex axis_title; axis_title.SetNDC(); axis_title.SetTextAlign(22); axis_title.SetTextSize(0.035);
        axis_title.DrawLatex(0.54, 0.04, x_title.c_str());
        axis_title.SetTextAngle(90);
        axis_title.DrawLatex(0.035, 0.55, y_title.c_str());
        if(text){
            TLatex label; label.SetTextAlign(22); label.SetTextSize(0.028);
            for(int ix=1; ix<=histogram->GetNbinsX(); ++ix){
                for(int iy=1; iy<=histogram->GetNbinsY(); ++iy){
                    const double value = histogram->GetBinContent(ix, iy);
                    label.SetTextColor(value > 0 && value < 45 ? kWhite : kBlack);
                    label.DrawLatex(histogram->GetXaxis()->GetBinCenter(ix), histogram->GetYaxis()->GetBinCenter(iy), Form("%.1f", value));
                }
            }
        }
    }

    vector<pair<string, TCanvas*>> draw(TH2F* reco_confusion) {
        vector<pair<string, TCanvas*>> canvases;
        TCanvas* confusion = new TCanvas("canvas_pid_confusion_normalized", "PID fractions per reco cluster", 1900, 850);
        confusion->Divide(2,1);
        for(int mode=0; mode<2; ++mode){
            confusion->cd(mode+1);
            TH2F* fractions = new TH2F(Form("pid_confusion_%s_percent", mode == 0 ? "truth" : "prediction"),
                mode == 0 ? "Reco clusters: each truth row = 100%;Predicted PID;Dominant truth species;Fraction (%)" :
                            "Reco clusters: each PID column = 100%;Predicted PID;Dominant truth species;Fraction (%)",
                kNPidSpecies, 0, kNPidSpecies, 6, 0, 6);
            fractions->SetDirectory(nullptr);
            for(int ip=0; ip<kNPidSpecies; ++ip){
                fractions->GetXaxis()->SetBinLabel(ip+1, kPredictedRegressionLabels[ip]);
                fractions->GetYaxis()->SetBinLabel(ip+1, kPredictedRegressionLabels[ip]);
            }
            fractions->GetYaxis()->SetBinLabel(6, "unknown / other");
            for(int it=1; it<=6; ++it){
                for(int ip=1; ip<=kNPidSpecies; ++ip){
                    double total = 0.0;
                    if(mode == 0) for(int j=1; j<=kNPidSpecies; ++j) total += reco_confusion->GetBinContent(it,j);
                    else for(int j=1; j<=6; ++j) total += reco_confusion->GetBinContent(j,ip);
                    fractions->SetBinContent(ip, it, total > 0 ? 100.0 * reco_confusion->GetBinContent(it,ip) / total : 0.0);
                }
            }
            drawPercentMatrix(fractions);
        }
        canvases.emplace_back("pid_confusion_normalized", confusion);

        TCanvas* plane = new TCanvas("canvas_clustering_effpur_pid_original", "Clustering quality by PID correctness", 1800, 800);
        plane->Divide(2,1);
        const double maximum = std::max(1.0, std::max(effpur[0]->GetMaximum(), effpur[1]->GetMaximum()));
        for(int correct=0; correct<2; ++correct){
            plane->cd(correct+1);
            gPad->SetLeftMargin(0.15); gPad->SetRightMargin(0.15); gPad->SetBottomMargin(0.14);
            effpur[correct]->SetStats(0); effpur[correct]->SetMaximum(maximum);
            effpur[correct]->GetYaxis()->SetTitleOffset(1.5);
            effpur[correct]->Draw("COLZ");
            TLine vertical(pid_cluster_success_efficiency, 0, pid_cluster_success_efficiency, 1);
            TLine horizontal(0, pid_cluster_success_purity, 1, pid_cluster_success_purity);
            vertical.SetLineStyle(2); horizontal.SetLineStyle(2);
            vertical.SetLineColor(kRed+1); horizontal.SetLineColor(kRed+1);
            vertical.DrawClone(); horizontal.DrawClone();
            TLatex label; label.SetNDC(); label.SetTextSize(0.033);
            label.DrawLatex(0.18, 0.84, Form("N = %.0f matched truth objects", effpur[correct]->GetEntries()));
        }
        canvases.emplace_back("clustering_effpur_pid_original", plane);

        TCanvas* success = new TCanvas("canvas_pid_clustering_success_original", "PID and clustering outcomes", 2100, 1100);
        success->Divide(2,1);
        const int colors[n_status] = {kGreen+2, kAzure+1, kOrange+1, kRed+1, kGray+1, kViolet-5};
        for(int panel=0; panel<2; ++panel){
            success->cd(panel+1);
            gPad->SetTopMargin(0.32); gPad->SetBottomMargin(0.20); gPad->SetLeftMargin(0.12);
            TH2F* counts = panel == 0 ? success_species : success_energy;
            THStack* stack = new THStack(Form("pid_clustering_success_%s_original", panel == 0 ? "species" : "energy"),
                Form("All known input truth objects (original overlap);%s;Truth objects (%%)", panel == 0 ? "Truth species" : "MC energy (GeV)"));
            TLegend* legend = new TLegend(0.12, 0.79, 0.96, 0.94);
            legend->SetNColumns(2); legend->SetBorderSize(0); legend->SetTextSize(0.025);
            for(int status=0; status<n_status; ++status){
                TH1F* fraction = new TH1F(Form("%s_status%d_percent", counts->GetName(), status), "",
                    counts->GetNbinsX(), 0, counts->GetNbinsX());
                fraction->SetDirectory(nullptr);
                fraction->SetFillColor(colors[status]); fraction->SetLineColor(kWhite);
                for(int ix=1; ix<=counts->GetNbinsX(); ++ix){
                    double total = 0.0;
                    for(int iy=1; iy<=n_status; ++iy) total += counts->GetBinContent(ix,iy);
                    fraction->GetXaxis()->SetBinLabel(ix, counts->GetXaxis()->GetBinLabel(ix));
                    fraction->SetBinContent(ix, total > 0 ? 100.0 * counts->GetBinContent(ix,status+1) / total : 0.0);
                }
                stack->Add(fraction); legend->AddEntry(fraction, statusLabel(status), "f");
            }
            stack->SetMinimum(0); stack->SetMaximum(110); stack->Draw("HIST");
            stack->GetXaxis()->SetLabelSize(0.03); stack->GetYaxis()->SetTitleOffset(1.4);
            stack->GetXaxis()->SetTitleSize(0);
            for(int ix=1; ix<counts->GetNbinsX(); ++ix){
                TLine separator(ix, 0, ix, 100);
                separator.SetLineColor(kWhite); separator.SetLineWidth(3); separator.DrawClone();
            }
            legend->Draw();
            TLatex label; label.SetNDC(); label.SetTextSize(0.027);
            label.DrawLatex(0.12, 0.74, Form("Cluster OK: efficiency #geq %.2f and purity #geq %.2f", pid_cluster_success_efficiency, pid_cluster_success_purity));
            label.SetTextAlign(22); label.SetTextSize(0.035);
            label.DrawLatex(0.55, 0.05, panel == 0 ? "Truth species" : "MC energy (GeV)");
            label.SetNDC(false); label.SetTextAlign(22); label.SetTextSize(0.022);
            for(int ix=1; ix<=counts->GetNbinsX(); ++ix){
                double total = 0.0;
                for(int iy=1; iy<=n_status; ++iy) total += counts->GetBinContent(ix,iy);
                label.DrawLatex(ix-0.5, 104, Form("N=%.0f", total));
            }
        }
        canvases.emplace_back("pid_clustering_success_original", success);
        for(const auto& matrix : overlaps){
            const string name = matrix.raw->GetName();
            TCanvas* canvas = new TCanvas(("canvas_" + name).c_str(), "Truth to reco overlap", 2400, 1200);
            canvas->Divide(2,1);
            const bool text = matrix.raw->GetNbinsX() <= 12 && matrix.raw->GetNbinsY() <= 12;
            canvas->cd(1); drawPercentMatrix(matrix.truth_percent, text);
            canvas->cd(2); drawPercentMatrix(matrix.reco_percent, text);
            canvases.emplace_back(name, canvas);
        }
        cout << "Original-overlap diagnostics: " << success_species->Integral(6,6,1,n_status)
             << " supported truth objects; " << unsupported_truth << " unsupported truth objects excluded; "
             << unknown_hits << " unknown input hits; " << missing_truth_matches << " t rows absent from prediction" << endl;
        return canvases;
    }

    void writeOverlapHistograms() const {
        for(const auto& matrix : overlaps){
            matrix.raw->Write(Form("%s_energy", matrix.raw->GetName()));
            matrix.truth_percent->Write(); matrix.reco_percent->Write();
        }
    }
};

struct CedricTruthParticle {
    int pdg = 0;
    int charge = 0;
    double mcen = 0.0;
    double energy = 0.0;
};

struct CedricRecoCluster {
    int nhits = 0;
    double energy = 0.0;
    bool has_track = false;
    map<int, double> overlap;
    map<int, bool> track_truth_ids;
};

void saveCanvas(TCanvas* canvas, const std::string& directory, const std::string& name,
                const std::string& readout = ""){
    if (!canvas) return;
    canvas->Update();
    const char* formats[] = {"png", "pdf", "eps"};
    for (const char* format : formats) {
        const std::string subdirectory = directory + "/" + format
            + (readout.empty() ? "" : "/" + readout);
        gSystem->mkdir(subdirectory.c_str(), true);

        const std::string filename = subdirectory + "/" + name + "." + format;
        canvas->SaveAs(filename.c_str());
    }
}

void efficiency_purity_check_reco_effpur_contiribution_w_Cedric(
    const char* input_file = nullptr,
    bool pandora_mode = default_pandora,
    const char* output_directory = nullptr,
    PidEnergyReadoutCache* readout_cache = nullptr,
    const char* image_directory = nullptr){
    const bool energy_only = readout_cache != nullptr;
    if (energy_only) beta_threshold = readout_cache->beta;
    const std::string analysis_method = energy_only ? readout_cache->method :
        (pandora_mode ? "pandora" : (use_truth_clustering ? "truth_clustering" : "reco"));
    const bool has_input_override = input_file && input_file[0] != '\0';
    int rawfilenum = energy_only ? readout_cache->inputs.size() : (has_input_override ? 1 : num_file);
    if (rawfilenum > num_file)
        throw std::runtime_error("Readout cache has more source files than the configured analysis capacity");
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
    string path_to_file = pandora_mode ? "pandora" : (use_truth_clustering ? "truth_clustering" : "reco");
    string picDirectory = Form("figures/nnqq2M_fixed_uds_%s/%dGeV/%s", model_condition.c_str(), qq_energy, path_to_file.c_str());
    if(output_directory && output_directory[0] != '\0') picDirectory = output_directory;
    std::string imageDirectory = energy_only && image_directory && image_directory[0]
        ? image_directory : picDirectory;

    if (energy_only) {
        for (int i=0; i<rawfilenum; ++i) fileNames_raw[i] = readout_cache->inputs[i].path;
    }
    else if(has_input_override) fileNames_raw[0] = input_file;
    else if(rawfilenum == 1) fileNames_raw[0] = Form("%s",fileName.c_str());
    else {
        for(int qq=0; qq<3; qq++){
            for(int i=1; i<=nfile; i++){
                if(exist_file(qq_energy, qqNames[qq], i)) continue;
                path_to_file = pandora_mode ? "pandora" : (use_truth_clustering ? "truth_clustering" : "tbeta090td050");
                // fileNames[qq][i] = Form("../output/energy_regression_1to1/skimmed/tc_fixed_uds/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/%s/%s_%03d.root", qq_energy, path_to_file.c_str(), qqNames[qq].c_str(), i);
                // fileNames[qq][i] = Form("../output/energy_regression_1to1/skimmed/tc_fixed_uds/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/perh5file/%s_%03d.root", qq_energy, qqNames[qq].c_str(), i);
                // fileNames[qq][i] = Form("../output/energy_regression_1to1/skimmed/tc_nnqq_2M/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/multi-head/%s/%s_%03d.root", qq_energy, path_to_file.c_str(), qqNames[qq].c_str(), i);
                if(pandora_mode)
                    fileNames[qq][i] = Form("../output/energy_regression_1to1/skimmed/tc_nnqq_2M/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/pandora/%s_%03d.root", qq_energy, qqNames[qq].c_str(), i);
                else
                    fileNames[qq][i] = Form("../output/energy_regression_1to1/skimmed/tc_nnqq_2M/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/%dGeV/%s/%s/%s_%03d.root", qq_energy, model_condition.c_str(), path_to_file.c_str(), qqNames[qq].c_str(), i);
                fileNames_raw[irawfilenum] = Form("%s",fileNames[qq][i].c_str());
                irawfilenum++;
            }
        }
    }
    cout << "input mode: " << (pandora_mode ? "Pandora" : "GravNet") << endl;
    if(has_input_override) cout << input_file << endl;
    else cout << fileName << endl;
    cout << picDirectory << endl;
    

    int event, hitid, mcid, truthid, mcpdg, mccharge, mcstatus, ncluster, matched_ncluster, matched_cluster, cond_track;
    double mcmass, mcpx, mcpy, mcpz, mcen, edep, edep_reco, edep_match, pred_edep, pred_edep_cluster, cond_beta, pred_alpha;

    int _event, _hitid, _mcid, _truthid, _mcpdg, _mccharge, _mcstatus, _ncluster, _matched_ncluster, _matched_cluster, _pred_alpha;
    double _mcmass, _mcpx, _mcpy, _mcpz, _mcen, _edep, _edep_reco, _edep_match, _pred_edep, _pred_edep_cluster, _pred_beta;

    int jet_event, n_jets;
    double jet_p4[2][4];

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
    struct CachedRecoRow {
        int event, cluster, mcid, pdg, charge, tracks, cond_track, matched_pdg, npdg;
        double beta, prediction, cluster_prediction, deposit, matched_fraction;
        std::array<int, 64> composition;
        PidEnergyReadout pid;
    };



    // TFile fileout("result_test_pandora.root","RECREATE");
    gSystem->mkdir(picDirectory.c_str(), true);
    bool expected_pid_input = energy_only;
    std::unique_ptr<TFile> first_input;
    if (rawfilenum > 0) {
        TDirectory::TContext context;
        first_input.reset(TFile::Open(fileNames_raw[0].c_str(), "READ"));
        if (!first_input || first_input->IsZombie())
            throw std::runtime_error("Cannot open input ROOT: " + fileNames_raw[0]);
        if (!pandora_mode && !expected_pid_input) {
            auto* reco = dynamic_cast<TTree*>(first_input->Get("reco"));
            expected_pid_input = reco && reco->GetBranch("energy_pid");
        }
    }
    const bool paired_pid = expected_pid_input && !energy_only;
    std::unique_ptr<PidSummaryStaging> common_staging;
    if (paired_pid) {
        common_staging.reset(new PidSummaryStaging());
        imageDirectory = common_staging->path;
    }
    const std::string result_directory = common_staging ? common_staging->path + "/common" : picDirectory;
    gSystem->mkdir(result_directory.c_str(), true);
    const char* result_name = energy_only ? "energy_regression_result.root" : "efficiency_purity_result.root";
    const string output_path = expected_pid_input || (output_directory && output_directory[0] != '\0')
        ? Form("%s/%s", result_directory.c_str(), result_name)
        : Form("../output/energy_regression_1to1/skimmed/results/tc_fixed_uds_brems/5D/%dGeV/%s.root", qq_energy, path_to_file.c_str());
    TFile fileout(output_path.c_str(),"RECREATE");
    pid_jer::Summary jet_resolution_summary(qq_energy, analysis_method,
        beta_threshold, energy_only ? readout_cache->mode : "saved");
    const char* paired_modes[2] = {"raw_head", "track_replaced"};
    std::unique_ptr<pid_jer::Summary> paired_jer[2];
    if (paired_pid) for (int mode=0; mode<2; ++mode)
        paired_jer[mode].reset(new pid_jer::Summary(qq_energy, analysis_method, beta_threshold, paired_modes[mode]));
    pid_singlepass::HistogramBanks energy_histograms;
    Long64_t paired_eligible = 0, paired_recovered = 0;


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

    double Eres_range = pandora_mode ? 0.05 : 1.5;
    // double Eres_range = 1.5;
    int Eres_nbin = 1200;
    double E_res_binWidth = Eres_range*2/Eres_nbin;
    double Eres_fit_range = 0.5;
    int Eres_fitbin_lower = Eres_range>Eres_fit_range ? (Eres_range-Eres_fit_range)/E_res_binWidth : 0;
    int Eres_fitbin_upper = Eres_range>Eres_fit_range ? Eres_nbin - (Eres_range-Eres_fit_range)/E_res_binWidth : -1;
    int rebin_factor = 0.025 / E_res_binWidth;

    const int nEnergy_jet = 12;
    const int nEnergy = kNTruthEnergyBins;
    TH1F *purity_cedric[nParticle];
    TH1F *purity_energy[nParticle][nEnergy];
    TH1F *purity_energy_normalize[nParticle][nEnergy];
    TH2F *purity2d[nParticle];
    TH1F *efficiency_cedric[nParticle];
    TH1F *purity[nParticle];
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
        purity_cedric[ip] = new TH1F(Form("purity_cedric_%d",ip), Form("%s Cedric purity (cluster-energy weighted)",particleNames[ip].c_str()), 101,0,1.01);
        purity_cedric[ip]->SetXTitle("purity W(P,C*)/W(C*)");
        purity2d[ip] = new TH2F(Form("purity2d_%d",ip), Form("%s Cedric purity",particleNames[ip].c_str()), 101,0,1.01, 24,0,12);
        purity2d[ip]->SetXTitle("purity W(P,C*)/W(C*)");
        purity2d[ip]->SetYTitle("MC energy (edep)");
        efficiency_cedric[ip] = new TH1F(Form("efficiency_cedric_%d",ip), Form("%s Cedric efficiency (truth-energy weighted)",particleNames[ip].c_str()), 101,0,1.01);
        efficiency_cedric[ip]->SetXTitle("efficiency W(P,C*)/W(P)");
        efficiency2d[ip] = new TH2F(Form("efficiency2d_%d",ip), Form("%s Cedric efficiency",particleNames[ip].c_str()), 101,0,1.01, 24,0,12);
        efficiency2d[ip]->SetXTitle("efficiency W(P,C*)/W(P)");
        efficiency2d[ip]->SetYTitle("MC energy (edep)");
        purity[ip] = new TH1F(Form("purity_%d",ip), Form("%s purity (MC energy>1 GeV)",particleNames[ip].c_str()), 101,0,1.01);
        purity[ip]->SetXTitle("purity (edep_match/edep_reco)");
        efficiency[ip] = new TH1F(Form("efficiency_%d",ip), Form("%s efficiency (MC energy>1 GeV)",particleNames[ip].c_str()), 101,0,1.01);
        efficiency[ip]->SetXTitle("efficiency (edep_match/edep)");
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
            purity_energy[ip][ie]->SetXTitle("purity W(P,C*)/W(C*)");
            purity_energy_normalize[ip][ie] = new TH1F(Form("purity_norm_%d_%d",ip,ie), title.c_str(), 101,0,1.01);
            purity_energy_normalize[ip][ie]->SetXTitle("purity W(P,C*)/W(C*)");
            title = ie==0 ? Form("%s efficiency",particleNames[ip].c_str()) : Form("%s efficiency %.0f-%.0f GeV", particleNames[ip].c_str(), kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]);
            efficiency_energy[ip][ie] = new TH1F(Form("efficiency_%d_%d",ip,ie), title.c_str(), 101,0,1.01);
            efficiency_energy[ip][ie]->SetXTitle("efficiency W(P,C*)/W(P)");
            efficiency_energy_normalize[ip][ie] = new TH1F(Form("efficiency_normalize_%d_%d",ip,ie), title.c_str(), 101,0,1.01);
            efficiency_energy_normalize[ip][ie]->SetXTitle("efficiency W(P,C*)/W(P)");

            title = ie==0 ? Form("%s ;predicted energy - MC truth energy",particleNames[ip].c_str()) : Form("%s (%.0f-%.0f GeV);predicted energy - MC truth energy",particleNames[ip].c_str(), kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]);
            // energy_diff_per_energy[ip][ie] = new TH1F(Form("energy_diff_per_energy%d_%d",ip,ie), title.c_str(), 120,-6,6);
            energy_diff_per_energy[ip][ie] = new TH1F(Form("energy_diff_per_energy%d_%d",ip,ie), title.c_str(), Eres_nbin,-Eres_range,Eres_range);
            title = ie==0 ? Form("%s ;(predicted - truth) / truth",particleNames[ip].c_str()) : Form("%s (%.0f-%.0f GeV);(predicted - truth) / truth",particleNames[ip].c_str(), kTruthEnergyBinEdges[ie], kTruthEnergyBinEdges[ie + 1]);
            // energy_resolution_per_energy[ip][ie] = new TH1F(Form("energy_resolution_per_energy_%d_%d",ip,ie), title.c_str(), 120,-6,6);
            energy_resolution_per_energy[ip][ie] = new TH1F(Form("energy_resolution_per_energy_%d_%d",ip,ie), title.c_str(), Eres_nbin,-Eres_range,Eres_range);
            energy_resolution_per_energy_cluster[ip][ie] = new TH1F(Form("energy_resolution_per_energy_cluster_%d_%d",ip,ie), title.c_str(), Eres_nbin,-Eres_range,Eres_range);
        }
    }
    // TH2F *jet_energy2d = new TH2F(Form("jet_energy2d"), Form(";MC jet energy;predicted jet energy"), 1200,0,120,1200,0,120);
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

    TH1F *energy_res_vs_abs_costheta[nParticle][kNAbsCosThetaBins];
    for(int ip=0; ip<nParticle; ip++){
        for(int ib=0; ib<kNAbsCosThetaBins; ib++){
            const string title = Form(
                "%s truth %.0f-%.0f GeV, |cos#theta|#in[%.2f,%.2f);(pred-truth)/truth",
                particleNames[ip].c_str(),
                kTruthEnergyBinEdges[kRms90VsAbsCosThetaTruthEnergyBin],
                kTruthEnergyBinEdges[kRms90VsAbsCosThetaTruthEnergyBin + 1],
                kAbsCosThetaEdges[ib],
                kAbsCosThetaEdges[ib + 1]);
            energy_res_vs_abs_costheta[ip][ib] = new TH1F(
                Form("energy_res_vs_abscosth_%d_%d", ip, ib),
                title.c_str(),
                Eres_nbin,
                -Eres_range,
                Eres_range);
        }
    }


    // PID diagnostics count each reco cluster once after the existing beta cut.
    TH2F *pid_confusion_reco = new TH2F("pid_confusion_reco", "Representative PID vs dominant truth;dominant truth species;predicted PID;clusters", 6, 0, 6, kNPidSpecies, 0, kNPidSpecies);
    TH2F *pid_energy_head_vs_prediction = new TH2F("pid_energy_head_vs_prediction", "Saved energy head vs PID;predicted PID;selected energy head;clusters", kNPidSpecies, 0, kNPidSpecies, 6, 0, 6);
    TH1F *pid_prediction_counts = new TH1F("pid_prediction_counts", "Representative PID;predicted PID;clusters", kNPidSpecies, 0, kNPidSpecies);
    TH1F *pid_energy_fallback_counts = new TH1F("pid_energy_fallback_counts", "Saved energy fallback;energy readout;clusters", 2, 0, 2);
    for(int ip=0; ip<kNPidSpecies; ++ip){
        pid_confusion_reco->GetXaxis()->SetBinLabel(ip+1, kPidSpecies[ip]);
        pid_confusion_reco->GetYaxis()->SetBinLabel(ip+1, kPidSpecies[ip]);
        pid_energy_head_vs_prediction->GetXaxis()->SetBinLabel(ip+1, kPidSpecies[ip]);
        pid_energy_head_vs_prediction->GetYaxis()->SetBinLabel(ip+1, kPidSpecies[ip]);
        pid_prediction_counts->GetXaxis()->SetBinLabel(ip+1, kPidSpecies[ip]);
    }
    pid_confusion_reco->GetXaxis()->SetBinLabel(6, "unknown/unsupported");
    pid_energy_head_vs_prediction->GetYaxis()->SetBinLabel(6, "legacy two-head");
    pid_energy_fallback_counts->GetXaxis()->SetBinLabel(1, "direct");
    pid_energy_fallback_counts->GetXaxis()->SetBinLabel(2, "trackless charged -> neutral");
    int pid_input_files = 0;
    int legacy_input_files = 0;

    // Use matched truth/cluster pairs from t, just like the original energy
    // scatter plot. A cluster matched to several truth particles can appear
    // more than once; these are not exclusive one-to-one matches.
    TH2F *head_regression_pid[kNPidSpecies][kNHeadRegressionColumns];
    TH2F *head_regression_truth[kNPidSpecies][kNHeadRegressionColumns];
    if (!energy_only) for(int ih=0; ih<kNPidSpecies; ++ih){
        for(int ic=0; ic<kNHeadRegressionColumns; ++ic){
            head_regression_pid[ih][ic] = new TH2F(
                Form("head_regression_pid_%s_%d", kPidSpecies[ih], ic),
                Form("%s | PID: %s;MC truth energy (GeV);Head energy (GeV)",
                     kHeadRegressionLabels[ih], kPredictedRegressionLabels[ic]),
                energyMax*10, 0, energyMax, energyMax*10, 0, energyMax);
            head_regression_truth[ih][ic] = new TH2F(
                Form("head_regression_truth_%s_%d", kPidSpecies[ih], ic),
                Form("%s | Truth: %s;MC truth energy (GeV);Head energy (GeV)",
                     kHeadRegressionLabels[ih], kTruthRegressionLabels[ic]),
                energyMax*10, 0, energyMax, energyMax*10, 0, energyMax);
        }
    }
    bool has_five_head_regression = false;
    Long64_t head_regression_matched_pairs = 0;
    Long64_t head_regression_unavailable[kNPidSpecies] = {};
    PidClusteringDiagnostics pid_clustering_diagnostics;

    if (paired_pid) {
        for (int ip=0; ip<nParticle; ++ip) {
            energy_histograms.add(energy[ip]);
            energy_histograms.add(energy_diff[ip]);
            energy_histograms.add(energy2d[ip]);
            energy_histograms.add(clusterenergy2d[ip]);
            energy_histograms.add(energy_resolution[ip]);
            energy_histograms.add(eff_vs_Ediff[ip]);
            energy_histograms.add(pur_vs_Ediff[ip]);
            energy_histograms.add(condbeta_vs_Ediff[ip]);
            energy_histograms.add(condbeta_vs_Ediff_track[ip]);
            energy_histograms.add(condbeta_vs_Ediff_nottrack[ip]);
            for (int ie=0; ie<nEnergy; ++ie) {
                energy_histograms.add(energy_diff_per_energy[ip][ie]);
                energy_histograms.add(energy_resolution_per_energy[ip][ie]);
                energy_histograms.add(energy_resolution_per_energy_cluster[ip][ie]);
            }
            for (int ib=0; ib<kNAbsCosThetaBins; ++ib)
                energy_histograms.add(energy_res_vs_abs_costheta[ip][ib]);
        }
        for (int ip=0; ip<kNTruthPhysCat; ++ip) {
            energy_histograms.add(energy2d_truth_phys[ip]);
            for (int ie=0; ie<nEnergy; ++ie)
                energy_histograms.add(energy_resolution_per_energy_truth_phys[ip][ie]);
        }
        energy_histograms.add(event_energy2d_reco_track);
        energy_histograms.add(event_energy2d_cond_track);
        energy_histograms.add(event_energy_sum_reco_track_distribution);
        energy_histograms.add(event_energy_sum_cond_track_distribution);
        energy_histograms.add(event_residual_all_reco_track);
        energy_histograms.add(event_residual_clean_reco_track);
        energy_histograms.add(event_residual_all_cond_track);
        energy_histograms.add(event_residual_clean_cond_track);
        for (int ib=0; ib<n_beta_thresholds; ++ib) {
            energy_histograms.add(event_energy_sum_reco_track_distribution_beta[ib]);
            energy_histograms.add(event_energy_sum_cond_track_distribution_beta[ib]);
        }
        for (int ie=0; ie<nEnergy_jet; ++ie) {
            energy_histograms.add(jet_energy_resolution_per_energy[ie]);
            energy_histograms.add(event_energy_diff_per_energy_reco_track[ie]);
            energy_histograms.add(event_energy_diff_per_energy_cond_track[ie]);
        }
        for (int category=0; category<6; ++category)
            energy_histograms.add(event_pred_energy_sum_by_track_category[category]);
        TIter remaining(fileout.GetList());
        while (auto* object = remaining())
            if (dynamic_cast<TH1*>(object) && pidReadoutDependentHistogram(object->GetName()))
                throw std::runtime_error(std::string("Unregistered paired energy histogram: ") + object->GetName());
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

    double cedric_eff_num_all = 0.0;
    double cedric_eff_den_all = 0.0;
    double cedric_pur_num_all = 0.0;
    double cedric_pur_den_all = 0.0;
    vector<double> cedric_eff_num(nParticle, 0.0);
    vector<double> cedric_eff_den(nParticle, 0.0);
    vector<double> cedric_pur_num(nParticle, 0.0);
    vector<double> cedric_pur_den(nParticle, 0.0);

    auto particleIndexFromPdg = [&](int pdg) {
        auto result = find(particledgValues.begin(), particledgValues.end(), pdg);
        if(result == particledgValues.end()) return -1;
        return particledgValues_itr[distance(particledgValues.begin(), result)];
    };

    auto processCedricEvent = [&](const vector<CedricHitRecord>& hits) {
        map<int, CedricTruthParticle> particles;
        map<int, CedricRecoCluster> clusters;

        for(const auto& hit : hits){
            if(hit.truthid > 0){
                auto& particle = particles[hit.truthid];
                particle.pdg = hit.pdg;
                particle.charge = hit.charge;
                particle.mcen = hit.mcen;
                particle.energy += hit.is_track ? 0.0 : hit.edep;
            }

            if(hit.cluster == master_unclustered_cluster_id) continue;
            auto& cluster = clusters[hit.cluster];
            cluster.nhits += 1;
            cluster.energy += hit.edep;
            cluster.has_track = cluster.has_track || (hit.is_track != 0);
            if(hit.truthid > 0){
                cluster.overlap[hit.truthid] += hit.is_track ? 0.0 : hit.edep;
                if(hit.is_track) cluster.track_truth_ids[hit.truthid] = true;
            }
        }

        map<int, const CedricRecoCluster*> kept_clusters;
        for(const auto& item : clusters){
            if(item.second.nhits >= cedric_n_min && item.second.energy >= cedric_e_min)
                kept_clusters[item.first] = &item.second;
        }

        // Charged C*: the kept cluster containing the particle's track hit.
        // If multiple charged tracks share a cluster, Cedric assigns it to the
        // charged particle with the largest calorimeter overlap.
        map<int, int> charged_star;
        for(const auto& item : kept_clusters){
            const int cluster_id = item.first;
            const CedricRecoCluster& cluster = *item.second;
            if(!cluster.has_track) continue;

            int owner = -1;
            double best_overlap = -std::numeric_limits<double>::infinity();
            for(const auto& track_item : cluster.track_truth_ids){
                const int truth_id = track_item.first;
                auto particle_it = particles.find(truth_id);
                if(particle_it == particles.end() || particle_it->second.charge == 0) continue;
                auto overlap_it = cluster.overlap.find(truth_id);
                const double overlap = overlap_it == cluster.overlap.end() ? 0.0 : overlap_it->second;
                if(overlap > best_overlap || (overlap == best_overlap && (owner < 0 || truth_id < owner))){
                    owner = truth_id;
                    best_overlap = overlap;
                }
            }
            if(owner >= 0) charged_star[owner] = cluster_id;
        }

        // Neutral ownership is evaluated only for track-free kept clusters.
        // Each cluster is owned by the neutral particle with the largest W(P,C),
        // and each neutral particle keeps only its largest-overlap owned cluster.
        map<int, vector<int>> neutral_owned_clusters;
        for(const auto& item : kept_clusters){
            const int cluster_id = item.first;
            const CedricRecoCluster& cluster = *item.second;
            if(cluster.has_track) continue;

            int owner = -1;
            double best_overlap = 0.0;
            for(const auto& overlap_item : cluster.overlap){
                const int truth_id = overlap_item.first;
                auto particle_it = particles.find(truth_id);
                if(particle_it == particles.end() || particle_it->second.charge != 0) continue;
                const double overlap = overlap_item.second;
                if(overlap > best_overlap || (overlap == best_overlap && overlap > 0.0 && (owner < 0 || truth_id < owner))){
                    owner = truth_id;
                    best_overlap = overlap;
                }
            }
            if(owner >= 0) neutral_owned_clusters[owner].push_back(cluster_id);
        }

        map<int, int> neutral_star;
        for(const auto& item : neutral_owned_clusters){
            const int truth_id = item.first;
            int best_cluster = -1;
            double best_overlap = -1.0;
            for(int cluster_id : item.second){
                const auto* cluster = kept_clusters[cluster_id];
                const double overlap = cluster->overlap.at(truth_id);
                if(overlap > best_overlap){
                    best_cluster = cluster_id;
                    best_overlap = overlap;
                }
            }
            neutral_star[truth_id] = best_cluster;
        }

        for(const auto& item : particles){
            const int truth_id = item.first;
            const CedricTruthParticle& particle = item.second;
            if(particle.energy <= 0.0) continue;

            const auto& star_map = particle.charge != 0 ? charged_star : neutral_star;
            const auto star_it = star_map.find(truth_id);
            const bool matched = star_it != star_map.end();
            double matched_energy = 0.0;
            double matched_cluster_energy = 0.0;
            if(matched){
                const auto cluster_it = kept_clusters.find(star_it->second);
                if(cluster_it != kept_clusters.end()){
                    matched_cluster_energy = cluster_it->second->energy;
                    const auto overlap_it = cluster_it->second->overlap.find(truth_id);
                    if(overlap_it != cluster_it->second->overlap.end())
                        matched_energy = overlap_it->second;
                }
            }

            cedric_eff_num_all += matched_energy;
            cedric_eff_den_all += particle.energy;
            const double efficiency_value = matched_energy / particle.energy;

            const int ip = particleIndexFromPdg(particle.pdg);
            const int ie = truthEnergyBinFromMcen(particle.mcen);
            if(ip >= 0){
                cedric_eff_num[ip] += matched_energy;
                cedric_eff_den[ip] += particle.energy;
                efficiency_cedric[ip]->Fill(efficiency_value, particle.energy);
                efficiency2d[ip]->Fill(efficiency_value, particle.energy, particle.energy);
                if(ie >= 0){
                    efficiency_energy[ip][ie]->Fill(efficiency_value, particle.energy);
                    efficiency_energy_normalize[ip][ie]->Fill(efficiency_value, particle.energy);
                }
            }

            if(matched && matched_cluster_energy > 0.0){
                cedric_pur_num_all += matched_energy;
                cedric_pur_den_all += matched_cluster_energy;
                const double purity_value = matched_energy / matched_cluster_energy;
                if(ip >= 0){
                    cedric_pur_num[ip] += matched_energy;
                    cedric_pur_den[ip] += matched_cluster_energy;
                    purity_cedric[ip]->Fill(purity_value, matched_cluster_energy);
                    purity2d[ip]->Fill(purity_value, particle.energy, matched_cluster_energy);
                    if(ie >= 0){
                        purity_energy[ip][ie]->Fill(purity_value, matched_cluster_energy);
                        purity_energy_normalize[ip][ie]->Fill(purity_value, matched_cluster_energy);
                    }
                }
            }
        }
    };

    for(int irawfile=0; irawfile<rawfilenum; irawfile++){
        if(rawfilenum>1) cout << irawfile << "/" << rawfilenum << "  ";// << endl;
        if (!energy_only) pid_clustering_diagnostics.beginFile();
        map<int, double> event_builder_mc_energy;

        cout << fileNames_raw[irawfile] << endl;
        std::unique_ptr<TFile> current_input(irawfile == 0 ? first_input.release() :
            TFile::Open(fileNames_raw[irawfile].c_str(), "READ"));
        filein[irawfile] = current_input.get();
        if (!filein[irawfile] || filein[irawfile]->IsZombie())
            throw std::runtime_error("Cannot open input ROOT: " + fileNames_raw[irawfile]);
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

        // Activate only fields consumed below; PID/head binders add their fields.
        tree[irawfile]->SetBranchStatus("*", 0);
        if(tree_reco[irawfile]) tree_reco[irawfile]->SetBranchStatus("*", 0);
        PidEnergyReadout truth_pid, reco_pid;
        truth_pid.bind(tree[irawfile]);
        reco_pid.bind(tree_reco[irawfile]);
        PidHeadEnergyReadout truth_heads;
        if(truth_pid.enabled && !energy_only){
            truth_heads.bind(tree[irawfile]);
            if(!truth_heads.enabled)
                cout << "  no saved energy_<species> branches: skipping five-head regression grids" << endl;
        }
        if(truth_pid.enabled != reco_pid.enabled)
            throw std::runtime_error("PID ROOT requires matching PID branches in t and reco: " + fileNames_raw[irawfile]);
        auto* source_metadata = dynamic_cast<TObjString*>(filein[irawfile]->Get("particle_eval_metadata"));
        const int summary_file_index = energy_only ? readout_cache->inputs[irawfile].file_index : irawfile;
        if (energy_only) {
            if (!reco_pid.enabled) throw std::runtime_error("Paired energy pass requires representative PID inputs");
            readout_cache->load(irawfile, source_metadata ? source_metadata->GetString().Data() : "");
        }
        jet_resolution_summary.addInput(summary_file_index, fileNames_raw[irawfile], reco_pid.enabled,
            source_metadata ? source_metadata->GetString().Data() : "");
        if (paired_pid) {
            if (!reco_pid.enabled || !source_metadata)
                throw std::runtime_error("Single-pass paired analysis requires uniform PID inputs with particle metadata");
            std::unique_ptr<char, decltype(&std::free)> canonical_path(
                ::realpath(fileNames_raw[irawfile].c_str(), nullptr), &std::free);
            if (!canonical_path) throw std::runtime_error("Cannot resolve paired source ROOT path");
            for (int mode=0; mode<2; ++mode)
                paired_jer[mode]->addInput(summary_file_index, canonical_path.get(), true,
                    source_metadata->GetString().Data());
        }
        if(truth_pid.enabled) ++pid_input_files;
        else ++legacy_input_files;
        // cout << "  energy readout: " << (truth_pid.enabled ? "energy_pid (saved representative PID)" : "legacy track/calo heads") << endl;


        const char* truth_branches[] = {"event", "mcid", "mcpdg", "mccharge", "mcen",
            "edep", "edep_reco", "edep_match", "pred_edep", "pred_edep_cluster",
            "cond_beta", "cond_track"};
        for(const char* branch_name : truth_branches)
            tree[irawfile]->SetBranchStatus(branch_name, 1);
        tree[irawfile]->SetBranchAddress("event", &event);
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
        if(truth_pid.enabled){
            if(!tree[irawfile]->GetBranch("truthid") || !tree[irawfile]->GetBranch("matched_cluster"))
                throw std::runtime_error("PID diagnostics require t/truthid and t/matched_cluster: " + fileNames_raw[irawfile]);
            tree[irawfile]->SetBranchStatus("truthid", 1);
            tree[irawfile]->SetBranchStatus("matched_cluster", 1);
            tree[irawfile]->SetBranchAddress("truthid", &truthid);
            tree[irawfile]->SetBranchAddress("matched_cluster", &matched_cluster);
        }

        unordered_map<int, double> jet_cos_theta_by_event;
        unordered_map<int, double> jet_thrust_by_event;
        if(jet_regression && tree_jet[irawfile]){
            tree_jet[irawfile]->SetBranchStatus("*", 0);
            for(const char* branch_name : {"event", "n_jets", "jet_p4"})
                tree_jet[irawfile]->SetBranchStatus(branch_name, 1);
            tree_jet[irawfile]->SetBranchAddress("event", &jet_event);
            tree_jet[irawfile]->SetBranchAddress("n_jets", &n_jets);
            tree_jet[irawfile]->SetBranchAddress("jet_p4", &jet_p4);
            for(int ientry=0; ientry<tree_jet[irawfile]->GetEntries(); ientry++){
                if(rawfilenum==1 && tree_jet[irawfile]->GetEntries()>1000 && ientry%100000==0) cout << "jet event  " << ientry << "/" << tree_jet[irawfile]->GetEntries() << endl;
                tree_jet[irawfile]->GetEntry(ientry);

                jet_thrust_by_event[jet_event] = pid_jer::upstreamThrust(jet_p4, n_jets);

                const double px = jet_p4[0][1];
                const double py = jet_p4[0][2];
                const double pz = jet_p4[0][3];
                const double amp = sqrt(px * px + py * py + pz * pz);
                if(amp <= 0.0) continue;
                const double costheta = pz / amp;
                jet_cos_theta_by_event[jet_event] = costheta;
            }
        }

        // Use the event-level energy from the H5 builder, as in the original
        // contribution macro. Summing t/mcen can double-count MC ancestors
        // and descendants and is not a valid event truth energy.
        const bool has_builder_mc_energy =
            tree_event[irawfile] &&
            tree_event[irawfile]->GetBranch("event") &&
            tree_event[irawfile]->GetBranch("MC_dijet_energy");
        if(has_builder_mc_energy){
            tree_event[irawfile]->SetBranchStatus("*", 0);
            tree_event[irawfile]->SetBranchStatus("event", 1);
            tree_event[irawfile]->SetBranchStatus("MC_dijet_energy", 1);
            int builder_event = -1;
            double builder_mc_event_energy = 0.0;
            std::set<int> builder_event_ids;
            if(tree_event[irawfile]->SetBranchAddress("event", &builder_event) < 0 ||
               tree_event[irawfile]->SetBranchAddress("MC_dijet_energy", &builder_mc_event_energy) < 0)
                throw std::runtime_error("Event truth branch type mismatch in " + fileNames_raw[irawfile]);
            for(Long64_t ientry=0; ientry<tree_event[irawfile]->GetEntries(); ++ientry){
                tree_event[irawfile]->GetEntry(ientry);
                if (paired_pid && !builder_event_ids.insert(builder_event).second)
                    throw std::runtime_error("Duplicate source event ID in PID event builder");
                if(builder_mc_event_energy > 0.0 && std::isfinite(builder_mc_event_energy))
                    event_builder_mc_energy[builder_event] = builder_mc_event_energy;
            }
            tree_event[irawfile]->ResetBranchAddresses();
        }else{
            cout << "   warning: event tree with event/MC_dijet_energy is missing; " << "skipping truth-dependent plots for this file" << endl;
        }
        std::vector<CachedRecoRow> cached_reco_rows;
        std::unique_ptr<pid_singlepass::FileReadouts> paired_readouts;
        if (tree_reco[irawfile]) {
            const char* reco_branches[] = {"event", "cluster", "mcid", "mcpdg", "mccharge",
                "edep_reco", "pred_edep", "pred_edep_cluster", "ntrack_hits", "cond_beta",
                "cond_is_track", "matched_truth_pdgid", "matched_truth_edep_frac",
                "npdg_comp", "pdg_comp_ids"};
            for(const char* branch_name : reco_branches)
                tree_reco[irawfile]->SetBranchStatus(branch_name, 1);
            tree_reco[irawfile]->SetBranchAddress("event", &reco_event);
            tree_reco[irawfile]->SetBranchAddress("cluster", &reco_cluster);
            tree_reco[irawfile]->SetBranchAddress("mcid", &reco_mcid);
            tree_reco[irawfile]->SetBranchAddress("mcpdg", &reco_mcpdg);
            tree_reco[irawfile]->SetBranchAddress("mccharge", &reco_mccharge);
            tree_reco[irawfile]->SetBranchAddress("edep_reco", &reco_edep_reco);
            tree_reco[irawfile]->SetBranchAddress("pred_edep", &reco_pred_edep);
            tree_reco[irawfile]->SetBranchAddress("pred_edep_cluster", &reco_pred_edep_cluster);
            tree_reco[irawfile]->SetBranchAddress("ntrack_hits", &reco_ntrack_hits);
            tree_reco[irawfile]->SetBranchAddress("cond_beta", &reco_cond_beta);
            tree_reco[irawfile]->SetBranchAddress("cond_is_track", &reco_cond_is_track);
            tree_reco[irawfile]->SetBranchAddress("matched_truth_pdgid", &reco_matched_truth_pdgid);
            tree_reco[irawfile]->SetBranchAddress("matched_truth_edep_frac", &reco_matched_truth_edep_frac);
            tree_reco[irawfile]->SetBranchAddress("npdg_comp", &reco_npdg_comp);
            tree_reco[irawfile]->SetBranchAddress("pdg_comp_ids", reco_pdg_comp_ids);
        }
        if (paired_pid) {
            if (!has_builder_mc_energy)
                throw std::runtime_error("Single-pass PID readout requires event/MC_dijet_energy");
            if (tree_reco[irawfile]->SetBranchAddress("event", &reco_event) < 0 ||
                tree_reco[irawfile]->SetBranchAddress("cluster", &reco_cluster) < 0 ||
                tree_reco[irawfile]->SetBranchAddress("ntrack_hits", &reco_ntrack_hits) < 0 ||
                tree_reco[irawfile]->SetBranchAddress("cond_beta", &reco_cond_beta) < 0)
                throw std::runtime_error("Invalid paired reco event/cluster/track-count/beta branch type");
            auto* unified_metadata = dynamic_cast<TObjString*>(filein[irawfile]->Get("save_root_metadata"));
            paired_readouts.reset(new pid_singlepass::FileReadouts(source_metadata->GetString().Data(),
                unified_metadata ? unified_metadata->GetString().Data() : "",
                common_staging->path, summary_file_index));
            PidHeadEnergyReadout reco_heads;
            reco_heads.bind(tree_reco[irawfile]);
            if (!reco_heads.enabled) throw std::runtime_error("Paired PID energies require all five saved raw heads");
            int accepted = 1;
            Long64_t seed = -1;
            if (tree_reco[irawfile]->GetBranch("cluster_accepted")) {
                tree_reco[irawfile]->SetBranchStatus("cluster_accepted", 1);
                if (tree_reco[irawfile]->SetBranchAddress("cluster_accepted", &accepted) < 0)
                    throw std::runtime_error("Invalid reco/cluster_accepted type");
            }
            if (tree_reco[irawfile]->GetBranch("pid_seed_input_row")) {
                tree_reco[irawfile]->SetBranchStatus("pid_seed_input_row", 1);
                if (tree_reco[irawfile]->SetBranchAddress("pid_seed_input_row", &seed) < 0)
                    throw std::runtime_error("Invalid reco/pid_seed_input_row type");
            }
            cached_reco_rows.reserve(tree_reco[irawfile]->GetEntries());
            for (Long64_t entry=0; entry<tree_reco[irawfile]->GetEntries(); ++entry) {
                tree_reco[irawfile]->GetEntry(entry);
                CachedRecoRow row;
                row.event=reco_event; row.cluster=reco_cluster; row.mcid=reco_mcid;
                row.pdg=reco_mcpdg; row.charge=reco_mccharge; row.tracks=reco_ntrack_hits;
                row.cond_track=reco_cond_is_track; row.matched_pdg=reco_matched_truth_pdgid;
                row.npdg=reco_npdg_comp; row.beta=reco_cond_beta;
                row.prediction=reco_pred_edep; row.cluster_prediction=reco_pred_edep_cluster;
                row.deposit=reco_edep_reco; row.matched_fraction=reco_matched_truth_edep_frac;
                row.composition.fill(0);
                const int components = std::max(0, std::min(64, reco_npdg_comp));
                std::copy(reco_pdg_comp_ids, reco_pdg_comp_ids+components, row.composition.begin());
                row.pid=reco_pid;
                cached_reco_rows.push_back(row);
                paired_readouts->add(reco_event, reco_cluster, reco_pid.select(0.), reco_pid.pid,
                    reco_pid.energy_head, reco_ntrack_hits, accepted != 0, seed, reco_heads.energy,
                    event_builder_mc_energy.count(reco_event) != 0);
            }
            tree_reco[irawfile]->ResetBranchAddresses();
            paired_readouts->prepareBridge(event_builder_mc_energy, jet_thrust_by_event);
        }
        if (!energy_only) {
        const char* cedric_required_branches[] = {"event", "truthid", "cluster", "mcpdg", "mccharge", "mcen", "edep_mc", "trackness"};
        if(!tree_pred[irawfile]){
            cerr << "ERROR: prediction tree is required for Cedric metrics: " << fileNames_raw[irawfile] << endl;
            return;
        }
        if(tree_pred[irawfile]->GetEntries() == 0){
            cerr << "ERROR: prediction tree is empty in " << fileNames_raw[irawfile] << endl << "Regenerate this ROOT file with save_root_reco_pid.py (full prediction tree)." << endl;
            return;
        }
        for(const char* branch_name : cedric_required_branches){
            if(!tree_pred[irawfile]->GetBranch(branch_name)){
                cerr << "ERROR: missing prediction/" << branch_name << " in " << fileNames_raw[irawfile] << endl << "Regenerate this ROOT file with save_root_reco_pid.py." << endl;
                return;
            }
        }

        // Read only the hit fields used by Cedric and common PID diagnostics.
        tree_pred[irawfile]->SetBranchStatus("*", 0);
        for(const char* branch_name : cedric_required_branches)
            tree_pred[irawfile]->SetBranchStatus(branch_name, 1);

        int cedric_event = -1;
        int cedric_truthid = 0;
        int cedric_cluster = master_unclustered_cluster_id;
        int cedric_pdg = 0;
        int cedric_charge = 0;
        int cedric_track = 0;
        int diagnostic_truth_valid = 1;
        double cedric_mcen = 0.0;
        double cedric_edep = 0.0;
        tree_pred[irawfile]->SetBranchAddress("event", &cedric_event);
        tree_pred[irawfile]->SetBranchAddress("truthid", &cedric_truthid);
        tree_pred[irawfile]->SetBranchAddress("cluster", &cedric_cluster);
        tree_pred[irawfile]->SetBranchAddress("mcpdg", &cedric_pdg);
        tree_pred[irawfile]->SetBranchAddress("mccharge", &cedric_charge);
        tree_pred[irawfile]->SetBranchAddress("mcen", &cedric_mcen);
        tree_pred[irawfile]->SetBranchAddress("edep_mc", &cedric_edep);
        tree_pred[irawfile]->SetBranchAddress("trackness", &cedric_track);
        if(tree_pred[irawfile]->GetBranch("truth_valid")){
            tree_pred[irawfile]->SetBranchStatus("truth_valid", 1);
            tree_pred[irawfile]->SetBranchAddress("truth_valid", &diagnostic_truth_valid);
        }

        vector<CedricHitRecord> cedric_hits;
        int current_cedric_event = -1;
        Long64_t recovery_input_row = -1;
        int recovery_hitid = -1;
        if (paired_readouts && paired_readouts->needsRecovery()) {
            if (!tree_pred[irawfile]->GetBranch("input_row") || !tree_pred[irawfile]->GetBranch("hitid"))
                throw std::runtime_error("Old PID ROOT needs prediction/input_row and hitid for original-H5 recovery");
            tree_pred[irawfile]->SetBranchStatus("input_row", 1);
            tree_pred[irawfile]->SetBranchStatus("hitid", 1);
            if (tree_pred[irawfile]->SetBranchAddress("input_row", &recovery_input_row) < 0 ||
                tree_pred[irawfile]->SetBranchAddress("hitid", &recovery_hitid) < 0)
                throw std::runtime_error("Old PID ROOT needs prediction/input_row and hitid for original-H5 recovery");
        }
        for(Long64_t ientry=0; ientry<tree_pred[irawfile]->GetEntries(); ++ientry){
            tree_pred[irawfile]->GetEntry(ientry);
            if (paired_readouts)
                paired_readouts->prediction(cedric_event, cedric_cluster, recovery_input_row,
                    recovery_hitid, cedric_track);
            if(current_cedric_event >= 0 && cedric_event != current_cedric_event){
                processCedricEvent(cedric_hits);
                if(truth_pid.enabled)
                    pid_clustering_diagnostics.observeEvent(irawfile, current_cedric_event, cedric_hits);
                cedric_hits.clear();
            }
            current_cedric_event = cedric_event;
            CedricHitRecord hit;
            hit.truthid = cedric_truthid;
            hit.cluster = cedric_cluster;
            hit.pdg = cedric_pdg;
            hit.charge = cedric_charge;
            hit.is_track = cedric_track;
            hit.mcen = cedric_mcen;
            hit.edep = cedric_edep;
            hit.truth_valid = diagnostic_truth_valid != 0;
            cedric_hits.push_back(hit);
        }
        if(!cedric_hits.empty()){
            processCedricEvent(cedric_hits);
            if(truth_pid.enabled)
                pid_clustering_diagnostics.observeEvent(irawfile, current_cedric_event, cedric_hits);
        }
        tree_pred[irawfile]->ResetBranchAddresses();
        tree_pred[irawfile]->SetBranchStatus("*", 1);
        }

        if (paired_readouts) {
            paired_readouts->recover(recoverMeasuredPidEnergy);
            paired_eligible += paired_readouts->eligible;
            paired_recovered += paired_readouts->recovered;
        }
        for(int ientry=0; ientry<entry_max[irawfile]; ientry++){
            if(rawfilenum==1 && entry_max[irawfile]>10000 && ientry%100000==0) cout << "t event " << ientry << "/" << entry_max[irawfile] << endl;
            tree[irawfile]->GetEntry(ientry);
            if(truth_pid.enabled){
                pred_edep = pred_edep_cluster = truth_pid.select(pred_edep);
                if (paired_readouts)
                    paired_readouts->select(event, matched_cluster, pred_edep, 0);
                if (energy_only)
                    pred_edep = pred_edep_cluster = readout_cache->select(event, matched_cluster, pred_edep);
                else
                    pid_clustering_diagnostics.matchTruth(event, truthid, matched_cluster, truth_pid.pid, cond_beta, edep, edep_reco, edep_match);
            }
            // Legacy Pandora stores one PFO energy in pred_edep for both
            // charged and neutral clusters; pred_edep_cluster is a zero
            // placeholder, not a neutral regression output.
            if(pandora_mode) pred_edep_cluster = pred_edep;

            if(event_builder_mc_energy.find(event) == event_builder_mc_energy.end()) continue;


            if(edep<=0 || edep_reco<=0 || edep_match<0) continue;
            if(cond_beta<beta_threshold) continue;
            // Keep the original matching and deposit/beta cuts. Fill before
            // the six-species filter so PID All also includes other known MC
            // species (e.g. charged kaons/protons). It is the sum of PID columns.
            if(truth_heads.enabled && truth_pid.energy_head >= 0){
                has_five_head_regression = true;
                if(truth_pid.pid < 0 || truth_pid.pid >= kNPidSpecies)
                    throw std::runtime_error("Invalid matched-cluster PID in " + fileNames_raw[irawfile]);
                if(mcid >= 0 && mcen > 0.0 && std::isfinite(mcen)){
                    ++head_regression_matched_pairs;
                    const int truth_column = headRegressionTruthColumn(mcpdg);
                    for(int ih=0; ih<kNPidSpecies; ++ih){
                        const double head_energy = truth_heads.energy[ih];
                        if(!std::isfinite(head_energy)){
                            ++head_regression_unavailable[ih];
                            continue;
                        }
                        head_regression_pid[ih][truth_pid.pid]->Fill(mcen, head_energy);
                        head_regression_pid[ih][kNHeadRegressionColumns-1]->Fill(mcen, head_energy);
                        if(truth_column >= 0)
                            head_regression_truth[ih][truth_column]->Fill(mcen, head_energy);
                    }
                }
            }
            auto result = find(particledgValues.begin(), particledgValues.end(), mcpdg);
            if(result == particledgValues.end()) continue;
            int itr = particledgValues_itr[distance(particledgValues.begin(), result)];

            const double purity_value = edep_match / edep_reco;
            const double efficiency_value = edep_match / edep;
            if(edep>1){
                purity[itr]->Fill(purity_value);
                efficiency[itr]->Fill(efficiency_value);
            }

            int energy_itr = truthEnergyBinFromMcen(mcen);

            MCtruth_energy[itr]->Fill(mcen);
            condbeta_vs_mcen[itr]->Fill(cond_beta,mcen);
            if(cond_track==1) condbeta_vs_mcen_track[itr]->Fill(cond_beta,mcen);
            else condbeta_vs_mcen_nottrack[itr]->Fill(cond_beta,mcen);
            auto fill_truth_energy = [&](int mode, double pred_edep, double pred_edep_cluster) {
                energy_histograms.at(energy[itr], mode)->Fill(pred_edep);

                energy_histograms.at(energy_diff[itr], mode)->Fill(pred_edep - mcen);
                energy_histograms.at(energy2d[itr], mode)->Fill(mcen,pred_edep);
                energy_histograms.at(clusterenergy2d[itr], mode)->Fill(mcen,pred_edep_cluster);
                energy_histograms.at(energy_resolution[itr], mode)->Fill( (pred_edep - mcen) / mcen );

                const int ipc = truthPhysicsCategoryFromMc(mcpdg, mccharge);
                const double ipc_energy = ipc == 0 ? pred_edep : pred_edep_cluster;
                energy_histograms.at(energy2d_truth_phys[ipc], mode)->Fill(mcen, ipc_energy);

                // cout << itr_energy << ", " << mcen << endl;
                if(energy_itr >= 0 && pred_edep>0.1){
                    energy_histograms.at(energy_diff_per_energy[itr][energy_itr], mode)->Fill(pred_edep - mcen);
                    energy_histograms.at(energy_resolution_per_energy[itr][energy_itr], mode)->Fill( (pred_edep - mcen) / mcen );
                }
                if(jet_regression && energy_itr == kRms90VsAbsCosThetaTruthEnergyBin && pred_edep > 0.1){
                    const auto jct = jet_cos_theta_by_event.find(event);
                    if(jct != jet_cos_theta_by_event.end()){
                        const double abs_ct = fabs(jct->second);
                        if(abs_ct <= 1.0 + 1e-6){
                            const int ib_ct = absCosThetaBinIndex(abs_ct);
                            if(ib_ct >= 0)
                                energy_histograms.at(energy_res_vs_abs_costheta[itr][ib_ct], mode)->Fill((pred_edep - mcen) / mcen);
                        }
                    }
                }
                if(energy_itr >= 0 && pred_edep_cluster>0.1){
                    energy_histograms.at(energy_resolution_per_energy_cluster[itr][energy_itr], mode)->Fill( (pred_edep_cluster - mcen) / mcen );
                }
                if(energy_itr >= 0){
                    energy_histograms.at(energy_resolution_per_energy_truth_phys[ipc][energy_itr], mode)->Fill((ipc_energy - mcen) / mcen);
                }

                energy_histograms.at(condbeta_vs_Ediff[itr], mode)->Fill(cond_beta,pred_edep-mcen);
                if(cond_track==1) energy_histograms.at(condbeta_vs_Ediff_track[itr], mode)->Fill(cond_beta,pred_edep-mcen);
                else energy_histograms.at(condbeta_vs_Ediff_nottrack[itr], mode)->Fill(cond_beta,pred_edep-mcen);
            };
            if (paired_pid) {
                for (int mode=0; mode<2; ++mode) {
                    const double selected = paired_readouts->select(event, matched_cluster, pred_edep, mode);
                    fill_truth_energy(mode, selected, selected);
                }
            } else fill_truth_energy(-1, pred_edep, pred_edep_cluster);

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

        if(tree_reco[irawfile]){
            std::array<map<int, EventRecoEnergySummary>, 2> event_energy_summaries;
            map<int, EventConfusionSummary> event_confusion_summary;
            std::array<map<int, EventTrackCategoryEnergySummary>, 2> event_category_summaries;
            const int energy_modes = paired_pid ? 2 : 1;
            for (int mode=0; mode<energy_modes; ++mode) for (const auto& truth : event_builder_mc_energy) {
                event_energy_summaries[mode][truth.first];
                event_category_summaries[mode][truth.first];
            }


            for(int ientry=0; ientry<tree_reco[irawfile]->GetEntries(); ientry++){
                if(rawfilenum==1 && tree_reco[irawfile]->GetEntries()>1000 && ientry%100000==0) cout << "reco event  " << ientry << "/" << tree_reco[irawfile]->GetEntries() << endl;
                if (paired_pid) {
                    const auto& row = cached_reco_rows.at(ientry);
                    reco_event=row.event; reco_cluster=row.cluster; reco_mcid=row.mcid;
                    reco_mcpdg=row.pdg; reco_mccharge=row.charge; reco_ntrack_hits=row.tracks;
                    reco_cond_is_track=row.cond_track; reco_matched_truth_pdgid=row.matched_pdg;
                    reco_npdg_comp=row.npdg; reco_cond_beta=row.beta;
                    reco_pred_edep=row.prediction; reco_pred_edep_cluster=row.cluster_prediction;
                    reco_edep_reco=row.deposit; reco_matched_truth_edep_frac=row.matched_fraction;
                    std::copy(row.composition.begin(), row.composition.end(), reco_pdg_comp_ids);
                    reco_pid=row.pid;
                } else tree_reco[irawfile]->GetEntry(ientry);
                if(reco_cond_beta<beta_threshold) continue;
                if(reco_pid.enabled){
                    if(reco_pid.pid < 0 || reco_pid.pid >= kNPidSpecies ||
                       reco_pid.energy_head < -1 || reco_pid.energy_head >= kNPidSpecies ||
                       (reco_pid.energy_fallback != 0 && reco_pid.energy_fallback != 1))
                        throw std::runtime_error("Invalid reco PID/head/fallback in " + fileNames_raw[irawfile]);
                    reco_pid.select(0.0);  // Validate even for an event without truth.
                    const int truth_species = reco_mcid >= 0 ? pidSpeciesFromMc(reco_mcpdg, reco_mccharge) : -1;
                    pid_confusion_reco->Fill((truth_species < 0 ? 5 : truth_species) + 0.5, reco_pid.pid + 0.5);
                    pid_prediction_counts->Fill(reco_pid.pid + 0.5);
                    pid_energy_head_vs_prediction->Fill(reco_pid.pid + 0.5,
                        (reco_pid.energy_head < 0 ? 5 : reco_pid.energy_head) + 0.5);
                    pid_energy_fallback_counts->Fill(reco_pid.energy_fallback + 0.5);
                }
                if(event_builder_mc_energy.find(reco_event) == event_builder_mc_energy.end()) continue;

                const double saved_reco = reco_pid.select(pandora_mode ? reco_pred_edep :
                    (reco_ntrack_hits>0 ? reco_pred_edep : reco_pred_edep_cluster));
                const double saved_cond = reco_pid.select(pandora_mode ? reco_pred_edep :
                    (reco_cond_is_track!=0 ? reco_pred_edep : reco_pred_edep_cluster));
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
                for (int mode=0; mode<energy_modes; ++mode) {
                auto &sum = event_energy_summaries[mode][reco_event];
                const double pred_reco = paired_pid ? paired_readouts->select(reco_event, reco_cluster, saved_reco, mode) : energy_only ? readout_cache->select(reco_event, reco_cluster, saved_reco) : saved_reco;
                const double pred_cond = paired_pid ? paired_readouts->select(reco_event, reco_cluster, saved_cond, mode) : energy_only ? readout_cache->select(reco_event, reco_cluster, saved_cond) : saved_cond;
                sum.pred_energy_reco_track += pred_reco;
                sum.pred_energy_cond_track += pred_cond;
                for(int ib=0; ib<n_beta_thresholds; ib++){
                    double beta_min = 0.1 * ib;
                    if(reco_cond_beta >= beta_min){
                        sum.pred_energy_reco_track_beta[ib] += pred_reco;
                        sum.pred_energy_cond_track_beta[ib] += pred_cond;
                    }
                }
                    event_category_summaries[mode][reco_event].pred_energy_sum[track_charge_category-1] += pred_reco;
                }

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

            for (int mode=0; mode<energy_modes; ++mode) {
                if (paired_pid) energy_histograms.activate(mode);
                const auto& event_energy_summary = event_energy_summaries[mode];
                const auto& event_track_category_energy_summary = event_category_summaries[mode];
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
                    if (mode == 0) c_event_distribution->Fill(c_event);
                }

                auto truth_it = event_builder_mc_energy.find(entry.first);
                if(truth_it == event_builder_mc_energy.end()) continue;
                const double truth_energy = truth_it->second;
                if(c_event>=0){
                    if (mode == 0) c_event_vs_truth_energy->Fill(truth_energy, c_event);
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
            // Persist every event with a valid builder truth energy, including
            // zero response when no reconstructed cluster survives the beta cut.
            for(const auto& truth : event_builder_mc_energy){
                const auto angle = jet_thrust_by_event.find(truth.first);
                const auto reco = event_energy_summary.find(truth.first);
                auto& summary = paired_pid ? *paired_jer[mode] : jet_resolution_summary;
                summary.observe(summary_file_index, truth.first,
                    angle == jet_thrust_by_event.end() ? std::numeric_limits<double>::quiet_NaN() : angle->second,
                    reco == event_energy_summary.end() ? 0. : reco->second.pred_energy_reco_track,
                    truth.second);
            }
            for(const auto &entry : event_track_category_energy_summary){
                if(event_builder_mc_energy.find(entry.first) == event_builder_mc_energy.end()) continue;
                const EventTrackCategoryEnergySummary &cat_sum = entry.second;
                for(int ic=0; ic<6; ic++){
                    event_pred_energy_sum_by_track_category[ic]->Fill(cat_sum.pred_energy_sum[ic]);
                }
            }
            }
        }

        if(truth_pid.enabled && !energy_only) pid_clustering_diagnostics.finishFile();
        filein[irawfile]->Close();
    }

    if(pid_input_files > 0){
        cout << "PID readout: " << pid_input_files << " PID input file(s), "
             << legacy_input_files << " legacy input file(s)" << endl;
        cout << "  reco clusters after beta cut = " << pid_prediction_counts->GetEntries()
             << ", trackless charged fallback = " << pid_energy_fallback_counts->GetBinContent(2) << endl;
        for(int ip=0; ip<kNPidSpecies; ++ip)
            cout << "  predicted " << kPidSpecies[ip] << " = " << pid_prediction_counts->GetBinContent(ip+1) << endl;
    }

    const int initial_opt_fit = gStyle->GetOptFit();
    auto cedricRatio = [](double numerator, double denominator) {
        return denominator > 0.0 ? numerator / denominator : -1.0;
    };
    cout << "Cedric clustering metrics (n_min=" << cedric_n_min
         << ", e_min=" << cedric_e_min << " GeV)" << endl;
    cout << "  inclusive efficiency = "
         << cedricRatio(cedric_eff_num_all, cedric_eff_den_all)
         << "  (" << cedric_eff_num_all << "/" << cedric_eff_den_all << ")" << endl;
    cout << "  inclusive purity     = "
         << cedricRatio(cedric_pur_num_all, cedric_pur_den_all)
         << "  (" << cedric_pur_num_all << "/" << cedric_pur_den_all << ")" << endl;
    for(int ip=0; ip<nParticle; ++ip){
        const double eff_value = cedricRatio(cedric_eff_num[ip], cedric_eff_den[ip]);
        const double pur_value = cedricRatio(cedric_pur_num[ip], cedric_pur_den[ip]);
        cout << "  " << particleNames[ip]
             << " efficiency=" << eff_value
             << " purity=" << pur_value << endl;
        efficiency_cedric[ip]->SetTitle(Form(
            "%s Cedric efficiency = %.4f (truth-energy weighted);efficiency W(P,C*)/W(P);weighted energy (GeV)",
            particleNames[ip].c_str(), eff_value));
        purity_cedric[ip]->SetTitle(Form(
            "%s Cedric purity = %.4f (cluster-energy weighted);purity W(P,C*)/W(C*);weighted energy (GeV)",
            particleNames[ip].c_str(), pur_value));
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

    
    TCanvas *compare_cedric = new TCanvas("compare_cedric","compare_cedric",2560,1440);
    compare_cedric->Divide(nParticle,2);
    for(int ip=0;ip<nParticle*2;ip++){
        compare_cedric->cd(ip+1);
        gPad->SetLogy();
        if(ip<nParticle) efficiency_cedric[ip]->Draw();
        else purity_cedric[ip-nParticle]->Draw();
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

    TCanvas *compare = new TCanvas("compare","compare",2560,1440);
    compare->Divide(nParticle,2);
    for(int ip=0;ip<nParticle*2;ip++){
        compare->cd(ip+1);
        gPad->SetLogy();
        if(ip<nParticle) efficiency[ip]->Draw();
        else purity[ip-nParticle]->Draw();
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

    
    gStyle->SetOptFit(1111);
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

    TCanvas *canvas_pid = nullptr;
    if(pid_input_files > 0){
        canvas_pid = new TCanvas("canvas_pid", "Representative PID diagnostics", 1800, 1200);
        canvas_pid->Divide(2,2);
        canvas_pid->cd(1);
        pid_confusion_reco->SetStats(0);
        pid_confusion_reco->Draw("COLZ TEXT");
        canvas_pid->cd(2);
        pid_energy_head_vs_prediction->SetStats(0);
        pid_energy_head_vs_prediction->Draw("COLZ TEXT");
        canvas_pid->cd(3);
        pid_prediction_counts->Draw("HIST");
        canvas_pid->cd(4);
        pid_energy_fallback_counts->Draw("HIST");
    }

    TCanvas *canvas_head_regression_pid = nullptr;
    TCanvas *canvas_head_regression_truth = nullptr;
    if(has_five_head_regression){
        cout << "Five-head regression grids: " << head_regression_matched_pairs
             << " matched truth/cluster pairs (same t matching and cuts as regression_cond_cluster)" << endl;
        for(int ih=0; ih<kNPidSpecies; ++ih)
            cout << "  " << kPidSpecies[ih] << ": " << head_regression_unavailable[ih]
                 << " unavailable candidate energies skipped" << endl;
        auto draw_head_grid = [&](const char* name, const char* title,
                                  TH2F* histograms[kNPidSpecies][kNHeadRegressionColumns]) {
            TCanvas* canvas = new TCanvas(name, title, 3000, 2250);
            canvas->Divide(kNHeadRegressionColumns, kNPidSpecies, 0.002, 0.002);
            for(int ih=0; ih<kNPidSpecies; ++ih){
                for(int ic=0; ic<kNHeadRegressionColumns; ++ic){
                    canvas->cd(ih*kNHeadRegressionColumns + ic + 1);
                    gPad->SetLeftMargin(0.15);
                    gPad->SetRightMargin(0.15);
                    gPad->SetBottomMargin(0.14);
                    gPad->SetTopMargin(0.10);
                    TH2F* histogram = histograms[ih][ic];
                    histogram->SetStats(0);
                    histogram->GetXaxis()->SetTitleSize(0.045);
                    histogram->GetYaxis()->SetTitleSize(0.045);
                    histogram->GetXaxis()->SetLabelSize(0.04);
                    histogram->GetYaxis()->SetLabelSize(0.04);
                    histogram->GetYaxis()->SetTitleOffset(1.5);
                    histogram->Draw("COLZ");
                    TLine diagonal(0, 0, energyMax, energyMax);
                    diagonal.SetLineStyle(2);
                    diagonal.SetLineColor(kGray+2);
                    diagonal.DrawClone();
                    TLatex label;
                    label.SetNDC();
                    label.SetTextSize(0.035);
                    label.DrawLatex(0.18, 0.83, Form("N = %.0f", histogram->GetEntries()));
                }
            }
            return canvas;
        };
        canvas_head_regression_pid = draw_head_grid("canvas_head_regression_pid",
            "Five energy heads by predicted PID", head_regression_pid);
        canvas_head_regression_truth = draw_head_grid("canvas_head_regression_truth",
            "Five energy heads by matched truth particle", head_regression_truth);
    }

    const bool split_pid = pid_input_files > 0;
    if (split_pid && legacy_input_files > 0)
        throw std::runtime_error("Paired PID energy analysis requires a uniform representative-PID input list");
    vector<pair<string, TCanvas*>> common_canvases = {
        {"efficiency_purity_cedric", compare_cedric},
        {"efficiency_purity", compare},
        {"category", canvas_clustering_quality}
    };
    if (canvas_pid) common_canvases.push_back({"pid_diagnostics", canvas_pid});
    if (canvas_head_regression_pid) common_canvases.push_back({"regression_heads_pid", canvas_head_regression_pid});
    if (canvas_head_regression_truth) common_canvases.push_back({"regression_heads_truth", canvas_head_regression_truth});
    if (split_pid && !energy_only) {
        const auto diagnostics = pid_clustering_diagnostics.draw(pid_confusion_reco);
        common_canvases.insert(common_canvases.end(), diagnostics.begin(), diagnostics.end());
    }
    if (saving_canvas && !energy_only)
        for (const auto& item : common_canvases)
            saveCanvas(item.second, imageDirectory, item.first);

    auto write_metadata = [&](const char* mode, bool selected) {
        TNamed("result_scope", selected ? "Selected-energy dependent histograms and plots" :
            (split_pid ? "Common clustering, PID and raw energy-head candidate diagnostics" : "Legacy analysis")).Write();
        TNamed("energy_readout", mode).Write();
        TNamed("method", analysis_method.c_str()).Write();
        TParameter<int>("beam_energy_GeV", qq_energy).Write();
        TParameter<double>("beta_threshold", beta_threshold).Write();
    };
    fileout.cd();
    write_metadata(energy_only ? readout_cache->mode.c_str() : (split_pid ? "common" : "saved"), energy_only);
    if (!energy_only) {
        for (const auto& item : common_canvases) item.second->Write(item.first.c_str());
        if (split_pid) {
            pid_clustering_diagnostics.writeOverlapHistograms();
            TNamed definitions("pid_clustering_diagnostics_definitions",
                Form("PID matrix: per reco cluster, dominant saved truth, unknown truth retained. "
                     "Success: per known input truth object in five PID classes from prediction; "
                     "saved t match after beta >= %g; efficiency=edep_match/edep, purity=edep_match/edep_reco. "
                     "Cluster OK requires efficiency >= %g and purity >= %g. "
                     "Unmatched and undefined overlap/PID remain in the denominator. "
                     "Overlap matrices: deposited input energy, including unknown/unclustered, Other bins aggregate omitted IDs; "
                     "sample limit %d across files. Clustering diagnostics use original definitions, not Cedric.",
                     beta_threshold, pid_cluster_success_efficiency, pid_cluster_success_purity, pid_overlap_event_limit));
            definitions.Write();
        }
    }
    if (!energy_only) {
        vector<pid_cluster::Row> clustering_rows;
        for (int ip=0; ip<nParticle; ++ip) {
            const int particle = !kaon_neutron && ip == 3 ? 5 : ip;
            pid_cluster::Row original;
            original.particle = particle;
            original.efficiency = pid_cluster::histogramMetric(*efficiency[ip]);
            original.purity = pid_cluster::histogramMetric(*purity[ip]);
            clustering_rows.push_back(original);
            pid_cluster::Row cedric;
            cedric.definition = 1; cedric.particle = particle;
            cedric.efficiency.numerator = cedric_eff_num[ip];
            cedric.efficiency.denominator = cedric_eff_den[ip];
            cedric.efficiency.entries = efficiency_cedric[ip]->GetEntries();
            cedric.purity.numerator = cedric_pur_num[ip];
            cedric.purity.denominator = cedric_pur_den[ip];
            cedric.purity.entries = purity_cedric[ip]->GetEntries();
            clustering_rows.push_back(cedric);
        }
        const std::string common_source = common_staging
            ? picDirectory + "/" + pid_jer::kSummaryName + ":/clustering/diagnostics" : output_path;
        pid_cluster::write(result_directory, qq_energy,
            pandora_mode ? "pandora" : (use_truth_clustering ? "truth_clustering" : "reco"),
            beta_threshold, clustering_rows, common_source);
    }

    auto render_energy = [&](const std::string& mode) {
        // Each bank is drawn/rebinned once with the same starting ROOT style.
        gStyle->SetOptFit(initial_opt_fit);
        gStyle->SetOptStat("rme");
        gStyle->SetStatX(0.55); gStyle->SetStatY(0.9);
        gStyle->SetStatH(0.3); gStyle->SetStatW(0.4);
        if(pid_input_files > 0 && legacy_input_files == 0){
            // Both panels display the same selected PID energy for this mode.
            // Retain histogram names so existing plotting workflows still work.
            event_energy_sum_reco_track_distribution->SetTitle("Event energy sum (representative PID);reconstructed event energy sum;entries");
            event_energy_sum_cond_track_distribution->SetTitle("Event energy sum (representative PID);reconstructed event energy sum;entries");
            event_energy2d_reco_track->SetTitle("Event energy (representative PID);truth event energy;reconstructed event energy");
            event_energy2d_cond_track->SetTitle("Event energy (representative PID);truth event energy;reconstructed event energy");
            for(int ib=0; ib<n_beta_thresholds; ++ib){
                const string title = Form("Event energy sum (representative PID, #beta>=%.1f);reconstructed event energy sum;entries", 0.1*ib);
                event_energy_sum_reco_track_distribution_beta[ib]->SetTitle(title.c_str());
                event_energy_sum_cond_track_distribution_beta[ib]->SetTitle(title.c_str());
            }
        }
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


// #if 0
    /*
    // jet_energy2d の Fill が無い現行 ROOT では未使用。復活させる場合は TH2F の生成と Fill を戻す。
    TCanvas *canvas_jet_energy_ = new TCanvas("canvas_jet_energy","canvas_jet_energy",1400,500);
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
    */
// #endif

    TCanvas *cv_rms90_abscostheta = nullptr;
    TGraphErrors *gr_rms90_norm_abscostheta[nParticle] = {};
    if(jet_regression){
        cv_rms90_abscostheta = new TCanvas("cv_rms90_abscostheta", "cv_rms90_abscostheta", 2400, 400);
        cv_rms90_abscostheta->Divide(nParticle, 1);
        const double Eref_gev = truthEnergyBinCenter(kRms90VsAbsCosThetaTruthEnergyBin);
        const double inv_sqrt_E = (Eref_gev > 0.0) ? (1.0 / sqrt(Eref_gev)) : 1.0;
        for(int ip=0; ip<nParticle; ip++){
            gr_rms90_norm_abscostheta[ip] = new TGraphErrors();
            gr_rms90_norm_abscostheta[ip]->SetName(Form("gr_rms90_norm_abscostheta_%d", ip));
            int npt = 0;
            for(int ib=0; ib<kNAbsCosThetaBins; ib++){
                TH1F *h = energy_res_vs_abs_costheta[ip][ib];
                if(h->GetEntries() < 30) continue;
                const double rms90 = calculateRMS90(h);
                const double nent = h->GetEffectiveEntries();
                const double y = rms90 * inv_sqrt_E;
                const double yerr = (nent > 1.0) ? (y / sqrt(nent)) : 0.0;
                const double xcenter = 0.5 * (kAbsCosThetaEdges[ib] + kAbsCosThetaEdges[ib + 1]);
                const double xhw = 0.5 * (kAbsCosThetaEdges[ib + 1] - kAbsCosThetaEdges[ib]);
                gr_rms90_norm_abscostheta[ip]->SetPoint(npt, xcenter, y);
                gr_rms90_norm_abscostheta[ip]->SetPointError(npt, xhw, yerr);
                npt++;
            }
            cv_rms90_abscostheta->cd(ip + 1);
            gr_rms90_norm_abscostheta[ip]->SetTitle(Form("%s (truth %.0f-%.0f GeV)",
                particleNames[ip].c_str(),
                kTruthEnergyBinEdges[kRms90VsAbsCosThetaTruthEnergyBin],
                kTruthEnergyBinEdges[kRms90VsAbsCosThetaTruthEnergyBin + 1]));
            gr_rms90_norm_abscostheta[ip]->GetXaxis()->SetTitle("|cos#theta|");
            gr_rms90_norm_abscostheta[ip]->GetYaxis()->SetTitle("RMS_{90} / #sqrt{E / GeV}");
            gr_rms90_norm_abscostheta[ip]->SetMarkerStyle(kFullCircle);
            gr_rms90_norm_abscostheta[ip]->SetMarkerColor(kBlack);
            gr_rms90_norm_abscostheta[ip]->SetLineColor(kBlack);
            if(npt > 0){
                gr_rms90_norm_abscostheta[ip]->Draw("APE");
                gr_rms90_norm_abscostheta[ip]->GetXaxis()->SetRangeUser(0.0, 1.0);
                gr_rms90_norm_abscostheta[ip]->GetYaxis()->SetRangeUser(0.0, 1.0);
            }else{
                auto *hframe = new TH2F(
                    Form("hframe_rms90_ct_%d", ip),
                    Form("%s;|cos#theta|;RMS_{90} / #sqrt{E / GeV}", particleNames[ip].c_str()),
                    10,
                    0.0,
                    1.0,
                    10,
                    0.0,
                    1.0);
                hframe->SetDirectory(nullptr);
                hframe->SetStats(0);
                hframe->Draw();
            }
        }
    }

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


    
    vector<pair<string, TCanvas*>> energy_canvases = {
        {"regression_cond_cluster", canvas_energy},
        {"energy_scan", canvas_energy_scan},
        {"energy_resolution_scan", canvas_energy_resolution_scan},
        {"pfa_category_energy_regression", canvas_truth_phys_energy_regression},
        {"energy_regression", canvas_energy_regression_result},
        {"dijet_energy_resolution", canvas_event_energy_resolution},
        {"dijet_energy_resolution_beta", canvas_event_energy_sum_by_beta},
        {"dijet_scatter", canvas_event_energy_2d},
        {"category_energy", canvas_event_pred_energy_by_track_category}
    };
    if (cv_rms90_abscostheta)
        energy_canvases.push_back({Form("rms90_vs_abs_costheta_truthBin%d_%.0f-%.0fGeV",
            kRms90VsAbsCosThetaTruthEnergyBin,
            kTruthEnergyBinEdges[kRms90VsAbsCosThetaTruthEnergyBin],
            kTruthEnergyBinEdges[kRms90VsAbsCosThetaTruthEnergyBin + 1]), cv_rms90_abscostheta});

        if (saving_canvas)
            for (const auto& item : energy_canvases)
                saveCanvas(item.second, imageDirectory, item.first, mode);
        for (const auto& item : energy_canvases) item.second->Write(item.first.c_str());
        canvas_confusion_eval->Write("confusion_term_evaluation");
        canvas_cluster_energy_resolution_scan->Write("cluster_energy_resolution_scan");
    };

    if (paired_pid) {
        for (int mode=0; mode<2; ++mode) {
            energy_histograms.activate(mode);
            const std::string mode_directory = common_staging->path + "/" + paired_modes[mode];
            gSystem->mkdir(mode_directory.c_str(), true);
            {
                TFile selected((mode_directory + "/energy_regression_result.root").c_str(), "RECREATE");
                if (selected.IsZombie()) throw std::runtime_error("Cannot stage paired energy diagnostics");
                write_metadata(paired_modes[mode], true);
                render_energy(paired_modes[mode]);
                energy_histograms.write(selected, mode);
                selected.Close();
            }
            paired_jer[mode]->write(mode_directory, saving_canvas, imageDirectory, paired_modes[mode]);
            TFile summary((mode_directory + "/" + pid_jer::kSummaryName).c_str(), "UPDATE");
            if (summary.IsZombie()) throw std::runtime_error("Cannot stage paired readout metadata");
            nlohmann::json recovery = {
                {"source_summary", picDirectory + "/" + pid_jer::kSummaryName + ":/" + paired_modes[mode]},
                {"energy_readout", paired_modes[mode]},
                {"single_track_charged_clusters", paired_eligible},
                {"h5_recovered_clusters", paired_recovered},
                {"recovery", "For h5_recovered_clusters: original H5 measured feature columns 7:10; input identity, float32 event truth/angle and all saved input rows validated. Other eligible clusters use the explicitly recorded measured-track policy and mass hypotheses."}
            };
            TObjString(recovery.dump().c_str()).Write("readout_recovery");
            summary.Close();
        }
    } else {
        fileout.cd();
        render_energy(energy_only ? readout_cache->mode : "");
    }
    fileout.cd();
    TIter next_object(fileout.GetList());
    while (auto* object = next_object()) {
        const bool dependent = pidReadoutDependentHistogram(object->GetName());
        if (!split_pid || dependent == energy_only) object->Write();
    }
    fileout.Close();
    if (!energy_only) {
        if (paired_pid) {
            savePairedPidEnergyReadouts(common_staging->path, picDirectory, saving_canvas);
        } else {
            jet_resolution_summary.write(picDirectory, saving_canvas);
        }
    }
}

// Public entry point called by the native executable.
void efficiency_purity_check_reco_effpur_contiribution_pid(
    const char* input_file = nullptr,
    bool pandora_mode = default_pandora,
    const char* output_directory = nullptr){
    efficiency_purity_check_reco_effpur_contiribution_w_Cedric(
        input_file, pandora_mode, output_directory);
}

// Internal native subprocess entry; the Python helper stages and publishes
// both modes together after each pass has successfully completed.
void efficiency_purity_check_pid_energy_readout(
    const char* readout_cache, const char* energy_readout, const char* output_directory,
    const char* image_directory) {
    if (!readout_cache || !readout_cache[0] || !output_directory || !output_directory[0])
        throw std::runtime_error("Paired energy pass requires cache and output directory");
    PidEnergyReadoutCache cache(readout_cache, energy_readout);
    efficiency_purity_check_reco_effpur_contiribution_w_Cedric(
        nullptr, false, output_directory, &cache, image_directory);
}
