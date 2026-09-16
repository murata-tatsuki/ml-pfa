#ifndef MURATA_SCAN_FAST_ROOT_WRITER_CXX
#define MURATA_SCAN_FAST_ROOT_WRITER_CXX

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>

#include "Compression.h"
#include "TFile.h"
#include "TTree.h"

// This class keeps the legacy TTree/leaf layout, but receives whole NumPy
// batches from Python.  The loops over TTree::Fill therefore run in C++ rather
// than crossing the Python/PyROOT boundary once per entry.
class ScanFastRootWriter {
public:
  explicit ScanFastRootWriter(const char *filename, int compression_level = 1)
      : file_(TFile::Open(filename, "RECREATE")) {
    if (!file_ || file_->IsZombie()) {
      throw std::runtime_error(std::string("Cannot create ROOT file: ") + filename);
    }
    file_->SetCompressionAlgorithm(ROOT::kLZ4);
    file_->SetCompressionLevel(compression_level);
    book_trees();
  }

  ~ScanFastRootWriter() { close(); }

  ScanFastRootWriter(const ScanFastRootWriter &) = delete;
  ScanFastRootWriter &operator=(const ScanFastRootWriter &) = delete;

  // int columns:
  // event, hitid, mcid, truthid, mcpdg, mccharge, mcstatus,
  // ncluster, matched_ncluster, matched_cluster, cond_track
  // double columns:
  // mcmass, mcpx, mcpy, mcpz, mcen, edep, edep_reco, edep_match,
  // pred_edep, pred_edep_cluster, pred_edep_weight, cond_beta, sed_radius,
  // pred_photon_energy, pred_charged_hadron_energy,
  // pred_neutral_hadron_energy, pred_muon_energy, pred_electron_energy
  void append_t(long long n, unsigned long long int_address,
                unsigned long long double_address) {
    const auto *iv = address<std::int32_t>(int_address);
    const auto *dv = address<double>(double_address);
    for (long long row = 0; row < n; ++row) {
      const auto *i = iv + row * 11;
      const auto *d = dv + row * 18;
      t_event_ = i[0];
      t_hitid_ = i[1];
      t_mcid_ = i[2];
      t_truthid_ = i[3];
      t_mcpdg_ = i[4];
      t_mccharge_ = i[5];
      t_mcstatus_ = i[6];
      t_ncluster_ = i[7];
      t_matched_ncluster_ = i[8];
      t_matched_cluster_ = i[9];
      t_cond_track_ = i[10];
      t_mcmass_ = d[0];
      t_mcpx_ = d[1];
      t_mcpy_ = d[2];
      t_mcpz_ = d[3];
      t_mcen_ = d[4];
      t_edep_ = d[5];
      t_edep_reco_ = d[6];
      t_edep_match_ = d[7];
      t_pred_edep_ = d[8];
      t_pred_edep_cluster_ = d[9];
      t_pred_edep_weight_ = d[10];
      t_cond_beta_ = d[11];
      t_sed_radius_ = d[12];
      t_pred_photon_energy_ = d[13];
      t_pred_charged_hadron_energy_ = d[14];
      t_pred_neutral_hadron_energy_ = d[15];
      t_pred_muon_energy_ = d[16];
      t_pred_electron_energy_ = d[17];
      t_->Fill();
    }
  }

  // scalar int columns:
  // event, cluster, nhits, mcid, mcpdg, mccharge, mcstatus, ntrack_hits,
  // cond_is_track, matched_truth_pdgid, npdg_comp
  // scalar double columns:
  // mcmass, mcpx, mcpy, mcpz, mcen, edep_reco, edep_mc, edep_match,
  // pred_edep, pred_edep_cluster, cond_beta, matched_truth_hit_frac,
  // matched_truth_edep_frac
  // composition int blocks: ids[64], hits[64], track_hits[64]
  // composition double blocks: hit_frac[64], edep_frac[64], edep[64],
  // truth_edep[64]
  void append_reco(long long n, unsigned long long int_address,
                   unsigned long long double_address,
                   unsigned long long comp_int_address,
                   unsigned long long comp_double_address) {
    const auto *iv = address<std::int32_t>(int_address);
    const auto *dv = address<double>(double_address);
    const auto *civ = address<std::int32_t>(comp_int_address);
    const auto *cdv = address<double>(comp_double_address);
    for (long long row = 0; row < n; ++row) {
      const auto *i = iv + row * 11;
      const auto *d = dv + row * 13;
      const auto *ci = civ + row * (3 * kMaxComposition);
      const auto *cd = cdv + row * (4 * kMaxComposition);
      reco_event_ = i[0];
      reco_cluster_ = i[1];
      reco_nhits_ = i[2];
      reco_mcid_ = i[3];
      reco_mcpdg_ = i[4];
      reco_mccharge_ = i[5];
      reco_mcstatus_ = i[6];
      reco_ntrack_hits_ = i[7];
      reco_cond_is_track_ = i[8];
      reco_matched_truth_pdgid_ = i[9];
      reco_npdg_comp_ = std::max(0, std::min(i[10], int(kMaxComposition)));
      reco_mcmass_ = d[0];
      reco_mcpx_ = d[1];
      reco_mcpy_ = d[2];
      reco_mcpz_ = d[3];
      reco_mcen_ = d[4];
      reco_edep_reco_ = d[5];
      reco_edep_mc_ = d[6];
      reco_edep_match_ = d[7];
      reco_pred_edep_ = d[8];
      reco_pred_edep_cluster_ = d[9];
      reco_cond_beta_ = d[10];
      reco_matched_truth_hit_frac_ = d[11];
      reco_matched_truth_edep_frac_ = d[12];
      std::memcpy(reco_pdg_comp_ids_, ci,
                  sizeof(std::int32_t) * kMaxComposition);
      std::memcpy(reco_pdg_comp_hits_, ci + kMaxComposition,
                  sizeof(std::int32_t) * kMaxComposition);
      std::memcpy(reco_pdg_comp_track_hits_, ci + 2 * kMaxComposition,
                  sizeof(std::int32_t) * kMaxComposition);
      std::memcpy(reco_pdg_comp_hit_frac_, cd,
                  sizeof(double) * kMaxComposition);
      std::memcpy(reco_pdg_comp_edep_frac_, cd + kMaxComposition,
                  sizeof(double) * kMaxComposition);
      std::memcpy(reco_pdg_comp_edep_, cd + 2 * kMaxComposition,
                  sizeof(double) * kMaxComposition);
      std::memcpy(reco_pdg_comp_truth_edep_, cd + 3 * kMaxComposition,
                  sizeof(double) * kMaxComposition);
      reco_->Fill();
    }
  }

  // int columns:
  // event, hitid, mcid, truthid, cluster, mcpdg, mccharge, mcstatus,
  // pred_alpha, trackness
  // double columns:
  // mcmass, mcpx, mcpy, mcpz, mcen, edep_mc, pred_edep,
  // pred_edep_cluster, pred_beta, and the five particle weights
  void append_prediction(long long n, unsigned long long int_address,
                         unsigned long long double_address) {
    const auto *iv = address<std::int32_t>(int_address);
    const auto *dv = address<double>(double_address);
    for (long long row = 0; row < n; ++row) {
      const auto *i = iv + row * 10;
      const auto *d = dv + row * 14;
      pred_event_ = i[0];
      pred_hitid_ = i[1];
      pred_mcid_ = i[2];
      pred_truthid_ = i[3];
      pred_cluster_ = i[4];
      pred_mcpdg_ = i[5];
      pred_mccharge_ = i[6];
      pred_mcstatus_ = i[7];
      pred_alpha_ = i[8];
      pred_trackness_ = i[9];
      pred_mcmass_ = d[0];
      pred_mcpx_ = d[1];
      pred_mcpy_ = d[2];
      pred_mcpz_ = d[3];
      pred_mcen_ = d[4];
      pred_edep_mc_ = d[5];
      pred_pred_edep_ = d[6];
      pred_pred_edep_cluster_ = d[7];
      pred_beta_ = d[8];
      pred_weight_photon_ = d[9];
      pred_weight_charged_hadron_ = d[10];
      pred_weight_neutral_hadron_ = d[11];
      pred_weight_muon_ = d[12];
      pred_weight_electron_ = d[13];
      prediction_->Fill();
    }
  }

  // int columns: event, ncluster
  // double columns: MC_dijet_energy, total_MC_energy_truth,
  // total_MC_energy_pred, total_predicted_energy_truth,
  // total_predicted_energy_pred
  void append_event(long long n, unsigned long long int_address,
                    unsigned long long double_address) {
    const auto *iv = address<std::int32_t>(int_address);
    const auto *dv = address<double>(double_address);
    for (long long row = 0; row < n; ++row) {
      const auto *i = iv + row * 2;
      const auto *d = dv + row * 5;
      event_event_ = i[0];
      event_ncluster_ = i[1];
      event_MC_dijet_energy_ = d[0];
      event_total_MC_energy_truth_ = d[1];
      event_total_MC_energy_pred_ = d[2];
      event_total_predicted_energy_truth_ = d[3];
      event_total_predicted_energy_pred_ = d[4];
      event_->Fill();
    }
  }

  // int columns: event, n_jets; double columns: jet_p4[2][4]
  void append_jet(long long n, unsigned long long int_address,
                  unsigned long long double_address) {
    const auto *iv = address<std::int32_t>(int_address);
    const auto *dv = address<double>(double_address);
    for (long long row = 0; row < n; ++row) {
      const auto *i = iv + row * 2;
      const auto *d = dv + row * 8;
      jet_event_ = i[0];
      jet_n_jets_ = i[1];
      std::memcpy(jet_p4_, d, sizeof(double) * 8);
      jet_->Fill();
    }
  }

  void close() {
    if (!file_)
      return;
    file_->cd();
    file_->Write();
    file_->Close();
    delete file_;
    file_ = nullptr;
  }

private:
  enum { kMaxComposition = 64 };

  template <typename T> static const T *address(unsigned long long value) {
    return reinterpret_cast<const T *>(static_cast<std::uintptr_t>(value));
  }

  void book_trees() {
    file_->cd();

    t_ = new TTree("t", "tree for MCParticle");
    t_->Branch("event", &t_event_, "event/I");
    t_->Branch("hitid", &t_hitid_, "hitid/I");
    t_->Branch("mcid", &t_mcid_, "mcid/I");
    t_->Branch("truthid", &t_truthid_, "truthid/I");
    t_->Branch("mcpdg", &t_mcpdg_, "mcpdg/I");
    t_->Branch("mccharge", &t_mccharge_, "mccharge/I");
    t_->Branch("mcmass", &t_mcmass_, "mcmass/D");
    t_->Branch("mcpx", &t_mcpx_, "mcpx/D");
    t_->Branch("mcpy", &t_mcpy_, "mcpy/D");
    t_->Branch("mcpz", &t_mcpz_, "mcpz/D");
    t_->Branch("mcen", &t_mcen_, "mcen/D");
    t_->Branch("mcstatus", &t_mcstatus_, "mcstatus/I");
    t_->Branch("edep", &t_edep_, "edep/D");
    t_->Branch("edep_reco", &t_edep_reco_, "edep_reco/D");
    t_->Branch("edep_match", &t_edep_match_, "edep_match/D");
    // Preserve the legacy binding: ncluster and matched_ncluster intentionally
    // read the same value in save_root_reco_w_Cedric.py.
    t_->Branch("ncluster", &t_matched_ncluster_, "ncluster/I");
    t_->Branch("matched_ncluster", &t_matched_ncluster_,
               "matched_ncluster/I");
    t_->Branch("matched_cluster", &t_matched_cluster_, "matched_cluster/I");
    t_->Branch("pred_edep", &t_pred_edep_, "pred_edep/D");
    t_->Branch("pred_edep_cluster", &t_pred_edep_cluster_,
               "pred_edep_cluster/D");
    t_->Branch("pred_edep_weight", &t_pred_edep_weight_,
               "pred_edep_weight/D");
    t_->Branch("cond_beta", &t_cond_beta_, "cond_beta/D");
    t_->Branch("cond_track", &t_cond_track_, "cond_track/I");
    t_->Branch("sed_radius", &t_sed_radius_, "sed_radius/D");
    t_->Branch("pred_photon_energy", &t_pred_photon_energy_,
               "pred_photon_energy/D");
    t_->Branch("pred_charged_hadron_energy", &t_pred_charged_hadron_energy_,
               "pred_charged_hadron_energy/D");
    t_->Branch("pred_neutral_hadron_energy", &t_pred_neutral_hadron_energy_,
               "pred_neutral_hadron_energy/D");
    t_->Branch("pred_muon_energy", &t_pred_muon_energy_,
               "pred_muon_energy/D");
    t_->Branch("pred_electron_energy", &t_pred_electron_energy_,
               "pred_electron_energy/D");

    reco_ = new TTree("reco", "tree for reconstructed clusters");
    reco_->Branch("event", &reco_event_, "event/I");
    reco_->Branch("cluster", &reco_cluster_, "cluster/I");
    reco_->Branch("nhits", &reco_nhits_, "nhits/I");
    reco_->Branch("mcid", &reco_mcid_, "mcid/I");
    reco_->Branch("mcpdg", &reco_mcpdg_, "mcpdg/I");
    reco_->Branch("mccharge", &reco_mccharge_, "mccharge/I");
    reco_->Branch("mcmass", &reco_mcmass_, "mcmass/D");
    reco_->Branch("mcpx", &reco_mcpx_, "mcpx/D");
    reco_->Branch("mcpy", &reco_mcpy_, "mcpy/D");
    reco_->Branch("mcpz", &reco_mcpz_, "mcpz/D");
    reco_->Branch("mcen", &reco_mcen_, "mcen/D");
    reco_->Branch("mcstatus", &reco_mcstatus_, "mcstatus/I");
    reco_->Branch("edep_reco", &reco_edep_reco_, "edep_reco/D");
    reco_->Branch("edep_mc", &reco_edep_mc_, "edep_mc/D");
    reco_->Branch("edep_match", &reco_edep_match_, "edep_match/D");
    reco_->Branch("pred_edep", &reco_pred_edep_, "pred_edep/D");
    reco_->Branch("pred_edep_cluster", &reco_pred_edep_cluster_,
                  "pred_edep_cluster/D");
    reco_->Branch("ntrack_hits", &reco_ntrack_hits_, "ntrack_hits/I");
    reco_->Branch("cond_beta", &reco_cond_beta_, "cond_beta/D");
    reco_->Branch("cond_is_track", &reco_cond_is_track_, "cond_is_track/I");
    reco_->Branch("matched_truth_pdgid", &reco_matched_truth_pdgid_,
                  "matched_truth_pdgid/I");
    reco_->Branch("matched_truth_hit_frac", &reco_matched_truth_hit_frac_,
                  "matched_truth_hit_frac/D");
    reco_->Branch("matched_truth_edep_frac", &reco_matched_truth_edep_frac_,
                  "matched_truth_edep_frac/D");
    reco_->Branch("npdg_comp", &reco_npdg_comp_, "npdg_comp/I");
    reco_->Branch("pdg_comp_ids", reco_pdg_comp_ids_,
                  "pdg_comp_ids[npdg_comp]/I");
    reco_->Branch("pdg_comp_hits", reco_pdg_comp_hits_,
                  "pdg_comp_hits[npdg_comp]/I");
    reco_->Branch("pdg_comp_hit_frac", reco_pdg_comp_hit_frac_,
                  "pdg_comp_hit_frac[npdg_comp]/D");
    reco_->Branch("pdg_comp_edep_frac", reco_pdg_comp_edep_frac_,
                  "pdg_comp_edep_frac[npdg_comp]/D");
    reco_->Branch("pdg_comp_edep", reco_pdg_comp_edep_,
                  "pdg_comp_edep[npdg_comp]/D");
    reco_->Branch("pdg_comp_truth_edep", reco_pdg_comp_truth_edep_,
                  "pdg_comp_truth_edep[npdg_comp]/D");
    reco_->Branch("pdg_comp_track_hits", reco_pdg_comp_track_hits_,
                  "pdg_comp_track_hits[npdg_comp]/I");

    prediction_ = new TTree("prediction", "tree for model output");
    prediction_->Branch("event", &pred_event_, "event/I");
    prediction_->Branch("hitid", &pred_hitid_, "hitid/I");
    prediction_->Branch("mcid", &pred_mcid_, "mcid/I");
    prediction_->Branch("truthid", &pred_truthid_, "truthid/I");
    prediction_->Branch("cluster", &pred_cluster_, "cluster/I");
    prediction_->Branch("mcpdg", &pred_mcpdg_, "mcpdg/I");
    prediction_->Branch("mccharge", &pred_mccharge_, "mccharge/I");
    prediction_->Branch("mcmass", &pred_mcmass_, "mcmass/D");
    prediction_->Branch("mcpx", &pred_mcpx_, "mcpx/D");
    prediction_->Branch("mcpy", &pred_mcpy_, "mcpy/D");
    prediction_->Branch("mcpz", &pred_mcpz_, "mcpz/D");
    prediction_->Branch("mcen", &pred_mcen_, "mcen/D");
    prediction_->Branch("mcstatus", &pred_mcstatus_, "mcstatus/I");
    prediction_->Branch("edep_mc", &pred_edep_mc_, "edep_mc/D");
    prediction_->Branch("pred_edep", &pred_pred_edep_, "pred_edep/D");
    prediction_->Branch("pred_edep_cluster", &pred_pred_edep_cluster_,
                        "pred_edep_cluster/D");
    prediction_->Branch("pred_beta", &pred_beta_, "pred_beta/D");
    prediction_->Branch("pred_alpha", &pred_alpha_, "pred_alpha/I");
    prediction_->Branch("trackness", &pred_trackness_, "trackness/I");
    prediction_->Branch("weight_photon", &pred_weight_photon_,
                        "weight_photon/D");
    prediction_->Branch("weight_charged_hadron", &pred_weight_charged_hadron_,
                        "weight_charged_hadron/D");
    prediction_->Branch("weight_neutral_hadron", &pred_weight_neutral_hadron_,
                        "weight_neutral_hadron/D");
    prediction_->Branch("weight_muon", &pred_weight_muon_, "weight_muon/D");
    prediction_->Branch("weight_electron", &pred_weight_electron_,
                        "weight_electron/D");

    event_ = new TTree("event", "tree for event");
    event_->Branch("event", &event_event_, "event/I");
    event_->Branch("ncluster", &event_ncluster_, "ncluster/I");
    event_->Branch("MC_dijet_energy", &event_MC_dijet_energy_,
                   "MC_dijet_energy/D");
    event_->Branch("total_MC_energy_truth", &event_total_MC_energy_truth_,
                   "total_MC_energy_truth/D");
    event_->Branch("total_MC_energy_pred", &event_total_MC_energy_pred_,
                   "total_MC_energy_pred/D");
    event_->Branch("total_predicted_energy_truth",
                   &event_total_predicted_energy_truth_,
                   "total_predicted_energy_truth/D");
    event_->Branch("total_predicted_energy_pred",
                   &event_total_predicted_energy_pred_,
                   "total_predicted_energy_pred/D");

    jet_ = new TTree("jet", "tree for jets");
    jet_->Branch("event", &jet_event_, "event/I");
    jet_->Branch("n_jets", &jet_n_jets_, "n_jets/I");
    jet_->Branch("jet_p4", jet_p4_, "jet_p4[2][4]/D");
  }

  TFile *file_ = nullptr;
  TTree *t_ = nullptr;
  TTree *reco_ = nullptr;
  TTree *prediction_ = nullptr;
  TTree *event_ = nullptr;
  TTree *jet_ = nullptr;

  std::int32_t t_event_ = 0, t_hitid_ = 0, t_mcid_ = 0, t_truthid_ = 0;
  std::int32_t t_mcpdg_ = 0, t_mccharge_ = 0, t_mcstatus_ = 0;
  std::int32_t t_ncluster_ = 0, t_matched_ncluster_ = 0;
  std::int32_t t_matched_cluster_ = 0, t_cond_track_ = 0;
  double t_mcmass_ = 0, t_mcpx_ = 0, t_mcpy_ = 0, t_mcpz_ = 0;
  double t_mcen_ = 0, t_edep_ = 0, t_edep_reco_ = 0, t_edep_match_ = 0;
  double t_pred_edep_ = 0, t_pred_edep_cluster_ = 0;
  double t_pred_edep_weight_ = 0, t_cond_beta_ = 0, t_sed_radius_ = 0;
  double t_pred_photon_energy_ = 0, t_pred_charged_hadron_energy_ = 0;
  double t_pred_neutral_hadron_energy_ = 0, t_pred_muon_energy_ = 0;
  double t_pred_electron_energy_ = 0;

  std::int32_t reco_event_ = 0, reco_cluster_ = 0, reco_nhits_ = 0;
  std::int32_t reco_mcid_ = 0, reco_mcpdg_ = 0, reco_mccharge_ = 0;
  std::int32_t reco_mcstatus_ = 0, reco_ntrack_hits_ = 0;
  std::int32_t reco_cond_is_track_ = 0, reco_matched_truth_pdgid_ = 0;
  std::int32_t reco_npdg_comp_ = 0;
  double reco_mcmass_ = 0, reco_mcpx_ = 0, reco_mcpy_ = 0;
  double reco_mcpz_ = 0, reco_mcen_ = 0, reco_edep_reco_ = 0;
  double reco_edep_mc_ = 0, reco_edep_match_ = 0, reco_pred_edep_ = 0;
  double reco_pred_edep_cluster_ = 0, reco_cond_beta_ = 0;
  double reco_matched_truth_hit_frac_ = 0;
  double reco_matched_truth_edep_frac_ = 0;
  std::int32_t reco_pdg_comp_ids_[kMaxComposition] = {};
  std::int32_t reco_pdg_comp_hits_[kMaxComposition] = {};
  std::int32_t reco_pdg_comp_track_hits_[kMaxComposition] = {};
  double reco_pdg_comp_hit_frac_[kMaxComposition] = {};
  double reco_pdg_comp_edep_frac_[kMaxComposition] = {};
  double reco_pdg_comp_edep_[kMaxComposition] = {};
  double reco_pdg_comp_truth_edep_[kMaxComposition] = {};

  std::int32_t pred_event_ = 0, pred_hitid_ = 0, pred_mcid_ = 0;
  std::int32_t pred_truthid_ = 0, pred_cluster_ = 0, pred_mcpdg_ = 0;
  std::int32_t pred_mccharge_ = 0, pred_mcstatus_ = 0, pred_alpha_ = 0;
  std::int32_t pred_trackness_ = 0;
  double pred_mcmass_ = 0, pred_mcpx_ = 0, pred_mcpy_ = 0;
  double pred_mcpz_ = 0, pred_mcen_ = 0, pred_edep_mc_ = 0;
  double pred_pred_edep_ = 0, pred_pred_edep_cluster_ = 0;
  double pred_beta_ = 0, pred_weight_photon_ = 0;
  double pred_weight_charged_hadron_ = 0;
  double pred_weight_neutral_hadron_ = 0, pred_weight_muon_ = 0;
  double pred_weight_electron_ = 0;

  std::int32_t event_event_ = 0, event_ncluster_ = 0;
  double event_MC_dijet_energy_ = 0, event_total_MC_energy_truth_ = 0;
  double event_total_MC_energy_pred_ = 0;
  double event_total_predicted_energy_truth_ = 0;
  double event_total_predicted_energy_pred_ = 0;

  std::int32_t jet_event_ = 0, jet_n_jets_ = 0;
  double jet_p4_[2][4] = {};
};

#endif
