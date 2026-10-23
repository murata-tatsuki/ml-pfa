#ifndef PID_SINGLE_PASS_ENERGY_H
#define PID_SINGLE_PASS_ENERGY_H

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <cmath>
#include <cctype>
#include <unistd.h>
#include <nlohmann/json.hpp>

namespace pid_singlepass {

// Bind the existing local histogram pointers to independent, detached banks.
// Each bank is filled and drawn once; the existing binning/fits stay unchanged.
class HistogramBanks {
    struct Slot {
        std::unique_ptr<TH1> histograms[2];
        std::function<void(TH1*)> bind;
    };
    std::vector<Slot> slots_;
    std::unordered_map<TH1*, std::array<TH1*, 2>> alternatives_;
public:
    template<class Histogram> void add(Histogram*& histogram) {
        Histogram** address = &histogram;
        histogram->SetDirectory(nullptr);
        Slot slot;
        slot.histograms[0].reset(histogram);
        slot.histograms[1].reset(static_cast<TH1*>(histogram->Clone()));
        slot.histograms[1]->SetDirectory(nullptr);
        const std::array<TH1*, 2> alternatives = {{slot.histograms[0].get(), slot.histograms[1].get()}};
        for (auto* value : alternatives) alternatives_.emplace(value, alternatives);
        slot.bind = [address](TH1* value) { *address = static_cast<Histogram*>(value); };
        slots_.push_back(std::move(slot));
    }
    void activate(int mode) {
        for (auto& slot : slots_) slot.bind(slot.histograms[mode].get());
    }
    template<class Histogram> Histogram* at(Histogram* histogram, int mode) const {
        if (mode < 0) return histogram;
        const auto found = alternatives_.find(histogram);
        if (found == alternatives_.end()) throw std::runtime_error("Unregistered paired energy fill");
        return static_cast<Histogram*>(found->second[mode]);
    }
    void write(TDirectory& destination, int mode) {
        TDirectory::TContext context(&destination);
        for (auto& slot : slots_)
            slot.histograms[mode]->Write(slot.histograms[mode]->GetName(), TObject::kOverwrite);
    }
};

inline bool measuredPolicy(const std::string& text) {
    const auto metadata = nlohmann::json::parse(text);
    const auto found = metadata.find("single_track_energy");
    if (found == metadata.end() || !found->is_object()) return false;
    const auto description_value = found->find("momentum");
    if (description_value == found->end() || !description_value->is_string()) return false;
    std::string description = description_value->get<std::string>();
    for (char& c : description) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (const char* word : {"measured", "calorimeter", "7:10"})
        if (description.find(word) == std::string::npos) return false;
    const auto masses = found->find("masses_GeV");
    if (masses == found->end() || !masses->is_object())
        throw std::runtime_error("Stored measured-track policy has no mass hypotheses");
    const char* species[] = {"charged_hadron", "electron", "muon"};
    const double expected[] = {0.13957039, 0.00051099895, 0.1056583755};
    for (int head=0; head<3; ++head) {
        const auto mass = masses->find(species[head]);
        if (mass == masses->end() || !mass->is_number() ||
            !std::isfinite(mass->get<double>()) || std::abs(mass->get<double>() - expected[head]) > 1e-12)
            throw std::runtime_error("Stored single-track mass hypotheses do not match pion/electron/muon");
    }
    return true;
}

class FileReadouts {
public:
    struct Energy {
        double saved, raw, track;
        int head;
        Long64_t seed;
        bool recover;
    };
    using Key = std::pair<int, int>;
private:
    std::map<Key, Energy> clusters_;
    std::set<int> recovery_events_;
    std::unique_ptr<TFile> bridge_;
    TTree* predictions_ = nullptr;
    int event_ = 0, cluster_ = 0, hitid_ = 0, trackness_ = 0;
    Long64_t input_row_ = -1;
    std::string metadata_, unified_metadata_, bridge_path_, recovery_path_;
    bool measured_;
public:
    Long64_t eligible = 0, recovered = 0;
    FileReadouts(const std::string& metadata, const std::string& unified_metadata,
                 const std::string& staging, int file_index)
        : metadata_(metadata), unified_metadata_(unified_metadata),
          bridge_path_(staging + "/measured_bridge_" + std::to_string(file_index) + ".root"),
          recovery_path_(staging + "/measured_recovery_" + std::to_string(file_index) + ".root"),
          measured_(measuredPolicy(metadata)) {}

    void add(int event, int cluster, double saved, int pid, int head, int tracks,
             bool accepted, Long64_t seed, const double* heads, bool valid_event) {
        if (cluster <= 0 || !std::isfinite(saved) || head < -1 || head > 4 || tracks < 0)
            throw std::runtime_error("Invalid saved PID energy/head/track count in single-pass input");
        const bool candidate = accepted && head >= 0 && head < 3 && tracks == 1;
        if (candidate && pid != head)
            throw std::runtime_error("Charged energy_head and predicted PID disagree");
        const bool replace = valid_event && candidate;
        const double raw = replace ? heads[head] : saved;
        if (!std::isfinite(raw))
            throw std::runtime_error("Invalid single-track charged raw energy/PID");
        if (replace && !measured_ && std::abs(saved - raw) > 1e-9 + 1e-12 * std::abs(raw))
            throw std::runtime_error("Adopted energy differs from raw head without an explicit measured-track policy");
        if (!clusters_.emplace(Key(event, cluster), Energy{saved, raw, saved, head, seed,
                                                         replace && !measured_}).second)
            throw std::runtime_error("Duplicate reco event/cluster identity");
        if (replace) ++eligible;
        if (replace && !measured_) recovery_events_.insert(event);
    }
    bool needsRecovery() const { return !recovery_events_.empty(); }

    void prepareBridge(const std::map<int, double>& truth,
                       const std::unordered_map<int, double>& angles) {
        if (!needsRecovery()) return;
        TDirectory::TContext context;
        bridge_.reset(TFile::Open(bridge_path_.c_str(), "RECREATE"));
        if (!bridge_ || bridge_->IsZombie()) throw std::runtime_error("Cannot create measured-energy bridge");
        TObjString(metadata_.c_str()).Write("particle_eval_metadata");
        if (!unified_metadata_.empty()) TObjString(unified_metadata_.c_str()).Write("save_root_metadata");
        TTree reco("reco", "Single-track charged clusters requiring measured momentum");
        int event = 0, cluster = 0, head = -1;
        Long64_t seed = -1;
        reco.Branch("event", &event, "event/I");
        reco.Branch("cluster", &cluster, "cluster/I");
        reco.Branch("energy_head", &head, "energy_head/I");
        reco.Branch("pid_seed_input_row", &seed, "pid_seed_input_row/L");
        for (const auto& row : clusters_) if (row.second.recover) {
            event = row.first.first; cluster = row.first.second;
            head = row.second.head; seed = row.second.seed;
            reco.Fill();
        }
        reco.Write();
        TTree events("event", "Validated source event truth and upstream thrust");
        double energy = 0., thrust = 0.;
        events.Branch("event", &event, "event/I");
        events.Branch("MC_dijet_energy", &energy, "MC_dijet_energy/D");
        events.Branch("thrust", &thrust, "thrust/D");
        for (int target : recovery_events_) {
            event = target; energy = truth.at(event);
            auto angle = angles.find(event);
            thrust = angle == angles.end() ? std::numeric_limits<double>::quiet_NaN() : angle->second;
            events.Fill();
        }
        events.Write();
        predictions_ = new TTree("prediction", "Frozen inputs copied during the common hit traversal");
        predictions_->Branch("event", &event_, "event/I");
        predictions_->Branch("cluster", &cluster_, "cluster/I");
        predictions_->Branch("input_row", &input_row_, "input_row/L");
        predictions_->Branch("hitid", &hitid_, "hitid/I");
        predictions_->Branch("trackness", &trackness_, "trackness/I");
    }
    void prediction(int event, int cluster, Long64_t row, int hitid, int track) {
        if (!predictions_ || !recovery_events_.count(event)) return;
        event_ = event; cluster_ = cluster; input_row_ = row; hitid_ = hitid; trackness_ = track;
        predictions_->Fill();
    }
    void recover(const std::function<void(const std::string&, const std::string&)>& run) {
        if (!needsRecovery()) return;
        {
            TDirectory::TContext context(bridge_.get());
            predictions_->Write();
            bridge_->Close();
            bridge_.reset(); predictions_ = nullptr;
        }
        run(bridge_path_, recovery_path_);
        TDirectory::TContext context;
        std::unique_ptr<TFile> result(TFile::Open(recovery_path_.c_str(), "READ"));
        auto* rows = result && !result->IsZombie() ? dynamic_cast<TTree*>(result->Get("track_replacements")) : nullptr;
        int event = -1, cluster = -1;
        double energy = 0.;
        if (!rows || rows->SetBranchAddress("event", &event) < 0 ||
            rows->SetBranchAddress("cluster", &cluster) < 0 ||
            rows->SetBranchAddress("track_replaced", &energy) < 0)
            throw std::runtime_error("Invalid measured-energy recovery output");
        for (Long64_t entry=0; entry<rows->GetEntries(); ++entry) {
            rows->GetEntry(entry);
            auto found = clusters_.find(Key(event, cluster));
            if (found == clusters_.end() || !found->second.recover || !std::isfinite(energy))
                throw std::runtime_error("Unexpected, duplicate or non-finite measured-energy recovery");
            found->second.track = energy;
            found->second.recover = false;
            ++recovered;
        }
        for (const auto& row : clusters_) if (row.second.recover)
            throw std::runtime_error("Missing measured-energy recovery for a single-track charged cluster");
        result->Close();
        result.reset();
        ::unlink(bridge_path_.c_str());
        ::unlink(recovery_path_.c_str());
    }
    double select(int event, int cluster, double saved, int mode) const {
        if (cluster <= 0) {
            if (saved != 0.) throw std::runtime_error("Unmatched truth row has nonzero saved PID energy");
            return saved;
        }
        const auto found = clusters_.find(Key(event, cluster));
        if (found == clusters_.end() || found->second.saved != saved)
            throw std::runtime_error("Truth/reco saved PID energy does not match the single-pass readout map");
        return mode == 0 ? found->second.raw : found->second.track;
    }
};

} // namespace pid_singlepass
#endif
