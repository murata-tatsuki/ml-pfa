// Lightweight beta/distance scan aggregation.
//
// This macro reads the per-H5 ROOT files directly.  It does not run hadd and
// it does not create plots while extracting the metrics.
//
// Usage (from master/macro):
//   root -l -b -q \
//     'src/make_beta_td_scan_summary.cxx("../output/.../91GeV/multi-head","scan_summary_91GeV.root",91)'
//
// The input directory must contain directories named tbetaXXXtdXXX, each of
// which contains the ROOT files made from the selected H5 files.

#include <TFile.h>
#include <TH1D.h>
#include <TList.h>
#include <TNamed.h>
#include <TParameter.h>
#include <TSystemDirectory.h>
#include <TSystemFile.h>
#include <TTree.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace beta_td_scan_summary {

constexpr int kUnclusteredClusterId = 0;
constexpr int kNCategory = 7;
const char* kCategoryNames[kNCategory] = {
    "inclusive", "electron", "pion", "photon", "neutron", "K0", "muon"
};

struct RunningStat {
    Long64_t n = 0;
    double mean_value = 0.0;
    double m2 = 0.0;

    void add(double value) {
        if (!std::isfinite(value)) return;
        ++n;
        const double delta = value - mean_value;
        mean_value += delta / static_cast<double>(n);
        m2 += delta * (value - mean_value);
    }

    double mean() const { return n > 0 ? mean_value : -1.0; }
    double standardError() const {
        if (n < 2) return -1.0;
        return std::sqrt(m2 / (static_cast<double>(n - 1) * n));
    }
};

// Ratio of sums with an event-clustered standard error.  x and y are first
// summed within an event, because particles/hits in one event are correlated.
struct RatioOfSums {
    double numerator = 0.0;
    double denominator = 0.0;
    std::vector<std::pair<double, double>> event_terms;

    void addEvent(double x, double y) {
        numerator += x;
        denominator += y;
        event_terms.emplace_back(x, y);
    }

    double value() const {
        return denominator > 0.0 ? numerator / denominator : -1.0;
    }

    double standardError() const {
        if (denominator <= 0.0 || event_terms.size() < 2) return -1.0;
        const double ratio = value();
        double residual2 = 0.0;
        for (const auto& term : event_terms) {
            const double residual = term.first - ratio * term.second;
            residual2 += residual * residual;
        }
        const double n = static_cast<double>(event_terms.size());
        return std::sqrt((n / (n - 1.0)) * residual2) / denominator;
    }
};

struct CategoryAccumulator {
    RunningStat legacy_efficiency;
    RunningStat legacy_purity;
    RatioOfSums cedric_efficiency;
    RatioOfSums cedric_purity;
    RatioOfSums cedric_charged_efficiency;
    RatioOfSums cedric_charged_purity;
    RatioOfSums cedric_charged_to_neutral_confusion;
    RatioOfSums cedric_neutral_efficiency;
    RatioOfSums cedric_neutral_purity;
    RatioOfSums cedric_neutral_to_charged_confusion;
    RatioOfSums cedric_coverage;
    RatioOfSums cedric_precision;
    Long64_t cedric_truth_particles = 0;
    Long64_t cedric_matched_particles = 0;
};

struct PointAccumulator {
    int files_found = 0;
    int files_opened = 0;
    int files_with_legacy = 0;
    int files_with_cedric = 0;
    int files_with_event = 0;
    Long64_t events = 0;
    std::array<CategoryAccumulator, kNCategory> category;
    std::vector<double> event_response;
};

struct CedricHitRecord {
    int truthid = 0;
    int cluster = kUnclusteredClusterId;
    int pdg = 0;
    int charge = 0;
    int is_track = 0;
    double mcen = 0.0;
    double edep = 0.0;
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
    std::map<int, double> overlap;
    std::map<int, bool> track_truth_ids;
};

struct BetaHitRecord {
    int truthid = 0;
    int is_track = 0;
    double beta = 0.0;
};

constexpr int kBetaBins = 1000;
struct BetaAccumulator {
    std::array<Long64_t, kBetaBins> all_hits{};
    std::array<Long64_t, kBetaBins> track_reference{};
    std::array<Long64_t, kBetaBins> calo_reference{};
    std::array<Long64_t, kBetaBins> other_valid{};
    Long64_t events = 0;

    static int bin(double beta) {
        if (!std::isfinite(beta)) return -1;
        if (beta <= 0.0) return 0;
        if (beta >= 1.0) return kBetaBins - 1;
        return std::min(kBetaBins - 1, static_cast<int>(beta * kBetaBins));
    }

    static void add(std::array<Long64_t, kBetaBins>& counts, double beta) {
        const int index = bin(beta);
        if (index >= 0) counts[index] += 1;
    }
};

double bounded01(double value) {
    if (!std::isfinite(value) || value < 0.0) return -1.0;
    return std::max(0.0, std::min(1.0, value));
}

double rejection(double confusion) {
    const double value = bounded01(confusion);
    return value >= 0.0 ? 1.0 - value : -1.0;
}

double geometricMean(const std::vector<double>& input) {
    double log_sum = 0.0;
    int valid_count = 0;
    for (double raw : input) {
        const double value = bounded01(raw);
        if (value < 0.0) continue;
        if (value <= 0.0) return 0.0;
        log_sum += std::log(std::max(value, 1e-12));
        valid_count += 1;
    }
    return valid_count > 0 ? std::exp(log_sum / valid_count) : -1.0;
}

double clusteringQuality(double efficiency,
                         double purity,
                         double confusion_quality) {
    const double values[3] = {efficiency, purity, confusion_quality};
    const double weights[3] = {0.40, 0.40, 0.20};
    double log_sum = 0.0;
    double weight_sum = 0.0;
    for (int index = 0; index < 3; ++index) {
        const double value = bounded01(values[index]);
        if (value < 0.0) continue;
        if (value <= 0.0) return 0.0;
        log_sum += weights[index] * std::log(std::max(value, 1e-12));
        weight_sum += weights[index];
    }
    return weight_sum > 0.0 ? std::exp(log_sum / weight_sum) : -1.0;
}

struct Rms90Result {
    double mean90 = -1.0;
    double rms90 = -1.0;
};

bool endsWith(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::string joinPath(const std::string& left, const std::string& right) {
    if (left.empty() || left.back() == '/') return left + right;
    return left + "/" + right;
}

std::vector<std::string> listEntries(const std::string& directory,
                                     bool want_directories) {
    std::vector<std::string> result;
    TSystemDirectory input_directory("scan_input", directory.c_str());
    TList* entries = input_directory.GetListOfFiles();
    if (!entries) return result;

    TIter next(entries);
    while (TSystemFile* entry = static_cast<TSystemFile*>(next())) {
        const std::string name = entry->GetName();
        if (name == "." || name == "..") continue;
        if (entry->IsDirectory() != want_directories) continue;
        result.push_back(name);
    }
    std::sort(result.begin(), result.end());
    return result;
}

int categoryFromPdg(int pdg) {
    const int absolute_pdg = std::abs(pdg);
    if (absolute_pdg == 11) return 1;
    if (absolute_pdg == 211) return 2;
    if (pdg == 22) return 3;
    if (pdg == 2112) return 4;
    if (pdg == 130) return 5;
    if (absolute_pdg == 13) return 6;
    return -1;
}

bool hasBranches(TTree* tree, const std::vector<std::string>& names) {
    if (!tree) return false;
    for (const auto& name : names) {
        if (!tree->GetBranch(name.c_str())) return false;
    }
    return true;
}

int inferEnergy(const std::string& path) {
    std::regex energy_pattern("([0-9]+)GeV");
    std::sregex_iterator begin(path.begin(), path.end(), energy_pattern);
    std::sregex_iterator end;
    int energy = -1;
    for (auto iterator = begin; iterator != end; ++iterator) {
        energy = std::stoi((*iterator)[1].str());
    }
    return energy;
}

Rms90Result calculateRms90(const std::vector<double>& input) {
    Rms90Result result;
    if (input.empty()) return result;

    std::vector<double> values;
    values.reserve(input.size());
    for (double value : input) {
        if (std::isfinite(value)) values.push_back(value);
    }
    if (values.empty()) return result;
    std::sort(values.begin(), values.end());

    const std::size_t window = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(0.9 * values.size())));
    std::size_t best_start = 0;
    double best_width = std::numeric_limits<double>::infinity();
    for (std::size_t start = 0; start + window <= values.size(); ++start) {
        const double width = values[start + window - 1] - values[start];
        if (width < best_width) {
            best_width = width;
            best_start = start;
        }
    }

    double sum = 0.0;
    double sum2 = 0.0;
    for (std::size_t index = best_start; index < best_start + window; ++index) {
        sum += values[index];
        sum2 += values[index] * values[index];
    }
    result.mean90 = sum / static_cast<double>(window);
    const double variance = sum2 / static_cast<double>(window) -
                            result.mean90 * result.mean90;
    result.rms90 = std::sqrt(std::max(0.0, variance));
    return result;
}

void addCedricEvent(const std::vector<CedricHitRecord>& hits,
                    int n_min,
                    double e_min,
                    PointAccumulator& output) {
    std::map<int, CedricTruthParticle> particles;
    std::map<int, CedricRecoCluster> clusters;

    for (const auto& hit : hits) {
        if (hit.truthid > 0) {
            auto& particle = particles[hit.truthid];
            particle.pdg = hit.pdg;
            particle.charge = hit.charge;
            particle.mcen = hit.mcen;
            particle.energy += hit.is_track ? 0.0 : hit.edep;
        }

        if (hit.cluster == kUnclusteredClusterId) continue;
        auto& cluster = clusters[hit.cluster];
        cluster.nhits += 1;
        cluster.energy += hit.edep;
        cluster.has_track = cluster.has_track || (hit.is_track != 0);
        if (hit.truthid > 0) {
            cluster.overlap[hit.truthid] += hit.is_track ? 0.0 : hit.edep;
            if (hit.is_track) cluster.track_truth_ids[hit.truthid] = true;
        }
    }

    std::map<int, const CedricRecoCluster*> kept_clusters;
    for (const auto& item : clusters) {
        if (item.second.nhits >= n_min && item.second.energy >= e_min) {
            kept_clusters[item.first] = &item.second;
        }
    }

    // Charged clusters are all kept clusters containing a track.  A charged
    // C* is owned by the charged truth particle whose track is in the cluster;
    // shared-track conflicts are resolved by the largest calorimeter overlap.
    std::map<int, int> charged_star;
    std::map<int, int> charged_owner;
    std::vector<int> charged_cluster_ids;
    for (const auto& item : kept_clusters) {
        const int cluster_id = item.first;
        const CedricRecoCluster& cluster = *item.second;
        if (!cluster.has_track) continue;
        charged_cluster_ids.push_back(cluster_id);

        int owner = -1;
        double best_overlap = -std::numeric_limits<double>::infinity();
        for (const auto& track_item : cluster.track_truth_ids) {
            const int truth_id = track_item.first;
            const auto particle = particles.find(truth_id);
            if (particle == particles.end() || particle->second.charge == 0) continue;
            const auto overlap = cluster.overlap.find(truth_id);
            const double value = overlap == cluster.overlap.end() ? 0.0 : overlap->second;
            if (value > best_overlap ||
                (value == best_overlap && (owner < 0 || truth_id < owner))) {
                owner = truth_id;
                best_overlap = value;
            }
        }
        if (owner >= 0) {
            charged_owner[cluster_id] = owner;
            charged_star[owner] = cluster_id;
        }
    }

    // Neutral ownership is evaluated only for track-free kept clusters.  All
    // neutral-owned clusters enter charged-to-neutral confusion.  Only the
    // largest-overlap cluster for each neutral particle is its C*; the rest
    // remain duplicate/contested clusters for the precision denominator.
    std::map<int, std::vector<int>> neutral_owned_clusters;
    std::map<int, int> neutral_owner;
    for (const auto& item : kept_clusters) {
        const int cluster_id = item.first;
        const CedricRecoCluster& cluster = *item.second;
        if (cluster.has_track) continue;

        int owner = -1;
        double best_overlap = 0.0;
        for (const auto& overlap_item : cluster.overlap) {
            const int truth_id = overlap_item.first;
            const auto particle = particles.find(truth_id);
            if (particle == particles.end() || particle->second.charge != 0) continue;
            const double value = overlap_item.second;
            if (value > best_overlap ||
                (value == best_overlap && value > 0.0 &&
                 (owner < 0 || truth_id < owner))) {
                owner = truth_id;
                best_overlap = value;
            }
        }
        if (owner >= 0) {
            neutral_owner[cluster_id] = owner;
            neutral_owned_clusters[owner].push_back(cluster_id);
        }
    }

    std::map<int, int> neutral_star;
    for (const auto& item : neutral_owned_clusters) {
        const int truth_id = item.first;
        int best_cluster = -1;
        double best_overlap = -1.0;
        for (int cluster_id : item.second) {
            const CedricRecoCluster* cluster = kept_clusters[cluster_id];
            const double overlap = cluster->overlap.at(truth_id);
            if (overlap > best_overlap) {
                best_cluster = cluster_id;
                best_overlap = overlap;
            }
        }
        neutral_star[truth_id] = best_cluster;
    }

    std::array<double, kNCategory> efficiency_numerator{};
    std::array<double, kNCategory> efficiency_denominator{};
    std::array<double, kNCategory> purity_numerator{};
    std::array<double, kNCategory> purity_denominator{};
    std::array<double, kNCategory> charged_efficiency_numerator{};
    std::array<double, kNCategory> charged_efficiency_denominator{};
    std::array<double, kNCategory> charged_purity_numerator{};
    std::array<double, kNCategory> charged_purity_denominator{};
    std::array<double, kNCategory> charged_confusion_numerator{};
    std::array<double, kNCategory> charged_confusion_denominator{};
    std::array<double, kNCategory> neutral_efficiency_numerator{};
    std::array<double, kNCategory> neutral_efficiency_denominator{};
    std::array<double, kNCategory> neutral_purity_numerator{};
    std::array<double, kNCategory> neutral_purity_denominator{};
    std::array<double, kNCategory> neutral_confusion_numerator{};
    std::array<double, kNCategory> neutral_confusion_denominator{};
    std::array<double, kNCategory> coverage_numerator{};
    std::array<double, kNCategory> coverage_denominator{};
    std::array<double, kNCategory> precision_numerator{};
    std::array<double, kNCategory> precision_denominator{};

    // Inclusive precision counts every kept cluster, including unowned,
    // orphan, and pure-noise clusters.  Per-category precision follows the
    // Cedric group convention and counts clusters attributed to that owner.
    precision_denominator[0] = static_cast<double>(kept_clusters.size());
    for (const auto& item : charged_owner) {
        precision_numerator[0] += 1.0;
        const auto particle = particles.find(item.second);
        if (particle == particles.end()) continue;
        const int category = categoryFromPdg(particle->second.pdg);
        if (category >= 0) {
            precision_numerator[category] += 1.0;
            precision_denominator[category] += 1.0;
        }
    }
    for (const auto& item : neutral_owner) {
        const int cluster_id = item.first;
        const int truth_id = item.second;
        const bool is_match = neutral_star.find(truth_id) != neutral_star.end() &&
                              neutral_star[truth_id] == cluster_id;
        if (is_match) precision_numerator[0] += 1.0;
        const auto particle = particles.find(truth_id);
        if (particle == particles.end()) continue;
        const int category = categoryFromPdg(particle->second.pdg);
        if (category >= 0) {
            precision_denominator[category] += 1.0;
            if (is_match) precision_numerator[category] += 1.0;
        }
    }

    for (const auto& item : particles) {
        const int truth_id = item.first;
        const CedricTruthParticle& particle = item.second;

        const std::map<int, int>& star_map =
            particle.charge != 0 ? charged_star : neutral_star;
        const auto star = star_map.find(truth_id);
        const bool matched = star != star_map.end();
        double matched_energy = 0.0;
        double matched_cluster_energy = 0.0;
        if (matched) {
            const auto cluster = kept_clusters.find(star->second);
            if (cluster != kept_clusters.end()) {
                matched_cluster_energy = cluster->second->energy;
                const auto overlap = cluster->second->overlap.find(truth_id);
                if (overlap != cluster->second->overlap.end()) {
                    matched_energy = overlap->second;
                }
            }
        }

        const int specific_category = categoryFromPdg(particle.pdg);
        const int categories[2] = {0, specific_category};
        const int number_of_categories = specific_category >= 0 ? 2 : 1;

        double confusion_energy = 0.0;
        if (particle.charge != 0) {
            for (const auto& cluster_item : neutral_owner) {
                const CedricRecoCluster* cluster = kept_clusters[cluster_item.first];
                const auto overlap = cluster->overlap.find(truth_id);
                if (overlap != cluster->overlap.end()) confusion_energy += overlap->second;
            }
        } else {
            for (int cluster_id : charged_cluster_ids) {
                const CedricRecoCluster* cluster = kept_clusters[cluster_id];
                const auto overlap = cluster->overlap.find(truth_id);
                if (overlap != cluster->overlap.end()) confusion_energy += overlap->second;
            }
        }

        for (int index = 0; index < number_of_categories; ++index) {
            const int category = categories[index];
            efficiency_numerator[category] += matched_energy;
            efficiency_denominator[category] += particle.energy;
            coverage_denominator[category] += 1.0;
            output.category[category].cedric_truth_particles += 1;
            if (matched) {
                coverage_numerator[category] += 1.0;
                purity_numerator[category] += matched_energy;
                purity_denominator[category] += matched_cluster_energy;
                output.category[category].cedric_matched_particles += 1;
            }

            if (particle.charge != 0) {
                charged_efficiency_numerator[category] += matched_energy;
                charged_efficiency_denominator[category] += particle.energy;
                charged_confusion_numerator[category] += confusion_energy;
                charged_confusion_denominator[category] += particle.energy;
                if (matched) {
                    charged_purity_numerator[category] += matched_energy;
                    charged_purity_denominator[category] += matched_cluster_energy;
                }
            } else {
                neutral_efficiency_numerator[category] += matched_energy;
                neutral_efficiency_denominator[category] += particle.energy;
                neutral_confusion_numerator[category] += confusion_energy;
                neutral_confusion_denominator[category] += particle.energy;
                if (matched) {
                    neutral_purity_numerator[category] += matched_energy;
                    neutral_purity_denominator[category] += matched_cluster_energy;
                }
            }
        }
    }

    for (int category = 0; category < kNCategory; ++category) {
        CategoryAccumulator& destination = output.category[category];
        destination.cedric_efficiency.addEvent(
            efficiency_numerator[category], efficiency_denominator[category]);
        destination.cedric_purity.addEvent(
            purity_numerator[category], purity_denominator[category]);
        destination.cedric_charged_efficiency.addEvent(
            charged_efficiency_numerator[category],
            charged_efficiency_denominator[category]);
        destination.cedric_charged_purity.addEvent(
            charged_purity_numerator[category], charged_purity_denominator[category]);
        destination.cedric_charged_to_neutral_confusion.addEvent(
            charged_confusion_numerator[category], charged_confusion_denominator[category]);
        destination.cedric_neutral_efficiency.addEvent(
            neutral_efficiency_numerator[category], neutral_efficiency_denominator[category]);
        destination.cedric_neutral_purity.addEvent(
            neutral_purity_numerator[category], neutral_purity_denominator[category]);
        destination.cedric_neutral_to_charged_confusion.addEvent(
            neutral_confusion_numerator[category], neutral_confusion_denominator[category]);
        destination.cedric_coverage.addEvent(
            coverage_numerator[category], coverage_denominator[category]);
        destination.cedric_precision.addEvent(
            precision_numerator[category], precision_denominator[category]);
    }
}

void processLegacyTree(TTree* tree, PointAccumulator& output) {
    int pdg = 0;
    double edep = 0.0;
    double edep_reco = 0.0;
    double edep_match = 0.0;
    double cond_beta = 0.0;

    tree->SetBranchStatus("*", 0);
    for (const char* name : {"mcpdg", "edep", "edep_reco", "edep_match"}) {
        tree->SetBranchStatus(name, 1);
    }
    const bool has_cond_beta = tree->GetBranch("cond_beta") != nullptr;
    if (has_cond_beta) tree->SetBranchStatus("cond_beta", 1);
    tree->SetBranchAddress("mcpdg", &pdg);
    tree->SetBranchAddress("edep", &edep);
    tree->SetBranchAddress("edep_reco", &edep_reco);
    tree->SetBranchAddress("edep_match", &edep_match);
    if (has_cond_beta) tree->SetBranchAddress("cond_beta", &cond_beta);

    for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
        tree->GetEntry(entry);
        if (edep <= 0.0 || edep_reco <= 0.0 || edep_match < 0.0) continue;
        if (has_cond_beta && cond_beta < 0.0) continue;
        if (edep <= 1.0) continue;

        const double efficiency = edep_match / edep;
        const double purity = edep_match / edep_reco;
        output.category[0].legacy_efficiency.add(efficiency);
        output.category[0].legacy_purity.add(purity);
        const int category = categoryFromPdg(pdg);
        if (category >= 0) {
            output.category[category].legacy_efficiency.add(efficiency);
            output.category[category].legacy_purity.add(purity);
        }
    }
    tree->ResetBranchAddresses();
}

void processCedricTree(TTree* tree,
                       int n_min,
                       double e_min,
                       PointAccumulator& output) {
    int event = -1;
    int truthid = 0;
    int cluster = kUnclusteredClusterId;
    int pdg = 0;
    int charge = 0;
    int track = 0;
    double mcen = 0.0;
    double edep = 0.0;

    tree->SetBranchStatus("*", 0);
    for (const char* name : {"event", "truthid", "cluster", "mcpdg",
                             "mccharge", "mcen", "edep_mc", "trackness"}) {
        tree->SetBranchStatus(name, 1);
    }
    tree->SetBranchAddress("event", &event);
    tree->SetBranchAddress("truthid", &truthid);
    tree->SetBranchAddress("cluster", &cluster);
    tree->SetBranchAddress("mcpdg", &pdg);
    tree->SetBranchAddress("mccharge", &charge);
    tree->SetBranchAddress("mcen", &mcen);
    tree->SetBranchAddress("edep_mc", &edep);
    tree->SetBranchAddress("trackness", &track);

    std::vector<CedricHitRecord> hits;
    int current_event = -1;
    for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
        tree->GetEntry(entry);
        if (current_event >= 0 && event != current_event) {
            addCedricEvent(hits, n_min, e_min, output);
            hits.clear();
        }
        current_event = event;
        CedricHitRecord hit;
        hit.truthid = truthid;
        hit.cluster = cluster;
        hit.pdg = pdg;
        hit.charge = charge;
        hit.is_track = track;
        hit.mcen = mcen;
        hit.edep = edep;
        hits.push_back(hit);
    }
    if (!hits.empty()) addCedricEvent(hits, n_min, e_min, output);
    tree->ResetBranchAddresses();
}

void addBetaEvent(const std::vector<BetaHitRecord>& hits,
                  BetaAccumulator& output) {
    if (hits.empty()) return;
    output.events += 1;
    std::map<int, std::vector<std::size_t>> truth_hits;
    for (std::size_t index = 0; index < hits.size(); ++index) {
        BetaAccumulator::add(output.all_hits, hits[index].beta);
        if (hits[index].truthid > 0) truth_hits[hits[index].truthid].push_back(index);
    }

    std::vector<bool> is_reference(hits.size(), false);
    for (const auto& item : truth_hits) {
        const std::vector<std::size_t>& indices = item.second;
        const bool has_track = std::any_of(
            indices.begin(), indices.end(), [&](std::size_t index) {
                return hits[index].is_track != 0;
            });
        std::size_t best = hits.size();
        double best_beta = -std::numeric_limits<double>::infinity();
        for (std::size_t index : indices) {
            if (has_track && hits[index].is_track == 0) continue;
            if (best == hits.size() || hits[index].beta > best_beta) {
                best = index;
                best_beta = hits[index].beta;
            }
        }
        if (best == hits.size()) continue;
        is_reference[best] = true;
        if (hits[best].is_track != 0) {
            BetaAccumulator::add(output.track_reference, hits[best].beta);
        } else {
            BetaAccumulator::add(output.calo_reference, hits[best].beta);
        }
    }

    for (std::size_t index = 0; index < hits.size(); ++index) {
        if (hits[index].truthid > 0 && !is_reference[index]) {
            BetaAccumulator::add(output.other_valid, hits[index].beta);
        }
    }
}

void processBetaTree(TTree* tree, BetaAccumulator& output) {
    int event = -1;
    int truthid = 0;
    int track = 0;
    double beta = 0.0;

    tree->SetBranchStatus("*", 0);
    for (const char* name : {"event", "truthid", "trackness", "pred_beta"}) {
        tree->SetBranchStatus(name, 1);
    }
    tree->SetBranchAddress("event", &event);
    tree->SetBranchAddress("truthid", &truthid);
    tree->SetBranchAddress("trackness", &track);
    tree->SetBranchAddress("pred_beta", &beta);

    std::vector<BetaHitRecord> hits;
    int current_event = -1;
    for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
        tree->GetEntry(entry);
        if (current_event >= 0 && event != current_event) {
            addBetaEvent(hits, output);
            hits.clear();
        }
        current_event = event;
        BetaHitRecord hit;
        hit.truthid = truthid;
        hit.is_track = track;
        hit.beta = beta;
        hits.push_back(hit);
    }
    if (!hits.empty()) addBetaEvent(hits, output);
    tree->ResetBranchAddresses();
}

void processEventTree(TTree* tree, PointAccumulator& output) {
    double truth_energy = 0.0;
    double predicted_energy = 0.0;
    tree->SetBranchStatus("*", 0);
    tree->SetBranchStatus("MC_dijet_energy", 1);
    tree->SetBranchStatus("total_predicted_energy_pred", 1);
    tree->SetBranchAddress("MC_dijet_energy", &truth_energy);
    tree->SetBranchAddress("total_predicted_energy_pred", &predicted_energy);
    for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
        tree->GetEntry(entry);
        if (truth_energy <= 0.0 || !std::isfinite(predicted_energy)) continue;
        output.event_response.push_back(predicted_energy / truth_energy);
        output.events += 1;
    }
    tree->ResetBranchAddresses();
}

}  // namespace beta_td_scan_summary

void make_beta_td_scan_summary(const char* scan_directory,
                               const char* output_file = "beta_td_scan_summary.root",
                               int energy_gev = -1,
                               int cedric_n_min = 2,
                               double cedric_e_min = 0.0) {
    using namespace beta_td_scan_summary;

    if (!scan_directory || scan_directory[0] == '\0') {
        std::cerr << "ERROR: scan_directory is required." << std::endl;
        return;
    }
    if (!output_file || output_file[0] == '\0') {
        std::cerr << "ERROR: output_file is required." << std::endl;
        return;
    }
    if (cedric_n_min < 1 || cedric_e_min < 0.0) {
        std::cerr << "ERROR: require cedric_n_min >= 1 and cedric_e_min >= 0."
                  << std::endl;
        return;
    }

    const std::string input_path = scan_directory;
    if (energy_gev < 0) energy_gev = inferEnergy(input_path);

    struct ScanPoint {
        std::string directory_name;
        int beta_code = 0;
        int distance_code = 0;
    };
    std::vector<ScanPoint> points;
    const std::regex point_pattern("^tbeta([0-9]{3})td([0-9]{3})$");
    for (const auto& name : listEntries(input_path, true)) {
        std::smatch match;
        if (!std::regex_match(name, match, point_pattern)) continue;
        ScanPoint point;
        point.directory_name = name;
        point.beta_code = std::stoi(match[1].str());
        point.distance_code = std::stoi(match[2].str());
        points.push_back(point);
    }
    std::sort(points.begin(), points.end(), [](const ScanPoint& left, const ScanPoint& right) {
        if (left.beta_code != right.beta_code) return left.beta_code < right.beta_code;
        return left.distance_code < right.distance_code;
    });
    if (points.empty()) {
        std::cerr << "ERROR: no tbetaXXXtdXXX directories under " << input_path
                  << std::endl;
        return;
    }

    TFile output(output_file, "RECREATE");
    if (output.IsZombie()) {
        std::cerr << "ERROR: cannot create " << output_file << std::endl;
        return;
    }

    TNamed input_directory_metadata("input_scan_directory", input_path.c_str());
    input_directory_metadata.Write();
    TNamed legacy_definition(
        "legacy_definition",
        "Per matched truth particle with edep>1 GeV: efficiency=edep_match/edep, "
        "purity=edep_match/edep_reco; reported value is the unweighted particle mean.");
    legacy_definition.Write();
    TNamed cedric_definition(
        "cedric_definition",
        "Cedric: W(P,C) is non-track edep overlap. Charged C* is track-anchored; "
        "neutral C* uses relative-majority ownership among track-free clusters. "
        "Efficiency=sum W(P,C*)/sum W(P); purity=sum W(P,C*)/sum W(C*). "
        "Qconf=GM(1-confusion_to_neutral,1-confusion_to_charged); "
        "ClusteringQuality=efficiency^0.40*purity^0.40*Qconf^0.20 with "
        "undefined components omitted and remaining weights renormalized.");
    cedric_definition.Write();
    TNamed resolution_definition(
        "resolution_definition",
        "Event response=event/total_predicted_energy_pred divided by event/MC_dijet_energy. "
        "Resolution is RMS90/Mean90 of the narrowest ceil(0.9*N) unbinned response sample.");
    resolution_definition.Write();
    TParameter<int>("energy_gev", energy_gev).Write();
    TParameter<int>("cedric_n_min", cedric_n_min).Write();
    TParameter<double>("cedric_e_min", cedric_e_min).Write();

    int manifest_beta_code = 0;
    int manifest_distance_code = 0;
    int manifest_open_ok = 0;
    int manifest_has_legacy = 0;
    int manifest_has_cedric = 0;
    int manifest_has_beta = 0;
    int manifest_has_event = 0;
    Long64_t manifest_legacy_entries = 0;
    Long64_t manifest_prediction_entries = 0;
    Long64_t manifest_event_entries = 0;
    std::string manifest_path;
    TTree manifest("scan_files", "Input files processed independently (no hadd)");
    manifest.Branch("beta_code", &manifest_beta_code);
    manifest.Branch("distance_code", &manifest_distance_code);
    manifest.Branch("file_path", &manifest_path);
    manifest.Branch("open_ok", &manifest_open_ok);
    manifest.Branch("has_legacy", &manifest_has_legacy);
    manifest.Branch("has_cedric", &manifest_has_cedric);
    manifest.Branch("has_beta", &manifest_has_beta);
    manifest.Branch("has_event", &manifest_has_event);
    manifest.Branch("legacy_entries", &manifest_legacy_entries);
    manifest.Branch("prediction_entries", &manifest_prediction_entries);
    manifest.Branch("event_entries", &manifest_event_entries);

    int row_energy = energy_gev;
    int row_beta_code = 0;
    int row_distance_code = 0;
    double row_beta = 0.0;
    double row_distance = 0.0;
    int row_category_id = 0;
    char row_category[32] = {0};
    int row_files_found = 0;
    int row_files_opened = 0;
    int row_files_legacy = 0;
    int row_files_cedric = 0;
    int row_files_event = 0;
    Long64_t row_events = 0;
    Long64_t row_legacy_n = 0;
    double row_legacy_efficiency = -1.0;
    double row_legacy_efficiency_error = -1.0;
    double row_legacy_purity = -1.0;
    double row_legacy_purity_error = -1.0;
    Long64_t row_cedric_truth_particles = 0;
    Long64_t row_cedric_matched_particles = 0;
    double row_cedric_efficiency_numerator = 0.0;
    double row_cedric_efficiency_denominator = 0.0;
    double row_cedric_efficiency = -1.0;
    double row_cedric_efficiency_error = -1.0;
    double row_cedric_purity_numerator = 0.0;
    double row_cedric_purity_denominator = 0.0;
    double row_cedric_purity = -1.0;
    double row_cedric_purity_error = -1.0;
    double row_cedric_charged_efficiency_numerator = 0.0;
    double row_cedric_charged_efficiency_denominator = 0.0;
    double row_cedric_charged_efficiency = -1.0;
    double row_cedric_charged_purity_numerator = 0.0;
    double row_cedric_charged_purity_denominator = 0.0;
    double row_cedric_charged_purity = -1.0;
    double row_cedric_charged_to_neutral_confusion_numerator = 0.0;
    double row_cedric_charged_to_neutral_confusion_denominator = 0.0;
    double row_cedric_charged_to_neutral_confusion = -1.0;
    double row_cedric_neutral_efficiency_numerator = 0.0;
    double row_cedric_neutral_efficiency_denominator = 0.0;
    double row_cedric_neutral_efficiency = -1.0;
    double row_cedric_neutral_purity_numerator = 0.0;
    double row_cedric_neutral_purity_denominator = 0.0;
    double row_cedric_neutral_purity = -1.0;
    double row_cedric_neutral_to_charged_confusion_numerator = 0.0;
    double row_cedric_neutral_to_charged_confusion_denominator = 0.0;
    double row_cedric_neutral_to_charged_confusion = -1.0;
    double row_cedric_coverage_numerator = 0.0;
    double row_cedric_coverage_denominator = 0.0;
    double row_cedric_coverage = -1.0;
    double row_cedric_precision_numerator = 0.0;
    double row_cedric_precision_denominator = 0.0;
    double row_cedric_precision = -1.0;
    double row_cedric_confusion_quality = -1.0;
    double row_cedric_charged_quality = -1.0;
    double row_cedric_neutral_quality = -1.0;
    double row_cedric_assignment_quality = -1.0;
    double row_cedric_clustering_quality = -1.0;
    double row_cedric_spurious_cluster_rate = -1.0;
    Long64_t row_response_n = 0;
    double row_response_mean = -1.0;
    double row_response_mean_error = -1.0;
    double row_mean90 = -1.0;
    double row_rms90 = -1.0;
    double row_resolution = -1.0;

    TTree summary("scan_summary", "One row per beta/distance/category");
    summary.Branch("energy_gev", &row_energy, "energy_gev/I");
    summary.Branch("beta_code", &row_beta_code, "beta_code/I");
    summary.Branch("distance_code", &row_distance_code, "distance_code/I");
    summary.Branch("beta", &row_beta, "beta/D");
    summary.Branch("distance", &row_distance, "distance/D");
    summary.Branch("category_id", &row_category_id, "category_id/I");
    summary.Branch("category", row_category, "category/C");
    summary.Branch("files_found", &row_files_found, "files_found/I");
    summary.Branch("files_opened", &row_files_opened, "files_opened/I");
    summary.Branch("files_legacy", &row_files_legacy, "files_legacy/I");
    summary.Branch("files_cedric", &row_files_cedric, "files_cedric/I");
    summary.Branch("files_event", &row_files_event, "files_event/I");
    summary.Branch("events", &row_events, "events/L");
    summary.Branch("legacy_n", &row_legacy_n, "legacy_n/L");
    summary.Branch("legacy_efficiency", &row_legacy_efficiency, "legacy_efficiency/D");
    summary.Branch("legacy_efficiency_error", &row_legacy_efficiency_error,
                   "legacy_efficiency_error/D");
    summary.Branch("legacy_purity", &row_legacy_purity, "legacy_purity/D");
    summary.Branch("legacy_purity_error", &row_legacy_purity_error,
                   "legacy_purity_error/D");
    summary.Branch("cedric_truth_particles", &row_cedric_truth_particles,
                   "cedric_truth_particles/L");
    summary.Branch("cedric_matched_particles", &row_cedric_matched_particles,
                   "cedric_matched_particles/L");
    summary.Branch("cedric_efficiency_numerator", &row_cedric_efficiency_numerator,
                   "cedric_efficiency_numerator/D");
    summary.Branch("cedric_efficiency_denominator", &row_cedric_efficiency_denominator,
                   "cedric_efficiency_denominator/D");
    summary.Branch("cedric_efficiency", &row_cedric_efficiency, "cedric_efficiency/D");
    summary.Branch("cedric_efficiency_error", &row_cedric_efficiency_error,
                   "cedric_efficiency_error/D");
    summary.Branch("cedric_purity_numerator", &row_cedric_purity_numerator,
                   "cedric_purity_numerator/D");
    summary.Branch("cedric_purity_denominator", &row_cedric_purity_denominator,
                   "cedric_purity_denominator/D");
    summary.Branch("cedric_purity", &row_cedric_purity, "cedric_purity/D");
    summary.Branch("cedric_purity_error", &row_cedric_purity_error,
                   "cedric_purity_error/D");
    summary.Branch("cedric_charged_efficiency_numerator",
                   &row_cedric_charged_efficiency_numerator,
                   "cedric_charged_efficiency_numerator/D");
    summary.Branch("cedric_charged_efficiency_denominator",
                   &row_cedric_charged_efficiency_denominator,
                   "cedric_charged_efficiency_denominator/D");
    summary.Branch("cedric_charged_efficiency", &row_cedric_charged_efficiency,
                   "cedric_charged_efficiency/D");
    summary.Branch("cedric_charged_purity_numerator",
                   &row_cedric_charged_purity_numerator,
                   "cedric_charged_purity_numerator/D");
    summary.Branch("cedric_charged_purity_denominator",
                   &row_cedric_charged_purity_denominator,
                   "cedric_charged_purity_denominator/D");
    summary.Branch("cedric_charged_purity", &row_cedric_charged_purity,
                   "cedric_charged_purity/D");
    summary.Branch("cedric_charged_to_neutral_confusion_numerator",
                   &row_cedric_charged_to_neutral_confusion_numerator,
                   "cedric_charged_to_neutral_confusion_numerator/D");
    summary.Branch("cedric_charged_to_neutral_confusion_denominator",
                   &row_cedric_charged_to_neutral_confusion_denominator,
                   "cedric_charged_to_neutral_confusion_denominator/D");
    summary.Branch("cedric_charged_to_neutral_confusion",
                   &row_cedric_charged_to_neutral_confusion,
                   "cedric_charged_to_neutral_confusion/D");
    summary.Branch("cedric_neutral_efficiency_numerator",
                   &row_cedric_neutral_efficiency_numerator,
                   "cedric_neutral_efficiency_numerator/D");
    summary.Branch("cedric_neutral_efficiency_denominator",
                   &row_cedric_neutral_efficiency_denominator,
                   "cedric_neutral_efficiency_denominator/D");
    summary.Branch("cedric_neutral_efficiency", &row_cedric_neutral_efficiency,
                   "cedric_neutral_efficiency/D");
    summary.Branch("cedric_neutral_purity_numerator",
                   &row_cedric_neutral_purity_numerator,
                   "cedric_neutral_purity_numerator/D");
    summary.Branch("cedric_neutral_purity_denominator",
                   &row_cedric_neutral_purity_denominator,
                   "cedric_neutral_purity_denominator/D");
    summary.Branch("cedric_neutral_purity", &row_cedric_neutral_purity,
                   "cedric_neutral_purity/D");
    summary.Branch("cedric_neutral_to_charged_confusion_numerator",
                   &row_cedric_neutral_to_charged_confusion_numerator,
                   "cedric_neutral_to_charged_confusion_numerator/D");
    summary.Branch("cedric_neutral_to_charged_confusion_denominator",
                   &row_cedric_neutral_to_charged_confusion_denominator,
                   "cedric_neutral_to_charged_confusion_denominator/D");
    summary.Branch("cedric_neutral_to_charged_confusion",
                   &row_cedric_neutral_to_charged_confusion,
                   "cedric_neutral_to_charged_confusion/D");
    summary.Branch("cedric_coverage_numerator", &row_cedric_coverage_numerator,
                   "cedric_coverage_numerator/D");
    summary.Branch("cedric_coverage_denominator", &row_cedric_coverage_denominator,
                   "cedric_coverage_denominator/D");
    summary.Branch("cedric_coverage", &row_cedric_coverage, "cedric_coverage/D");
    summary.Branch("cedric_precision_numerator", &row_cedric_precision_numerator,
                   "cedric_precision_numerator/D");
    summary.Branch("cedric_precision_denominator", &row_cedric_precision_denominator,
                   "cedric_precision_denominator/D");
    summary.Branch("cedric_precision", &row_cedric_precision, "cedric_precision/D");
    summary.Branch("cedric_confusion_quality", &row_cedric_confusion_quality,
                   "cedric_confusion_quality/D");
    summary.Branch("cedric_charged_quality", &row_cedric_charged_quality,
                   "cedric_charged_quality/D");
    summary.Branch("cedric_neutral_quality", &row_cedric_neutral_quality,
                   "cedric_neutral_quality/D");
    summary.Branch("cedric_assignment_quality", &row_cedric_assignment_quality,
                   "cedric_assignment_quality/D");
    summary.Branch("cedric_clustering_quality", &row_cedric_clustering_quality,
                   "cedric_clustering_quality/D");
    summary.Branch("cedric_spurious_cluster_rate", &row_cedric_spurious_cluster_rate,
                   "cedric_spurious_cluster_rate/D");
    summary.Branch("response_n", &row_response_n, "response_n/L");
    summary.Branch("response_mean", &row_response_mean, "response_mean/D");
    summary.Branch("response_mean_error", &row_response_mean_error,
                   "response_mean_error/D");
    summary.Branch("mean90", &row_mean90, "mean90/D");
    summary.Branch("rms90", &row_rms90, "rms90/D");
    summary.Branch("resolution", &row_resolution, "resolution/D");

    const std::vector<std::string> legacy_branches = {
        "mcpdg", "edep", "edep_reco", "edep_match"
    };
    const std::vector<std::string> cedric_branches = {
        "event", "truthid", "cluster", "mcpdg", "mccharge",
        "mcen", "edep_mc", "trackness"
    };
    const std::vector<std::string> beta_branches = {
        "event", "truthid", "trackness", "pred_beta"
    };
    const std::vector<std::string> event_branches = {
        "MC_dijet_energy", "total_predicted_energy_pred"
    };

    BetaAccumulator beta_accumulator;
    std::set<std::string> beta_source_files;

    std::cout << "Found " << points.size() << " scan-point directories." << std::endl;
    for (std::size_t point_index = 0; point_index < points.size(); ++point_index) {
        const ScanPoint& point = points[point_index];
        const std::string point_path = joinPath(input_path, point.directory_name);
        PointAccumulator accumulator;

        std::vector<std::string> root_files;
        for (const auto& name : listEntries(point_path, false)) {
            if (endsWith(name, ".root")) root_files.push_back(name);
        }
        accumulator.files_found = static_cast<int>(root_files.size());
        std::cout << "[" << point_index + 1 << "/" << points.size() << "] "
                  << point.directory_name << ": " << root_files.size()
                  << " ROOT files" << std::endl;

        for (const auto& root_name : root_files) {
            manifest_beta_code = point.beta_code;
            manifest_distance_code = point.distance_code;
            manifest_path = joinPath(point_path, root_name);
            manifest_open_ok = 0;
            manifest_has_legacy = 0;
            manifest_has_cedric = 0;
            manifest_has_beta = 0;
            manifest_has_event = 0;
            manifest_legacy_entries = 0;
            manifest_prediction_entries = 0;
            manifest_event_entries = 0;

            TFile input(manifest_path.c_str(), "READ");
            if (input.IsZombie()) {
                std::cerr << "  WARNING: cannot open " << manifest_path << std::endl;
                manifest.Fill();
                continue;
            }
            manifest_open_ok = 1;
            accumulator.files_opened += 1;

            TTree* legacy_tree = static_cast<TTree*>(input.Get("t"));
            TTree* prediction_tree = static_cast<TTree*>(input.Get("prediction"));
            TTree* event_tree = static_cast<TTree*>(input.Get("event"));

            manifest_has_legacy = hasBranches(legacy_tree, legacy_branches);
            manifest_has_cedric = hasBranches(prediction_tree, cedric_branches) &&
                                  prediction_tree->GetEntries() > 0;
            manifest_has_beta = hasBranches(prediction_tree, beta_branches) &&
                                prediction_tree->GetEntries() > 0;
            manifest_has_event = hasBranches(event_tree, event_branches);
            if (legacy_tree) manifest_legacy_entries = legacy_tree->GetEntries();
            if (prediction_tree) manifest_prediction_entries = prediction_tree->GetEntries();
            if (event_tree) manifest_event_entries = event_tree->GetEntries();

            if (manifest_has_legacy) {
                processLegacyTree(legacy_tree, accumulator);
                accumulator.files_with_legacy += 1;
            }
            if (manifest_has_cedric) {
                processCedricTree(prediction_tree, cedric_n_min, cedric_e_min, accumulator);
                accumulator.files_with_cedric += 1;
            }
            // Beta is identical at every threshold point.  Accumulate each
            // per-H5 ROOT basename once rather than counting its 81 copies.
            if (manifest_has_beta && beta_source_files.insert(root_name).second) {
                processBetaTree(prediction_tree, beta_accumulator);
            }
            if (manifest_has_event) {
                processEventTree(event_tree, accumulator);
                accumulator.files_with_event += 1;
            }
            manifest.Fill();
            input.Close();
        }

        RunningStat response_stat;
        for (double value : accumulator.event_response) response_stat.add(value);
        const Rms90Result rms90 = calculateRms90(accumulator.event_response);

        row_beta_code = point.beta_code;
        row_distance_code = point.distance_code;
        row_beta = point.beta_code / 100.0;
        row_distance = point.distance_code / 100.0;
        row_files_found = accumulator.files_found;
        row_files_opened = accumulator.files_opened;
        row_files_legacy = accumulator.files_with_legacy;
        row_files_cedric = accumulator.files_with_cedric;
        row_files_event = accumulator.files_with_event;
        row_events = accumulator.events;
        row_response_n = response_stat.n;
        row_response_mean = response_stat.mean();
        row_response_mean_error = response_stat.standardError();
        row_mean90 = rms90.mean90;
        row_rms90 = rms90.rms90;
        row_resolution = rms90.mean90 > 0.0 ? rms90.rms90 / rms90.mean90 : -1.0;

        for (int category = 0; category < kNCategory; ++category) {
            const CategoryAccumulator& source = accumulator.category[category];
            row_category_id = category;
            std::snprintf(row_category, sizeof(row_category), "%s", kCategoryNames[category]);
            row_legacy_n = source.legacy_efficiency.n;
            row_legacy_efficiency = source.legacy_efficiency.mean();
            row_legacy_efficiency_error = source.legacy_efficiency.standardError();
            row_legacy_purity = source.legacy_purity.mean();
            row_legacy_purity_error = source.legacy_purity.standardError();
            row_cedric_truth_particles = source.cedric_truth_particles;
            row_cedric_matched_particles = source.cedric_matched_particles;
            row_cedric_efficiency_numerator = source.cedric_efficiency.numerator;
            row_cedric_efficiency_denominator = source.cedric_efficiency.denominator;
            row_cedric_efficiency = source.cedric_efficiency.value();
            row_cedric_efficiency_error = source.cedric_efficiency.standardError();
            row_cedric_purity_numerator = source.cedric_purity.numerator;
            row_cedric_purity_denominator = source.cedric_purity.denominator;
            row_cedric_purity = source.cedric_purity.value();
            row_cedric_purity_error = source.cedric_purity.standardError();

            row_cedric_charged_efficiency_numerator =
                source.cedric_charged_efficiency.numerator;
            row_cedric_charged_efficiency_denominator =
                source.cedric_charged_efficiency.denominator;
            row_cedric_charged_efficiency = source.cedric_charged_efficiency.value();
            row_cedric_charged_purity_numerator =
                source.cedric_charged_purity.numerator;
            row_cedric_charged_purity_denominator =
                source.cedric_charged_purity.denominator;
            row_cedric_charged_purity = source.cedric_charged_purity.value();
            row_cedric_charged_to_neutral_confusion_numerator =
                source.cedric_charged_to_neutral_confusion.numerator;
            row_cedric_charged_to_neutral_confusion_denominator =
                source.cedric_charged_to_neutral_confusion.denominator;
            row_cedric_charged_to_neutral_confusion =
                source.cedric_charged_to_neutral_confusion.value();

            row_cedric_neutral_efficiency_numerator =
                source.cedric_neutral_efficiency.numerator;
            row_cedric_neutral_efficiency_denominator =
                source.cedric_neutral_efficiency.denominator;
            row_cedric_neutral_efficiency = source.cedric_neutral_efficiency.value();
            row_cedric_neutral_purity_numerator =
                source.cedric_neutral_purity.numerator;
            row_cedric_neutral_purity_denominator =
                source.cedric_neutral_purity.denominator;
            row_cedric_neutral_purity = source.cedric_neutral_purity.value();
            row_cedric_neutral_to_charged_confusion_numerator =
                source.cedric_neutral_to_charged_confusion.numerator;
            row_cedric_neutral_to_charged_confusion_denominator =
                source.cedric_neutral_to_charged_confusion.denominator;
            row_cedric_neutral_to_charged_confusion =
                source.cedric_neutral_to_charged_confusion.value();

            row_cedric_coverage_numerator = source.cedric_coverage.numerator;
            row_cedric_coverage_denominator = source.cedric_coverage.denominator;
            row_cedric_coverage = source.cedric_coverage.value();
            row_cedric_precision_numerator = source.cedric_precision.numerator;
            row_cedric_precision_denominator = source.cedric_precision.denominator;
            row_cedric_precision = source.cedric_precision.value();

            const double charged_rejection = rejection(
                row_cedric_charged_to_neutral_confusion);
            const double neutral_rejection = rejection(
                row_cedric_neutral_to_charged_confusion);
            row_cedric_confusion_quality = geometricMean(
                {charged_rejection, neutral_rejection});
            row_cedric_charged_quality = geometricMean(
                {row_cedric_charged_efficiency, row_cedric_charged_purity,
                 charged_rejection});
            row_cedric_neutral_quality = geometricMean(
                {row_cedric_neutral_efficiency, row_cedric_neutral_purity,
                 neutral_rejection});
            row_cedric_assignment_quality = geometricMean(
                {row_cedric_coverage, row_cedric_precision});
            row_cedric_clustering_quality = clusteringQuality(
                row_cedric_efficiency, row_cedric_purity,
                row_cedric_confusion_quality);
            row_cedric_spurious_cluster_rate = row_cedric_precision >= 0.0
                ? 1.0 - bounded01(row_cedric_precision)
                : -1.0;
            summary.Fill();
        }

        if (accumulator.files_with_cedric != accumulator.files_opened) {
            std::cout << "  NOTE: Cedric inputs available in "
                      << accumulator.files_with_cedric << "/"
                      << accumulator.files_opened << " opened files." << std::endl;
        }
    }

    output.cd();
    TH1D beta_all_hits(
        "cedric_beta_all_hits", "All hits;GNN beta score;hits / 0.001", kBetaBins, 0.0, 1.0);
    TH1D beta_track_reference(
        "cedric_beta_track_reference",
        "Truth representatives with track;GNN beta score;hits / 0.001",
        kBetaBins, 0.0, 1.0);
    TH1D beta_calo_reference(
        "cedric_beta_calo_reference",
        "Calorimeter truth representatives;GNN beta score;hits / 0.001",
        kBetaBins, 0.0, 1.0);
    TH1D beta_other_valid(
        "cedric_beta_other_valid", "Other valid hits;GNN beta score;hits / 0.001",
        kBetaBins, 0.0, 1.0);
    for (int bin = 1; bin <= kBetaBins; ++bin) {
        beta_all_hits.SetBinContent(bin, beta_accumulator.all_hits[bin - 1]);
        beta_track_reference.SetBinContent(bin, beta_accumulator.track_reference[bin - 1]);
        beta_calo_reference.SetBinContent(bin, beta_accumulator.calo_reference[bin - 1]);
        beta_other_valid.SetBinContent(bin, beta_accumulator.other_valid[bin - 1]);
    }
    beta_all_hits.Write();
    beta_track_reference.Write();
    beta_calo_reference.Write();
    beta_other_valid.Write();
    TParameter<Long64_t>("cedric_beta_events", beta_accumulator.events).Write();
    summary.Write();
    manifest.Write();
    output.Close();
    std::cout << "Wrote " << output_file << " (no hadd was performed)." << std::endl;
}
