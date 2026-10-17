#ifndef PID_CLUSTERING_SUMMARY_H
#define PID_CLUSTERING_SUMMARY_H

// Small, independent tables for the multi-energy plotter. Preserve the actual
// unbinned histogram statistics rather than reconstructing means from bins.
#include "pid_jet_resolution_summary.h"
#include "TH1.h"

namespace pid_cluster {
static const char* const kSchema = "pid-clustering-summary-1";
static const char* const kSummaryName = "clustering_summary.root";
static const char* const kDefinitions[2] = {"original", "cedric"};
static const char* const kParticles[6] = {"electron", "pion", "photon", "neutron", "K0", "muon"};
static const char* const kOriginalDefinition =
    "Unweighted mean of matched t rows: edep>1 GeV, edep_reco>0, edep_match>=0, cond_beta>=beta_threshold; efficiency=edep_match/edep; purity=edep_match/edep_reco; unmatched truth excluded";
static const char* const kCedricDefinition =
    "Exclusive charged/neutral ownership, n_inputs>=2, cluster edep>=0; efficiency=sum W(P,C*)/sum W(P), unmatched truth included; purity=sum W(P,C*)/sum W(C*) for matched truth; calorimeter deposited-energy weights";

struct Metric {
    double numerator = 0., denominator = 0.;
    Long64_t entries = 0;
    double value() const { return denominator > 0. ? numerator/denominator : std::numeric_limits<double>::quiet_NaN(); }
    bool valid() const { return denominator > 0. && std::isfinite(numerator) && std::isfinite(denominator); }
};
struct Row {
    int definition = 0, particle = 0;
    Metric efficiency, purity;
};
inline Metric histogramMetric(const TH1& histogram) {
    double stats[4] = {};
    histogram.GetStats(stats);
    Metric metric;
    metric.numerator = stats[2]; metric.denominator = stats[0];
    metric.entries = histogram.GetEntries();
    return metric;
}
inline void bind(TTree& tree, Row& row, bool writing) {
    auto field = [&](const char* name, void* address, const char* type) {
        if (writing) tree.Branch(name, address, (std::string(name)+"/"+type).c_str());
        else if (!tree.GetBranch(name) || tree.SetBranchAddress(name, address) < 0)
            throw std::runtime_error(std::string("Missing clustering summary branch: ")+name);
    };
    field("definition", &row.definition, "I"); field("particle", &row.particle, "I");
    field("eff_num", &row.efficiency.numerator, "D"); field("eff_den", &row.efficiency.denominator, "D");
    field("pur_num", &row.purity.numerator, "D"); field("pur_den", &row.purity.denominator, "D");
    field("eff_entries", &row.efficiency.entries, "L"); field("pur_entries", &row.purity.entries, "L");
}
inline void write(const std::string& directory, int energy, const std::string& method,
                  double beta, const std::vector<Row>& values, const std::string& source) {
    TDirectory::TContext context;
    gSystem->mkdir(directory.c_str(), true);
    const std::string path = directory+"/"+kSummaryName;
    TFile output(path.c_str(), "RECREATE");
    if (output.IsZombie()) throw std::runtime_error("Cannot create "+path);
    TNamed("schema", kSchema).Write(); TNamed("method", method.c_str()).Write();
    TNamed("original_definition", kOriginalDefinition).Write();
    TNamed("cedric_definition", kCedricDefinition).Write();
    TNamed("particle_order", "electron,pion,photon,neutron,K0,muon").Write();
    TNamed("source", source.c_str()).Write();
    TParameter<int>("beam_energy_GeV", energy).Write();
    TParameter<double>("beta_threshold", beta).Write();
    TTree tree("clustering", "Numerators, denominators and counts for the two definitions");
    Row row; bind(tree, row, true);
    for (const auto& value : values) { row = value; tree.Fill(); }
    tree.Write();
    std::ofstream text(directory+"/clustering_summary.txt");
    if (!text) throw std::runtime_error("Cannot create clustering summary text");
    text << "# " << kSchema << " beam_energy_GeV=" << energy << " method=" << method << '\n'
         << "# original: " << kOriginalDefinition << '\n'
         << "# cedric: " << kCedricDefinition << '\n'
         << "definition particle efficiency purity eff_num eff_den pur_num pur_den eff_entries pur_entries\n"
         << std::setprecision(17);
    for (const auto& value : values)
        text << kDefinitions[value.definition] << ' ' << kParticles[value.particle] << ' '
             << value.efficiency.value() << ' ' << value.purity.value() << ' '
             << value.efficiency.numerator << ' ' << value.efficiency.denominator << ' '
             << value.purity.numerator << ' ' << value.purity.denominator << ' '
             << value.efficiency.entries << ' ' << value.purity.entries << '\n';
    std::cout << "Saved clustering comparison data: " << path << std::endl;
}
} // namespace pid_cluster
#endif
