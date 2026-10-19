// Read single-energy outputs without re-reading inference ROOTs or running GNN.
// From macro/:
//   root -l -b -q 'plot_jet_energy_resolution_pid.cxx()'
//   root -l -b -q 'plot_jet_energy_resolution_pid.cxx("figures/nnqq2M_fixed_uds_5head_pid","figures/pid_comparison")'
// Optional third argument: a separate base directory for Pandora summaries.
// Without that override, missing Pandora summaries use the previously saved
// H5 reference in /home/murata/research/studies/pandora_eval_validation/
// pandora_h5_all_energies_20260925. The optional fourth argument changes that
// reference directory (tables/resolutions.csv and tables/idr_evaluation.root);
// an empty fourth argument disables it. Only no-neutrino-correction rows are
// used, after checking their bins and values against the saved histograms.
// Inputs: BASE/{40,91,200,350,500}GeV/{reco,truth_clustering,pandora}/
//         jet_energy_resolution_summary.root
// Outputs: angle_all_energies, angle_energy_panels, central_vs_jet_energy,
//          forward_vs_jet_energy in png/pdf/eps, plus a comparison ROOT file.
// GNN summaries come from efficiency_purity_check_reco_effpur_contiribution_pid
// for each configured qq_energy/use_truth_clustering. Its existing input
// arguments remain unchanged; the third argument chooses ENERGYGeV/METHOD.
// An output directory override does not change qq_energy/method. The saved
// Pandora reference requires no rerun of the efficiency macro or inference.
// Axis/energy/RMS90 definitions are checked before plotting. Missing input
// files and statistically invalid points are explicitly skipped, never zeroed.
// Angular bins match the existing Pandora H5 comparison, ending at
// 0.95-0.97 and 0.97-0.98. Older summaries ending at 0.975 and 1.0 are
// regrouped from their saved events in memory; source ROOTs and the stored
// central/forward results are retained. No inference or hit-level read is needed.
// Also saves event_energy_ratio_ENERGYGeV_{central,forward,inclusive} and
// efficiency_purity_bar_{original,cedric}_{reco,truth_clustering}.
// Ratios use visible MC event energy and a Gaussian fit over [0.9,1.1].
// Bars read clustering_summary.root alongside each single-energy summary.
// Pandora reference numbers are stored separately in data/pandora_reference_20260925.txt.
// Missing CSV/energy ROOT files fall back to this text snapshot automatically.
// The optional fifth argument selects that file; an empty fifth argument disables
// all use of the snapshot (including saved Pandora efficiency/purity means).

#include "root_common_includes.h"
#include "src/pid_jet_resolution_summary.h"
#include "src/pid_pandora_reference.h"
#include "TLatex.h"
#include "TNamed.h"
#include "TParameter.h"
#include <array>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>

static const string model_condition = "5head_pid_onehot";  // 5head_pid 5head_pid_onehot


namespace pid_jer_plot {
static const int kEnergies[5] = {40, 91, 200, 350, 500};
static const char* const kMethods[3] = {"reco", "truth_clustering", "pandora"};
static const char* const kLabels[3] = {"GNN (PID)", "GNN truth clustering (PID)", "Pandora"};
static const int kMarkers[3] = {26, 25, 24};
static const int kColors[5] = {kMagenta+1, kBlue, kGreen+2, kRed, kBlack};
static const int kMethodColors[3] = {kOrange+7, kGreen+2, kBlue};
static const char* const kYTitle = "RMS_{90}(E_{j}) / Mean_{90}(E_{j}) [%]";
static const char* const kPandoraReferenceDirectory =
    "/home/murata/research/studies/pandora_eval_validation/pandora_h5_all_energies_20260925";

// Resolve once when loading the macro, independently of the plotting CWD.
inline std::string defaultPandoraSnapshotFile() {
    std::string directory = gSystem->DirName(__FILE__);
    if (!gSystem->IsAbsoluteFileName(directory.c_str()))
        directory = std::string(gSystem->WorkingDirectory()) + "/" + directory;
    return directory + "/data/pandora_reference_20260925.txt";
}
static const std::string kPandoraSnapshotFile = defaultPandoraSnapshotFile();

struct Sample {
    bool present = false;
    bool angular_recomputed = false;
    bool has_beta_threshold = true;
    double beta = 0.;
    std::string path;
    std::string source_kind = "single_energy_summary";
    std::string histogram_path;
    std::string event_path;
    std::shared_ptr<const pid_pandora_reference::Snapshot> pandora_snapshot;
    std::array<pid_jer::Point, pid_jer::kBins+2> points;
};
using Samples = std::array<std::array<Sample,5>,3>;

inline std::string named(TFile& file, const char* key) {
    auto* value = dynamic_cast<TNamed*>(file.Get(key));
    if (!value) throw std::runtime_error(std::string("Missing summary metadata: ") + key);
    return value->GetTitle();
}

bool hasAngularEdges(const Sample& sample, const double* edges) {
    for (int i = 0; i < pid_jer::kBins; ++i) {
        const auto& p = sample.points[i];
        if (std::fabs(p.low-edges[i]) > 1e-9 ||
            std::fabs(p.high-edges[i+1]) > 1e-9) return false;
    }
    return true;
}

void recomputeAngularPoints(Sample& sample, TFile& file, const std::string& path) {
    auto* events = dynamic_cast<TTree*>(file.Get("events"));
    double thrust = 0., energy = 0.;
    if (!events || !events->GetBranch("thrust") || !events->GetBranch("reco_energy") ||
        events->SetBranchAddress("thrust", &thrust) < 0 ||
        events->SetBranchAddress("reco_energy", &energy) < 0)
        throw std::runtime_error("Old angular bins require saved events/thrust and reco_energy: " + path);
    events->SetBranchStatus("*", 0);
    events->SetBranchStatus("thrust", 1);
    events->SetBranchStatus("reco_energy", 1);
    std::array<std::unique_ptr<TH1F>, pid_jer::kBins> histograms;
    std::array<Long64_t, pid_jer::kBins> invalid{};
    for (int i = 0; i < pid_jer::kBins; ++i) {
        histograms[i].reset(new TH1F(Form("rebinned_energy_angle_%d", i), "", 100000, 0., 5000.));
        histograms[i]->SetDirectory(nullptr);
    }
    for (Long64_t i = 0, n = events->GetEntries(); i < n; ++i) {
        if (events->GetEntry(i) <= 0)
            throw std::runtime_error("Unreadable saved event while regrouping angular bins: " + path);
        const int bin = pid_jer::angleBin(thrust);
        if (bin < 0) continue;
        if (std::isfinite(energy)) histograms[bin]->Fill(energy);
        else ++invalid[bin];
    }
    events->ResetBranchAddresses();
    for (int i = 0; i < pid_jer::kBins; ++i) {
        auto p = pid_jer::statistics(*histograms[i], invalid[i]);
        p.angle_bin = i; p.low = pid_jer::kEdges[i]; p.high = pid_jer::kEdges[i+1];
        sample.points[i] = p;
    }
    sample.angular_recomputed = true;
    std::cout << "Regrouped angular bins from saved events (through 0.97 and 0.98): "
              << path << std::endl;
}

bool load(Sample& sample, const std::string& path, int method, int energy) {
    sample.path = path;
    if (gSystem->AccessPathName(path.c_str())) {
        std::cerr << "Missing summary (skipped): " << path << std::endl;
        return false;
    }
    TDirectory::TContext context;
    TFile file(path.c_str(), "READ");
    if (file.IsZombie()) throw std::runtime_error("Cannot read " + path);
    if (named(file,"schema") != pid_jer::kSchema ||
        named(file,"angle_definition") != pid_jer::kAngleDefinition ||
        named(file,"metric_definition") != pid_jer::kMetricDefinition)
        throw std::runtime_error("Incompatible angle/metric summary: " + path);
    const std::string energy_definition = named(file,"energy_definition");
    if (energy_definition.find("no neutrino correction") == std::string::npos ||
        energy_definition.find("not Ereco/Etruth") == std::string::npos)
        throw std::runtime_error("Incompatible energy/correction definition: " + path);
    if (named(file,"method") != kMethods[method])
        throw std::runtime_error("Summary method disagrees with its directory: " + path);
    auto* beam = dynamic_cast<TParameter<int>*>(file.Get("beam_energy_GeV"));
    auto* beta = dynamic_cast<TParameter<double>*>(file.Get("beta_threshold"));
    if (!beam || beam->GetVal() != energy || !beta || !std::isfinite(beta->GetVal()))
        throw std::runtime_error("Summary beam energy/configuration mismatch: " + path);
    sample.beta = beta->GetVal();
    auto* tree = dynamic_cast<TTree*>(file.Get("resolution"));
    if (!tree || tree->GetEntries() != pid_jer::kBins+2)
        throw std::runtime_error("Missing/incomplete resolution table: " + path);
    pid_jer::Point p;
    auto bind = [&](const char* name, void* address) {
        if (!tree->GetBranch(name) || tree->SetBranchAddress(name, address) < 0)
            throw std::runtime_error(std::string("Invalid summary branch: ") + name + " in " + path);
    };
    bind("angle_bin", &p.angle_bin); bind("low", &p.low); bind("high", &p.high);
    bind("entries", &p.entries); bind("invalid_entries", &p.invalid_entries);
    bind("underflow", &p.underflow); bind("overflow", &p.overflow);
    bind("mean90", &p.mean90); bind("rms90", &p.rms90);
    bind("resolution_percent", &p.resolution_percent); bind("error_percent", &p.error_percent);
    bind("valid", &p.valid);
    std::array<bool,pid_jer::kBins+2> found{};
    for (Long64_t i = 0; i < tree->GetEntries(); ++i) {
        if (tree->GetEntry(i) <= 0) throw std::runtime_error("Unreadable summary row: " + path);
        const int index = p.angle_bin >= 0 ? p.angle_bin :
            (p.angle_bin == -1 ? pid_jer::kBins : (p.angle_bin == -2 ? pid_jer::kBins+1 : -1));
        if (index < 0 || index >= int(found.size()) || found[index])
            throw std::runtime_error("Invalid/duplicate summary bin: " + path);
        const bool wrong_region = index >= pid_jer::kBins &&
            (std::fabs(p.low-(p.angle_bin == -1 ? 0. : .7)) > 1e-9 ||
             std::fabs(p.high-(p.angle_bin == -1 ? .7 : .98)) > 1e-9);
        if (!std::isfinite(p.low) || !std::isfinite(p.high) || wrong_region ||
            (p.valid && (p.entries < 5 || p.invalid_entries || p.underflow || p.overflow ||
                        !std::isfinite(p.resolution_percent) || !std::isfinite(p.error_percent) ||
                        !(p.mean90 > 0.) || p.resolution_percent < 0. || p.error_percent < 0.)))
            throw std::runtime_error("Invalid summary statistics/boundaries: " + path);
        found[index] = true;
        sample.points[index] = p;
    }
    tree->ResetBranchAddresses();
    if (!hasAngularEdges(sample, pid_jer::kEdges)) {
        static const double old_edges[pid_jer::kBins+1] = {
            0., .1, .2, .3, .4, .5, .6, .7, .8, .9, .925, .950, .975, 1.
        };
        if (!hasAngularEdges(sample, old_edges))
            throw std::runtime_error("Unsupported summary angular boundaries: " + path);
        recomputeAngularPoints(sample, file, path);
    }
    sample.present = true;
    return true;
}

// The text file contains saved statistics, not reconstructed/synthetic events.
bool loadPandoraTextReference(Sample& sample, const std::string& path, int energy) {
    if (path.empty() || gSystem->AccessPathName(path.c_str())) {
        std::cerr << "Missing Pandora text reference (skipped): " << path << std::endl;
        return false;
    }
    auto snapshot = std::make_shared<pid_pandora_reference::Snapshot>(
        pid_pandora_reference::load(path));
    sample = Sample{};
    sample.points = snapshot->points[pid_pandora_reference::energyIndex(energy)];
    sample.path = snapshot->path;
    sample.source_kind = "pandora_text_reference";
    sample.has_beta_threshold = false;
    sample.beta = std::numeric_limits<double>::quiet_NaN();
    sample.pandora_snapshot = std::move(snapshot);
    sample.present = true;
    std::cout << "Using Pandora text snapshot: " << energy << " GeV, " << path << std::endl;
    return true;
}

bool loadPandoraReference(Sample& sample, const std::string& directory, int energy,
                          const char* snapshotFile = nullptr) {
    const std::string csv_path = directory + "/tables/resolutions.csv";
    const std::string root_path = directory + "/tables/idr_evaluation.root";
    if (gSystem->AccessPathName(csv_path.c_str()) || gSystem->AccessPathName(root_path.c_str())) {
        const std::string snapshot_path = snapshotFile ? snapshotFile : kPandoraSnapshotFile;
        if (!snapshot_path.empty())
            return loadPandoraTextReference(sample,snapshot_path,energy);
        std::cerr << "Missing saved Pandora reference (skipped): " << directory << std::endl;
        return false;
    }
    std::ifstream table(csv_path);
    std::string line;
    if (!std::getline(table, line)) throw std::runtime_error("Cannot read " + csv_path);
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "dijet_energy_GeV,jet_energy_GeV,region,angle_low,angle_high,neutrino_correction,events,resolution_percent,error_percent,underflow,overflow,histogram")
        throw std::runtime_error("Unsupported Pandora reference CSV columns: " + csv_path);
    TDirectory::TContext context;
    TFile file(root_path.c_str(), "READ");
    if (file.IsZombie()) throw std::runtime_error("Cannot read " + root_path);
    std::array<bool, pid_jer::kBins+2> found{};
    while (std::getline(table, line)) {
        if (line.empty() || line == "\r") continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        int beam = 0;
        Long64_t entries = 0;
        double jet = 0., low = 0., high = 0., resolution = 0., error = 0., under = 0., over = 0.;
        std::string region, corrected, histogram, extra;
        if (!(row >> beam >> jet >> region >> low >> high >> corrected >> entries
                  >> resolution >> error >> under >> over >> histogram) || (row >> extra))
            throw std::runtime_error("Malformed Pandora reference CSV row: " + csv_path);
        if (corrected != "False" && corrected != "True")
            throw std::runtime_error("Invalid Pandora neutrino correction flag: " + csv_path);
        if (beam != energy || corrected == "True") continue;
        int index = -1;
        for (int i = 0; i < pid_jer::kBins; ++i)
            if (region == Form("angle_%02d", i)) index = i;
        if (region == "central") index = pid_jer::kBins;
        if (region == "forward") index = pid_jer::kBins+1;
        if (index < 0 || found[index])
            throw std::runtime_error("Unknown/duplicate Pandora reference region: " + region);
        const double expected_low = index < pid_jer::kBins ? pid_jer::kEdges[index] : (index == pid_jer::kBins ? 0. : .7);
        const double expected_high = index < pid_jer::kBins ? pid_jer::kEdges[index+1] : (index == pid_jer::kBins ? .7 : .98);
        const std::string expected_name = "E" + std::to_string(energy) + "_" + region + "_no_nu_correction";
        if (!std::isfinite(jet) || !std::isfinite(low) || !std::isfinite(high) ||
            std::fabs(jet-energy/2.) > 1e-9 || std::fabs(low-expected_low) > 1e-9 ||
            std::fabs(high-expected_high) > 1e-9 || histogram != expected_name)
            throw std::runtime_error("Incompatible Pandora reference energy/angular bins: " + csv_path);
        auto* hist = dynamic_cast<TH1F*>(file.Get(histogram.c_str()));
        if (!hist || hist->GetNbinsX() != 100000 || hist->GetXaxis()->GetXmin() != 0. ||
            hist->GetXaxis()->GetXmax() != 5000.)
            throw std::runtime_error("Missing/incompatible Pandora energy histogram: " + histogram);
        auto p = pid_jer::statistics(*hist);
        if (!p.valid || !std::isfinite(resolution) || !std::isfinite(error) ||
            resolution <= 0. || error < 0. || p.entries != entries ||
            p.underflow != under || p.overflow != over ||
            std::fabs(p.resolution_percent-resolution) > 1e-6*resolution ||
            std::fabs(p.error_percent-error) > 1e-6*std::max(error, 1e-12))
            throw std::runtime_error("Pandora CSV and saved histogram disagree: " + histogram);
        p.angle_bin = index < pid_jer::kBins ? index : (index == pid_jer::kBins ? -1 : -2);
        p.low = low; p.high = high;
        p.resolution_percent = resolution; p.error_percent = error;
        sample.points[index] = p;
        found[index] = true;
    }
    if (!table.eof() || std::find(found.begin(), found.end(), false) != found.end())
        throw std::runtime_error("Incomplete Pandora reference table: " + csv_path);
    sample.path = csv_path;
    sample.event_path = directory + "/tables/event_energy_" + std::to_string(energy) + "GeV.root";
    sample.histogram_path = root_path;
    sample.source_kind = "pandora_h5_reference";
    sample.has_beta_threshold = false;  // The full-PFO reference has no GNN beta cut.
    sample.beta = std::numeric_limits<double>::quiet_NaN();
    sample.present = true;
    std::cout << "Loaded saved Pandora reference: " << energy << " GeV, no neutrino correction, "
              << csv_path << std::endl;
    return true;
}

std::unique_ptr<TGraphErrors> angleGraph(const Sample& sample, int method, int energy_index, bool energy_color) {
    std::unique_ptr<TGraphErrors> graph(new TGraphErrors());
    graph->SetName(Form("angle_%s_%d", kMethods[method], kEnergies[energy_index]));
    graph->SetMarkerStyle(kMarkers[method]); graph->SetMarkerSize(1.5);
    graph->SetMarkerColor(energy_color ? kColors[energy_index] : kMethodColors[method]);
    graph->SetLineColor(graph->GetMarkerColor()); graph->SetLineWidth(2);
    if (sample.present) for (int i = 0; i < pid_jer::kBins; ++i) {
        const auto& p = sample.points[i];
        if (!p.valid) continue;
        const int n = graph->GetN();
        graph->SetPoint(n, (p.low+p.high)*.5, p.resolution_percent);
        graph->SetPointError(n, (p.high-p.low)*.5, p.error_percent);
    }
    return graph;
}

double ymax(const Samples& samples, bool angular) {
    double result = 20.;
    for (const auto& method : samples) for (const auto& sample : method) {
        if (!sample.present) continue;
        for (int i = angular ? 0 : pid_jer::kBins; i < (angular ? pid_jer::kBins : pid_jer::kBins+2); ++i) {
            const auto& p = sample.points[i];
            if (p.valid) result = std::max(result, 1.12*(p.resolution_percent+p.error_percent));
        }
    }
    return result;
}

void stylePad(TPad* pad, double top = .06) {
    pad->SetLeftMargin(.14); pad->SetRightMargin(.035);
    pad->SetBottomMargin(.14); pad->SetTopMargin(top); pad->SetTicks(1,1);
}
void styleFrame(TH1* frame) {
    frame->GetXaxis()->CenterTitle(); frame->GetYaxis()->CenterTitle();
    frame->GetXaxis()->SetTitleFont(42); frame->GetYaxis()->SetTitleFont(42);
    frame->GetXaxis()->SetLabelFont(42); frame->GetYaxis()->SetLabelFont(42);
}
void footer() {
    TLatex text;
    text.SetNDC(); text.SetTextFont(42); text.SetTextSize(.025);
    text.DrawLatex(.15,.015,"Pandora upstream thrust definition; no neutrino correction");
}
void finish(TCanvas& canvas, TFile& output, const std::string& directory, const char* name) {
    canvas.Modified(); canvas.Update();
    pid_jer::saveCanvas(&canvas, directory, name);
    output.cd(); canvas.Write(name);
}

void drawAll(const Samples& samples, TFile& output, const std::string& directory) {
    std::vector<std::unique_ptr<TGraphErrors>> graphs;
    TCanvas canvas("pid_angle_all_energies", "All jet energies and reconstruction methods", 1500, 1200);
    stylePad(&canvas, .25);
    auto* frame = canvas.DrawFrame(0.,0.,1.,ymax(samples,true));
    frame->SetTitle(Form(";|cos(#theta_{thrust})|;%s",kYTitle));
    styleFrame(frame);
    frame->GetYaxis()->SetTitleOffset(1.35);
    for (int m=0; m<3; ++m) for (int e=0; e<5; ++e) {
        graphs.push_back(angleGraph(samples[m][e],m,e,true));
        graphs.back()->Draw("P SAME");
    }
    TLegend energies(.15,.80,.57,.98); energies.SetNColumns(2); energies.SetBorderSize(0); energies.SetFillStyle(0); energies.SetTextFont(42);
    for (int e=0; e<5; ++e) energies.AddEntry(graphs[e].get(),Form("%g GeV Jets",kEnergies[e]/2.),"p");
    energies.Draw();
    TLegend methods(.60,.79,.96,.98); methods.SetBorderSize(0); methods.SetFillStyle(0); methods.SetTextFont(42);
    for (int m=0; m<3; ++m) {
        std::unique_ptr<TGraphErrors> key(new TGraphErrors());
        key->SetMarkerStyle(kMarkers[m]); key->SetMarkerColor(kBlack); key->SetMarkerSize(1.3);
        methods.AddEntry(key.get(),kLabels[m],"p"); graphs.push_back(std::move(key));
    }
    methods.Draw(); footer();
    finish(canvas,output,directory,"angle_all_energies");
}

void drawPanels(const Samples& samples, TFile& output, const std::string& directory) {
    std::vector<std::unique_ptr<TGraphErrors>> graphs;
    TCanvas canvas("pid_angle_energy_panels", "Angular resolution by jet energy", 1800, 1100);
    canvas.Divide(3,2,.002,.002);
    for (int e=0; e<5; ++e) {
        auto* pad = static_cast<TPad*>(canvas.cd(e+1)); stylePad(pad);
        auto* frame = pad->DrawFrame(0.,0.,1.,ymax(samples,true));
        frame->SetTitle(Form(";|cos(#theta_{thrust})|;%s",kYTitle));
        styleFrame(frame);
        frame->GetYaxis()->SetTitleOffset(1.4);
        int points = 0;
        for (int m=0; m<3; ++m) {
            graphs.push_back(angleGraph(samples[m][e],m,e,false));
            points += graphs.back()->GetN(); graphs.back()->Draw("P SAME");
        }
        TLatex label; label.SetNDC(); label.SetTextFont(42); label.SetTextSize(.052);
        label.DrawLatex(.19,.84,Form("%g GeV Jets",kEnergies[e]/2.));
        if (!points) { label.SetTextSize(.035); label.DrawLatex(.22,.5,"No valid resolution points"); }
    }
    canvas.cd(6);
    TLegend legend(.08,.39,.94,.72); legend.SetBorderSize(0); legend.SetFillStyle(0); legend.SetTextFont(42); legend.SetTextSize(.055);
    for (int m=0; m<3; ++m) legend.AddEntry(graphs[m].get(),kLabels[m],"pe");
    legend.Draw();
    TLatex note; note.SetNDC(); note.SetTextFont(42); note.SetTextSize(.038);
    note.DrawLatex(.08,.27,"Pandora upstream thrust definition");
    note.DrawLatex(.08,.20,"No neutrino correction");
    finish(canvas,output,directory,"angle_energy_panels");
}

void drawRegion(const Samples& samples, int region, TFile& output, const std::string& directory) {
    std::vector<std::unique_ptr<TGraphErrors>> graphs;
    const char* name = region == pid_jer::kBins ? "central_vs_jet_energy" : "forward_vs_jet_energy";
    TCanvas canvas(Form("pid_%s",name), "Jet resolution versus energy", 1100, 900);
    stylePad(&canvas);
    auto* frame = canvas.DrawFrame(0.,0.,275.,ymax(samples,false));
    frame->SetTitle(Form(";E_{jet} [GeV];%s",kYTitle));
    styleFrame(frame);
    frame->GetYaxis()->SetTitleOffset(1.35);
    TLegend legend(.18,.61,.69,.79); legend.SetBorderSize(0); legend.SetFillStyle(0); legend.SetTextFont(42);
    for (int m=0; m<3; ++m) {
        std::unique_ptr<TGraphErrors> graph(new TGraphErrors());
        graph->SetName(Form("%s_%s",name,kMethods[m]));
        graph->SetMarkerStyle(kMarkers[m]); graph->SetMarkerSize(1.3);
        graph->SetMarkerColor(kMethodColors[m]); graph->SetLineColor(kMethodColors[m]); graph->SetLineWidth(2);
        for (int e=0; e<5; ++e) {
            const auto& sample = samples[m][e]; const auto& p = sample.points[region];
            if (!sample.present || !p.valid) continue;
            const int n = graph->GetN();
            graph->SetPoint(n,kEnergies[e]/2.,p.resolution_percent);
            graph->SetPointError(n,0.,p.error_percent);
        }
        graph->Draw("LP SAME"); legend.AddEntry(graph.get(),kLabels[m],"lp");
        output.cd(); graph->Write(); graphs.push_back(std::move(graph));
    }
    legend.Draw();
    TLatex label; label.SetNDC(); label.SetTextFont(42); label.SetTextSize(.04);
    label.DrawLatex(.38,.85,region == pid_jer::kBins
        ? "|cos(#theta_{thrust})| < 0.7" : "0.7 < |cos(#theta_{thrust})| < 0.98");
    footer(); finish(canvas,output,directory,name);
}
} // namespace pid_jer_plot

#include "src/pid_comparison_distributions.h"

void plot_jet_energy_resolution_pid(
    const char* baseDirectory = Form("figures/nnqq2M_fixed_uds_%s",model_condition.c_str()),
    const char* outputDirectory = "",
    const char* pandoraBaseDirectory = "",
    const char* pandoraReferenceDirectory = pid_jer_plot::kPandoraReferenceDirectory,
    const char* pandoraSnapshotFile = nullptr) {
    using namespace pid_jer_plot;
    Samples samples;
    int loaded = 0;
    double beta = std::numeric_limits<double>::quiet_NaN();
    for (int m=0; m<3; ++m) for (int e=0; e<5; ++e) {
        const std::string base = m == 2 && pandoraBaseDirectory[0] ? pandoraBaseDirectory : baseDirectory;
        const std::string path = base + "/" + std::to_string(kEnergies[e]) + "GeV/" + kMethods[m] + "/" + pid_jer::kSummaryName;
        auto& sample = samples[m][e];
        const bool use_reference = m == 2 && !pandoraBaseDirectory[0] &&
            pandoraReferenceDirectory && pandoraReferenceDirectory[0] && gSystem->AccessPathName(path.c_str());
        if (use_reference) {
            if (!loadPandoraReference(sample,pandoraReferenceDirectory,kEnergies[e],pandoraSnapshotFile)) continue;
            const std::string event_cache = base + "/" + std::to_string(kEnergies[e])+"GeV/pandora/event_energy_reference.root";
            if (sample.source_kind == "pandora_h5_reference" && !gSystem->AccessPathName(event_cache.c_str())) sample.event_path = event_cache;
        } else if (!load(sample,path,m,kEnergies[e])) continue;
        if (sample.has_beta_threshold) {
            if (std::isfinite(beta) && std::fabs(beta-sample.beta) > 1e-12)
                throw std::runtime_error("Cannot compare summaries with different beta cuts: " + path);
            beta = sample.beta;
        }
        ++loaded;
    }
    if (!loaded) throw std::runtime_error("No jet_energy_resolution_summary.root inputs found");
    const std::string directory = outputDirectory[0] ? outputDirectory : std::string(baseDirectory)+"/comparison";
    gSystem->mkdir(directory.c_str(),true);
    TDirectory::TContext context;
    TFile output((directory+"/jet_energy_resolution_comparison.root").c_str(),"RECREATE");
    if (output.IsZombie()) throw std::runtime_error("Cannot create comparison ROOT");
    TNamed("schema",pid_jer::kSchema).Write();
    TNamed("angle_definition",pid_jer::kAngleDefinition).Write();
    TNamed("metric_definition",pid_jer::kMetricDefinition).Write();
    TNamed("angular_bin_definition","Pandora H5 bins; final bins [0.95,0.97) and [0.97,0.98)").Write();
    TNamed("neutrino_correction","none").Write();
    TTree inputs("inputs","Source summaries and saved Pandora reference used in this comparison");
    std::string path, method, source_kind, histogram_path, event_path;
    int energy = 0;
    bool angular_recomputed = false;
    inputs.Branch("path",&path); inputs.Branch("method",&method); inputs.Branch("beam_energy_GeV",&energy,"beam_energy_GeV/I");
    inputs.Branch("angular_recomputed_from_events",&angular_recomputed,"angular_recomputed_from_events/O");
    inputs.Branch("source_kind",&source_kind); inputs.Branch("histogram_path",&histogram_path);
    inputs.Branch("event_path",&event_path);
    for (int m=0; m<3; ++m) for (int e=0; e<5; ++e) if (samples[m][e].present) {
        path = samples[m][e].path; method = kMethods[m]; energy = kEnergies[e];
        source_kind = samples[m][e].source_kind; histogram_path = samples[m][e].histogram_path;
        event_path = samples[m][e].event_path.empty() ? path : samples[m][e].event_path;
        angular_recomputed = samples[m][e].angular_recomputed; inputs.Fill();
        auto graph = angleGraph(samples[m][e],m,e,true); graph->Write();
    }
    inputs.Write();
    gStyle->SetOptStat(0);
    drawAll(samples,output,directory); drawPanels(samples,output,directory);
    drawRegion(samples,pid_jer::kBins,output,directory);
    drawRegion(samples,pid_jer::kBins+1,output,directory);
    drawRatios(samples,output,directory);
    ClusteringSamples clustering;
    TTree clustering_table("clustering_comparison","Original and Cedric metrics used in the bar plots");
    pid_cluster::Row row;
    std::string clustering_method, clustering_path, clustering_source_kind;
    int clustering_energy = 0;
    clustering_table.Branch("method",&clustering_method); clustering_table.Branch("source",&clustering_path);
    clustering_table.Branch("source_kind",&clustering_source_kind);
    clustering_table.Branch("beam_energy_GeV",&clustering_energy,"beam_energy_GeV/I");
    pid_cluster::bind(clustering_table,row,true);
    for (int m=0; m<3; ++m) for (int e=0; e<5; ++e) {
        const std::string base = m == 2 && pandoraBaseDirectory[0] ? pandoraBaseDirectory : baseDirectory;
        const std::string path = base + "/" + std::to_string(kEnergies[e])+"GeV/"+kMethods[m]+"/"+pid_cluster::kSummaryName;
        if (m == 2 && !pandoraBaseDirectory[0] && gSystem->AccessPathName(path.c_str())) {
            if (!loadSavedPandoraBars(clustering[m][e],e,pandoraSnapshotFile)) continue;
        } else if (!loadClustering(clustering[m][e],path,m,kEnergies[e],beta)) continue;
        clustering_method = kMethods[m]; clustering_energy = kEnergies[e]; clustering_path = clustering[m][e].path;
        clustering_source_kind = clustering[m][e].source_kind;
        for (int d=0; d<2; ++d) for (int p=0; p<6; ++p) if (clustering[m][e].found[d][p]) {
            row = clustering[m][e].rows[d][p]; clustering_table.Fill();
        }
    }
    output.cd(); clustering_table.Write();
    TNamed("clustering_original_definition",pid_cluster::kOriginalDefinition).Write();
    TNamed("clustering_cedric_definition",pid_cluster::kCedricDefinition).Write();
    TNamed("clustering_pandora_marker","Black points and horizontal marker guides; no statistical error bars").Write();
    TNamed("clustering_saved_plot_means","saved_plot_means_no_counts: numerator/denominator=published mean/1; entries=0 means unavailable, not zero events; original definition only").Write();
    for (int d=0; d<2; ++d) for (int m=0; m<2; ++m) drawClustering(clustering,m,d,output,directory);
    std::cout << "Read " << loaded << "/15 energy/method results; saved comparison figures to " << directory << std::endl;
}
