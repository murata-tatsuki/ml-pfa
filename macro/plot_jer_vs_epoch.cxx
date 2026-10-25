// Plot existing epoch-scan metrics.csv; no inference or ROOT-data rereading.
// From master/macro/:
//   root -l -b -q 'plot_jer_vs_epoch.cxx()'
//   root -l -b -q 'plot_jer_vs_epoch.cxx("../output/jer_epoch_scan/5head_pid_onehot_2026_10_09_154602/200GeV")'
// Arguments: input, output directory, readout, method, region, zoom upper JER [%],
// training log (empty=auto; none=disable), beta start epoch, energy start epoch.
// Loss plots use epoch-matched training logs; all original JER plots remain available.
// Input: a model/run directory, an energy directory, or a metrics.csv file.
// method: both/reco/truth_clustering; region: both/central/forward.
// zoom=0 disables the additional zoom plots. Full-range plots are always saved.

#include "src/jer_epoch_csv.h"
#include "src/jer_training_loss.h"

#include <TCanvas.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH1F.h>
#include <TLatex.h>
#include <TLegend.h>
#include <TNamed.h>
#include <TPad.h>
#include <TROOT.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <regex>

namespace jer_epoch_plot {

using Key = std::tuple<int, std::string, std::string>; // energy, region, method
using Series = std::map<Key, std::vector<CsvRow>>;

struct Data {
    Series series;
    std::string model, run, readout, method, region;
    std::vector<std::string> sources, regions, methods;
    std::set<int> energies;
    double beta = 0;
};

std::string join(const std::vector<std::string>& values, const std::string& separator) {
    std::ostringstream out;
    for (std::size_t i = 0; i < values.size(); ++i) out << (i ? separator : "") << values[i];
    return out.str();
}

std::string number(double value, int precision = 2) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string safeName(const std::string& value) {
    std::string out = value;
    for (char& ch : out)
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_') ch = '_';
    return out;
}

std::string parent(const std::string& path) {
    const std::size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

std::string absolute(const std::string& path) {
    TString expanded(path.c_str());
    gSystem->ExpandPathName(expanded);
    if (!gSystem->IsAbsoluteFileName(expanded.Data()))
        expanded = TString(gSystem->WorkingDirectory()) + "/" + expanded;
    return expanded.Data();
}

bool isDirectory(const std::string& path) {
    FileStat_t info;
    return gSystem->GetPathInfo(path.c_str(), info) == 0 && R_ISDIR(info.fMode);
}

bool isFile(const std::string& path) {
    FileStat_t info;
    return gSystem->GetPathInfo(path.c_str(), info) == 0 && R_ISREG(info.fMode);
}

std::vector<std::string> discover(const std::string& input) {
    if (isFile(input)) return {input};
    if (!isDirectory(input)) throw std::runtime_error("Input does not exist: " + input);
    if (isFile(input + "/metrics.csv")) return {input + "/metrics.csv"};
    std::vector<std::string> files;
    void* directory = gSystem->OpenDirectory(input.c_str());
    if (!directory) throw std::runtime_error("Cannot list directory: " + input);
    while (const char* item = gSystem->GetDirEntry(directory)) {
        const std::string name(item);
        if (name.size() <= 3 || name.substr(name.size() - 3) != "GeV") continue;
        const std::string digits = name.substr(0, name.size() - 3);
        if (digits.find_first_not_of("0123456789") != std::string::npos) continue;
        const std::string path = input + "/" + name;
        if (!isDirectory(path)) continue;
        if (isFile(path + "/metrics.csv")) files.push_back(path + "/metrics.csv");
        else std::cerr << "[skip] No metrics.csv (run --collect first): " << path << '\n';
    }
    gSystem->FreeDirectory(directory);
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::runtime_error("No metrics.csv found; supply a model/run or energy directory: " + input);
    return files;
}

Data readData(const std::string& input, const std::string& readout,
              const std::string& method, const std::string& region) {
    if (readout != "saved" && readout != "raw_head" && readout != "track_replaced")
        throw std::runtime_error("readout must be saved, raw_head, or track_replaced");
    if (method != "both" && method != "reco" && method != "truth_clustering")
        throw std::runtime_error("method must be both, reco, or truth_clustering");
    if (region != "both" && region != "central" && region != "forward")
        throw std::runtime_error("region must be both, central, or forward");
    Data data;
    data.readout = readout; data.method = method; data.region = region;
    data.sources = discover(input);
    data.methods = method == "both" ? std::vector<std::string>{"reco", "truth_clustering"} : std::vector<std::string>{method};
    data.regions = region == "both" ? std::vector<std::string>{"central", "forward"} : std::vector<std::string>{region};
    std::map<int, std::tuple<std::string, long long, long long>> coverage;
    std::map<std::pair<int, std::string>, std::tuple<double, double, long long>> bounds;
    std::map<std::string, std::pair<double, double>> regionBounds;
    std::set<std::tuple<int, int, std::string, std::string>> seen;
    bool first = true;
    for (const std::string& path : data.sources) {
        const auto rows = loadMetrics(path);
        std::size_t selected = 0;
        for (const CsvRow& row : rows) {
            if (row.readout != readout || (method != "both" && row.method != method)) continue;
            if (row.method != "reco" && row.method != "truth_clustering") continue;
            if (std::find(data.regions.begin(), data.regions.end(), row.region) == data.regions.end()) continue;
            const int bin = row.region == "central" ? -1 : -2;
            if (row.angleBin != bin) throw std::runtime_error("Inconsistent region/angle_bin in " + path);
            if (first) {
                data.model = row.model; data.run = row.run; data.beta = row.betaThreshold;
                first = false;
            }
            if (row.model != data.model || row.run != data.run || row.betaThreshold != data.beta)
                throw std::runtime_error("Mixed model/run or analysis beta threshold; plot separately");
            const auto identity = std::make_tuple(row.digest, row.events, row.expectedFiles);
            if (coverage.count(row.energy) && coverage.at(row.energy) != identity)
                throw std::runtime_error("Event selection or input coverage differs across epochs/methods for " + std::to_string(row.energy) + " GeV");
            coverage[row.energy] = identity;
            const auto angularKey = std::make_pair(row.energy, row.region);
            const auto angular = std::make_tuple(row.low, row.high, row.entries);
            if (bounds.count(angularKey) && bounds.at(angularKey) != angular)
                throw std::runtime_error("Angular bounds or event counts differ across epochs/methods");
            bounds[angularKey] = angular;
            const auto interval = std::make_pair(row.low, row.high);
            if (regionBounds.count(row.region) && regionBounds.at(row.region) != interval)
                throw std::runtime_error("Angular bounds differ across energies; plot separately");
            regionBounds[row.region] = interval;
            if (!seen.emplace(row.energy, row.epoch, row.method, row.region).second)
                throw std::runtime_error("Duplicate plot point across input CSV files");
            data.series[Key(row.energy, row.region, row.method)].push_back(row);
            data.energies.insert(row.energy);
            ++selected;
        }
        std::cout << "[read] " << path << ": " << selected << " selected points\n";
    }
    if (data.series.empty()) throw std::runtime_error("No valid, complete, eligible rows match the selection");
    for (auto& item : data.series) {
        auto& rows = item.second;
        std::sort(rows.begin(), rows.end(), [](const CsvRow& a, const CsvRow& b) { return a.epoch < b.epoch; });
        for (std::size_t i = 1; i < rows.size(); ++i)
            if (rows[i].epoch != rows[i - 1].epoch + 1)
                std::cerr << "[gap] " << std::get<0>(item.first) << " GeV " << std::get<1>(item.first)
                          << ' ' << std::get<2>(item.first) << ": " << rows[i - 1].epoch << " -> " << rows[i].epoch << '\n';
        const CsvRow& minimum = *std::min_element(rows.begin(), rows.end(), [](const CsvRow& a, const CsvRow& b) {
            return a.resolution < b.resolution;
        });
        std::cout << "[minimum in scan] " << minimum.energy << " GeV " << minimum.region << ' ' << minimum.method
                  << ": epoch=" << minimum.epoch << " JER=" << number(minimum.resolution, 5)
                  << " +/- " << number(minimum.error, 5) << "% (N=" << minimum.events << ")\n";
    }
    return data;
}

const CsvRow& minimum(const std::vector<CsvRow>& rows) {
    return *std::min_element(rows.begin(), rows.end(), [](const CsvRow& a, const CsvRow& b) {
        return a.resolution < b.resolution;
    });
}

int energyColor(int energy) {
    switch (energy) {
        case 40: return kViolet + 1;
        case 91: return kGreen + 2;
        case 200: return kAzure + 2;
        case 350: return kOrange + 7;
        case 500: return kRed + 1;
        default: return kGray + 2;
    }
}

std::unique_ptr<TGraphErrors> graph(const std::vector<CsvRow>& rows, std::size_t begin,
                                  std::size_t end, int color, int style) {
    std::unique_ptr<TGraphErrors> result(new TGraphErrors(static_cast<int>(end - begin)));
    for (std::size_t i = begin; i < end; ++i) {
        result->SetPoint(static_cast<int>(i - begin), rows[i].epoch, rows[i].resolution);
        result->SetPointError(static_cast<int>(i - begin), 0, rows[i].error);
    }
    result->SetLineColor(color); result->SetMarkerColor(color);
    result->SetLineStyle(style); result->SetLineWidth(2);
    result->SetMarkerStyle(style == 1 ? 20 : 24); result->SetMarkerSize(0.45);
    return result;
}

void label(double x, double y, const std::string& value, double size = 0.028) {
    TLatex text;
    text.SetNDC(); text.SetTextFont(42); text.SetTextSize(size);
    text.DrawLatex(x, y, value.c_str());
}

std::string suffix(const Data& data) {
    return "_" + data.readout + (data.method == "both" ? "" : "_" + data.method)
        + (data.region == "both" ? "" : "_" + data.region);
}

// Energy-specific images share one directory; comparison and common figures
// stay in the model/run output directory. ROOT object names remain unchanged.
std::string energyDirectory(const std::string& output, int energy) {
    if (energy <= 0) return output;
    const std::string directory = output + "/" + std::to_string(energy) + "GeV";
    if (gSystem->mkdir(directory.c_str(), true) != 0 && !isDirectory(directory))
        throw std::runtime_error("Cannot create energy plot directory: " + directory);
    return directory;
}

void draw(const Data& data, int energy, double zoomMax, const std::string& output, TFile& root) {
    const std::string plotOutput = energyDirectory(output, energy);
    const bool overview = energy == 0;
    const std::string overviewMethod = data.method == "truth_clustering" ? "truth_clustering" : "reco";
    const std::string scope = overview ? "overview" : std::to_string(energy) + "GeV";
    const std::string name = "jer_vs_epoch_" + scope + suffix(data) + (zoomMax > 0 ? "_zoom" : "");
    // Keep graph/legend objects alive until after the canvas has been saved/destroyed.
    std::vector<std::unique_ptr<TGraphErrors>> graphs;
    std::vector<std::unique_ptr<TLegend>> legends;
    TCanvas canvas(name.c_str(), name.c_str(), data.regions.size() == 2 ? 1500 : 900, 760);
    canvas.Divide(static_cast<int>(data.regions.size()), 1, 0.005, 0.005);
    for (std::size_t panel = 0; panel < data.regions.size(); ++panel) {
        canvas.cd(static_cast<int>(panel + 1));
        gPad->SetLeftMargin(0.13); gPad->SetRightMargin(0.04);
        gPad->SetBottomMargin(0.13); gPad->SetTopMargin(0.20);
        gPad->SetGridx(); gPad->SetGridy();
        std::vector<const std::vector<CsvRow>*> selected;
        for (const auto& item : data.series) {
            if (std::get<1>(item.first) != data.regions[panel]) continue;
            if (overview ? std::get<2>(item.first) != overviewMethod : std::get<0>(item.first) != energy) continue;
            selected.push_back(&item.second);
        }
        if (selected.empty()) {
            label(0.15, 0.6, "No valid complete data for " + data.regions[panel], 0.04);
            continue;
        }
        double lowX = selected.front()->front().epoch, highX = lowX, highY = 0;
        std::size_t clipped = 0;
        for (const auto* rows : selected) for (const CsvRow& row : *rows) {
            lowX = std::min(lowX, static_cast<double>(row.epoch));
            highX = std::max(highX, static_cast<double>(row.epoch));
            highY = std::max(highY, row.resolution + row.error);
            if (zoomMax > 0 && row.resolution > zoomMax) ++clipped;
        }
        const double padding = std::max(0.5, (highX - lowX) * 0.025);
        const double upperY = zoomMax > 0 ? zoomMax : std::max(1.0, highY * 1.15);
        TH1F* frame = gPad->DrawFrame(lowX - padding, 0, highX + padding, upperY);
        frame->SetTitle(""); frame->GetXaxis()->SetTitle("Epoch");
        frame->GetYaxis()->SetTitle("JER [%]");
        frame->GetXaxis()->SetTitleSize(0.044); frame->GetYaxis()->SetTitleSize(0.044);
        frame->GetXaxis()->SetLabelSize(0.035); frame->GetYaxis()->SetLabelSize(0.035);
        frame->GetYaxis()->SetTitleOffset(1.25);
        frame->GetXaxis()->SetNdivisions(510); frame->GetXaxis()->SetNoExponent();
        frame->GetXaxis()->SetDecimals(false);
        std::unique_ptr<TLegend> legend(new TLegend(0.43, 0.77 - 0.047 * selected.size(), 0.94, 0.77));
        legend->SetBorderSize(0); legend->SetFillStyle(1001); legend->SetFillColor(kWhite);
        legend->SetTextFont(42); legend->SetTextSize(0.027);
        for (const auto* rows : selected) {
            const CsvRow& first = rows->front();
            const bool reco = first.method == "reco";
            const int color = overview ? energyColor(first.energy) : (reco ? kAzure + 2 : kOrange + 7);
            const int style = reco ? 1 : 2;
            auto all = graph(*rows, 0, rows->size(), color, style);
            const CsvRow& best = minimum(*rows);
            std::string entry = overview ? std::to_string(first.energy) + " GeV" : (reco ? "GNN reco" : "Truth clustering");
            const bool markMinimum = overview || reco || data.method == "truth_clustering";
            if (markMinimum) entry += "; min e" + std::to_string(best.epoch) + " = " + number(best.resolution) + "%";
            legend->AddEntry(all.get(), entry.c_str(), "lep");
            graphs.push_back(std::move(all));
            // Draw only consecutive-epoch segments; missing epochs never get bridged.
            for (std::size_t begin = 0; begin < rows->size();) {
                std::size_t end = begin + 1;
                while (end < rows->size() && (*rows)[end].epoch == (*rows)[end - 1].epoch + 1) ++end;
                auto segment = graph(*rows, begin, end, color, style);
                segment->Draw("LPZ SAME");
                graphs.push_back(std::move(segment));
                begin = end;
            }
            if (markMinimum && (zoomMax == 0 || best.resolution <= zoomMax)) {
                std::unique_ptr<TGraphErrors> star(new TGraphErrors(1));
                star->SetPoint(0, best.epoch, best.resolution);
                star->SetMarkerStyle(29); star->SetMarkerSize(2.0); star->SetMarkerColor(color);
                star->Draw("P SAME"); graphs.push_back(std::move(star));
            }
        }
        legend->Draw(); legends.push_back(std::move(legend));
        const CsvRow& first = selected.front()->front();
        label(0.13, 0.965, data.model + " / " + data.run, 0.026);
        label(0.13, 0.918, (overview ? "Energy comparison (" + overviewMethod + ")" : std::to_string(energy) + " GeV sample")
              + " | " + data.regions[panel] + ": " + number(first.low, 3)
              + (first.region == "forward" ? " < a < " : " #leq a < ") + number(first.high, 3), 0.029);
        label(0.13, 0.875, "Readout: " + data.readout + "; #beta cut = " + number(data.beta)
              + (overview ? "" : "; N = " + std::to_string(first.events)), 0.027);
        label(0.13, 0.828, "JER = 100 #sqrt{2} RMS_{90}(E_{reco}) / Mean_{90}(E_{reco});  a = #LT|cos#theta_{q}|#GT_{E}", 0.024);
        label(0.14, 0.028, zoomMax > 0 ? "Zoom: " + std::to_string(clipped) + " points above range; see full plot"
              : "Stars: minima in this scan; error bars: CSV error_percent", 0.023);
    }
    canvas.Update();
    root.cd(); canvas.Write(name.c_str());
    for (const std::string& extension : {"png", "pdf"}) {
        const std::string path = plotOutput + "/" + name + "." + extension;
        canvas.SaveAs(path.c_str());
        if (!isFile(path)) throw std::runtime_error("Could not save plot: " + path);
    }
}

// Find this run's rank-0 epoch summaries, never a log from another run.
std::string discoverTrainingLog(const std::string& repo, const std::string& run) {
    const std::string directory = repo + "/shell/tmp";
    void* handle = gSystem->OpenDirectory(directory.c_str());
    if (!handle) return "";
    std::vector<std::string> candidates;
    while (const char* entry = gSystem->GetDirEntry(handle)) {
        const std::string name(entry);
        if (name.find(run) == std::string::npos) continue;
        const std::string log = directory + "/" + name + "/rank0.log";
        if (isFile(log)) candidates.push_back(absolute(log));
    }
    gSystem->FreeDirectory(handle);
    if (candidates.size() > 1)
        throw std::runtime_error("Multiple training logs match run " + run + "; specify argument 7 explicitly: "
                                 + join(candidates, ", "));
    return candidates.empty() ? "" : candidates.front();
}

LossHistory trainingLoss(const Data& data, const std::string& macroDirectory,
                         const char* requested, int betaStart, int energyStart) {
    if (betaStart < -2 || energyStart < -2)
        throw std::runtime_error("Loss start epochs must be >=0, -1 (unknown), or -2 (auto)");
    const std::string choice = requested ? requested : "";
    LossHistory result;
    if (choice == "none") return result;
    const bool automatic = choice.empty() || choice == "auto";
    const std::string source = automatic ? discoverTrainingLog(parent(macroDirectory), data.run) : absolute(choice);
    if (source.empty()) {
        std::cerr << "[loss] No rank0.log found for run " << data.run
                  << "; JER plots only. Supply argument 7 for loss plots.\n";
        return result;
    }
    // An explicit renamed log is allowed; a different timestamp in its path is
    // rejected rather than silently pairing checkpoints with a different run.
    const std::regex runPattern("[0-9]{4}_[0-9]{2}_[0-9]{2}_[0-9]{6}");
    std::smatch match;
    if (std::regex_search(source, match, runPattern) && match.str() != data.run)
        throw std::runtime_error("Training log timestamp differs from JER run: " + source);
    result = readTrainingLoss(source);
    const bool known = data.model == "5head_pid_onehot" && data.run == "2026_10_09_154602";
    result.betaStart = betaStart == -2 ? (known ? 2 : -1) : betaStart;
    result.energyStart = energyStart == -2 ? (known ? 4 : -1) : energyStart;
    result.activationSource = known && betaStart == -2 && energyStart == -2
        ? "Recorded 2026_10_09_154602 schedule: beta epoch>1, energy epoch>3"
        : (known ? "Recorded schedule with explicit argument overrides; -1 means unknown"
                 : "Explicit activation epochs; -1 means unknown");
    if (!known && betaStart == -2 && energyStart == -2)
        result.activationSource = "Unknown activation schedule: specify beta/energy starts in arguments 8/9";
    for (const auto& warning : result.warnings) std::cerr << "[loss] " << warning << '\n';
    std::set<int> matched;
    for (const auto& series : data.series) for (const auto& row : series.second)
        if (result.epochs.count(row.epoch)) matched.insert(row.epoch);
    if (matched.empty()) throw std::runtime_error("No common epochs between training log and selected JER: " + source);
    std::cout << "[loss] Matched " << matched.size() << " checkpoint epochs to JER\n";
    std::cout << "[loss] " << result.source << ": " << result.epochs.size()
              << " complete epochs; beta starts=" << result.betaStart
              << ", energy starts=" << result.energyStart << '\n';
    return result;
}

} // namespace jer_epoch_plot

#include "src/jer_loss_plots.h"
#include "src/jer_energy_loss_plots.h"

void plot_jer_vs_epoch(
    const char* input = "",
    const char* outputDirectory = "",
    const char* readout = "saved",
    const char* method = "both",
    const char* region = "both",
    double zoomMax = 15.0,
    const char* trainingLog = "",
    int betaStart = -2,
    int energyStart = -2
) {
    using namespace jer_epoch_plot;
    if (!std::isfinite(zoomMax) || zoomMax < 0) throw std::runtime_error("zoom upper JER must be nonnegative and finite");
    const std::string macroDirectory = parent(absolute(__FILE__));
    const std::string source = absolute(input[0] ? input : (macroDirectory + "/../output/jer_epoch_scan/5head_pid_onehot_2026_10_09_154602").c_str());
    const Data data = readData(source, readout, method, region);
    const LossHistory losses = trainingLoss(data, macroDirectory, trainingLog, betaStart, energyStart);
    const std::string output = absolute(outputDirectory[0] ? outputDirectory
        : (macroDirectory + "/figures/jer_epoch_scan/" + safeName(data.model + "_" + data.run)).c_str());
    if (gSystem->mkdir(output.c_str(), true) != 0 && !isDirectory(output))
        throw std::runtime_error("Cannot create plot output directory: " + output);
    std::vector<std::string> energyLabels;
    for (int energy : data.energies) energyLabels.push_back(std::to_string(energy));
    const std::string rootPath = output + "/jer_vs_epoch_" + join(energyLabels, "-") + "GeV" + suffix(data) + ".root";
    TFile root(rootPath.c_str(), "RECREATE");
    if (root.IsZombie()) throw std::runtime_error("Cannot create plot ROOT: " + rootPath);
    TNamed("source_csv_files", join(data.sources, "\n").c_str()).Write();
    TNamed("model", data.model.c_str()).Write(); TNamed("run_id", data.run.c_str()).Write();
    TNamed("energy_readout", data.readout.c_str()).Write();
    TNamed("minimum_definition", "Minimum observed JER in the selected valid, complete scan; not a statistically independent checkpoint selection").Write();
    TNamed("angular_coordinate", "a = upstream energy-weighted quark abs(pz)/p; bounds from CSV low/high").Write();
    TNamed("metric_definition", "100*sqrt(2)*RMS90(Ereco_event)/Mean90(Ereco_event); values and errors copied from CSV").Write();
    TNamed("zoom_upper_percent", number(zoomMax, 6).c_str()).Write();
    for (const auto& item : data.series) {
        const auto& rows = item.second;
        const CsvRow& first = rows.front();
        const std::string key = std::to_string(first.energy) + "GeV_" + first.region + "_" + first.method;
        auto points = graph(rows, 0, rows.size(), energyColor(first.energy), first.method == "reco" ? 1 : 2);
        points->SetName(("points_" + key).c_str());
        points->SetTitle((key + ";Epoch;JER [%]").c_str()); points->Write();
        TNamed(("event_selection_" + key).c_str(), first.digest.c_str()).Write();
    }
    for (int energy : data.energies) {
        draw(data, energy, 0, output, root);
        if (zoomMax > 0) draw(data, energy, zoomMax, output, root);
    }
    if (data.energies.size() > 1) {
        draw(data, 0, 0, output, root);
        if (zoomMax > 0) draw(data, 0, zoomMax, output, root);
    }
    if (!losses.epochs.empty()) {
        drawLossPlots(data, losses, zoomMax, output, root);
        drawEnergyLossPlots(data, losses, zoomMax, output, root);
    }
    root.Close();
    std::cout << "Saved epoch JER plots: " << output << "\nSaved graphs and canvases: " << rootPath << '\n';
}
