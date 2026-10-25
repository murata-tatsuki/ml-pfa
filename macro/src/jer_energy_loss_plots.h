#ifndef JER_ENERGY_LOSS_PLOTS_H
#define JER_ENERGY_LOSS_PLOTS_H

// Included after jer_loss_plots.h: reuse its strict epoch matching and drawing.
#include "jer_loss_correlations.h"
#include <TH2D.h>
#include <TTree.h>

#include <array>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace jer_epoch_plot {
namespace energy_loss_plot {

using Matches = loss_plot::Matches;

const std::array<std::string, 5>& components() {
    static const std::array<std::string, 5> names{{"L_E", "L_E_cond", "L_E_cluster", "L_E_photon", "L_E_neutral_hadron"}};
    return names;
}

std::string title(const std::string& component) {
    if (component == "L_E") return "All energy (L_{E})";
    if (component == "L_E_cond") return "Charged (L_{E,cond})";
    if (component == "L_E_cluster") return "Neutral (L_{E,cluster})";
    if (component == "L_E_photon") return "Photon (L_{E,#gamma})";
    return "Neutral hadron (L_{E,nh})";
}

std::string methodTitle(const std::string& method) {
    return method == "reco" ? "GNN reco" : "Truth clustering";
}

struct Correlation {
    int energy = 0, firstEpoch = -1, lastEpoch = -1;
    std::string region, method, readout, component, window;
    long long n = 0;
    double rho = std::numeric_limits<double>::quiet_NaN();
};

Correlation correlation(const Matches& rows, int energy, const std::string& region,
                        const std::string& method, const std::string& readout,
                        const std::string& component, int minimumEpoch, const std::string& window) {
    Correlation result;
    result.energy = energy; result.region = region; result.method = method;
    result.readout = readout; result.component = component; result.window = window;
    std::vector<double> x, y;
    for (const auto& row : rows) {
        if (row.epoch < minimumEpoch) continue;
        if (!std::isfinite(row.loss) || !std::isfinite(row.jer)) continue;
        if (result.n == 0) result.firstEpoch = result.lastEpoch = row.epoch;
        result.firstEpoch = std::min(result.firstEpoch, row.epoch);
        result.lastEpoch = std::max(result.lastEpoch, row.epoch);
        ++result.n; x.push_back(row.loss); y.push_back(row.jer);
    }
    result.rho = loss_correlation::spearman(x, y);
    return result;
}

std::string rhoText(const Correlation& value) {
    return std::isfinite(value.rho) ? number(value.rho, 3) : "undefined";
}

Matches matched(const Data& data, const LossHistory& losses, int energy,
                const std::string& region, const std::string& method, const std::string& component) {
    const auto found = data.series.find(Key(energy, region, method));
    return found == data.series.end() ? Matches() : loss_plot::match(found->second, losses, component);
}

std::string identity(int energy, const std::string& region, const std::string& method,
                     const std::string& readout, const std::string& component) {
    return std::to_string(energy) + "GeV_" + region + "_" + method + "_" + readout + "_" + component;
}

void savePoints(const Matches& rows, const std::string& key, TFile& root) {
    root.cd();
    auto graph = loss_plot::scatterGraph(rows, 20);
    graph->SetName(("energy_loss_jer_" + key).c_str());
    graph->SetTitle((key + ";Validation energy-loss contribution;JER [%]").c_str());
    graph->Write();
    TGraph epochs(static_cast<int>(rows.size()));
    epochs.SetName(("energy_epoch_for_point_" + key).c_str());
    epochs.SetTitle("Scatter point index to checkpoint epoch;Point index;Epoch");
    for (std::size_t i = 0; i < rows.size(); ++i) epochs.SetPoint(static_cast<int>(i), i, rows[i].epoch);
    epochs.Write();
}

void scatter(const Data& data, const LossHistory& losses, int energy, const std::string& region,
             const std::string& method, double zoomMax, const std::string& output, TFile& root) {
    const std::string name = "jer_vs_energy_loss_" + std::to_string(energy) + "GeV_" + region + "_"
        + method + "_" + data.readout + (zoomMax > 0 ? "_zoom" : "");
    loss_plot::Objects objects;
    TCanvas canvas(name.c_str(), name.c_str(), 1800, 1100); canvas.Divide(3, 2, 0.005, 0.005);
    int first = std::numeric_limits<int>::max(), last = std::numeric_limits<int>::min();
    std::array<Matches, 5> all;
    for (std::size_t i = 0; i < components().size(); ++i) {
        all[i] = matched(data, losses, energy, region, method, components()[i]);
        for (const auto& row : all[i]) { first = std::min(first, row.epoch); last = std::max(last, row.epoch); }
    }
    const bool any = first <= last;
    if (!any) { first = 0; last = 1; }
    for (std::size_t panel = 0; panel < components().size(); ++panel) {
        const auto& component = components()[panel]; const auto& rows = all[panel];
        canvas.cd(static_cast<int>(panel + 1)); loss_plot::setupPad(0.25);
        double lowX = std::numeric_limits<double>::infinity(), highX = -lowX, highY = 0;
        std::size_t clipped = 0;
        for (const auto& row : rows) {
            lowX = std::min(lowX, row.loss); highX = std::max(highX, row.loss);
            highY = std::max(highY, row.jer + row.error);
            if (zoomMax > 0 && row.jer > zoomMax) ++clipped;
        }
        if (rows.empty()) { lowX = 0; highX = 1; }
        const double padding = std::max(0.001, (highX - lowX) * 0.10);
        auto* frame = gPad->DrawFrame(lowX - padding, 0, highX + padding,
                                     zoomMax > 0 ? zoomMax : std::max(1., highY * 1.15));
        loss_plot::axes(frame, "Validation " + title(component), "JER [%]");
        const auto full = correlation(rows, energy, region, method, data.readout, component, -1, "all_active");
        const auto late = correlation(rows, energy, region, method, data.readout, component, 60, "epoch_ge60");
        label(0.14, 0.957, std::to_string(energy) + " GeV | " + region + " | " + methodTitle(method), 0.031);
        label(0.14, 0.900, title(component), 0.036);
        label(0.14, 0.845, "Spearman #rho = " + rhoText(full) + "   N = " + std::to_string(full.n), 0.031);
        if (late.n >= 3)
            label(0.14, 0.790, "epoch #geq 60: #rho = " + rhoText(late) + "   N = " + std::to_string(late.n), 0.027);
        if (zoomMax > 0)
            label(0.15, 0.035, "Zoom: " + std::to_string(clipped) + " points above range; #rho uses ALL matched points", 0.023);
        else label(0.15, 0.035, "#rho uses all matched finite points; missing values omitted", 0.023);
        if (rows.empty()) label(0.20, 0.46, "Component unavailable / no matched epochs", 0.036);
        auto graph = loss_plot::scatterGraph(rows, method == "reco" ? 20 : 24);
        graph->Draw("PZ SAME"); objects.graphs.push_back(std::move(graph));
        for (const auto& row : rows) {
            std::unique_ptr<TMarker> marker(new TMarker(row.loss, row.jer, method == "reco" ? 20 : 24));
            marker->SetMarkerColor(loss_plot::epochColor(row.epoch, first, last));
            marker->SetMarkerSize(method == "reco" ? 0.75 : 0.95); marker->Draw(); objects.markers.push_back(std::move(marker));
        }
        if (zoomMax == 0) savePoints(rows, identity(energy, region, method, data.readout, component), root);
    }
    canvas.cd(6); gPad->Range(0, 0, 1, 1);
    label(0.07, 0.95, data.model + " / " + data.run, 0.034);
    label(0.07, 0.87, methodTitle(method) + " | readout: " + data.readout, 0.041);
    label(0.07, 0.79, loss_plot::activeNote(losses), 0.030);
    label(0.07, 0.72, "Charged = charged hadron + electron + muon", 0.030);
    label(0.07, 0.66, "Neutral = photon + neutral hadron", 0.030);
    label(0.07, 0.60, "Species contributions share the total denominator", 0.028);
    label(0.07, 0.54, "They are contributions, not class-wise means", 0.028);
    label(0.07, 0.47, "Gray error bars: JER error from metrics.csv", 0.028);
    label(0.07, 0.41, "Positive #rho: lower loss accompanies lower JER", 0.028);
    if (any) {
        for (int i = 0; i < 128; ++i) {
            const double y1 = 0.12 + 0.23 * i / 128., y2 = 0.12 + 0.23 * (i + 1) / 128.;
            std::unique_ptr<TBox> box(new TBox(0.76, y1, 0.81, y2));
            const int epoch = first + static_cast<int>(std::round((last - first) * i / 127.));
            box->SetFillColor(loss_plot::epochColor(epoch, first, last)); box->SetLineColor(box->GetFillColor());
            box->Draw(); objects.boxes.push_back(std::move(box));
        }
        label(0.70, 0.37, "Epoch", 0.031);
        label(0.83, 0.12, std::to_string(first), 0.030);
        label(0.83, 0.33, std::to_string(last), 0.030);
    }
    label(0.07, 0.26, "Separate plot for each method", 0.031);
    label(0.07, 0.19, "No causal conclusion from correlation alone", 0.027);
    label(0.07, 0.05, "Undefined #rho: fewer than 3 points or constant values", 0.026);
    loss_plot::save(canvas, name, energyDirectory(output, energy), root);
}

void saveCorrelations(const Data& data, const std::vector<Correlation>& records,
                      const std::string& output, TFile& root) {
    const std::string path = output + "/energy_loss_correlations" + suffix(data) + ".csv";
    std::ofstream csv(path.c_str());
    if (!csv) throw std::runtime_error("Cannot create energy loss correlations CSV: " + path);
    csv << "energy,region,method,readout,component,window,first_epoch,last_epoch,n,spearman_rho,defined\n";
    root.cd();
    TTree tree("energy_loss_correlations", "Unweighted Spearman correlation across matched checkpoint epochs");
    int energy = 0, firstEpoch = -1, lastEpoch = -1;
    long long n = 0; double rho = 0; bool defined = false;
    std::string region, method, readout, component, window;
    tree.Branch("energy", &energy); tree.Branch("region", &region); tree.Branch("method", &method);
    tree.Branch("readout", &readout); tree.Branch("component", &component); tree.Branch("window", &window);
    tree.Branch("first_epoch", &firstEpoch); tree.Branch("last_epoch", &lastEpoch);
    tree.Branch("n", &n); tree.Branch("spearman_rho", &rho); tree.Branch("defined", &defined);
    csv << std::setprecision(17);
    for (const auto& row : records) {
        energy = row.energy; region = row.region; method = row.method; readout = row.readout;
        component = row.component; window = row.window; firstEpoch = row.firstEpoch; lastEpoch = row.lastEpoch;
        n = row.n; rho = row.rho; defined = std::isfinite(rho);
        csv << energy << ',' << region << ',' << method << ',' << readout << ',' << component << ',' << window << ','
            << firstEpoch << ',' << lastEpoch << ',' << n << ',' << rho << ',' << (defined ? "true" : "false") << '\n';
        tree.Fill();
    }
    csv.close();
    if (!csv) throw std::runtime_error("Failed writing energy loss correlations CSV: " + path);
    tree.Write();
}

int correlationColor(double rho) {
    if (!std::isfinite(rho)) return kGray;
    const double strength = std::min(1., std::abs(rho));
    // White at zero; blue at -1, red at +1. No global palette mutation.
    const double r = rho < 0 ? 1. - .78 * strength : 1. - .22 * strength;
    const double g = 1. - .70 * strength;
    const double b = rho < 0 ? 1. - .18 * strength : 1. - .78 * strength;
    return TColor::GetColor(static_cast<Float_t>(r), static_cast<Float_t>(g), static_cast<Float_t>(b));
}

void overview(const Data& data, const LossHistory& losses, const std::vector<Correlation>& records,
              const std::string& output, TFile& root) {
    const std::string name = "energy_loss_correlation_overview" + suffix(data);
    loss_plot::Objects objects;
    std::vector<std::unique_ptr<TH2D>> histograms;
    std::vector<std::pair<int, std::string>> columns;
    for (int energy : data.energies) for (const auto& region : data.regions) columns.push_back({energy, region});
    // Keep the cell width readable as more energy samples are collected.
    const int panelWidth = (data.methods.size() == 2 ? 925 : 1000)
        * std::max(4, static_cast<int>(columns.size())) / 4;
    TCanvas canvas(name.c_str(), name.c_str(), panelWidth * static_cast<int>(data.methods.size()), 900);
    canvas.Divide(static_cast<int>(data.methods.size()), 1, .005, .005);
    for (std::size_t panel = 0; panel < data.methods.size(); ++panel) {
        const auto& method = data.methods[panel]; canvas.cd(static_cast<int>(panel + 1));
        gPad->SetLeftMargin(.24); gPad->SetRightMargin(.13); gPad->SetBottomMargin(.17); gPad->SetTopMargin(.22);
        const std::string histName = "energy_loss_correlation_matrix_" + method;
        std::unique_ptr<TH2D> matrix(new TH2D(histName.c_str(), "", static_cast<int>(columns.size()), 0., columns.size(), 5, 0., 5.));
        matrix->SetDirectory(nullptr); matrix->SetStats(false); matrix->SetMinimum(-1); matrix->SetMaximum(1);
        for (std::size_t i = 0; i < columns.size(); ++i)
            matrix->GetXaxis()->SetBinLabel(static_cast<int>(i + 1), (std::to_string(columns[i].first) + " GeV / " + columns[i].second).c_str());
        for (std::size_t i = 0; i < components().size(); ++i)
            matrix->GetYaxis()->SetBinLabel(static_cast<int>(5 - i), title(components()[i]).c_str());
        matrix->GetXaxis()->SetLabelSize(.032); matrix->GetYaxis()->SetLabelSize(.036);
        matrix->GetXaxis()->SetTickLength(0); matrix->GetYaxis()->SetTickLength(0);
        matrix->Draw("AXIS");
        for (std::size_t row = 0; row < components().size(); ++row) for (std::size_t col = 0; col < columns.size(); ++col) {
            const Correlation* found = nullptr;
            for (const auto& record : records)
                if (record.energy == columns[col].first && record.region == columns[col].second && record.method == method
                    && record.component == components()[row] && record.window == "all_active") { found = &record; break; }
            const double rho = found ? found->rho : std::numeric_limits<double>::quiet_NaN();
            const double bottom = 4. - row;
            matrix->SetBinContent(static_cast<int>(col + 1), static_cast<int>(5 - row), rho);
            std::unique_ptr<TBox> box(new TBox(col, bottom, col + 1., bottom + 1.));
            box->SetFillColor(correlationColor(rho)); box->SetLineColor(kWhite); box->Draw(); objects.boxes.push_back(std::move(box));
            TLatex text; text.SetTextAlign(22); text.SetTextFont(42); text.SetTextSize(.046);
            text.DrawLatex(col + .5, bottom + .62, (std::isfinite(rho) ? number(rho, 3) : "undefined").c_str());
            text.SetTextSize(.030); text.DrawLatex(col + .5, bottom + .29, ("N=" + std::to_string(found ? found->n : 0)).c_str());
        }
        const double legendX = columns.size() + .09 * columns.size(), legendWidth = .035 * columns.size();
        for (int i = 0; i < 128; ++i) {
            const double low = 5. * i / 128., high = 5. * (i + 1) / 128.;
            std::unique_ptr<TBox> box(new TBox(legendX, low, legendX + legendWidth, high));
            box->SetFillColor(correlationColor(-1. + 2. * i / 127.)); box->SetLineColor(box->GetFillColor());
            box->Draw(); objects.boxes.push_back(std::move(box));
        }
        label(.875, .17, "-1", .028); label(.875, .46, "0", .028); label(.875, .74, "+1", .028);
        label(.24, .958, data.model + " / " + data.run, .029);
        label(.24, .91, methodTitle(method) + " | " + data.readout + " | Spearman #rho", .035);
        label(.24, .862, loss_plot::activeNote(losses), .026);
        label(.24, .812, "Red: positive; blue: negative; gray: undefined", .025);
        label(.24, .08, "Full matched data; fixed color range [-1, 1]", .026);
        label(.24, .035, "Species terms are contributions with a common denominator", .024);
        root.cd(); matrix->Write(); histograms.push_back(std::move(matrix));
    }
    loss_plot::save(canvas, name, output, root);
}

double epochStd(const loss_plot::Points& values) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    double mean = 0; for (const auto& row : values) mean += row.value; mean /= values.size();
    double variance = 0; for (const auto& row : values) variance += (row.value - mean) * (row.value - mean);
    return std::sqrt(variance / values.size());
}

void componentHistory(const Data& data, const LossHistory& losses, const std::string& output, TFile& root) {
    const std::string name = "energy_loss_components_vs_epoch" + suffix(data);
    const int start = loss_plot::activeStart(losses);
    std::array<std::array<loss_plot::Points, 3>, 2> series;
    double low = std::numeric_limits<double>::infinity(), high = -low;
    const std::array<std::string, 3> terms{{"L_E", "L_E_cond", "L_E_cluster"}};
    for (int panel = 0; panel < 2; ++panel) for (std::size_t i = 0; i < terms.size(); ++i) {
        auto& rows = series[panel][i]; rows = loss_plot::points(losses, terms[i], panel == 1);
        rows.erase(std::remove_if(rows.begin(), rows.end(), [start](const loss_plot::Point& row) {
            return start >= 0 && row.epoch < start;
        }), rows.end());
        for (const auto& row : rows) { low = std::min(low, row.value); high = std::max(high, row.value); }
    }
    if (!std::isfinite(low)) { low = 0; high = 1; }
    const double pad = std::max(.02, (high - low) * .12);
    const auto x = loss_plot::epochRange(data, losses);
    loss_plot::Objects objects;
    TCanvas canvas(name.c_str(), name.c_str(), 1700, 800); canvas.Divide(2, 1, .005, .005);
    const int colors[3] = {kBlack, kAzure + 2, kOrange + 7};
    for (int panel = 0; panel < 2; ++panel) {
        canvas.cd(panel + 1); loss_plot::setupPad(.23);
        auto* frame = gPad->DrawFrame(x.first, std::max(0., low - pad), x.second, high + pad);
        loss_plot::axes(frame, "Epoch", "Energy-loss contribution (same scale)");
        std::unique_ptr<TLegend> legend(new TLegend(.39, .57, .94, .76));
        legend->SetBorderSize(0); legend->SetFillColor(kWhite); legend->SetTextSize(.029);
        for (std::size_t i = 0; i < terms.size(); ++i) {
            auto* graph = loss_plot::drawLoss(series[panel][i], colors[i], 1, objects);
            if (graph) legend->AddEntry(graph, (title(terms[i]) + "; #sigma_{epochs}=" + number(epochStd(series[panel][i]), 4)).c_str(), "lp");
        }
        legend->Draw(); objects.legends.push_back(std::move(legend));
        label(.14, .955, data.model + " / " + data.run, .029);
        label(.14, .90, std::string(panel == 0 ? "Train" : "Validation") + " | total / charged / neutral energy loss", .033);
        label(.14, .845, loss_plot::activeNote(losses), .027);
        label(.15, .04, "Same y scale in both panels; #sigma is variation over epochs", .025);
    }
    loss_plot::save(canvas, name, output, root);
}

} // namespace energy_loss_plot

void drawEnergyLossPlots(const Data& data, const LossHistory& losses, double zoomMax,
                         const std::string& output, TFile& root) {
    if (losses.epochs.empty()) return;
    const std::array<std::string, 5> species{{"L_E_charged_hadron", "L_E_electron", "L_E_muon", "L_E_photon", "L_E_neutral_hadron"}};
    bool fiveHeadMetrics = false;
    for (const auto& epoch : losses.epochs) {
        bool complete = true;
        for (const auto& name : species) {
            const auto value = epoch.second.validation.find(name);
            complete = complete && value != epoch.second.validation.end() && std::isfinite(value->second);
        }
        if (complete) { fiveHeadMetrics = true; break; }
    }
    if (!fiveHeadMetrics) {
        std::cerr << "[energy loss] No complete five-species validation metrics; skip charged/neutral plots (existing loss/JER plots retained)\n";
        return;
    }
    root.cd();
    TNamed("energy_loss_component_definition", "Five-head convention: L_E_cond=charged hadron+electron+muon; L_E_cluster=photon+neutral_hadron; species losses are additive contributions using the common all-object denominator, not separate class means").Write();
    TNamed("energy_loss_correlation_definition", "Unweighted Spearman across matched checkpoint epochs; averaged ranks for ties; fewer than 3 points or constant values yield undefined; full and zoom use identical all-matched data; no causal inference or p-value claimed").Write();
    for (const auto& component : energy_loss_plot::components()) for (bool validation : {false, true}) {
        const auto rows = loss_plot::points(losses, component, validation);
        auto graph = loss_plot::lossGraph(rows, 0, rows.size(), validation ? kOrange + 7 : kAzure + 2, 1);
        const std::string key = std::string(validation ? "validation_" : "train_") + component;
        graph->SetName(("energy_loss_vs_epoch_" + key).c_str());
        graph->SetTitle((key + ";Epoch;Energy-loss contribution").c_str()); graph->Write();
    }
    std::vector<energy_loss_plot::Correlation> records;
    for (int energy : data.energies) for (const auto& region : data.regions) for (const auto& method : data.methods) {
        for (const auto& component : energy_loss_plot::components()) {
            const auto rows = energy_loss_plot::matched(data, losses, energy, region, method, component);
            records.push_back(energy_loss_plot::correlation(rows, energy, region, method, data.readout, component, -1, "all_active"));
            records.push_back(energy_loss_plot::correlation(rows, energy, region, method, data.readout, component, 60, "epoch_ge60"));
        }
        energy_loss_plot::scatter(data, losses, energy, region, method, 0, output, root);
        if (zoomMax > 0) energy_loss_plot::scatter(data, losses, energy, region, method, zoomMax, output, root);
    }
    energy_loss_plot::saveCorrelations(data, records, output, root);
    energy_loss_plot::overview(data, losses, records, output, root);
    energy_loss_plot::componentHistory(data, losses, output, root);
}

} // namespace jer_epoch_plot
#endif
