#ifndef JER_LOSS_PLOTS_H
#define JER_LOSS_PLOTS_H

// Included after Data and the training-log reader by plot_jer_vs_epoch.cxx.
#include <TBox.h>
#include <TColor.h>
#include <TGraph.h>
#include <TLine.h>
#include <TMarker.h>

#include <array>
#include <limits>

namespace jer_epoch_plot {
namespace loss_plot {

struct Point { int epoch; double value; };
using Points = std::vector<Point>;

// Declare the objects before a canvas: pads must be destroyed first.
struct Objects {
    std::vector<std::unique_ptr<TGraphErrors>> graphs;
    std::vector<std::unique_ptr<TLegend>> legends;
    std::vector<std::unique_ptr<TLine>> lines;
    std::vector<std::unique_ptr<TMarker>> markers;
    std::vector<std::unique_ptr<TBox>> boxes;
};

const std::array<std::string, 5>& components() {
    static const std::array<std::string, 5> names{{"total", "L_E", "L_V", "L_beta", "L_PID"}};
    return names;
}

std::string componentTitle(const std::string& component) {
    if (component == "total") return "Total loss";
    if (component == "L_beta") return "L_{#beta}";
    if (component == "L_PID") return "L_{PID}";
    if (component == "L_E") return "L_{E}";
    return "L_{V}";
}

bool value(const LossEpoch& row, const std::string& component, bool validation, double& result) {
    if (component == "total") result = validation ? row.validationTotal : row.trainTotal;
    else {
        const auto& values = validation ? row.validation : row.train;
        const auto found = values.find(component);
        if (found == values.end()) return false;
        result = found->second;
    }
    return std::isfinite(result);
}

int activeStart(const LossHistory& losses) {
    return losses.betaStart >= 0 && losses.energyStart >= 0
        ? std::max(losses.betaStart, losses.energyStart) : -1;
}

std::string activeNote(const LossHistory& losses) {
    const int start = activeStart(losses);
    return start >= 0 ? "Loss comparison: epoch #geq " + std::to_string(start) + " (all loss terms active)"
        : "Activation boundary unknown: all matched epochs";
}

Points points(const LossHistory& losses, const std::string& component, bool validation) {
    Points result;
    for (const auto& item : losses.epochs) {
        double loss = 0;
        if (value(item.second, component, validation, loss)) result.push_back({item.first, loss});
    }
    return result;
}

std::unique_ptr<TGraphErrors> lossGraph(const Points& rows, std::size_t begin, std::size_t end,
                                     int color, int style) {
    std::unique_ptr<TGraphErrors> result(new TGraphErrors(static_cast<int>(end - begin)));
    for (std::size_t i = begin; i < end; ++i) {
        result->SetPoint(static_cast<int>(i - begin), rows[i].epoch, rows[i].value);
        result->SetPointError(static_cast<int>(i - begin), 0, 0);
    }
    result->SetLineColor(color); result->SetMarkerColor(color);
    result->SetLineWidth(2); result->SetLineStyle(style);
    result->SetMarkerStyle(style == 1 ? 20 : 24); result->SetMarkerSize(0.45);
    return result;
}

TGraphErrors* drawLoss(const Points& rows, int color, int style, Objects& objects) {
    if (rows.empty()) return nullptr;
    auto full = lossGraph(rows, 0, rows.size(), color, style);
    TGraphErrors* handle = full.get();
    objects.graphs.push_back(std::move(full));
    for (std::size_t begin = 0; begin < rows.size();) {
        std::size_t end = begin + 1;
        while (end < rows.size() && rows[end].epoch == rows[end - 1].epoch + 1) ++end;
        auto segment = lossGraph(rows, begin, end, color, style);
        segment->Draw("LPZ SAME"); objects.graphs.push_back(std::move(segment));
        begin = end;
    }
    return handle;
}

std::pair<double, double> range(const Points& train, const Points& validation) {
    double lower = std::numeric_limits<double>::infinity(), upper = -lower;
    for (const auto* rows : {&train, &validation}) for (const Point& row : *rows) {
        lower = std::min(lower, row.value); upper = std::max(upper, row.value);
    }
    if (!std::isfinite(lower)) return {0, 1};
    const double padding = std::max(0.02, (upper - lower) * 0.12);
    return {lower < 0 ? lower - padding : std::max(0., lower - padding), upper + padding};
}

void save(TCanvas& canvas, const std::string& name, const std::string& output, TFile& root) {
    canvas.Update(); root.cd(); canvas.Write(name.c_str());
    for (const std::string& extension : {"png", "pdf"}) {
        const std::string path = output + "/" + name + "." + extension;
        canvas.SaveAs(path.c_str());
        if (!isFile(path)) throw std::runtime_error("Could not save loss plot: " + path);
    }
}

void axes(TH1F* frame, const std::string& x, const std::string& y) {
    frame->SetTitle(""); frame->GetXaxis()->SetTitle(x.c_str()); frame->GetYaxis()->SetTitle(y.c_str());
    frame->GetXaxis()->SetTitleSize(0.045); frame->GetYaxis()->SetTitleSize(0.045);
    frame->GetXaxis()->SetLabelSize(0.036); frame->GetYaxis()->SetLabelSize(0.036);
    frame->GetYaxis()->SetTitleOffset(1.35);
    if (x == "Epoch") {
        frame->GetXaxis()->SetNdivisions(510); frame->GetXaxis()->SetNoExponent();
        frame->GetXaxis()->SetDecimals(false);
    }
}

void setupPad(double top = 0.18) {
    gPad->SetLeftMargin(0.14); gPad->SetRightMargin(0.045);
    gPad->SetBottomMargin(0.14); gPad->SetTopMargin(top);
    gPad->SetGridx(); gPad->SetGridy();
}

TLine* vertical(int epoch, double lower, double upper, int color, int style, Objects& objects) {
    std::unique_ptr<TLine> line(new TLine(epoch, lower, epoch, upper));
    line->SetLineColor(color); line->SetLineStyle(style); line->SetLineWidth(2);
    TLine* handle = line.get(); line->Draw(); objects.lines.push_back(std::move(line));
    return handle;
}

void activationLines(const LossHistory& losses, double lowX, double highX,
                     double lowY, double highY, Objects& objects) {
    for (int epoch : {losses.betaStart, losses.energyStart})
        if (epoch >= lowX && epoch <= highX) vertical(epoch, lowY, highY, kGray + 1, 3, objects);
}

std::string phaseNote(const LossHistory& losses) {
    std::vector<std::string> notes;
    if (losses.betaStart >= 0) notes.push_back("#beta starts e" + std::to_string(losses.betaStart));
    if (losses.energyStart >= 0) notes.push_back("E starts e" + std::to_string(losses.energyStart));
    return notes.empty() ? "Loss activation epochs: unknown" : "Gray dotted: " + join(notes, "; ");
}

const Point* validationMinimum(const Points& validation, const LossHistory& losses,
                               const std::set<int>* eligible = nullptr) {
    const Point* best = nullptr;
    const int start = activeStart(losses);
    for (const Point& row : validation)
        if ((start < 0 || row.epoch >= start) && (!eligible || eligible->count(row.epoch))
            && (!best || row.value < best->value)) best = &row;
    return best;
}

std::pair<double, double> epochRange(const Data& data, const LossHistory& losses, int energy = 0) {
    double low = std::numeric_limits<double>::infinity(), high = -low;
    for (const auto& item : losses.epochs) {
        low = std::min(low, static_cast<double>(item.first)); high = std::max(high, static_cast<double>(item.first));
    }
    for (const auto& item : data.series) {
        if (energy && std::get<0>(item.first) != energy) continue;
        for (const CsvRow& row : item.second) {
            low = std::min(low, static_cast<double>(row.epoch)); high = std::max(high, static_cast<double>(row.epoch));
        }
    }
    const double padding = std::max(0.5, (high - low) * 0.025);
    return {low - padding, high + padding};
}

void epochComparison(const Data& data, const LossHistory& losses, int energy,
                     double zoomMax, const std::string& output, TFile& root) {
    const std::string name = "jer_loss_vs_epoch_" + std::to_string(energy) + "GeV" + suffix(data)
        + (zoomMax > 0 ? "_zoom" : "");
    Objects objects;
    TCanvas canvas(name.c_str(), name.c_str(), data.regions.size() == 2 ? 1600 : 900, 1050);
    canvas.Divide(static_cast<int>(data.regions.size()), 2, 0.005, 0.005);
    const Points train = points(losses, "total", false), validation = points(losses, "total", true);
    const auto x = epochRange(data, losses, energy), yLoss = range(train, validation);
    for (std::size_t panel = 0; panel < data.regions.size(); ++panel) {
        const std::string& region = data.regions[panel];
        std::set<int> observed;
        for (const auto& item : data.series)
            if (std::get<0>(item.first) == energy && std::get<1>(item.first) == region)
                for (const CsvRow& row : item.second) observed.insert(row.epoch);
        const Point* bestLoss = validationMinimum(validation, losses, &observed);
        const auto reco = data.series.find(Key(energy, region, "reco"));
        const CsvRow* bestJer = reco == data.series.end() ? nullptr : &minimum(reco->second);
        canvas.cd(static_cast<int>(panel + 1)); setupPad(0.24);
        TH1F* frame = gPad->DrawFrame(x.first, yLoss.first, x.second, yLoss.second);
        axes(frame, "Epoch", "Total loss (linear)");
        activationLines(losses, x.first, x.second, yLoss.first, yLoss.second, objects);
        TLine* lossLine = bestLoss ? vertical(bestLoss->epoch, yLoss.first, yLoss.second, kGreen + 2, 2, objects) : nullptr;
        TLine* jerLine = bestJer ? vertical(bestJer->epoch, yLoss.first, yLoss.second, kMagenta + 1, 7, objects) : nullptr;
        TGraphErrors* trainGraph = drawLoss(train, kAzure + 2, 1, objects);
        TGraphErrors* validationGraph = drawLoss(validation, kOrange + 7, 1, objects);
        std::unique_ptr<TLegend> legend(new TLegend(0.45, 0.52, 0.94, 0.75));
        legend->SetBorderSize(0); legend->SetFillColor(kWhite); legend->SetTextSize(0.029);
        if (trainGraph) legend->AddEntry(trainGraph, "Train total", "lp");
        if (validationGraph) legend->AddEntry(validationGraph, "Validation total", "lp");
        if (lossLine) legend->AddEntry(lossLine, ("Validation min: e" + std::to_string(bestLoss->epoch)).c_str(), "l");
        if (jerLine) legend->AddEntry(jerLine, ("Reco JER min: e" + std::to_string(bestJer->epoch)).c_str(), "l");
        legend->Draw(); objects.legends.push_back(std::move(legend));
        label(0.14, 0.955, data.model + " / " + data.run, 0.029);
        label(0.14, 0.902, std::to_string(energy) + " GeV | " + region + " | " + data.readout, 0.033);
        label(0.14, 0.852, activeNote(losses), 0.027);
        label(0.14, 0.80, phaseNote(losses), 0.026);
        if (train.empty() && validation.empty()) label(0.19, 0.4, "Total loss unavailable", 0.045);

        canvas.cd(static_cast<int>(data.regions.size() + panel + 1)); setupPad(0.11);
        std::vector<const std::vector<CsvRow>*> selected;
        double highY = 0; std::size_t clipped = 0;
        for (const std::string& method : data.methods) {
            const auto found = data.series.find(Key(energy, region, method));
            if (found == data.series.end()) continue;
            selected.push_back(&found->second);
            for (const CsvRow& row : found->second) {
                highY = std::max(highY, row.resolution + row.error);
                if (zoomMax > 0 && row.resolution > zoomMax) ++clipped;
            }
        }
        const double upper = zoomMax > 0 ? zoomMax : std::max(1., highY * 1.14);
        frame = gPad->DrawFrame(x.first, 0, x.second, upper); axes(frame, "Epoch", "JER [%]");
        activationLines(losses, x.first, x.second, 0, upper, objects);
        if (bestLoss) vertical(bestLoss->epoch, 0, upper, kGreen + 2, 2, objects);
        if (bestJer) vertical(bestJer->epoch, 0, upper, kMagenta + 1, 7, objects);
        std::unique_ptr<TLegend> lowerLegend(new TLegend(0.46, 0.72, 0.94, 0.87));
        lowerLegend->SetBorderSize(0); lowerLegend->SetFillColor(kWhite); lowerLegend->SetTextSize(0.031);
        for (const auto* rows : selected) {
            const bool isReco = rows->front().method == "reco";
            const int color = isReco ? kAzure + 2 : kOrange + 7, style = isReco ? 1 : 2;
            auto full = graph(*rows, 0, rows->size(), color, style);
            lowerLegend->AddEntry(full.get(), isReco ? "GNN reco" : "Truth clustering", "lep");
            objects.graphs.push_back(std::move(full));
            for (std::size_t begin = 0; begin < rows->size();) {
                std::size_t end = begin + 1;
                while (end < rows->size() && (*rows)[end].epoch == (*rows)[end - 1].epoch + 1) ++end;
                auto segment = graph(*rows, begin, end, color, style);
                segment->Draw("LPZ SAME"); objects.graphs.push_back(std::move(segment)); begin = end;
            }
        }
        lowerLegend->Draw(); objects.legends.push_back(std::move(lowerLegend));
        label(0.14, 0.94, region + " | error bars from metrics.csv", 0.030);
        label(0.15, 0.035, zoomMax > 0 ? "Zoom: " + std::to_string(clipped) + " points above range; see full plot"
            : "Minima use epochs observed in JER; loss min uses active phase", 0.025);
        if (selected.empty()) label(0.19, 0.45, "No matching JER data", 0.045);
    }
    save(canvas, name, energyDirectory(output, energy), root);
}

void componentHistory(const Data& data, const LossHistory& losses, const std::string& output, TFile& root,
                      bool activeOnly = false) {
    const int start = activeStart(losses);
    const std::string name = "loss_components_vs_epoch" + suffix(data) + (activeOnly ? "_active" : "");
    Objects objects;
    TCanvas canvas(name.c_str(), name.c_str(), 1450, 1000); canvas.Divide(2, 2, 0.005, 0.005);
    auto x = epochRange(data, losses);
    if (activeOnly) x.first = start - std::max(0.5, (x.second - start) * 0.025);
    for (std::size_t panel = 1; panel < components().size(); ++panel) {
        const std::string& component = components()[panel];
        Points train = points(losses, component, false), validation = points(losses, component, true);
        if (activeOnly) for (auto* rows : {&train, &validation})
            rows->erase(std::remove_if(rows->begin(), rows->end(), [start](const Point& row) {
                return row.epoch < start;
            }), rows->end());
        const auto y = range(train, validation);
        canvas.cd(static_cast<int>(panel)); setupPad(0.22);
        TH1F* frame = gPad->DrawFrame(x.first, y.first, x.second, y.second);
        axes(frame, "Epoch", componentTitle(component) + " (linear)");
        activationLines(losses, x.first, x.second, y.first, y.second, objects);
        TGraphErrors* trainGraph = drawLoss(train, kAzure + 2, 1, objects);
        TGraphErrors* validationGraph = drawLoss(validation, kOrange + 7, 1, objects);
        std::unique_ptr<TLegend> legend(new TLegend(0.58, 0.62, 0.94, 0.77));
        legend->SetBorderSize(0); legend->SetFillColor(kWhite); legend->SetTextSize(0.031);
        if (trainGraph) legend->AddEntry(trainGraph, "Train", "lp");
        if (validationGraph) legend->AddEntry(validationGraph, "Validation", "lp");
        legend->Draw(); objects.legends.push_back(std::move(legend));
        label(0.14, 0.95, data.model + " / " + data.run, 0.028);
        label(0.14, 0.895, componentTitle(component) + " | recorded loss components", 0.034);
        label(0.14, 0.84, activeOnly ? activeNote(losses) : phaseNote(losses), 0.025);
        if (train.empty() && validation.empty()) label(0.21, 0.48, "Component unavailable in this log", 0.04);
        else if (train.empty() || validation.empty()) label(0.15, 0.035,
            train.empty() ? "Train component unavailable" : "Validation component unavailable", 0.028);
        else label(0.15, 0.035, "Missing epochs are left as gaps; no zero filling", 0.026);
    }
    save(canvas, name, output, root);
}

int epochColor(int epoch, int low, int high) {
    static const int first = []() {
        double stop[5] = {0, 0.25, 0.5, 0.75, 1};
        double red[5] = {0.267, 0.230, 0.128, 0.369, 0.993};
        double green[5] = {0.005, 0.322, 0.567, 0.789, 0.906};
        double blue[5] = {0.329, 0.546, 0.551, 0.383, 0.144};
        return TColor::CreateGradientColorTable(5, stop, red, green, blue, 128);
    }();
    const double fraction = high == low ? 0.5 : std::max(0., std::min(1., (epoch - low) / static_cast<double>(high - low)));
    return first + static_cast<int>(std::round(fraction * 127));
}

struct Match { int epoch; double loss, jer, error; };
using Matches = std::vector<Match>;

Matches match(const std::vector<CsvRow>& rows, const LossHistory& losses, const std::string& component) {
    Matches result;
    const int start = activeStart(losses);
    for (const CsvRow& row : rows) {
        if (start >= 0 && row.epoch < start) continue;
        const auto found = losses.epochs.find(row.epoch);
        double loss = 0;
        if (found != losses.epochs.end() && value(found->second, component, true, loss))
            result.push_back({row.epoch, loss, row.resolution, row.error});
    }
    return result;
}

std::unique_ptr<TGraphErrors> scatterGraph(const Matches& rows, int markerStyle) {
    std::unique_ptr<TGraphErrors> result(new TGraphErrors(static_cast<int>(rows.size())));
    for (std::size_t i = 0; i < rows.size(); ++i) {
        result->SetPoint(static_cast<int>(i), rows[i].loss, rows[i].jer);
        result->SetPointError(static_cast<int>(i), 0, rows[i].error);
    }
    result->SetLineColor(kGray + 1); result->SetMarkerColor(kGray + 1);
    result->SetMarkerStyle(markerStyle); result->SetMarkerSize(0.5);
    return result;
}

void scatter(const Data& data, const LossHistory& losses, int energy, const std::string& region,
             double zoomMax, const std::string& output, TFile& root) {
    const std::string name = "jer_vs_validation_loss_" + std::to_string(energy) + "GeV_" + region
        + "_" + data.readout + (data.method == "both" ? "" : "_" + data.method)
        + (zoomMax > 0 ? "_zoom" : "");
    Objects objects;
    TCanvas canvas(name.c_str(), name.c_str(), 1800, 1080); canvas.Divide(3, 2, 0.005, 0.005);
    int firstEpoch = std::numeric_limits<int>::max(), lastEpoch = std::numeric_limits<int>::min();
    std::size_t totalMatched = 0;
    for (const std::string& method : data.methods) {
        const auto found = data.series.find(Key(energy, region, method));
        if (found == data.series.end()) continue;
        for (const std::string& component : components()) for (const Match& row : match(found->second, losses, component)) {
            firstEpoch = std::min(firstEpoch, row.epoch); lastEpoch = std::max(lastEpoch, row.epoch);
            ++totalMatched;
        }
    }
    if (!totalMatched) { firstEpoch = 0; lastEpoch = 1; }
    for (std::size_t panel = 0; panel < components().size(); ++panel) {
        const std::string& component = components()[panel];
        canvas.cd(static_cast<int>(panel + 1)); setupPad(0.17);
        std::map<std::string, Matches> selected;
        double lowX = std::numeric_limits<double>::infinity(), highX = -lowX, highY = 0;
        std::size_t matched = 0, clipped = 0;
        for (const std::string& method : data.methods) {
            const auto found = data.series.find(Key(energy, region, method));
            if (found == data.series.end()) continue;
            Matches rows = match(found->second, losses, component);
            matched += rows.size();
            for (const Match& row : rows) {
                lowX = std::min(lowX, row.loss); highX = std::max(highX, row.loss);
                highY = std::max(highY, row.jer + row.error);
                if (zoomMax > 0 && row.jer > zoomMax) ++clipped;
            }
            selected.emplace(method, std::move(rows));
        }
        if (!matched) { lowX = 0; highX = 1; }
        const double padding = std::max(0.02, (highX - lowX) * 0.09);
        TH1F* frame = gPad->DrawFrame(lowX - padding, 0, highX + padding,
                                    zoomMax > 0 ? zoomMax : std::max(1., highY * 1.15));
        axes(frame, "Validation " + componentTitle(component), "JER [%]");
        label(0.14, 0.95, std::to_string(energy) + " GeV | " + region + " | " + data.readout, 0.031);
        label(0.14, 0.885, componentTitle(component) + " | matched points: " + std::to_string(matched), 0.032);
        if (zoomMax > 0) label(0.15, 0.035, "Zoom: " + std::to_string(clipped) + " points above range; see full plot", 0.026);
        if (!matched) label(0.20, 0.47, "No matching validation component", 0.038);
        for (const auto& item : selected) {
            const bool reco = item.first == "reco";
            const std::string identity = std::to_string(energy) + "GeV_" + region + "_" + item.first
                + "_" + data.readout + "_" + component;
            auto points = scatterGraph(item.second, reco ? 20 : 24);
            points->SetName(("loss_jer_validation_" + identity).c_str());
            points->SetTitle((identity + ";Validation " + componentTitle(component) + ";JER [%]").c_str());
            if (zoomMax == 0) {
                root.cd(); points->Write();
                // A parallel index graph records the exact epoch attached to each scatter point.
                TGraph epochs(static_cast<int>(item.second.size()));
                epochs.SetName(("epoch_for_point_" + identity).c_str());
                epochs.SetTitle("Scatter point index to training epoch;Point index;Epoch");
                for (std::size_t i = 0; i < item.second.size(); ++i) epochs.SetPoint(static_cast<int>(i), i, item.second[i].epoch);
                epochs.Write();
            }
            canvas.cd(static_cast<int>(panel + 1));
            points->Draw("PZ SAME"); objects.graphs.push_back(std::move(points));
            for (const Match& row : item.second) {
                std::unique_ptr<TMarker> marker(new TMarker(row.loss, row.jer, reco ? 20 : 24));
                marker->SetMarkerColor(epochColor(row.epoch, firstEpoch, lastEpoch));
                marker->SetMarkerSize(reco ? 0.65 : 0.85); marker->Draw(); objects.markers.push_back(std::move(marker));
            }
        }
    }
    canvas.cd(6); gPad->Range(0, 0, 1, 1);
    label(0.07, 0.94, data.model + " / " + data.run, 0.035);
    label(0.07, 0.86, "Validation loss versus JER", 0.043);
    label(0.07, 0.78, activeNote(losses), 0.031);
    label(0.07, 0.71, "Each point = one matched checkpoint epoch", 0.032);
    label(0.07, 0.64, "Gray error bars: JER uncertainty from CSV", 0.030);
    label(0.07, 0.57, "Missing components/epochs are omitted", 0.030);
    label(0.07, 0.50, "All axes linear; no zero filling", 0.030);
    std::unique_ptr<TLegend> legend(new TLegend(0.06, 0.22, 0.67, 0.44));
    legend->SetBorderSize(0); legend->SetFillStyle(0); legend->SetTextSize(0.038);
    for (const std::string& method : data.methods) {
        std::unique_ptr<TGraphErrors> sample(new TGraphErrors(1));
        sample->SetMarkerStyle(method == "reco" ? 20 : 24); sample->SetMarkerSize(method == "reco" ? 0.8 : 1.0);
        legend->AddEntry(sample.get(), method == "reco" ? "GNN reco" : "Truth clustering", "p");
        objects.graphs.push_back(std::move(sample));
    }
    legend->Draw(); objects.legends.push_back(std::move(legend));
    if (totalMatched) {
        for (int i = 0; i < 128; ++i) {
            const double y1 = 0.14 + 0.26 * i / 128., y2 = 0.14 + 0.26 * (i + 1) / 128.;
            std::unique_ptr<TBox> box(new TBox(0.77, y1, 0.82, y2));
            const int epoch = firstEpoch + static_cast<int>(std::round((lastEpoch - firstEpoch) * i / 127.));
            box->SetFillColor(epochColor(epoch, firstEpoch, lastEpoch)); box->SetLineColor(box->GetFillColor());
            box->Draw(); objects.boxes.push_back(std::move(box));
        }
        label(0.72, 0.45, "Epoch", 0.033);
        label(0.84, 0.14, std::to_string(firstEpoch), 0.032);
        label(0.84, 0.38, std::to_string(lastEpoch), 0.032);
    }
    label(0.07, 0.07, "Correlation describes this training trajectory only", 0.028);
    save(canvas, name, energyDirectory(output, energy), root);
}

} // namespace loss_plot

void drawLossPlots(const Data& data, const LossHistory& losses, double zoomMax,
                   const std::string& output, TFile& root) {
    if (losses.epochs.empty()) throw std::runtime_error("Training loss history is empty");
    root.cd();
    TNamed("training_loss_source", losses.source.c_str()).Write();
    TNamed("training_loss_activation", loss_plot::activeNote(losses).c_str()).Write();
    TNamed("training_loss_activation_source", losses.activationSource.c_str()).Write();
    TNamed("training_loss_beta_start_epoch", std::to_string(losses.betaStart).c_str()).Write();
    TNamed("training_loss_energy_start_epoch", std::to_string(losses.energyStart).c_str()).Write();
    TNamed("training_loss_plot_definition", "Training/validation values copied from log; scatter joins epoch exactly; missing values omitted; linear axes; component percentages are not used to reconstruct total").Write();
    TNamed("training_loss_minimum_definition", "Lowest finite logged validation total among JER-observed epochs after all known loss activations; unknown activation boundaries use all JER-observed epochs").Write();
    TNamed("training_loss_parse_warnings", join(losses.warnings, "\n").c_str()).Write();
    for (const std::string& component : loss_plot::components()) for (bool validation : {false, true}) {
        const auto points = loss_plot::points(losses, component, validation);
        auto graph = loss_plot::lossGraph(points, 0, points.size(), validation ? kOrange + 7 : kAzure + 2, 1);
        const std::string key = std::string(validation ? "validation_" : "train_") + component;
        graph->SetName(("loss_vs_epoch_" + key).c_str());
        graph->SetTitle((key + ";Epoch;" + loss_plot::componentTitle(component)).c_str());
        graph->Write();
    }
    loss_plot::componentHistory(data, losses, output, root);
    if (loss_plot::activeStart(losses) > 0 && losses.epochs.rbegin()->first >= loss_plot::activeStart(losses))
        loss_plot::componentHistory(data, losses, output, root, true);
    for (int energy : data.energies) {
        loss_plot::epochComparison(data, losses, energy, 0, output, root);
        if (zoomMax > 0) loss_plot::epochComparison(data, losses, energy, zoomMax, output, root);
        for (const std::string& region : data.regions) {
            loss_plot::scatter(data, losses, energy, region, 0, output, root);
            if (zoomMax > 0) loss_plot::scatter(data, losses, energy, region, zoomMax, output, root);
        }
    }
}

} // namespace jer_epoch_plot
#endif
