#ifndef PID_JET_RESOLUTION_SUMMARY_H
#define PID_JET_RESOLUTION_SUMMARY_H

// Shared, versioned output contract for the single-energy PID analysis and
// plot_jet_energy_resolution_pid.cxx. No inference ROOT/H5 is modified.
#include "TCanvas.h"
#include "TDirectory.h"
#include "TFile.h"
#include "TGraphErrors.h"
#include "TH1F.h"
#include "TNamed.h"
#include "TObjString.h"
#include "TParameter.h"
#include "TSystem.h"
#include "TTree.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pid_jer {
static const int kBins = 13;
static const double kEdges[kBins + 1] = {
    0., .1, .2, .3, .4, .5, .6, .7, .8, .9, .925, .950, .970, .980
};
static const char* const kSchema = "pid-jet-resolution-1";
static const char* const kAngleDefinition = "Pandora upstream thrust: sum_q E_q*abs(pz_q)/|p_q| / sum_q E_q; float32";
static const char* const kMetricDefinition = "sqrt(2)*RMS90(Ereco_event)/Mean90(Ereco_event)*100; upstream minimum-RMS 90-percent histogram interval";
static const char* const kSummaryName = "jet_energy_resolution_summary.root";

// Match PfoAnalysis.cc and the H5 event_eval/thrust calculation, including
// float32 intermediate rounding. This is not an optimized final-state thrust
// axis and is not the direction of only the first quark.
inline double upstreamThrust(const double (*p4)[4], int count) {
    if (count != 2) return std::numeric_limits<double>::quiet_NaN();
    float energy = 0.f, weighted = 0.f;
    for (int i = 0; i < count; ++i) {
        const float e = p4[i][0], px = p4[i][1], py = p4[i][2], pz = p4[i][3];
        const float momentum = std::sqrt(px*px + py*py + pz*pz);
        if (!(e > 0.f) || !(momentum > 0.f) || !std::isfinite(e) || !std::isfinite(momentum))
            return std::numeric_limits<double>::quiet_NaN();
        energy += e;
        weighted += (std::fabs(pz) / momentum) * e;
    }
    const float angle = weighted / energy;
    return std::isfinite(angle) && angle >= 0.f && angle <= 1.f
        ? double(angle) : std::numeric_limits<double>::quiet_NaN();
}

inline int angleBin(double value) {
    if (!std::isfinite(value)) return -1;
    // Match the existing Pandora H5 comparison: decimal double boundaries,
    // lower inclusive/upper exclusive, with no angular bin at or above 0.98.
    // The separately pooled central/forward selections below are unchanged.
    for (int i = 0; i < kBins; ++i)
        if (value >= kEdges[i] && value < kEdges[i+1]) return i;
    return -1;
}

struct Point {
    int angle_bin = 0;             // 0..12: angle; -1: central; -2: forward
    double low = 0., high = 0.;
    Long64_t entries = 0, invalid_entries = 0;
    double underflow = 0., overflow = 0.;
    double mean90 = std::numeric_limits<double>::quiet_NaN();
    double rms90 = std::numeric_limits<double>::quiet_NaN();
    double resolution_percent = std::numeric_limits<double>::quiet_NaN();
    double error_percent = std::numeric_limits<double>::quiet_NaN();
    bool valid = false;
};

// Same minimum-RMS interval and float moments as the vendored, unmodified
// pandora_reference/AnalysisHelper.cc. Prefix sums avoid scanning 100000
// histogram bins again for each candidate interval. Do not use the original
// contribution macro's different minimum-WIDTH interval for these figures.
inline Point statistics(const TH1F& hist, Long64_t invalid = 0) {
    Point out;
    out.entries = hist.GetEntries();
    out.invalid_entries = invalid;
    out.underflow = hist.GetBinContent(0);
    out.overflow = hist.GetBinContent(hist.GetNbinsX()+1);
    if (out.entries < 5 || invalid || out.underflow || out.overflow) return out;
    const int n = hist.GetNbinsX();
    std::vector<double> counts(n+2, 0.), sumx(n+2, 0.), sumxx(n+2, 0.);
    for (int i = 0; i <= n; ++i) {
        const float x = hist.GetBinLowEdge(i) + .5 * hist.GetBinWidth(i);
        const float y = hist.GetBinContent(i);
        counts[i+1] = counts[i] + y;
        sumx[i+1] = sumx[i] + float(y*x);
        sumxx[i+1] = sumxx[i] + float(float(y*x)*x);
    }
    const float total = counts[n+1];
    float best = std::numeric_limits<float>::max(), bestMean = 0.f;
    for (int start = 0; start <= n; ++start) {
        if (counts[start] >= total / 10.) break;
        if (!hist.GetBinContent(start)) continue;
        const double target = counts[start] + .9 * total;
        auto end = std::lower_bound(counts.begin()+start+1, counts.end(), target);
        if (end == counts.end()) continue;
        const int stop = end - counts.begin();
        const double entries = counts[stop] - counts[start];
        const float mean = (sumx[stop] - sumx[start]) / entries;
        const float mean2 = (sumxx[stop] - sumxx[start]) / entries;
        const float rms = std::sqrt(mean2 - mean*mean);
        if (rms < best) { best = rms; bestMean = mean; }
    }
    if (best == std::numeric_limits<float>::max() || !(bestMean > 0.f)) return out;
    out.mean90 = bestMean;
    out.rms90 = best;
    // Preserve upstream float rounding for the quoted value and error.
    const float fraction = best / bestMean * std::sqrt(2.) * 100.;
    const float error = fraction / std::sqrt(total);
    out.resolution_percent = fraction;
    out.error_percent = error;
    out.valid = std::isfinite(fraction) && std::isfinite(error);
    return out;
}

inline void saveCanvas(TCanvas* canvas, const std::string& directory, const std::string& name) {
    for (const char* format : {"png", "pdf", "eps"}) {
        const std::string dir = directory + "/" + format;
        gSystem->mkdir(dir.c_str(), true);
        canvas->SaveAs((dir + "/" + name + "." + format).c_str());
    }
}

class Summary {
    struct Input { int file_index; std::string path, metadata; bool pid_input; };
    struct Event {
        int file_index, event;
        double thrust, reco_energy, truth_energy;
        bool valid_angle, valid_energy;
    };
    int energy_;
    std::string method_;
    double beta_;
    std::array<std::unique_ptr<TH1F>, kBins+2> hist_;
    std::array<Long64_t, kBins+2> invalid_{};
    std::vector<Event> events_;
    std::vector<Input> inputs_;
public:
    Summary(int beam_energy, const std::string& method, double beta)
        : energy_(beam_energy), method_(method), beta_(beta) {
        for (int i = 0; i < kBins+2; ++i) {
            const std::string name = i < kBins ? "energy_angle_" + std::to_string(i)
                : (i == kBins ? "energy_central" : "energy_forward");
            hist_[i].reset(new TH1F(name.c_str(), ";E_{reco}^{event} [GeV];Events", 100000, 0., 5000.));
            hist_[i]->SetDirectory(nullptr);
        }
    }
    void addInput(int index, const std::string& path, bool pid_input, const std::string& metadata) {
        if (method_ == "pandora" && pid_input)
            throw std::runtime_error("Cannot label a representative-PID GNN input as Pandora: " + path);
        inputs_.push_back({index, path, metadata, pid_input});
    }
    void observe(int file_index, int event, double thrust, double reco, double truth) {
        const bool angle_ok = std::isfinite(thrust) && thrust >= 0. && thrust <= 1.;
        const bool energy_ok = std::isfinite(reco);
        events_.push_back({file_index, event, thrust, reco, truth, angle_ok, energy_ok});
        if (!angle_ok) return;
        auto fill = [&](int i) { if (energy_ok) hist_[i]->Fill(reco); else ++invalid_[i]; };
        const int bin = angleBin(thrust);
        if (bin >= 0) fill(bin);
        if (thrust < .7f) fill(kBins);
        if (thrust > .7f && thrust < .98f) fill(kBins+1);
    }
    void write(const std::string& directory, bool draw) const {
        TDirectory::TContext context;
        gSystem->mkdir(directory.c_str(), true);
        const std::string path = directory + "/" + kSummaryName;
        TFile output(path.c_str(), "RECREATE");
        if (output.IsZombie()) throw std::runtime_error("Cannot create " + path);
        TNamed("schema", kSchema).Write();
        TNamed("method", method_.c_str()).Write();
        TNamed("angle_definition", kAngleDefinition).Write();
        TNamed("metric_definition", kMetricDefinition).Write();
        TNamed("energy_definition", "Raw saved reco energy; PID=sum(reco/energy_pid); Pandora=sum(reco/pred_edep); no neutrino correction; not Ereco/Etruth").Write();
        TNamed("region_definition", "central: thrust<0.7f; forward: 0.7f<thrust<0.98f; angle: Pandora H5 decimal edges through 0.97 and 0.98, lower inclusive/upper exclusive").Write();
        TNamed("selection", "All valid event/MC_dijet_energy events in readable t/reco files; zero energy retained if no reco cluster passes beta cut").Write();
        TParameter<int>("beam_energy_GeV", energy_).Write();
        TParameter<double>("jet_energy_GeV", energy_ / 2.).Write();
        TParameter<double>("beta_threshold", beta_).Write();
        TParameter<int>("minimum_entries", 5).Write();
        TTree input_rows("inputs", "Source ROOT paths and unmodified PID metadata");
        Input input{};
        input_rows.Branch("file_index", &input.file_index, "file_index/I");
        input_rows.Branch("path", &input.path); input_rows.Branch("metadata", &input.metadata);
        input_rows.Branch("pid_input", &input.pid_input, "pid_input/O");
        for (const auto& value : inputs_) { input = value; input_rows.Fill(); }
        input_rows.Write();
        TTree rows("events", "Inputs to the angular/region energy distributions");
        Event event{};
        rows.Branch("file_index", &event.file_index, "file_index/I");
        rows.Branch("event", &event.event, "event/I");
        rows.Branch("thrust", &event.thrust, "thrust/D");
        rows.Branch("reco_energy", &event.reco_energy, "reco_energy/D");
        rows.Branch("truth_energy", &event.truth_energy, "truth_energy/D");
        rows.Branch("valid_angle", &event.valid_angle, "valid_angle/O");
        rows.Branch("valid_energy", &event.valid_energy, "valid_energy/O");
        for (const auto& value : events_) { event = value; rows.Fill(); }
        rows.Write();
        TTree results("resolution", "Angular bins and directly pooled central/forward regions");
        Point p;
        results.Branch("angle_bin", &p.angle_bin, "angle_bin/I");
        results.Branch("low", &p.low, "low/D"); results.Branch("high", &p.high, "high/D");
        results.Branch("entries", &p.entries, "entries/L");
        results.Branch("invalid_entries", &p.invalid_entries, "invalid_entries/L");
        results.Branch("underflow", &p.underflow, "underflow/D");
        results.Branch("overflow", &p.overflow, "overflow/D");
        results.Branch("mean90", &p.mean90, "mean90/D"); results.Branch("rms90", &p.rms90, "rms90/D");
        results.Branch("resolution_percent", &p.resolution_percent, "resolution_percent/D");
        results.Branch("error_percent", &p.error_percent, "error_percent/D");
        results.Branch("valid", &p.valid, "valid/O");
        std::ofstream text(directory + "/jet_energy_resolution_summary.txt");
        if (!text) throw std::runtime_error("Cannot write resolution summary text");
        text << "# " << kSchema << " beam_energy_GeV=" << energy_ << " method=" << method_ << '\n'
             << "# " << kAngleDefinition << '\n' << "# " << kMetricDefinition << '\n'
             << "# no neutrino correction; invalid points are not zero-resolution measurements\n"
             << "angle_bin low high entries invalid_entries underflow overflow Mean90 RMS90 resolution_percent error_percent valid\n"
             << std::setprecision(12);
        TGraphErrors graph;
        graph.SetName("resolution_vs_thrust");
        graph.SetTitle(";|cos(#theta_{thrust})|;RMS_{90}(E_{j}) / Mean_{90}(E_{j}) [%]");
        for (int i = 0; i < kBins+2; ++i) {
            p = statistics(*hist_[i], invalid_[i]);
            p.angle_bin = i < kBins ? i : (i == kBins ? -1 : -2);
            p.low = i < kBins ? kEdges[i] : (i == kBins ? 0. : .7);
            p.high = i < kBins ? kEdges[i+1] : (i == kBins ? .7 : .98);
            results.Fill(); hist_[i]->Write();
            text << p.angle_bin << ' ' << p.low << ' ' << p.high << ' ' << p.entries << ' '
                 << p.invalid_entries << ' ' << p.underflow << ' ' << p.overflow << ' '
                 << p.mean90 << ' ' << p.rms90 << ' ' << p.resolution_percent << ' '
                 << p.error_percent << ' ' << p.valid << '\n';
            if (i < kBins && p.valid) {
                const int n = graph.GetN();
                graph.SetPoint(n, (p.low+p.high)/2., p.resolution_percent);
                graph.SetPointError(n, (p.high-p.low)/2., p.error_percent);
            }
        }
        results.Write(); graph.Write();
        if (draw) {
            TCanvas canvas("pid_jet_resolution_vs_thrust", "Jet energy resolution", 1000, 750);
            canvas.SetLeftMargin(.14); canvas.SetBottomMargin(.14); canvas.SetTicks(1,1);
            double ymax = 20.;
            for (int i = 0; i < graph.GetN(); ++i) ymax = std::max(ymax, 1.15*(graph.GetY()[i]+graph.GetErrorY(i)));
            auto* frame = canvas.DrawFrame(0., 0., 1., ymax);
            frame->SetTitle(graph.GetTitle());
            graph.SetMarkerStyle(24); graph.SetMarkerColor(kBlue+1); graph.SetLineColor(kBlue+1);
            graph.Draw("P SAME");
            saveCanvas(&canvas, directory, "jet_energy_resolution_vs_thrust");
            canvas.Write("jet_energy_resolution_vs_thrust");
        }
        std::cout << "Saved angular resolution data: " << path << std::endl;
    }
};
} // namespace pid_jer
#endif
