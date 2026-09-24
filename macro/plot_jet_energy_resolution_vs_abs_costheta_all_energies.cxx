// Overlay all jet energies in separate reco, truth-clustering, and PandoraPFA plots.
// Numerical values are read from logs produced by
// efficiency_purity_check_reco_effpur_contiribution.cxx.
//
// Usage from macro/:
//   root -l -b -q plot_jet_energy_resolution_vs_abs_costheta_all_energies.cxx

#include "root_common_includes.h"
#include "TLatex.h"

#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

namespace {

const int kNEnergies = 5;
const int kBeamEnergies[kNEnergies] = {40, 91, 200, 350, 500};
const double kJetEnergies[kNEnergies] = {20.0, 45.5, 100.0, 175.0, 250.0};

const int kNAbsCosThetaBins = 13;
const double kAbsCosThetaEdges[kNAbsCosThetaBins + 1] = {
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    0.925, 0.950, 0.975, 1.0
};

enum MethodIndex {
    kReco = 0,
    kTruthClustering = 1,
    kPandora = 2,
    kNMethods = 3
};

const char* kMethodNames[kNMethods] = {"reco", "truth_clustering", "pandora"};
const char* kMethodLabels[kNMethods] = {"GNN", "GNN perfect clustering", "PandoraPFA"};

// Number of events in each |cos(theta_q)| bin. It is common to the three
// reconstruction methods at a fixed beam energy.
Long64_t kEntries[kNEnergies][kNAbsCosThetaBins] = {};

// sqrt(2) * RMS90(Ereco_event / Etrue_builder_event) [%]
// Index order: [method][beam energy][|cos(theta_q)| bin].
double kResolution[kNMethods][kNEnergies][kNAbsCosThetaBins] = {};

// Mean90(Ereco_event / Etrue_builder_event), used only for the additional
// response-normalized plots. Index order is the same as kResolution.
double kMean90[kNMethods][kNEnergies][kNAbsCosThetaBins] = {};

int findEnergyIndex(int beamEnergy) {
    for (int ie = 0; ie < kNEnergies; ++ie) {
        if (kBeamEnergies[ie] == beamEnergy) return ie;
    }
    return -1;
}

int findMethodIndex(const string& method) {
    if (method == "reco") return kReco;
    if (method == "truth") return kTruthClustering;
    if (method == "pandora") return kPandora;
    return -1;
}

bool loadResults(const char* inputPath) {
    ifstream input(inputPath);
    if (!input) {
        cerr << "Error: cannot open input file: " << inputPath << endl;
        return false;
    }

    bool valueLoaded[kNMethods][kNEnergies][kNAbsCosThetaBins] = {};
    bool entriesLoaded[kNEnergies][kNAbsCosThetaBins] = {};
    int currentEnergy = -1;
    int currentMethod = -1;
    int currentBin = 0;
    int lineNumber = 0;
    string line;

    while (getline(input, line)) {
        ++lineNumber;

        int beamEnergy = 0;
        string gev;
        string method;
        istringstream header(line);
        if ((header >> beamEnergy >> gev >> method) && gev == "GeV") {
            const int energyIndex = findEnergyIndex(beamEnergy);
            const int methodIndex = findMethodIndex(method);
            if (energyIndex >= 0 && methodIndex >= 0) {
                currentEnergy = energyIndex;
                currentMethod = methodIndex;
                currentBin = 0;
                continue;
            }
        }

        const size_t firstCharacter = line.find_first_not_of(" \t");
        if (firstCharacter == string::npos || line[firstCharacter] != '[') continue;
        if (currentEnergy < 0 || currentMethod < 0) continue;

        double lowerEdge = 0.0;
        double upperEdge = 0.0;
        long long entries = 0;
        double mean90 = 0.0;
        double rms90 = 0.0;
        double resolution = 0.0;
        const int fieldsRead = sscanf(
            line.c_str(),
            " [%lf, %lf): entries=%lld, Mean90=%lf, RMS90=%lf, resolution=%lf%%",
            &lowerEdge, &upperEdge, &entries, &mean90, &rms90, &resolution);

        if (fieldsRead != 6 || currentBin >= kNAbsCosThetaBins) {
            cerr << "Error: invalid data at " << inputPath << ':' << lineNumber
                 << ": " << line << endl;
            return false;
        }
        if (fabs(lowerEdge - kAbsCosThetaEdges[currentBin]) > 1e-9 ||
            fabs(upperEdge - kAbsCosThetaEdges[currentBin + 1]) > 1e-9) {
            cerr << "Error: unexpected |cos(theta_q)| bin at " << inputPath
                 << ':' << lineNumber << endl;
            return false;
        }
        if (valueLoaded[currentMethod][currentEnergy][currentBin]) {
            cerr << "Error: duplicate data at " << inputPath << ':' << lineNumber
                 << endl;
            return false;
        }
        if (entriesLoaded[currentEnergy][currentBin] &&
            kEntries[currentEnergy][currentBin] != entries) {
            cerr << "Error: inconsistent entries at " << inputPath << ':'
                 << lineNumber << endl;
            return false;
        }

        kEntries[currentEnergy][currentBin] = static_cast<Long64_t>(entries);
        kMean90[currentMethod][currentEnergy][currentBin] = mean90;
        kResolution[currentMethod][currentEnergy][currentBin] = resolution;
        entriesLoaded[currentEnergy][currentBin] = true;
        valueLoaded[currentMethod][currentEnergy][currentBin] = true;
        ++currentBin;
    }

    for (int im = 0; im < kNMethods; ++im) {
        for (int ie = 0; ie < kNEnergies; ++ie) {
            for (int ib = 0; ib < kNAbsCosThetaBins; ++ib) {
                if (!valueLoaded[im][ie][ib]) {
                    cerr << "Error: missing data for " << kBeamEnergies[ie]
                         << " GeV, " << kMethodNames[im] << ", bin " << ib
                         << " in " << inputPath << endl;
                    return false;
                }
            }
        }
    }

    cout << "Loaded jet energy resolution data from: " << inputPath << endl;
    return true;
}

void drawMethodPlot(
    int methodIndex,
    bool normalizeByMean90,
    const char* outputDirectory
) {
    const string method = kMethodNames[methodIndex];
    const string metricName = normalizeByMean90 ? "rms90_over_mean90" : "rms90_over_etrue";
    const int colors[kNEnergies] = {
        kGreen + 2, kAzure + 2, kOrange + 7, kRed + 1, kBlack
    };
    const int markerStyles[kNEnergies] = {
        kFullSquare, kFullCircle, kOpenCircle, kFullTriangleDown, kFullCircle
    };

    TCanvas* canvas = new TCanvas(
        Form("canvas_all_energies_%s_%s", method.c_str(), metricName.c_str()),
        Form("jet energy resolution vs truth jet direction: %s", method.c_str()),
        1100,
        750);
    canvas->SetLeftMargin(0.14);
    canvas->SetRightMargin(0.25);
    canvas->SetTopMargin(0.07);
    canvas->SetBottomMargin(0.13);
    canvas->SetGridx();
    canvas->SetGridy();

    TH2F* frame = new TH2F(
        Form("frame_all_energies_%s_%s", method.c_str(), metricName.c_str()),
        normalizeByMean90
            ? ";|cos(#theta_{q})|;#sqrt{2} RMS_{90}(E_{reco}) / Mean_{90}(E_{reco}) [%]"
            : ";|cos(#theta_{q})|;#sqrt{2} RMS_{90}(E_{reco}) / E_{true} [%]",
        100, 0.0, 1.0,
        100, 0.0, normalizeByMean90 ? 22.0 : 15.0);
    frame->SetStats(0);
    frame->GetXaxis()->SetTitleSize(0.055);
    frame->GetXaxis()->SetLabelSize(0.045);
    frame->GetYaxis()->SetTitleSize(0.047);
    frame->GetYaxis()->SetLabelSize(0.045);
    frame->GetYaxis()->SetTitleOffset(1.35);
    frame->Draw();

    TLegend* legend = new TLegend(0.77, 0.57, 0.98, 0.91);
    legend->SetBorderSize(1);
    legend->SetFillStyle(1001);
    legend->SetFillColor(kWhite);
    legend->SetTextSize(0.038);

    for (int ie = 0; ie < kNEnergies; ie++) {
        const int beamEnergy = kBeamEnergies[ie];
        TGraphErrors* graph = new TGraphErrors(kNAbsCosThetaBins);
        graph->SetName(Form(
            "graph_%s_%s_%dGeV",
            method.c_str(),
            metricName.c_str(),
            beamEnergy));
        graph->SetMarkerStyle(markerStyles[ie]);
        graph->SetMarkerSize(1.15);
        graph->SetMarkerColor(colors[ie]);
        graph->SetLineColor(colors[ie]);
        graph->SetLineWidth(2);

        for (int ib = 0; ib < kNAbsCosThetaBins; ib++) {
            const double x = 0.5 * (kAbsCosThetaEdges[ib] + kAbsCosThetaEdges[ib + 1]);
            const double xError = 0.5 * (kAbsCosThetaEdges[ib + 1] - kAbsCosThetaEdges[ib]);
            const double mean90 = kMean90[methodIndex][ie][ib];
            const double resolution = normalizeByMean90
                ? kResolution[methodIndex][ie][ib] / mean90
                : kResolution[methodIndex][ie][ib];
            const double n90 = 0.9 * static_cast<double>(kEntries[ie][ib]);
            // The logs do not contain the Mean90 uncertainty, so the normalized
            // plot propagates only the original RMS90 statistical uncertainty.
            const double yError = n90 > 1.0
                ? resolution / sqrt(2.0 * (n90 - 1.0))
                : 0.0;
            graph->SetPoint(ib, x, resolution);
            graph->SetPointError(ib, xError, yError);
        }

        graph->Draw("P E1 SAME");
        const string jetEnergyLabel = fabs(kJetEnergies[ie] - floor(kJetEnergies[ie])) < 1e-9
            ? Form("%.0f GeV jets", kJetEnergies[ie])
            : Form("%.1f GeV jets", kJetEnergies[ie]);
        legend->AddEntry(graph, jetEnergyLabel.c_str(), "lp");
    }

    legend->Draw();

    TLatex label;
    label.SetNDC();
    label.SetTextFont(42);
    label.SetTextSize(0.050);
    label.DrawLatex(0.49, 0.86, "Z/#gamma^{*} #rightarrow uds");
    label.SetTextAlign(22);
    label.SetTextSize(0.038);
    label.DrawLatex(0.565, 0.79, kMethodLabels[methodIndex]);

    canvas->RedrawAxis();
    const string outputPath = Form(
        "%s/jet_energy_resolution_vs_abs_costheta_all_energies_%s%s.png",
        outputDirectory,
        method.c_str(),
        normalizeByMean90 ? "_rms90_over_mean90" : "");
    canvas->SaveAs(outputPath.c_str());
    cout << "Saved: " << outputPath << endl;
}

}  // namespace

void plot_jet_energy_resolution_vs_abs_costheta_all_energies(
    const char* outputDirectory = "figures/nnqq2M_fixed_uds",
    const char* inputPath = "mono-head.txt"
) {
    if (!loadResults(inputPath)) return;

    gStyle->SetOptStat(0);
    gSystem->mkdir(outputDirectory, true);

    // Keep the original RMS90/Etrue plots.
    drawMethodPlot(kPandora, false, outputDirectory);
    drawMethodPlot(kTruthClustering, false, outputDirectory);
    drawMethodPlot(kReco, false, outputDirectory);

    // Add response-normalized RMS90/Mean90 plots.
    drawMethodPlot(kPandora, true, outputDirectory);
    drawMethodPlot(kTruthClustering, true, outputDirectory);
    drawMethodPlot(kReco, true, outputDirectory);
}
