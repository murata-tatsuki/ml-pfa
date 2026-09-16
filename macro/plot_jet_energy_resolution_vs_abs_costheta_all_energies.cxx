// Overlay all jet energies in separate reco, truth-clustering, and PandoraPFA plots.
// Numerical values are hard-coded from logs produced by
// efficiency_purity_check_reco_effpur_contiribution.cxx.
//
// Usage from macro/:
//   root -l -b -q plot_jet_energy_resolution_vs_abs_costheta_all_energies.cxx

#include "root_common_includes.h"
#include "TLatex.h"

#include <cmath>
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
const Long64_t kEntries[kNEnergies][kNAbsCosThetaBins] = {
    {11431, 11610, 12122, 12622, 13572, 14787, 15960, 17446, 19332, 5150, 5213, 5324, 5431},
    {11061, 11415, 11937, 12648, 13581, 14520, 16124, 17467, 19497, 5184, 5394, 5551, 5621},
    {11105, 11513, 11956, 12685, 13718, 14774, 15884, 17233, 19369, 5179, 5301, 5324, 5559},
    {11263, 11380, 11986, 12606, 13555, 14771, 15870, 17766, 19284, 5100, 5396, 5423, 5400},
    {11246, 11531, 11907, 12637, 13290, 14702, 16075, 17668, 19270, 5073, 5227, 5497, 5677}
};

// sqrt(2) * RMS90(Ereco_event / Etrue_builder_event) [%]
// Index order: [method][beam energy][|cos(theta_q)| bin].
const double kResolution[kNMethods][kNEnergies][kNAbsCosThetaBins] = {
    { // reco
        {8.22372, 8.24182, 7.92002, 8.07217, 8.14117, 8.37525, 8.81083, 9.51417, 9.68236, 9.74219, 9.75779, 10.2959, 13.5611},
        {4.6106, 4.32408, 4.13981, 4.14345, 4.17392, 4.31697, 4.48191, 4.98368, 5.18659, 5.30617, 5.33669, 5.89437, 11.804},
        {4.66907, 3.96787, 3.62938, 3.4568, 3.37207, 3.38691, 3.28524, 3.4489, 3.72939, 4.28825, 4.62046, 5.42998, 12.5678},
        {8.79048, 7.88371, 7.56028, 7.38579, 7.20649, 7.07017, 6.99442, 6.81289, 6.91657, 7.8441, 8.40093, 9.0403, 13.5985},
        {12.1928, 11.7339, 11.3952, 11.3727, 11.4389, 11.1496, 11.5413, 11.6484, 11.2875, 11.5755, 11.7484, 12.2112, 13.7202}
    },
    { // truth clustering
        {6.08172, 6.14622, 5.95783, 6.06041, 6.21486, 6.36011, 6.65329, 7.27786, 7.64663, 7.75417, 7.80337, 8.48038, 12.1306},
        {3.64624, 3.49652, 3.38572, 3.33219, 3.40559, 3.48684, 3.67393, 4.08986, 4.33098, 4.48799, 4.48409, 5.16056, 11.3308},
        {4.37128, 3.55738, 3.23855, 3.11533, 3.00689, 2.9415, 2.93192, 3.08961, 3.34655, 3.80641, 4.28543, 5.02391, 12.2211},
        {8.6227, 7.68284, 7.24524, 7.1061, 6.92909, 6.73936, 6.63799, 6.58744, 6.61878, 7.34037, 7.95616, 8.7248, 13.5831},
        {12.0582, 11.6468, 11.2717, 11.1803, 11.2942, 10.944, 11.3036, 11.3006, 11.0332, 11.4066, 11.5157, 11.8866, 13.6303}
    },
    { // PandoraPFA
        {6.43956, 6.43602, 6.49152, 6.458, 6.46833, 6.37797, 6.49586, 6.65104, 6.75303, 6.95596, 7.0711, 7.68242, 10.9135},
        {4.1211, 4.05153, 4.00636, 4.01674, 3.97227, 3.98498, 3.98389, 4.26638, 4.373, 4.63845, 4.84101, 5.29527, 9.76413},
        {3.20151, 2.96315, 2.94207, 2.93893, 2.9152, 2.86591, 3.00415, 3.18816, 3.14858, 3.30588, 3.61586, 4.04438, 9.53964},
        {3.16598, 2.8123, 2.77301, 2.74867, 2.6465, 2.63291, 2.78396, 2.96133, 2.77342, 3.05687, 3.20561, 3.65559, 8.36541},
        {3.27823, 2.80712, 2.78187, 2.69787, 2.64471, 2.65462, 2.85357, 2.97345, 2.70795, 3.00734, 3.18327, 3.62521, 8.35294}
    }
};

// Mean90(Ereco_event / Etrue_builder_event), used only for the additional
// response-normalized plots. Index order is the same as kResolution.
const double kMean90[kNMethods][kNEnergies][kNAbsCosThetaBins] = {
    { // reco
        {1.02496, 1.02876, 1.02804, 1.03244, 1.03499, 1.03905, 1.04667, 1.05434, 1.05451, 1.05374, 1.04796, 1.04016, 0.996734},
        {0.987417, 0.99212, 0.992695, 0.99517, 0.998135, 0.999495, 1.00185, 1.00562, 1.00559, 1.00331, 1.00072, 0.990984, 0.928032},
        {0.949363, 0.956534, 0.96004, 0.962851, 0.964802, 0.965383, 0.967804, 0.968989, 0.968969, 0.961238, 0.957554, 0.94598, 0.875577},
        {0.877663, 0.892264, 0.896091, 0.89837, 0.901901, 0.905277, 0.907885, 0.9094, 0.906342, 0.891478, 0.88396, 0.871899, 0.808408},
        {0.785791, 0.801348, 0.808632, 0.811821, 0.816102, 0.821994, 0.820386, 0.816915, 0.815891, 0.803276, 0.790508, 0.777713, 0.715037}
    },
    { // truth clustering
        {0.978627, 0.981556, 0.982194, 0.985752, 0.987703, 0.990182, 0.99259, 0.997051, 0.998408, 0.995903, 0.991497, 0.984361, 0.939854},
        {0.969485, 0.972316, 0.97567, 0.976139, 0.979326, 0.978937, 0.981336, 0.984843, 0.983897, 0.978557, 0.978527, 0.96585, 0.904718},
        {0.941589, 0.949895, 0.952789, 0.954839, 0.957193, 0.958785, 0.959287, 0.960778, 0.960341, 0.953033, 0.946893, 0.935505, 0.869122},
        {0.874787, 0.889379, 0.895103, 0.896931, 0.900326, 0.904006, 0.906736, 0.907019, 0.904652, 0.891317, 0.883739, 0.868959, 0.803607},
        {0.78671, 0.798992, 0.807547, 0.81176, 0.814774, 0.822078, 0.821349, 0.821676, 0.81886, 0.80457, 0.793463, 0.781598, 0.72583}
    },
    { // PandoraPFA
        {0.922893, 0.923812, 0.928061, 0.929657, 0.929732, 0.933458, 0.933538, 0.934952, 0.934868, 0.932527, 0.929004, 0.920727, 0.868749},
        {0.959581, 0.962132, 0.964221, 0.96475, 0.966449, 0.967256, 0.970068, 0.96988, 0.971106, 0.971192, 0.970145, 0.959944, 0.907259},
        {0.974978, 0.979022, 0.980249, 0.979422, 0.981532, 0.98293, 0.983516, 0.985407, 0.988771, 0.98625, 0.985656, 0.979657, 0.924554},
        {0.974878, 0.981934, 0.982835, 0.983792, 0.98524, 0.984731, 0.984609, 0.986359, 0.993601, 0.989999, 0.98888, 0.983626, 0.937616},
        {0.974511, 0.981044, 0.982326, 0.984223, 0.985179, 0.985416, 0.98315, 0.985751, 0.994605, 0.991407, 0.99001, 0.9843, 0.937806}
    }
};

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
    const char* outputDirectory = "figures/nnqq2M_fixed_uds"
) {
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
