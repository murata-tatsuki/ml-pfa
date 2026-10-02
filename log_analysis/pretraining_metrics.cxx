#include <TCanvas.h>
#include <TFile.h>
#include <TGraph.h>
#include <TLegend.h>
#include <TLine.h>
#include <TMultiGraph.h>
#include <TStyle.h>
#include <TSystem.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <vector>

using namespace std;

namespace {

struct MetricPoint {
  int epoch = -1;
  long long n = 0;
  double mean = 0.;
  double stddev = 0.;
  double rms90 = 0.;
};

struct MetricGroup {
  vector<string> labels;
  map<string, vector<MetricPoint>> points;
};

struct ParsedMetrics {
  MetricGroup energyBins;
  MetricGroup particles;
  vector<int> epochs;
};

enum class Section { kNone, kEnergyBins, kParticles };

// const char *kDefaultInput = "../shell/tmp/singleParticleEvents_outputD5_2026_09_17_123323_alpha_tracker_diff_log_perCluster_multihead_ranks/rank0_pretraining_metrics.log";
// const char *kDefaultInput = "../shell/tmp/singleParticleEvents_outputD5_2026_09_19_053627_log_ratio_mse_multihead_ranks/rank0_pretraining_metrics.log";
// const char *kDefaultInput = "../shell/tmp/singleParticleEvents_outputD5_2026_09_19_221359_log_scaled_relative_multihead_ranks/rank0_pretraining_metrics.log";
const char *kDefaultInput = "../shell/tmp/singleParticleEvents_outputD5_2026_10_01_071122_alpha_tracker_diff_log_perCluster_multihead_ranks/rank0_pretraining_metrics.log";

string trim(const string &text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == string::npos) return "";
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

string safeName(string text) {
  for (char &c : text) {
    if (!isalnum(static_cast<unsigned char>(c))) c = '_';
  }
  while (text.find("__") != string::npos) {
    text.replace(text.find("__"), 2, "_");
  }
  if (!text.empty() && text.back() == '_') text.pop_back();
  return text;
}

void addPoint(MetricGroup &group, const string &label,
              const MetricPoint &point) {
  if (group.points.find(label) == group.points.end()) {
    group.labels.push_back(label);
  }
  group.points[label].push_back(point);
}

bool parseLog(const string &fileName, ParsedMetrics &result) {
  ifstream input(fileName);
  if (!input) {
    cerr << "Error: could not open " << fileName << endl;
    return false;
  }

  const string number =
      R"([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)";
  const regex epochPattern(R"(^\s*Epoch\s+([0-9]+)\s*$)");
  const regex metricPattern("^\\s*(.+?):\\s*n=([0-9]+)\\s+Mean=(" +
                            number + ")\\s+Std=(" + number +
                            ")\\s+RMS90=(" + number + ")\\s*$");

  int currentEpoch = -1;
  Section section = Section::kNone;
  string line;
  size_t lineNumber = 0;
  set<int> seenEpochs;

  while (getline(input, line)) {
    ++lineNumber;
    smatch match;
    if (regex_match(line, match, epochPattern)) {
      currentEpoch = stoi(match[1].str());
      section = Section::kNone;
      if (seenEpochs.insert(currentEpoch).second) result.epochs.push_back(currentEpoch);
      continue;
    }
    if (line.find("by true-energy bin:") != string::npos) {
      section = Section::kEnergyBins;
      continue;
    }
    if (line.find("by particle species:") != string::npos) {
      section = Section::kParticles;
      continue;
    }
    if (trim(line).empty()) continue;

    if (regex_match(line, match, metricPattern)) {
      if (currentEpoch < 0 || section == Section::kNone) {
        cerr << "Warning: ignored metric without Epoch/section at line "
             << lineNumber << endl;
        continue;
      }
      MetricPoint point;
      point.epoch = currentEpoch;
      point.n = stoll(match[2].str());
      point.mean = stod(match[3].str());
      point.stddev = stod(match[4].str());
      point.rms90 = stod(match[5].str());
      const string label = trim(match[1].str());
      if (section == Section::kEnergyBins)
        addPoint(result.energyBins, label, point);
      else
        addPoint(result.particles, label, point);
    }
  }

  sort(result.epochs.begin(), result.epochs.end());
  auto sortPoints = [](MetricGroup &group) {
    for (auto &entry : group.points) {
      sort(entry.second.begin(), entry.second.end(),
           [](const MetricPoint &a, const MetricPoint &b) {
             return a.epoch < b.epoch;
           });
    }
  };
  sortPoints(result.energyBins);
  sortPoints(result.particles);

  if (result.epochs.empty() || result.particles.points.empty()) {
    cerr << "Error: no pretraining validation metrics were found in "
         << fileName << endl;
    return false;
  }
  return true;
}

double metricValue(const MetricPoint &point, const string &metric) {
  if (metric == "Mean") return point.mean;
  if (metric == "Std") return point.stddev;
  return point.rms90;
}

int graphColor(size_t index) {
  const int colors[] = {kBlue + 1, kRed + 1,     kGreen + 2, kMagenta + 1,
                        kOrange + 7, kCyan + 2,  kViolet + 1, kGray + 2,
                        kPink + 7,   kSpring + 5};
  return colors[index % (sizeof(colors) / sizeof(colors[0]))];
}

void drawPanel(const MetricGroup &group, const string &groupName,
               const string &metric, bool zoomMean, bool logY,
               TFile &rootOutput) {
  auto *multiGraph = new TMultiGraph();
  auto *legend = new TLegend(0.72, 0.64, 0.94, 0.90);
  legend->SetBorderSize(0);
  legend->SetFillStyle(0);
  legend->SetTextSize(0.032);

  for (size_t i = 0; i < group.labels.size(); ++i) {
    const string &label = group.labels[i];
    const auto found = group.points.find(label);
    if (found == group.points.end() || found->second.empty()) continue;

    const vector<MetricPoint> &points = found->second;
    auto *graph = new TGraph(points.size());
    graph->SetName((safeName(groupName) + "_" + safeName(label) + "_" +
                    safeName(metric) + (zoomMean ? "_zoom" : ""))
                       .c_str());
    graph->SetTitle(label.c_str());
    graph->SetLineColor(graphColor(i));
    graph->SetMarkerColor(graphColor(i));
    graph->SetLineWidth(2);
    graph->SetMarkerStyle(20 + (i % 10));
    graph->SetMarkerSize(0.45);
    for (size_t j = 0; j < points.size(); ++j) {
      graph->SetPoint(j, points[j].epoch, metricValue(points[j], metric));
    }
    multiGraph->Add(graph, "LP");
    legend->AddEntry(graph, label.c_str(), "lp");
    rootOutput.cd();
    graph->Write();
  }

  string title = groupName + " - " + metric;
  if (zoomMean) title += " (zoom)";
  title += ";Epoch;" + metric + " of E_{pred}/E_{true}";
  multiGraph->SetTitle(title.c_str());
  multiGraph->Draw("A");
  gPad->SetGrid();
  gPad->SetLogy(logY);
  if (zoomMean) {
    multiGraph->SetMinimum(0.5);
    multiGraph->SetMaximum(1.5);
  } else if (!logY) {
    multiGraph->SetMinimum(0.0);
  }
  gPad->Modified();
  gPad->Update();

  if (metric == "Mean") {
    const double xMin = gPad->GetUxmin();
    const double xMax = gPad->GetUxmax();
    auto *reference = new TLine(xMin, 1.0, xMax, 1.0);
    reference->SetLineColor(kBlack);
    reference->SetLineStyle(2);
    reference->SetLineWidth(2);
    reference->Draw("same");
  }
  legend->Draw();
}

TCanvas *makeSummaryCanvas(const MetricGroup &group, const string &groupName,
                           const string &canvasName, TFile &rootOutput) {
  auto *canvas = new TCanvas(canvasName.c_str(), groupName.c_str(), 1500, 1000);
  canvas->Divide(2, 2);
  canvas->cd(1);
  drawPanel(group, groupName, "Mean", false, false, rootOutput);
  canvas->cd(2);
  drawPanel(group, groupName, "Mean", true, false, rootOutput);
  canvas->cd(3);
  drawPanel(group, groupName, "Std", false, true, rootOutput);
  canvas->cd(4);
  drawPanel(group, groupName, "RMS90", false, true, rootOutput);
  canvas->Update();
  rootOutput.cd();
  canvas->Write();
  return canvas;
}

void writeCsvGroup(ofstream &csv, const string &groupName,
                   const MetricGroup &group) {
  for (const string &label : group.labels) {
    const auto found = group.points.find(label);
    if (found == group.points.end()) continue;
    for (const MetricPoint &point : found->second) {
      csv << point.epoch << ',' << groupName << ',' << '"' << label << '"'
          << ',' << point.n << ',' << setprecision(10) << point.mean << ','
          << point.stddev << ',' << point.rms90 << '\n';
    }
  }
}

bool writeCsv(const string &fileName, const ParsedMetrics &metrics) {
  ofstream csv(fileName);
  if (!csv) {
    cerr << "Error: could not write " << fileName << endl;
    return false;
  }
  csv << "epoch,group,category,n,mean,std,rms90\n";
  writeCsvGroup(csv, "energy_bin", metrics.energyBins);
  writeCsvGroup(csv, "particle", metrics.particles);
  return true;
}

void printLatestGroup(const string &title, const MetricGroup &group,
                      int latestEpoch) {
  cout << '\n' << title << " (Epoch " << latestEpoch << ")\n";
  cout << left << setw(20) << "Category" << right << setw(10) << "n"
       << setw(13) << "Mean" << setw(13) << "Std" << setw(13) << "RMS90"
       << '\n';
  cout << string(69, '-') << '\n';
  for (const string &label : group.labels) {
    const auto found = group.points.find(label);
    if (found == group.points.end()) continue;
    const auto point = find_if(found->second.rbegin(), found->second.rend(),
                               [latestEpoch](const MetricPoint &value) {
                                 return value.epoch == latestEpoch;
                               });
    if (point == found->second.rend()) {
      cout << left << setw(20) << label << "  (no entry)\n";
      continue;
    }
    cout << left << setw(20) << label << right << setw(10) << point->n
         << fixed << setprecision(6) << setw(13) << point->mean << setw(13)
         << point->stddev << setw(13) << point->rms90 << '\n';
  }
}

void reportMissingEntries(const MetricGroup &group, const string &groupName,
                          size_t expectedEpochs) {
  for (const string &label : group.labels) {
    const size_t actual = group.points.at(label).size();
    if (actual != expectedEpochs) {
      cerr << "Warning: " << groupName << " '" << label << "' has " << actual
           << " entries; expected " << expectedEpochs << endl;
    }
  }
}

}  // namespace

// Usage from /home/murata/master/log_analysis:
//   root -l -q 'pretraining_metrics.cxx()'
//   root -l -q 'pretraining_metrics.cxx("/path/to/rank0_pretraining_metrics.log")'
//   root -l -q 'pretraining_metrics.cxx("input.log","output_directory")'
void pretraining_metrics(const char *inputFile = kDefaultInput,
                         const char *outputDir = "pretraining_metrics_output") {
  ParsedMetrics metrics;
  if (!parseLog(inputFile, metrics)) return;

  reportMissingEntries(metrics.energyBins, "energy bin", metrics.epochs.size());
  reportMissingEntries(metrics.particles, "particle", metrics.epochs.size());

  if (gSystem->mkdir(outputDir, true) != 0 &&
      gSystem->AccessPathName(outputDir)) {
    cerr << "Error: could not create output directory " << outputDir << endl;
    return;
  }

  const string outputBase = outputDir;
  const string rootName = outputBase + "/pretraining_metrics.root";
  TFile rootOutput(rootName.c_str(), "RECREATE");
  if (rootOutput.IsZombie()) {
    cerr << "Error: could not create " << rootName << endl;
    return;
  }

  gStyle->SetOptStat(0);
  gStyle->SetTitleFontSize(0.04);
  TCanvas *particleCanvas = makeSummaryCanvas(
      metrics.particles, "Validation by particle species", "c_particles",
      rootOutput);
  TCanvas *energyCanvas = makeSummaryCanvas(
      metrics.energyBins, "Validation by true-energy bin", "c_energy_bins",
      rootOutput);

  auto *summaryCanvas = new TCanvas(
      "c_pretraining_metrics_summary", "Pretraining validation summary", 1500,
      2000);
  summaryCanvas->Divide(1, 2, 0.0, 0.0);
  summaryCanvas->cd(1);
  particleCanvas->DrawClonePad();
  summaryCanvas->cd(2);
  energyCanvas->DrawClonePad();
  summaryCanvas->Update();

  const string summaryPng =
      outputBase + "/pretraining_metrics_summary.png";
  const string particlePng = outputBase + "/particles.png";
  const string energyBinPng = outputBase + "/energy_bins.png";
  summaryCanvas->Print(summaryPng.c_str());
  particleCanvas->Print(particlePng.c_str());
  energyCanvas->Print(energyBinPng.c_str());
  rootOutput.cd();
  summaryCanvas->Write();

  const string csvName = outputBase + "/pretraining_metrics.csv";
  writeCsv(csvName, metrics);
  rootOutput.Close();

  const int latestEpoch = metrics.epochs.back();
  cout << "Read " << metrics.epochs.size() << " epochs from " << inputFile
       << endl;
  printLatestGroup("Particle species", metrics.particles, latestEpoch);
  printLatestGroup("True-energy bins", metrics.energyBins, latestEpoch);
  cout << "\nSaved:\n"
       << "  " << summaryPng << '\n'
       << "  " << particlePng << '\n'
       << "  " << energyBinPng << '\n'
       << "  " << csvName << '\n'
       << "  " << rootName << endl;
}
