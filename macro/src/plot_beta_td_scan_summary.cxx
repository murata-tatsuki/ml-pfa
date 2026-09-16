// Plot the compact ROOT file made by make_beta_td_scan_summary.cxx.
//
// Usage (from master/macro):
//   root -l -b -q \
//     'src/plot_beta_td_scan_summary.cxx("scan_summary_91GeV.root","scan_plots_91GeV","all")'
//
// The third argument may be inclusive, electron, pion, photon, neutron, K0,
// muon, or all.  No Pandora overlay is made.

#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TLine.h>
#include <TMarker.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TSystem.h>
#include <TTree.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace beta_td_scan_plot {

struct ScanRow {
    int energy_gev = -1;
    int beta_code = 0;
    int distance_code = 0;
    double beta = 0.0;
    double distance = 0.0;
    int category_id = 0;
    std::string category;
    double legacy_efficiency = -1.0;
    double legacy_purity = -1.0;
    double cedric_efficiency = -1.0;
    double cedric_purity = -1.0;
    double cedric_clustering_quality = -1.0;
    double cedric_confusion_quality = -1.0;
    double cedric_charged_quality = -1.0;
    double cedric_neutral_quality = -1.0;
    double cedric_assignment_quality = -1.0;
    double cedric_charged_efficiency = -1.0;
    double cedric_charged_purity = -1.0;
    double cedric_charged_to_neutral_confusion = -1.0;
    double cedric_neutral_efficiency = -1.0;
    double cedric_neutral_purity = -1.0;
    double cedric_neutral_to_charged_confusion = -1.0;
    double cedric_coverage = -1.0;
    double cedric_precision = -1.0;
    double cedric_spurious_cluster_rate = -1.0;
    double response_mean = -1.0;
    double resolution = -1.0;
};

std::vector<double> makeBinEdges(const std::set<double>& input) {
    std::vector<double> centers(input.begin(), input.end());
    std::vector<double> edges;
    if (centers.empty()) return edges;
    if (centers.size() == 1) {
        edges.push_back(centers.front() - 0.05);
        edges.push_back(centers.front() + 0.05);
        return edges;
    }
    edges.resize(centers.size() + 1);
    edges.front() = centers.front() - 0.5 * (centers[1] - centers[0]);
    for (std::size_t index = 1; index < centers.size(); ++index) {
        edges[index] = 0.5 * (centers[index - 1] + centers[index]);
    }
    edges.back() = centers.back() +
                   0.5 * (centers.back() - centers[centers.size() - 2]);
    return edges;
}

std::string cleanName(std::string name) {
    for (char& character : name) {
        if (!std::isalnum(static_cast<unsigned char>(character))) character = '_';
    }
    return name;
}

double harmonicMean(double first, double second) {
    return first >= 0.0 && second >= 0.0 && first + second > 0.0
               ? 2.0 * first * second / (first + second)
               : -1.0;
}

std::string thresholdLabel(double value) {
    std::string label = std::to_string(value);
    while (!label.empty() && label.back() == '0') label.pop_back();
    if (!label.empty() && label.back() == '.') label.pop_back();
    return label;
}

void configureMetricMap(TH2D* histogram,
                        const std::set<double>& beta_values,
                        const std::set<double>& distance_values,
                        bool unit_range = true) {
    histogram->SetStats(false);
    histogram->GetXaxis()->SetTitle("#beta threshold");
    histogram->GetYaxis()->SetTitle("distance threshold");
    histogram->GetXaxis()->SetTitleOffset(1.15);
    histogram->GetYaxis()->SetTitleOffset(1.25);
    histogram->GetZaxis()->SetTitleOffset(1.25);
    int bin = 1;
    for (double value : beta_values) {
        histogram->GetXaxis()->SetBinLabel(bin++, thresholdLabel(value).c_str());
    }
    bin = 1;
    for (double value : distance_values) {
        histogram->GetYaxis()->SetBinLabel(bin++, thresholdLabel(value).c_str());
    }
    if (unit_range) {
        histogram->SetMinimum(0.0);
        histogram->SetMaximum(1.0);
    }
}

}  // namespace beta_td_scan_plot

void plot_beta_td_scan_summary(const char* summary_file,
                               const char* output_directory = "beta_td_scan_plots",
                               const char* category_to_plot = "all") {
    using namespace beta_td_scan_plot;

    if (!summary_file || summary_file[0] == '\0') {
        std::cerr << "ERROR: summary_file is required." << std::endl;
        return;
    }
    const std::string output_path =
        output_directory && output_directory[0] != '\0'
            ? output_directory
            : "beta_td_scan_plots";
    const std::string requested_category =
        category_to_plot && category_to_plot[0] != '\0'
            ? category_to_plot
            : "all";

    TFile input(summary_file, "READ");
    TTree* tree = static_cast<TTree*>(input.Get("scan_summary"));
    if (input.IsZombie() || !tree) {
        std::cerr << "ERROR: cannot read scan_summary from " << summary_file
                  << std::endl;
        return;
    }

    const char* required_branches[] = {
        "energy_gev", "beta_code", "distance_code", "beta", "distance",
        "category_id", "category", "legacy_efficiency", "legacy_purity",
        "cedric_efficiency", "cedric_purity", "response_mean", "resolution"
    };
    for (const char* name : required_branches) {
        if (!tree->GetBranch(name)) {
            std::cerr << "ERROR: missing scan_summary/" << name << std::endl;
            return;
        }
    }

    const char* cedric_quality_branches[] = {
        "cedric_clustering_quality", "cedric_confusion_quality",
        "cedric_charged_quality", "cedric_neutral_quality",
        "cedric_assignment_quality", "cedric_charged_efficiency",
        "cedric_charged_purity", "cedric_charged_to_neutral_confusion",
        "cedric_neutral_efficiency", "cedric_neutral_purity",
        "cedric_neutral_to_charged_confusion", "cedric_coverage",
        "cedric_precision", "cedric_spurious_cluster_rate"
    };
    bool has_cedric_quality_branches = true;
    std::vector<std::string> missing_cedric_quality_branches;
    for (const char* name : cedric_quality_branches) {
        if (!tree->GetBranch(name)) {
            has_cedric_quality_branches = false;
            missing_cedric_quality_branches.push_back(name);
        }
    }
    if (!has_cedric_quality_branches) {
        std::cerr
            << "WARNING: this summary predates the full Cedric metrics; "
            << "Cedric quality/component plots will be skipped. Missing branches:";
        for (const std::string& name : missing_cedric_quality_branches) {
            std::cerr << " " << name;
        }
        std::cerr << "\nRe-run make_beta_td_scan_summary.cxx on ROOT files "
                  << "containing the hit-level prediction tree." << std::endl;
    }

    int energy_gev = -1;
    int beta_code = 0;
    int distance_code = 0;
    double beta = 0.0;
    double distance = 0.0;
    int category_id = 0;
    char category[32] = {0};
    double legacy_efficiency = -1.0;
    double legacy_purity = -1.0;
    double cedric_efficiency = -1.0;
    double cedric_purity = -1.0;
    double cedric_clustering_quality = -1.0;
    double cedric_confusion_quality = -1.0;
    double cedric_charged_quality = -1.0;
    double cedric_neutral_quality = -1.0;
    double cedric_assignment_quality = -1.0;
    double cedric_charged_efficiency = -1.0;
    double cedric_charged_purity = -1.0;
    double cedric_charged_to_neutral_confusion = -1.0;
    double cedric_neutral_efficiency = -1.0;
    double cedric_neutral_purity = -1.0;
    double cedric_neutral_to_charged_confusion = -1.0;
    double cedric_coverage = -1.0;
    double cedric_precision = -1.0;
    double cedric_spurious_cluster_rate = -1.0;
    double response_mean = -1.0;
    double resolution = -1.0;

    tree->SetBranchAddress("energy_gev", &energy_gev);
    tree->SetBranchAddress("beta_code", &beta_code);
    tree->SetBranchAddress("distance_code", &distance_code);
    tree->SetBranchAddress("beta", &beta);
    tree->SetBranchAddress("distance", &distance);
    tree->SetBranchAddress("category_id", &category_id);
    tree->SetBranchAddress("category", category);
    tree->SetBranchAddress("legacy_efficiency", &legacy_efficiency);
    tree->SetBranchAddress("legacy_purity", &legacy_purity);
    tree->SetBranchAddress("cedric_efficiency", &cedric_efficiency);
    tree->SetBranchAddress("cedric_purity", &cedric_purity);
    if (has_cedric_quality_branches) {
        tree->SetBranchAddress("cedric_clustering_quality", &cedric_clustering_quality);
        tree->SetBranchAddress("cedric_confusion_quality", &cedric_confusion_quality);
        tree->SetBranchAddress("cedric_charged_quality", &cedric_charged_quality);
        tree->SetBranchAddress("cedric_neutral_quality", &cedric_neutral_quality);
        tree->SetBranchAddress("cedric_assignment_quality", &cedric_assignment_quality);
        tree->SetBranchAddress("cedric_charged_efficiency", &cedric_charged_efficiency);
        tree->SetBranchAddress("cedric_charged_purity", &cedric_charged_purity);
        tree->SetBranchAddress("cedric_charged_to_neutral_confusion",
                               &cedric_charged_to_neutral_confusion);
        tree->SetBranchAddress("cedric_neutral_efficiency", &cedric_neutral_efficiency);
        tree->SetBranchAddress("cedric_neutral_purity", &cedric_neutral_purity);
        tree->SetBranchAddress("cedric_neutral_to_charged_confusion",
                               &cedric_neutral_to_charged_confusion);
        tree->SetBranchAddress("cedric_coverage", &cedric_coverage);
        tree->SetBranchAddress("cedric_precision", &cedric_precision);
        tree->SetBranchAddress("cedric_spurious_cluster_rate",
                               &cedric_spurious_cluster_rate);
    }
    tree->SetBranchAddress("response_mean", &response_mean);
    tree->SetBranchAddress("resolution", &resolution);

    std::vector<ScanRow> rows;
    std::set<double> beta_values;
    std::set<double> distance_values;
    std::map<int, std::string> categories;
    for (Long64_t entry = 0; entry < tree->GetEntries(); ++entry) {
        tree->GetEntry(entry);
        ScanRow row;
        row.energy_gev = energy_gev;
        row.beta_code = beta_code;
        row.distance_code = distance_code;
        row.beta = beta;
        row.distance = distance;
        row.category_id = category_id;
        row.category = category;
        row.legacy_efficiency = legacy_efficiency;
        row.legacy_purity = legacy_purity;
        row.cedric_efficiency = cedric_efficiency;
        row.cedric_purity = cedric_purity;
        row.cedric_clustering_quality = cedric_clustering_quality;
        row.cedric_confusion_quality = cedric_confusion_quality;
        row.cedric_charged_quality = cedric_charged_quality;
        row.cedric_neutral_quality = cedric_neutral_quality;
        row.cedric_assignment_quality = cedric_assignment_quality;
        row.cedric_charged_efficiency = cedric_charged_efficiency;
        row.cedric_charged_purity = cedric_charged_purity;
        row.cedric_charged_to_neutral_confusion =
            cedric_charged_to_neutral_confusion;
        row.cedric_neutral_efficiency = cedric_neutral_efficiency;
        row.cedric_neutral_purity = cedric_neutral_purity;
        row.cedric_neutral_to_charged_confusion =
            cedric_neutral_to_charged_confusion;
        row.cedric_coverage = cedric_coverage;
        row.cedric_precision = cedric_precision;
        row.cedric_spurious_cluster_rate = cedric_spurious_cluster_rate;
        row.response_mean = response_mean;
        row.resolution = resolution;
        rows.push_back(row);
        beta_values.insert(beta);
        distance_values.insert(distance);
        categories[category_id] = category;
    }
    tree->ResetBranchAddresses();
    if (rows.empty()) {
        std::cerr << "ERROR: scan_summary is empty." << std::endl;
        return;
    }

    std::vector<int> categories_to_draw;
    if (requested_category == "all") {
        for (const auto& item : categories) categories_to_draw.push_back(item.first);
    } else {
        for (const auto& item : categories) {
            if (item.second == requested_category) categories_to_draw.push_back(item.first);
        }
        if (categories_to_draw.empty()) {
            std::cerr << "ERROR: category '" << requested_category
                      << "' is not present in the summary." << std::endl;
            return;
        }
    }

    const std::vector<double> beta_edges = makeBinEdges(beta_values);
    const std::vector<double> distance_edges = makeBinEdges(distance_values);
    if (beta_edges.size() < 2 || distance_edges.size() < 2) {
        std::cerr << "ERROR: cannot construct scan axes." << std::endl;
        return;
    }

    gROOT->SetBatch(kTRUE);
    gStyle->SetOptStat(0);
    gStyle->SetPaintTextFormat(".3f");
    gSystem->mkdir(output_path.c_str(), true);
    const std::string root_output_path = output_path + "/beta_td_scan_plots.root";
    TFile output(root_output_path.c_str(), "RECREATE");
    if (output.IsZombie()) {
        std::cerr << "ERROR: cannot create " << root_output_path << std::endl;
        return;
    }

    for (int selected_category : categories_to_draw) {
        const std::string label = categories[selected_category];
        const std::string clean_label = cleanName(label);
        const std::string title_suffix =
            rows.front().energy_gev >= 0
                ? ", " + std::to_string(rows.front().energy_gev) + " GeV"
                : "";

        TH2D legacy_efficiency_map(
            ("legacy_efficiency_" + clean_label).c_str(),
            ("Efficiency: " + label + title_suffix).c_str(),
            static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
            static_cast<int>(distance_edges.size() - 1), distance_edges.data());
        TH2D legacy_purity_map(
            ("legacy_purity_" + clean_label).c_str(),
            ("Purity: " + label + title_suffix).c_str(),
            static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
            static_cast<int>(distance_edges.size() - 1), distance_edges.data());
        TH2D cedric_efficiency_map(
            ("cedric_efficiency_" + clean_label).c_str(),
            ("Cedric efficiency: " + label + title_suffix).c_str(),
            static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
            static_cast<int>(distance_edges.size() - 1), distance_edges.data());
        TH2D cedric_purity_map(
            ("cedric_purity_" + clean_label).c_str(),
            ("Cedric purity: " + label + title_suffix).c_str(),
            static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
            static_cast<int>(distance_edges.size() - 1), distance_edges.data());

        configureMetricMap(&legacy_efficiency_map, beta_values, distance_values);
        configureMetricMap(&legacy_purity_map, beta_values, distance_values);
        configureMetricMap(&cedric_efficiency_map, beta_values, distance_values);
        configureMetricMap(&cedric_purity_map, beta_values, distance_values);

        double best_legacy_score = -1.0;
        double best_cedric_score = -1.0;
        const ScanRow* best_legacy = nullptr;
        const ScanRow* best_cedric = nullptr;
        for (const ScanRow& row : rows) {
            if (row.category_id != selected_category) continue;
            const int xbin = legacy_efficiency_map.GetXaxis()->FindBin(row.beta);
            const int ybin = legacy_efficiency_map.GetYaxis()->FindBin(row.distance);
            if (row.legacy_efficiency >= 0.0) {
                legacy_efficiency_map.SetBinContent(xbin, ybin, row.legacy_efficiency);
            }
            if (row.legacy_purity >= 0.0) {
                legacy_purity_map.SetBinContent(xbin, ybin, row.legacy_purity);
            }
            if (row.cedric_efficiency >= 0.0) {
                cedric_efficiency_map.SetBinContent(xbin, ybin, row.cedric_efficiency);
            }
            if (row.cedric_purity >= 0.0) {
                cedric_purity_map.SetBinContent(xbin, ybin, row.cedric_purity);
            }

            const double legacy_score = harmonicMean(
                row.legacy_efficiency, row.legacy_purity);
            if (legacy_score > best_legacy_score) {
                best_legacy_score = legacy_score;
                best_legacy = &row;
            }
            const double cedric_score = row.cedric_clustering_quality;
            if (cedric_score > best_cedric_score) {
                best_cedric_score = cedric_score;
                best_cedric = &row;
            }
        }

        const std::string canvas_name = "effpur_" + clean_label;
        TCanvas canvas(canvas_name.c_str(), canvas_name.c_str(), 1500, 1200);
        canvas.Divide(2, 2);
        TH2D* maps[4] = {
            &legacy_efficiency_map, &legacy_purity_map,
            &cedric_efficiency_map, &cedric_purity_map
        };
        for (int pad = 0; pad < 4; ++pad) {
            canvas.cd(pad + 1);
            gPad->SetLeftMargin(0.13);
            gPad->SetRightMargin(0.15);
            gPad->SetBottomMargin(0.13);
            gPad->SetTopMargin(0.10);
            maps[pad]->Draw("COLZ TEXT");
        }
        output.cd();
        legacy_efficiency_map.Write();
        legacy_purity_map.Write();
        cedric_efficiency_map.Write();
        cedric_purity_map.Write();
        canvas.Write();
        canvas.SaveAs((output_path + "/" + canvas_name + ".png").c_str());

        if (has_cedric_quality_branches && best_cedric) {
            TH2D clustering_quality_map(
                ("cedric_clustering_quality_" + clean_label).c_str(),
                ("Cedric clustering quality: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D confusion_quality_map(
                ("cedric_confusion_quality_" + clean_label).c_str(),
                ("Confusion quality: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D charged_quality_map(
                ("cedric_charged_quality_" + clean_label).c_str(),
                ("Charged quality: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D neutral_quality_map(
                ("cedric_neutral_quality_" + clean_label).c_str(),
                ("Neutral quality: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D assignment_quality_map(
                ("cedric_assignment_quality_" + clean_label).c_str(),
                ("Assignment quality: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D charged_efficiency_map(
                ("cedric_charged_efficiency_" + clean_label).c_str(),
                ("Charged efficiency: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D charged_purity_map(
                ("cedric_charged_purity_" + clean_label).c_str(),
                ("Charged purity: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D charged_confusion_map(
                ("cedric_charged_to_neutral_confusion_" + clean_label).c_str(),
                ("Charged to neutral confusion: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D neutral_efficiency_map(
                ("cedric_neutral_efficiency_" + clean_label).c_str(),
                ("Neutral efficiency: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D neutral_purity_map(
                ("cedric_neutral_purity_" + clean_label).c_str(),
                ("Neutral purity: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D neutral_confusion_map(
                ("cedric_neutral_to_charged_confusion_" + clean_label).c_str(),
                ("Neutral to charged confusion: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D coverage_map(
                ("cedric_coverage_" + clean_label).c_str(),
                ("Coverage: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D precision_map(
                ("cedric_precision_" + clean_label).c_str(),
                ("Precision: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());
            TH2D spurious_rate_map(
                ("cedric_spurious_cluster_rate_" + clean_label).c_str(),
                ("Spurious cluster rate: " + label + title_suffix).c_str(),
                static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
                static_cast<int>(distance_edges.size() - 1), distance_edges.data());

            TH2D* cedric_maps[] = {
                &clustering_quality_map, &confusion_quality_map,
                &charged_quality_map, &neutral_quality_map,
                &assignment_quality_map, &charged_efficiency_map,
                &charged_purity_map, &charged_confusion_map,
                &neutral_efficiency_map, &neutral_purity_map,
                &neutral_confusion_map, &coverage_map, &precision_map,
                &spurious_rate_map
            };
            for (TH2D* map : cedric_maps) {
                configureMetricMap(map, beta_values, distance_values);
            }

            for (const ScanRow& row : rows) {
                if (row.category_id != selected_category) continue;
                const int xbin = clustering_quality_map.GetXaxis()->FindBin(row.beta);
                const int ybin = clustering_quality_map.GetYaxis()->FindBin(row.distance);
                const double values[] = {
                    row.cedric_clustering_quality, row.cedric_confusion_quality,
                    row.cedric_charged_quality, row.cedric_neutral_quality,
                    row.cedric_assignment_quality, row.cedric_charged_efficiency,
                    row.cedric_charged_purity,
                    row.cedric_charged_to_neutral_confusion,
                    row.cedric_neutral_efficiency, row.cedric_neutral_purity,
                    row.cedric_neutral_to_charged_confusion, row.cedric_coverage,
                    row.cedric_precision, row.cedric_spurious_cluster_rate
                };
                for (std::size_t index = 0;
                     index < sizeof(cedric_maps) / sizeof(cedric_maps[0]); ++index) {
                    if (values[index] >= 0.0) {
                        cedric_maps[index]->SetBinContent(xbin, ybin, values[index]);
                    }
                }
            }

            const bool on_boundary =
                best_cedric->beta == *beta_values.begin() ||
                best_cedric->beta == *beta_values.rbegin() ||
                best_cedric->distance == *distance_values.begin() ||
                best_cedric->distance == *distance_values.rbegin();
            std::string quality_title =
                "Cedric clustering quality: " + label + title_suffix +
                ", best #beta=" + thresholdLabel(best_cedric->beta) +
                ", distance=" + thresholdLabel(best_cedric->distance);
            if (on_boundary) quality_title += " (boundary)";
            clustering_quality_map.SetTitle(quality_title.c_str());

            const std::string quality_canvas_name =
                "cedric_clustering_quality_" + clean_label;
            TCanvas quality_canvas(
                quality_canvas_name.c_str(), quality_canvas_name.c_str(), 850, 720);
            gPad->SetLeftMargin(0.13);
            gPad->SetRightMargin(0.16);
            gPad->SetBottomMargin(0.13);
            gPad->SetTopMargin(0.12);
            clustering_quality_map.Draw("COLZ TEXT");
            TMarker best_marker(best_cedric->beta, best_cedric->distance, 29);
            best_marker.SetMarkerColor(kRed + 1);
            best_marker.SetMarkerSize(2.2);
            best_marker.Draw("SAME");
            output.cd();
            clustering_quality_map.Write();
            quality_canvas.Write();
            quality_canvas.SaveAs(
                (output_path + "/" + quality_canvas_name + ".png").c_str());

            const std::string component_canvas_name =
                "cedric_quality_components_" + clean_label;
            TCanvas component_canvas(
                component_canvas_name.c_str(), component_canvas_name.c_str(), 1500, 1200);
            component_canvas.Divide(2, 2);
            TH2D* component_maps[] = {
                &confusion_quality_map, &charged_quality_map,
                &neutral_quality_map, &assignment_quality_map
            };
            for (int pad = 0; pad < 4; ++pad) {
                component_canvas.cd(pad + 1);
                gPad->SetLeftMargin(0.13);
                gPad->SetRightMargin(0.15);
                gPad->SetBottomMargin(0.13);
                gPad->SetTopMargin(0.10);
                component_maps[pad]->Draw("COLZ TEXT");
            }
            output.cd();
            for (TH2D* map : component_maps) map->Write();
            component_canvas.Write();
            component_canvas.SaveAs(
                (output_path + "/" + component_canvas_name + ".png").c_str());

            const std::string energy_canvas_name =
                "cedric_energy_metrics_" + clean_label;
            TCanvas energy_canvas(
                energy_canvas_name.c_str(), energy_canvas_name.c_str(), 2000, 1000);
            energy_canvas.Divide(4, 2);
            TH2D* energy_maps[] = {
                &cedric_efficiency_map, &cedric_purity_map,
                &charged_efficiency_map, &charged_purity_map,
                &charged_confusion_map, &neutral_efficiency_map,
                &neutral_purity_map, &neutral_confusion_map
            };
            for (int pad = 0; pad < 8; ++pad) {
                energy_canvas.cd(pad + 1);
                gPad->SetLeftMargin(0.14);
                gPad->SetRightMargin(0.16);
                gPad->SetBottomMargin(0.14);
                gPad->SetTopMargin(0.10);
                energy_maps[pad]->Draw("COLZ TEXT");
            }
            output.cd();
            for (int index = 2; index < 8; ++index) energy_maps[index]->Write();
            energy_canvas.Write();
            energy_canvas.SaveAs(
                (output_path + "/" + energy_canvas_name + ".png").c_str());

            const std::string assignment_canvas_name =
                "cedric_assignment_metrics_" + clean_label;
            TCanvas assignment_canvas(
                assignment_canvas_name.c_str(), assignment_canvas_name.c_str(), 1800, 600);
            assignment_canvas.Divide(3, 1);
            TH2D* assignment_maps[] = {
                &coverage_map, &precision_map, &spurious_rate_map
            };
            for (int pad = 0; pad < 3; ++pad) {
                assignment_canvas.cd(pad + 1);
                gPad->SetLeftMargin(0.14);
                gPad->SetRightMargin(0.16);
                gPad->SetBottomMargin(0.14);
                gPad->SetTopMargin(0.10);
                assignment_maps[pad]->Draw("COLZ TEXT");
            }
            output.cd();
            for (TH2D* map : assignment_maps) map->Write();
            assignment_canvas.Write();
            assignment_canvas.SaveAs(
                (output_path + "/" + assignment_canvas_name + ".png").c_str());
        } else if (has_cedric_quality_branches) {
            std::cerr << "WARNING: no valid Cedric clustering quality for category '"
                      << label << "'. The input scan ROOT files probably do not "
                      << "contain usable hit-level prediction entries." << std::endl;
        }

        if (best_legacy) {
            std::cout << label << " best legacy harmonic mean = "
                      << best_legacy_score << " at beta=" << best_legacy->beta
                      << ", distance=" << best_legacy->distance << std::endl;
        }
        if (best_cedric) {
            std::cout << label << " best Cedric clustering quality = "
                      << best_cedric_score << " at beta=" << best_cedric->beta
                      << ", distance=" << best_cedric->distance << std::endl;
            std::cout << "  score inputs: efficiency=" << best_cedric->cedric_efficiency
                      << ", purity=" << best_cedric->cedric_purity
                      << ", confusion_quality="
                      << best_cedric->cedric_confusion_quality
                      << ", coverage=" << best_cedric->cedric_coverage
                      << ", precision=" << best_cedric->cedric_precision << std::endl;
            std::cout << "  charged: efficiency="
                      << best_cedric->cedric_charged_efficiency
                      << ", purity=" << best_cedric->cedric_charged_purity
                      << ", to-neutral confusion="
                      << best_cedric->cedric_charged_to_neutral_confusion << std::endl;
            std::cout << "  neutral: efficiency="
                      << best_cedric->cedric_neutral_efficiency
                      << ", purity=" << best_cedric->cedric_neutral_purity
                      << ", to-charged confusion="
                      << best_cedric->cedric_neutral_to_charged_confusion << std::endl;
            const bool best_on_boundary =
                best_cedric->beta == *beta_values.begin() ||
                best_cedric->beta == *beta_values.rbegin() ||
                best_cedric->distance == *distance_values.begin() ||
                best_cedric->distance == *distance_values.rbegin();
            if (best_on_boundary) {
                std::cout << "  WARNING: Cedric argmax is on the scan boundary; "
                          << "extend the scan before treating it as an interior optimum."
                          << std::endl;
            }
        }
    }

    // Cedric's threshold scanner also shows the beta-score separation used to
    // interpret the selected beta cut.  The summary maker accumulates this
    // once per input H5 file even though each threshold ROOT repeats the tree.
    const ScanRow* inclusive_best_cedric = nullptr;
    double inclusive_best_quality = -1.0;
    if (has_cedric_quality_branches) {
        for (const ScanRow& row : rows) {
            if (row.category_id == 0 &&
                row.cedric_clustering_quality > inclusive_best_quality) {
                inclusive_best_quality = row.cedric_clustering_quality;
                inclusive_best_cedric = &row;
            }
        }
    }
    TH1D* beta_all_input = static_cast<TH1D*>(input.Get("cedric_beta_all_hits"));
    TH1D* beta_track_input =
        static_cast<TH1D*>(input.Get("cedric_beta_track_reference"));
    TH1D* beta_calo_input =
        static_cast<TH1D*>(input.Get("cedric_beta_calo_reference"));
    TH1D* beta_other_input =
        static_cast<TH1D*>(input.Get("cedric_beta_other_valid"));
    const bool has_beta_distribution =
        beta_all_input && beta_track_input && beta_calo_input && beta_other_input &&
        beta_all_input->Integral() > 0.0;
    if (inclusive_best_cedric && has_beta_distribution) {
        TH1D* beta_full[] = {
            static_cast<TH1D*>(beta_all_input->Clone("cedric_beta_all_hits_plot")),
            static_cast<TH1D*>(beta_track_input->Clone("cedric_beta_track_reference_plot")),
            static_cast<TH1D*>(beta_calo_input->Clone("cedric_beta_calo_reference_plot")),
            static_cast<TH1D*>(beta_other_input->Clone("cedric_beta_other_valid_plot"))
        };
        TH1D* beta_zoom[] = {
            static_cast<TH1D*>(beta_all_input->Clone("cedric_beta_all_hits_zoom")),
            static_cast<TH1D*>(beta_track_input->Clone("cedric_beta_track_reference_zoom")),
            static_cast<TH1D*>(beta_calo_input->Clone("cedric_beta_calo_reference_zoom")),
            static_cast<TH1D*>(beta_other_input->Clone("cedric_beta_other_valid_zoom"))
        };
        const int colors[] = {kBlack, kBlue + 1, kGreen + 2, kOrange + 7};
        const char* labels[] = {
            "All hits", "Truth representatives with track",
            "Calorimeter truth representatives", "Other valid hits"
        };
        for (int index = 0; index < 4; ++index) {
            beta_full[index]->SetDirectory(nullptr);
            beta_zoom[index]->SetDirectory(nullptr);
            beta_full[index]->SetLineColor(colors[index]);
            beta_zoom[index]->SetLineColor(colors[index]);
            beta_full[index]->SetLineWidth(index == 0 ? 2 : 1);
            beta_zoom[index]->SetLineWidth(index == 0 ? 2 : 1);
            beta_full[index]->SetStats(false);
            beta_zoom[index]->SetStats(false);
        }
        beta_full[0]->SetTitle("GNN beta-score distribution;GNN beta score;hits / 0.001");
        beta_zoom[0]->SetTitle(
            "Scanned beta-threshold region;GNN beta score;hits / 0.001");

        const double beta_min = *beta_values.begin();
        const double beta_max = *beta_values.rbegin();
        const double beta_margin = std::max(0.02, 0.4 * (beta_max - beta_min));
        const double zoom_min = std::max(0.0, beta_min - beta_margin);
        const double zoom_max = std::min(1.0, beta_max + beta_margin);
        beta_zoom[0]->GetXaxis()->SetRangeUser(zoom_min, zoom_max);

        TCanvas beta_canvas("cedric_beta_distribution", "cedric_beta_distribution",
                            1500, 600);
        beta_canvas.Divide(2, 1);
        TLegend beta_legend(0.12, 0.12, 0.68, 0.34);
        beta_legend.SetBorderSize(0);
        beta_legend.SetFillStyle(0);
        for (int pad = 0; pad < 2; ++pad) {
            beta_canvas.cd(pad + 1);
            gPad->SetLogy();
            gPad->SetLeftMargin(0.12);
            gPad->SetRightMargin(0.04);
            gPad->SetBottomMargin(0.13);
            TH1D** histograms = pad == 0 ? beta_full : beta_zoom;
            histograms[0]->SetMinimum(0.5);
            histograms[0]->Draw("HIST");
            for (int index = 1; index < 4; ++index) histograms[index]->Draw("HIST SAME");
            if (pad == 1) {
                for (int index = 0; index < 4; ++index) {
                    beta_legend.AddEntry(histograms[index], labels[index], "l");
                }
                beta_legend.AddEntry(static_cast<TObject*>(nullptr),
                    ("Selected #beta=" + thresholdLabel(inclusive_best_cedric->beta)).c_str(),
                    "");
                beta_legend.Draw();
            }
        }
        beta_canvas.cd(1);
        TLine best_beta_full(inclusive_best_cedric->beta, 0.5,
                             inclusive_best_cedric->beta,
                             std::max(1.0, beta_full[0]->GetMaximum()));
        best_beta_full.SetLineColor(kMagenta + 1);
        best_beta_full.SetLineStyle(2);
        best_beta_full.SetLineWidth(3);
        best_beta_full.Draw("SAME");
        beta_canvas.cd(2);
        TLine best_beta_zoom(inclusive_best_cedric->beta, 0.5,
                             inclusive_best_cedric->beta,
                             std::max(1.0, beta_zoom[0]->GetMaximum()));
        best_beta_zoom.SetLineColor(kMagenta + 1);
        best_beta_zoom.SetLineStyle(2);
        best_beta_zoom.SetLineWidth(3);
        best_beta_zoom.Draw("SAME");
        output.cd();
        for (TH1D* histogram : beta_full) histogram->Write();
        beta_canvas.Write();
        beta_canvas.SaveAs((output_path + "/cedric_beta_distribution.png").c_str());
    } else if (has_cedric_quality_branches) {
        std::cerr << "WARNING: Cedric beta-distribution plot was skipped because "
                  << "the summary has no usable pred_beta histogram or no valid "
                  << "inclusive clustering-quality point." << std::endl;
    }

    // Event resolution is category-independent and is plotted once.  There is
    // intentionally no Pandora overlay.
    TH2D response_map(
        "event_response_mean", "Mean event response;#beta threshold;distance threshold",
        static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
        static_cast<int>(distance_edges.size() - 1), distance_edges.data());
    TH2D resolution_map(
        "event_resolution", "Event energy resolution;#beta threshold;distance threshold",
        static_cast<int>(beta_edges.size() - 1), beta_edges.data(),
        static_cast<int>(distance_edges.size() - 1), distance_edges.data());
    configureMetricMap(&response_map, beta_values, distance_values, false);
    configureMetricMap(&resolution_map, beta_values, distance_values, false);
    response_map.GetZaxis()->SetTitle("Mean E_{reco}/E_{true}");
    resolution_map.GetZaxis()->SetTitle("RMS90/Mean90 [%]");
    response_map.SetMinimum(0.0);
    resolution_map.SetMinimum(0.0);

    double best_resolution_value = std::numeric_limits<double>::infinity();
    const ScanRow* best_resolution = nullptr;
    for (const ScanRow& row : rows) {
        if (row.category_id != 0) continue;
        const int xbin = response_map.GetXaxis()->FindBin(row.beta);
        const int ybin = response_map.GetYaxis()->FindBin(row.distance);
        if (row.response_mean >= 0.0) {
            response_map.SetBinContent(xbin, ybin, row.response_mean);
        }
        if (row.resolution >= 0.0) {
            resolution_map.SetBinContent(xbin, ybin, 100.0 * row.resolution);
            if (row.resolution < best_resolution_value) {
                best_resolution_value = row.resolution;
                best_resolution = &row;
            }
        }
    }

    TCanvas resolution_canvas("energy_resolution", "energy_resolution", 1500, 600);
    resolution_canvas.Divide(2, 1);
    resolution_canvas.cd(1);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.15);
    gPad->SetBottomMargin(0.13);
    gPad->SetTopMargin(0.10);
    response_map.Draw("COLZ TEXT");
    resolution_canvas.cd(2);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.15);
    gPad->SetBottomMargin(0.13);
    gPad->SetTopMargin(0.10);
    resolution_map.Draw("COLZ TEXT");
    output.cd();
    response_map.Write();
    resolution_map.Write();
    resolution_canvas.Write();
    resolution_canvas.SaveAs((output_path + "/energy_resolution.png").c_str());

    if (best_resolution) {
        std::cout << "Best event RMS90/Mean90 = "
                  << 100.0 * best_resolution_value << " %"
                  << " at beta=" << best_resolution->beta
                  << ", distance=" << best_resolution->distance << std::endl;
    }

    output.Close();
    input.Close();
    std::cout << "Wrote plots under " << output_path << std::endl;
}
