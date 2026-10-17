#ifndef PID_COMPARISON_DISTRIBUTIONS_H
#define PID_COMPARISON_DISTRIBUTIONS_H

#include "pid_clustering_summary.h"
#include "TFitResult.h"
#include "TFitResultPtr.h"
#include "TLine.h"

// Included after the main plotter's input and canvas helpers.
namespace pid_jer_plot {
static const char* const kRegions[3] = {"central", "forward", "inclusive"};
static const char* const kRegionLabels[3] = {
    "|cos(#theta_{thrust})| < 0.7",
    "0.7 < |cos(#theta_{thrust})| < 0.98", "No angular cut"
};
static const char* const kRatioDefinition =
    "Event Ereco / visible MC event/MC_dijet_energy; no neutrino correction; 100 bins [0.5,1.5); normalized by all valid selected events including under/overflow";
static const int kRatioColors[3] = {kOrange+7,kAzure+2,kGreen+2};

struct RatioSample {
    std::array<std::unique_ptr<TH1F>,3> hist;
    Long64_t invalid_energy = 0, invalid_angle = 0;
    std::string path;
    bool present = false;
};
using Ratios = std::array<std::array<RatioSample,5>,3>;

bool loadRatios(RatioSample& result, const Sample& sample, int method, int energy) {
    if (!sample.present) return false;
    if (method == 2 && sample.source_kind == "pandora_text_reference") {
        if (!sample.pandora_snapshot) throw std::runtime_error("Missing Pandora text snapshot");
        const auto& saved = sample.pandora_snapshot->ratios[pid_pandora_reference::energyIndex(energy)];
        result.path = sample.pandora_snapshot->path;
        for (int r=0; r<3; ++r) {
            result.hist[r].reset(new TH1F(Form("event_energy_ratio_pandora_%dGeV_%s",energy,kRegions[r]),
                ";E_{reco}^{event} / E_{true,visible}^{event};Event Fraction",100,.5,1.5));
            auto& hist = *result.hist[r]; hist.SetDirectory(nullptr); hist.Sumw2();
            for (int bin=0; bin<102; ++bin) {
                hist.SetBinContent(bin,saved[r].contents[bin]);
                hist.GetSumw2()->SetAt(saved[r].sumw2[bin],bin);
            }
            double stats[4]; std::copy(saved[r].stats.begin(),saved[r].stats.end(),stats);
            hist.PutStats(stats); hist.SetEntries(saved[r].entries);
        }
        result.invalid_energy = saved[0].invalid_energy;
        result.invalid_angle = saved[0].invalid_angle;
        result.present = true;
        return true;
    }
    result.path = sample.event_path.empty() ? sample.path : sample.event_path;
    if (gSystem->AccessPathName(result.path.c_str())) {
        std::cerr << "Missing saved event energies (ratio plots skipped): " << result.path << std::endl;
        return false;
    }
    TDirectory::TContext context;
    TFile file(result.path.c_str(), "READ");
    if (file.IsZombie()) throw std::runtime_error("Cannot read "+result.path);
    if (sample.source_kind == "pandora_h5_reference") {
        auto* beam = dynamic_cast<TParameter<int>*>(file.Get("beam_energy_GeV"));
        if (named(file,"schema") != "pid-pandora-event-energy-1" || !beam || beam->GetVal() != energy ||
            named(file,"angle_definition") != pid_jer::kAngleDefinition ||
            named(file,"truth_definition") != "visible MC event/MC_dijet_energy")
            throw std::runtime_error("Incompatible Pandora event energy cache: "+result.path);
    }
    auto* events = dynamic_cast<TTree*>(file.Get("events"));
    if (!events) {
        std::cerr << "No events tree (ratio plots skipped): " << result.path << std::endl;
        return false;
    }
    double thrust = 0., reco = 0., truth = 0.;
    events->SetBranchStatus("*",0);
    auto bind = [&](const char* key, double* address) {
        events->SetBranchStatus(key,1);
        if (!events->GetBranch(key) || events->SetBranchAddress(key,address) < 0)
            throw std::runtime_error(std::string("Missing saved event field: ")+key+" in "+result.path);
    };
    bind("thrust",&thrust); bind("reco_energy",&reco); bind("truth_energy",&truth);
    for (int r=0; r<3; ++r) {
        result.hist[r].reset(new TH1F(Form("event_energy_ratio_%s_%dGeV_%s",kMethods[method],energy,kRegions[r]),
            ";E_{reco}^{event} / E_{true,visible}^{event};Event Fraction",100,.5,1.5));
        result.hist[r]->SetDirectory(nullptr); result.hist[r]->Sumw2();
    }
    for (Long64_t i=0, n=events->GetEntries(); i<n; ++i) {
        if (events->GetEntry(i) <= 0) throw std::runtime_error("Unreadable saved event: "+result.path);
        if (!std::isfinite(reco) || reco < 0. || !std::isfinite(truth) || !(truth > 0.) || !std::isfinite(reco/truth)) {
            ++result.invalid_energy; continue;
        }
        const double ratio = reco/truth;
        result.hist[2]->Fill(ratio);
        if (!std::isfinite(thrust) || thrust < 0. || thrust > 1.) { ++result.invalid_angle; continue; }
        // Same separately pooled central/forward cuts as the resolution plots.
        if (thrust < .7f) result.hist[0]->Fill(ratio);
        if (thrust > .7f && thrust < .98f) result.hist[1]->Fill(ratio);
    }
    events->ResetBranchAddresses();
    for (auto& hist : result.hist) if (hist->GetEntries() > 0.) hist->Scale(1./hist->GetEntries());
    result.present = true;
    return true;
}

struct RatioFit {
    int beam_energy = 0, status = -1;
    std::string method, region, source;
    Long64_t entries = 0, invalid_energy = 0, invalid_angle = 0;
    double underflow = 0., overflow = 0.;
    double mean = NAN, sigma = NAN, sigma_over_mean_percent = NAN;
    double mean_error = NAN, sigma_error = NAN;
    bool valid = false;
};
RatioFit fitRatio(TH1F& hist) {
    RatioFit fit;
    fit.entries = hist.GetEntries();
    fit.underflow = hist.GetBinContent(0); fit.overflow = hist.GetBinContent(hist.GetNbinsX()+1);
    if (fit.entries < 30) return fit;
    TF1 gaussian((std::string(hist.GetName())+"_fit").c_str(),"gaus",.9,1.1);
    gaussian.SetParameters(hist.GetMaximum(),1.,.05);
    gaussian.SetParLimits(1,.9,1.1); gaussian.SetParLimits(2,.001,.3);
    const TFitResultPtr result = hist.Fit(&gaussian,"QNR0S");
    fit.status = int(result);
    if (fit.status || !result.Get() || !(gaussian.GetParameter(1)>0.) ||
        !(gaussian.GetParameter(2)>0.)) return fit;
    fit.mean = gaussian.GetParameter(1); fit.sigma = gaussian.GetParameter(2);
    fit.mean_error = gaussian.GetParError(1); fit.sigma_error = gaussian.GetParError(2);
    fit.sigma_over_mean_percent = 100.*fit.sigma/fit.mean;
    fit.valid = std::isfinite(fit.sigma_over_mean_percent);
    return fit;
}

void drawRatios(const Samples& samples, TFile& output, const std::string& directory) {
    Ratios ratios;
    TTree fits("event_energy_ratio_fits","Gaussian fit over [0.9,1.1]; sigma/mu without sqrt(2)");
    RatioFit row;
    fits.Branch("beam_energy_GeV",&row.beam_energy,"beam_energy_GeV/I");
    fits.Branch("method",&row.method); fits.Branch("region",&row.region); fits.Branch("source",&row.source);
    fits.Branch("entries",&row.entries,"entries/L"); fits.Branch("invalid_energy",&row.invalid_energy,"invalid_energy/L");
    fits.Branch("invalid_angle",&row.invalid_angle,"invalid_angle/L");
    fits.Branch("underflow_fraction",&row.underflow,"underflow_fraction/D");
    fits.Branch("overflow_fraction",&row.overflow,"overflow_fraction/D");
    fits.Branch("fit_status",&row.status,"fit_status/I"); fits.Branch("fit_valid",&row.valid,"fit_valid/O");
    fits.Branch("mean",&row.mean,"mean/D"); fits.Branch("sigma",&row.sigma,"sigma/D");
    fits.Branch("mean_error",&row.mean_error,"mean_error/D"); fits.Branch("sigma_error",&row.sigma_error,"sigma_error/D");
    fits.Branch("sigma_over_mean_percent",&row.sigma_over_mean_percent,"sigma_over_mean_percent/D");
    TNamed("event_energy_ratio_definition",kRatioDefinition).Write();
    TNamed("event_energy_ratio_fit_definition","Gaussian [0.9,1.1]; sigma/mu [%], no sqrt(2); failed fits marked invalid").Write();
    for (int m=0; m<3; ++m) for (int e=0; e<5; ++e) loadRatios(ratios[m][e],samples[m][e],m,kEnergies[e]);
    for (int e=0; e<5; ++e) for (int r=0; r<3; ++r) {
        const std::string name = Form("event_energy_ratio_%dGeV_%s",kEnergies[e],kRegions[r]);
        double ymax = 0.; int available = 0;
        for (int m=0; m<3; ++m) if (ratios[m][e].present && ratios[m][e].hist[r]->GetEntries()) {
            ymax = std::max(ymax,double(ratios[m][e].hist[r]->GetMaximum())); ++available;
        }
        if (!available) continue;
        TCanvas canvas(("pid_"+name).c_str(),"Event energy response",1200,750);
        stylePad(&canvas); canvas.SetGridx(); canvas.SetGridy();
        auto* frame = canvas.DrawFrame(.5,0.,1.5,1.25*ymax);
        frame->SetTitle(";E_{reco}^{event} / E_{true,visible}^{event};Event Fraction"); styleFrame(frame);
        TLegend legend(.16,.56,.51,.89); legend.SetBorderSize(0); legend.SetFillStyle(0);
        legend.SetTextFont(42); legend.SetTextSize(.029);
        for (int m=0; m<3; ++m) {
            auto& data = ratios[m][e];
            if (!data.present || !data.hist[r]->GetEntries()) continue;
            auto& hist = *data.hist[r];
            hist.SetLineColor(kRatioColors[m]); hist.SetLineWidth(3); hist.Draw("HIST SAME");
            row = fitRatio(hist); row.beam_energy = kEnergies[e]; row.method = kMethods[m]; row.region = kRegions[r];
            row.source = data.path; row.invalid_energy = data.invalid_energy; row.invalid_angle = data.invalid_angle;
            fits.Fill();
            legend.AddEntry(&hist,kLabels[m],"l");
            legend.AddEntry(static_cast<TObject*>(nullptr),row.valid
                ? Form("#sigma/#mu = %.2f%%  (N = %lld)",row.sigma_over_mean_percent,row.entries)
                : Form("Gaussian fit unavailable  (N = %lld)",row.entries),"");
            output.cd(); hist.Write();
        }
        legend.Draw();
        TLatex label; label.SetNDC(); label.SetTextFont(42); label.SetTextSize(.035);
        label.DrawLatex(.16,.95,Form("Z/#gamma* #rightarrow q#bar{q} (q=u,d,s), #sqrt{s}=%d GeV",kEnergies[e]));
        label.SetTextSize(.029); label.DrawLatex(.57,.86,kRegionLabels[r]);
        label.SetTextSize(.023); label.DrawLatex(.57,.80,"Gaussian fit: 0.9 < E_{reco}/E_{true} < 1.1");
        finish(canvas,output,directory,name.c_str());
    }
    output.cd(); fits.Write();
}

struct ClusteringSample {
    bool present = false;
    std::string path, source_kind = "single_energy_summary";
    std::array<std::array<pid_cluster::Row,6>,2> rows;
    std::array<std::array<bool,6>,2> found{};
};
using ClusteringSamples = std::array<std::array<ClusteringSample,5>,3>;

// Original-definition means only; the text file has no Cedric means or counts.
bool loadSavedPandoraBars(ClusteringSample& sample, int energy_index,
                         const char* snapshotFile = nullptr) {
    const std::string snapshot_path = snapshotFile ? snapshotFile : kPandoraSnapshotFile;
    if (snapshot_path.empty() || gSystem->AccessPathName(snapshot_path.c_str())) {
        std::cerr << "Missing Pandora text means (bar series skipped)" << std::endl;
        return false;
    }
    const auto snapshot = pid_pandora_reference::load(snapshot_path);
    sample.present = true; sample.source_kind = "saved_plot_means_no_counts";
    sample.path = snapshot.path;
    for (int p=0; p<5; ++p) {
        auto& row = sample.rows[0][p]; row.particle = p;
        row.efficiency.numerator = snapshot.bars[energy_index][p].efficiency;
        row.efficiency.denominator = 1.;
        row.purity.numerator = snapshot.bars[energy_index][p].purity;
        row.purity.denominator = 1.;
        // A denominator of 1 encodes a saved mean, not an event/energy count.
        sample.found[0][p] = true;
    }
    return true;
}

bool loadClustering(ClusteringSample& result, const std::string& path, int method, int energy, double beta) {
    result.path = path;
    if (gSystem->AccessPathName(path.c_str())) {
        std::cerr << "Missing clustering summary (bar series skipped): " << path << std::endl; return false;
    }
    TDirectory::TContext context;
    TFile input(path.c_str(),"READ");
    if (input.IsZombie()) throw std::runtime_error("Cannot read "+path);
    auto* beam = dynamic_cast<TParameter<int>*>(input.Get("beam_energy_GeV"));
    auto* cut = dynamic_cast<TParameter<double>*>(input.Get("beta_threshold"));
    if (named(input,"schema") != pid_cluster::kSchema || named(input,"method") != kMethods[method] ||
        named(input,"original_definition") != pid_cluster::kOriginalDefinition ||
        named(input,"cedric_definition") != pid_cluster::kCedricDefinition ||
        !beam || beam->GetVal() != energy || !cut ||
        (method != 2 && (!std::isfinite(cut->GetVal()) || (std::isfinite(beta) && std::fabs(cut->GetVal()-beta)>1e-12))))
        throw std::runtime_error("Incompatible clustering summary: "+path);
    auto* tree = dynamic_cast<TTree*>(input.Get("clustering"));
    if (!tree) throw std::runtime_error("Missing clustering table: "+path);
    pid_cluster::Row row; pid_cluster::bind(*tree,row,false);
    for (Long64_t i=0; i<tree->GetEntries(); ++i) {
        if (tree->GetEntry(i)<=0 || row.definition<0 || row.definition>1 || row.particle<0 || row.particle>5 ||
            result.found[row.definition][row.particle]) throw std::runtime_error("Invalid/duplicate clustering row: "+path);
        for (const auto* value : {&row.efficiency,&row.purity})
            if (!std::isfinite(value->numerator) || !std::isfinite(value->denominator) ||
                value->numerator<0. || value->denominator<0. || value->entries<0)
                throw std::runtime_error("Invalid clustering statistics: "+path);
        result.found[row.definition][row.particle] = true; result.rows[row.definition][row.particle] = row;
    }
    tree->ResetBranchAddresses(); result.present = true; return true;
}

void drawClustering(const ClusteringSamples& samples, int method, int definition,
                    TFile& output, const std::string& directory) {
    const int colors[5] = {kRed+1,kOrange+1,kBlue+1,kGreen+2,kViolet-6};
    const std::string name = std::string("efficiency_purity_bar_")+pid_cluster::kDefinitions[definition]+"_"+kMethods[method];
    bool any = false;
    for (int e=0; e<5; ++e) any = any || samples[method][e].present;
    if (!any) return;
    TCanvas canvas(("pid_"+name).c_str(),"Clustering efficiency and purity",1500,950); canvas.Divide(1,2,.001,.001);
    std::vector<std::unique_ptr<TH1F>> histograms;
    std::vector<std::unique_ptr<TGraph>> graphs;
    std::vector<std::unique_ptr<TLine>> ticks;
    std::vector<std::unique_ptr<TLegend>> legends;
    for (int panel=0; panel<2; ++panel) {
        const size_t first_tick = ticks.size();
        auto* pad = static_cast<TPad*>(canvas.cd(panel+1));
        pad->SetLeftMargin(.09); pad->SetRightMargin(.21); pad->SetBottomMargin(.16); pad->SetTopMargin(.10);
        pad->SetGridy(); pad->SetTicks(1,1);
        double minimum = panel == 0 ? 60. : 50.;
        const double maximum = 100.;
        for (int m : {method,2}) for (int e=0; e<5; ++e) for (int p=0; p<5; ++p) {
            const auto& sample = samples[m][e];
            if (!sample.present || !sample.found[definition][p]) continue;
            const auto& row = sample.rows[definition][p]; const auto& metric = panel ? row.purity : row.efficiency;
            if (metric.valid()) minimum = std::min(minimum,100.*metric.value()-5.);
        }
        minimum = std::max(0.,5.*std::floor(minimum/5.));
        std::unique_ptr<TH1F> frame(new TH1F((name+Form("_frame_%d",panel)).c_str(),"",5,0.,5.));
        frame->SetDirectory(nullptr); frame->SetMinimum(minimum); frame->SetMaximum(maximum);
        frame->GetYaxis()->SetTitle(panel ? "Purity [%]" : "Efficiency [%]");
        frame->GetYaxis()->SetTitleSize(0.); frame->GetYaxis()->SetLabelSize(.045);
        frame->GetXaxis()->SetLabelSize(0.);
        for (int p=0; p<5; ++p) frame->GetXaxis()->SetBinLabel(p+1,pid_cluster::kParticles[p]);
        frame->Draw(); histograms.push_back(std::move(frame));
        bool has_pandora = false;
        std::unique_ptr<TLegend> legend(new TLegend(.805,.28,.995,.86));
        legend->SetBorderSize(0); legend->SetFillStyle(0); legend->SetTextFont(42); legend->SetTextSize(.043);
        for (int e=0; e<5; ++e) {
            const double offset = .095+.17*e;
            std::unique_ptr<TH1F> bar(new TH1F((name+Form("_%s_%dGeV",panel ? "purity" : "efficiency",kEnergies[e])).c_str(),"",5,0.,5.));
            bar->SetDirectory(nullptr); bar->SetFillColor(colors[e]); bar->SetLineColor(colors[e]);
            bar->SetBarWidth(.13); bar->SetBarOffset(offset);
            std::unique_ptr<TGraph> halo(new TGraph()), points(new TGraph());
            points->SetName((name+Form("_%s_pandora_%dGeV",panel ? "purity" : "efficiency",kEnergies[e])).c_str());
            halo->SetMarkerStyle(20); halo->SetMarkerColor(kWhite); halo->SetMarkerSize(1.65);
            points->SetMarkerStyle(20); points->SetMarkerColor(kBlack); points->SetMarkerSize(1.05);
            for (int p=0; p<5; ++p) {
                for (int m : {method,2}) {
                    const auto& sample = samples[m][e];
                    if (!sample.present || !sample.found[definition][p]) continue;
                    const auto& row = sample.rows[definition][p]; const auto& value = panel ? row.purity : row.efficiency;
                    if (!value.valid()) continue;
                    if (m == method) bar->SetBinContent(p+1,100.*value.value());
                    else {
                        const double x = p+offset+.065, y = 100.*value.value();
                        halo->SetPoint(halo->GetN(),x,y); points->SetPoint(points->GetN(),x,y);
                        has_pandora = true;
                        std::unique_ptr<TLine> tick(new TLine(x-.04,y,x+.04,y)); tick->SetLineWidth(2); ticks.push_back(std::move(tick));
                    }
                }
            }
            bar->Draw("BAR SAME");
            if (points->GetN()) { halo->Draw("P SAME"); points->Draw("P SAME"); }
            legend->AddEntry(bar.get(),Form("%d GeV",kEnergies[e]),"f");
            output.cd(); bar->Write(); points->Write();
            histograms.push_back(std::move(bar)); graphs.push_back(std::move(halo)); graphs.push_back(std::move(points));
        }
        // Horizontal ticks are marker guides, as in plot_efficiency_purity_bar;
        // they do not represent statistical errors.
        for (size_t i=first_tick; i<ticks.size(); ++i) ticks[i]->Draw("SAME");
        if (has_pandora) legend->AddEntry(graphs.back().get(),"Pandora (black)","p");
        legend->Draw(); legends.push_back(std::move(legend));
        TLatex label; label.SetNDC(); label.SetTextFont(42); label.SetTextSize(.043);
        label.DrawLatex(.10,.94,Form("%s definition: %s bars%s",definition ? "Cedric" : "Original",kLabels[method],
            has_pandora ? " / Pandora points" : ""));
        if (!has_pandora) {
            label.SetTextSize(.033); label.DrawLatex(.81,.20,"Pandora: no saved"); label.DrawLatex(.81,.15,"Cedric reference");
        }
        // Place labels after all bars in pad coordinates. They stay visible
        // when ROOT paints bars below a nonzero displayed y-axis minimum.
        label.SetTextSize(.045); label.SetTextAlign(23);
        for (int p=0; p<5; ++p) label.DrawLatex(.09+.70*(p+.5)/5.,.135,pid_cluster::kParticles[p]);
        label.SetTextSize(.055); label.SetTextAlign(22); label.SetTextAngle(90.);
        label.DrawLatex(.025,.58,panel ? "Purity [%]" : "Efficiency [%]");
    }
    finish(canvas,output,directory,name.c_str());
}
} // namespace pid_jer_plot
#endif
