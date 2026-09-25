#ifndef PANDORA_EVAL_ANALYSIS_H
#define PANDORA_EVAL_ANALYSIS_H
#include "TChain.h"
#include "TChainElement.h"
#include "TFile.h"
#include "TH1F.h"
#include "TTree.h"
#include "TObjString.h"
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TSystem.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
// Unmodified LCPandoraAnalysis helper, commit 61b8993d121e84efa07bd74678b2ef687597ba77.
#include "pandora_reference/AnalysisHelper.cc"

namespace pandora_eval {
inline void Run(const char *input, const char *output) {
    TChain events("eval_events"), matches("eval_matches");
    if (!events.Add(input) || !events.GetEntries()) throw std::runtime_error("No eval_events input");
    matches.Add(input);
    std::string configuration;
    for (auto *item : *events.GetListOfFiles()) {
        const std::string path = static_cast<TChainElement *>(item)->GetTitle();
        if (path == output) throw std::runtime_error("Output must differ from input");
        TFile file(path.c_str());
        if (!file.Get("pandora_eval_metadata")) throw std::runtime_error("Missing comparison metadata: " + path);
        auto *config = dynamic_cast<TObjString *>(file.Get("comparison_configuration"));
        if (!config) throw std::runtime_error("Missing comparison configuration: " + path);
        if (!configuration.empty() && configuration != config->GetString().Data())
            throw std::runtime_error("Cannot mix different model/input/evaluation configurations");
        configuration = config->GetString().Data();
    }
    if (!gSystem->AccessPathName(output)) throw std::runtime_error("Output exists; choose a new file");
    TFile file(output,"RECREATE");
    if (file.IsZombie()) throw std::runtime_error("Cannot create analysis output");
    const std::array<float,14> edges{{0,.1,.2,.3,.4,.5,.6,.7,.8,.9,.925,.95,.975,1}};
    const std::array<std::string,7> names{{"pandora_reference", "pandora_common", "gnn_common",
        "pandora_three_way", "gnn_three_way", "truth_three_way", "truth_labelled_partial"}};
    std::array<TH1F *,7> central;
    std::array<std::array<TH1F *,13>,7> angular;
    for (size_t a=0;a<names.size();++a) {
        central[a] = new TH1F((names[a]+"_central").c_str(),
            (names[a]+";E_{visible}+E_{#nu,target} [GeV];Events").c_str(),100000,0,5000);
        for (int b=0;b<13;++b) angular[a][b]=new TH1F((names[a]+"_angle_"+std::to_string(b)).c_str(),
            (names[a]+";E_{visible} [GeV];Events").c_str(),100000,0,5000);
    }
    Long64_t q=0, ref=0, common=0, three=0, tv=0, complete=0, source_index=0, requested=0;
    Float_t pfo=0, nu=0, thrust=0;
    Double_t gnn=0, truth=0;
    std::string *source=nullptr;
    events.SetBranchAddress("qPdg",&q); events.SetBranchAddress("reference_valid",&ref);
    events.SetBranchAddress("common_valid",&common); events.SetBranchAddress("three_way_valid",&three);
    events.SetBranchAddress("truth_valid",&tv); events.SetBranchAddress("truth_complete",&complete);
    events.SetBranchAddress("pfoEnergyTotal",&pfo); events.SetBranchAddress("mcEnergyENu",&nu);
    events.SetBranchAddress("thrust",&thrust); events.SetBranchAddress("gnn_energy",&gnn);
    events.SetBranchAddress("truth_energy",&truth); events.SetBranchAddress("source_id",&source);
    events.SetBranchAddress("source_index",&source_index);
    events.SetBranchAddress("inference_requested",&requested);
    std::set<std::pair<std::string,Long64_t>> seen;
    std::set<std::pair<int,Long64_t>> selected_events;
    auto fill=[&](int a,float energy) {
        if (!std::isfinite(energy)) throw std::runtime_error("Non-finite energy in valid event");
        if (thrust<=.7f) central[a]->Fill(float(energy+nu));
        for (int b=0;b<13;++b) if (thrust>=edges[b] && thrust<edges[b+1]) angular[a][b]->Fill(energy);
    };
    Long64_t event_number=0;
    events.SetBranchAddress("event",&event_number);
    for (Long64_t i=0;i<events.GetEntries();++i) {
        events.GetEntry(i);
        if (!seen.emplace(*source,source_index).second) throw std::runtime_error("Duplicate source event in inputs");
        if (!ref || q<1 || q>3) continue;
        fill(0,pfo);
        if (common) {fill(1,pfo); fill(2,float(gnn));}
        if (three) {fill(3,pfo); fill(4,float(gnn)); fill(5,float(truth));}
        if (common && tv && !complete) fill(6,float(truth));
        if (common || !requested) selected_events.emplace(events.GetTreeNumber(),event_number);
    }
    events.ResetBranchAddresses();
    TTree results("resolution","Upstream CalculatePerformance, sqrt(2)*RMS90/Mean90 [%]");
    Int_t sample=0,bin=0,valid=0;
    Double_t entries=0,underflow=0,overflow=0;
    Float_t resolution=0,error=0;
    results.Branch("sample",&sample,"sample/I"); results.Branch("angle_bin",&bin,"angle_bin/I");
    results.Branch("valid",&valid,"valid/I"); results.Branch("entries",&entries,"entries/D");
    results.Branch("underflow",&underflow,"underflow/D"); results.Branch("overflow",&overflow,"overflow/D");
    results.Branch("resolution_percent",&resolution,"resolution_percent/F");
    results.Branch("error_percent",&error,"error_percent/F");
    std::array<TGraphErrors *,7> graphs;
    for (sample=0;sample<7;++sample) {
        graphs[sample]=new TGraphErrors(); graphs[sample]->SetName((names[sample]+"_resolution_vs_thrust").c_str());
        for (bin=-1;bin<13;++bin) {
            TH1F *h=bin<0 ? central[sample] : angular[sample][bin];
            entries=h->GetEntries(); underflow=h->GetBinContent(0); overflow=h->GetBinContent(h->GetNbinsX()+1);
            resolution=error=std::numeric_limits<float>::quiet_NaN();
            valid=entries>=5 && underflow==0 && overflow==0;
            // Do not turn a <5-event skipped result into a zero-resolution point.
            if (valid) {
                pandora_analysis::AnalysisHelper::CalculatePerformance(h,resolution,error,true,false);
                valid=std::isfinite(resolution) && resolution<std::numeric_limits<float>::max();
            }
            results.Fill();
            if (valid && bin>=0) {
                int point=graphs[sample]->GetN();
                graphs[sample]->SetPoint(point,(edges[bin]+edges[bin+1])*.5,resolution);
                graphs[sample]->SetPointError(point,(edges[bin+1]-edges[bin])*.5,error);
            }
        }
        graphs[sample]->Write();
    }
    // Particle overlap diagnostics are explicitly conditional on labelled
    // inputs. Include unmatched truth in efficiency; never clip energy tails.
    std::array<TH1F *,3> eff,pur,residual;
    for (int a=0;a<3;++a) {
        const std::string tag=std::array<std::string,3>{{"pandora","gnn","truth"}}[a];
        eff[a]=new TH1F((tag+"_efficiency_labelled").c_str(),"Labelled-input overlap;Efficiency;Truth particles",101,0,1.01);
        pur[a]=new TH1F((tag+"_purity_labelled").c_str(),"Labelled-input overlap;Purity;Matched truth particles",101,0,1.01);
        residual[a]=new TH1F((tag+"_energy_residual").c_str(),"Matched energy residual;(E_{reco}-E_{truth})/E_{truth};Truth particles",400,-2,2);
        residual[a]->SetCanExtend(TH1::kXaxis);
    }
    Long64_t algorithm=0,matched=0,match_event=0;
    Double_t efficiency=0,purity=0,reco_energy=0,truth_energy=0;
    matches.SetBranchAddress("event",&match_event); matches.SetBranchAddress("algorithm",&algorithm);
    matches.SetBranchAddress("matched",&matched); matches.SetBranchAddress("efficiency",&efficiency);
    matches.SetBranchAddress("purity",&purity); matches.SetBranchAddress("reco_energy",&reco_energy);
    matches.SetBranchAddress("truth_energy",&truth_energy);
    for (Long64_t i=0;i<matches.GetEntries();++i) {
        matches.GetEntry(i);
        if (!selected_events.count({matches.GetTreeNumber(),match_event})) continue;
        if (algorithm<0 || algorithm>2) throw std::runtime_error("Invalid algorithm ID");
        if (std::isfinite(efficiency)) eff[algorithm]->Fill(efficiency);
        if (matched && std::isfinite(purity)) pur[algorithm]->Fill(purity);
        if (matched && truth_energy>0 && std::isfinite(reco_energy)) residual[algorithm]->Fill((reco_energy-truth_energy)/truth_energy);
    }
    matches.ResetBranchAddresses();
    results.Write();
    TObjString(configuration.c_str()).Write("comparison_configuration");
    TObjString("sample 0: full Pandora reference; 1/2: identical Pandora/GNN events; 3/4/5: identical complete-truth events; 6: partial truth diagnostic only. angle_bin=-1: central with neutrino correction; 0..12: uncorrected angular bins.").Write("evaluation_definition");
    TCanvas canvas("common_energy","Central energy on identical Pandora/GNN events",1000,700);
    central[1]->SetLineColor(kGreen+2); central[2]->SetLineColor(kBlue+1);
    central[1]->SetMaximum(1.15*std::max(central[1]->GetMaximum(),central[2]->GetMaximum()));
    int last=std::max(central[1]->FindLastBinAbove(0),central[2]->FindLastBinAbove(0));
    if (last>0) central[1]->GetXaxis()->SetRangeUser(0,1.1*central[1]->GetXaxis()->GetBinUpEdge(last));
    central[1]->Draw("HIST"); central[2]->Draw("HIST SAME");
    TLegend legend(.6,.7,.88,.88); legend.AddEntry(central[1],"Pandora (common events)","l");
    legend.AddEntry(central[2],"GNN (common events)","l"); legend.Draw(); canvas.Write();
    TCanvas angleCanvas("common_resolution", "Common-event angular resolution",1000,700);
    if (graphs[1]->GetN() || graphs[2]->GetN()) {
        graphs[1]->SetLineColor(kGreen+2); graphs[1]->SetMarkerColor(kGreen+2); graphs[1]->SetMarkerStyle(20);
        graphs[2]->SetLineColor(kBlue+1); graphs[2]->SetMarkerColor(kBlue+1); graphs[2]->SetMarkerStyle(21);
        const int first=graphs[1]->GetN() ? 1 : 2;
        graphs[first]->SetTitle("Common events;quark energy-weighted |cos(#theta)|;#sqrt{2} RMS_{90}/Mean_{90} [%]");
        graphs[first]->Draw("AP"); graphs[3-first]->Draw("P SAME");
    }
    angleCanvas.Write();
    file.Write(); file.Close();
}
}
#endif
