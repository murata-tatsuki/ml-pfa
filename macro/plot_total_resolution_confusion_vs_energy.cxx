// Plot total resolution and confusion term vs jet energy,
// and truth-physics-category resolution vs truth energy (second canvas; you fill arrays).
// Usage:
//   root -l "plot_total_resolution_confusion_vs_energy.cxx"

#include "TCanvas.h"
#include "TGraphErrors.h"
#include "TGraph.h"
#include "TH1F.h"
#include "TLegend.h"

using namespace std;

void plot_total_resolution_confusion_vs_energy() {
    const int nEnergy = 5;
    const double jetEnergy[nEnergy] = {40, 91, 200, 350, 500};

    // Values are sigma of (pred-truth)/truth.

    // // mono-head nnqq
    // const double totalResolutionReco[nEnergy] = {
    //     2.754/40*sqrt(2),   // 40 GeV
    //     3.778/91*sqrt(2),  // 91 GeV
    //     8.622/200*sqrt(2),  // 200 GeV
    //     22.515/350*sqrt(2),  // 350 GeV
    //     38.336/500*sqrt(2)   // 500 GeV
    // };
    // const double totalResolutionPerfectReco[nEnergy] = {
    //     2.435/40*sqrt(2),     // 40 GeV
    //     3.373/91*sqrt(2),     // 91 GeV
    //     7.977/200*sqrt(2),    // 200 GeV
    //     21.432/350*sqrt(2),       // 350 GeV
    //     36.831/500*sqrt(2)     // 500 GeV
    // };

    // const double totalResolutionCond[nEnergy] = {
    //     2.749/40*sqrt(2),   // 40 GeV
    //     3.773/91*sqrt(2),   // 91 GeV
    //     8.627/200*sqrt(2),  // 200 GeV
    //     22.563/350*sqrt(2),   // 350 GeV
    //     38.079/500*sqrt(2)   // 500 GeV
    // };
    // const double totalResolutionPerfectCond[nEnergy] = {
    //     2.432/40*sqrt(2),     // 40 GeV
    //     3.363/91*sqrt(2),     // 91 GeV
    //     7.978/200*sqrt(2),    // 200 GeV
    //     21.281/350*sqrt(2),       // 350 GeV
    //     34.873/500*sqrt(2)     // 500 GeV
    // };

    // nnqq 2M multi-head track
    const double totalResolutionReco[nEnergy] = {
        2.61317/40*sqrt(2),   // 40 GeV
        3.20276/91*sqrt(2),  // 91 GeV
        5.84847/200*sqrt(2),  // 200 GeV
        19.7041/350*sqrt(2),  // 350 GeV
        44.7095/500*sqrt(2)   // 500 GeV
    };
    const double totalResolutionPerfectReco[nEnergy] = {
        2.06266/40*sqrt(2),     // 40 GeV
        2.76575/91*sqrt(2),     // 91 GeV
        5.38477/200*sqrt(2),    // 200 GeV
        19.0420/350*sqrt(2),       // 350 GeV
        44.1410/500*sqrt(2)     // 500 GeV
    };

    const double totalResolutionCond[nEnergy] = {
        2.61156/40*sqrt(2),   // 40 GeV
        3.20176/91*sqrt(2),   // 91 GeV
        5.84991/200*sqrt(2),  // 200 GeV
        19.7073/350*sqrt(2),   // 350 GeV
        44.7138/500*sqrt(2)   // 500 GeV
    };
    const double totalResolutionPerfectCond[nEnergy] = {
        2.05791/40*sqrt(2),     // 40 GeV
        2.66986/91*sqrt(2),     // 91 GeV
        5.37478/200*sqrt(2),    // 200 GeV
        18.7537/350*sqrt(2),       // 350 GeV
        44.1347/500*sqrt(2)     // 500 GeV
    };

    /*
    // nnqq 2M multi-head-alpha
    const double totalResolutionReco[nEnergy] = {
        2.60921/40*sqrt(2),   // 40 GeV
        3.33072/91*sqrt(2),  // 91 GeV
        6.34621/200*sqrt(2),  // 200 GeV
        18.8725/350*sqrt(2),  // 350 GeV
        37.8825/500*sqrt(2)   // 500 GeV
    };
    const double totalResolutionPerfectReco[nEnergy] = {
        2.06124/40*sqrt(2),     // 40 GeV
        2.81488/91*sqrt(2),     // 91 GeV
        5.64418/200*sqrt(2),    // 200 GeV
        17.9835/350*sqrt(2),       // 350 GeV
        38.3638/500*sqrt(2)     // 500 GeV
    };

    const double totalResolutionCond[nEnergy] = {
        2.60792/40*sqrt(2),   // 40 GeV
        3.32920/91*sqrt(2),   // 91 GeV
        6.34776/200*sqrt(2),  // 200 GeV
        18.8746/350*sqrt(2),   // 350 GeV
        37.8878/500*sqrt(2)   // 500 GeV
    };
    const double totalResolutionPerfectCond[nEnergy] = {
        2.05699/40*sqrt(2),     // 40 GeV
        2.80867/91*sqrt(2),     // 91 GeV
        5.63542/200*sqrt(2),    // 200 GeV
        17.9854/350*sqrt(2),       // 350 GeV
        38.4720/500*sqrt(2)     // 500 GeV
    };
    */

    /*
    // nnqq 2M mono-head track
    const double totalResolutionReco[nEnergy] = {
        4.39437/40*sqrt(2),   // 40 GeV
        7.42121/91*sqrt(2),  // 91 GeV
        11.9602/200*sqrt(2),  // 200 GeV
        44.1819/350*sqrt(2),  // 350 GeV
        70.2977/500*sqrt(2)   // 500 GeV
    };
    const double totalResolutionPerfectReco[nEnergy] = {
        4.39422/40*sqrt(2),     // 40 GeV
        7.47764/91*sqrt(2),     // 91 GeV
        11.9504/200*sqrt(2),    // 200 GeV
        44.3238/350*sqrt(2),       // 350 GeV
        70.3617/500*sqrt(2)     // 500 GeV
    };

    const double totalResolutionCond[nEnergy] = {
        4.39432/40*sqrt(2),   // 40 GeV
        7.42131/91*sqrt(2),   // 91 GeV
        11.9607/200*sqrt(2),  // 200 GeV
        44.1898/350*sqrt(2),   // 350 GeV
        70.303/500*sqrt(2)   // 500 GeV
    };
    const double totalResolutionPerfectCond[nEnergy] = {
        4.39417/40*sqrt(2),     // 40 GeV
        7.47762/91*sqrt(2),     // 91 GeV
        11.9510/200*sqrt(2),    // 200 GeV
        44.3314/350*sqrt(2),       // 350 GeV
        70.2841/500*sqrt(2)     // 500 GeV
    };
    */


    // PandoraPFA has only one reconstructed result (no perfect-clustering result).
    // Replace -1.0 with measured RMS90/Ejet values; negative values are not drawn.
    const double totalResolutionPandora[nEnergy] = {
        2.04433/40*sqrt(2),     // 40 GeV
        2.89583/91*sqrt(2),     // 91 GeV
        4.79710/200*sqrt(2),    // 200 GeV
        7.59841/350*sqrt(2),       // 350 GeV
        10.9865/500*sqrt(2)     // 500 GeV
    };

    double confusionReco[nEnergy];
    double confusionCond[nEnergy];
    for(int i=0; i<nEnergy; i++){
        confusionReco[i] = (totalResolutionReco[i]==-1 || totalResolutionPerfectReco[i]==-1 ) ?  -1 : pow(totalResolutionReco[i],2) - pow(totalResolutionPerfectReco[i],2)>=0 ? sqrt(pow(totalResolutionReco[i],2) - pow(totalResolutionPerfectReco[i],2)) : -1;
        confusionCond[i] = (totalResolutionCond[i]==-1 || totalResolutionPerfectCond[i]==-1 ) ?  -1 : pow(totalResolutionCond[i],2) - pow(totalResolutionPerfectCond[i],2)>=0 ? sqrt(pow(totalResolutionCond[i],2) - pow(totalResolutionPerfectCond[i],2)) : -1;
    }


    auto buildGraphPercent = [&](const double *values, int color, int markerStyle) {
        TGraph *g = new TGraph();
        int ip = 0;
        for (int i = 0; i < nEnergy; i++) {
            if (values[i] < 0) continue;  // skip placeholder points
            g->SetPoint(ip, jetEnergy[i]/2., values[i] * 100.0);
            ip++;
        }
        g->SetLineColor(color);
        g->SetMarkerColor(color);
        g->SetMarkerStyle(markerStyle);
        g->SetLineWidth(2);
        return g;
    };

    TGraph *gTotalReco = buildGraphPercent(totalResolutionReco, kBlack, 20);
    TGraph *gPerfectReco = buildGraphPercent(totalResolutionPerfectReco, kBlue + 1, 20);
    TGraph *gConfReco = buildGraphPercent(confusionReco, kRed + 1, 24);
    TGraph *gTotalCond = buildGraphPercent(totalResolutionCond, kBlack, 20);
    TGraph *gPerfectCond = buildGraphPercent(totalResolutionPerfectCond, kBlue + 1, 20);
    TGraph *gConfCond = buildGraphPercent(confusionCond, kRed + 1, 24);
    TGraph *gPandora = buildGraphPercent(totalResolutionPandora, kGreen + 2, 22);
    gPandora->SetLineStyle(2);
    const bool hasPandora = gPandora->GetN() > 0;

    TCanvas *c = new TCanvas("c_total_resolution_confusion", "total resolution and confusion term", 1400, 550);
    c->Divide(2, 1);

    c->cd(1);
    gTotalReco->SetTitle("Reco-track based;jet energy [GeV];RMS_{90}/E_{jet} [%]");
    // gTotalReco->SetTitle("Reco-track based;jet energy [GeV];#sigma/E_{jet} [%]");
    gTotalReco->SetMinimum(0.0);
    gTotalReco->SetMaximum(15.0);
    gTotalReco->Draw("APL");
    gPerfectReco->Draw("PL SAME");
    gConfReco->Draw("PL SAME");
    if (hasPandora) gPandora->Draw("PL SAME");
    TLegend *leg1 = new TLegend(0.47, 0.66, 0.88, 0.88);
    leg1->SetFillStyle(0);
    leg1->AddEntry(gTotalReco, "GNN total resolution", "lp");
    leg1->AddEntry(gPerfectReco, "GNN perfect clustering resolution", "lp");
    leg1->AddEntry(gConfReco, "GNN confusion term", "lp");
    if (hasPandora) leg1->AddEntry(gPandora, "PandoraPFA resolution", "lp");
    leg1->Draw();

    c->cd(2);
    gTotalCond->SetTitle("Cond-track based;jet energy [GeV];RMS_{90}/E_{jet} [%]");
    // gTotalCond->SetTitle("Cond-track based;jet energy [GeV];#sigma/E_{jet} [%]");
    gTotalCond->SetMinimum(0.0);
    gTotalCond->SetMaximum(15.0);
    gTotalCond->Draw("APL");
    gPerfectCond->Draw("PL SAME");
    gConfCond->Draw("PL SAME");
    if (hasPandora) gPandora->Draw("PL SAME");
    TLegend *leg2 = new TLegend(0.47, 0.66, 0.88, 0.88);
    leg2->SetFillStyle(0);
    leg2->AddEntry(gTotalCond, "GNN total resolution", "lp");
    leg2->AddEntry(gPerfectCond, "GNN perfect clustering resolution", "lp");
    leg2->AddEntry(gConfCond, "GNN confusion term", "lp");
    if (hasPandora) leg2->AddEntry(gPandora, "PandoraPFA resolution", "lp");
    leg2->Draw();



    // ----- canvas 2: charged / photon / neutral_hadron vs MC truth energy (values: you provide) -----
    const int kNTruthPhysCat = 3;
    const char* kTruthPhysName[kNTruthPhysCat] = {"charged", "photon", "neutral_hadron"};

    // x = truth energy (GeV), ex = x error (e.g. half bin width; use 0 if unused)

    // y < 0 skips that point for that curve

    // // mono-head
    // const double physRms[kNTruthPhysCat][nEnergy] = {
    //     {0.3638, 0.6864, 1.282, 2.672, 4.373}, // charged   TODO
    //     {0.6894, 1.046, 2.144, 4.672, 6.567}, // photon    TODO
    //     {1.03, 1.912, 3.093, 3.985, 4.791}  // neutral_hadron TODO
    // };
    // const double physRms90[kNTruthPhysCat][nEnergy] = {
    //     {0.1894, 0.2369, 0.4554, 1.565, 2.862},
    //     {0.5396, 0.801, 1.475, 3.613, 5.261},
    //     {0.8314, 1.528, 2.291, 1.372, 0.7038}};
    // const double physSigma[kNTruthPhysCat][nEnergy] = {
    //     {0.1319, 0.1821, 0.4951, 1.555, 3.05},
    //     {0.6602, 0.9819, 1.879, 4.267, 9.271},
    //     {1.619, 3.671, 6.574, 12.52, 17.53}};
    // const double physSigmaErr[kNTruthPhysCat][nEnergy] = {
    //     {0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0},
    //     {0, 0, 0, 0, 0}};

    // multi-head nnqq 2M
    const double physRms[kNTruthPhysCat][nEnergy] = {
        {0.49067, 0.831419, 1.70732, 3.7842, 7.30537}, // charged   TODO
        {0.588848, 1.03921, 3.2476, 7.37563, 11.7731}, // photon    TODO
        {0.963951, 1.70608, 3.40742, 6.70547, 9.74683}  // neutral_hadron TODO
    };
    const double physRms90[kNTruthPhysCat][nEnergy] = {
        {0.198462, 0.241082, 0.2843, 0.566876, 0.877563},
        {0.449903, 0.707181, 2.14864, 5.97484, 9.8303},
        {0.713717, 1.2884, 2.61444, 4.52544, 6.04141}};
    const double physSigma[kNTruthPhysCat][nEnergy] = {
        {0.0310694, 0.101744, 0.102698, 0.22529, 0.397916},
        {0.547697, 0.816189, 2.57254, 14.9693, 23.322},
        {1.67681, 1.849, 4.89949, 16.6873, 18.6309}};
    const double physSigmaErr[kNTruthPhysCat][nEnergy] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}};

    /*// multi-head-alpha nnqq 2M
    const double physRms[kNTruthPhysCat][nEnergy] = {
        {0.514704, 0.879288, 1.85765, 4.16506, 8.37568}, // charged   TODO
        {0.606172, 1.13038, 3.40947, 7.28137, 11.7077}, // photon    TODO
        {0.999574, 1.7696, 3.53684, 6.62971, 9.48969}  // neutral_hadron TODO
    };
    const double physRms90[kNTruthPhysCat][nEnergy] = {
        {0.178516, 0.212667, 0.255373, 0.527869, 1.35156},
        {0.46093, 0.745156, 2.3427, 5.85885, 9.84396},
        {0.759971, 1.36083, 2.75425, 4.54328, 5.73153}};
    const double physSigma[kNTruthPhysCat][nEnergy] = {
        {0.0831808, 0.0457639, 0.0952789, 0.366668, 0.771051},
        {0.549334, 0.840452, 2.31393, 15.1642, 25.2794},
        {1.65963, 2.28525, 4.54464, 14.1314, 21.4386}};
    const double physSigmaErr[kNTruthPhysCat][nEnergy] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}};
    */

    /*// mono-head nnqq 2M
    const double physRms[kNTruthPhysCat][nEnergy] = {
        {1.70518, 3.07464, 6.4711, 8.88649, 10.3475}, // charged   TODO
        {1.40655, 3.06351, 6.41035, 10.9079, 15.0855}, // photon    TODO
        {1.08196, 2.30275, 4.94225, 8.62768, 12.0267}  // neutral_hadron TODO
    };
    const double physRms90[kNTruthPhysCat][nEnergy] = {
        {0.432463, 0.415821, 0.569413, 3.67087, 5.62898},
        {0.778195, 1.95298, 4.53404, 7.52037, 7.70273},
        {0.443242, 0.963721, 3.20101, 5.78448, 8.00419}};
    const double physSigma[kNTruthPhysCat][nEnergy] = {
        {3.27317, 8.83466, 13.8139, 26.3256, 32.0797},
        {2.61316, 5.03829, 10.2298, 16.6418, 20.7807},
        {1.74959, 4.48634, 9.62229, 15.6483, 18.5087}};
    const double physSigmaErr[kNTruthPhysCat][nEnergy] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}};
    */

    auto buildGraphErrorsFromArrays = [&](const double* y, const double* ey) {
        TGraphErrors* g = new TGraphErrors();
        int ip = 0;
        for (int i = 0; i < nEnergy; i++) {
            if (y[i] < 0) continue;
            g->SetPoint(ip, jetEnergy[i]/2, y[i]/jetEnergy[i]*sqrt(2)*100);
            g->SetMarkerStyle(20);
            g->SetLineWidth(2);
            ip++;
        }
        return g;
    };

    TCanvas* cPhys = new TCanvas("c_truth_phys_resolution_vs_energy", "truth phys: resolution vs truth E", 1800, 600);
    cPhys->Divide(kNTruthPhysCat, 1);
    const int max_y[kNTruthPhysCat] = {2,4,15};

    for (int ic = 0; ic < kNTruthPhysCat; ic++) {
        TGraphErrors* gRms = buildGraphErrorsFromArrays(physRms[ic], nullptr);
        TGraphErrors* gR90 = buildGraphErrorsFromArrays(physRms90[ic], nullptr);
        TGraphErrors* gSig = buildGraphErrorsFromArrays(physSigma[ic], physSigmaErr[ic]);

        gRms->SetLineColor(kRed);
        gRms->SetMarkerColor(kRed);
        gRms->SetLineWidth(2);
        gR90->SetLineColor(kGreen + 1);
        gR90->SetMarkerColor(kGreen + 1);
        gR90->SetLineWidth(2);
        gSig->SetLineColor(kBlue);
        gSig->SetMarkerColor(kBlue);
        gSig->SetLineWidth(2);

        cPhys->cd(ic + 1);
        const int n90 = gR90->GetN();
        const int nRm = gRms->GetN();
        const int nSg = gSig->GetN();
        TString title = TString::Format("%s (truth category);jet energy (GeV);#sigma/E_{jet} [%%]",kTruthPhysName[ic]);
        if (n90 == 0 && nRm == 0 && nSg == 0) {
            TH1F* hf = new TH1F(TString::Format("h_phys_empty_%d", ic), title, 1, 0., 50.);
            hf->SetMinimum(0);
            hf->SetMaximum(0.5);
            hf->SetStats(0);
            hf->Draw();
            TLegend* legP = new TLegend(0.35, 0.75, 0.95, 0.92);
            legP->SetFillStyle(0);
            legP->AddEntry((TObject*)nullptr, "fill physRms / physRms90 / physSigma arrays", "");
            legP->Draw();
            continue;
        }
        TGraphErrors* gFirst = n90 > 0 ? gR90 : (nRm > 0 ? gRms : gSig);
        gFirst->SetTitle(title);
        gFirst->SetMinimum(0);
        gFirst->SetMaximum(max_y[ic]);
        gFirst->Draw("APL");
        if (gFirst != gRms && nRm > 0) gRms->Draw("PL SAME");
        if (gFirst != gR90 && n90 > 0) gR90->Draw("PL SAME");
        if (gFirst != gSig && nSg > 0) gSig->Draw("PL SAME");

        TLegend* legP = new TLegend(0.45, 0.65, 0.9, 0.9);
        legP->SetFillStyle(0);
        legP->AddEntry(gRms, "rms", "lp");
        legP->AddEntry(gR90, "rms_{90}", "lp");
        legP->AddEntry(gSig, "#sigma_{gaus}", "lp");
        legP->Draw();
    }

    c->SaveAs("figures/nnqq2M_fixed_uds/jet_energy_rezolution.png");
    cPhys->SaveAs("figures/nnqq2M_fixed_uds/jet_energy_rezolution_component.png");
}
