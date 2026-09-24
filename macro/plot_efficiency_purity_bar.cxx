// Plot efficiency and purity bar charts from clustering results at 40, 91, 200, 350, 500 GeV.
// Data extracted from histograms (Mean values).
// Usage:
//   root -l "plot_efficiency_purity_bar.cxx"

using namespace std;

const string save_dir = "_multi-head-alpha";

void plot_efficiency_purity_bar() {
    const int nParticle = 5;
    const int nEnergy = 5;
    const char* particleNames[nParticle] = {"electron", "pion", "photon", "neutron", "kaon"};
    const char* energyLabels[nEnergy] = {"40 GeV", "91 GeV", "200 GeV", "350 GeV", "500 GeV"};
    const double energies[nEnergy] = {40, 91, 200, 350, 500};

    /*
    // nnqq mono-head
    const double efficiency[nParticle][nEnergy] = {
        {0.9517, 0.9459, 0.9223, 0.8852, 0.8498},  // electron
        {0.8622, 0.8742, 0.8529, 0.8247, 0.7982},  // pion
        {0.9514, 0.9366, 0.9095, 0.8677, 0.8291},  // photon
        {0.8290, 0.7950, 0.7486, 0.7242, 0.7068},  // neutron
        {0.8211, 0.7846, 0.7378, 0.7150, 0.6973}   // K0
    };
    const double purity[nParticle][nEnergy] = {
        {0.8486, 0.8077, 0.7486, 0.6766, 0.6066},  // electron
        {0.9798, 0.9495, 0.8948, 0.8225, 0.7515},  // pion
        {0.9782, 0.9373, 0.8557, 0.7464, 0.6413},  // photon
        {0.9456, 0.8709, 0.7523, 0.6352, 0.5496},  // neutron
        {0.9634, 0.9009, 0.7773, 0.6483, 0.5570}   // K0
    };
    */

    /*
    // nnqq 2M multi-head
    const double efficiency[nParticle][nEnergy] = {
        {0.9637, 0.9629, 0.9518, 0.9358, 0.9213},  // electron
        {0.8572, 0.8664, 0.8543, 0.8361, 0.8205},  // pion
        {0.9726, 0.9624, 0.9445, 0.9229, 0.9033},  // photon
        {0.8723, 0.8403, 0.8039, 0.7781, 0.7645},  // neutron
        {0.8623, 0.8332, 0.7953, 0.7683, 0.7543}   // K0
    };
    const double purity[nParticle][nEnergy] = {
        {0.8557, 0.8212, 0.7718, 0.7182, 0.6737},  // electron
        {0.9802, 0.9527, 0.9033, 0.8494, 0.8041},  // pion
        {0.9885, 0.9666, 0.9161, 0.8471, 0.7861},  // photon
        {0.9415, 0.8740, 0.7718, 0.6734, 0.6040},  // neutron
        {0.9594, 0.9062, 0.8064, 0.6987, 0.6213}   // K0
    };
    */

    /*
    // nnqq 2M multi-head-alpha
    const double efficiency[nParticle][nEnergy] = {
        {0.9622, 0.9600, 0.9465, 0.9297, 0.9151},  // electron
        {0.8506, 0.8587, 0.8457, 0.8265, 0.8105},  // pion
        {0.9619, 0.9500, 0.9306, 0.9090, 0.8906},  // photon
        {0.8598, 0.8256, 0.7887, 0.7640, 0.7487},  // neutron
        {0.8483, 0.8174, 0.7769, 0.7522, 0.7383}   // K0
    };
    const double purity[nParticle][nEnergy] = {
        {0.8543, 0.8188, 0.7681, 0.7157, 0.6731},  // electron
        {0.9786, 0.9502, 0.8981, 0.8406, 0.7931},  // pion
        {0.9890, 0.9657, 0.9108, 0.8493, 0.7798},  // photon
        {0.9414, 0.8747, 0.7683, 0.6717, 0.6060},  // neutron
        {0.9591, 0.9053, 0.8050, 0.6966, 0.6234}   // K0
    };
    */

    /*
    // nnqq 2M multi-head-alpha
    const double efficiency[nParticle][nEnergy] = {
        {0.9622, 0.9600, 0.9465, 0.9297, 0.9151},  // electron
        {0.8506, 0.8587, 0.8457, 0.8265, 0.8105},  // pion
        {0.9619, 0.9500, 0.9306, 0.9090, 0.8906},  // photon
        {0.8598, 0.8256, 0.7887, 0.7640, 0.7487},  // neutron
        {0.8483, 0.8174, 0.7769, 0.7522, 0.7383}   // K0
    };
    const double purity[nParticle][nEnergy] = {
        {0.8543, 0.8188, 0.7681, 0.7157, 0.6731},  // electron
        {0.9786, 0.9502, 0.8981, 0.8406, 0.7931},  // pion
        {0.9890, 0.9657, 0.9108, 0.8493, 0.7798},  // photon
        {0.9414, 0.8747, 0.7683, 0.6717, 0.6060},  // neutron
        {0.9591, 0.9053, 0.8050, 0.6966, 0.6234}   // K0
    };
    */


    // nnqq 2M mono-head
    const double efficiency[nParticle][nEnergy] = {
        {, , , , },  // electron
        {, , , , },  // pion
        {, , , , },  // photon
        {, , , , },  // neutron
        {, , , , }   // K0
    };
    const double purity[nParticle][nEnergy] = {
        {, , , , },  // electron
        {, , , , },  // pion
        {, , , , },  // photon
        {, , , , },  // neutron
        {, , , , }   // K0
    };



    // PandoraPFA results [particle][energy]. nnqq 2M
    const double pandoraEfficiency[nParticle][nEnergy] = {
        {0.9836, 0.9748, 0.9577, 0.9409, 0.9286},  // electron
        {0.8798, 0.8760, 0.8607, 0.8396, 0.8207},  // pion
        {0.9893, 0.9799, 0.9588, 0.9375, 0.9227},  // photon
        {0.8530, 0.8431, 0.8230, 0.7969, 0.7757},  // neutron
        {0.8475, 0.8405, 0.8164, 0.7862, 0.7631}   // K0
    };
    const double pandoraPurity[nParticle][nEnergy] = {
        {0.7987, 0.7548, 0.7190, 0.6897, 0.6662},  // electron
        {0.9619, 0.9350, 0.8948, 0.8531, 0.8195},  // pion
        {0.9747, 0.9330, 0.8947, 0.8598, 0.8285},  // photon
        {0.9279, 0.8466, 0.7463, 0.6745, 0.6241},  // neutron
        {0.9586, 0.8975, 0.8030, 0.7234, 0.6702}   // K0
    };

    // Colors for 5 energies
    const int colors[nEnergy] = {kRed + 1, kOrange + 1, kBlue + 1, kGreen + 2, kViolet - 6};
    const int gnnFillStyle = 1001;

    TCanvas *c = new TCanvas("c_eff_pur", "Efficiency and Purity", 1200, 800);
    c->Divide(1, 2);

    // Draw one GNN bar per energy and put the corresponding PandoraPFA result
    // at the center of the bar as a black marker.
    const double barWidth = 0.13;
    const double energyGap = 0.04;
    const double pandoraTickHalfWidth = 0.04;
    const double totalBarWidth = nEnergy * barWidth + (nEnergy - 1) * energyGap;
    const double firstBarOffset = (1.0 - totalBarWidth) / 2.0;

    int panelCount = 0;
    auto drawBarPanel = [&](const char* title, const char* ylabel,
                           const double data[nParticle][nEnergy],
                           const double pandoraData[nParticle][nEnergy],
                           double ymin, double ymax) {
        const int pc = panelCount++;
        TH1F *frame = new TH1F(Form("frame_%d", pc), title, nParticle, 0, nParticle);
        frame->SetStats(0);
        frame->SetMinimum(ymin * 100);
        frame->SetMaximum(ymax * 100);
        frame->GetYaxis()->SetTitle(Form("%s (%%)", ylabel));
        frame->GetYaxis()->SetTitleSize(0.06);
        frame->GetYaxis()->SetLabelSize(0.05);
        frame->GetXaxis()->SetLabelSize(0.06);
        for (int ip = 0; ip < nParticle; ip++) {
            frame->GetXaxis()->SetBinLabel(ip + 1, particleNames[ip]);
        }
        frame->Draw();

        TH1F *hGnn[nEnergy];
        TGraph *gPandoraHalo[nEnergy];
        TGraph *gPandora[nEnergy];
        for (int ie = 0; ie < nEnergy; ie++) {
            const double barOffset = firstBarOffset + ie * (barWidth + energyGap);

            hGnn[ie] = new TH1F(Form("h_gnn_%d_%d", pc, ie), "", nParticle, 0, nParticle);
            hGnn[ie]->SetStats(0);
            hGnn[ie]->SetFillColor(colors[ie]);
            hGnn[ie]->SetFillStyle(gnnFillStyle);
            hGnn[ie]->SetLineColor(colors[ie]);
            hGnn[ie]->SetBarWidth(barWidth);
            hGnn[ie]->SetBarOffset(barOffset);

            gPandoraHalo[ie] = new TGraph();
            gPandoraHalo[ie]->SetName(Form("g_pandora_halo_%d_%d", pc, ie));
            gPandoraHalo[ie]->SetMarkerStyle(20);
            gPandoraHalo[ie]->SetMarkerColor(kWhite);
            gPandoraHalo[ie]->SetMarkerSize(1.65);

            gPandora[ie] = new TGraph();
            gPandora[ie]->SetName(Form("g_pandora_%d_%d", pc, ie));
            gPandora[ie]->SetMarkerStyle(20);
            gPandora[ie]->SetMarkerColor(kBlack);
            gPandora[ie]->SetMarkerSize(1.05);

            int nPandoraPoints = 0;
            for (int ip = 0; ip < nParticle; ip++) {
                if (data[ip][ie] >= 0.0) {
                    hGnn[ie]->SetBinContent(ip + 1, data[ip][ie] * 100.0);
                }
                if (pandoraData[ip][ie] >= 0.0) {
                    const double x = ip + barOffset + 0.5 * barWidth;
                    const double y = pandoraData[ip][ie] * 100.0;
                    gPandoraHalo[ie]->SetPoint(nPandoraPoints, x, y);
                    gPandora[ie]->SetPoint(nPandoraPoints, x, y);
                    nPandoraPoints++;
                }
            }
            hGnn[ie]->Draw("BAR SAME");
            // The white marker underneath acts as a halo on dark-colored bars.
            gPandoraHalo[ie]->Draw("P SAME");
            gPandora[ie]->Draw("P SAME");
            // Draw the tick last so it remains black across the white halo.
            for (int ip = 0; ip < nParticle; ip++) {
                if (pandoraData[ip][ie] < 0.0) continue;
                const double x = ip + barOffset + 0.5 * barWidth;
                const double y = pandoraData[ip][ie] * 100.0;
                TLine *pandoraTick = new TLine(x - pandoraTickHalfWidth, y,
                                               x + pandoraTickHalfWidth, y);
                pandoraTick->SetLineColor(kBlack);
                pandoraTick->SetLineWidth(2);
                pandoraTick->Draw("SAME");
            }
        }
        return frame;
    };

    // Top: Efficiency
    c->cd(1);
    gPad->SetGridy();
    gPad->SetBottomMargin(0.15);
    gPad->SetRightMargin(0.20);
    drawBarPanel("Efficiency", "Efficiency", efficiency, pandoraEfficiency, 0.60, 1.00);

    // Bottom: Purity
    c->cd(2);
    gPad->SetGridy();
    gPad->SetBottomMargin(0.15);
    gPad->SetRightMargin(0.20);
    drawBarPanel("Purity", "Purity", purity, pandoraPurity, 0.50, 1.);

    // Combined legend in the upper-right margin: color represents energy and
    // the bar/marker symbols represent the reconstruction method.
    c->cd(1);
    TLegend *leg = new TLegend(0.81, 0.46, 0.98, 0.93);
    leg->SetFillStyle(1001);
    leg->SetFillColor(kWhite);
    leg->SetBorderSize(1);
    leg->SetTextSize(0.032);
    for (int ie = 0; ie < nEnergy; ie++) {
        TH1F *hlegEnergy = new TH1F(Form("hleg_energy_%d", ie), "", 1, 0, 1);
        hlegEnergy->SetFillColor(colors[ie]);
        hlegEnergy->SetFillStyle(gnnFillStyle);
        hlegEnergy->SetLineColor(colors[ie]);
        leg->AddEntry(hlegEnergy, energyLabels[ie], "f");
    }

    TH1F *hlegGnn = new TH1F("hleg_gnn", "", 1, 0, 1);
    hlegGnn->SetFillColor(kGray + 1);
    hlegGnn->SetFillStyle(gnnFillStyle);
    hlegGnn->SetLineColor(kGray + 1);
    leg->AddEntry(hlegGnn, "GNN", "f");

    TGraph *glegPandora = new TGraph();
    glegPandora->SetMarkerStyle(20);
    glegPandora->SetMarkerColor(kBlack);
    glegPandora->SetMarkerSize(1.05);
    glegPandora->SetLineColor(kBlack);
    glegPandora->SetLineWidth(2);
    leg->AddEntry(glegPandora, "PandoraPFA", "lp");
    leg->Draw();

    c->Update();
    // c->SaveAs("efficiency_purity_bar.pdf");
    c->SaveAs(Form("figures/nnqq2M_fixed_uds%s/efficiency_purity_bar.png",save_dir.c_str()));
    // cout << "Saved: efficiency_purity_bar.pdf, efficiency_purity_bar.png" << endl;
    cout << "Saved: efficiency_purity_bar.png" << endl;
}
