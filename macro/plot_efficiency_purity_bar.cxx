// Plot efficiency and purity bar charts from clustering results at 40, 91, 200, 350, 500 GeV.
// Data extracted from histograms (Mean values).
// Usage:
//   root -l "plot_efficiency_purity_bar.cxx"

using namespace std;

void plot_efficiency_purity_bar() {
    const int nParticle = 5;
    const int nEnergy = 5;
    const char* particleNames[nParticle] = {"electron", "pion", "photon", "neutron", "kaon"};
    const char* energyLabels[nEnergy] = {"40 GeV", "91 GeV", "200 GeV", "350 GeV", "500 GeV"};
    const double energies[nEnergy] = {40, 91, 200, 350, 500};

    // Efficiency [particle][energy] - 40, 91, 200, 350, 500 GeV (350/500は後で追加)
    const double efficiency[nParticle][nEnergy] = {
        {0.9517, 0.9459, 0.9223, 0.8852, 0.8498},  // electron
        {0.8622, 0.8742, 0.8529, 0.8247, 0.7982},  // pion
        {0.9514, 0.9366, 0.9095, 0.8677, 0.8291},  // photon
        {0.8290, 0.7950, 0.7486, 0.7242, 0.7068},  // neutron
        {0.8211, 0.7846, 0.7378, 0.7150, 0.6973}   // K0
    };

    // Purity [particle][energy]
    const double purity[nParticle][nEnergy] = {
        {0.8486, 0.8077, 0.7486, 0.6766, 0.6066},  // electron
        {0.9798, 0.9495, 0.8948, 0.8225, 0.7515},  // pion
        {0.9782, 0.9373, 0.8557, 0.7464, 0.6413},  // photon
        {0.9456, 0.8709, 0.7523, 0.6352, 0.5496},  // neutron
        {0.9634, 0.9009, 0.7773, 0.6483, 0.5570}   // K0
    };

    // 350/500 GeV のデータ追加時は上記の 0.0 を実測値に置き換える

    // Colors for 5 energies
    const int colors[nEnergy] = {kRed + 1, kOrange + 1, kBlue + 1, kGreen + 2, kViolet - 6};
    const int fillStyles[nEnergy] = {1001, 1001, 1001, 1001, 1001};

    TCanvas *c = new TCanvas("c_eff_pur", "Efficiency and Purity", 900, 700);
    c->Divide(1, 2);

    const double barWidth = 0.15;  // 5 energies 用に調整
    const double barGap = 0.02;
    const double barShiftRight = 0.45;  // バー全体を右にずらす
    const double totalBarWidth = nEnergy * barWidth + (nEnergy - 1) * barGap;
    const double barOffset = totalBarWidth / 2.0 - barWidth / 2.0;

    int panelCount = 0;
    auto drawBarPanel = [&](const char* title, const char* ylabel,
                           const double data[nParticle][nEnergy], double ymin, double ymax) {
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

        TH1F *h[nEnergy];
        for (int ie = 0; ie < nEnergy; ie++) {
            h[ie] = new TH1F(Form("h_%d_%d", pc, ie), "", nParticle, 0, nParticle);
            h[ie]->SetStats(0);
            h[ie]->SetFillColor(colors[ie]);
            h[ie]->SetFillStyle(fillStyles[ie]);
            h[ie]->SetBarWidth(barWidth);
            h[ie]->SetBarOffset(-barOffset + ie * (barWidth + barGap) + barShiftRight);
            for (int ip = 0; ip < nParticle; ip++) {
                h[ie]->SetBinContent(ip + 1, data[ip][ie] * 100.0);
            }
            h[ie]->Draw("BAR SAME");
        }
        return frame;
    };

    // Top: Efficiency
    c->cd(1);
    gPad->SetGridy();
    gPad->SetBottomMargin(0.15);
    drawBarPanel("Efficiency", "Efficiency", efficiency, 0.60, 1.);

    // Bottom: Purity
    c->cd(2);
    gPad->SetGridy();
    gPad->SetBottomMargin(0.15);
    drawBarPanel("Purity", "Purity", purity, 0.50, 1.);

    // Legend (draw on top panel)
    c->cd(1);
    TLegend *leg = new TLegend(0.60, 0.6, 0.9, 0.9);
    leg->SetFillStyle(0);
    for (int ie = 0; ie < nEnergy; ie++) {
        TH1F *hleg = new TH1F(Form("hleg_%d", ie), "", 1, 0, 1);
        hleg->SetFillColor(colors[ie]);
        hleg->SetFillStyle(fillStyles[ie]);
        leg->AddEntry(hleg, energyLabels[ie], "f");
    }
    leg->Draw();

    c->Update();
    // c->SaveAs("efficiency_purity_bar.pdf");
    c->SaveAs("figures/fixed_uds/efficiency_purity_bar.png");
    // cout << "Saved: efficiency_purity_bar.pdf, efficiency_purity_bar.png" << endl;
    cout << "Saved: efficiency_purity_bar.png" << endl;
}
