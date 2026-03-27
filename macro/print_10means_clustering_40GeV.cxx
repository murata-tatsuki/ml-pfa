// Print 10 means (5 efficiency + 5 purity) for 40 GeV dijet sample.
// Usage:
//   root -l "print_10means_clustering_40GeV.cxx"
//   root -l 'print_10means_clustering_40GeV.cxx("path/to/input.root")'

using namespace std;

void print_10means_clustering_40GeV(
    const string fileName = "../output/energy_regression_1to1/skimmed/tc_fixed_uds/5D/E_regression/tbeta_td_scan/qmin02_lr5e-4/40GeV/tbeta090td050.root",
    const double beta_threshold = 0.0
) {
    TFile *filein = TFile::Open(fileName.c_str(), "READ");
    if (!filein || filein->IsZombie()) {
        cout << "[ERROR] failed to open file: " << fileName << endl;
        return;
    }

    TTree *tree = (TTree*)filein->Get("t");
    if (!tree) {
        cout << "[ERROR] tree \"t\" is not found in: " << fileName << endl;
        filein->Close();
        return;
    }

    const int nParticle = 5;
    string particleNames[nParticle] = {"electron", "pion", "photon", "neutron", "K0"};

    TH1F *efficiency[nParticle];
    TH1F *purity[nParticle];
    for (int ip = 0; ip < nParticle; ip++) {
        efficiency[ip] = new TH1F(Form("efficiency_%d", ip), "", 101, 0.0, 1.01);
        purity[ip] = new TH1F(Form("purity_%d", ip), "", 101, 0.0, 1.01);
    }

    int mcpdg = 0;
    double edep = 0.0;
    double edep_reco = 0.0;
    double edep_match = 0.0;
    double cond_beta = 0.0;

    tree->SetBranchAddress("mcpdg", &mcpdg);
    tree->SetBranchAddress("edep", &edep);
    tree->SetBranchAddress("edep_reco", &edep_reco);
    tree->SetBranchAddress("edep_match", &edep_match);
    tree->SetBranchAddress("cond_beta", &cond_beta);

    const Long64_t nEntries = tree->GetEntries();
    for (Long64_t i = 0; i < nEntries; i++) {
        tree->GetEntry(i);

        if (edep <= 0.0 || edep_reco <= 0.0 || edep_match < 0.0) continue;
        if (cond_beta < beta_threshold) continue;

        int itr = -1;
        if (mcpdg == 11 || mcpdg == -11) itr = 0;
        else if (mcpdg == 211 || mcpdg == -211) itr = 1;
        else if (mcpdg == 22) itr = 2;
        else if (mcpdg == 2112) itr = 3;
        else if (mcpdg == 130) itr = 4;
        else continue;

        const double eff = edep_match / edep;
        const double pur = edep_match / edep_reco;

        if (edep > 1.0) {
            efficiency[itr]->Fill(eff);
            purity[itr]->Fill(pur);
        }
    }

    cout << "input file: " << fileName << endl;
    cout << "beta threshold: " << beta_threshold << endl;
    cout << "================ 10 means ================" << endl;
    for (int ip = 0; ip < nParticle; ip++) {
        cout << Form("%-8s efficiency mean = %.4f", particleNames[ip].c_str(), efficiency[ip]->GetMean()) << endl;
    }
    for (int ip = 0; ip < nParticle; ip++) {
        cout << Form("%-8s purity     mean = %.4f", particleNames[ip].c_str(), purity[ip]->GetMean()) << endl;
    }
    cout << "==========================================" << endl;

    filein->Close();
}
