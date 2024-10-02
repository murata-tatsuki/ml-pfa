using namespace std;

// efficiency, purityを表示して保存するマクロ

const string test_particle_types = {"ntau_10GeV_10", "uds"};

// conditions
const bool saving_canvas = false;
const string train_particle_type = "ntau_10GeV_10";      // ntau_10GeV_10    uds91   ntau_10to100GeV_10
const string test_particle_type = "uds91";      // ntau_10GeV_10    uds91   ntau_10to100GeV_10


const bool hyper_parameter = false;
const int dimension = 5;    // output dimensions  (one for beta, others are for coordinates)
// 8が先

const bool fine_tuning = false;
const int epoch = 25;       // 20   25
const int train_epoch = epoch*2-1;

const bool beta_scan = false; 
const bool tbeta_td_scan = true; 
const bool tbeta_td_scan_below01 = false; 
const bool tbeta_td_scan_below01_add = false; 


const bool cout_eff_pur = true;





const double Pandora_eff_tau[3] = {0.993, 0.940, 0.991};    // GNN と同じ方法でenergy sumを計算した結果 tau
const double Pandora_pur_tau[3] = {0.918, 0.946, 0.972};    // GNN と同じ方法でenergy sumを計算した結果 tau
const double Pandora_eff_uds[3] = {0.802, 0.904, 0.790};    // GNN と同じ方法でenergy sumを計算した結果 uds91
const double Pandora_pur_uds[3] = {0.750, 0.906, 0.777};    // GNN と同じ方法でenergy sumを計算した結果 uds91
double Pandora_eff[3] = {};
double Pandora_pur[3] = {};

// const double Pandora_eff[3] = {0.993, 0.940, 0.991};    // ILCSoft でenergy sumを計算した結果
// const double Pandora_pur[3] = {0.918, 0.946, 0.972};    // ILCSoft でenergy sumを計算した結果



void tbeta_td_result(){ 
    int rawfilenum = 1;
    int rawfilenum_below01 = 0;
    int nbeta = 10;
    int ndiameter = 9;
    if(beta_scan) rawfilenum = 100;
    if(tbeta_td_scan){
        // rawfilenum = 90;
        rawfilenum = 90;
        if(tbeta_td_scan_below01_add) rawfilenum_below01 = 9;
    }

    for(int i=0;i<3;i++){
        Pandora_eff[i] = test_particle_type == "ntau_10GeV_10" ? Pandora_eff_tau[i] : (test_particle_type == "uds91" ? Pandora_eff_uds[i] : 0);
        Pandora_pur[i] = test_particle_type == "ntau_10GeV_10" ? Pandora_pur_tau[i] : (test_particle_type == "uds91" ? Pandora_pur_uds[i] : 0);
    }

    if((hyper_parameter && fine_tuning && beta_scan && tbeta_td_scan) || (tbeta_td_scan_below01 && tbeta_td_scan_below01_add)){ // condition check
        cout << "something wrong with setting boolean " << endl;
        abort();
    }
    if( (tbeta_td_scan_below01 && !tbeta_td_scan) || (tbeta_td_scan_below01_add && !tbeta_td_scan) ){ // condition check
        cout << "something wrong with setting boolean " << endl;
        abort();
    }

    TFile *filein[rawfilenum];
    TTree *tree[rawfilenum];
    int entry_max[rawfilenum];
    TFile *filein_below01[rawfilenum];
    TTree *tree_below01[rawfilenum];
    int entry_max_below01[rawfilenum];
    int total_entry_max=0;
    string picDirectory;

    if(hyper_parameter){
        filein[0] = new TFile(Form("../output/hyper_parameter/dimensions/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_%dD_49_%s.root",dimension,test_particle_type.c_str()));
        picDirectory = Form("../pic/output_dimension/D%d",dimension);
    }
    if(fine_tuning){
        filein[0] = new TFile(Form("../output/fine_tuning/tc_ntau_10GeV_10_uds_timingcut_forcealpha_thetaphi_%d_%s.root",train_epoch,test_particle_type.c_str()));
        picDirectory = Form("../pic/fine_tuning");
    }
    if(beta_scan){
        for(int irawfile=0; irawfile<rawfilenum; irawfile++){
            filein[irawfile] = new TFile(Form("../output/hyper_parameter/beta_threshold/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10/beta%03d.root",irawfile));
        }
    }
    if(tbeta_td_scan){
        for(int irawfile=0; irawfile<rawfilenum; irawfile++){
            int itbeta = irawfile / ndiameter;
            int itd = irawfile % ndiameter + 1;
            if(tbeta_td_scan_below01) filein[irawfile] = new TFile(Form("../output/hyper_parameter/tbeta_td/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10/tbeta001td010/tbeta%03dtd%02d0.root",itbeta,itd));
            else {
                filein[irawfile] = new TFile(Form("../output/hyper_parameter/tbeta_td/tc_%s_timingcut_forcealpha_thetaphi_5D_49_%s/tbeta010td010/tbeta%02d0td%02d0.root",train_particle_type.c_str(),test_particle_type.c_str(),itbeta,itd));
                // filein[irawfile] = new TFile(Form("../output/hyper_parameter/tbeta_td/tc_%s_timingcut_forcealpha_thetaphi_5D_49_%s/tbeta%02d0td%02d0.root",train_particle_type.c_str(),test_particle_type.c_str(),itbeta,itd));
                // filein[irawfile] = new TFile(Form("../output/new_clustering/hyper_parameter/tbeta_td/tc_%s_timingcut_forcealpha_thetaphi_5D_49_%s/tbeta010td010/tbeta%03dtd%02d0.root",train_particle_type.c_str(),test_particle_type.c_str(),itbeta,itd));
            }
            picDirectory = Form("../pic/tbeta_td_scan");
        }
    }
    if(tbeta_td_scan_below01_add){
        for(int irawfile=0; irawfile<rawfilenum_below01; irawfile++){
            int itbeta = irawfile / ndiameter;
            int itd = irawfile % ndiameter + 1;
            filein_below01[irawfile] = new TFile(Form("../output/hyper_parameter/tbeta_td/tc_%s_timingcut_forcealpha_thetaphi_5D_49_%s/tbeta005td%02d0.root",train_particle_type.c_str(),test_particle_type.c_str(),itd));
        }
    }

    cout << picDirectory << endl;

    for(int i=0; i<rawfilenum; i++){
        tree[i] = (TTree*) filein[i]->Get("t");
        entry_max[i] = tree[i]->GetEntries();
    }
    for(int i=0; i<rawfilenum_below01; i++){
        tree_below01[i] = (TTree*) filein_below01[i]->Get("t");
        entry_max_below01[i] = tree_below01[i]->GetEntries();
    }


    int event, hitid, mcid, truthid, mcpdg, mccharge, mcstatus, ncluster, matched_ncluster, matched_cluster;
    double mcmass, mcpx, mcpy, mcpz, mcen, edep, edep_reco, edep_match;



    // TFile fileout("result_test_pandora.root","RECREATE");
    TFile fileout("result/result_test.root","RECREATE");


    // string particleNames[3] = {"electron", "pion", "gamma"};
    string particleNames[3] = {"electron", "pion", "photon"};
    vector<int> particledgValues = {11,-11, 211,-211, 22};
    // map<int,int> pdgParticle;
    // pdgParticle[11] = 0;
    // pdgParticle[-11] = 0;
    // pdgParticle[211] = 1;
    // pdgParticle[-211] = 1;
    // pdgParticle[22] = 2;
    
    const int nParticle = 3;
    const int nEnergy = 10;
    TH1F *purity[nParticle][rawfilenum];
    TH1F *purity_energy[nParticle][nEnergy][rawfilenum];
    TH1F *purity_energy_normalize[nParticle][nEnergy][rawfilenum];
    TH2F *purity2d[nParticle][rawfilenum];
    TH1F *efficiency[nParticle][rawfilenum];
    TH1F *efficiency_energy[nParticle][nEnergy][rawfilenum];
    TH1F *efficiency_energy_normalize[nParticle][nEnergy][rawfilenum];
    TH2F *efficiency2d[nParticle][rawfilenum];
    TGraph *eff_vs_pur[nParticle];
    TH2F *tbeta_vs_td_eff[nParticle];
    TH2F *tbeta_vs_td_pur[nParticle];
    TH1F *purity_below01[nParticle][rawfilenum_below01];
    TH1F *efficiency_below01[nParticle][rawfilenum_below01];
    for(int ip=0;ip<nParticle;ip++){
        eff_vs_pur[ip] = new TGraph();
        eff_vs_pur[ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
        eff_vs_pur[ip]->GetXaxis()->SetTitle("efficiency");
        eff_vs_pur[ip]->GetYaxis()->SetTitle("puriy");
        tbeta_vs_td_eff[ip] = new TH2F(Form("tbeta_vs_td_eff_%d",ip),Form("%s efficiency;beta threshold;diameter threshold",particleNames[ip].c_str()),10,0,1, 10,0,1);
        tbeta_vs_td_pur[ip] = new TH2F(Form("tbeta_vs_td_pur_%d",ip),Form("%s purity;beta threshold;diameter threshold",particleNames[ip].c_str()),10,0,1, 10,0,1);
        for(int irawfile=0; irawfile<rawfilenum; irawfile++){
            purity[ip][irawfile] = new TH1F(Form("purity_%d_%d",ip,irawfile), Form("%s purity (MC energy>1 GeV) tbeta=%d",particleNames[ip].c_str(),irawfile), 101,0,1.01);
            purity[ip][irawfile]->SetXTitle("purity (edep_match/edep_reco)");
            purity2d[ip][irawfile] = new TH2F(Form("purity2d_%d_%d",ip,irawfile), Form("%s purity tbeta=%d",particleNames[ip].c_str(),irawfile), 101,0,1.01, 24,0,12);
            purity2d[ip][irawfile]->SetXTitle("purity (edep_match/edep_reco)");
            purity2d[ip][irawfile]->SetYTitle("MC energy (edep)");
            efficiency[ip][irawfile] = new TH1F(Form("efficiency_%d_%d",ip,irawfile), Form("%s efficiency (MC energy>1 GeV) tbeta=%d",particleNames[ip].c_str(),irawfile), 101,0,1.01);
            efficiency[ip][irawfile]->SetXTitle("efficiency (edep_match/edep)");
            efficiency2d[ip][irawfile] = new TH2F(Form("efficiency2d_%d_%d",ip,irawfile), Form("%s efficiency tbeta=%d",particleNames[ip].c_str(),irawfile), 101,0,1.01, 24,0,12);
            efficiency2d[ip][irawfile]->SetXTitle("efficiency (edep_match/edep)");
            efficiency2d[ip][irawfile]->SetYTitle("MC energy (edep)");
            for(int ie=0;ie<nEnergy;ie++){
                string title = ie==0 ? Form("%s purity",particleNames[ip].c_str()) : Form("%s purity %d-%d GeV tbeta=%d",particleNames[ip].c_str(),ie,ie+1,irawfile);
                purity_energy[ip][ie][irawfile] = new TH1F(Form("purity_%d_%d_%d",ip,ie,irawfile), title.c_str(), 51,0,1.02);
                purity_energy[ip][ie][irawfile]->SetXTitle("purity (edep_match/edep_reco)");
                purity_energy_normalize[ip][ie][irawfile] = new TH1F(Form("purity_norm_%d_%d_%d",ip,ie,irawfile), title.c_str(), 51,0,1.02);
                purity_energy_normalize[ip][ie][irawfile]->SetXTitle("purity (edep_match/edep_reco)");
                title = ie==0 ? Form("%s efficiency",particleNames[ip].c_str()) : Form("%s efficiency %d-%d GeV tbeta=%d",particleNames[ip].c_str(),ie,ie+1,irawfile);
                efficiency_energy[ip][ie][irawfile] = new TH1F(Form("efficiency_%d_%d_%d",ip,ie,irawfile), title.c_str(), 51,0,1.02);
                efficiency_energy[ip][ie][irawfile]->SetXTitle("efficiency (edep_match/edep)");
                efficiency_energy_normalize[ip][ie][irawfile] = new TH1F(Form("efficiency_normalize_%d_%d_%d",ip,ie,irawfile), title.c_str(), 51,0,1.02);
                efficiency_energy_normalize[ip][ie][irawfile]->SetXTitle("efficiency (edep_match/edep)");
            }
        }
        for(int irawfile=0; irawfile<rawfilenum_below01; irawfile++){
            purity_below01[ip][irawfile] = new TH1F(Form("purity_below01_%d_%d",ip,irawfile), Form("%s purity (MC energy>1 GeV) tbeta=%d",particleNames[ip].c_str(),irawfile), 101,0,1.01);
            purity_below01[ip][irawfile]->SetXTitle("purity (edep_match/edep_reco)");
            efficiency_below01[ip][irawfile] = new TH1F(Form("efficiency_below01_%d_%d",ip,irawfile), Form("%s efficiency (MC energy>1 GeV) tbeta=%d",particleNames[ip].c_str(),irawfile), 101,0,1.01);
            efficiency_below01[ip][irawfile]->SetXTitle("efficiency (edep_match/edep)");
        }
    }
    TGraph *eff_vs_pur_[nbeta][nParticle];
    TGraph *eff_vs_pur_sametbeta[nbeta][nParticle];
    int line_color = 1;
    for(int ibeta=0;ibeta<nbeta;ibeta++){
        if(ibeta==4 || ibeta==8) line_color++;
        for(int ip=0;ip<nParticle;ip++){
            eff_vs_pur_[ibeta][ip] = new TGraph();
            eff_vs_pur_[ibeta][ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
            eff_vs_pur_[ibeta][ip]->GetXaxis()->SetTitle("efficiency");
            eff_vs_pur_[ibeta][ip]->GetYaxis()->SetTitle("puriy");
            eff_vs_pur_[ibeta][ip]->SetMarkerStyle(4);
            eff_vs_pur_[ibeta][ip]->SetMarkerSize(0.5);
            eff_vs_pur_[ibeta][ip]->SetMarkerColor(ibeta+line_color);

            eff_vs_pur_sametbeta[ibeta][ip] = new TGraph();
            eff_vs_pur_sametbeta[ibeta][ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
            eff_vs_pur_sametbeta[ibeta][ip]->GetXaxis()->SetTitle("efficiency");
            eff_vs_pur_sametbeta[ibeta][ip]->GetYaxis()->SetTitle("puriy");
            eff_vs_pur_sametbeta[ibeta][ip]->SetMarkerStyle(4);
            eff_vs_pur_sametbeta[ibeta][ip]->SetMarkerSize(0.5);
            eff_vs_pur_sametbeta[ibeta][ip]->SetMarkerColor(ibeta+line_color);
            eff_vs_pur_sametbeta[ibeta][ip]->SetLineColor(ibeta+line_color);
        }
    }
    TGraph *eff_vs_pur_below01[nParticle];
    if(tbeta_td_scan_below01_add){
        for(int ip=0;ip<nParticle;ip++){
            eff_vs_pur_below01[ip] = new TGraph();
            eff_vs_pur_below01[ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
            eff_vs_pur_below01[ip]->GetXaxis()->SetTitle("efficiency");
            eff_vs_pur_below01[ip]->GetYaxis()->SetTitle("puriy");
            eff_vs_pur_below01[ip]->SetMarkerStyle(4);
            eff_vs_pur_below01[ip]->SetMarkerSize(0.5);
            eff_vs_pur_below01[ip]->SetMarkerColor(kViolet);
            eff_vs_pur_below01[ip]->SetLineColor(kViolet);
        }
    }

    double efficiency_scan[nParticle][rawfilenum];
    double purity_scan[nParticle][rawfilenum];
    double efficiency_scan_below01[nParticle][rawfilenum_below01];
    double purity_scan_below01[nParticle][rawfilenum_below01];


    // data をとってきてる
    for(int irawfile=0; irawfile<rawfilenum; irawfile++){
        // if(rawfilenum>1) cout << irawfile << "/" << rawfilenum << endl;
        double den = tbeta_td_scan_below01 ? 100.0 : 10.0;
        if(cout_eff_pur) cout << "tbeta:" << irawfile/9/den << " td:" << (irawfile%9+1)/10.0 << endl;

        tree[irawfile]->SetBranchAddress("event", &event);
        tree[irawfile]->SetBranchAddress("hitid", &hitid);
        tree[irawfile]->SetBranchAddress("mcid", &mcid);
        tree[irawfile]->SetBranchAddress("mcpdg", &mcpdg);
        tree[irawfile]->SetBranchAddress("mcen", &mcen);
        tree[irawfile]->SetBranchAddress("edep", &edep);
        tree[irawfile]->SetBranchAddress("edep_reco", &edep_reco);
        tree[irawfile]->SetBranchAddress("edep_match", &edep_match);

        for(int ientry=0; ientry<entry_max[irawfile]; ientry++){
            tree[irawfile]->GetEntry(ientry);

            if(edep<=0 || edep_reco<=0 || edep_match<0) continue;
            if(edep>10){
                // cout << "edep>10  event:" << event << "  hitid:" << hitid << "  edep:" << edep << endl;
                // continue;
            }
            auto result = find(particledgValues.begin(), particledgValues.end(), mcpdg);
            if(result == particledgValues.end()) continue;
            int itr = distance(particledgValues.begin(), result) / 2;

            double pur = edep_match / edep_reco;
            double eff = edep_match / edep;
            // cout << itr << ", " << edep << ", " << edep_match << ", " << edep_reco << ", " << pur << ", " << eff << ", " << endl;
            if(pur<0.1){
                // cout << "pur<0.1  event:" << event << "  hitid:" << hitid << "  itr:" << itr << ", " << edep << ", " << edep_match << ", " << edep_reco << ", " << pur << ", " << eff << ",   " << (int)edep << endl;
                // continue;
            }

            if(edep>1) purity[itr][irawfile]->Fill(pur);
            purity2d[itr][irawfile]->Fill(pur,edep);
            if(edep<10){
                purity_energy[itr][(int)edep][irawfile]->Fill(pur);
                purity_energy_normalize[itr][(int)edep][irawfile]->Fill(pur);
                efficiency_energy[itr][(int)edep][irawfile]->Fill(eff);
                efficiency_energy_normalize[itr][(int)edep][irawfile]->Fill(eff);
            }
            if(edep>1) efficiency[itr][irawfile]->Fill(eff);
            efficiency2d[itr][irawfile]->Fill(eff,edep);
        }
        
        for(int ip=0;ip<nParticle;ip++){
            efficiency_scan[ip][irawfile] = efficiency[ip][irawfile]->GetMean();
            purity_scan[ip][irawfile] = purity[ip][irawfile]->GetMean();

            eff_vs_pur[ip]->SetPoint(irawfile, efficiency_scan[ip][irawfile], purity_scan[ip][irawfile]);
            if(irawfile%9==0) eff_vs_pur_[irawfile%9][ip]->SetPoint(irawfile, efficiency_scan[ip][irawfile], purity_scan[ip][irawfile]);
            eff_vs_pur_sametbeta[irawfile/9][ip]->SetPoint(irawfile%9, efficiency_scan[ip][irawfile], purity_scan[ip][irawfile]);
            if(cout_eff_pur) cout << "  " << particleNames[ip] << ", eff:" <<efficiency_scan[ip][irawfile] << " pur:" << purity_scan[ip][irawfile] << endl;
        }

        if(tbeta_td_scan){
            for(int ip=0;ip<nParticle;ip++){
                int itbeta = irawfile / 9 + 1;
                int itd = irawfile % 9 + 1;
                tbeta_vs_td_eff[ip]->SetBinContent(itbeta+1, itd+1, purity[ip][irawfile]->GetMean());
                tbeta_vs_td_pur[ip]->SetBinContent(itbeta+1, itd+1, efficiency[ip][irawfile]->GetMean());
            }
        }


        // for(int ip=0;ip<nParticle;ip++){
        //     purity[ip]->Reset();
        //     purity[ip]->ResetStats();
        //     efficiency[ip]->Reset();
        //     efficiency[ip]->ResetStats();
        // }

        // if(beta_scan || tbeta_td_scan) delete filein[irawfile];
        delete filein[irawfile];
    }
    for(int irawfile=0; irawfile<rawfilenum_below01; irawfile++){
        // if(rawfilenum>1) cout << irawfile << "/" << rawfilenum << endl;
        if(cout_eff_pur) cout << "tbeta:" << 0.05 << " td:" << (irawfile%9+1)/10.0 << endl;

        tree_below01[irawfile]->SetBranchAddress("event", &event);
        tree_below01[irawfile]->SetBranchAddress("hitid", &hitid);
        tree_below01[irawfile]->SetBranchAddress("mcid", &mcid);
        tree_below01[irawfile]->SetBranchAddress("mcpdg", &mcpdg);
        tree_below01[irawfile]->SetBranchAddress("mcen", &mcen);
        tree_below01[irawfile]->SetBranchAddress("edep", &edep);
        tree_below01[irawfile]->SetBranchAddress("edep_reco", &edep_reco);
        tree_below01[irawfile]->SetBranchAddress("edep_match", &edep_match);

        for(int ientry=0; ientry<entry_max_below01[irawfile]; ientry++){
            tree_below01[irawfile]->GetEntry(ientry);

            if(edep<=0 || edep_reco<=0 || edep_match<0) continue;
            auto result = find(particledgValues.begin(), particledgValues.end(), mcpdg);
            if(result == particledgValues.end()) continue;
            int itr = distance(particledgValues.begin(), result) / 2;

            double pur = edep_match / edep_reco;
            double eff = edep_match / edep;

            if(edep>1){
                purity_below01[itr][irawfile]->Fill(pur);
                efficiency_below01[itr][irawfile]->Fill(eff);
            }
        }
        
        for(int ip=0;ip<nParticle;ip++){
            efficiency_scan_below01[ip][irawfile] = purity_below01[ip][irawfile]->GetMean();
            purity_scan_below01[ip][irawfile] = efficiency_below01[ip][irawfile]->GetMean();

            eff_vs_pur[ip]->SetPoint(irawfile, efficiency_scan_below01[ip][irawfile], purity_scan_below01[ip][irawfile]);
            eff_vs_pur_below01[ip]->SetPoint(irawfile%9, efficiency_scan_below01[ip][irawfile], purity_scan_below01[ip][irawfile]);
            if(cout_eff_pur) cout << "  " << particleNames[ip] << ", eff:" <<efficiency_scan[ip][irawfile] << " pur:" << purity_scan[ip][irawfile] << endl;
        }
        delete filein_below01[irawfile];
    }

    // efficiency[0]->Draw();
    // purity_energy[0][0]->Draw();
    


    // gStyle->SetStatX(0.35);
    // gStyle->SetStatY(0.9);
    // // legends をもう少し大きくする

    TGraph *Pandora_eff_vs_pur[nParticle];
    for(int ip=0;ip<nParticle;ip++){
        Pandora_eff_vs_pur[ip] = new TGraph();
        Pandora_eff_vs_pur[ip]->SetTitle(Form("Pandora %s",particleNames[ip].c_str()));
        Pandora_eff_vs_pur[ip]->GetXaxis()->SetTitle("efficiency");
        Pandora_eff_vs_pur[ip]->GetYaxis()->SetTitle("puriy");
        Pandora_eff_vs_pur[ip]->SetMarkerStyle(3);
        // Pandora_eff_vs_pur[ip]->SetMarkerSize(1);
        Pandora_eff_vs_pur[ip]->SetMarkerColor(kRed);
        Pandora_eff_vs_pur[ip]->SetPoint(0, Pandora_eff[ip], Pandora_pur[ip]);
        cout << Pandora_eff[ip] << ", " << Pandora_pur[ip] << endl;
    }

    TCanvas *compare = new TCanvas("compare","compare",1400,500);
    compare->Divide(nParticle);
    TLegend *legends_tbeta = new TLegend( 0.101, 0.101, 0.451, 0.401);
    double denominator = tbeta_td_scan_below01 ? 100.0 : 10.0;
    for(int ibeta=0;ibeta<nbeta;ibeta++) legends_tbeta->AddEntry(eff_vs_pur_sametbeta[ibeta][0], Form("tbeta=%3.2f",(ibeta)/denominator) , "l");
    if(tbeta_td_scan_below01_add) legends_tbeta->AddEntry(eff_vs_pur_below01[0], Form("tbeta=0.05") , "l");
    if(test_particle_type == "ntau_10GeV_10") eff_vs_pur[1]->SetMinimum(0.935);
    if(test_particle_type == "uds91"){
        eff_vs_pur[2]->SetMinimum(0.6);
        eff_vs_pur[2]->GetXaxis()->SetLimits(0.65,1.01);
    }
    for(int ip=0;ip<nParticle;ip++){
        // TPaveStats *s = (TPaveStats*) gPad->GetPrimitive("stats"); s->SetTextSize(0.1); s->SetX1NDC(0.5); s->SetY1NDC(0.5);
        compare->cd(ip+1);
        // gPad->SetLogy();
        // if(test_particle_type == "uds91"){
        //     eff_vs_pur[ip]->SetMaximum(1.01);
        //     eff_vs_pur[ip]->GetXaxis()->SetLimits(0.73,1.01);
        // }
        eff_vs_pur[ip]->Draw("AP");
        eff_vs_pur_[0][ip]->Draw("P");
        Pandora_eff_vs_pur[ip]->Draw("P");

        for(int ibeta=0;ibeta<nbeta;ibeta++) {
            eff_vs_pur_sametbeta[ibeta][ip]->Draw("L");
        }
        if(tbeta_td_scan_below01_add) eff_vs_pur_below01[ip]->Draw("L");
        gStyle->SetLegendFont(60);
        legends_tbeta->Draw("same");
    }

    TCanvas *compare_tbeta_td = new TCanvas("compare_tbeta_td","compare_tbeta_td",1);
    compare_tbeta_td->Divide(nParticle,2);
    if(tbeta_td_scan){
        for(int ip=0;ip<nParticle;ip++){
            compare_tbeta_td->cd(ip+1);
            gStyle->SetOptStat(0);
            tbeta_vs_td_eff[ip]->SetMinimum(0.8);
            tbeta_vs_td_eff[ip]->Draw("colz");
            compare_tbeta_td->cd(ip+1+nParticle);
            gStyle->SetOptStat(0);
            tbeta_vs_td_pur[ip]->SetMinimum(0.8);
            tbeta_vs_td_pur[ip]->Draw("colz");
        }
    }




    if(saving_canvas){  // saving canvases
        string suffix = "";
        if(hyper_parameter) suffix = Form("_%s",test_particle_type.c_str());
        if(fine_tuning)     suffix = Form("_epoch%d_%s",epoch,test_particle_type.c_str());
        // if(tbeta_td_scan)   suffix = Form("_epoch%d_%s",epoch,test_particle_type.c_str());
        if(tbeta_td_scan_below01) suffix = Form("_tbeta_below01");
        
        if(tbeta_td_scan){
            compare->SaveAs(Form("%s/efficiency_vs_purity%s.pdf",picDirectory.c_str(),suffix.c_str()));
            compare_tbeta_td->SaveAs(Form("%s/tbeta_td_2d%s.pdf",picDirectory.c_str(),suffix.c_str()));
        }
    }

}
