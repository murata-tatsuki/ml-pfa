using namespace std;

// efficiency, purityを表示して保存するマクロ

const string test_particle_types = {"ntau_10GeV_10", "uds"};

// conditions
const bool saving_canvas = false;
const string train_particle_type = "uds91";      // ntau_10GeV_10    uds91   ntau_10to100GeV_10
const string test_particle_type = train_particle_type;   // ntau_10GeV_10    uds91   ntau_10to100GeV_10
const string short_particle_type = train_particle_type == "ntau_10GeV_10" ? "ntau" : (train_particle_type == "uds91" ? "uds" : "ntau_10to100GeV_10");               // ntau    uds


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
const bool new_clustering = true;
const bool skimmed = false;


const bool cout_eff_pur = true;




// split をあわせたときの値
const double Pandora_eff_tau[3] = {0.993, 0.940, 0.991};    // GNN と同じ方法でenergy sumを計算した結果 tau
const double Pandora_pur_tau[3] = {0.918, 0.946, 0.972};    // GNN と同じ方法でenergy sumを計算した結果 tau
const double Pandora_eff_uds[3] = {0.802, 0.904, 0.790};    // GNN と同じ方法でenergy sumを計算した結果 uds91
const double Pandora_pur_uds[3] = {0.750, 0.906, 0.777};    // GNN と同じ方法でenergy sumを計算した結果 uds91
// 1to1のときの値
// const double Pandora_eff_tau[3] = {0.987, 0.885, 0.990};    // GNN と同じ方法でenergy sumを計算した結果 tau
// const double Pandora_pur_tau[3] = {0.946, 0.991, 0.984};    // GNN と同じ方法でenergy sumを計算した結果 tau
// const double Pandora_eff_uds[3] = {0.977, 0.878, 0.981};    // GNN と同じ方法でenergy sumを計算した結果 uds91
// const double Pandora_pur_uds[3] = {0.715, 0.838, 0.840};    // GNN と同じ方法でenergy sumを計算した結果 uds91
double Pandora_eff[3] = {};
double Pandora_pur[3] = {};

// const double Pandora_eff[3] = {0.993, 0.940, 0.991};    // ILCSoft でenergy sumを計算した結果
// const double Pandora_pur[3] = {0.918, 0.946, 0.972};    // ILCSoft でenergy sumを計算した結果


const int energyMax = test_particle_type == "ntau_10GeV_10" ? 12 : (test_particle_type == "uds91" ? 40 : 100 );
const int energyMaximum = test_particle_type == "ntau_10GeV_10" ? 10 : (test_particle_type == "uds91" ? 40 : 100 );



void tbeta_td_result_energy(){ 
    int rawfilenum = 1;
    int rawfilenum_below01 = 0;
    int nbeta = 9;
    int ndiameter = 9;
    int betas[9] = {1,2,3,4,5,6,7,8,9};
    int diameters[9] = {1,2,3,4,5,6,7,8,9};
    // int nbeta = 1;
    // int ndiameter = 9;
    // int betas[1] = {9};
    // int diameters[9] = {1,2,3,4,5,6,7,8,9};
    if(beta_scan) rawfilenum = 81;
    if(tbeta_td_scan) rawfilenum = 9;

    for(int i=0;i<3;i++){
        Pandora_eff[i] = test_particle_type == "ntau_10GeV_10" ? Pandora_eff_tau[i] : (test_particle_type == "uds91" ? Pandora_eff_uds[i] : 0);
        Pandora_pur[i] = test_particle_type == "ntau_10GeV_10" ? Pandora_pur_tau[i] : (test_particle_type == "uds91" ? Pandora_pur_uds[i] : 0);
    }

    if((hyper_parameter && fine_tuning && beta_scan && tbeta_td_scan) || (tbeta_td_scan_below01 && tbeta_td_scan_below01_add)){ // condition check
        cout << "something wrong with setting boolean " << endl;
        abort();
    }

    TFile *filein[nbeta][ndiameter];
    TTree *tree[nbeta][ndiameter];
    int entry_max[nbeta][ndiameter];
    int total_entry_max=0;
    string picDirectory;
    
    if(tbeta_td_scan){
        for(int itbeta=0; itbeta<nbeta; itbeta++){
            for(int itd=0; itd<ndiameter; itd++){
                // cout << Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/alpha_LE16_010_testDetected_499_tbeta_td_scan/tbeta%02d0td%02d0.root",betas[itbeta],diameters[itd]) << endl;
                // filein[itbeta][itd] = new TFile(Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/alpha_LE16_010_testDetected_499_tbeta_td_scan/tbeta%02d0td%02d0.root",betas[itbeta],diameters[itd]));
                // filein[itbeta][itd] = new TFile(Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/cluster_energy_regression/alpha_LE16_010_ERcluster_2025_02_07_150149/tbeta_td_scan/tbeta%02d0td%02d0.root",betas[itbeta],diameters[itd]));
                // filein[irawfile] = new TFile(Form("../output/hyper_parameter/tbeta_td/tc_%s_timingcut_forcealpha_thetaphi_5D_49_%s/tbeta%02d0td%02d0.root",train_particle_type.c_str(),test_particle_type.c_str(),itbeta,itd));
                // filein[irawfile] = new TFile(Form("../output/new_clustering/hyper_parameter/tbeta_td/tc_%s_timingcut_forcealpha_thetaphi_5D_49_%s/tbeta010td010/tbeta%03dtd%02d0.root",train_particle_type.c_str(),test_particle_type.c_str(),itbeta,itd));

                // filein[itbeta][itd] = new TFile(Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/LEweight0/alpha_0_2025_02_19_1537484_499_tbeta_td_scan/tbeta%02d0td%02d0.root",betas[itbeta],diameters[itd]));
                filein[itbeta][itd] = new TFile(Form("../output/energy_regression/new_clustering/energyTree/restartPeriod30/tc_ntau_10GeV_10_5D_59_ntau_10GeV_10_momentum/cluster_energy_regression/alpha_LE16_010_ERcluster_2025_02_07_150149/tbeta_td_scan/tbeta%02d0td%02d0.root",betas[itbeta],diameters[itd]));
                
                // filein[itbeta][itd] = new TFile(Form("../output/energy_regression_1to1/tc_uds91_5D/fine_tuning_2025_03_04_162442/tbeta_td_scan/tbeta%02d0td%02d0.root",betas[itbeta],diameters[itd]));
                // filein[itbeta][itd] = new TFile(Form("../output/energy_regression_1to1/tc_ntau_10GeV_10_5D/cluster_energy_regression/alpha_LE16_010_ERcluster_2025_02_07_150149/tbeta_td_scan/tbeta%02d0td%02d0.root",betas[itbeta],diameters[itd]));
                // filein[itbeta][itd] = new TFile(Form("../output/energy_regression_1to1/tc_ntau_10GeV_10_5D/LEweight0/alpha_0_2025_02_19_1537484/tbeta_td_scan/tbeta%02d0td%02d0.root",betas[itbeta],diameters[itd]));
                // filein[itbeta][itd] = new TFile(Form("../output/energy_regression_1to1/tc_uds91_5D/cluster_energy_regression/fine_tuning_2025_03_12_141129/tbeta_td_scan/tbeta%02d0td%02d0.root",betas[itbeta],diameters[itd]));
                // filein[itbeta][itd] = new TFile(Form("../output/energy_regression_1to1/tc_ntau_10GeV_10_5D/no_E_regression/tbeta_td_scan/tbeta%02d0td%02d0.root",betas[itbeta],diameters[itd]));
                // filein[itbeta][itd] = new TFile(Form("../output/energy_regression_1to1/tc_uds91_5D/no_E_regression/tbeta_td_scan/tbeta%02d0td%02d0.root",betas[itbeta],diameters[itd]));
            }
            picDirectory = Form("../pic/tbeta_td_scan");
        }
    }

    cout << picDirectory << endl;

    for(int itbeta=0; itbeta<nbeta; itbeta++){
        for(int itd=0; itd<ndiameter; itd++){
            tree[itbeta][itd] = (TTree*) filein[itbeta][itd]->Get("t");
            entry_max[itbeta][itd] = tree[itbeta][itd]->GetEntries();
        }
    }


    int event, hitid, mcid, truthid, mcpdg, mccharge, mcstatus, ncluster, matched_ncluster, matched_cluster, cond_track;
    double mcmass, mcpx, mcpy, mcpz, mcen, edep, edep_reco, edep_match, pred_edep, pred_edep_cluster, cond_beta, pred_alpha;



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
    TH1F *purity[nParticle][nbeta][ndiameter];
    TH1F *purity_energy[nParticle][nEnergy][nbeta][ndiameter];
    TH1F *purity_energy_normalize[nParticle][nEnergy][nbeta][ndiameter];
    TH2F *purity2d[nParticle][nbeta][ndiameter];
    TH1F *efficiency[nParticle][nbeta][ndiameter];
    TH1F *efficiency_energy[nParticle][nEnergy][nbeta][ndiameter];
    TH1F *efficiency_energy_normalize[nParticle][nEnergy][nbeta][ndiameter];
    TH2F *efficiency2d[nParticle][nbeta][ndiameter];
    TGraph *eff_vs_pur[nParticle];
    TH2F *tbeta_vs_td_eff[nParticle];
    TH2F *tbeta_vs_td_pur[nParticle];
    for(int ip=0;ip<nParticle;ip++){
        eff_vs_pur[ip] = new TGraph();
        eff_vs_pur[ip]->SetTitle(Form("%s",particleNames[ip].c_str()));
        eff_vs_pur[ip]->GetXaxis()->SetTitle("efficiency");
        eff_vs_pur[ip]->GetYaxis()->SetTitle("puriy");
        tbeta_vs_td_eff[ip] = new TH2F(Form("tbeta_vs_td_eff_%d",ip),Form("%s efficiency;beta threshold;diameter threshold",particleNames[ip].c_str()),10,0,1, 10,0,1);
        tbeta_vs_td_pur[ip] = new TH2F(Form("tbeta_vs_td_pur_%d",ip),Form("%s purity;beta threshold;diameter threshold",particleNames[ip].c_str()),10,0,1, 10,0,1);
        for(int itbeta=0; itbeta<nbeta; itbeta++){
            for(int itd=0; itd<ndiameter; itd++){
                purity[ip][itbeta][itd] = new TH1F(Form("purity_%d_%d_%d",ip,itbeta,itd), Form("%s purity (MC energy>1 GeV) tbeta=%d, td=%d",particleNames[ip].c_str(),itbeta,itd), 101,0,1.01);
                purity[ip][itbeta][itd]->SetXTitle("purity (edep_match/edep_reco)");
                purity2d[ip][itbeta][itd] = new TH2F(Form("purity2d_%d_%d_%d",ip,itbeta,itd), Form("%s purity tbeta=%d, td=%d",particleNames[ip].c_str(),itbeta,itd), 101,0,1.01, 24,0,12);
                purity2d[ip][itbeta][itd]->SetXTitle("purity (edep_match/edep_reco)");
                purity2d[ip][itbeta][itd]->SetYTitle("MC energy (edep)");
                efficiency[ip][itbeta][itd] = new TH1F(Form("efficiency_%d_%d_%d",ip,itbeta,itd), Form("%s efficiency (MC energy>1 GeV) tbeta=%d, td=%d",particleNames[ip].c_str(),itbeta,itd), 101,0,1.01);
                efficiency[ip][itbeta][itd]->SetXTitle("efficiency (edep_match/edep)");
                efficiency2d[ip][itbeta][itd] = new TH2F(Form("efficiency2d_%d_%d_%d",ip,itbeta,itd), Form("%s efficiency tbeta=%d, td=%d",particleNames[ip].c_str(),itbeta,itd), 101,0,1.01, 24,0,12);
                efficiency2d[ip][itbeta][itd]->SetXTitle("efficiency (edep_match/edep)");
                efficiency2d[ip][itbeta][itd]->SetYTitle("MC energy (edep)");
                for(int ie=0;ie<nEnergy;ie++){
                    string title = ie==0 ? Form("%s purity",particleNames[ip].c_str()) : Form("%s purity %d-%d GeV tbeta=%d, td=%d",particleNames[ip].c_str(),ie,ie+1,itbeta,itd);
                    purity_energy[ip][ie][itbeta][itd] = new TH1F(Form("purity_%d_%d_%d_%d",ip,ie,itbeta,itd), title.c_str(), 51,0,1.02);
                    purity_energy[ip][ie][itbeta][itd]->SetXTitle("purity (edep_match/edep_reco)");
                    purity_energy_normalize[ip][ie][itbeta][itd] = new TH1F(Form("purity_norm_%d_%d_%d_%d",ip,ie,itbeta,itd), title.c_str(), 51,0,1.02);
                    purity_energy_normalize[ip][ie][itbeta][itd]->SetXTitle("purity (edep_match/edep_reco)");
                    title = ie==0 ? Form("%s efficiency",particleNames[ip].c_str()) : Form("%s efficiency %d-%d GeV tbeta=%d, td=%d",particleNames[ip].c_str(),ie,ie+1,itbeta,itd);
                    efficiency_energy[ip][ie][itbeta][itd] = new TH1F(Form("efficiency_%d_%d_%d_%d",ip,ie,itbeta,itd), title.c_str(), 51,0,1.02);
                    efficiency_energy[ip][ie][itbeta][itd]->SetXTitle("efficiency (edep_match/edep)");
                    efficiency_energy_normalize[ip][ie][itbeta][itd] = new TH1F(Form("efficiency_normalize_%d_%d_%d_%d",ip,ie,itbeta,itd), title.c_str(), 51,0,1.02);
                    efficiency_energy_normalize[ip][ie][itbeta][itd]->SetXTitle("efficiency (edep_match/edep)");
                }
            }
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

    // energy regresion related figures
    double Eres_range = 1.5;
    int Eres_nbin = 1200;
    double E_res_binWidth = Eres_range*2/Eres_nbin;
    double Eres_fit_range = 0.5;
    int Eres_fitbin_lower = Eres_range>Eres_fit_range ? (Eres_range-Eres_fit_range)/E_res_binWidth : 0;
    int Eres_fitbin_upper = Eres_range>Eres_fit_range ? Eres_nbin - (Eres_range-Eres_fit_range)/E_res_binWidth : -1;
    int rebin_factor = 0.025 / E_res_binWidth;

    const double energy_interval = energyMaximum / nEnergy;
    TH1F *energy_resolution_per_energy[nParticle][nEnergy][nbeta][ndiameter];
    for(int ip=0;ip<nParticle;ip++){
        for(int ie=0;ie<nEnergy;ie++){
            for(int itbeta=0; itbeta<nbeta; itbeta++){
                for(int itd=0; itd<ndiameter; itd++){
                    string title = ie==0 ? Form("%s ;(predicted - truth) / truth",particleNames[ip].c_str()) : Form("%s (%d-%d GeV);(predicted - truth) / truth",particleNames[ip].c_str(),(int)(ie*energy_interval),(int)((ie+1)*energy_interval));
                    energy_resolution_per_energy[ip][ie][itbeta][itd] = new TH1F(Form("energy_resolution_per_energy_%d_%d_%d_%d",ip,ie,itbeta,itd), title.c_str(), Eres_nbin,-Eres_range,Eres_range);
                }
            }
        }
    }

    double efficiency_scan[nParticle][nbeta][ndiameter];
    double purity_scan[nParticle][nbeta][ndiameter];


    // data をとってきてる
    int irawfile = 0;
    for(int itbeta=0; itbeta<nbeta; itbeta++){
        for(int itd=0; itd<ndiameter; itd++){
            // if(rawfilenum>1) cout << irawfile << "/" << rawfilenum << endl;
            double den = 10.0;
            if(cout_eff_pur) cout << "tbeta:" << betas[itbeta]/10.0 << " td:" << diameters[itd]/10.0 << endl;

            tree[itbeta][itd]->SetBranchAddress("event", &event);
            tree[itbeta][itd]->SetBranchAddress("hitid", &hitid);
            tree[itbeta][itd]->SetBranchAddress("mcid", &mcid);
            tree[itbeta][itd]->SetBranchAddress("mcpdg", &mcpdg);
            tree[itbeta][itd]->SetBranchAddress("mcen", &mcen);
            tree[itbeta][itd]->SetBranchAddress("edep", &edep);
            tree[itbeta][itd]->SetBranchAddress("edep_reco", &edep_reco);
            tree[itbeta][itd]->SetBranchAddress("edep_match", &edep_match);
            tree[itbeta][itd]->SetBranchAddress("pred_edep", &pred_edep);
            tree[itbeta][itd]->SetBranchAddress("pred_edep_cluster", &pred_edep_cluster);
            tree[itbeta][itd]->SetBranchAddress("cond_beta", &cond_beta);
            tree[itbeta][itd]->SetBranchAddress("cond_track", &cond_track);

            for(int ientry=0; ientry<entry_max[itbeta][itd]; ientry++){
                tree[itbeta][itd]->GetEntry(ientry);

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

                if(edep>1) purity[itr][itbeta][itd]->Fill(pur);
                purity2d[itr][itbeta][itd]->Fill(pur,edep);
                if(edep<10){
                    purity_energy[itr][(int)edep][itbeta][itd]->Fill(pur);
                    purity_energy_normalize[itr][(int)edep][itbeta][itd]->Fill(pur);
                    efficiency_energy[itr][(int)edep][itbeta][itd]->Fill(eff);
                    efficiency_energy_normalize[itr][(int)edep][itbeta][itd]->Fill(eff);
                }
                if(edep>1) efficiency[itr][itbeta][itd]->Fill(eff);
                efficiency2d[itr][itbeta][itd]->Fill(eff,edep);

                int itr_energy = mcen / energy_interval;
                // cout << itr_energy << ", " << mcen << ", " << energy_interval << endl;
                if(itr_energy<nEnergy && pred_edep>0.1){
                    // energy_diff_per_energy[itr][itr_energy]->Fill(pred_edep/scaling - mcen);
                    energy_resolution_per_energy[itr][itr_energy][itbeta][itd]->Fill( (pred_edep - mcen) / mcen );
                }
            }

            for(int ip=0;ip<nParticle;ip++){
                efficiency_scan[ip][itbeta][itd] = efficiency[ip][itbeta][itd]->GetMean();
                purity_scan[ip][itbeta][itd] = purity[ip][itbeta][itd]->GetMean();

                eff_vs_pur[ip]->SetPoint(irawfile, efficiency_scan[ip][itbeta][itd], purity_scan[ip][itbeta][itd]);
                if(itd==0){
                    eff_vs_pur_[itbeta][ip]->SetPoint(itd, efficiency_scan[ip][itbeta][itd], purity_scan[ip][itbeta][itd]);
                }
                eff_vs_pur_sametbeta[itbeta][ip]->SetPoint(itd, efficiency_scan[ip][itbeta][itd], purity_scan[ip][itbeta][itd]);
                if(cout_eff_pur) cout << "  " << particleNames[ip] << ", eff:" <<efficiency_scan[ip][itbeta][itd] << " pur:" << purity_scan[ip][itbeta][itd] << endl;
            }

            if(tbeta_td_scan){
                for(int ip=0;ip<nParticle;ip++){
                    tbeta_vs_td_eff[ip]->SetBinContent(betas[itbeta]+1, diameters[itd]+1, purity[ip][itbeta][itd]->GetMean());
                    tbeta_vs_td_pur[ip]->SetBinContent(betas[itbeta]+1, diameters[itd]+1, efficiency[ip][itbeta][itd]->GetMean());
                }
            }



            delete filein[itbeta][itd];
            irawfile++;
        }
    }

    // efficiency[0]->Draw();
    // purity_energy[0][0]->Draw();
    


    // gStyle->SetStatX(0.35);
    // gStyle->SetStatY(0.9);

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
    for(int ibeta=0;ibeta<nbeta;ibeta++) legends_tbeta->AddEntry(eff_vs_pur_sametbeta[ibeta][0], Form("tbeta=%3.2f",(betas[ibeta])/10.0) , "l");
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
        // eff_vs_pur_[0][ip]->Draw("P");
        Pandora_eff_vs_pur[ip]->Draw("P");

        for(int ibeta=0;ibeta<nbeta;ibeta++) {
            eff_vs_pur_[ibeta][ip]->Draw("P");
            eff_vs_pur_sametbeta[ibeta][ip]->Draw("L");
        }
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




    TCanvas *canvas_energy_resolution_scan = new TCanvas("canvas_energy_resolution_scan","canvas_energy_resolution_scan",1400,500);
    canvas_energy_resolution_scan->Divide(nParticle,1);
    // TCanvas *canvas_energy_resolution_scan = new TCanvas("canvas_energy_resolution_scan","canvas_energy_resolution_scan",1);
    canvas_energy_resolution_scan->cd();
    double resolution_sigma[nParticle][nEnergy][nbeta][ndiameter];
    double resolution_sigma_error[nParticle][nEnergy][nbeta][ndiameter];
    TF1 *gaus = new TF1("gaus", "gaus", -3,3);
    TGraphErrors *energy_resolution_sigma[nParticle][nbeta][ndiameter];
    TLegend *legend_res = new TLegend( 0.5, 0.6, 0.9, 0.9);
    // legend_res->AddEntry(energy_resolution_sigma[0], Form("gaussian sigma") , "l");
    for(int ip=0;ip<nParticle;ip++){
        cout << particleNames[ip] << endl;
        line_color = 1;
        for(int itbeta=0; itbeta<nbeta; itbeta++){
            if(itbeta==4 || itbeta==8) line_color++;
            for(int itd=0; itd<ndiameter; itd++){
                energy_resolution_sigma[ip][itbeta][itd] = new TGraphErrors();
                energy_resolution_sigma[ip][itbeta][itd]->SetTitle(Form("%s beta:%.1f, diameter:%.1f",particleNames[ip].c_str(),itbeta/10.0,itd/10.0));
                energy_resolution_sigma[ip][itbeta][itd]->SetTitle(Form("%s beta:%.1f, diameter:%.1f",particleNames[ip].c_str(),itbeta/10.0,itd/10.0));
                energy_resolution_sigma[ip][itbeta][itd]->SetLineColor(itd+line_color);
                energy_resolution_sigma[ip][itbeta][itd]->SetMarkerColor(itd+line_color);
                energy_resolution_sigma[ip][itbeta][itd]->SetMaximum(0.5);
                energy_resolution_sigma[ip][itbeta][itd]->SetMinimum(0);
                if(ip==0) { 
                    legend_res->AddEntry(energy_resolution_sigma[0][itbeta][itd], Form("beta:%.1f, diameter:%.1f",itbeta/10.0,itd/10.0) , "l");
                }
                canvas_energy_resolution_scan->cd(ip+1);
                for(int ie=0;ie<nEnergy;ie++){
                    energy_resolution_per_energy[ip][ie][itbeta][itd]->SetAxisRange(-Eres_fit_range,Eres_fit_range);
                    if(energy_resolution_per_energy[ip][ie][itbeta][itd]->Integral(Eres_fitbin_lower,Eres_fitbin_upper)>30){
                        energy_resolution_per_energy[ip][ie][itbeta][itd]->Fit("gaus","NQ","",-Eres_fit_range,Eres_fit_range);
                        resolution_sigma[ip][ie][itbeta][itd] = gaus->GetParameter(2);
                        resolution_sigma_error[ip][ie][itbeta][itd] = gaus->GetParError(2);
                    } else {
                        resolution_sigma[ip][ie][itbeta][itd] = -1;
                        resolution_sigma_error[ip][ie][itbeta][itd] = 0;
                    }

                    energy_resolution_sigma[ip][itbeta][itd]->SetPoint(ie, (ie+itbeta/10.+itd/100.)*energy_interval, resolution_sigma[ip][ie][itbeta][itd]);
                    energy_resolution_sigma[ip][itbeta][itd]->SetPointError(ie, 0, resolution_sigma_error[ip][ie][itbeta][itd]);

                    // if( (ip<2 && ie>6) || (ip==2 && (ie>1 && ie<5)) ){
                    //     string energy_range = Form("%d-%d GeV",(int)(ie*energy_interval),(int)((ie+1)*energy_interval));
                    //     cout << "  " << energy_range << " : " << resolution_sigma[ip][ie] << endl;
                    // }
                }
                if(itbeta==0&&itd==0) energy_resolution_sigma[ip][itbeta][itd]->Draw("AP");
                energy_resolution_sigma[ip][itbeta][itd]->Draw("P");
            }
        }
        legend_res->Draw("same");
    }
    




    if(saving_canvas){  // saving canvases
        string suffix = "";
        if(hyper_parameter) suffix = Form("_%s",test_particle_type.c_str());
        if(fine_tuning)     suffix = Form("_epoch%d_%s",epoch,test_particle_type.c_str());
        // if(tbeta_td_scan)   suffix = Form("_epoch%d_%s",epoch,test_particle_type.c_str());
        
        if(tbeta_td_scan){
            compare->SaveAs(Form("%s/efficiency_vs_purity%s.pdf",picDirectory.c_str(),suffix.c_str()));
            compare_tbeta_td->SaveAs(Form("%s/tbeta_td_2d%s.pdf",picDirectory.c_str(),suffix.c_str()));
        }
    }

}
