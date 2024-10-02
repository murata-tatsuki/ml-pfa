using namespace std;

// efficiency, purityを表示して保存するマクロ

const string test_particle_types = {"ntau_10GeV_10", "uds"};

// conditions
const bool saving_canvas = false;
const string train_particle_type = "uds91";      // ntau_10GeV_10    uds   ntau_10to100GeV_10
const string test_particle_type = "uds";      // ntau_10GeV_10    uds   ntau_10to100GeV_10


const bool hyper_parameter = false;
const int dimension = 5;    // output dimensions  (one for beta, others are for coordinates)
// 8が先

const bool fine_tuning = false;
const int epoch = 25;       // 20   25
const int train_epoch = epoch*2-1;


void check_result(){ 
    int rawfilenum = 1;

    if(hyper_parameter && fine_tuning){ // condition check
        cout << "something wrong with setting boolian " << endl;
        abort();
    }

    TFile *filein[rawfilenum];
    TTree *tree[rawfilenum];
    int entry_max[rawfilenum];
    int total_entry_max=0;
    string picDirectory;
    // filein[0] = new TFile("../output/test_pandora.root");
    // filein[0] = new TFile("../output/test/test_100_pandora.root");
    // filein[0] = new TFile("../output/test/test_100_.root");
    // filein[0] = new TFile("../output/test.root");

    // filein[0] = new TFile("../output/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49.root");
    // filein[0] = new TFile("../output/ntau_to_uds/gravnet/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49_usd.root");
    // filein[0] = new TFile("../output/ntau_to_ntau/pandora/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49.root");
    // filein[0] = new TFile("../output/ntau_to_ntau/gravnet/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_49.root");
    // filein[0] = new TFile("../output/uds_to_uds/gravnet/tc_usd91_timingcut_forcealpha_thetaphi_19_usd.root");
    // filein[0] = new TFile("../output/uds_to_uds/pandora/tc_usd91_timingcut_forcealpha_thetaphi_19_usd.root");
    // filein[0] = new TFile("../output/hyper_parameter/dimensions/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10.root");
    // filein[0] = new TFile("../output/hyper_parameter/dimensions/uds_to_uds/tc_uds91_timingcut_forcealpha_thetaphi_5D_49_uds.root");
    // filein[0] = new TFile("../output/skimmed/ntau_to_ntau/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10.root");

        // new clustering
    // filein[0] = new TFile("../output/new_clustering/hyper_parameter/dimensions/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_3D_49_ntau_10GeV_10.root");
    // filein[0] = new TFile("../output/new_clustering/hyper_parameter/dimensions/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_4D_49_ntau_10GeV_10.root");
    // filein[0] = new TFile("../output/new_clustering/hyper_parameter/dimensions/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10.root");
    // filein[0] = new TFile("../output/new_clustering/hyper_parameter/dimensions/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_9D_49_ntau_10GeV_10.root");
    // filein[0] = new TFile("../output/new_clustering/hyper_parameter/dimensions/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_17D_49_ntau_10GeV_10.root");
    
    // learning rate
    // filein[0] = new TFile("../output/hyper_parameter/learning_rate/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10_lr1e-4.root");
    // filein[0] = new TFile("../output/hyper_parameter/learning_rate/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10_lr5e-5.root");
    // filein[0] = new TFile("../output/hyper_parameter/learning_rate/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10_lr2e-5.root");
    // filein[0] = new TFile("../output/hyper_parameter/learning_rate/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_5D_49_ntau_10GeV_10_lr9e-6.root");

    // pandora
    // filein[0] = new TFile("../output/skimmed/ntau_to_ntau/pandora/tc_ntau_timingcut_forcealpha_thetaphi_49_ntau.root");
    // filein[0] = new TFile("../output/skimmed/uds_to_uds/pandora/tc_uds_timingcut_forcealpha_thetaphi_49_uds.root");

    // energy regression
    // filein[0] = new TFile("../output/energy_regression/new_clustering/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10.root");
    // filein[0] = new TFile("../output/energy_regression/new_clustering/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10_MSE_edit.root");
    // filein[0] = new TFile("../output/energy_regression/new_clustering/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10_MSE.root");
    filein[0] = new TFile("../output/energy_regression/new_clustering/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10_MSE_edit_energyTree.root");
    filein[0] = new TFile("../output/energy_regression/new_clustering/tc_ntau_10GeV_10_5D_49_ntau_10GeV_10_MSE_edit_energyTree.root");

    if(hyper_parameter){
        const string rootDir = train_particle_type == "uds91" ? "uds_to_uds/" : "";
        filein[0] = new TFile(Form("../output/hyper_parameter/dimensions/%stc_%s_timingcut_forcealpha_thetaphi_%dD_49_%s.root",rootDir.c_str(),train_particle_type.c_str(),dimension,test_particle_type.c_str()));
        picDirectory = Form("../pic/output_dimension/D%d",dimension);
    }
    if(fine_tuning){
        filein[0] = new TFile(Form("../output/fine_tuning/tc_ntau_10GeV_10_uds_timingcut_forcealpha_thetaphi_%d_%s.root",train_epoch,test_particle_type.c_str()));
        picDirectory = Form("../pic/fine_tuning");
    }

    cout << picDirectory << endl;

    for(int i=0; i<rawfilenum; i++){
        tree[i] = (TTree*) filein[i]->Get("t");
        entry_max[i] = tree[i]->GetEntries();
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
    TH1F *purity[nParticle];
    TH1F *purity_energy[nParticle][nEnergy];
    TH1F *purity_energy_normalize[nParticle][nEnergy];
    TH2F *purity2d[nParticle];
    TH1F *efficiency[nParticle];
    TH1F *efficiency_energy[nParticle][nEnergy];
    TH1F *efficiency_energy_normalize[nParticle][nEnergy];
    TH2F *efficiency2d[nParticle];
    for(int ip=0;ip<nParticle;ip++){
        purity[ip] = new TH1F(Form("purity_%d",ip), Form("%s purity (MC energy>1 GeV)",particleNames[ip].c_str()), 101,0,1.01);
        purity[ip]->SetXTitle("purity (edep_match/edep_reco)");
        purity2d[ip] = new TH2F(Form("purity2d_%d",ip), Form("%s purity",particleNames[ip].c_str()), 101,0,1.01, 24,0,12);
        purity2d[ip]->SetXTitle("purity (edep_match/edep_reco)");
        purity2d[ip]->SetYTitle("MC energy (edep)");
        efficiency[ip] = new TH1F(Form("efficiency_%d",ip), Form("%s efficiency (MC energy>1 GeV)",particleNames[ip].c_str()), 101,0,1.01);
        efficiency[ip]->SetXTitle("efficiency (edep_match/edep)");
        efficiency2d[ip] = new TH2F(Form("efficiency2d_%d",ip), Form("%s efficiency",particleNames[ip].c_str()), 101,0,1.01, 24,0,12);
        efficiency2d[ip]->SetXTitle("efficiency (edep_match/edep)");
        efficiency2d[ip]->SetYTitle("MC energy (edep)");
        for(int ie=0;ie<nEnergy;ie++){
            string title = ie==0 ? Form("%s purity",particleNames[ip].c_str()) : Form("%s purity %d-%d GeV",particleNames[ip].c_str(),ie,ie+1);
            purity_energy[ip][ie] = new TH1F(Form("purity_%d_%d",ip,ie), title.c_str(), 51,0,1.02);
            purity_energy[ip][ie]->SetXTitle("purity (edep_match/edep_reco)");
            purity_energy_normalize[ip][ie] = new TH1F(Form("purity_norm_%d_%d",ip,ie), title.c_str(), 51,0,1.02);
            purity_energy_normalize[ip][ie]->SetXTitle("purity (edep_match/edep_reco)");
            title = ie==0 ? Form("%s efficiency",particleNames[ip].c_str()) : Form("%s efficiency %d-%d GeV",particleNames[ip].c_str(),ie,ie+1);
            efficiency_energy[ip][ie] = new TH1F(Form("efficiency_%d_%d",ip,ie), title.c_str(), 51,0,1.02);
            efficiency_energy[ip][ie]->SetXTitle("efficiency (edep_match/edep)");
            efficiency_energy_normalize[ip][ie] = new TH1F(Form("efficiency_normalize_%d_%d",ip,ie), title.c_str(), 51,0,1.02);
            efficiency_energy_normalize[ip][ie]->SetXTitle("efficiency (edep_match/edep)");
        }
    }


    // data をとってきてる
    for(int irawfile=0; irawfile<rawfilenum; irawfile++){
        if(rawfilenum>1) cout << irawfile << "/" << rawfilenum << endl;

        // ssa
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

            if(edep>1) purity[itr]->Fill(pur);
            purity2d[itr]->Fill(pur,edep);
            if(edep<10){
                purity_energy[itr][(int)edep]->Fill(pur);
                purity_energy_normalize[itr][(int)edep]->Fill(pur);
                efficiency_energy[itr][(int)edep]->Fill(eff);
                efficiency_energy_normalize[itr][(int)edep]->Fill(eff);
            }
            if(edep>1) efficiency[itr]->Fill(eff);
            efficiency2d[itr]->Fill(eff,edep);
        }
    }

    // efficiency[0]->Draw();
    // purity_energy[0][0]->Draw();
    


    // gStyle->SetStatX(0.35);
    gStyle->SetOptStat("rme");
    gStyle->SetStatX(0.55);
    gStyle->SetStatY(0.9);
    gStyle->SetStatH(0.3);
    gStyle->SetStatW(0.4);
    // legends をもう少し大きくする

    TCanvas *compare = new TCanvas("compare","compare",1);
    compare->Divide(nParticle,2);
    for(int ip=0;ip<nParticle*2;ip++){
        // TPaveStats *s = (TPaveStats*) gPad->GetPrimitive("stats"); s->SetTextSize(0.1); s->SetX1NDC(0.5); s->SetY1NDC(0.5);
        compare->cd(ip+1);
        gPad->SetLogy();
        if(ip<nParticle) efficiency[ip]->Draw();
        else purity[ip-nParticle]->Draw();
    }


    TCanvas *compare2d = new TCanvas("compare2d","compare2d",1);
    compare2d->Divide(nParticle,2);
    for(int ip=0;ip<nParticle*2;ip++){
        compare2d->cd(ip+1);
        gPad->SetLogz();
        if(ip<nParticle) efficiency2d[ip]->Draw("colz");
        else purity2d[ip-nParticle]->Draw("colz");
    }


    int yaxis_height[nParticle*2];
    for(int ip=0;ip<nParticle*2;ip++){
        yaxis_height[ip] = 0;
        for(int ie=0;ie<nEnergy;ie++){
            if(ip<nParticle) yaxis_height[ip] = yaxis_height[ip]<efficiency_energy[ip][ie]->GetMaximum() ? efficiency_energy[ip][ie]->GetMaximum() : yaxis_height[ip];
            else yaxis_height[ip] = yaxis_height[ip]<purity_energy[ip-nParticle][ie]->GetMaximum() ? purity_energy[ip-nParticle][ie]->GetMaximum() : yaxis_height[ip];
        }
    }
    TCanvas *compare_energy = new TCanvas("compare_energy","compare_energy",1);
    compare_energy->Divide(nParticle,2);
    TLegend *legend[nParticle][2];

    for(int ip=0;ip<nParticle*2;ip++){
        compare_energy->cd(ip+1);
        gPad->SetLogy();

        legend[ip%nParticle][ip/nParticle] = new TLegend( 0.4, 0.6, 0.8, 0.9) ;

        for(int ie=0;ie<nEnergy;ie++){
            string drawOption = ie==0 ? "" : "same";
            int colorId = ie<9 ? ie+1 : ie+2;

            if(ip<nParticle){
                // efficiency_energy[ip][ie]->Rebin(2);
                efficiency_energy[ip][ie]->SetLineColor(colorId);
                efficiency_energy[ip][ie]->SetMarkerColor(colorId);
                efficiency_energy[ip][ie]->SetMaximum(yaxis_height[ip]*2);
                efficiency_energy[ip][ie]->Draw(drawOption.c_str());
                legend[ip%nParticle][ip/nParticle]->AddEntry(efficiency_energy[ip][ie], Form("%d-%d GeV",ie,ie+1) , "l");
                legend[ip%nParticle][ip/nParticle]->Draw();
            } else {
                // purity_energy[ip-nParticle][ie]->Rebin(2);
                purity_energy[ip-nParticle][ie]->SetLineColor(colorId);
                purity_energy[ip-nParticle][ie]->SetMarkerColor(colorId);
                purity_energy[ip-nParticle][ie]->SetMaximum(yaxis_height[ip]*2);
                purity_energy[ip-nParticle][ie]->Draw(drawOption.c_str());
                // compare_energy->cd(ip+1)->BuildLegend();
                legend[ip%nParticle][ip/nParticle]->AddEntry(purity_energy[ip-nParticle][ie], Form("%d-%d GeV",ie,ie+1) , "l");
                legend[ip%nParticle][ip/nParticle]->Draw();
            }
        }
    }


    TCanvas *compare_energy_normalized = new TCanvas("compare_energy_normalized","compare_energy_normalized",1);
    compare_energy_normalized->Divide(nParticle,2);
    for(int ip=0;ip<nParticle*2;ip++){
        compare_energy_normalized->cd(ip+1);
        gPad->SetLogy();

        for(int ie=0;ie<nEnergy;ie++){
            string drawOption = ie==0 ? "HIST" : "same HIST";
            int colorId = ie<9 ? ie+1 : ie+2;

            if(ip<nParticle){
                efficiency_energy_normalize[ip][ie]->SetLineColor(colorId);
                efficiency_energy_normalize[ip][ie]->SetMarkerColor(colorId);
                // efficiency_energy[ip][ie]->SetMaximum(yaxis_height[ip]*2);
                efficiency_energy_normalize[ip][ie]->Scale(1./efficiency_energy[ip][ie]->GetEntries());
                efficiency_energy_normalize[ip][ie]->Draw(drawOption.c_str());
                // legend[ip%nParticle][ip/nParticle]->AddEntry(efficiency_energy[ip][ie], Form("%d-%d GeV",ie,ie+1) , "l");
                // legend[ip%nParticle][ip/nParticle]->Draw();
            } else {
                purity_energy_normalize[ip-nParticle][ie]->SetLineColor(colorId);
                purity_energy_normalize[ip-nParticle][ie]->SetMarkerColor(colorId);
                // purity_energy[ip-nParticle][ie]->SetMaximum(yaxis_height[ip]*2);
                purity_energy_normalize[ip-nParticle][ie]->Scale(1./purity_energy[ip-nParticle][ie]->GetEntries());
                purity_energy_normalize[ip-nParticle][ie]->Draw(drawOption.c_str());
                // legend[ip%nParticle][ip/nParticle]->AddEntry(purity_energy[ip-nParticle][ie], Form("%d-%d GeV",ie,ie+1) , "l");
                // legend[ip%nParticle][ip/nParticle]->Draw();
            }
        }
    }



    if(saving_canvas){  // saving canvases
        string suffix = "";
        if(hyper_parameter) suffix = Form("_%s",test_particle_type.c_str());
        if(fine_tuning)     suffix = Form("_epoch%d_%s",epoch,test_particle_type.c_str());
        
        compare->SaveAs(Form("%s/efficiency_purity%s.pdf",picDirectory.c_str(),suffix.c_str()));
        compare2d->SaveAs(Form("%s/efficiency_purity_vs_energy%s.pdf",picDirectory.c_str(),suffix.c_str()));
        compare_energy->SaveAs(Form("%s/per_energy%s.pdf",picDirectory.c_str(),suffix.c_str()));
        compare_energy_normalized->SaveAs(Form("%s/per_energy_norm%s.pdf",picDirectory.c_str(),suffix.c_str()));
    }

}
