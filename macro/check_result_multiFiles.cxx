#include <cmath>

using namespace std;

// efficiency, purityを表示して保存するマクロ

const int Ntest_particle_type = 1;                                // 3;
const string test_particle_types[2] = {"ntau_10GeV_10", "uds"};   // {ntau_10GeV_10, uds, ntau_10to100GeV_10}
const string test_type[2] = {"tau", "jet"};   // {ntau_10GeV_10, uds, ntau_10to100GeV_10}

const int nParticle = 3;
string particleNames[3] = {"electron", "pion", "photon"};

const int Ndimension = 5;              // 5;
const int dimensions[5] = {3, 4, 5, 9, 17};   // {3, 4, 5, 9, 17};
// tg->GetXaxis()->SetBinLabel(tg->GetXaxis()->FindBin(i + 1.), dimensions[i]);

const bool jet_result = false;  // if true, tau train, jet test results are also drawn

// conditions
const bool saving_canvas = true;
// const string test_particle_type = "uds";      // ntau_10GeV_10    uds   ntau_10to100GeV_10


const bool hyper_parameter = true;
// const int dimension = 5;    // output dimensions  (one for beta, others are for coordinates)

const bool fine_tuning = false;
const int epoch = 25;       // 20   25
const int train_epoch = epoch*2-1;

const bool new_clustering = true;


float round_n(float num, double digit){
    num = num * pow(10,digit-1);
    num = round(num);
    return num / pow(10,digit-1);
}

void check_result_multiFiles(){ 
    int rawfilenum = Ndimension * Ntest_particle_type;

    if(hyper_parameter && fine_tuning){ // condition check
        cout << "something wrong with setting boolian " << endl;
        abort();
    }

    TFile *filein[rawfilenum];
    TTree *tree[rawfilenum];
    int entry_max[rawfilenum];
    int total_entry_max=0;
    string picDirectory;
    
    if(hyper_parameter){
        for(int irawfile=0;irawfile<rawfilenum;irawfile++){
            string ptype = test_particle_types[irawfile%Ntest_particle_type];
            int dimension = dimensions[irawfile/Ntest_particle_type];
            // cout << dimension << " " << ptype << endl;
            if(!new_clustering) filein[irawfile] = new TFile(Form("../output/hyper_parameter/dimensions/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_%dD_49_%s.root",dimension,ptype.c_str()));
            else filein[irawfile] = new TFile(Form("../output/new_clustering/hyper_parameter/dimensions/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_%dD_49_%s.root",dimension,ptype.c_str()));
        }
        picDirectory = Form("../pic/output_dimension");
    }
    if(fine_tuning){
        // filein[0] = new TFile(Form("../output/fine_tuning/tc_ntau_10GeV_10_uds_timingcut_forcealpha_thetaphi_%d_%s.root",train_epoch,test_particle_type.c_str()));
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
    vector<int> particledgValues = {11,-11, 211,-211, 22};
    // map<int,int> pdgParticle;
    // pdgParticle[11] = 0;
    // pdgParticle[-11] = 0;
    // pdgParticle[211] = 1;
    // pdgParticle[-211] = 1;
    // pdgParticle[22] = 2;
    
    const int nEnergy = 10;
    TH1F *purity[nParticle][Ntest_particle_type][Ndimension];
    TH1F *efficiency[nParticle][Ntest_particle_type][Ndimension];
    // TH1F *purity_energy[nParticle][nEnergy][rawfilenum];
    // TH1F *purity_energy_normalize[nParticle][nEnergy][rawfilenum];
    // TH2F *purity2d[nParticle][Ntest_particle_type][Ndimension];
    // TH1F *efficiency_energy[nParticle][nEnergy][rawfilenum];
    // TH1F *efficiency_energy_normalize[nParticle][nEnergy][rawfilenum];
    // TH2F *efficiency2d[nParticle][Ntest_particle_type][Ndimension];
    for(int ip=0;ip<nParticle;ip++){
        for(int itype=0; itype<Ntest_particle_type; itype++){
            for(int dim=0; dim<Ndimension; dim++){
                purity[ip][itype][dim] = new TH1F(Form("purity_%d_%d_%d",ip,itype,dim), Form("%s purity %dD (MC energy>1 GeV, %s test)",particleNames[ip].c_str(),dimensions[dim]-1,test_particle_types[itype].c_str()), 101,0,1.01);
                purity[ip][itype][dim]->SetXTitle("purity (edep_match/edep_reco)");
                efficiency[ip][itype][dim] = new TH1F(Form("efficiency_%d_%d_%d",ip,itype,dim), Form("%s efficiency %dD (MC energy>1 GeV, %s test)",particleNames[ip].c_str(),dimensions[dim]-1,test_particle_types[itype].c_str()), 101,0,1.01);
                efficiency[ip][itype][dim]->SetXTitle("efficiency (edep_match/edep)");
            }
          /*
            purity2d[ip][irawfile] = new TH2F(Form("purity2d_%d_%d",ip,irawfile), Form("%s purity",particleNames[ip].c_str()), 101,0,1.01, 24,0,12);
            purity2d[ip][irawfile]->SetXTitle("purity (edep_match/edep_reco)");
            purity2d[ip][irawfile]->SetYTitle("MC energy (edep)");
            efficiency2d[ip][irawfile] = new TH2F(Form("efficiency2d_%d_%d",ip,irawfile), Form("%s efficiency",particleNames[ip].c_str()), 101,0,1.01, 24,0,12);
            efficiency2d[ip][irawfile]->SetXTitle("efficiency (edep_match/edep)");
            efficiency2d[ip][irawfile]->SetYTitle("MC energy (edep)");
            
            for(int ie=0;ie<nEnergy;ie++){
                string title = ie==0 ? Form("%s purity",particleNames[ip].c_str()) : Form("%s purity %d-%d GeV",particleNames[ip].c_str(),ie,ie+1);
                purity_energy[ip][ie][irawfile] = new TH1F(Form("purity_%d_%d",ip,ie), title.c_str(), 51,0,1.02);
                purity_energy[ip][ie][irawfile]->SetXTitle("purity (edep_match/edep_reco)");
                purity_energy_normalize[ip][ie][irawfile] = new TH1F(Form("purity_norm_%d_%d",ip,ie), title.c_str(), 51,0,1.02);
                purity_energy_normalize[ip][ie][irawfile]->SetXTitle("purity (edep_match/edep_reco)");
                title = ie==0 ? Form("%s efficiency",particleNames[ip].c_str()) : Form("%s efficiency %d-%d GeV",particleNames[ip].c_str(),ie,ie+1);
                efficiency_energy[ip][ie][irawfile] = new TH1F(Form("efficiency_%d_%d",ip,ie), title.c_str(), 51,0,1.02);
                efficiency_energy[ip][ie][irawfile]->SetXTitle("efficiency (edep_match/edep)");
                efficiency_energy_normalize[ip][ie][irawfile] = new TH1F(Form("efficiency_normalize_%d_%d",ip,ie), title.c_str(), 51,0,1.02);
                efficiency_energy_normalize[ip][ie][irawfile]->SetXTitle("efficiency (edep_match/edep)");
            }
          */
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

            int itype = irawfile%Ntest_particle_type;
            int dim = irawfile/Ntest_particle_type;
            
            if(edep>1) purity[itr][itype][dim]->Fill(pur);
            // purity2d[itr][irawfile]->Fill(pur,edep);
            // if(edep<10){
            //     purity_energy[itr][(int)edep]->Fill(pur);
            //     purity_energy_normalize[itr][(int)edep]->Fill(pur);
            //     efficiency_energy[itr][(int)edep]->Fill(eff);
            //     efficiency_energy_normalize[itr][(int)edep]->Fill(eff);
            // }
            if(edep>1) efficiency[itr][itype][dim]->Fill(eff);
            // efficiency2d[itr][irawfile]->Fill(eff,edep);
        }
    }

    // efficiency[0]->Draw();
    // purity_energy[0][0]->Draw();
    


    gStyle->SetStatX(0.35);
    gStyle->SetStatY(0.9);
    // legends をもう少し大きくする
    gStyle->SetOptStat(0);

    TCanvas *compare[Ntest_particle_type];
    for(int itype=0;itype<Ntest_particle_type;itype++){
        compare[itype] = new TCanvas(Form("compare_%d",itype),Form("compare histograms %s",test_particle_types[itype].c_str()),1);
        compare[itype]->Divide(nParticle,2);
    }
    TGraph *dim_evolve_eff[Ntest_particle_type][nParticle];
    TGraph *dim_evolve_pur[Ntest_particle_type][nParticle];
    TLegend *legends_eff[Ntest_particle_type][nParticle];
    TLegend *legends_pur[Ntest_particle_type][nParticle];
    for(int itype=0;itype<Ntest_particle_type;itype++){
        for(int ip=0;ip<nParticle*2;ip++){
            // TPaveStats *s = (TPaveStats*) gPad->GetPrimitive("stats"); s->SetTextSize(0.1); s->SetX1NDC(0.5); s->SetY1NDC(0.5);
            compare[itype]->cd();
            compare[itype]->cd(ip+1);
            gPad->SetLogy();

            if(ip<nParticle){
                legends_eff[itype][ip] = new TLegend( 0.2, 0.6, 0.55, 0.9);
                dim_evolve_eff[itype][ip] = new TGraph();
            }
            // else {
            //     if(test_particle_types[itype]=="uds" && ip==nParticle*2-1) legends_pur[itype][ip-nParticle] = new TLegend( 0.5, 0.2, 0.85, 0.5);
            //     else legends_pur[itype][ip-nParticle] = new TLegend( 0.2, 0.6, 0.55, 0.9);
            // }
            else{
                legends_pur[itype][ip-nParticle] = new TLegend( 0.2, 0.65, 0.55, 0.9);
                dim_evolve_pur[itype][ip-nParticle] = new TGraph();
            }


            int line_color = 1;
            for(int dim=0; dim<Ndimension; dim++){
                string options = dim == 0 ? "" : "same";
                if(dimensions[dim]-1==16) line_color++;
                if(ip<nParticle){
                    string title_eff = Form("%s efficiency (MC energy>1 GeV, %s test)",particleNames[ip].c_str(),test_particle_types[itype].c_str());
                    float mean = round_n(efficiency[ip][itype][dim]->GetMean(),4.0);
                    dim_evolve_eff[itype][ip]->SetPoint(dim, dimensions[dim]-1, mean);
                    legends_eff[itype][ip]->AddEntry(efficiency[ip][itype][dim], Form("%dD mean=%4.3f",dimensions[dim]-1,mean) , "l");
                    if(dim==0) efficiency[ip][itype][dim]->SetTitle(title_eff.c_str());
                    efficiency[ip][itype][dim]->SetLineColor(dim+line_color);
                    efficiency[ip][itype][dim]->Draw(options.c_str());
                } else {
                    string title_pur = Form("%s purity (MC energy>1 GeV, %s test)",particleNames[ip-nParticle].c_str(),test_particle_types[itype].c_str());
                    float mean = round_n(purity[ip-nParticle][itype][dim]->GetMean(),4.0);
                    dim_evolve_pur[itype][ip-nParticle]->SetPoint(dim, dimensions[dim]-1, mean);
                    legends_pur[itype][ip-nParticle]->AddEntry(purity[ip-nParticle][itype][dim], Form("%dD mean=%4.3f",dimensions[dim]-1,mean) , "l");
                    if(dim==0) purity[ip-nParticle][itype][dim]->SetTitle(title_pur.c_str());
                    purity[ip-nParticle][itype][dim]->SetLineColor(dim+line_color);
                    purity[ip-nParticle][itype][dim]->Draw(options.c_str());
                }
            }
            if(ip<nParticle) legends_eff[itype][ip]->Draw("same");
            else legends_pur[itype][ip-nParticle]->Draw("same");
        }
    }
  /*
    for(int itype=0;itype<Ntest_particle_type;itype++){
        for(int ip=0;ip<nParticle*2;ip++){
            // TPaveStats *s = (TPaveStats*) gPad->GetPrimitive("stats"); s->SetTextSize(0.1); s->SetX1NDC(0.5); s->SetY1NDC(0.5);
            compare[itype]->cd();
            compare[itype]->cd(ip+1);
            gPad->SetLogy();

            legends_eff[itype][ip] = new TLegend( 0.2, 0.65, 0.55, 0.9);
            legends_pur[itype][ip] = new TLegend( 0.2, 0.65, 0.55, 0.9);

            int line_color = 1;
            for(int dim=0; dim<Ndimension; dim++){
                string options = dim == 0 ? "" : "same";
                if(dimensions[dim]-1==16) line_color++;
                if(ip<nParticle){
                    float mean = round_n(efficiency[ip][itype][dim]->GetMean(),4.0);
                    legends_eff[itype][ip]->AddEntry(efficiency[ip][itype][dim], Form("%dD mean=%4.3f",dimensions[dim]-1,mean) , "l");
                    efficiency[ip][itype][dim]->SetLineColor(dim+line_color);
                    efficiency[ip][itype][dim]->Draw(options.c_str());
                } else {
                    float mean = round_n(purity[ip-nParticle][itype][dim]->GetMean(),4.0);
                    legends_pur[itype][ip]->AddEntry(purity[ip-nParticle][itype][dim], Form("%dD mean=%4.3f",dimensions[dim]-1,mean) , "l");
                    purity[ip-nParticle][itype][dim]->SetLineColor(dim+line_color);
                    purity[ip-nParticle][itype][dim]->Draw(options.c_str());
                }
            }
            if(ip<nParticle) legends_eff[itype][ip]->Draw("same");
            else legends_pur[itype][ip]->Draw("same");
        }
    }
  */

  /*
    TCanvas *compare2d[Ntest_particle_type] = new TCanvas("compare2d","compare2d",1);
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
  */

    int particle_color[3] = {1, 2, 4};     // color of graph for pdg(e, pi, gamma)
    int test_marker[2] = {2, 3};           // marker of graph for test particle(tau, jets).
    TCanvas *compare_outDim[2];
    TLegend *legends_eff_outDim = new TLegend( 0.5, 0.11, 0.85, 0.41);
    // TLegend *legends_pur_outDim;
    double pandora_results[6] = {0.993, 0.940, 0.991, 0.918, 0.946, 0.972};
    TF1 *pandora_eff[nParticle];
    TF1 *pandora_pur[nParticle];
    double graph_minimum_eff = jet_result ? 0.8 : 0.92;
    double graph_minimum_pur = jet_result ? 0.6 : 0.75;
    for(int ip=0;ip<nParticle;ip++){
        pandora_eff[ip] = new TF1(Form("pandora_eff_%d",ip),Form("%f",pandora_results[ip]),-1,20);
        pandora_eff[ip]->SetLineWidth(1);
        pandora_eff[ip]->SetLineStyle(7);
        pandora_pur[ip] = new TF1(Form("pandora_pur_%d",ip),Form("%f",pandora_results[ip+nParticle]),-1,20);
        pandora_pur[ip]->SetLineWidth(1);
        pandora_pur[ip]->SetLineStyle(7);
    }
    for(int ep=0;ep<2;ep++){
        string title = ep==0 ? "efficiency" : "purity";
        compare_outDim[ep] = new TCanvas(Form("compare_outDim_%d",ep),Form("%s",title.c_str()),1);
        compare_outDim[ep]->Divide(nParticle,2);
    }
    for(int ep=0;ep<2;ep++){
        int ngraph = 0;
        compare_outDim[ep]->cd();
        for(int itype=0;itype<Ntest_particle_type;itype++){
            if(!jet_result && test_type[itype]=="jet") continue; 
            for(int ip=0;ip<nParticle;ip++){
                string options = ngraph == 0 ? "APL" : "PL";
                if(ep==0){
                    dim_evolve_eff[itype][ip]->GetXaxis()->SetTitle("output dimensions");
                    dim_evolve_eff[itype][ip]->GetYaxis()->SetTitle("efficiency");
                    dim_evolve_eff[itype][ip]->SetMaximum(1);
                    dim_evolve_eff[itype][ip]->SetMinimum(graph_minimum_eff);
                    dim_evolve_eff[itype][ip]->SetLineColor(particle_color[ip]);
                    dim_evolve_eff[itype][ip]->SetMarkerColor(particle_color[ip]);
                    dim_evolve_eff[itype][ip]->SetMarkerStyle(test_marker[itype]);
                    dim_evolve_eff[itype][ip]->Draw(options.c_str());
                    string legend_content = jet_result ? Form("  (GNN %s test)",test_type[itype].c_str()) : "  (GNN)";
                    legends_eff_outDim->AddEntry(dim_evolve_eff[itype][ip], Form("%s%s",particleNames[ip].c_str(),legend_content.c_str()), "p");
                    
                    pandora_eff[ip]->SetLineColor(particle_color[ip]);
                    pandora_eff[ip]->Draw("same");
                } else {
                    dim_evolve_pur[itype][ip]->GetXaxis()->SetTitle("output dimensions");
                    dim_evolve_pur[itype][ip]->GetYaxis()->SetTitle("purity");
                    dim_evolve_pur[itype][ip]->SetMaximum(1);
                    dim_evolve_pur[itype][ip]->SetMinimum(graph_minimum_pur);
                    dim_evolve_pur[itype][ip]->SetLineColor(particle_color[ip]);
                    dim_evolve_pur[itype][ip]->SetMarkerColor(particle_color[ip]);
                    dim_evolve_pur[itype][ip]->SetMarkerStyle(test_marker[itype]);
                    dim_evolve_pur[itype][ip]->Draw(options.c_str());
                    pandora_pur[ip]->SetLineColor(particle_color[ip]);
                    pandora_pur[ip]->Draw("same");
                }
                ngraph++;
            }
        }
        if(ep==0) for(int ip=0;ip<nParticle;ip++) legends_eff_outDim->AddEntry(pandora_eff[ip], Form("%s  (PandoraPFA)",particleNames[ip].c_str()), "l");
        if(!jet_result || ep==0) legends_eff_outDim->Draw("same");
    }


    if(saving_canvas){  // saving canvases
        string suffix = "";
        
        for(int itype=0;itype<Ntest_particle_type;itype++){
            if(hyper_parameter){
                if(!new_clustering) suffix = Form("_%s",test_particle_types[itype].c_str());
                else suffix = Form("_%s_new_clustering",test_particle_types[itype].c_str());
            }
            if(fine_tuning)     suffix = Form("_epoch%d_%s",epoch,test_particle_types[itype].c_str());

            compare[itype]->SaveAs(Form("%s/efficiency_purity%s.pdf",picDirectory.c_str(),suffix.c_str()));
        // compare2d->SaveAs(Form("%s/efficiency_purity_vs_energy%s.pdf",picDirectory.c_str(),suffix.c_str()));
        // compare_energy->SaveAs(Form("%s/per_energy%s.pdf",picDirectory.c_str(),suffix.c_str()));
        // compare_energy_normalized->SaveAs(Form("%s/per_energy_norm%s.pdf",picDirectory.c_str(),suffix.c_str()));
        }

        string suf = jet_result ? "_withJets" : "";
        if(new_clustering)  suffix = Form("%s_new_clustering",suf.c_str());
        compare_outDim[0]->SaveAs(Form("%s/efficiency%s.pdf",picDirectory.c_str(),suffix.c_str()));
        compare_outDim[1]->SaveAs(Form("%s/purity%s.pdf",picDirectory.c_str(),suffix.c_str()));
    }

}
