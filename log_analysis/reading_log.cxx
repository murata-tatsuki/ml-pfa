#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;


vector<string> split(string str, char del) {
    int first = 0;
    int last = str.find_first_of(del);

    vector<string> result;

    while (first < str.size()) {
        string subStr(str, first, last - first);

        result.push_back(subStr);

        first = last + 1;
        last = str.find_first_of(del, first);

        if (last == string::npos) {
            last = str.size();
        }
    }

    return result;
}



// void reading_log(string logfilePath){ 
  // ifstream file(logfilePath);  // 読み込むファイルのパスを指定
void reading_log(){ 
  int nepoch = -1;
  vector<double> learning_rates;
  vector<double> train_loss;
  vector<double> l_v;
  vector<double> l_beta;
  vector<double> l_e;
  vector<double> returning;




  // ifstream file("../log/energy_regression/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD5_2024_11_22_144511_alpha_momentum.log");  // long tau task
  // ifstream file("../log/energy_regression/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD5_2025_01_08_185604_alpha_tracker_modifing_momentum.log");  // 読み込むファイルのパスを指定
  // ifstream file("../log/energy_regression/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD5_2025_01_30_161918_alpha_momentum.log");  // 読み込むファイルのパスを指定
  ifstream file("../log/energy_regression/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD5_2025_02_10_170311_alpha_tracker_momentum.log");  // 読み込むファイルのパスを指定
  // ifstream file("../log/energy_regression/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD5_2025_02_07_150149_alpha_momentum.log");  // 読み込むファイルのパスを指定
  // ifstream file("../log/energy_regression/tc_uds91_timingcut_forcealpha_thetaphi_outputD5_2025_01_30_164401_alpha_momentum.log");  // 読み込むファイルのパスを指定
  string line;

  while(getline(file, line)){  // 1行ずつ読み込む

    // getting number of epochs
    if(line.find("epoch ")!=string::npos){
      line.erase(0,line.find("epoch ")+6);
      nepoch = stoi(line);
    }

    // getting learning rates
    if(line.find("learning rate :  ")!=string::npos){
      learning_rates.clear();
      line.erase(0,line.find("epoch ")+19);
      line.erase(line.end()-1,line.end());

      string lr;
      string del = ", ";
      stringstream ssline(line);
      while(getline(ssline, lr, ' ')){
        if(lr.find(",")!=string::npos) lr.erase(lr.end()-1,lr.end());
        // vector<string> linee = split(line,", ");
        double doublelr = stod(lr);
        learning_rates.push_back(doublelr);
      }
    }

    // getting train loss
    if(line.find("train loss : ")!=string::npos){
      line.erase(0,line.find("train loss :  ")+14);
      train_loss.push_back(stod(line));
    }

    // getting total validation loss
    if(line.find("Returning ")!=string::npos){
      line.erase(0,line.find("Returning ")+10);
      returning.push_back(stod(line));
    }

    // getting validation l_v
    if(line.find("L_V ")!=string::npos){
      // cout << line << endl;
      line.erase(0,line.find("= ")+3);
      // cout << line << endl;
      // cout << stod(line) << endl;
      l_v.push_back(stod(line));
    }

    // getting validation l_beta
    if(line.find("L_beta ")!=string::npos){
      line.erase(0,line.find("= ")+3);
      // cout << stod(line) << endl;
      l_beta.push_back(stod(line));
    }

    // getting validation l_e
    if(line.find("L_E ")!=string::npos){
      line.erase(0,line.find("= ")+3);
      // cout << stod(line) << endl;
      l_e.push_back(stod(line));
    }

    // cout << line << endl;
  }



  TGraph *g_lr = new TGraph();
  g_lr->GetXaxis()->SetTitle("epoch"); 
  g_lr->GetYaxis()->SetTitle("learning rate"); 
  g_lr->SetMinimum(0); 
  // g_loss->SetMaximum(6); 
  for(int i=0;i<nepoch;i++){
    g_lr->SetPoint(i,i,learning_rates[i]);
  }
  TCanvas *c_lr = new TCanvas("c_lr","c_lr",1);
  c_lr->cd();
  g_lr->Draw();






  TGraph *g_loss = new TGraph();
  TGraph *g_LV = new TGraph();
  TGraph *g_Lbeta = new TGraph();
  TGraph *g_LE = new TGraph();
  g_loss->GetXaxis()->SetTitle("epoch"); 
  g_loss->GetYaxis()->SetTitle("loss"); 
  // g_loss->SetTitle("beta * E"); 
  g_loss->SetMinimum(0); 
  // g_loss->SetMaximum(6); 
  g_loss->SetLineColor(1); 
  g_LV->SetLineColor(2); 
  g_LV->GetXaxis()->SetTitle("epoch"); 
  g_LV->GetYaxis()->SetTitle("loss"); 
  g_Lbeta->SetLineColor(3); 
  g_LE->SetLineColor(4); 

  TLegend *legend = new TLegend( 0.4, 0.48, 0.8, 0.78);
  legend->AddEntry( g_loss, "total loss", "l"); // AddEntry( pointer , "interpretation" , "option" )
  legend->AddEntry( g_LV, "L_V", "l"); // option は　"f"=box, "l"="L"=line, "p"=marker
  legend->AddEntry( g_Lbeta, "L_beta", "l") ;
  legend->AddEntry( g_LE, "L_E", "l") ;
  // legend->AddEntry( g5, "total loss w/o L_E", "l") ;
  legend->SetFillColor(0);


  for(int i=0;i<nepoch;i++){
    // g5->SetPoint(i,i,returning5[i]);

    g_loss->SetPoint(i,i,returning[i]);
    g_LV->SetPoint(i,i,l_v[i]);
    g_Lbeta->SetPoint(i,i,l_beta[i]);
    g_LE->SetPoint(i,i,l_e[i]);
  }

  TCanvas *c1 = new TCanvas("c1","c1",1);
  c1->cd();
  g_loss->Draw();
  // g5->Draw("same");
  g_LV->Draw("same");
  g_Lbeta->Draw("same");
  g_LE->Draw("same");
  // legend->Draw();

  TCanvas *c2 = new TCanvas("c2","c2",1);
  c2->cd();
  g_LV->SetMaximum(6); 
  g_LV->SetMinimum(0); 
  g_LV->Draw();
  g_loss->Draw("same");
  // g5->Draw("same");
  g_Lbeta->Draw("same");
  g_LE->Draw("same");
  legend->Draw();


}
