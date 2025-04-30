#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

using namespace std;

vector<double> l_v;
vector<double> l_v_att_charged;
vector<double> l_v_att_neutral;
vector<double> l_v_rep_charged;
vector<double> l_v_rep_neutral;
vector<double> l_beta;
vector<double> l_e;
vector<double> l_e_tracker;
vector<double> l_e_cond;
vector<double> l_e_cluster;
vector<double> returning;
vector<double> train_loss;
vector<double> train_l_v;
vector<double> train_l_v_att_charged;
vector<double> train_l_v_att_neutral;
vector<double> train_l_v_rep_charged;
vector<double> train_l_v_rep_neutral;
vector<double> train_l_beta;
vector<double> train_l_e;
vector<double> train_l_e_tracker;
vector<double> train_l_e_cond;
vector<double> train_l_e_cluster;

int epoch_noLE = -1;
// int epoch_noLE = 15;


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

void getting_loss(string line, bool get_train){
    // getting  l_v
    if(line.find("L_V ")!=string::npos){
      // cout << line << endl;
      line.erase(0,line.find("= ")+3);
      // cout << line << endl;
      // cout << stod(line) << endl;
      if(!get_train) l_v.push_back(stod(line));
      else train_l_v.push_back(stod(line));
    }

    // getting  L_V_attractive_charged
    if(line.find("L_V_attractive_charged ")!=string::npos){
      line.erase(0,line.find("= ")+3);
      if(!get_train) l_v_att_charged.push_back(stod(line));
      else train_l_v_att_charged.push_back(stod(line));
    }

    // getting  L_V_attractive_neutral
    if(line.find("L_V_attractive_neutral ")!=string::npos){
      line.erase(0,line.find("= ")+3);
      if(!get_train) l_v_att_neutral.push_back(stod(line));
      else train_l_v_att_neutral.push_back(stod(line));
    }

    // getting  L_V_attractive_charged
    if(line.find("L_V_repulsive_charged ")!=string::npos){
      line.erase(0,line.find("= ")+3);
      if(!get_train) l_v_rep_charged.push_back(stod(line));
      else train_l_v_rep_charged.push_back(stod(line));
    }

    // getting  L_V_attractive_neutral
    if(line.find("L_V_repulsive_neutral ")!=string::npos){
      line.erase(0,line.find("= ")+3);
      if(!get_train) l_v_rep_neutral.push_back(stod(line));
      else train_l_v_rep_neutral.push_back(stod(line));
    }

    // getting l_beta
    if(line.find("L_beta ")!=string::npos){
      line.erase(0,line.find("= ")+3);
      if(!get_train) l_beta.push_back(stod(line));
      else train_l_beta.push_back(stod(line));
    }

    // getting l_e
    if(line.find("L_E ")!=string::npos){
      line.erase(0,line.find("= ")+3);
      if(!get_train) l_e.push_back(stod(line));
      else train_l_e.push_back(stod(line));
    }

    // getting L_E_tracker
    if(line.find("L_E_tracker ")!=string::npos){
      line.erase(0,line.find("= ")+3);
      if(!get_train) l_e_tracker.push_back(stod(line));
      else train_l_e_tracker.push_back(stod(line));
    }

    if(line.find("L_E_cond ")!=string::npos){
      line.erase(0,line.find("= ")+3);
      if(!get_train) l_e_cond.push_back(stod(line));
      else train_l_e_cond.push_back(stod(line));
    }

    // getting L_E_cluster
    if(line.find("L_E_cluster ")!=string::npos){
      line.erase(0,line.find("= ")+3);
      if(!get_train) l_e_cluster.push_back(stod(line));
      else train_l_e_cluster.push_back(stod(line));
    }
}



// void reading_log(string logfilePath){ 
  // ifstream file(logfilePath);  // 読み込むファイルのパスを指定
void reading_log_cluster(){ 
  int nepoch = -1;
  vector<double> learning_rates, grads;
  vector<vector<double>> gradients;
  int grad_max_digit=0, grad_min_digit=0;
  
  




  // ifstream file("../log/energy_regression/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD5_2024_11_22_144511_alpha_momentum.log");  // long tau task
  // ifstream file("../log/energy_regression/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD5_2025_01_08_185604_alpha_tracker_modifing_momentum.log");  // 読み込むファイルのパスを指定
  // ifstream file("../log/energy_regression/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD5_2025_01_30_161918_alpha_momentum.log");
  // ifstream file("../log/energy_regression/tc_uds91_timingcut_forcealpha_thetaphi_outputD5_2025_03_12_141129_alpha_tracker_momentum.log");
  // ifstream file("../log/energy_regression/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD5_2025_02_07_150149_alpha_momentum.log");
  // ifstream file("../log/energy_regression/tc_uds91_timingcut_forcealpha_thetaphi_outputD5_2025_01_30_164401_alpha_momentum.log");


  ifstream file("../log/tc_ntau_10GeV_10_timingcut_forcealpha_thetaphi_outputD17_2025_04_25_174831.log");
  string line;
  bool gradient_check = false;

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
      continue;
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

    getting_loss(line, line.find("train")!=string::npos);
    // cout << line << endl;

    // getting gradient
    if((line.find("[")!=string::npos || gradient_check) && nepoch>=0){
      // cout << line << "      ";
      gradient_check = true;
      line.erase(0,1);
      string grad;
      stringstream ssline(line);
      while(getline(ssline, grad, ' ')){
        double doublegrad = stod(grad);
        // cout << doublegrad << " ";
        grads.push_back(doublegrad);
      }
      if(line.find("]")!=string::npos){
        gradient_check = false;
        gradients.push_back(grads);
        // for(int i=0;i<grads.size();i++) cout << grads[i] << " ";
        // cout << endl;
        grads.clear();
      }
      // finding maximum/minimum digits
      string dig;
      line.erase(0,line.find("e")+1);
      stringstream digits(line);
      while(getline(digits, dig, 'e')){
        double intdig = stoi(dig);
        if(grad_max_digit<intdig) grad_max_digit=intdig;
        if(grad_min_digit>intdig) grad_min_digit=intdig;
        // cout << intdig << " ";
        // grads.push_back(doublegrad);
      }
      // cout << endl;
    }
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
  TGraph *g_LV_att_charged = new TGraph();
  TGraph *g_LV_att_neutral = new TGraph();
  TGraph *g_LV_rep_charged = new TGraph();
  TGraph *g_LV_rep_neutral = new TGraph();
  TGraph *g_Lbeta = new TGraph();
  TGraph *g_LE = new TGraph();
  TGraph *g_LEcond = new TGraph();
  TGraph *g_LEcluster = new TGraph();
  g_loss->GetXaxis()->SetTitle("epoch"); 
  g_loss->GetYaxis()->SetTitle("validation loss"); 
  // g_loss->SetTitle("beta * E"); 
  g_loss->SetMinimum(0); 
  // g_loss->SetMaximum(6); 
  g_loss->SetLineColor(1); 
  g_LV->SetLineColor(2); 
  g_LV->GetXaxis()->SetTitle("epoch"); 
  g_LV->GetYaxis()->SetTitle("loss"); 
  g_Lbeta->SetLineColor(3); 
  g_LE->SetLineColor(4); 
  g_LEcond->SetLineColor(7); 
  g_LEcluster->SetLineColor(9); 
  g_LV_att_charged->SetLineColor(1); 
  g_LV_att_neutral->SetLineColor(2); 
  g_LV_rep_charged->SetLineColor(3); 
  g_LV_rep_neutral->SetLineColor(4); 

  TLegend *legend = new TLegend( 0.4, 0.48, 0.8, 0.78);
  legend->AddEntry( g_loss, "total loss", "l"); // AddEntry( pointer , "interpretation" , "option" )
  legend->AddEntry( g_LV, "L_V", "l"); // option は　"f"=box, "l"="L"=line, "p"=marker
  legend->AddEntry( g_Lbeta, "L_beta", "l") ;
  legend->AddEntry( g_LE, "L_E", "l") ;
  legend->AddEntry( g_LEcond, "L_E_condpoint", "l") ;
  legend->AddEntry( g_LEcluster, "L_E_cluster", "l") ;
  // legend->AddEntry( g5, "total loss w/o L_E", "l") ;
  legend->SetFillColor(0);


  TGraph *g_train_loss = new TGraph();
  TGraph *g_train_LV = new TGraph();
  TGraph *g_train_LV_att_charged = new TGraph();
  TGraph *g_train_LV_att_neutral = new TGraph();
  TGraph *g_train_LV_rep_charged = new TGraph();
  TGraph *g_train_LV_rep_neutral = new TGraph();
  TGraph *g_train_Lbeta = new TGraph();
  TGraph *g_train_LE = new TGraph();
  TGraph *g_train_LEtracker = new TGraph();
  TGraph *g_train_LEcluster = new TGraph();
  g_train_loss->GetXaxis()->SetTitle("epoch"); 
  g_train_loss->GetYaxis()->SetTitle("train loss"); 
  g_train_loss->SetMinimum(0); 
  g_train_loss->SetLineColor(1); 
  g_train_LV->SetLineColor(2); 
  g_train_LV->GetXaxis()->SetTitle("epoch"); 
  g_train_LV->GetYaxis()->SetTitle("train loss"); 
  g_train_Lbeta->SetLineColor(3); 
  g_train_LE->SetLineColor(4); 
  g_train_LEtracker->SetLineColor(7); 
  g_train_LEcluster->SetLineColor(9); 
  g_train_LV_att_charged->SetLineColor(1); 
  g_train_LV_att_neutral->SetLineColor(2); 
  g_train_LV_rep_charged->SetLineColor(3); 
  g_train_LV_rep_neutral->SetLineColor(4); 

  TLegend *legend_train = new TLegend( 0.4, 0.48, 0.8, 0.78);
  legend_train->AddEntry( g_train_loss, "total loss", "l"); // AddEntry( pointer , "interpretation" , "option" )
  legend_train->AddEntry( g_train_LV, "L_V", "l"); // option は　"f"=box, "l"="L"=line, "p"=marker
  legend_train->AddEntry( g_train_Lbeta, "L_beta", "l") ;
  legend_train->AddEntry( g_train_LE, "L_E", "l") ;
  legend_train->AddEntry( g_train_LEtracker, "L_E_condpoint", "l") ;
  legend_train->AddEntry( g_train_LEcluster, "L_E_cluster", "l") ;
  legend_train->SetFillColor(0);

  for(int i=0;i<nepoch;i++){
    g_loss->SetPoint(i,i,i>epoch_noLE ? returning[i] : returning[i] - l_e[i]);
    g_LV->SetPoint(i,i,l_v[i]);
    g_LV_att_charged->SetPoint(i,i,l_v_att_charged[i]);
    g_LV_att_neutral->SetPoint(i,i,l_v_att_neutral[i]);
    g_LV_rep_charged->SetPoint(i,i,l_v_rep_charged[i]);
    g_LV_rep_neutral->SetPoint(i,i,l_v_rep_neutral[i]);
    g_Lbeta->SetPoint(i,i,l_beta[i]);
    if(i>epoch_noLE){
      g_LE->SetPoint(i-epoch_noLE-1,i,l_e[i]);
      if(l_e_tracker.size()>0) g_LEcond->SetPoint(i-epoch_noLE-1,i,l_e_tracker[i]);
      if(l_e_cond.size()>0) g_LEcond->SetPoint(i-epoch_noLE-1,i,l_e_cond[i]);
      // g_LEcond->SetPoint(i,i,l_e_tracker[i]-l_e_cluster[i]);
      g_LEcluster->SetPoint(i-epoch_noLE-1,i,l_e_cluster[i]);
    }

    if(train_l_v.size()==0) continue;
    g_train_loss->SetPoint(i,i,i>epoch_noLE ? train_l_v[i]+train_l_beta[i]+train_l_e[i] : train_l_v[i]+train_l_beta[i]);
    g_train_LV->SetPoint(i,i,train_l_v[i]);
    g_train_LV_att_charged->SetPoint(i,i,train_l_v_att_charged[i]);
    g_train_LV_att_neutral->SetPoint(i,i,train_l_v_att_neutral[i]);
    g_train_LV_rep_charged->SetPoint(i,i,train_l_v_rep_charged[i]);
    g_train_LV_rep_neutral->SetPoint(i,i,train_l_v_rep_neutral[i]);
    g_train_Lbeta->SetPoint(i,i,train_l_beta[i]);
    if(i>epoch_noLE){
      g_train_LE->SetPoint(i-epoch_noLE-1,i,train_l_e[i]);
      if(train_l_e_tracker.size()>0) g_train_LEtracker->SetPoint(i-epoch_noLE-1,i,train_l_e_tracker[i]);
      if(train_l_e_cond.size()>0) g_train_LEtracker->SetPoint(i-epoch_noLE-1,i,train_l_e_cond[i]);
      g_train_LEcluster->SetPoint(i-epoch_noLE-1,i,train_l_e_cluster[i]);
    }
  }

  TCanvas *c1 = new TCanvas("c1","c1",1);
  c1->cd();
  g_loss->Draw();
  // g5->Draw("same");
  g_LV->Draw("same");
  g_Lbeta->Draw("same");
  g_LE->Draw("same");
  g_LEcond->Draw("same");
  g_LEcluster->Draw("same");
  // legend->Draw();

  TCanvas *c2 = new TCanvas("c2","c2",1);
  c2->cd();
  c2->SetGrid();
  g_LV->SetMaximum(6); 
  g_LV->SetMinimum(0); 
  g_LV->Draw();
  g_loss->Draw("same");
  // g5->Draw("same");
  g_Lbeta->Draw("same");
  g_LE->Draw("same");
  g_LEcond->Draw("same");
  g_LEcluster->Draw("same");
  legend->Draw();

  TCanvas *c1_train = new TCanvas("c1_train","c1_train",1);
  c1_train->cd();
  g_train_loss->Draw();
  g_train_LV->Draw("same");
  g_train_Lbeta->Draw("same");
  g_train_LE->Draw("same");
  g_train_LEtracker->Draw("same");
  g_train_LEcluster->Draw("same");

  TCanvas *c2_train = new TCanvas("c2_train","c2_train",1);
  c2_train->cd();
  c2_train->SetGrid();
  g_train_LV->SetMaximum(6); 
  g_train_LV->SetMinimum(0); 
  g_train_LV->Draw();
  g_train_loss->Draw("same");
  g_train_Lbeta->Draw("same");
  g_train_LE->Draw("same");
  g_train_LEtracker->Draw("same");
  g_train_LEcluster->Draw("same");
  legend_train->Draw();








  // assert(gradients.size()!=nepoch);

  // gradient
  int digit_range = grad_max_digit - grad_min_digit + 1;
  cout << "digits : " << grad_max_digit << " , " <<  grad_min_digit << endl;
  int ngrad = gradients[0].size();

  int last_epoch_digits[ngrad];
  for(int igrad=0;igrad<ngrad;igrad++){
    int dig = log10(gradients[nepoch-1][igrad]) > 0 ? log10(gradients[nepoch-1][igrad]) : log10(gradients[nepoch-1][igrad]) - 1;
    // cout << gradients[nepoch-1][igrad] << ", " << dig << endl;
    last_epoch_digits[igrad] = dig;
  }





  TGraph *g_gradients[ngrad];
  for(int igrad=0;igrad<ngrad;igrad++){
    g_gradients[igrad] = new TGraph();
    g_gradients[igrad]->GetXaxis()->SetTitle("epoch"); 
    g_gradients[igrad]->GetYaxis()->SetTitle("gradient"); 
    // g_gradients[i]->SetMinimum(0); 
    // g_gradients[i]->SetLineColor(1); 
  }
  TCanvas *c_gradients[digit_range];
  TLegend *c_legend[digit_range];
  int ngraph_digit[digit_range], line_color[digit_range];
  for(int idig=0;idig<digit_range;idig++){
    c_gradients[idig] = new TCanvas(Form("c_gradients_%d",idig+grad_min_digit),Form("c_gradients_%d",idig+grad_min_digit),1);
    c_legend[idig] = new TLegend( 0.5, 0.6, 0.9, 0.9);
    ngraph_digit[idig] = 0;
    line_color[idig] = 1;
  }

  for(int iepoch=0;iepoch<nepoch;iepoch++){
    for(int igrad=0;igrad<ngrad;igrad++){
      g_gradients[igrad]->SetPoint(iepoch,iepoch,gradients[iepoch][igrad]);
    }
  }

  for(int igrad=0;igrad<ngrad;igrad++){
    c_gradients[last_epoch_digits[igrad]-grad_min_digit]->cd();
    string drawOption = ngraph_digit[last_epoch_digits[igrad]-grad_min_digit]==0 ? "" : "same";
    g_gradients[igrad]->SetLineColor(ngraph_digit[last_epoch_digits[igrad]-grad_min_digit] + line_color[last_epoch_digits[igrad]-grad_min_digit]); 
    c_legend[last_epoch_digits[igrad]-grad_min_digit]->AddEntry(g_gradients[igrad], Form("gradient norm of parameter %d",igrad) , "l");
    g_gradients[igrad]->Draw(drawOption.c_str());
    ngraph_digit[last_epoch_digits[igrad]-grad_min_digit]++;
    if(ngraph_digit[last_epoch_digits[igrad]-grad_min_digit]==4 || ngraph_digit[last_epoch_digits[igrad]-grad_min_digit]==8) line_color[last_epoch_digits[igrad]-grad_min_digit]++;
  }

  for(int idig=0;idig<digit_range;idig++){
    c_gradients[idig]->cd();
    c_legend[idig]->Draw("same");
  }
  



  TLegend *legend_LV = new TLegend( 0.4, 0.48, 0.8, 0.78);
  legend_LV->AddEntry( g_LV_att_charged, "attractive +-", "l");
  legend_LV->AddEntry( g_LV_att_neutral, "attractive 0", "l");
  legend_LV->AddEntry( g_LV_rep_charged, "repulsive +-", "l");
  legend_LV->AddEntry( g_LV_rep_neutral, "repulsive 0", "l");
  legend_LV->SetFillColor(0);
  TCanvas *c_l_v = new TCanvas("c_l_v","c_l_v",1);
  c_l_v->cd();
  c_l_v->SetGrid();
  g_LV_rep_neutral->SetMinimum(0); 
  g_LV_rep_neutral->Draw();
  g_LV_att_charged->Draw("same");
  g_LV_att_neutral->Draw("same");
  g_LV_rep_charged->Draw("same");
  legend_LV->Draw("same");


  TLegend *legend_train_LV = new TLegend( 0.4, 0.48, 0.8, 0.78);
  legend_train_LV->AddEntry( g_train_LV_att_charged, "attractive +-", "l");
  legend_train_LV->AddEntry( g_train_LV_att_neutral, "attractive 0", "l");
  legend_train_LV->AddEntry( g_train_LV_rep_charged, "repulsive +-", "l");
  legend_train_LV->AddEntry( g_train_LV_rep_neutral, "repulsive 0", "l");
  legend_train_LV->SetFillColor(0);
  TCanvas *c_l_v_train = new TCanvas("c_l_v_train","c_l_v_train",1);
  c_l_v_train->cd();
  c_l_v_train->SetGrid();
  g_train_LV_rep_neutral->SetMinimum(0); 
  g_train_LV_rep_neutral->Draw();
  g_train_LV_att_charged->Draw("same");
  g_train_LV_att_neutral->Draw("same");
  g_train_LV_rep_charged->Draw("same");
  legend_train_LV->Draw("same");


}
