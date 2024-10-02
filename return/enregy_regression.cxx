
using namespace std;


void enregy_regression(){ 

  double returning5[50] = {3.3099231719970703, 3.312429904937744, 3.5421433448791504, 3.8623576164245605, 4.180744647979736, 4.468781471252441, 4.7387542724609375, 4.9926958084106445, 2.516710042953491, 2.342529058456421, 2.1439993381500244, 2.1316959857940674, 2.1174652576446533, 2.098949909210205, 2.1026790142059326, 2.0793020725250244, 2.0580270290374756, 2.065371513366699, 2.0202250480651855, 2.0314743518829346, 2.013171911239624, 2.0043935775756836, 2.0212857723236084, 1.9770739078521729, 1.9837214946746826, 1.961472749710083, 1.9709796905517578, 1.9497346878051758, 1.9483168125152588, 1.9371148347854614, 1.9338716268539429, 1.9248974323272705, 1.903326153755188, 1.9091007709503174, 1.9102119207382202, 1.8726075887680054, 1.8844894170761108, 1.8732366561889648, 1.8780766725540161, 1.8632919788360596, 1.8812166452407837, 1.869850516319275, 1.8526830673217773, 1.8591461181640625, 1.8397884368896484, 1.843465805053711, 1.819946050643921, 1.838031530380249, 1.8135267496109009, 1.7918754816055298};
  TGraph *g5 = new TGraph();
  g5->SetLineColor(38); 

  double MSE_loss[50] = {3.93650484085083, 3.4546549320220947, 3.513397455215454, 3.6601505279541016, 3.8369641304016113, 4.054523468017578, 4.280764102935791, 4.5416131019592285, 2.3826541900634766, 2.430546760559082, 2.4255387783050537, 2.3959157466888428, 2.386561870574951, 2.366328716278076, 2.3609421253204346, 2.351844310760498, 2.337934970855713, 2.3169851303100586, 2.3086180686950684, 2.2888364791870117, 2.2679967880249023, 2.2928998470306396, 2.2801952362060547, 2.282601833343506, 2.2504289150238037, 2.2501285076141357, 2.244048595428467, 2.2247838973999023, 2.240537405014038, 2.21301531791687, 2.2257397174835205, 2.2118396759033203, 2.1968460083007812, 2.1896541118621826, 2.194591522216797, 2.1845014095306396, 2.189572334289551, 2.1971263885498047, 2.1760900020599365, 2.168835401535034, 2.1699976921081543, 2.1706628799438477, 2.1433112621307373, 2.1533455848693848, 2.1414003372192383, 2.1507222652435303, 2.1401383876800537, 2.1286749839782715, 2.137079954147339, 2.1213903427124023};
  double MSE_LV[50] = {1.5204, 0.7677, 0.6121, 0.5398, 0.5073, 0.4850, 0.4665, 0.4435, 0.8657, 1.0232, 1.0451, 1.0401, 1.0441, 1.0450, 1.0632, 1.0506, 1.0538, 1.0365, 1.0327, 1.0201, 1.0030, 1.0504, 1.0363, 1.0384, 1.0083, 1.0085, 1.0025, 0.9933, 1.0142, 0.9918, 1.0163, 0.9950, 0.9804, 0.9784, 0.9797, 0.9748, 0.9881, 1.0004, 0.9745, 0.9660, 0.9803, 0.9744, 0.9469, 0.9576, 0.9549, 0.9619, 0.9586, 0.9463, 0.9527, 0.9418};
  double MSE_Lbeta[50] = {1.4161, 1.6870, 1.9013, 2.1203, 2.3296, 2.5695, 2.8142, 3.0981, 0.5170, 0.4073, 0.3805, 0.3558, 0.3424, 0.3213, 0.2978, 0.3012, 0.2842, 0.2805, 0.2760, 0.2688, 0.2650, 0.2425, 0.2439, 0.2442, 0.2421, 0.2416, 0.2416, 0.2315, 0.2264, 0.2212, 0.2095, 0.2168, 0.2165, 0.2112, 0.2149, 0.2097, 0.2015, 0.1967, 0.2016, 0.2029, 0.1897, 0.1963, 0.1964, 0.1958, 0.1865, 0.1888, 0.1815, 0.1823, 0.1843, 0.1796};
  double MSE_LE[50] = {0.5123, 0.4790, 0.4612, 0.4490, 0.4400, 0.4317, 0.4227, 0.4149, 0.4208, 0.4192, 0.4164, 0.4148, 0.4118, 0.4067, 0.4015, 0.3972, 0.3922, 0.3876, 0.3835, 0.3789, 0.3767, 0.3675, 0.3632, 0.3598, 0.3579, 0.3496, 0.3486, 0.3444, 0.3397, 0.3349, 0.3331, 0.3306, 0.3248, 0.3237, 0.3190, 0.3180, 0.3119, 0.3096, 0.3081, 0.3068, 0.3036, 0.3000, 0.3006, 0.2961, 0.2945, 0.2929, 0.2897, 0.2884, 0.2851, 0.2841};

  double MSE_loss_betaMSE[50] = {3.839707136154175, 3.403385639190674, 3.573150396347046, 3.8325700759887695, 4.142667293548584, 4.496973037719727, 4.862890243530273, 5.24053430557251, 2.5415847301483154, 2.352144956588745, 2.3344950675964355, 2.3344154357910156, 2.323958396911621, 2.309264898300171, 2.307952642440796, 2.2884321212768555, 2.2849721908569336, 2.2727835178375244, 2.248910427093506, 2.227130651473999, 2.2323179244995117, 2.2116892337799072, 2.208051919937134, 2.1983866691589355, 2.1931824684143066, 2.1841981410980225, 2.1797304153442383, 2.1701653003692627, 2.1717443466186523, 2.168219804763794, 2.1698834896087646, 2.1482748985290527, 2.147284746170044, 2.1594600677490234, 2.1396079063415527, 2.1167407035827637, 2.1143953800201416, 2.113311529159546, 2.117901563644409, 2.1168277263641357, 2.0982284545898438, 2.1017067432403564, 2.1059906482696533, 2.10835337638855, 2.09149169921875, 2.083705425262451, 2.064852714538574, 2.0809059143066406, 2.0792579650878906, 2.0533905029296875};
  double MSE_LV_betaMSE[50] = {1.4591, 0.7603, 0.6013, 0.5330, 0.4765, 0.4384, 0.4225, 0.3951, 0.6461, 0.8299, 0.8861, 0.9232, 0.9381, 0.9427, 0.9557, 0.9484, 0.9550, 0.9543, 0.9408, 0.9308, 0.9403, 0.9190, 0.9292, 0.9299, 0.9280, 0.9104, 0.9131, 0.9008, 0.9141, 0.9238, 0.9166, 0.8959, 0.8979, 0.9129, 0.8997, 0.8742, 0.8713, 0.8743, 0.8890, 0.8881, 0.8625, 0.8790, 0.8818, 0.8888, 0.8646, 0.8609, 0.8442, 0.8621, 0.8723, 0.8259};
  double MSE_Lbeta_betaMSE[50] = {1.1526, 1.6159, 1.9651, 2.2976, 2.6655, 3.0584, 3.4403, 3.8454, 0.8453, 0.4706, 0.4014, 0.3681, 0.3449, 0.3265, 0.3141, 0.3020, 0.2935, 0.2816, 0.2725, 0.2607, 0.2572, 0.2590, 0.2455, 0.2336, 0.2302, 0.2431, 0.2355, 0.2395, 0.2260, 0.2125, 0.2230, 0.2222, 0.2200, 0.2160, 0.2104, 0.2118, 0.2129, 0.2091, 0.1987, 0.1988, 0.2072, 0.1934, 0.1947, 0.1902, 0.1982, 0.1929, 0.1907, 0.1903, 0.1768, 0.2006};
  double MSE_LE_betaMSE[50] = {0.2280, 0.0273, 0.0067, 0.0020, 0.0006, 0.0002, 0.0001, 0.0000, 0.0502, 0.0516, 0.0470, 0.0432, 0.0410, 0.0401, 0.0382, 0.0380, 0.0365, 0.0369, 0.0356, 0.0357, 0.0348, 0.0337, 0.0333, 0.0349, 0.0349, 0.0307, 0.0311, 0.0299, 0.0316, 0.0319, 0.0302, 0.0302, 0.0294, 0.0306, 0.0295, 0.0307, 0.0302, 0.0300, 0.0302, 0.0299, 0.0285, 0.0293, 0.0294, 0.0294, 0.0288, 0.0298, 0.0300, 0.0285, 0.0301, 0.0269};


  TGraph *g_MSE_loss = new TGraph();
  TGraph *g_MSE_LV = new TGraph();
  TGraph *g_MSE_Lbeta = new TGraph();
  TGraph *g_MSE_LE = new TGraph();
  g_MSE_loss->GetXaxis()->SetTitle("epoch"); 
  g_MSE_loss->GetYaxis()->SetTitle("loss"); 
  g_MSE_loss->SetMinimum(0); 
  g_MSE_loss->SetMaximum(6); 
  g_MSE_loss->SetLineColor(1); 
  g_MSE_LV->SetLineColor(2); 
  g_MSE_Lbeta->SetLineColor(3); 
  g_MSE_LE->SetLineColor(4); 
  
  TLegend *legend = new TLegend( 0.4, 0.48, 0.8, 0.78);
  legend->AddEntry( g_MSE_loss, "total loss", "l"); // AddEntry( pointer , "interpretation" , "option" )
  legend->AddEntry( g_MSE_LV, "L_V", "l"); // option は　"f"=box, "l"="L"=line, "p"=marker
  legend->AddEntry( g_MSE_Lbeta, "L_beta", "l") ;
  legend->AddEntry( g_MSE_LE, "L_E", "l") ;
  legend->AddEntry( g5, "total loss w/o L_E", "l") ;
  legend->SetFillColor(0);



  TGraph *g_MSE_loss_betaMSE = new TGraph();
  TGraph *g_MSE_LV_betaMSE = new TGraph();
  TGraph *g_MSE_Lbeta_betaMSE = new TGraph();
  TGraph *g_MSE_LE_betaMSE = new TGraph();
  g_MSE_loss_betaMSE->GetXaxis()->SetTitle("epoch"); 
  g_MSE_loss_betaMSE->GetYaxis()->SetTitle("loss"); 
  g_MSE_loss_betaMSE->SetMinimum(0); 
  g_MSE_loss_betaMSE->SetMaximum(6); 
  g_MSE_loss_betaMSE->SetLineColor(1); 
  g_MSE_LV_betaMSE->SetLineColor(2); 
  g_MSE_Lbeta_betaMSE->SetLineColor(3); 
  g_MSE_LE_betaMSE->SetLineColor(4); 

  TLegend *legend_betaMSE = new TLegend( 0.4, 0.48, 0.8, 0.78);
  legend_betaMSE->AddEntry( g_MSE_loss_betaMSE, "total loss", "l"); // AddEntry( pointer , "interpretation" , "option" )
  legend_betaMSE->AddEntry( g_MSE_LV_betaMSE, "L_V", "l"); // option は　"f"=box, "l"="L"=line, "p"=marker
  legend_betaMSE->AddEntry( g_MSE_Lbeta_betaMSE, "L_beta", "l") ;
  legend_betaMSE->AddEntry( g_MSE_LE_betaMSE, "L_E", "l") ;
  legend_betaMSE->AddEntry( g5, "total loss w/o L_E", "l") ;
  legend_betaMSE->SetFillColor(0);


  for(int i=0;i<50;i++){
    g5->SetPoint(i,i,returning5[i]);

    g_MSE_loss->SetPoint(i,i,MSE_loss[i]);
    g_MSE_LV->SetPoint(i,i,MSE_LV[i]);
    g_MSE_Lbeta->SetPoint(i,i,MSE_Lbeta[i]);
    g_MSE_LE->SetPoint(i,i,MSE_LE[i]);

    g_MSE_loss_betaMSE->SetPoint(i,i,MSE_loss_betaMSE[i]);
    g_MSE_LV_betaMSE->SetPoint(i,i,MSE_LV_betaMSE[i]);
    g_MSE_Lbeta_betaMSE->SetPoint(i,i,MSE_Lbeta_betaMSE[i]);
    g_MSE_LE_betaMSE->SetPoint(i,i,MSE_LE_betaMSE[i]);
  }

  g_MSE_loss->Draw();
  g5->Draw("same");
  g_MSE_LV->Draw("same");
  g_MSE_Lbeta->Draw("same");
  g_MSE_LE->Draw("same");
  legend->Draw();


  TCanvas *c2 = new TCanvas("c2","c2",1);
  c2->cd();
  g_MSE_loss_betaMSE->Draw();
  g5->Draw("same");
  g_MSE_LV_betaMSE->Draw("same");
  g_MSE_Lbeta_betaMSE->Draw("same");
  g_MSE_LE_betaMSE->Draw("same");
  legend_betaMSE->Draw();


}