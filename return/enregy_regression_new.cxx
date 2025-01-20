
using namespace std;


void enregy_regression_new(){ 

  double returning5[50] = {3.3099231719970703, 3.312429904937744, 3.5421433448791504, 3.8623576164245605, 4.180744647979736, 4.468781471252441, 4.7387542724609375, 4.9926958084106445, 2.516710042953491, 2.342529058456421, 2.1439993381500244, 2.1316959857940674, 2.1174652576446533, 2.098949909210205, 2.1026790142059326, 2.0793020725250244, 2.0580270290374756, 2.065371513366699, 2.0202250480651855, 2.0314743518829346, 2.013171911239624, 2.0043935775756836, 2.0212857723236084, 1.9770739078521729, 1.9837214946746826, 1.961472749710083, 1.9709796905517578, 1.9497346878051758, 1.9483168125152588, 1.9371148347854614, 1.9338716268539429, 1.9248974323272705, 1.903326153755188, 1.9091007709503174, 1.9102119207382202, 1.8726075887680054, 1.8844894170761108, 1.8732366561889648, 1.8780766725540161, 1.8632919788360596, 1.8812166452407837, 1.869850516319275, 1.8526830673217773, 1.8591461181640625, 1.8397884368896484, 1.843465805053711, 1.819946050643921, 1.838031530380249, 1.8135267496109009, 1.7918754816055298};
  TGraph *g5 = new TGraph();
  g5->SetLineColor(38); 

  double MSE_loss_alphaMSE[50] = {311.77203369140625, 305.6438293457031, 304.2088623046875, 298.4476318359375, 292.3844299316406, 289.4076843261719, 287.65716552734375, 284.0992736816406, 278.8836364746094, 274.39105224609375, 269.3546447753906, 265.6130065917969, 262.4834289550781, 260.98809814453125, 259.9219665527344, 257.74383544921875, 94.85491180419922, 77.7738265991211, 70.88126373291016, 68.01258087158203, 63.44456100463867, 61.5815315246582, 59.201744079589844, 57.12360763549805, 56.670928955078125, 55.39116668701172, 53.555809020996094, 51.822532653808594, 51.577945709228516, 50.10015106201172, 49.50216293334961, 49.48790740966797, 48.55936813354492, 48.18036651611328, 47.244972229003906, 47.508056640625, 46.61693572998047, 45.728660583496094, 45.177860260009766, 44.90422058105469, 44.417903900146484, 44.05051803588867, 43.7349967956543, 43.12415313720703, 42.75998306274414, 42.80250930786133, 42.326026916503906, 42.45901107788086, 43.4248046875, 43.06492233276367};
  double MSE_LV_alphaMSE[50] = {2.3141, 1.1911, 0.8605, 0.6406, 0.5406, 0.4965, 0.4720, 0.4577, 0.6509, 0.6584, 0.7071, 0.7855, 0.8221, 0.8575, 0.8730, 0.9052, 1.5474, 1.2899, 1.1412, 1.0831, 1.0196, 0.9669, 0.9388, 0.9134, 0.8919, 0.8734, 0.8851, 0.8681, 0.8590, 0.8542, 0.8597, 0.8603, 0.8601, 0.8504, 0.8775, 0.8858, 0.8830, 0.9076, 0.9021, 0.9073, 0.8925, 0.9222, 0.9153, 0.9199, 0.9300, 0.9538, 0.9539, 0.9471, 0.9612, 0.9810};
  double MSE_Lbeta_alphaMSE[50] = {0.9604, 1.1601, 1.3606, 1.5243, 1.6804, 1.8071, 1.9289, 2.0349, 0.8635, 0.7814, 0.6450, 0.4817, 0.4079, 0.3625, 0.3313, 0.3064, 1.0464, 1.0451, 0.9853, 0.9531, 0.9336, 0.9210, 0.9075, 0.8817, 0.8720, 0.8435, 0.8299, 0.8151, 0.7927, 0.7763, 0.7595, 0.7314, 0.7119, 0.6976, 0.6625, 0.6349, 0.6252, 0.6020, 0.5923, 0.5710, 0.5686, 0.5467, 0.5441, 0.5277, 0.5151, 0.5064, 0.4985, 0.4938, 0.4844, 0.4738};
  double MSE_LE_alphaMSE[50] = {307.4976, 302.2926, 300.9878, 295.2827, 289.1634, 286.1041, 284.2563, 280.6067, 276.3692, 271.9513, 267.0026, 263.3458, 260.2534, 258.7681, 257.7177, 255.5323, 91.2611, 74.4388, 67.7548, 64.9764, 60.4914, 58.6937, 56.3555, 54.3285, 53.9070, 52.6743, 50.8408, 49.1393, 48.9263, 47.4697, 46.8830, 46.8962, 45.9874, 45.6324, 44.7050, 44.9874, 44.1087, 43.2191, 42.6835, 42.4259, 41.9568, 41.5817, 41.2756, 40.6766, 40.3149, 40.3424, 39.8736, 40.0181, 40.9792, 40.6101};

  double MSE_loss_betaE[50] = {4.011220932006836, 4.186608791351318, 5.425604343414307, 7.798770427703857, 11.651430130004883, 19.453664779663086, 34.10069274902344, 61.09080123901367, 2.6710753440856934, 2.6020281314849854, 2.575838804244995, 2.576538324356079, 2.5625948905944824, 2.551370620727539, 2.5430526733398438, 2.5452024936676025, 2.471195697784424, 2.470613956451416, 2.4436843395233154, 2.4380879402160645, 2.430426836013794, 2.428960084915161, 2.4266631603240967, 2.390263795852661, 2.404404401779175, 2.40230131149292, 2.3978943824768066, 2.3825652599334717, 2.377206325531006, 2.358394145965576, 2.335132360458374, 2.3486435413360596, 2.332282781600952, 2.344268560409546, 2.3262014389038086, 2.3003036975860596, 2.3108229637145996, 2.297447443008423, 2.2850735187530518, 2.299931287765503, 2.2748067378997803, 2.28478741645813, 2.2666828632354736, 2.277327299118042, 2.2773516178131104, 2.2467362880706787, 2.2486608028411865, 2.2460317611694336, 2.231459140777588};
  double MSE_LV_betaE[50] = {1.4396, 0.7938, 0.6475, 0.5512, 0.5007, 0.4624, 0.4380, 0.4125, 0.7059, 0.8642, 0.8852, 0.9288, 0.9342, 0.9297, 0.9520, 0.9663, 0.9989, 1.0255, 1.0029, 1.0094, 0.9983, 1.0137, 1.0221, 0.9731, 1.0061, 1.0029, 1.0025, 0.9965, 0.9903, 0.9763, 0.9529, 0.9737, 0.9625, 0.9826, 0.9581, 0.9283, 0.9539, 0.9429, 0.9291, 0.9502, 0.9295, 0.9431, 0.9318, 0.9469, 0.9625, 0.9215, 0.9219, 0.9217, 0.9230};
  double MSE_Lbeta_betaE[50] = {1.1761, 1.5265, 1.7868, 2.0349, 2.2651, 2.5489, 2.8177, 3.1195, 0.6267, 0.4099, 0.3695, 0.3315, 0.3140, 0.3091, 0.2824, 0.2735, 0.2701, 0.2582, 0.2606, 0.2521, 0.2577, 0.2450, 0.2374, 0.2506, 0.2351, 0.2369, 0.2344, 0.2269, 0.2296, 0.2251, 0.2275, 0.2217, 0.2190, 0.2132, 0.2198, 0.2230, 0.2116, 0.2092, 0.2109, 0.2069, 0.2048, 0.2001, 0.1954, 0.1913, 0.1780, 0.1889, 0.1900, 0.1890, 0.1741};
  double MSE_LE_betaE[50] = {0.3955, 0.8663, 1.9912, 4.2126, 7.8856, 15.4423, 29.8450, 56.5588, 0.3385, 0.3279, 0.3211, 0.3163, 0.3145, 0.3126, 0.3087, 0.3055, 0.2022, 0.1869, 0.1803, 0.1766, 0.1744, 0.1702, 0.1672, 0.1665, 0.1632, 0.1624, 0.1610, 0.1591, 0.1574, 0.1570, 0.1547, 0.1532, 0.1509, 0.1485, 0.1483, 0.1490, 0.1454, 0.1454, 0.1451, 0.1428, 0.1406, 0.1415, 0.1394, 0.1392, 0.1368, 0.1364, 0.1367, 0.1353, 0.1344};





double MSE_loss_alpha_momentum[50] = {413.70135498046875, 434.57855224609375, 462.095703125, 490.3146667480469, 510.64385986328125, 529.6590576171875, 545.4608764648438, 560.5513916015625, 492.3951416015625, 509.18115234375, 523.6689453125, 531.5469970703125, 540.137939453125, 546.6045532226562, 552.1828002929688, 552.1266479492188, 120.84819793701172, 150.022216796875, 98.06794738769531, 91.48311614990234, 87.58417510986328, 81.9701919555664, 78.71429443359375, 75.71642303466797, 122.81358337402344, 70.30873107910156, 66.38294982910156, 63.3106689453125, 60.978816986083984, 58.974464416503906, 57.525291442871094, 55.65523910522461, 53.10529327392578, 51.92455291748047, 50.92137908935547, 49.25210189819336, 49.03129577636719, 48.382694244384766, 46.33823013305664, 45.0791130065918, 43.9316291809082, 43.421390533447266, 43.505455017089844, 42.51667404174805, 41.52317428588867, 41.555572509765625, 40.91287612915039, 40.50226974487305, 40.00469970703125, 41.64023208618164};
double MSE_LV_alpha_momentum[50] = {4.5483, 2.1834, 1.4958, 1.1434, 0.9949, 0.8858, 0.7958, 0.7334, 0.8639, 0.9070, 0.9915, 1.0657, 1.1274, 1.1380, 1.1701, 1.1561, 2.0622, 2.1392, 1.5523, 1.4909, 1.4436, 1.4140, 1.3559, 1.3361, 1.4269, 1.3346, 1.2846, 1.3184, 1.3089, 1.3375, 1.3173, 1.2749, 1.3732, 1.3963, 1.3764, 1.3436, 1.3725, 1.3699, 1.3606, 1.3734, 1.3585, 1.3886, 1.4729, 1.4118, 1.3712, 1.4116, 1.3868, 1.4307, 1.3725, 1.3262};
double MSE_Lbeta_alpha_momentum[50] = {1.0135, 1.1806, 1.3254, 1.4729, 1.6017, 1.7139, 1.8227, 1.9476, 0.9228, 0.7955, 0.7027, 0.6487, 0.6078, 0.6019, 0.5742, 0.5645, 1.2751, 1.0930, 1.0250, 0.9888, 0.9613, 0.9273, 0.9103, 0.9024, 0.9544, 0.8489, 0.8531, 0.8158, 0.7932, 0.7590, 0.7456, 0.7375, 0.6893, 0.6713, 0.6550, 0.6507, 0.6424, 0.6251, 0.6219, 0.5999, 0.5918, 0.5855, 0.5702, 0.5623, 0.5611, 0.5495, 0.5527, 0.5429, 0.5323, 0.5559};
double MSE_LE_alpha_momentum[50] = {407.1396, 430.2145, 458.2744, 486.6983, 507.0472, 526.0593, 541.8423, 556.8704, 489.6085, 506.4786, 520.9748, 528.8326, 537.4027, 543.8647, 549.4385, 549.4061, 116.5108, 145.7901, 94.4906, 88.0034, 84.1793, 78.6289, 75.4481, 72.4779, 119.4323, 67.1253, 63.2452, 60.1764, 57.8767, 55.8779, 54.4624, 52.6429, 50.0427, 48.8569, 47.8900, 46.2578, 46.0164, 45.3877, 43.3557, 42.1058, 40.9813, 40.4473, 40.4623, 39.5425, 38.5908, 38.5944, 37.9735, 37.5287, 37.0999, 38.7582};



double MSE_loss_alpha_momentum_editloss[50] = {345.60791015625, 333.9302978515625, 322.94598388671875, 316.0450439453125, 310.7818908691406, 312.44183349609375, 312.1851501464844, 312.55377197265625, 338.14459228515625, 344.4854431152344, 345.2275390625, 347.12646484375, 345.7498474121094, 348.21484375, 350.816162109375, 351.51885986328125, 116.74578857421875, 109.52081298828125, 104.3308334350586, 99.71048736572266, 93.33544158935547, 89.21296691894531, 81.19178771972656, 78.71533966064453, 75.84228515625, 72.2516098022461, 67.72355651855469, 64.2964859008789, 65.38297271728516, 64.43379211425781, 64.78995513916016, 61.96194076538086, 62.26935577392578, 60.90884017944336, 58.11948776245117, 58.24091339111328, 57.24394607543945, 56.30430603027344, 57.311058044433594, 55.33735656738281, 53.57692337036133, 52.910091400146484, 54.23872375488281, 53.25480270385742, 51.147682189941406, 53.58781814575195, 52.21281051635742, 50.433013916015625, 49.41612243652344, 50.1043701171875};
double MSE_LV_alpha_momentum_editloss[50] = {4.4490, 2.2352, 1.5780, 1.3334, 1.1991, 1.1014, 1.0475, 1.0046, 1.1827, 1.3463, 1.4763, 1.4667, 1.5288, 1.5102, 1.5226, 1.5225, 2.0731, 1.8968, 1.8630, 1.8070, 1.9840, 1.8900, 1.8593, 1.9185, 1.9703, 1.9263, 1.9637, 1.9253, 1.9393, 2.0073, 1.9209, 1.9381, 1.9552, 1.9726, 1.8916, 1.8662, 1.8985, 1.8685, 1.8581, 1.8211, 1.8401, 1.8373, 1.8539, 1.8306, 1.7725, 1.8163, 1.8651, 1.8078, 1.8510, 1.8219};
double MSE_Lbeta_alpha_momentum_editloss[50] = {0.9807, 1.2134, 1.4140, 1.5740, 1.7085, 1.8384, 1.9545, 2.1087, 1.0052, 0.8324, 0.7788, 0.7641, 0.7461, 0.7327, 0.7213, 0.7161, 0.9769, 0.8902, 0.8458, 0.8297, 0.7710, 0.7721, 0.7610, 0.7536, 0.7363, 0.7354, 0.7187, 0.7055, 0.7053, 0.6959, 0.6988, 0.6862, 0.6806, 0.6700, 0.6713, 0.6745, 0.6614, 0.6631, 0.6585, 0.6560, 0.6509, 0.6460, 0.6420, 0.6371, 0.6428, 0.6434, 0.6276, 0.6301, 0.6245, 0.6226};
double MSE_LE_alpha_momentum_editloss[50] = {339.1782, 329.4817, 318.9539, 312.1376, 306.8743, 308.5021, 308.1832, 308.4406, 334.9566, 341.3067, 341.9725, 343.8956, 342.4750, 344.9719, 347.5722, 348.2802, 112.6958, 105.7338, 100.6220, 96.0738, 89.5804, 85.5508, 77.5716, 75.0433, 72.1357, 68.5898, 64.0411, 60.6657, 61.7384, 60.7306, 61.1703, 58.3377, 58.6336, 57.2662, 54.5567, 54.7002, 53.6840, 52.7727, 53.7944, 51.8602, 50.0858, 49.4267, 50.7429, 49.7872, 47.7324, 50.1281, 48.7201, 46.9951, 45.9406, 46.6599};





  TGraph *g_MSE_loss_betaE = new TGraph();
  TGraph *g_MSE_LV_betaE = new TGraph();
  TGraph *g_MSE_Lbeta_betaE = new TGraph();
  TGraph *g_MSE_LE_betaE = new TGraph();
  g_MSE_loss_betaE->GetXaxis()->SetTitle("epoch"); 
  g_MSE_loss_betaE->GetYaxis()->SetTitle("loss"); 
  g_MSE_loss_betaE->SetTitle("beta * E"); 
  g_MSE_loss_betaE->SetMinimum(0); 
  g_MSE_loss_betaE->SetMaximum(6); 
  g_MSE_loss_betaE->SetLineColor(1); 
  g_MSE_LV_betaE->SetLineColor(2); 
  g_MSE_Lbeta_betaE->SetLineColor(3); 
  g_MSE_LE_betaE->SetLineColor(4); 

  TLegend *legend_betaE = new TLegend( 0.4, 0.48, 0.8, 0.78);
  legend_betaE->AddEntry( g_MSE_loss_betaE, "total loss", "l"); // AddEntry( pointer , "interpretation" , "option" )
  legend_betaE->AddEntry( g_MSE_LV_betaE, "L_V", "l"); // option は　"f"=box, "l"="L"=line, "p"=marker
  legend_betaE->AddEntry( g_MSE_Lbeta_betaE, "L_beta", "l") ;
  legend_betaE->AddEntry( g_MSE_LE_betaE, "L_E", "l") ;
  legend_betaE->AddEntry( g5, "total loss w/o L_E", "l") ;
  legend_betaE->SetFillColor(0);




  TGraph *g_MSE_loss_alphaMSE = new TGraph();
  TGraph *g_MSE_LV_alphaMSE = new TGraph();
  TGraph *g_MSE_Lbeta_alphaMSE = new TGraph();
  TGraph *g_MSE_LE_alphaMSE = new TGraph();
  g_MSE_loss_alphaMSE->GetXaxis()->SetTitle("epoch"); 
  g_MSE_loss_alphaMSE->GetYaxis()->SetTitle("loss"); 
  g_MSE_loss_alphaMSE->SetTitle("alpha MSE"); 
  g_MSE_loss_alphaMSE->SetMinimum(0); 
  // g_MSE_loss_alphaMSE->SetMaximum(6); 
  g_MSE_loss_alphaMSE->SetLineColor(1); 
  g_MSE_LV_alphaMSE->SetLineColor(2); 
  g_MSE_Lbeta_alphaMSE->SetLineColor(3); 
  g_MSE_LE_alphaMSE->SetLineColor(4); 

  TLegend *legend_alphaMSE = new TLegend( 0.4, 0.48, 0.8, 0.78);
  legend_alphaMSE->AddEntry( g_MSE_loss_alphaMSE, "total loss", "l"); // AddEntry( pointer , "interpretation" , "option" )
  legend_alphaMSE->AddEntry( g_MSE_LV_alphaMSE, "L_V", "l"); // option は　"f"=box, "l"="L"=line, "p"=marker
  legend_alphaMSE->AddEntry( g_MSE_Lbeta_alphaMSE, "L_beta", "l") ;
  legend_alphaMSE->AddEntry( g_MSE_LE_alphaMSE, "L_E", "l") ;
  legend_alphaMSE->AddEntry( g5, "total loss w/o L_E", "l") ;
  legend_alphaMSE->SetFillColor(0);




  TGraph *g_MSE_loss_alpha_momentum = new TGraph();
  TGraph *g_MSE_LV_alpha_momentum = new TGraph();
  TGraph *g_MSE_Lbeta_alpha_momentum = new TGraph();
  TGraph *g_MSE_LE_alpha_momentum = new TGraph();
  g_MSE_loss_alpha_momentum->GetXaxis()->SetTitle("epoch"); 
  g_MSE_loss_alpha_momentum->GetYaxis()->SetTitle("loss"); 
  g_MSE_loss_alpha_momentum->SetTitle("alpha momentum"); 
  g_MSE_loss_alpha_momentum->SetMinimum(0); 
  g_MSE_loss_alpha_momentum->SetMaximum(6); 
  g_MSE_loss_alpha_momentum->SetLineColor(1); 
  g_MSE_LV_alpha_momentum->SetLineColor(2); 
  g_MSE_Lbeta_alpha_momentum->SetLineColor(3); 
  g_MSE_LE_alpha_momentum->SetLineColor(4); 

  TLegend *legend_alpha_momentum = new TLegend( 0.4, 0.48, 0.8, 0.78);
  legend_alpha_momentum->AddEntry( g_MSE_loss_alpha_momentum, "total loss", "l"); // AddEntry( pointer , "interpretation" , "option" )
  legend_alpha_momentum->AddEntry( g_MSE_LV_alpha_momentum, "L_V", "l"); // option は　"f"=box, "l"="L"=line, "p"=marker
  legend_alpha_momentum->AddEntry( g_MSE_Lbeta_alpha_momentum, "L_beta", "l") ;
  legend_alpha_momentum->AddEntry( g_MSE_LE_alpha_momentum, "L_E", "l") ;
  legend_alpha_momentum->AddEntry( g5, "total loss w/o L_E", "l") ;
  legend_alpha_momentum->SetFillColor(0);



  TGraph *g_MSE_loss_alpha_momentum_editloss = new TGraph();
  TGraph *g_MSE_LV_alpha_momentum_editloss = new TGraph();
  TGraph *g_MSE_Lbeta_alpha_momentum_editloss = new TGraph();
  TGraph *g_MSE_LE_alpha_momentum_editloss = new TGraph();
  g_MSE_loss_alpha_momentum_editloss->GetXaxis()->SetTitle("epoch"); 
  g_MSE_loss_alpha_momentum_editloss->GetYaxis()->SetTitle("loss"); 
  g_MSE_loss_alpha_momentum_editloss->SetTitle("alpha momentum"); 
  g_MSE_loss_alpha_momentum_editloss->SetMinimum(0); 
  // g_MSE_loss_alpha_momentum_editloss->SetMaximum(6); 
  g_MSE_loss_alpha_momentum_editloss->SetLineColor(1); 
  g_MSE_LV_alpha_momentum_editloss->SetLineColor(2); 
  g_MSE_Lbeta_alpha_momentum_editloss->SetLineColor(3); 
  g_MSE_LE_alpha_momentum_editloss->SetLineColor(4); 

  TLegend *legend_alpha_momentum_editloss = new TLegend( 0.4, 0.48, 0.8, 0.78);
  legend_alpha_momentum_editloss->AddEntry( g_MSE_loss_alpha_momentum_editloss, "total loss", "l"); // AddEntry( pointer , "interpretation" , "option" )
  legend_alpha_momentum_editloss->AddEntry( g_MSE_LV_alpha_momentum_editloss, "L_V", "l"); // option は　"f"=box, "l"="L"=line, "p"=marker
  legend_alpha_momentum_editloss->AddEntry( g_MSE_Lbeta_alpha_momentum_editloss, "L_beta", "l") ;
  legend_alpha_momentum_editloss->AddEntry( g_MSE_LE_alpha_momentum_editloss, "L_E", "l") ;
  legend_alpha_momentum_editloss->AddEntry( g5, "total loss w/o L_E", "l") ;
  legend_alpha_momentum_editloss->SetFillColor(0);


  for(int i=0;i<50;i++){
    g5->SetPoint(i,i,returning5[i]);

    g_MSE_loss_betaE->SetPoint(i,i,MSE_loss_betaE[i]);
    g_MSE_LV_betaE->SetPoint(i,i,MSE_LV_betaE[i]);
    g_MSE_Lbeta_betaE->SetPoint(i,i,MSE_Lbeta_betaE[i]);
    g_MSE_LE_betaE->SetPoint(i,i,MSE_LE_betaE[i]);

    g_MSE_loss_alphaMSE->SetPoint(i,i,MSE_loss_alphaMSE[i]);
    g_MSE_LV_alphaMSE->SetPoint(i,i,MSE_LV_alphaMSE[i]);
    g_MSE_Lbeta_alphaMSE->SetPoint(i,i,MSE_Lbeta_alphaMSE[i]);
    g_MSE_LE_alphaMSE->SetPoint(i,i,MSE_LE_alphaMSE[i]);

    g_MSE_loss_alpha_momentum->SetPoint(i,i,MSE_loss_alpha_momentum[i]);
    g_MSE_LV_alpha_momentum->SetPoint(i,i,MSE_LV_alpha_momentum[i]);
    g_MSE_Lbeta_alpha_momentum->SetPoint(i,i,MSE_Lbeta_alpha_momentum[i]);
    g_MSE_LE_alpha_momentum->SetPoint(i,i,MSE_LE_alpha_momentum[i]);

    g_MSE_loss_alpha_momentum_editloss->SetPoint(i,i,MSE_loss_alpha_momentum_editloss[i]);
    g_MSE_LV_alpha_momentum_editloss->SetPoint(i,i,MSE_LV_alpha_momentum_editloss[i]);
    g_MSE_Lbeta_alpha_momentum_editloss->SetPoint(i,i,MSE_Lbeta_alpha_momentum_editloss[i]);
    g_MSE_LE_alpha_momentum_editloss->SetPoint(i,i,MSE_LE_alpha_momentum_editloss[i]);
  }


  TCanvas *c2 = new TCanvas("c2","c2",1);
  c2->cd();
  g_MSE_loss_betaE->Draw();
  g5->Draw("same");
  g_MSE_LV_betaE->Draw("same");
  g_MSE_Lbeta_betaE->Draw("same");
  g_MSE_LE_betaE->Draw("same");
  legend_betaE->Draw();


  TCanvas *c4 = new TCanvas("c4","c4",1);
  c4->cd();
  g_MSE_loss_alphaMSE->Draw();
  g5->Draw("same");
  g_MSE_LV_alphaMSE->Draw("same");
  g_MSE_Lbeta_alphaMSE->Draw("same");
  g_MSE_LE_alphaMSE->Draw("same");
  legend_alphaMSE->Draw();

  
  TCanvas *c5 = new TCanvas("c5","c5",1);
  c5->cd();
  g_MSE_loss_alpha_momentum->Draw();
  g5->Draw("same");
  g_MSE_LV_alpha_momentum->Draw("same");
  g_MSE_Lbeta_alpha_momentum->Draw("same");
  g_MSE_LE_alpha_momentum->Draw("same");
  legend_alpha_momentum->Draw();


  TCanvas *c6 = new TCanvas("c6","c6",1);
  c6->cd();
  g_MSE_loss_alpha_momentum_editloss->Draw();
  g5->Draw("same");
  g_MSE_LV_alpha_momentum_editloss->Draw("same");
  g_MSE_Lbeta_alpha_momentum_editloss->Draw("same");
  g_MSE_LE_alpha_momentum_editloss->Draw("same");
  legend_alpha_momentum_editloss->Draw();


}