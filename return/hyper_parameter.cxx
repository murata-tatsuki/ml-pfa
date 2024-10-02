/*
const int nepoch = 20;
const int nrun = 11;
int irun=0;
vector<double> lrwd[nrun];
vector<double> returning[nrun];


D9のほうが時間が前の方
uds, mixはudsもほうが時間が前の方

D9

3.2475175857543945
3.284655809402466
3.4999561309814453
3.772411346435547
4.096988677978516
4.405715465545654
4.69852352142334


D17

3.286836624145508
3.2998013496398926
3.520970106124878
3.7784066200256348
4.059854984283447
4.355417251586914

*/

using namespace std;


void hyper_parameter(){ 
  const bool mix_jets = true;

  double returning3[50] = {3.8006234169006348, 3.6424477100372314, 3.8571619987487793, 4.207303047180176, 4.679628849029541, 5.109182357788086, 5.3312273025512695, 5.60181999206543, 2.497852325439453, 2.374774694442749, 2.374459981918335, 2.3676342964172363, 2.344287157058716, 2.331014633178711, 2.316375255584717, 2.3220632076263428, 2.292168140411377, 2.26031231880188, 2.2995898723602295, 2.2591519355773926, 2.2716269493103027, 2.273881196975708, 2.2662038803100586, 2.2626888751983643, 2.224613666534424, 2.22839617729187, 2.2046313285827637, 2.2242209911346436, 2.231212615966797, 2.2551159858703613, 2.197127103805542, 2.228247880935669, 2.21693754196167, 2.2319886684417725, 2.1820590496063232, 2.193791627883911, 2.1785974502563477, 2.1736090183258057, 2.1665871143341064, 2.1662561893463135, 2.185634136199951, 2.162515163421631, 2.1726577281951904, 2.1511149406433105, 2.1452693939208984, 2.167165994644165, 2.133772373199463, 2.1498427391052246, 2.135873556137085, 2.157715320587158};
  double returning4[50] = {3.4548983573913574, 3.394765853881836, 3.6248679161071777, 3.958784580230713, 4.332741737365723, 4.687068939208984, 5.056465148925781, 5.320186614990234, 2.5107297897338867, 2.208482503890991, 2.1227431297302246, 2.1051218509674072, 2.109790325164795, 2.0810422897338867, 2.0924124717712402, 2.104797840118408, 2.0577495098114014, 2.069084882736206, 2.028956890106201, 2.0415496826171875, 2.0244193077087402, 2.045563220977783, 2.023719072341919, 2.0134358406066895, 2.001085042953491, 1.9894592761993408, 2.0055713653564453, 1.979004979133606, 1.977959394454956, 1.9677495956420898, 1.9566656351089478, 1.9451497793197632, 1.9599920511245728, 1.9549139738082886, 1.9465242624282837, 1.924999475479126, 1.9252493381500244, 1.928422451019287, 1.9213244915008545, 1.9194576740264893, 1.902466058731079, 1.913806438446045, 1.9002915620803833, 1.8989492654800415, 1.8894038200378418, 1.8691178560256958, 1.875684142112732, 1.8469626903533936, 1.8605403900146484, 1.839210867881775};
  double returning5[50] = {3.3099231719970703, 3.312429904937744, 3.5421433448791504, 3.8623576164245605, 4.180744647979736, 4.468781471252441, 4.7387542724609375, 4.9926958084106445, 2.516710042953491, 2.342529058456421, 2.1439993381500244, 2.1316959857940674, 2.1174652576446533, 2.098949909210205, 2.1026790142059326, 2.0793020725250244, 2.0580270290374756, 2.065371513366699, 2.0202250480651855, 2.0314743518829346, 2.013171911239624, 2.0043935775756836, 2.0212857723236084, 1.9770739078521729, 1.9837214946746826, 1.961472749710083, 1.9709796905517578, 1.9497346878051758, 1.9483168125152588, 1.9371148347854614, 1.9338716268539429, 1.9248974323272705, 1.903326153755188, 1.9091007709503174, 1.9102119207382202, 1.8726075887680054, 1.8844894170761108, 1.8732366561889648, 1.8780766725540161, 1.8632919788360596, 1.8812166452407837, 1.869850516319275, 1.8526830673217773, 1.8591461181640625, 1.8397884368896484, 1.843465805053711, 1.819946050643921, 1.838031530380249, 1.8135267496109009, 1.7918754816055298};
  double returning9[50] = {3.2475175857543945, 3.284655809402466, 3.4999561309814453, 3.772411346435547, 4.096988677978516, 4.405715465545654, 4.69852352142334, 4.992194652557373, 2.470101833343506, 2.2702078819274902, 2.2027320861816406, 2.181360960006714, 2.1790003776550293, 2.141350507736206, 2.1362476348876953, 2.1110692024230957, 2.0912461280822754, 2.0675597190856934, 2.0693154335021973, 2.0859696865081787, 2.0455024242401123, 2.0388331413269043, 2.0177531242370605, 2.022265911102295, 2.039381265640259, 2.0109076499938965, 2.007634401321411, 2.0012526512145996, 1.9915680885314941, 2.018231153488159, 1.9815261363983154, 1.9695795774459839};
  returning9[36] = 1.8449394702911377;
  returning9[37] = 1.8210935592651367;
  returning9[38] = 1.813805341720581;
  returning9[39] = 1.7980328798294067;
  returning9[40] = 1.955081820487976;
  returning9[41] = 1.934873342514038;
  returning9[42] = 1.9284491539001465;
  returning9[43] = 1.937288522720337;
  returning9[44] = 1.9279670715332031;
  returning9[45] = 1.9228771924972534;
  returning9[46] = 1.9108747243881226;
  returning9[47] = 1.8980752229690552;
  returning9[48] = 1.9024237394332886;
  returning9[49] = 1.8989543914794922;
  double returning17[50] = {3.286836624145508, 3.2998013496398926, 3.520970106124878, 3.7784066200256348, 4.059854984283447, 4.355417251586914, 4.63107442855835, 4.88042688369751, 2.5089855194091797, 2.4037094116210938, 2.3318841457366943, 2.216315984725952, 2.1146492958068848, 2.065817356109619, 2.0299012660980225, 2.004786729812622, 1.9949252605438232, 1.990755558013916, 1.9872071743011475, 1.973129391670227, 1.971125602722168, 1.9726868867874146, 1.9386122226715088, 1.9320454597473145, 1.9227877855300903, 1.9354074001312256, 1.9073216915130615, 1.8952640295028687, 1.8846580982208252, 1.8879928588867188, 1.8723344802856445, 1.866936445236206, 1.8595986366271973, 1.8437252044677734, 1.8468351364135742, 1.8313947916030884, 1.8431113958358765, 1.8220854997634888, 1.8131614923477173, 1.7980406284332275, 1.8079819679260254, 1.8085283041000366, 1.8021482229232788, 1.781205654144287, 1.7836928367614746, 1.7713452577590942, 1.7797044515609741, 1.7726491689682007, 1.7671897411346436, 1.7539931535720825};
  double returning9_20240704110317[50] = {3.292630672454834, 3.3278512954711914, 3.5467755794525146, 3.8615078926086426, 4.166678428649902, 4.472689151763916, 4.791111946105957, 5.091063499450684, 2.5935049057006836, 2.40419864654541, 2.262566328048706, 2.1166279315948486, 2.0918831825256348, 2.086486577987671, 2.0967657566070557, 2.075841188430786, 2.0725436210632324, 2.063311815261841, 2.063300132751465, 2.0575673580169678, 2.0411107540130615, 2.0372543334960938, 2.026972770690918, 2.019362688064575, 2.0045647621154785, 1.9963769912719727, 1.9971133470535278, 1.9788464307785034, 1.9852005243301392, 1.9709984064102173, 1.9567636251449585, 1.9532614946365356, 1.9530757665634155, 1.9531139135360718, 1.9397668838500977, 1.9343405961990356, 1.9445120096206665, 1.9276790618896484, 1.9189536571502686, 1.9147412776947021, 1.9113320112228394, 1.9076613187789917, 1.9270902872085571, 1.9027948379516602, 1.897904396057129, 1.879157543182373, 1.8769949674606323, 1.884661316871643, 1.860723614692688, 1.8619627952575684};
  double returning_jet5[50] = {4.061908721923828, 4.1292290687561035, 4.4246826171875, 4.766791343688965, 5.068028450012207, 5.278223037719727, 5.426862716674805, 5.564116954803467, 3.1948719024658203, 2.992388963699341, 3.0633811950683594, 3.0956945419311523, 3.113849639892578, 3.085331916809082, 3.08756685256958, 3.0864906311035156, 3.0857560634613037, 3.058253765106201, 3.07224178314209, 2.997197151184082, 2.996516227722168, 2.9837639331817627, 2.9876596927642822, 2.9660260677337646, 2.9605603218078613, 2.988619804382324, 2.919567823410034, 2.867964029312134, 2.862112522125244, 2.8603737354278564, 2.8972811698913574, 2.8751752376556396, 2.8426952362060547, 2.8311941623687744, 2.8296892642974854, 2.8395984172821045, 2.787858486175537, 2.826279878616333, 2.8218870162963867, 2.788161516189575, 2.7956225872039795, 2.7637929916381836, 2.788208246231079, 2.7897355556488037, 2.736499786376953, 2.7529141902923584, 2.740583896636963, 2.749462842941284, 2.7646241188049316, 2.74556040763855};
  double returning_mix5[50] = {3.829582691192627, 4.1187920570373535, 4.626526355743408, 5.01945686340332, 5.394181251525879, 5.691390037536621, 5.8505425453186035, 6.020639896392822, 2.744948387145996, 2.616532325744629, 2.639695167541504, 2.617037057876587, 2.5952982902526855, 2.5741076469421387, 2.578068971633911, 2.552891492843628, 2.540018320083618, 2.50384783744812, 2.483628749847412, 2.500298023223877, 2.470770835876465, 2.4693117141723633, 2.457653522491455, 2.426398277282715, 2.4412689208984375};
  double tauD5_lr2e_5[50] = {3.3085713386535645, 3.2289977073669434, 3.4018187522888184, 3.551161766052246, 3.7608909606933594, 4.006103515625, 4.327524185180664, 4.654053688049316, 2.1738064289093018, 2.1302897930145264, 2.1174535751342773, 2.1073217391967773, 2.1050562858581543, 2.0800747871398926, 2.0711870193481445, 2.055189847946167, 2.0424656867980957, 2.0382332801818848, 2.029289484024048, 1.9894274473190308, 1.9870151281356812, 1.9446771144866943, 1.9305665493011475, 1.938077688217163, 1.9051826000213623, 1.9120104312896729, 1.8691891431808472, 1.8765095472335815, 1.8704496622085571, 1.8640859127044678, 1.8545081615447998, 1.8544126749038696, 1.8380076885223389, 1.8368947505950928, 1.8407975435256958, 1.8227124214172363, 1.822160243988037, 1.8125379085540771, 1.8232576847076416, 1.8003357648849487, 1.7912955284118652, 1.788845181465149, 1.7856184244155884, 1.7966078519821167, 1.795357346534729, 1.7734652757644653, 1.7632105350494385, 1.7703816890716553, 1.7646740674972534, 1.7496743202209473};
  double tauD5_lr5e_5[50] = {3.255052328109741, 3.4143972396850586, 3.5303258895874023, 3.669780969619751, 3.8393349647521973, 3.9939022064208984, 4.189491271972656, 4.416443347930908, 2.0531692504882812, 2.0368285179138184, 1.9936351776123047, 1.967560052871704, 1.9408468008041382, 1.9041889905929565, 1.9000389575958252, 1.8437634706497192, 1.840084195137024, 1.8159935474395752, 1.8077008724212646, 1.8032422065734863, 1.761430263519287, 1.7483054399490356, 1.6921131610870361, 1.7111878395080566, 1.6887753009796143, 1.6952520608901978, 1.6969316005706787, 1.706167221069336, 1.6793177127838135, 1.6637243032455444, 1.657518744468689, 1.679374098777771, 1.6409564018249512, 1.6846115589141846, 1.5981858968734741, 1.6130203008651733, 1.621119737625122, 1.6091477870941162, 1.5843932628631592, 1.5889427661895752, 1.5791313648223877, 1.5877870321273804, 1.5629078149795532, 1.5757277011871338, 1.547433853149414, 1.5485578775405884};
  
  TGraph *g3 = new TGraph();
  TGraph *g4 = new TGraph();
  TGraph *g5 = new TGraph();
  TGraph *g9 = new TGraph();
  TGraph *g17 = new TGraph();
  g3->GetXaxis()->SetTitle("epoch"); 
  g3->GetYaxis()->SetTitle("loss"); 
  g3->SetMinimum(1); 
  g3->SetMaximum(6); 
  g3->SetLineColor(1); 
  g4->SetLineColor(2); 
  g5->SetLineColor(3); 
  g9->SetLineColor(4); 
  g17->SetLineColor(6); 

  TGraph *gjet = new TGraph();
  TGraph *gmix = new TGraph();
  gjet->SetLineColor(7); 
  gmix->SetLineColor(8); 

  TGraph *d5lr2e_5 = new TGraph();
  TGraph *d5lr5e_5 = new TGraph();
  d5lr2e_5->SetLineColor(9); 
  d5lr5e_5->SetLineColor(11); 

  TLegend *legend = new TLegend( 0.6, 0.48, 0.8, 0.78);
  legend->AddEntry( g3, "D=2", "l"); // AddEntry( pointer , "interpretation" , "option" )
  legend->AddEntry( g4, "D=3", "l"); // option は　"f"=box, "l"="L"=line, "p"=marker
  legend->AddEntry( g5, "D=4", "l") ;
  legend->AddEntry( g9, "D=8", "l") ;
  legend->AddEntry( g17, "D=16", "l") ;
  legend->SetFillColor(0);
  if(mix_jets){
    legend->AddEntry( gjet, "D=5 jet", "l") ;
    legend->AddEntry( gmix, "D=5 mix", "l") ;
  }

  TLegend *legend_lr = new TLegend( 0.6, 0.48, 0.9, 0.78);
  legend_lr->AddEntry( g5, "D=4 lr=9e-6", "l");
  legend_lr->AddEntry( d5lr2e_5, "D=4 lr=2e-5", "l");
  legend_lr->AddEntry( d5lr5e_5, "D=4 lr=5e-5", "l");


  for(int i=0;i<50;i++){
    g3->SetPoint(i,i,returning3[i]);
    g4->SetPoint(i,i,returning4[i]);
    g5->SetPoint(i,i,returning5[i]);
    // g9->SetPoint(i,i,returning9[i]);
    g9->SetPoint(i,i,returning9_20240704110317[i]);
    g17->SetPoint(i,i,returning17[i]);
    if(mix_jets){
      gjet->SetPoint(i,i,returning_jet5[i]);
      if(i<25) gmix->SetPoint(i,i,returning_mix5[i]);
    }
    d5lr2e_5->SetPoint(i,i,tauD5_lr2e_5[i]);
    d5lr5e_5->SetPoint(i,i,tauD5_lr5e_5[i]);
  }

  g3->Draw();
  g4->Draw("same");
  g5->Draw("same");
  g9->Draw("same");
  g17->Draw("same");
  if(mix_jets){
    gjet->Draw("same");
    gmix->Draw("same");
  }
  legend->Draw();

  TCanvas *c_lr = new TCanvas(1);
  c_lr->cd();
  g5->GetXaxis()->SetTitle("epoch"); 
  g5->GetYaxis()->SetTitle("loss"); 
  g5->SetMinimum(1); 
  g5->SetMaximum(6); 
  g5->Draw();
  d5lr2e_5->Draw("same");
  d5lr5e_5->Draw("same");
  legend_lr->Draw();

}