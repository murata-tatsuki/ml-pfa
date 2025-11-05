class Prediction:
    #NoiseFilter
    #def __init__(self, pass_noise_filter, pred_betas, pred_cluster_space_coords):
    #    self.pass_noise_filter = pass_noise_filter
    #    self.pred_betas = pred_betas
    #    self.pred_cluster_space_coords = pred_cluster_space_coords
        #print(f"prediction.pass_noise_filter : {self.pass_noise_filter}")
        #print(f"prediction.pred_betas : {self.pred_betas}")
        #print(f"pred_cluster_space_coords : {self.pred_cluster_space_coords}")

    #w/o NoiseFilter
    def __init__(self,pred_betas, pred_cluster_space_coords, pred_charge_track_likeness, charged_hits, pred_tracker_energy=None, pred_cluster_energy=None, pred_weight_photon=None, pred_weight_charged_hadron=None, pred_weight_neutral_hadron=None, pred_weight_muon=None, pred_weight_electron=None):
        self.pred_betas = pred_betas
        self.pred_cluster_space_coords = pred_cluster_space_coords
        self.pred_charge_track_likeness = pred_charge_track_likeness
        self.charged_hits = charged_hits
        self.pred_tracker_energy = pred_tracker_energy
        self.pred_cluster_energy = pred_cluster_energy
        self.pred_weight_photon = pred_weight_photon
        self.pred_weight_charged_hadron = pred_weight_charged_hadron
        self.pred_weight_neutral_hadron = pred_weight_neutral_hadron
        self.pred_weight_muon = pred_weight_muon
        self.pred_weight_electron = pred_weight_electron
