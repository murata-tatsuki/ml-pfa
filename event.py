import numpy as np
from torch_geometric.data import Data

class Event:
    def __init__(self, data: Data, pandora=False):
        self.x = data.x.numpy()
        self.y = data.y.numpy()
        if hasattr(data, 'truth_cluster_props'):
            self.truth_cluster_props = data.truth_cluster_props.numpy()
        else:
            self.truth_cluster_props = np.zeros((self.x.shape[0], 5))
        #if hasattr(data, 'inpz'):
        #    self.inpz = int(data.inpz[0].item())
        self.batch = data.batch.numpy()
        self.feat = data.feat
        self.label = data.label
        self.pand = data.pand if pandora else None

    @property
    def truth_e_bound(self):
        return self.truth_cluster_props[:,0]

    @property
    def truth_x_bound(self):
        return self.truth_cluster_props[:,1]

    @property
    def truth_y_bound(self):
        return self.truth_cluster_props[:,2]

    @property
    def truth_time(self):
        return self.truth_cluster_props[:,3]

    @property
    def truth_pdgid(self):
        return self.truth_cluster_props[:,4]

    # Getters for a single truth id

    def index_by_id(self, id):
        return (self.y == id).argmax()

    def truth_e_bound_by_id(self, id):
        return self.truth_e_bound[self.index_by_id(id)]

    def truth_x_bound_by_id(self, id):
        return self.truth_x_bound[self.index_by_id(id)]

    def truth_y_bound_by_id(self, id):
        return self.truth_y_bound[self.index_by_id(id)]

    def truth_time_by_id(self, id):
        return self.truth_time[self.index_by_id(id)]

    def truth_pdgid_by_id(self, id):
        return self.truth_pdgid[self.index_by_id(id)]

    @property
    def energy(self):
        return self.feat[:,0]

    @property
    def momentumVector(self):
        return self.feat[:,7:10]

    @property
    def time(self):
        return self.x[:,8]

    @property
    def etahit(self):
        return self.x[:,1]

    @property
    def zerofeature(self):
        return self.x[:,2]

    @property
    def thetahit(self):
        return self.x[:,3]

    @property
    def rhit(self):
        return self.x[:,4]
    
    @property
    def xhit(self):
        return self.x[:,5]

    @property
    def yhit(self):
        return self.x[:,6]

    @property
    def zhit(self):
        return self.x[:,7]

    @property
    def time(self):
        return self.x[:,8]

    @property
    def select_em_hits(self):
        return np.isin(np.abs(self.truth_pdgid), np.array([11, 22, 111]))

    @property
    def select_mip_hits(self):
        return np.isin(np.abs(self.truth_pdgid), np.array([13]))

    @property
    def select_noise_hits(self):
        return self.y <= 0

    @property
    def select_signal_hits(self):
        return self.y > 0

    @property
    def select_had_hits(self):
        return (self.select_signal_hits & (~self.select_em_hits) & (~self.select_mip_hits))

    @property
    def em_energy_fraction(self):
        return self.energy[self.select_em_hits].sum() / self.energy[self.select_signal_hits].sum()

    @property
    def had_energy_fraction(self):
        return self.energy[self.select_had_hits].sum() / self.energy[self.select_signal_hits].sum()

    @property
    def mip_energy_fraction(self):
        return self.energy[self.select_mip_hits].sum() / self.energy[self.select_signal_hits].sum()
