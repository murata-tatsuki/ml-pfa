import numpy as np
import torch
from torch_geometric.loader import DataLoader
from model import get_model
#from dataset import get_dataset
from event import Event
from prediction import Prediction
from clustering import cluster
from matching import make_matches

class TestYielder:
    def __init__(self, model=None, dataset=None, ckpt=None, timingCut=False, use_charge_track_likeness=False):
        self.model = get_model(jit=False) if model is None else model
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location=torch.device('cpu'))['model'])
        self.dataset = dataset
        #self.dataset = get_dataset(timingCut=timingCut) if dataset is None else dataset
        self.use_charge_track_likeness = use_charge_track_likeness
        self.reset_loader()

    def reset_loader(self):
        self.loader = DataLoader(self.dataset, batch_size=1, shuffle=False)

    def event_filter(self, event):
        """Subclassable to make an event-level filter before any model inference (for speed)"""
        return True

    def _iter_data(self, nmax=None):
        for i, data in enumerate(self.loader):
            if nmax is not None and i >= nmax:
                break
            yield i, data

    def iter(self, nmax=None):
        for i, data in self._iter_data(nmax):
            event = Event(data)
            if not self.event_filter(event): continue
            yield event

    def iter_pred(self, nmax=None, pandora=False, energyRegression=False):
        with torch.no_grad():
            self.model.eval()
            for i, data in self._iter_data(nmax):
                event = Event(data, pandora)

                label=event.y
                unique_label=np.unique(label)
                # print(f"true clusters = {len(unique_label)}")
                #nclus = len(unique_label)
                #if nclus == 1: continue
                
                if not self.event_filter(event): continue
                if len(data.x) < 50: continue

                # print(f"{len(data.x)=},{len(data.batch)=}")
                # print(f"{data.x=}")
                #print(f"{data.batch=}")

                #_,pass_noise_filter,out_gravnet = self.model(data.x, data.batch) #NoiseFilter
                out_gravnet = self.model(data.x, data.batch) #w/o NoiseFilter
                #pass_noise_filter = pass_noise_filter.numpy() #NoiseFilter
                pred_betas = torch.sigmoid(out_gravnet[:,0]).numpy()

                if (not energyRegression):
                    if (self.use_charge_track_likeness):
                        pred_charge_track_likeness = torch.sigmoid(out_gravnet[:,1]).numpy()
                        pred_cluster_space_coords = out_gravnet[:,2:].numpy()
                        pred_cluster_energy = None
                    else:
                        pred_charge_track_likeness = None
                        pred_cluster_space_coords = out_gravnet[:,1:].numpy()
                        pred_cluster_energy = None

                    #prediction = Prediction(pass_noise_filter, pred_betas, pred_cluster_space_coords)

                    # add track hits info
                    charged_hits = event.x[:,4]
                else :
                    if (self.use_charge_track_likeness):
                        pred_charge_track_likeness = torch.sigmoid(out_gravnet[:,1]).numpy()
                        pred_cluster_energy = out_gravnet[:,2].numpy()
                        pred_cluster_space_coords = out_gravnet[:,3:].numpy()
                    else:
                        pred_charge_track_likeness = None
                        pred_cluster_energy = out_gravnet[:,1].numpy()
                        pred_cluster_space_coords = out_gravnet[:,2:].numpy()
                    # add track hits info
                    charged_hits = event.x[:,4]

                # print("*******************")
                # for i,x in enumerate(charged_hits):
                #     print(f"[{i:03}] charged_hits={x:03}")

                prediction = Prediction(pred_betas, pred_cluster_space_coords, pred_charge_track_likeness, charged_hits, pred_cluster_energy) #w/o noise
                #f.write(f"prediction pass_noise_filter : {prediction.pass_noise_filter}\n")
                yield event, prediction

    def iter_clustering(self, tbeta=0.7, td=0.5, nmax=None, pandora=False, energyRegression=False, clustering_td_momentum=False):
        for event, prediction in self.iter_pred(nmax, pandora, energyRegression):
            clustering = cluster(event, prediction, tbeta, td, clustering_td_momentum)
            pandora_clustering = np.array(event.pand, dtype=int).flatten() + 1 if pandora else None
            yield event, prediction, clustering, pandora_clustering

    def iter_matches(self, tbeta=0.7, td=0.5, nmax=None, pandora=False, energyRegression=False, clustering_td_momentum=False):
        for event, prediction, clustering, pandora_clustering in self.iter_clustering(tbeta, td, nmax, pandora, energyRegression, clustering_td_momentum):
            if not pandora:
                matches = make_matches(event, prediction, clustering=clustering)
            else:
                matches = make_matches(event, prediction, clustering=pandora_clustering)
            cluster = clustering if not pandora else pandora_clustering
            yield event, prediction, cluster, matches


class TestYielderEM(TestYielder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_em_fraction = 1.0

    def event_filter(self, event):
        return event.em_energy_fraction >= self.min_em_fraction

class TestYielderHAD(TestYielder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_had_fraction = 1.0

    def event_filter(self, event):
        return event.had_energy_fraction >= self.min_had_fraction

class TestYielderMIP(TestYielder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_mip_fraction = 1.0

    def event_filter(self, event):
        return event.mip_energy_fraction >= self.min_mip_fraction


# class TestYielderSinglePhoton(TestYielder):
#     def __init__(self, *args, **kwargs):
#         kwargs['dataset'] = -1
#         super().__init__(*args, **kwargs)

#     def reset_loader(self):
#         self.loader = single_photon_dataset()()

class TestYielderSingleTruthShower(TestYielder):
    def event_filter(self, event: Event):
        total_energy = event.energy[event.select_signal_hits].sum()
        for id in np.unique(event.y):
            if id == 0: continue
            shower_energy = event.energy[event.y==id].sum()
            if shower_energy / total_energy > .95:
                print(
                    f'{shower_energy=}, {total_energy=}, '
                    f'r={shower_energy/total_energy}, pdgid={event.truth_pdgid_by_id(id)}'
                    )
                return True
        return False
