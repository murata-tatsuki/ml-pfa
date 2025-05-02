import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from model import get_model
#from dataset import get_dataset
from event import Event
from prediction import Prediction
from clustering import cluster
from matching import make_matches

class TestYielder:
    def __init__(self, model=None, dataset=None, ckpt=None, device='cpu', timingCut=False, use_charge_track_likeness=False, pandora=False):
        self.model = get_model(jit=False) if model is None else model
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location=torch.device(device))['model'])
        self.dataset = dataset
        #self.dataset = get_dataset(timingCut=timingCut) if dataset is None else dataset
        self.use_charge_track_likeness = use_charge_track_likeness
        self.device = device
        self.reset_loader()
        self.pandora = pandora

    def reset_loader(self):
        self.batch_size = 1 if self.device=='cpu' else 20
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def event_filter(self, event):
        """Subclassable to make an event-level filter before any model inference (for speed)"""
        return True

    def _iter_data(self, nmax=None):
        for i, data in enumerate(self.loader):
            if nmax is not None and i >= nmax: break

            if self.device=='cpu':
                data.to(self.device)
                out_gravnet = self.model(data.x, data.batch).to(self.device) if not self.pandora else None
                yield i, data, out_gravnet
            else:
                for event_number, event_data, event_out_gravnet in self.iter_event(i, data):
                    event_num = event_number + i * self.batch_size
                    yield event_num, event_data, event_out_gravnet

    def iter_event(self, i, data):
        data.to(self.device)
        out_gravnet = self.model(data.x, data.batch).to(self.device) if not self.pandora else None
        # data.to('cpu')
        # out_gravnet.to('cpu')
        data_list = data.to_data_list()
        for batch_id, data_batch in zip(torch.unique(data.batch), data_list):
            same_batch = data.batch==batch_id
            out_gravnet_batch = out_gravnet[same_batch]
            data_batch = Batch.from_data_list([data_batch])
            yield i, data_batch, out_gravnet_batch

    def iter_pred(self, nmax=None, energyRegression=False, energyRegressionCluster=False):
        with torch.no_grad():
            self.model.eval()
            for i, data, out_gravnet in self._iter_data(nmax):
                if self.device!='cpu':
                    data=data.to('cpu')
                    out_gravnet=out_gravnet.to('cpu')
                event = Event(data, self.pandora)

                # label=event.y
                # unique_label=np.unique(label)
                # print(f"true clusters = {len(unique_label)}")
                #nclus = len(unique_label)
                #if nclus == 1: continue

                if not self.event_filter(event): continue
                if len(data.x) < 50: continue

                # print(f"{len(data.x)=},{len(data.batch)=}")
                # print(f"{data.x=}")
                #print(f"{data.batch=}")

                if not self.pandora:
                    # if self.device!='cpu': data.to(self.device)
                    #_,pass_noise_filter,out_gravnet = self.model(data.x, data.batch) #NoiseFilter
                    # out_gravnet = self.model(data.x, data.batch) #w/o NoiseFilter
                    # if self.device!='cpu': 
                    #     data.to('cpu')
                    #     out_gravnet.to('cpu')
                    #pass_noise_filter = pass_noise_filter.numpy() #NoiseFilter
                    pred_betas = torch.sigmoid(out_gravnet[:,0]).numpy()

                    if (not energyRegression):
                        if (self.use_charge_track_likeness):
                            pred_charge_track_likeness = torch.sigmoid(out_gravnet[:,1]).numpy()
                            pred_cluster_space_coords = out_gravnet[:,2:].numpy()
                            pred_tracker_energy = None
                            pred_cluster_energy = None
                        else:
                            pred_charge_track_likeness = None
                            pred_cluster_space_coords = out_gravnet[:,1:].numpy()
                            pred_tracker_energy = None
                            pred_cluster_energy = None

                        # add track hits info
                        charged_hits = event.x[:,4]
                    else :
                        if (not energyRegressionCluster):
                            if (self.use_charge_track_likeness):
                                pred_charge_track_likeness = torch.sigmoid(out_gravnet[:,1]).numpy()
                                pred_tracker_energy = out_gravnet[:,2].numpy()
                                pred_cluster_energy = None
                                pred_cluster_space_coords = out_gravnet[:,3:].numpy()
                            else:
                                pred_charge_track_likeness = None
                                pred_tracker_energy = out_gravnet[:,1].numpy()
                                pred_cluster_energy = None
                                pred_cluster_space_coords = out_gravnet[:,2:].numpy()
                        else:
                            if (self.use_charge_track_likeness):
                                pred_charge_track_likeness = torch.sigmoid(out_gravnet[:,1]).numpy()
                                pred_tracker_energy = out_gravnet[:,2].numpy()
                                pred_cluster_energy = out_gravnet[:,3].numpy()
                                pred_cluster_space_coords = out_gravnet[:,4:].numpy()
                            else:
                                pred_charge_track_likeness = None
                                pred_tracker_energy = out_gravnet[:,1].numpy()
                                pred_cluster_energy = out_gravnet[:,2].numpy()
                                pred_cluster_space_coords = out_gravnet[:,3:].numpy()
                        # add track hits info
                        charged_hits = event.x[:,4]

                    prediction = Prediction(pred_betas, pred_cluster_space_coords, pred_charge_track_likeness, charged_hits, pred_tracker_energy, pred_cluster_energy) #w/o noise
                else:
                    prediction = Prediction(None, None, None, event.x[:,4], event.pand[:,2], None) #w/o noise
                    # print(event.pand)
                #f.write(f"prediction pass_noise_filter : {prediction.pass_noise_filter}\n")
                yield event, prediction

    def iter_clustering(self, tbeta=0.7, td=0.5, nmax=None, energyRegression=False, energyRegressionCluster=False, clustering_td_momentum=False):
        for event, prediction in self.iter_pred(nmax, energyRegression, energyRegressionCluster):
            if not self.pandora:
                clustering, condensation_points = cluster(event, prediction, tbeta, td, clustering_td_momentum)
            else:
                clustering = None
                condensation_points = None
            pandora_clustering = np.array(event.pand[:,0], dtype=int).flatten() + 1 if self.pandora else None
            yield event, prediction, clustering, pandora_clustering, condensation_points

    def iter_matches(self, tbeta=0.7, td=0.5, nmax=None, energyRegression=False, energyRegressionCluster=False, clustering_td_momentum=False):
        for event, prediction, clustering, pandora_clustering, condensation_points in self.iter_clustering(tbeta, td, nmax, energyRegression, energyRegressionCluster, clustering_td_momentum):
            if not self.pandora:
                matches = make_matches(event, prediction, clustering=clustering)
            else:
                matches = make_matches(event, prediction, clustering=pandora_clustering)
            cluster = clustering if not self.pandora else pandora_clustering
            yield event, prediction, cluster, matches, condensation_points


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
