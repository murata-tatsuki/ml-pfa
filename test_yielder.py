import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from model import get_model
#from dataset import get_dataset
from event import Event
from prediction import Prediction
from clustering import cluster
from matching import make_matches, matching_hungarian_set_bbox_only
from torch.nn.utils.rnn import pad_sequence
from train_clustering_ddp import pdg_id_to_class, query_construction

class TestYielder:
    def __init__(self, model=None, dataset=None, ckpt=None, device='cpu', timingCut=False, use_charge_track_likeness=False, pandora=False, event_energy=False):
        self.model = get_model(jit=False) if model is None else model
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location=torch.device(device))['model'])
        self.dataset = dataset
        #self.dataset = get_dataset(timingCut=timingCut) if dataset is None else dataset
        self.use_charge_track_likeness = use_charge_track_likeness
        self.device = device
        self.reset_loader()
        self.pandora = pandora
        self.event_energy = event_energy

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
            out_gravnet_batch = out_gravnet[same_batch] if not self.pandora else None
            data_batch = Batch.from_data_list([data_batch])
            yield i, data_batch, out_gravnet_batch

    def iter_pred(self, nmax=None, energyRegression=False, energyRegressionCluster=False, energyRegressionWeight=False):
        with torch.no_grad():
            self.model.eval()
            for i, data, out_gravnet in self._iter_data(nmax):
                if self.device!='cpu':
                    data=data.to('cpu')
                    out_gravnet=out_gravnet.to('cpu') if not self.pandora else None
                event = Event(data, self.pandora, self.event_energy)

                # label=event.y
                # unique_label=np.unique(label)
                # print(f"true clusters = {len(unique_label)}")
                #nclus = len(unique_label)
                #if nclus == 1: continue

                if not self.event_filter(event): continue
                # if len(data.x) < 50: continue

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

                    pred_tracker_energy = None
                    pred_cluster_energy = None
                    pred_weight_photon = None
                    pred_weight_hadron = None
                    pred_weight_muon = None
                    pred_weight_electron = None

                    if (not energyRegression):
                        if (self.use_charge_track_likeness):
                            pred_charge_track_likeness = torch.sigmoid(out_gravnet[:,1]).numpy()
                            pred_cluster_space_coords = out_gravnet[:,2:].numpy()
                        else:
                            pred_charge_track_likeness = None
                            pred_cluster_space_coords = out_gravnet[:,1:].numpy()

                        # add track hits info
                        charged_hits = event.x[:,4]
                    else:
                        if not energyRegressionWeight:
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
                        else:
                            pred_charge_track_likeness = None
                            pred_weight_photon = out_gravnet[:,1].numpy()
                            pred_weight_hadron = out_gravnet[:,2].numpy()
                            pred_weight_muon = out_gravnet[:,3].numpy()
                            pred_weight_electron = out_gravnet[:,4].numpy()
                            pred_cluster_space_coords = out_gravnet[:,5:].numpy()
                        # add track hits info
                        charged_hits = event.x[:,4]

                    prediction = Prediction(pred_betas, pred_cluster_space_coords, pred_charge_track_likeness, charged_hits, pred_tracker_energy, pred_cluster_energy, pred_weight_photon, pred_weight_hadron, pred_weight_muon, pred_weight_electron) #w/o noise
                else:
                    prediction = Prediction(None, None, None, event.x[:,4], event.pand[:,2], None, None, None, None, None) #w/o noise
                    # print(event.pand)
                #f.write(f"prediction pass_noise_filter : {prediction.pass_noise_filter}\n")
                yield event, prediction

    def iter_clustering(self, tbeta=0.7, td=0.5, nmax=None, energyRegression=False, energyRegressionCluster=False, clustering_td_momentum=False, energyRegressionWeight=False):
        for event, prediction in self.iter_pred(nmax, energyRegression, energyRegressionCluster, energyRegressionWeight):
            if not self.pandora:
                clustering, condensation_points = cluster(event, prediction, tbeta, td, clustering_td_momentum)
            else:
                clustering = None
                condensation_points = None
            pandora_clustering = np.array(event.pand[:,0], dtype=int).flatten() + 1 if self.pandora else None
            yield event, prediction, clustering, pandora_clustering, condensation_points

    def iter_matches(self, tbeta=0.7, td=0.5, nmax=None, energyRegression=False, energyRegressionCluster=False, clustering_td_momentum=False, energyRegressionWeight=False):
        for event, prediction, clustering, pandora_clustering, condensation_points in self.iter_clustering(tbeta, td, nmax, energyRegression, energyRegressionCluster, clustering_td_momentum, energyRegressionWeight):
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



# class with machine learned clustering result
class TestYielderWithMLClustering(TestYielder):
    def __init__(self, model_clustering=None, classification=False, pid=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classification = classification
        self.pid = pid
        self.model_clustering = get_model(jit=False) if model_clustering is None else model_clustering
    
    def reset_loader(self):
        self.batch_size = 1 if self.device=='cpu' else 100
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def pad_and_mask_batch(self, tensor, batch):
        """
        Args:
            tensor: (sum BM, D) - GNNの出力（embed, featなど）
            batch:  (sum BM,)   - 各行が属するイベントID
        Returns:
            padded_tensor: (B, N_max, D)
            mask:          (B, N_max) - 1 for valid, 0 for padded
        """
        device = tensor.device
        batch_ids = batch.unique(sorted=True)

        # イベント単位にテンソルを分割（List[(M_i, D)])
        split_tensors = [tensor[batch == b_id] for b_id in batch_ids]

        # パディングして (B, N_max, D)
        padded_tensor = pad_sequence(split_tensors, batch_first=True)  # padding = 0 by default

        # mask 生成: True where data is valid
        lengths = [t.size(0) for t in split_tensors]
        max_len = padded_tensor.size(1)
        mask = torch.zeros(len(lengths), max_len, dtype=torch.bool, device=device)
        for i, l in enumerate(lengths):
            mask[i, :l] = True

        return padded_tensor, mask

    def split_by_batch(self, tensor, batch):
        """
        Args:
            tensor: (N, D) - GNNの出力（embed, featなど）
            batch:  (N,)   - 各行が属するイベントID
        Returns:
            split_tensors: List[Tensor] - イベントごとのテンソル [(M_0, D), (M_1, D), ..., (M_B-1, D)]
        """
        batch_ids = batch.unique(sorted=True)
        split_tensors = [tensor[batch == b_id] for b_id in batch_ids]
        return split_tensors

    def pdg_id_to_class(self, pdg_ids):
        """
        pdg_ids: torch.Tensor or np.ndarray, shape (N,)
        Return: torch.Tensor, shape (N,) – class indices (0~9)
        """
        abs_ids = torch.abs(pdg_ids)
        cls = torch.full_like(pdg_ids, fill_value=9)  # default: other neutral hadron (ID=10)

        # Explicit mappings
        cls[pdg_ids == 22] = 0      # photon
        cls[abs_ids == 11] = 1      # electron / positron
        cls[abs_ids == 13] = 2      # muon
        cls[abs_ids == 211] = 3     # charged pion
        cls[abs_ids == 111] = 4     # neutral pion
        cls[abs_ids == 321] = 5     # charged kaon
        cls[abs_ids == 130] = 6     # K_L0
        cls[abs_ids == 310] = 6     # K_S0
        cls[abs_ids == 2212] = 7    # proton
        cls[abs_ids == 2112] = 8    # neutron

        return cls

    def get_gnn_output(self, batched_data, use_charged_cluster_loss=False, energy_regression=True, energy_regression_cluster=True):
        with torch.no_grad():
            gnn_outputs: torch.Tensor = self.model(batched_data.x, batched_data.batch).to(self.device)
            
            pred_betas = torch.sigmoid(gnn_outputs[:,0])
            if energy_regression:
                if not energy_regression_cluster:
                    if use_charged_cluster_loss:
                        pred_charge_track_likeness = torch.sigmoid(gnn_outputs[:,1])
                        pred_tracker_energy = gnn_outputs[:,2]
                        pred_cluster_energy = None
                        pred_cluster_space_coords = gnn_outputs[:,3:]
                        assert(pred_charge_track_likeness.device == self.device)
                    else:
                        pred_charge_track_likeness = None
                        pred_tracker_energy = gnn_outputs[:,1]
                        pred_cluster_energy = None
                        pred_cluster_space_coords = gnn_outputs[:,2:]
                else:
                    if use_charged_cluster_loss:
                        pred_charge_track_likeness = torch.sigmoid(gnn_outputs[:,1])
                        pred_tracker_energy = gnn_outputs[:,2]
                        pred_cluster_energy = gnn_outputs[:,3]
                        pred_cluster_space_coords = gnn_outputs[:,4:]
                        assert(pred_charge_track_likeness.device == self.device)
                    else:
                        pred_charge_track_likeness = None
                        pred_tracker_energy = gnn_outputs[:,1]
                        pred_cluster_energy = gnn_outputs[:,2]
                        pred_cluster_space_coords = gnn_outputs[:,3:]
            else:
                pred_tracker_energy = None
                pred_cluster_energy = None
                if use_charged_cluster_loss:
                    pred_charge_track_likeness = torch.sigmoid(gnn_outputs[:,1])
                    pred_cluster_space_coords = gnn_outputs[:,2:]
                    assert(pred_charge_track_likeness.device == self.device)
                else:
                    pred_charge_track_likeness = None
                    pred_cluster_space_coords = gnn_outputs[:,1:]
            cluster_track_index = batched_data.y[:,1]

            # assert all(t.device == self.device for t in [pred_betas, pred_cluster_space_coords, batched_data.y, batched_data.batch,])
            true_energy = torch.sqrt(torch.sum(torch.square(batched_data.label[:,4:8]), 1))
            detected_energy = batched_data.feat[:,0]

        return pred_cluster_space_coords, pred_betas
    
    def _iter_data(self, nmax=None):
        with torch.no_grad():
            self.model.eval()
            self.model_clustering.eval()

            for i, data in enumerate(self.loader):
                if nmax is not None and i >= nmax: break

                for event_number, event_data, pred_fourvec, truth_four_vector, mask, attn in self.iter_event(i, data):
                    event_num = event_number + i * self.batch_size
                    yield event_num, event_data, pred_fourvec, truth_four_vector, mask, attn

    def iter_event(self, i, data, nmax=None):
        data.to(self.device)

        pred_cluster_space_coords, pred_betas = self.get_gnn_output(data)
        hit_embed, hit_mask = self.pad_and_mask_batch(pred_cluster_space_coords, data.batch)
        hit_beta, _ = self.pad_and_mask_batch(pred_betas.unsqueeze(-1), data.batch)
        hit_beta = hit_beta.squeeze(-1)
        hit_feat, _ = self.pad_and_mask_batch(data.feat, data.batch)

        if self.classification or self.pid:
            pred_fourvec, pred_cls, attn_w = self.model_clustering(hit_embed, hit_beta, hit_feat, hit_mask)
        else:
            pred_fourvec, attn_w = self.model_clustering(hit_embed, hit_beta, hit_feat, hit_mask)
        if self.classification:
            unique_label = self.split_by_batch(data.label, data.batch)
            unique_label = [torch.unique(t[:,1], dim=0) for t in unique_label]
        if self.pid:
            unique_label = self.split_by_batch(data.label, data.batch)
            unique_label = [torch.unique(t[:,1:3], dim=0) for t in unique_label]
            unique_label = [self.pdg_id_to_class(t[:,1]) for t in unique_label]
        
        true_energy = torch.sqrt(torch.sum(torch.square(data.label[:,4:8]), 1))
        # truth_four_vector = torch.cat((data.label[:,5:8], true_energy.reshape(-1,1)), dim=1)
        truth_four_vector = torch.cat((true_energy.reshape(-1,1), data.label[:,5:8]), dim=1)
        truth_four_vector = self.split_by_batch(truth_four_vector, data.batch)
        truth_four_vector = [torch.unique(t, dim=0) for t in truth_four_vector]
        # 必要なら yield に含める要素を追加してください
        # yield i, data, pred_fourvec, pred_cls, truth_four_vector, unique_label

        # print(pred_fourvec, len(pred_fourvec[0]), truth_four_vector, len(truth_four_vector[0]))

        assert(torch.unique(data.batch).shape[0] == len(pred_fourvec))

        data_list = data.to_data_list()
        for batch_id, data_batch, pred_vecs, truth_vecs, mask, attn in zip(torch.unique(data.batch), data_list, pred_fourvec, truth_four_vector, hit_mask, attn_w):
            data_batch = Batch.from_data_list([data_batch])
            yield i, data_batch.to('cpu'), pred_vecs.to('cpu'), truth_vecs.to('cpu'), mask.to('cpu'), attn.to('cpu')


class TestYielder_transformer_Like_Clustering:
    def __init__(self, model=None, dataset=None, ckpt=None, device='cpu', timingCut=False, use_charge_track_likeness=False, pandora=False, event_energy=False):
        self.model = get_model(jit=False) if model is None else model
        if ckpt:
            model.load_state_dict(torch.load(ckpt, map_location=torch.device(device))['model'])
        self.dataset = dataset
        #self.dataset = get_dataset(timingCut=timingCut) if dataset is None else dataset
        self.use_charge_track_likeness = use_charge_track_likeness
        self.device = device
        self.reset_loader()
        self.pandora = pandora
        self.event_energy = event_energy

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
                print(data.x.shape, data.batch.shape, data.y.shape)
                out_gravnet = self.model(data.x, data.batch).to(self.device) if not self.pandora else None
                print(data.x.shape, data.batch.shape)
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
            out_gravnet_batch = out_gravnet[same_batch] if not self.pandora else None
            data_batch = Batch.from_data_list([data_batch])
            yield i, data_batch, out_gravnet_batch

    def iter_pred(self, nmax=None, energyRegression=False, energyRegressionCluster=False, energyRegressionWeight=False):
        with torch.no_grad():
            self.model.eval()
            for i, data, out_gravnet in self._iter_data(nmax):
                if self.device!='cpu':
                    data=data.to('cpu')
                    out_gravnet=out_gravnet.to('cpu') if not self.pandora else None
                event = Event(data, self.pandora, self.event_energy)

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

                    pred_tracker_energy = None
                    pred_cluster_energy = None
                    pred_weight_photon = None
                    pred_weight_hadron = None
                    pred_weight_muon = None
                    pred_weight_electron = None

                    if (not energyRegression):
                        if (self.use_charge_track_likeness):
                            pred_charge_track_likeness = torch.sigmoid(out_gravnet[:,1]).numpy()
                            pred_cluster_space_coords = out_gravnet[:,2:].numpy()
                        else:
                            pred_charge_track_likeness = None
                            pred_cluster_space_coords = out_gravnet[:,1:].numpy()

                        # add track hits info
                        charged_hits = event.x[:,4]
                    else:
                        if not energyRegressionWeight:
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
                        else:
                            pred_charge_track_likeness = None
                            pred_weight_photon = out_gravnet[:,1].numpy()
                            pred_weight_hadron = out_gravnet[:,2].numpy()
                            pred_weight_muon = out_gravnet[:,3].numpy()
                            pred_weight_electron = out_gravnet[:,4].numpy()
                            pred_cluster_space_coords = out_gravnet[:,5:].numpy()
                        # add track hits info
                        charged_hits = event.x[:,4]

                    prediction = Prediction(pred_betas, pred_cluster_space_coords, pred_charge_track_likeness, charged_hits, pred_tracker_energy, pred_cluster_energy, pred_weight_photon, pred_weight_hadron, pred_weight_muon, pred_weight_electron) #w/o noise
                else:
                    prediction = Prediction(None, None, None, event.x[:,4], event.pand[:,2], None, None, None, None, None) #w/o noise
                    # print(event.pand)
                #f.write(f"prediction pass_noise_filter : {prediction.pass_noise_filter}\n")
                yield event, data, prediction

    def iter_clustering(self, tbeta=0.7, td=0.5, nmax=None, energyRegression=False, energyRegressionCluster=False, clustering_td_momentum=False, energyRegressionWeight=False):
        for event, data, prediction in self.iter_pred(nmax, energyRegression, energyRegressionCluster, energyRegressionWeight):
            if not self.pandora:
                clustering, condensation_points = cluster(event, prediction, tbeta, td, clustering_td_momentum)
            else:
                clustering = None
                condensation_points = None
            pandora_clustering = np.array(event.pand[:,0], dtype=int).flatten() + 1 if self.pandora else None
            yield event, data, prediction, clustering, pandora_clustering, condensation_points

    def iter_matches(self, tbeta=0.7, td=0.5, nmax=None, energyRegression=False, energyRegressionCluster=False, clustering_td_momentum=False, energyRegressionWeight=False):
        for event, data, prediction, clustering, pandora_clustering, condensation_points in self.iter_clustering(tbeta, td, nmax, energyRegression, energyRegressionCluster, clustering_td_momentum, energyRegressionWeight):
            if not self.pandora:
                matches = make_matches(event, prediction, clustering=clustering)
            else:
                matches = make_matches(event, prediction, clustering=pandora_clustering)
            cluster = clustering if not self.pandora else pandora_clustering
            yield event, data, prediction, cluster, matches, condensation_points


def match_hits_to_queries(attn_weights):
    """
    すでに計算済みの attention weight (B, N_query, N_hit) を用いて、
    各 hit が最も強く結びついている query の index を返す関数。

    Args:
        attn_weights: Tensor, shape = (B, N_query, N_hit)
                      query→hit の attention 確率（softmax後）

    Returns:
        hit_to_query: Tensor, shape = (B, N_hit)
                      各 hit の最大 attention を持つ query index
    """
    # --- Step 1 ---
    # attention map を転置して、(B, N_hit, N_query にする)
    # こうすることで hit ごとの query 比較が簡単にできる
    attn_hit_view = attn_weights.transpose(1, 2)  # (B, N_hit, N_query)

    # --- Step 2 ---
    # 各 hit について attention が最大の query を選ぶ
    # dim=-1 は query 次元に沿って argmax を取る
    hit_to_query = torch.argmax(attn_hit_view, dim=-1)  # (B, N_hit)

    return hit_to_query

class TestYielderWithMLClustering_trackQuery(TestYielder):
    def __init__(self, model_clustering=None, energyRegression=False, energyRegressionCluster=False, energyRegressionWeight=False, classification=False, pid=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classification = classification
        self.pid = pid
        self.model_clustering = get_model(jit=False) if model_clustering is None else model_clustering
        self.energyRegression=energyRegression
        self.energyRegressionCluster=energyRegressionCluster
        self.energyRegressionWeight=energyRegressionWeight
    
    def reset_loader(self):
        self.batch_size = 1 if self.device=='cpu' else 20
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
    
    def get_gnn_output_allFeat(self, batched_data):
        with torch.no_grad():
            gnn_outputs: torch.Tensor = self.model(batched_data.x, batched_data.batch)
        return gnn_outputs, torch.sigmoid(gnn_outputs[:,0])
    
    def feat_format(self, gnn_output, feat):
        gnn_output[:,0] = torch.sigmoid(gnn_output[:,0])
        feat[:,1:4] = feat[:,1:4] / 2000
        feat[:,7:10] = feat[:,7:10] / 3
        return torch.cat((gnn_output, feat), dim=-1)


    def pad_and_mask_batch(self, tensor, batch):
        """
        Args:
            tensor: (sum BM, D) - GNNの出力（embed, featなど）
            batch:  (sum BM,)   - 各行が属するイベントID
        Returns:
            padded_tensor: (B, N_max, D)
            mask:          (B, N_max) - 1 for valid, 0 for padded
        """
        device = tensor.device
        batch_ids = batch.unique(sorted=True)

        # イベント単位にテンソルを分割（List[(M_i, D)])
        split_tensors = [tensor[batch == b_id] for b_id in batch_ids]

        # パディングして (B, N_max, D)
        padded_tensor = pad_sequence(split_tensors, batch_first=True)  # padding = 0 by default

        # mask 生成: True where data is valid
        lengths = [t.size(0) for t in split_tensors]
        max_len = padded_tensor.size(1)
        mask = torch.zeros(len(lengths), max_len, dtype=torch.bool, device=device)
        for i, l in enumerate(lengths):
            mask[i, :l] = True

        return padded_tensor, mask

    def split_by_batch(self, tensor, batch):
        """
        Args:
            tensor: (N, D) - GNNの出力（embed, featなど）
            batch:  (N,)   - 各行が属するイベントID
        Returns:
            split_tensors: List[Tensor] - イベントごとのテンソル [(M_0, D), (M_1, D), ..., (M_B-1, D)]
        """
        batch_ids = batch.unique(sorted=True)
        split_tensors = [tensor[batch == b_id] for b_id in batch_ids]
        return split_tensors
    
    def split_by_batch_unique(self, tensor, batch):
        """
        Args:
            tensor: (N, D) - GNNの出力（embed, featなど）
            batch:  (N,)   - 各行が属するイベントID
        Returns:
            split_tensors: List[Tensor] - イベントごとのテンソル [(M_0, D), (M_1, D), ..., (M_B-1, D)]
        """
        batch_ids = batch.unique(sorted=True)
        split_tensors = [torch.unique(tensor[batch == b_id], dim=0) for b_id in batch_ids]
        return split_tensors

    def split_by_batch_padded(self, tensor, batch):
        """
        Args:
            tensor: (N, D)
            batch:  (N,)
        Returns:
            padded_tensor: (B, N_max, D)
            mask: (B, N_max)  # True=有効
        """
        batch_ids = batch.unique(sorted=True)
        split_tensors = [tensor[batch == b_id] for b_id in batch_ids]

        max_len = max(t.size(0) for t in split_tensors)
        feat_dim = tensor.size(-1)
        device = tensor.device
        B = len(split_tensors)

        padded = torch.zeros(B, max_len, feat_dim, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)

        for i, t in enumerate(split_tensors):
            n = t.size(0)
            padded[i, :n, :] = t
            mask[i, :n] = True

        return padded, mask

    def pdg_id_to_class(self, pdg_ids):
        """
        pdg_ids: torch.Tensor or np.ndarray, shape (N,)
        Return: torch.Tensor, shape (N,) – class indices (0~9)
        """
        abs_ids = torch.abs(pdg_ids)
        cls = torch.full_like(pdg_ids, fill_value=9)  # default: other neutral hadron (ID=10)

        # Explicit mappings
        cls[pdg_ids == 22] = 0      # photon
        cls[abs_ids == 11] = 1      # electron / positron
        cls[abs_ids == 13] = 2      # muon
        cls[abs_ids == 211] = 3     # charged pion
        cls[abs_ids == 111] = 4     # neutral pion
        cls[abs_ids == 321] = 5     # charged kaon
        cls[abs_ids == 130] = 6     # K_L0
        cls[abs_ids == 310] = 6     # K_S0
        cls[abs_ids == 2212] = 7    # proton
        cls[abs_ids == 2112] = 8    # neutron

        return cls

    def iter_clustering_model(self, nmax=None):
        with torch.no_grad():
            self.model_clustering.eval()
            for i, data, out_gravnet in self._iter_data(nmax):
                if self.device!='cpu':
                    data=data.to('cpu')
                    out_gravnet=out_gravnet.to('cpu') if not self.pandora else None
                event = Event(data, self.pandora, self.event_energy)

                hit_features            = self.feat_format(out_gravnet, data.feat[:,:-3])
                hit_embed, hit_mask     = self.pad_and_mask_batch(hit_features, data.batch)

                query, seed_padding_mask, query_indices_in_key, seed_track_mask = query_construction(hit_embed, hit_mask=hit_mask)
                pred_fourvec, particle_prob, particle_cls_logits, attn_w = self.model_clustering(hit_embed, query, hit_mask=hit_mask)

                unique_label = self.split_by_batch(data.label[:,1:4], data.batch)
                unique_label = [torch.unique(t, dim=0) for t in unique_label]
                true_cls = [pdg_id_to_class(t[:,1], t[:,2]) for t in unique_label]
                true_energy = torch.sqrt(torch.sum(torch.square(data.label[:,4:8]), 1))
                # truth_four_vector = torch.cat((data.label[:,5:8], true_energy.reshape(true_energy.shape[0],1)), 1)
                truth_four_vector_torchTensor = torch.cat((true_energy.reshape(true_energy.shape[0],1), data.label[:,5:8]), 1)
                truth_four_vector = self.split_by_batch_unique(truth_four_vector_torchTensor, data.batch)
                truth_four_vector_torchTensor, true_mask = self.split_by_batch_padded(truth_four_vector_torchTensor, data.batch)

                yield i, data, out_gravnet, pred_fourvec, truth_four_vector, hit_mask, particle_prob, particle_cls_logits, attn_w

    def iter_pred(self, nmax=None):
        with torch.no_grad():
            self.model.eval()
            for i, data, out_gravnet, pred_fourvec, truth_four_vector, mask, pcl_prob, cls_logits, attn in self.iter_clustering_model(nmax):
                
                event = Event(data, self.pandora, self.event_energy)

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

                    pred_tracker_energy = None
                    pred_cluster_energy = None
                    pred_weight_photon = None
                    pred_weight_hadron = None
                    pred_weight_muon = None
                    pred_weight_electron = None
                    pred_charge_track_likeness = None

                    if (not self.energyRegression):
                        if (self.use_charge_track_likeness):
                            pred_charge_track_likeness = torch.sigmoid(out_gravnet[:,1]).numpy()
                            pred_cluster_space_coords = out_gravnet[:,2:].numpy()
                        else:
                            pred_cluster_space_coords = out_gravnet[:,1:].numpy()

                        # add track hits info
                        charged_hits = event.x[:,4]
                    else:
                        if not self.energyRegressionWeight:
                            if (not self.energyRegressionCluster):
                                if (self.use_charge_track_likeness):
                                    pred_charge_track_likeness = torch.sigmoid(out_gravnet[:,1]).numpy()
                                    pred_tracker_energy = out_gravnet[:,2].numpy()
                                    pred_cluster_space_coords = out_gravnet[:,3:].numpy()
                                else:
                                    pred_tracker_energy = out_gravnet[:,1].numpy()
                                    pred_cluster_space_coords = out_gravnet[:,2:].numpy()
                            else:
                                if (self.use_charge_track_likeness):
                                    pred_charge_track_likeness = torch.sigmoid(out_gravnet[:,1]).numpy()
                                    pred_tracker_energy = out_gravnet[:,2].numpy()
                                    pred_cluster_energy = out_gravnet[:,3].numpy()
                                    pred_cluster_space_coords = out_gravnet[:,4:].numpy()
                                else:
                                    pred_tracker_energy = out_gravnet[:,1].numpy()
                                    pred_cluster_energy = out_gravnet[:,2].numpy()
                                    pred_cluster_space_coords = out_gravnet[:,3:].numpy()
                        else:
                            pred_weight_photon = out_gravnet[:,1].numpy()
                            pred_weight_hadron = out_gravnet[:,2].numpy()
                            pred_weight_muon = out_gravnet[:,3].numpy()
                            pred_weight_electron = out_gravnet[:,4].numpy()
                            pred_cluster_space_coords = out_gravnet[:,5:].numpy()
                        # add track hits info
                        charged_hits = event.x[:,4]

                    prediction = Prediction(pred_betas, pred_cluster_space_coords, pred_charge_track_likeness, charged_hits, pred_tracker_energy, pred_cluster_energy, pred_weight_photon, pred_weight_hadron, pred_weight_muon, pred_weight_electron) #w/o noise
                else:
                    prediction = Prediction(None, None, None, event.x[:,4], event.pand[:,2], None, None, None, None, None) #w/o noise
                    # print(event.pand)
                #f.write(f"prediction pass_noise_filter : {prediction.pass_noise_filter}\n")
                yield i, event, prediction, pred_fourvec, truth_four_vector, mask, pcl_prob, cls_logits, attn

    def iter_clustering(self, nmax=None):
        for event_num, event, prediction, pred_fourvec, truth_four_vector, mask, pcl_prob, cls_logits, attn in self.iter_pred(nmax):
            attn_weight = match_hits_to_queries(attn[-1]).detach().numpy().copy()
            trans_clustering = attn_weight[0]
            yield event_num, event, prediction, pred_fourvec, truth_four_vector, mask, pcl_prob, cls_logits, attn, trans_clustering

    def iter_matches(self, nmax=None):
        for event_num, event, prediction, pred_fourvec, truth_four_vector, mask, pcl_prob, cls_logits, attn, clustering in self.iter_clustering(nmax):
            matches = make_matches(event, None, clustering=clustering)
            yield event_num, event, prediction, pred_fourvec, truth_four_vector, mask, pcl_prob, cls_logits, attn, clustering, matches