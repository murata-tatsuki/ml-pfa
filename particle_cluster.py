"""Opt-in particle-cluster energy and additional-particle score.

The GNN is shared and trained jointly. Membership is discrete; pooling and both
outputs remain differentiable with respect to the shared hit features. Detector
measurements alone enter the head; truth is used exclusively by target helpers.
"""
import math
import torch
from torch import nn
from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min


_SCALARS = (
    'log_energy', 'log_ecal_energy', 'log_hcal_energy', 'log_other_energy',
    'log_unknown_energy', 'log_calo_count', 'log_track_count',
    'log_mean_hit_energy', 'log_hit_energy_variance', 'log_max_hit_energy',
    'centroid_x_over_2000mm', 'centroid_y_over_2000mm', 'centroid_z_over_2000mm',
    'rms_x_over_2000mm', 'rms_y_over_2000mm', 'rms_z_over_2000mm',
    'span_x_over_2000mm', 'span_y_over_2000mm', 'span_z_over_2000mm',
    'log_nearest_track_distance_mm', 'log_nearest_track_momentum_GeV',
    'log_cluster_track_momentum_sum_GeV', 'log_cluster_track_momentum_max_GeV',
    'log_cluster_track_vector_magnitude_GeV', 'has_nearest_track',
    'log_energy_minus_log_nearest_track_momentum',
)
PC_FEATURE_SCHEMA = dict(version=1, embedding_pooling=['sum', 'mean', 'max'],
                         scalar_features=list(_SCALARS),
                         detector_ids=dict(ECAL=1, HCAL=2),
                         missing_detector='unknown_energy',
                         coordinate_units='mm', energy_units='GeV',
                         target='deposit_fraction_with_matched_track_accounting_v1',
                         score='untracked_truth_deposit_fraction_v1')


def add_arguments(parser):
    parser.add_argument('--particle-cluster-energy', action='store_true',
        help='Joint cluster-level calorimeter energy output in the shared five-head GNN')
    parser.add_argument('--particle-cluster-score', action='store_true',
        help='Joint additional-particle score for trackless reconstructed clusters')
    parser.add_argument('--particle-cluster-source', choices=('truth', 'predicted'), default='truth',
        help='Training membership; truth score training also uses artificial charged fragments')
    parser.add_argument('--particle-cluster-tbeta', type=float, default=0.9)
    parser.add_argument('--particle-cluster-td', type=float, default=0.5)
    parser.add_argument('--particle-cluster-score-weight', type=float, default=1.)


def enabled(args):
    return bool(getattr(args, 'particle_cluster_energy', False) or
                getattr(args, 'particle_cluster_score', False))


def validate_arguments(args):
    if not enabled(args):
        return
    if not (getattr(args, 'use_multihead_model', False) and
            getattr(args, 'five_particle_energy_heads', False) and
            getattr(args, 'pid_head', False)):
        raise ValueError('Particle cluster options require --use-multihead-model '
                         '--five-particle-energy-heads --pid-head')
    for flag in ('jit', 'dp', 'energy_branch', 'energy_regression_weight', 'cluster_energy_pooling'):
        if getattr(args, flag, False):
            raise ValueError('Particle cluster options do not support --' + flag.replace('_', '-'))
    from cluster_energy import validate_settings
    validate_settings(args.particle_cluster_source, args.particle_cluster_tbeta, args.particle_cluster_td)
    if not math.isfinite(args.particle_cluster_score_weight) or args.particle_cluster_score_weight < 0:
        raise ValueError('--particle-cluster-score-weight must be finite and nonnegative')


def _cluster_lookup(batch, assignments, cluster_batch, cluster_ids):
    """Map nodes, including tracks, to the calo-containing cluster table."""
    count = len(cluster_ids)
    keys = torch.cat((torch.stack((cluster_batch, cluster_ids), 1),
                      torch.stack((batch, assignments), 1)), 0)
    unique_keys, inverse = torch.unique(keys, sorted=True, dim=0, return_inverse=True)
    table = torch.full((len(unique_keys),),
                       -1, dtype=torch.long, device=batch.device)
    table[inverse[:count]] = torch.arange(count, device=batch.device)
    result = table[inverse[count:]]
    return torch.where(assignments > 0, result, torch.full_like(result, -1))


class ParticleClusterHead(nn.Module):
    """One shared pooled representation, optional energy and score projections."""
    def __init__(self, feature_dim=128, energy=False, score=False):
        super().__init__()
        if not (energy or score):
            raise ValueError('ParticleClusterHead requires energy or score')
        self.energy_enabled, self.score_enabled = bool(energy), bool(score)
        self.feature_dim = feature_dim
        self.shared = nn.Sequential(nn.Linear(3 * feature_dim + len(_SCALARS), 128),
                                    nn.ReLU(), nn.Linear(128, 64), nn.ReLU())
        if energy:
            self.energy_output = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
        if score:
            self.score_output = nn.Linear(64, 1)

    def forward(self, features, batch, assignments, raw_features, detector_ids=None):
        if raw_features is None or raw_features.ndim != 2 or raw_features.shape[1] < 10:
            raise ValueError('Particle cluster head requires raw detector features with at least 10 columns')
        if features.shape != (len(batch), self.feature_dim) or assignments.shape != batch.shape:
            raise ValueError('Particle cluster feature or membership shape mismatch')
        if len(raw_features) != len(batch) or not torch.isfinite(raw_features).all():
            raise ValueError('Invalid raw detector features in particle cluster head')
        raw = raw_features.detach().float()
        tracks = raw[:, 5] > .5
        selected = (~tracks) & (assignments > 0)
        # Group all nodes once. The selected group IDs retain the exact sorted
        # (event, cluster) order, while the same lookup also maps track nodes.
        all_keys, all_inverse = torch.unique(torch.stack((batch, assignments), 1),
                                            sorted=True, dim=0, return_inverse=True)
        selected_groups, inverse = torch.unique(all_inverse[selected], sorted=True,
                                                return_inverse=True)
        keys = all_keys[selected_groups]
        count = len(keys)
        group_table = torch.full((len(all_keys),), -1, dtype=torch.long, device=batch.device)
        group_table[selected_groups] = torch.arange(count, device=batch.device)
        node_clusters = group_table[all_inverse]
        h = features[selected].float()
        deposit = raw[:, 0].clamp_min(0)
        e = deposit[selected]
        n = scatter_add(e.new_ones(e.shape), inverse, dim_size=count)
        esum = scatter_add(e, inverse, dim_size=count)
        weights = torch.where(esum[inverse] > 0,
                              e / esum[inverse].clamp_min(1e-20), 1. / n[inverse])
        in_cluster = tracks & (node_clusters >= 0)
        trk_index = node_clusters[in_cluster]
        ntrack = scatter_add(raw.new_ones(trk_index.shape), trk_index, dim_size=count)
        momentum = raw[:, 7:10].norm(dim=1)
        track_sum = scatter_add(momentum[in_cluster], trk_index, dim_size=count)
        track_max = scatter_max(momentum[in_cluster], trk_index, dim_size=count)[0]
        track_vector = scatter_add(raw[in_cluster, 7:10], trk_index, dim=0, dim_size=count).norm(dim=1)
        if detector_ids is None:
            det = torch.zeros_like(batch)
        else:
            if detector_ids.shape != batch.shape or not torch.isfinite(detector_ids).all():
                raise ValueError('Invalid detector identity shape or values')
            det = detector_ids.detach().long()
        detectors = det[selected]
        ecal = scatter_add(e * (detectors == 1), inverse, dim_size=count)
        hcal = scatter_add(e * (detectors == 2), inverse, dim_size=count)
        unknown = scatter_add(e * (detectors == 0), inverse, dim_size=count)
        other = scatter_add(e * ((detectors != 0) & (detectors != 1) & (detectors != 2)), inverse, dim_size=count)
        mean_energy = esum / n.clamp_min(1)
        variance = scatter_mean((e - mean_energy[inverse]).square(), inverse, dim_size=count)
        maximum_energy = scatter_max(e, inverse, dim_size=count)[0]
        xyz = raw[selected, 1:4]
        centroid = scatter_add(xyz * weights[:, None], inverse, dim=0, dim_size=count)
        spatial_variance = scatter_add((xyz - centroid[inverse]).square() * weights[:, None],
                                       inverse, dim=0, dim_size=count)
        rms = spatial_variance.clamp_min(0).sqrt()
        extent = (scatter_max(xyz, inverse, dim=0, dim_size=count)[0] -
                  scatter_min(xyz, inverse, dim=0, dim_size=count)[0])
        nearest_distance, nearest_momentum, has_track = esum.new_zeros(count), esum.new_zeros(count), esum.new_zeros(count)
        # Event-local context. Chunk to avoid an unbounded clusters x tracks allocation.
        for event in torch.unique(keys[:, 0]):
            ci = torch.nonzero(keys[:, 0] == event, as_tuple=False).flatten()
            ti = torch.nonzero((batch == event) & tracks, as_tuple=False).flatten()
            if not ti.numel():
                continue
            for chunk in ci.split(256):
                distances = torch.cdist(centroid[chunk], raw[ti, 1:4])
                values, nearest = distances.min(dim=1)
                nearest_distance[chunk] = values
                nearest_momentum[chunk] = momentum[ti[nearest]]
                has_track[chunk] = 1
        scalars = torch.cat((torch.stack((esum.log1p(), ecal.log1p(), hcal.log1p(),
                         other.log1p(), unknown.log1p(), n.log1p(), ntrack.log1p(),
                         mean_energy.log1p(), variance.log1p(), maximum_energy.log1p()), 1),
                         centroid / 2000., rms / 2000., extent / 2000.,
                         torch.stack((nearest_distance.log1p(), nearest_momentum.log1p(),
                         track_sum.log1p(), track_max.log1p(), track_vector.log1p(), has_track,
                         esum.log1p() - nearest_momentum.log1p()), 1)), 1)
        pooled_features = torch.cat((scatter_add(h, inverse, dim=0, dim_size=count),
                        scatter_mean(h, inverse, dim=0, dim_size=count),
                        scatter_max(h, inverse, dim=0, dim_size=count)[0], scalars), 1)
        hidden = self.shared(pooled_features)
        indices = torch.full_like(batch, -1, dtype=torch.long)
        indices[selected] = inverse
        result = dict(cluster_batch=keys[:, 0], cluster_ids=keys[:, 1], assignments=assignments,
                      selected=selected, hit_cluster_index=indices, counts=n, track_counts=ntrack,
                      features=pooled_features, node_cluster_index=node_clusters)
        if self.energy_enabled:
            predicted = self.energy_output(hidden).squeeze(-1).float()
            hit_energy = deposit.new_zeros(len(batch))
            hit_energy[selected] = predicted[inverse] * weights
            # Connect empty batches/heads to DDP without artificial loss values.
            result.update(energy=predicted, hit_energy=hit_energy + predicted.sum() * 0)
        if self.score_enabled:
            result['score_logits'] = self.score_output(hidden).squeeze(-1).float()
        return result


@torch.no_grad()
def augment_trackless_assignments(assignments, batch, track, raw_features):
    """Make detached calo fragments from tracked clusters for score negatives.

    All original tracks remain in their original clusters. Their calorimeter
    hits become two new trackless groups, split on the widest physical axis.
    Single-hit showers form one fragment. Existing trackless clusters remain.
    """
    result = assignments.detach().long().clone()
    track = track.bool()
    for event in torch.unique(batch):
        event_mask = batch == event
        next_id = int(assignments[event_mask].max()) + 1 if event_mask.any() else 1
        clusters = torch.unique(assignments[event_mask & track & (assignments > 0)])
        for cluster in clusters:
            calo = torch.nonzero(event_mask & (~track) & (assignments == cluster), as_tuple=False).flatten()
            if not calo.numel():
                continue
            xyz = raw_features[calo, 1:4].detach()
            axis = (xyz.max(0)[0] - xyz.min(0)[0]).argmax()
            ordered = calo[torch.argsort(xyz[:, axis])]
            split = max(1, len(ordered) // 2)
            result[ordered[:split]] = next_id
            next_id += 1
            if split < len(ordered):
                result[ordered[split:]] = next_id
                next_id += 1
    return result


def training_readout(head, features, clustering, batch, track_mask, raw_features,
                     detector_ids=None, truth_ids=None, source='truth', tbeta=.9,
                     td=.5, coordinate_start=1):
    from cluster_energy import cluster_membership
    assignments = cluster_membership(clustering, batch, track_mask, source,
        truth_ids=truth_ids, tbeta=tbeta, td=td, coordinate_start=coordinate_start)
    pooled = head(features, batch, assignments, raw_features, detector_ids)
    if source == 'truth' and head.score_enabled:
        augmented = augment_trackless_assignments(assignments, batch, track_mask, raw_features)
        pooled['score_augmented'] = head(features, batch, augmented, raw_features, detector_ids)
    return pooled


@torch.no_grad()
def cluster_targets(pooled, data, need_score=True):
    """Energy and score targets with explicit track-based double-count prevention.

    Truth energy is distributed over ALL calo hits of each truth object, including
    unassigned hits. A charged contribution is suppressed only when a matching
    input track exists elsewhere, outside the candidate cluster. Neutral energy
    and charged energy without an input track remain. For score, charged truth
    with any matching input track is not an additional independent particle.

    Mixed clusters receive a deposited-energy-weighted independent fraction as a
    soft score target. Unknown truth makes the whole cluster unsupervised. These
    targets cannot resolve parent/daughter double counting not encoded in labels.
    """
    from particle_heads import species_ids
    batch = data.batch
    truth_ids = data.y[:, 0].long()
    tracks = data.feat[:, 5] > .5
    count = len(pooled['cluster_ids'])
    deposit = data.feat[:, 0].detach().float().clamp_min(0)
    teacher_valid = getattr(data, 'truth_valid', None)
    teacher_valid = (torch.ones_like(batch, dtype=torch.bool) if teacher_valid is None
                     else teacher_valid.bool())
    species = species_ids(data.label[:, 2], data.label[:, 3])
    known = teacher_valid & (truth_ids > 0) & (species >= 0)
    truth_keys, inverse = torch.unique(torch.stack((batch[known], truth_ids[known]), 1),
                                       sorted=True, dim=0, return_inverse=True)
    ntruth = len(truth_keys)
    truth_index = torch.full_like(batch, -1, dtype=torch.long)
    truth_index[known] = inverse
    target_energy = data.label[:, 4:8].detach().float().square().sum(1).sqrt()
    if not torch.isfinite(target_energy[known]).all():
        raise ValueError('Non-finite labelled energy in particle cluster targets')
    truth_e = scatter_max(target_energy[known], inverse, dim_size=ntruth)[0]
    ntrack = scatter_add(tracks[known].long(), inverse, dim_size=ntruth)
    truth_track = torch.zeros_like(batch, dtype=torch.bool)
    truth_track[known] = ntrack[inverse] > 0
    charged = species < 3
    selected_calo = known & (~tracks)
    truth_calo_index = truth_index[selected_calo]
    denominator = scatter_add(deposit[selected_calo], truth_calo_index, dim_size=ntruth)
    nhit = scatter_add(deposit.new_ones(truth_calo_index.shape), truth_calo_index, dim_size=ntruth)
    fraction = deposit.new_zeros(len(batch))
    fraction[selected_calo] = torch.where(denominator[truth_calo_index] > 0,
        deposit[selected_calo] / denominator[truth_calo_index].clamp_min(1e-20),
        1. / nhit[truth_calo_index].clamp_min(1))
    # Determine whether each calorimeter hit's matching truth track is in C.
    node_cluster = pooled.get('node_cluster_index')
    if node_cluster is None:
        node_cluster = _cluster_lookup(batch, pooled['assignments'], pooled['cluster_batch'], pooled['cluster_ids'])
    matching_track = tracks & known & (node_cluster >= 0)
    pair_stride = count + 1
    track_pairs = torch.unique(truth_index[matching_track] * pair_stride + node_cluster[matching_track])
    has_track_inside = torch.zeros_like(batch, dtype=torch.bool)
    can_match = selected_calo & (node_cluster >= 0)
    pairs = truth_index[can_match] * pair_stride + node_cluster[can_match]
    if track_pairs.numel():
        positions = torch.searchsorted(track_pairs, pairs)
        found = positions < len(track_pairs)
        safe = positions.clamp_max(len(track_pairs) - 1)
        has_track_inside[can_match] = found & (track_pairs[safe] == pairs)
    allowed = known & ((~charged) | (~truth_track) | has_track_inside)
    hit_target = deposit.new_zeros(len(batch))
    hit_target[selected_calo] = fraction[selected_calo] * truth_e[truth_calo_index]
    selected = pooled['selected']
    selected_index = pooled['hit_cluster_index'][selected]
    energy = scatter_add((hit_target * allowed)[selected], selected_index, dim_size=count)
    score = None
    if need_score:
        # Deposited independent-signal fraction; use equal weights at zero deposit.
        independent = known & ((~charged) | (~truth_track))
        counts = pooled['counts']
        esum = scatter_add(deposit[selected], selected_index, dim_size=count)
        independent_sum = scatter_add((deposit * independent)[selected], selected_index, dim_size=count)
        independent_count = scatter_add(independent[selected].float(), selected_index, dim_size=count)
        score = torch.where(esum > 0, independent_sum / esum.clamp_min(1e-20),
                            independent_count / counts.clamp_min(1)).clamp(0, 1)
    unknown = (~teacher_valid) | ((truth_ids > 0) & (species < 0))
    nunknown = scatter_add(unknown[selected].long(), selected_index, dim_size=count)
    valid = nunknown == 0
    return dict(energy=energy, score=score, valid=valid,
                score_valid=valid & (pooled['track_counts'] == 0))
