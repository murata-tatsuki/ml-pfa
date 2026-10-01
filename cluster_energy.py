"""Optional cluster-level calorimeter regression, separate from legacy paths.

Clustering membership is discrete/detached. Pooling, the energy MLP and the
shared hit encoder are differentiable. Truth labels never enter predicted
membership or the energy head; they are used only to construct training targets.
"""
import numpy as np
import torch
from torch import nn
from torch_scatter import scatter_add, scatter_max, scatter_mean


POOLED_LOSSES = ('sum', 'sum_log', 'sum_log_perCluster',
                 'log_ratio_mse', 'log_scaled_relative')


def validate_settings(source, tbeta, td):
    if source not in ('truth', 'predicted'):
        raise ValueError('cluster energy source must be truth or predicted')
    if not np.isfinite(tbeta) or not 0 <= tbeta <= 1:
        raise ValueError('cluster energy tbeta must be in [0, 1]')
    if not np.isfinite(td) or td <= 0:
        raise ValueError('cluster energy td must be finite and positive')


@torch.no_grad()
def cluster_membership(clustering, batch, track_mask, source, truth_ids=None,
                       tbeta=0.7, td=0.5, coordinate_start=1):
    """Return event-local positive IDs; 0 denotes noise/unassigned hits."""
    validate_settings(source, tbeta, td)
    if source == 'truth':
        if truth_ids is None:
            raise ValueError('Truth pooling requires explicit truth_cluster_index')
        if truth_ids.shape != batch.shape:
            raise ValueError('truth_cluster_index and batch must have identical shapes')
        return truth_ids.detach().long().clamp_min(0)

    # This is the same membership implementation used by clustering.cluster.
    # Import lazily to avoid a circular import through gravnet_model.
    from objectcondensation import get_clustering_np_new
    out = torch.zeros_like(batch, dtype=torch.long)
    beta = clustering[:, 0].detach().float().sigmoid().cpu().numpy()
    coords = clustering[:, coordinate_start:].detach().float().cpu().numpy()
    tracks = track_mask.detach().cpu().numpy()
    batches = batch.detach().cpu().numpy()
    if not np.isfinite(beta).all() or not np.isfinite(coords).all():
        raise ValueError('Non-finite beta or coordinates in predicted pooling')
    for event_id in np.unique(batches):
        positions = np.flatnonzero(batches == event_id)
        if len(positions) == 1:
            # The legacy singleton-track merge needs at least two clusters.
            labels = np.zeros(1, dtype=np.int64)
        else:
            labels, _ = get_clustering_np_new(
                event=None, betas=beta[positions], X=coords[positions],
                charged_hits=tracks[positions], tbeta=tbeta, td=td)
        out[torch.as_tensor(positions, device=batch.device)] = torch.as_tensor(
            labels + 1, device=batch.device, dtype=torch.long)
    return out


class ClusterEnergyHead(nn.Module):
    """Pool calorimeter hit embeddings before a nonlinear cluster energy MLP."""
    def __init__(self, feature_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * feature_dim + 2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus())

    def forward(self, features, batch, assignments, track_mask, detected_energy):
        if detected_energy is None or detected_energy.shape != batch.shape:
            raise ValueError('Cluster energy pooling requires raw detected_energy per hit')
        energy = detected_energy.detach().float()
        if not torch.isfinite(energy).all():
            raise ValueError('Non-finite detected energy in cluster pooling')
        energy = energy.clamp_min(0)
        selected = (~track_mask.bool()) & (assignments > 0)
        keys, inverse = torch.unique(
            torch.stack((batch[selected], assignments[selected]), dim=1),
            dim=0, sorted=True, return_inverse=True)
        n_clusters = keys.shape[0]
        h = features[selected].float()
        count = scatter_add(h.new_ones(h.shape[0]), inverse, dim_size=n_clusters)
        energy_sum = scatter_add(energy[selected], inverse, dim_size=n_clusters)
        summed = scatter_add(h, inverse, dim=0, dim_size=n_clusters)
        mean = scatter_mean(h, inverse, dim=0, dim_size=n_clusters)
        maximum = scatter_max(h, inverse, dim=0, dim_size=n_clusters)[0]
        pooled = torch.cat((summed, mean, maximum,
                            energy_sum.log1p()[:, None], count.log1p()[:, None]), dim=1)
        predicted = self.mlp(pooled).squeeze(-1).float()
        # Compatibility view only: existing ROOT writers sum per-hit values.
        # The head predicts ONE energy per cluster, not per-hit corrections.
        denominator = torch.where(energy_sum > 0, energy_sum, torch.ones_like(energy_sum))
        fraction = energy[selected] / denominator[inverse]
        fraction = torch.where(energy_sum[inverse] > 0, fraction, 1. / count[inverse])
        hit_energy = energy.new_zeros(energy.shape)
        hit_energy[selected] = predicted[inverse] * fraction
        # Keep a differentiable zero even for batches without calorimeter hits.
        hit_energy = hit_energy + predicted.sum() * 0
        hit_cluster_index = torch.full_like(batch, -1, dtype=torch.long)
        hit_cluster_index[selected] = inverse
        return dict(energy=predicted, hit_energy=hit_energy,
                    assignments=assignments, hit_cluster_index=hit_cluster_index,
                    cluster_batch=keys[:, 0], cluster_ids=keys[:, 1],
                    selected=selected, counts=count)


@torch.no_grad()
def deposited_energy_targets(pooled, truth_ids, truth_energy, detected_energy,
                             batch, track_mask, truth_valid=None):
    """Allocate each truth energy by its deposited-energy fractions.

The denominator includes ALL calorimeter hits of the truth particle, including
unassigned hits. Splitting therefore does not duplicate the full truth energy.
Noise contributes zero. A zero-deposit truth particle uses equal hit fractions.
"""
    truth_energy = truth_energy.detach().float()
    energy = detected_energy.detach().float().clamp_min(0)
    known = truth_ids > 0 if truth_valid is None else truth_valid.bool() & (truth_ids > 0)
    if not torch.isfinite(truth_energy[known]).all() or not torch.isfinite(energy).all():
        raise ValueError('Non-finite energy in pooled regression target')
    signal = (~track_mask.bool()) & known
    _, truth_index = torch.unique(torch.stack((batch[signal], truth_ids[signal]), dim=1),
                                  dim=0, return_inverse=True)
    n_truth = int(truth_index.max().item()) + 1 if truth_index.numel() else 0
    deposit_sum = scatter_add(energy[signal], truth_index, dim_size=n_truth)
    truth_e = scatter_max(truth_energy[signal], truth_index, dim_size=n_truth)[0]
    counts = scatter_add(energy.new_ones(truth_index.shape), truth_index, dim_size=n_truth)
    denominator = torch.where(deposit_sum > 0, deposit_sum, torch.ones_like(deposit_sum))
    fractions = energy[signal] / denominator[truth_index]
    fractions = torch.where(deposit_sum[truth_index] > 0, fractions, 1. / counts[truth_index])
    hit_target = energy.new_zeros(energy.shape)
    hit_target[signal] = fractions * truth_e[truth_index]
    selected = pooled['selected']
    return scatter_add(hit_target[selected], pooled['hit_cluster_index'][selected],
                       dim_size=pooled['energy'].numel())


def pooled_supervision_mask(pooled, truth_valid=None):
    """A cluster containing any unlabelled calo hit has no complete energy target."""
    if truth_valid is None:
        return torch.ones_like(pooled['energy'], dtype=torch.bool)
    selected = pooled['selected']
    unknown_count = scatter_add((~truth_valid[selected].bool()).long(),
        pooled['hit_cluster_index'][selected], dim_size=pooled['energy'].numel())
    return unknown_count == 0


def pooled_energy_loss(pooled, truth_ids, truth_energy, detected_energy, batch,
                       track_mask, loss_name, epsilon=1e-3, truth_valid=None):
    if loss_name not in POOLED_LOSSES:
        raise ValueError(f'Pooled energy supports {POOLED_LOSSES}; got {loss_name}')
    valid = pooled_supervision_mask(pooled, truth_valid)
    predicted = pooled['energy'][valid].float()
    target = deposited_energy_targets(pooled, truth_ids, truth_energy, detected_energy,
                                      batch, track_mask, truth_valid)[valid]
    if loss_name == 'sum':
        # Preserve calc_L_E's existing LE_cluster_coef for the 'sum' mode.
        loss = 5 * (predicted - target).square()
    elif loss_name in ('sum_log', 'sum_log_perCluster'):
        loss = (predicted - target).abs().log1p()
    else:
        from objectcondensation import scale_invariant_energy_loss
        loss = scale_invariant_energy_loss(predicted, target, loss_name, epsilon)
    batch_size = int(batch.max().item()) + 1 if batch.numel() else 1
    if loss_name in ('sum_log_perCluster', 'log_ratio_mse', 'log_scaled_relative'):
        counts = scatter_add(loss.new_ones(loss.shape), pooled['cluster_batch'][valid], dim_size=batch_size)
        by_event = scatter_add(loss, pooled['cluster_batch'][valid], dim_size=batch_size)
        return (by_event / counts.clamp_min(1)).sum() / batch_size
    return loss.sum() / batch_size


def add_training_arguments(parser):
    parser.add_argument('--cluster-energy-pooling', action='store_true',
                        help='Pool calorimeter features per cluster before the energy MLP (multihead only)')
    parser.add_argument('--cluster-energy-source', choices=['truth', 'predicted'], default='truth',
                        help='Membership used for energy pooling during training and validation')
    parser.add_argument('--cluster-energy-tbeta', type=float, default=0.7)
    parser.add_argument('--cluster-energy-td', type=float, default=0.5)


def validate_training_arguments(args):
    if not getattr(args, 'cluster_energy_pooling', False):
        return
    if not (args.use_multihead_model and args.energy_regression and args.energy_regression_cluster):
        raise ValueError('--cluster-energy-pooling requires --use-multihead-model '
                         '--energy-regression --energy-regression-cluster')
    if args.LE_cluster not in POOLED_LOSSES:
        raise ValueError('--cluster-energy-pooling requires --LE-cluster in ' + ', '.join(POOLED_LOSSES))
    if args.jit or args.dp or args.energy_branch or args.energy_regression_weight:
        raise ValueError('Cluster energy pooling supports eager single-device or DDP multihead models; '
                         'jit, dp, energy-branch and energy-regression-weight are not supported with it')
    validate_settings(args.cluster_energy_source, args.cluster_energy_tbeta, args.cluster_energy_td)


def training_forward(model, data, args, epoch=None):
    if args.use_multihead_model:
        kwargs = dict(epoch=epoch, return_dict=True)
        if getattr(args, 'cluster_energy_pooling', False):
            kwargs.update(truth_cluster_index=data.y[:, 0], detected_energy=data.feat[:, 0])
        return model(data.x, data.batch, **kwargs)
    return model(data.x, data.batch)


def replace_calo_loss(legacy_energy_loss, components, pooled, data, args, coefficient):
    """Legacy loss was called with Ecl_regression=False; add the pooled term."""
    if not getattr(args, 'cluster_energy_pooling', False):
        return legacy_energy_loss, components
    if pooled is None:
        raise ValueError('Cluster energy pooling was enabled but forward returned no cluster energies')
    truth_energy = data.label[:, 4:8].float().square().sum(dim=1).sqrt()
    calo_loss = pooled_energy_loss(
        pooled, data.y[:, 0], truth_energy, data.feat[:, 0], data.batch,
        data.x[:, 4] > 0.5, args.LE_cluster, args.epsilon,
        truth_valid=getattr(data, 'truth_valid', None)) * coefficient
    result = legacy_energy_loss + calo_loss
    components = dict(components)
    components['L_E_cluster'] = calo_loss.detach()
    components['L_E'] = result.detach()
    return result, components


def pooling_model(model):
    while hasattr(model, 'module') or hasattr(model, 'model'):
        model = model.module if hasattr(model, 'module') else model.model
    return model if getattr(model, 'cluster_energy_pooling', False) else None


def checkpoint_payload(model, epoch=None, args=None):
    raw = model.module if hasattr(model, 'module') else model
    result = dict(model=raw.state_dict())
    from particle_heads import checkpoint_config
    config = checkpoint_config(raw, epoch)
    if config is not None:
        if args is not None:
            config.update(pid_loss_weight=getattr(args, 'pid_loss_weight', 1.),
                          epochs_noPID=getattr(args, 'epochs_noPID', -1))
        result['particle_heads_config'] = config
    if args is not None and getattr(args, 'extended_h5_input', False):
        result['training_input_config'] = dict(schema='pandora-eval-1',
            exclude_gap_hits=args.exclude_gap_hits, unknown_truth='input_only',
            unknown_calo_cluster_loss='skip_incomplete_cluster')
    pooled = pooling_model(raw)
    if pooled is not None:
        result['cluster_energy_config'] = dict(version=1, source=pooled.cluster_energy_source,
            tbeta=pooled.cluster_energy_tbeta, td=pooled.cluster_energy_td,
            coordinate_start=pooled.cluster_energy_coordinate_start,
            interaction_start_epoch=pooled.interaction_start_epoch,
            inference_epoch=epoch if epoch is not None else pooled.interaction_start_epoch,
            target='truth_energy_deposit_fraction')
    return result


def inference_forward(model, data):
    pooled = pooling_model(model)
    if pooled is None:
        return model(data.x, data.batch)
    kwargs = dict(detected_energy=data.feat[:, 0])
    if pooled.cluster_energy_source == 'truth':
        if not hasattr(data, 'y') or data.y is None:
            raise ValueError('Explicit truth pooling requires truth labels')
        kwargs['truth_cluster_index'] = data.y[:, 0]
    return model(data.x, data.batch, **kwargs)


def configure_inference(model, source=None, tbeta=None, td=None):
    pooled = pooling_model(model)
    if pooled is None:
        return
    source = pooled.cluster_energy_source if source is None else source
    tbeta = pooled.cluster_energy_tbeta if tbeta is None else tbeta
    td = pooled.cluster_energy_td if td is None else td
    validate_settings(source, tbeta, td)
    pooled.cluster_energy_source = source
    pooled.cluster_energy_tbeta = tbeta
    pooled.cluster_energy_td = td


def add_inference_arguments(parser):
    parser.add_argument('--cluster-energy-source', choices=['truth', 'predicted'], default=None,
                        help='Pooling for a cluster-energy checkpoint; default follows --truth-clustering, '
                             'otherwise predicted. Ignored by legacy checkpoints.')


def inference_loader_kwargs(args):
    return dict(cluster_energy_source=getattr(args, 'cluster_energy_source', None) or
                ('truth' if getattr(args, 'truth_clustering', False) else 'predicted'),
                cluster_energy_tbeta=args.tbeta, cluster_energy_td=args.td)
