"""Mask unavailable teachers after forward, preserving unlabelled model inputs."""
import numpy as np
import torch

import objectcondensation as oc


COMPONENTS = (
    'L_V', 'L_V_attractive', 'L_V_attractive_charged', 'L_V_attractive_neutral',
    'L_V_repulsive', 'L_V_repulsive_charged', 'L_V_repulsive_neutral',
    'L_charged_cluster', 'L_beta', 'L_beta_noise', 'L_beta_sig', 'L_beta_suppress',
    'L_beta_track', 'L_E', 'L_E_charge', 'L_E_cond', 'L_E_cluster',
    'L_E_w_photon', 'L_E_w_charged_hadron', 'L_E_w_neutral_hadron',
    'L_E_w_muon', 'L_E_w_electron')
HIT_ARGUMENTS = ('cluster_track_index', 'tracker_energy', 'detected_energy',
    'pred_cluster_energy', 'mcpdg', 'mccharge', 'weight_photon',
    'weight_charged_hadron', 'weight_neutral_hadron', 'weight_muon', 'weight_electron')
PREDICTIONS = ('tracker_energy', 'pred_cluster_energy', 'weight_photon',
    'weight_charged_hadron', 'weight_neutral_hadron', 'weight_muon', 'weight_electron')


def prepare_supervised_metadata(data):
    """Build batch-local truth metadata before H2D, without removing model inputs."""
    batch, truth_ids = data.batch, data.y[:, 0].long()
    track = data.y[:, 1]
    if any(t.device.type != 'cpu' for t in (batch, truth_ids, track)):
        raise ValueError('Supervised metadata must be prepared before data.to(device)')
    valid = getattr(data, 'truth_valid', None)
    selected, scale = None, 1.
    if valid is not None:
        if valid.device.type != 'cpu' or valid.shape != batch.shape:
            raise ValueError('truth_valid must be a CPU flag per model input')
        mask = valid.numpy().astype(bool, copy=False)
        if np.any(truth_ids.numpy()[mask] <= 0):
            raise ValueError('Known extended-H5 truth must have a positive cluster ID')
        if not np.any(mask):
            return dict(mode='empty', n_hits=batch.numel())
        if not np.all(mask):
            positions = np.flatnonzero(mask)
            selected = torch.from_numpy(positions)
            selected_events = batch.numpy()[positions]
            if np.all(selected_events[1:] >= selected_events[:-1]):
                starts = np.r_[True, selected_events[1:] != selected_events[:-1]]
                inverse = starts.cumsum(dtype=np.int64) - 1
                n_events = int(starts.sum())
            else:
                events, inverse = np.unique(selected_events, return_inverse=True)
                n_events = len(events)
            selected_batch = torch.from_numpy(inverse)
            scale = n_events / (int(batch.numpy().max()) + 1)
    # The legacy unmasked loss does not accept empty batches. Keep its fallback.
    if not batch.numel():
        return None
    loss_batch = batch if selected is None else selected_batch
    loss_ids = truth_ids if selected is None else torch.from_numpy(truth_ids.numpy()[positions])
    loss_track = track if selected is None else torch.from_numpy(track.numpy()[positions])
    return dict(mode='all' if selected is None else 'partial', n_hits=batch.numel(),
                selected=selected, batch=None if selected is None else loss_batch, scale=scale,
                oc=oc.prepare_oc_metadata(loss_ids, loss_batch, loss_track))


def calc_supervised_loss(beta, coords, charged, truth_ids, truth_energy, batch,
                         truth_valid=None, truth_metadata=None, **kwargs):
    """Unlabelled rows are neither objects nor noise nor repulsion targets.

Use the legacy OC/track loss unchanged on labelled outputs. Events without
labels contribute zero, while normalization still uses the full event batch.
This slices predictions AFTER the encoder; unknown inputs remain in its graph.
"""
    if truth_metadata is not None:
        if truth_metadata['n_hits'] != batch.numel():
            raise ValueError('Supervised metadata does not match input shape')
        mode = truth_metadata['mode']
        if mode == 'all':
            return oc.calc_LV_Lbeta(beta, coords, charged, truth_ids, truth_energy, batch,
                                   oc_metadata=truth_metadata['oc'], **kwargs)
        if mode == 'partial':
            mask = truth_metadata['selected']
            selected_batch = truth_metadata['batch']
            scale = truth_metadata['scale']
        elif mode != 'empty':
            raise ValueError('Invalid supervised metadata mode')
    else:
        if truth_valid is None:
            return oc.calc_LV_Lbeta(beta, coords, charged, truth_ids, truth_energy, batch, **kwargs)
        if truth_valid.shape != batch.shape:
            raise ValueError('truth_valid must have one flag per model input')
        mask = truth_valid.bool()
        if torch.any(truth_ids[mask] <= 0):
            raise ValueError('Known extended-H5 truth must have a positive cluster ID')
        if bool(mask.all()) and mask.numel():
            return oc.calc_LV_Lbeta(beta, coords, charged, truth_ids, truth_energy, batch, **kwargs)
        mode = 'partial' if bool(mask.any()) else 'empty'
    if mode == 'empty':
        # An empty slice provides an autograd connection without reading any
        # unavailable teacher or prediction value, including on an empty rank.
        predicted = [beta, coords, charged] + [kwargs.get(k) for k in PREDICTIONS]
        zero = sum(t.reshape(-1)[:0].sum() for t in predicted if t is not None)
        keys = COMPONENTS + (('L_beta_norms_term', 'L_beta_logbeta_term')
                             if kwargs.get('beta_term_option') == 'short-range-potential' else ())
        return zero, zero, zero, zero, {k: zero.detach().clone() for k in keys}

    selected_kwargs = dict(kwargs)
    for name in HIT_ARGUMENTS:
        value = kwargs.get(name)
        if value is not None:
            selected_kwargs[name] = value[mask]
    if truth_metadata is None:
        events, selected_batch = torch.unique(batch[mask], sorted=True, return_inverse=True)
        # Preserve normalization including events without any labels.
        n_events = int(batch.max().item()) + 1
        scale = events.numel() / n_events
    else:
        selected_kwargs['oc_metadata'] = truth_metadata['oc']
    result = oc.calc_LV_Lbeta(beta[mask], coords[mask], None if charged is None else charged[mask],
        truth_ids[mask], truth_energy[mask], selected_batch, **selected_kwargs)
    return tuple(x * scale for x in result[:4]) + ({k: v * scale for k, v in result[4].items()},)
