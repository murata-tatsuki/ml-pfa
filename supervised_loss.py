"""Mask unavailable teachers after forward, preserving unlabelled model inputs."""
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


def calc_supervised_loss(beta, coords, charged, truth_ids, truth_energy, batch,
                         truth_valid=None, **kwargs):
    """Unlabelled rows are neither objects nor noise nor repulsion targets.

Use the legacy OC/track loss unchanged on labelled outputs. Events without
labels contribute zero, while normalization still uses the full event batch.
This slices predictions AFTER the encoder; unknown inputs remain in its graph.
"""
    if truth_valid is None:
        return oc.calc_LV_Lbeta(beta, coords, charged, truth_ids, truth_energy, batch, **kwargs)
    if truth_valid.shape != batch.shape:
        raise ValueError('truth_valid must have one flag per model input')
    mask = truth_valid.bool()
    if torch.any(truth_ids[mask] <= 0):
        raise ValueError('Known extended-H5 truth must have a positive cluster ID')
    if bool(mask.all()) and mask.numel():
        return oc.calc_LV_Lbeta(beta, coords, charged, truth_ids, truth_energy, batch, **kwargs)
    if not bool(mask.any()):
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
    events, selected_batch = torch.unique(batch[mask], sorted=True, return_inverse=True)
    # Preserve the original batch normalization even when some events have no labels.
    n_events = int(batch.max().item()) + 1
    scale = events.numel() / n_events
    result = oc.calc_LV_Lbeta(beta[mask], coords[mask], None if charged is None else charged[mask],
        truth_ids[mask], truth_energy[mask], selected_batch, **selected_kwargs)
    return tuple(x * scale for x in result[:4]) + ({k: v * scale for k, v in result[4].items()},)
