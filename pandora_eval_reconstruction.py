"""Reconstruction and labelled-overlap diagnostics for the common H5 inputs."""
from types import SimpleNamespace
import numpy as np
import torch
from cluster_energy import inference_forward, configure_inference, pooling_model
from clustering import cluster
from prediction import Prediction


def configure_regression_output(model, activation):
    """Explicit compatibility with multihead checkpoints trained before Softplus.

    Parameter-free activations are absent from state_dict, so loading weights
    cannot infer this choice. Do not guess it from the checkpoint filename.
    """
    if activation == 'current':
        return
    if activation != 'linear':
        raise ValueError(f'Unknown regression output activation: {activation}')
    raw = model.model if hasattr(model, 'model') else model
    if not hasattr(raw, 'head_output') or not hasattr(raw, 'head_specs'):
        raise ValueError('Linear regression compatibility requires a multihead model')
    if getattr(raw, 'cluster_energy_pooling', False):
        raise ValueError('Linear compatibility is for legacy per-hit regression, not pooled heads')
    for i, spec in enumerate(raw.head_specs):
        if spec['kind'] != 'regression':
            continue
        head = raw.head_output[i]
        layers = list(head.children())
        if layers and isinstance(layers[-1], torch.nn.Softplus):
            replacement = torch.nn.Sequential(*layers[:-1])
            replacement.train(head.training)
            raw.head_output[i] = replacement
        elif not layers or not isinstance(layers[-1], torch.nn.Linear):
            raise ValueError('Unexpected regression head; cannot restore linear output')


def predict(model, data, device, tbeta, td, calo_head=True, truth=False):
    configure_inference(model, source='truth' if truth else 'predicted', tbeta=tbeta, td=td)
    data = data.clone().to(device)
    data.batch = torch.zeros(len(data.x), dtype=torch.long, device=device)
    with torch.no_grad():
        output = inference_forward(model, data).detach().cpu().numpy()
    expected_minimum = 4 if calo_head else 3
    if output.ndim != 2 or len(output) != len(data.x) or output.shape[1] < expected_minimum:
        raise ValueError(f'Unexpected model output shape: {output.shape}')
    if not np.isfinite(output).all():
        raise ValueError('Non-finite model output')
    beta = torch.sigmoid(torch.from_numpy(output[:, 0])).numpy()
    tracker = output[:, 1]
    calo = output[:, 2] if calo_head else None
    coordinates = output[:, 3:] if calo_head else output[:, 2:]
    track = data.x[:, 4].detach().cpu().numpy()
    if truth:
        assignments = data.y[:, 0].detach().cpu().numpy().astype(np.int64)
    elif len(beta) == 1:
        assignments = np.ones(1, dtype=np.int64)
    else:
        prediction = Prediction(beta, coordinates, None, track, tracker, calo)
        assignments, _ = cluster(SimpleNamespace(), prediction, tbeta, td)
    return dict(beta=beta, tracker=tracker, calo=calo, assignments=assignments)


def energy_clusters(data, prediction, policy='alpha'):
    """One energy per cluster; no truth momentum or particle type is used.

    alpha: original event writer's highest-beta seed rule.
    any-track: highest-beta track's regressed energy if any track is present.
    Both values are saved so the distinction cannot silently vary by plot.
    """
    assignments = prediction['assignments']
    track = data.x[:, 4].numpy() > .5
    result = []
    for ci in np.unique(assignments):
        if ci <= 0:
            continue
        indices = np.flatnonzero(assignments == ci)
        seed = indices[np.argsort(-prediction['beta'][indices])[0]]
        tracks = indices[track[indices]]
        # Compatibility with existing per-hit energy heads: sum the same
        # column as the old ROOT writer. Pooled heads produce zeros on tracks.
        calorimeter = (float(np.sum(prediction['calo'][indices], dtype=np.float64))
                       if prediction['calo'] is not None else float(prediction['tracker'][seed]))
        alpha = float(prediction['tracker'][seed]) if track[seed] else calorimeter
        any_track = float(prediction['tracker'][tracks[np.argsort(-prediction['beta'][tracks])[0]]]) if len(tracks) else calorimeter
        result.append(dict(cluster=int(ci), energy=alpha if policy == 'alpha' else any_track,
            energy_alpha=alpha, energy_any_track=any_track, n_tracks=len(tracks),
            n_inputs=len(indices), seed_input_row=int(data.input_row[seed]),
            members=set(map(int, data.input_row[indices].numpy()))))
    return result


def pandora_clusters(event):
    members = {int(p[0]): set() for p in event['pfo']}
    for link in event['pfo_links']:
        members[int(link[0])].add(int(link[3]))
    return [dict(cluster=int(p[0]), energy=float(p[2]), energy_alpha=float(p[2]),
                 energy_any_track=float(p[2]), n_tracks=int(p[7]),
                 n_inputs=len(members[int(p[0])]), seed_input_row=-1,
                 members=members[int(p[0])]) for p in event['pfo']]


def overlap_metrics(event, clusters, selected_rows):
    """Best deposited-overlap match per truth label (not a bijective matching).

    Only labelled model-input rows enter either denominator. Unknown truth
    contributes to a separately stored coverage diagnostic, never a fake MC.
    Track-only truth objects use hit overlap to find a match, but energy-based
    efficiency/purity remain NaN when their denominator is zero.
    """
    selected = np.zeros(len(event['feature']), dtype=bool)
    selected[np.asarray(selected_rows, dtype=np.int64)] = True
    known = selected & (event['row_info'][:, 5] > 0)
    labels = event['label'][:, 1].astype(np.int64)
    edep = np.where(event['row_info'][:, 0] == 0, event['feature'][:, 0], 0.)
    # Avoid 0*NaN for rows excluded from the evaluation input set.
    edep = np.where(known & np.isfinite(edep), edep, 0.)
    truth = {int(t[0]): t for t in event['truth_particles']}
    cluster_info = []
    for c in clusters:
        members = np.array(sorted(c['members']), dtype=np.int64)
        labelled = members[known[members]]
        cluster_info.append((c, labelled, float(edep[labelled].sum())))
    for label in np.unique(labels[known]):
        rows = np.flatnonzero(known & (labels == label))
        denom = float(edep[rows].sum())
        best, best_score = None, (0., 0)
        for c, members, cluster_edep in cluster_info:
            overlap = members[labels[members] == label]
            score = (float(edep[overlap].sum()), len(overlap))
            if score > best_score:
                best_score = score
                best = c, cluster_edep
        t = truth[int(label)]
        c, reco_denom = best if best is not None else (None, 0.)
        yield dict(truth_label=int(label), truth_pdg=int(t[2]), truth_energy=float(t[3]),
            matched=int(c is not None), cluster=c['cluster'] if c else -1,
            reco_energy=c['energy'] if c else float('nan'), overlap_energy=best_score[0],
            truth_deposit=denom, reco_known_deposit=reco_denom,
            efficiency=best_score[0]/denom if denom > 0 else float('nan'),
            purity=best_score[0]/reco_denom if reco_denom > 0 else float('nan'),
            truth_n_inputs=len(rows), overlap_n_inputs=best_score[1])
