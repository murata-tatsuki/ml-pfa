"""Opt-in node PID and five species energy losses; legacy paths are untouched."""
import math
import torch
from torch.nn import functional as F
from torch_scatter import scatter_add, scatter_max, scatter_min

SPECIES = ('charged_hadron', 'electron', 'muon', 'photon', 'neutral_hadron')
CHARGED_MASSES_GEV = (0.13957039, 0.00051099895, 0.1056583755)


def add_pid_readout_argument(parser):
    parser.add_argument('--pid-readout', choices=('auto', 'representative'), default='auto',
        help='representative: require a trained PID head; retain beta/coordinate clustering '
             'and use the highest-beta track, or highest-beta node without tracks. '
             'auto uses this readout when the checkpoint has PID.')


def resolve_pid_readout(model, requested='auto'):
    if requested not in ('auto', 'representative'):
        raise ValueError('Unknown --pid-readout: ' + str(requested))
    raw = raw_model(model) if model is not None else None
    pid = bool(getattr(raw, 'pid_head', False))
    if requested == 'representative' and not pid:
        raise ValueError('--pid-readout representative requires a checkpoint with a trained PID head')
    if getattr(raw, 'five_particle_energy_heads', False) and not pid:
        raise ValueError('Normal five-head inference requires PID; use diagnose_particle_heads.py for truth-PID diagnosis')
    return 'representative' if pid else 'none'


def enabled(args):
    return bool(getattr(args, 'pid_head', False) or getattr(args, 'five_particle_energy_heads', False))


def add_arguments(parser):
    parser.add_argument('--pid-head', action='store_true', help='Add node PID logits; CE at one track-prioritized truth alpha per particle')
    parser.add_argument('--five-particle-energy-heads', action='store_true', help='Use five species regressors: charged track alpha / neutral calo sum; log1p absolute error')
    parser.add_argument('--pid-loss-weight', type=float, default=1., help='Multiplier for event-averaged PID cross entropy')
    parser.add_argument('--epochs-noPID', type=int, default=-1, help='Enable PID loss after this epoch (default: epoch 0)')
    from particle_cluster import add_arguments as add_cluster_arguments
    add_cluster_arguments(parser)


def validate_arguments(args):
    from particle_cluster import validate_arguments as validate_cluster_arguments
    validate_cluster_arguments(args)
    if not enabled(args):
        return
    if not args.use_multihead_model:
        raise ValueError('Particle heads require --use-multihead-model')
    for flag in ('jit', 'dp', 'energy_branch', 'energy_regression_weight', 'cluster_energy_pooling'):
        if getattr(args, flag, False):
            raise ValueError('Particle heads do not support --' + flag.replace('_', '-'))
    if not math.isfinite(args.pid_loss_weight) or args.pid_loss_weight < 0:
        raise ValueError('--pid-loss-weight must be finite and nonnegative')
    if args.five_particle_energy_heads:
        if not (args.energy_regression and args.energy_regression_cluster):
            raise ValueError('Five energy heads require --energy-regression --energy-regression-cluster')
        if args.LE_track != 'alpha_tracker_diff_log_perCluster' or args.LE_cluster != 'sum_log_perCluster':
            raise ValueError('Five energy heads require --LE-track alpha_tracker_diff_log_perCluster --LE-cluster sum_log_perCluster')
        # This flag is authoritative; do not require a redundant --multihead-regression-heads 5.
        args.multihead_regression_heads = 5


def species_ids(pdg, charge):
    """Use MC PDG and charge, never track availability, for the true PID."""
    finite = torch.isfinite(pdg) & torch.isfinite(charge)
    p = torch.where(finite, pdg.abs(), torch.zeros_like(pdg))
    t = torch.full_like(pdg, -1, dtype=torch.long)
    hadron = finite & (p >= 100) & (p == p.round())
    t[hadron & (charge != 0)] = 0
    t[hadron & (charge == 0)] = 4
    t[finite & (p == 11)] = 1
    t[finite & (p == 13)] = 2
    t[finite & (p == 22)] = 3
    return t


def event_average(values, object_batch, n_events):
    """Mean within each event, then mean across ALL events, including empty ones."""
    sums = scatter_add(values, object_batch, dim_size=n_events)
    counts = scatter_add(torch.ones_like(values), object_batch, dim_size=n_events)
    return (sums / counts.clamp_min(1)).sum() / max(1, n_events)


def truth_objects(data, beta):
    n_events = int(data.batch.max()) + 1 if data.batch.numel() else 0
    known = data.y[:, 0] > 0
    if getattr(data, 'truth_valid', None) is not None:
        known = known & data.truth_valid.bool()
    nodes = torch.nonzero(known, as_tuple=False).flatten()
    keys, inverse = torch.unique(torch.stack((data.batch[nodes], data.y[nodes, 0].long()), 1),
                                 dim=0, sorted=True, return_inverse=True)
    n = len(keys)
    first = scatter_min(nodes, inverse, dim_size=n)[0]
    track = data.y[nodes, 1].bool()
    track_count = scatter_add(track.long(), inverse, dim_size=n)
    calo_count = scatter_add((~track).long(), inverse, dim_size=n)
    scores = beta.detach()[nodes]
    _, any_alpha = scatter_max(scores, inverse, dim_size=n)
    _, track_alpha = scatter_max(scores.masked_fill(~track, -float('inf')), inverse, dim_size=n)
    alpha = nodes[torch.where(track_count > 0, track_alpha, any_alpha)]
    t_node = species_ids(data.label[nodes, 2], data.label[nodes, 3])
    t = species_ids(data.label[first, 2], data.label[first, 3])
    if n and not torch.equal(t_node, t[inverse]):
        raise ValueError('Inconsistent truth PID inside a truth object')
    target = data.label[first, 4:8].float().square().sum(1).sqrt()
    valid = (t >= 0) & torch.isfinite(target) & (target > 0)
    energy_valid = valid & torch.where(t < 3, track_count > 0, calo_count > 0)
    return dict(nodes=nodes, inverse=inverse, first=first, alpha=alpha, species=t,
                batch=keys[:, 0], target=target, valid=valid, energy_valid=energy_valid,
                track=track, track_count=track_count, calo_count=calo_count, n_events=n_events)


def object_energies(regression_heads, objects, *, charged_only=False):
    if len(regression_heads) != 5:
        raise ValueError('Expected exactly five energy heads')
    o = objects
    if charged_only:
        # Pooled energy replaces the neutral node sums. Gather only the small
        # representative table needed by the retained three charged losses.
        alpha_values = torch.cat([head[o['alpha']] for head in regression_heads[:3]], dim=1).float()
        return alpha_values.gather(1, o['species'].clamp(0, 2)[:, None]).flatten()
    energies = torch.cat(regression_heads, dim=1).float()
    node_calo = o['nodes'][~o['track']]
    sums = scatter_add(energies[node_calo], o['inverse'][~o['track']], dim=0, dim_size=len(o['alpha']))
    alpha_values = energies[o['alpha']]
    values = torch.where((o['species'] < 3)[:, None], alpha_values, sums)
    return values.gather(1, o['species'].clamp_min(0)[:, None]).flatten()


def auxiliary_losses(out, regression_heads, pid_logits, data, args, epoch, coefficient,
                     particle_clusters=None, particle_cluster_augmented=None):
    o = truth_objects(data, torch.sigmoid(out[:, 0].float()))
    # Empty sums connect all outputs without evaluating unavailable rows.
    zero = out.reshape(-1)[:0].sum()
    for head in regression_heads or []:
        zero = zero + head.reshape(-1)[:0].sum()
    if pid_logits is not None:
        zero = zero + pid_logits.reshape(-1)[:0].sum()
    energy_loss, pid_loss = zero, zero
    metrics = {}
    pool_energy = bool(getattr(args, 'particle_cluster_energy', False))
    pool_score = bool(getattr(args, 'particle_cluster_score', False))
    if getattr(args, 'five_particle_energy_heads', False):
        pred = object_energies(regression_heads, o, charged_only=pool_energy)
        valid = o['energy_valid']
        losses = torch.log1p((pred[valid] - o['target'][valid]).abs())
        if pool_energy:
            # Keep ALL eligible objects in the denominator, as before, while
            # avoiding the discarded neutral scatter and regression errors.
            losses = losses.masked_fill(o['species'][valid] >= 3, 0.)
        energy_loss = event_average(losses, o['batch'][valid], o['n_events']) * coefficient + zero
        for i, name in enumerate(SPECIES):
            if pool_energy and i >= 3:
                metrics['L_E_' + name] = zero.detach()
                continue
            chosen = o['species'][valid] == i
            # Contributions sum to L_E; denominator is all eligible objects, not five class means.
            contribution = event_average(losses * chosen, o['batch'][valid], o['n_events']) * coefficient
            metrics['L_E_' + name] = contribution.detach()
        metrics['N_E_missing_track'] = ((o['valid']) & (o['species'] < 3) & (o['track_count'] == 0)).sum().float()
        metrics['L_E'] = energy_loss.detach()
        metrics['L_E_cond'] = sum(metrics['L_E_' + s] for s in SPECIES[:3])
        metrics['L_E_cluster'] = sum(metrics['L_E_' + s] for s in SPECIES[3:])
    if pool_energy or pool_score:
        from particle_cluster import cluster_targets
        if particle_clusters is None:
            raise ValueError('Particle cluster options require pooled outputs from training_forward')
        targets = cluster_targets(particle_clusters, data, need_score=pool_score)
        if pool_energy:
            # Retain the original charged-head term and its object normalization;
            # replace the two node-summed calorimeter terms with the pooled head.
            charged_loss = energy_loss
            pv = targets['valid']
            pooled_losses = (particle_clusters['energy'][pv].float() - targets['energy'][pv]).abs().log1p()
            pooled_loss = event_average(pooled_losses, particle_clusters['cluster_batch'][pv], o['n_events']) * coefficient
            pooled_loss = pooled_loss + particle_clusters['energy'].sum() * 0
            energy_loss = charged_loss + pooled_loss + zero
            for name in SPECIES[3:]:
                metrics['L_E_' + name] = zero.detach()
            metrics.update(L_E=energy_loss.detach(), L_E_cond=charged_loss.detach(),
                           L_E_cluster=pooled_loss.detach(), L_E_particle_cluster=pooled_loss.detach(),
                           N_E_particle_cluster=pv.sum().float())
        if pool_score:
            score_values, score_batch = [], []
            pools = [(particle_clusters, targets)]
            if particle_cluster_augmented is not None:
                pools.append((particle_cluster_augmented, cluster_targets(particle_cluster_augmented, data)))
            score_zero = zero
            for pool, teacher in pools:
                sv = teacher['score_valid']
                # Augmentation preserves real trackless clusters. Supervise those
                # once, and use the augmented view only for newly split fragments.
                if pool is particle_cluster_augmented:
                    original = torch.stack((particle_clusters['cluster_batch'], particle_clusters['cluster_ids']), 1)
                    augmented = torch.stack((pool['cluster_batch'], pool['cluster_ids']), 1)
                    combined, inverse = torch.unique(torch.cat((original, augmented), 0), dim=0, return_inverse=True)
                    seen = torch.zeros(len(combined), dtype=torch.bool, device=out.device)
                    seen[inverse[:len(original)]] = True
                    sv = sv & (~seen[inverse[len(original):]])
                score_values.append(F.binary_cross_entropy_with_logits(pool['score_logits'][sv].float(), teacher['score'][sv], reduction='none'))
                score_batch.append(pool['cluster_batch'][sv])
                score_zero = score_zero + pool['score_logits'].sum() * 0
            values, batches = torch.cat(score_values), torch.cat(score_batch)
            raw_score = event_average(values, batches, o['n_events']) + score_zero
            # Keep this tensor differentiable: compose_loss adds this independent
            # term from epoch zero, even while the energy ramp is disabled.
            metrics['L_particle_score'] = raw_score * args.particle_cluster_score_weight
            metrics['L_particle_score_unweighted'] = raw_score.detach()
            metrics['N_particle_score'] = values.new_tensor(values.numel())
    if getattr(args, 'pid_head', False):
        if pid_logits is None or pid_logits.shape != (len(out), 5):
            raise ValueError('PID enabled but five node logits are missing')
        valid = o['valid']
        logits = pid_logits[o['alpha'][valid]].float()
        labels = o['species'][valid]
        ce = F.cross_entropy(logits, labels, reduction='none')
        raw = event_average(ce, o['batch'][valid], o['n_events']) + zero
        pid_loss = raw * args.pid_loss_weight * (epoch > args.epochs_noPID)
        metrics['L_PID'] = pid_loss.detach()
        metrics['L_PID_unweighted'] = raw.detach()
        metrics['PID_accuracy_event_mean'] = event_average((logits.argmax(1) == labels).float(), o['batch'][valid], o['n_events']).detach()
        metrics['N_PID'] = valid.sum().float()
    return energy_loss, pid_loss, metrics


def compose_loss(components, args, epoch, offset=1.):
    result = components['L_V'] + offset
    if epoch > args.epochs_nobeta:
        result = result + components['L_beta']
    if epoch > args.epochs_noLE:
        result = result + components['L_E']
    if getattr(args, 'particle_cluster_score', False):
        result = result + components['L_particle_score']
    return result + components.get('L_PID', 0.)


def five_head_training_loss(out, regressions, pid_logits, data, args, epoch, qmin,
                            return_components=False, offset=1., *,
                            particle_clusters=None, particle_cluster_augmented=None):
    from supervised_loss import calc_supervised_loss
    coord_start = 2 if args.use_charged_cluster_loss else 1
    target = data.label[:, 4:8].float().square().sum(1).sqrt()
    lv, lb, _, _, components = calc_supervised_loss(
        out[:, 0].sigmoid(), out[:, coord_start:],
        out[:, 1].sigmoid() if args.use_charged_cluster_loss else None,
        data.y[:, 0].long(), target, data.batch, return_components=return_components,
        beta_term_option='short-range-potential', beta_track_term=args.beta_track,
        beta_track_term_beginning=args.beta_track_beginning, force_track_alpha=args.force_track_alpha,
        cluster_track_index=data.y[:, 1], qmin=qmin, tracker_energy=None,
        detected_energy=data.feat[:, 0], Ecl_regression=False, weight_regression=False,
        l_beta_suppression=args.l_beta_suppression, epoch=epoch, epsilon=args.epsilon,
        truth_valid=getattr(data, 'truth_valid', None),
        truth_metadata=getattr(data, 'truth_metadata', None))
    ramp = min(1., max(0., (epoch - args.epochs_noLE) / 10.) ** 2)
    coefficient = args.regression_coefficinet * (ramp if args.LE_gradually else 1.)
    le, lp, extra = auxiliary_losses(out, regressions, pid_logits, data, args, epoch, coefficient,
        particle_clusters=particle_clusters, particle_cluster_augmented=particle_cluster_augmented)
    components = dict(components, **extra)
    if return_components:
        return components
    active = dict(components, L_V=lv, L_beta=lb, L_E=le, L_PID=lp)
    return compose_loss(active, args, epoch, offset), components


def pretraining_responses(out, regressions, data, particle_clusters=None):
    o = truth_objects(data, out[:, 0].sigmoid())
    pooled_energy = particle_clusters is not None and 'energy' in particle_clusters
    pred = object_energies(regressions, o, charged_only=pooled_energy)
    v = o['energy_valid'] & torch.isfinite(pred)
    if pooled_energy:
        from particle_cluster import cluster_targets
        targets = cluster_targets(particle_clusters, data, need_score=False)
        v = v & (o['species'] < 3)
        pv = targets['valid'] & (targets['energy'] > 0) & torch.isfinite(particle_clusters['energy'])
        energy = torch.cat((o['target'][v], targets['energy'][pv]))
        predicted = torch.cat((pred[v], particle_clusters['energy'][pv]))
        # Mixed predicted clusters have no single PDG. Zero labels this response
        # as a pooled calorimeter candidate in the optional pretraining report.
        pdg = torch.cat((data.label[o['first'][v], 2].long(), torch.zeros_like(targets['energy'][pv], dtype=torch.long)))
        return dict(ratio=(predicted / energy).detach().cpu().numpy(),
                    energy=energy.detach().cpu().numpy(), pdg=pdg.detach().cpu().numpy())
    return dict(ratio=(pred[v] / o['target'][v]).detach().cpu().numpy(),
                energy=o['target'][v].detach().cpu().numpy(),
                pdg=data.label[o['first'][v], 2].detach().long().cpu().numpy())


def format_metrics(components):
    keys = [k for k in components if k.startswith(('L_PID', 'PID_', 'N_PID', 'N_E_', 'L_particle_', 'N_particle_')) or k in ['L_E_' + s for s in SPECIES] + ['L_E_particle_cluster']]
    return ''.join('\n    {} = {:.6g}'.format(k, float(components[k])) for k in keys)


def raw_model(model):
    while hasattr(model, 'module') or hasattr(model, 'model'):
        model = model.module if hasattr(model, 'module') else model.model
    return model


def checkpoint_config(model, epoch):
    m = raw_model(model)
    if not (getattr(m, 'pid_head', False) or getattr(m, 'five_particle_energy_heads', False)):
        return None
    config = dict(version=1, species=list(SPECIES), pid_head=m.pid_head,
                five_particle_energy_heads=m.five_particle_energy_heads,
                pid_alpha='track_if_present_else_max_beta', neutral_energy='calo_sum',
                energy_loss='log1p_absolute_GeV', interaction_start_epoch=m.interaction_start_epoch,
                coordinate_start=m.cluster_energy_coordinate_start,
                inference_pid='all_five', trackless_charged='best_neutral_probability_fallback',
                inference_epoch=epoch if epoch is not None else m.interaction_start_epoch)
    if getattr(m, 'particle_cluster_energy', False) or getattr(m, 'particle_cluster_score', False):
        from particle_cluster import PC_FEATURE_SCHEMA
        if m.particle_cluster_energy:
            config['neutral_energy'] = 'particle_cluster_pooled_scalar'
        config['particle_cluster_config'] = dict(version=1, feature_schema=PC_FEATURE_SCHEMA,
            energy=m.particle_cluster_energy, score=m.particle_cluster_score,
            source=m.particle_cluster_source, tbeta=m.particle_cluster_tbeta, td=m.particle_cluster_td)
    return config


def validate_checkpoint_config(state, config):
    pid = any(k.startswith('pid_output.') for k in state)
    five = '_particle_energy_version' in state
    pc_weights = any(k.startswith('particle_cluster_head.') for k in state)
    if config is None:
        if pid or five or pc_weights:
            raise ValueError('Particle head checkpoint is missing particle_heads_config')
        return {}
    if config.get('version') != 1 or config.get('species') != list(SPECIES):
        raise ValueError('Unsupported particle head checkpoint version or species order')
    if (pid, five) != (config.get('pid_head'), config.get('five_particle_energy_heads')):
        raise ValueError('Particle head checkpoint metadata disagrees with its weights')
    if five and int(state['_particle_energy_version']) != 1:
        raise ValueError('Unsupported five-head weights version')
    pc = config.get('particle_cluster_config')
    if bool(pc) != pc_weights:
        raise ValueError('Particle cluster checkpoint metadata disagrees with its weights')
    if pc:
        from particle_cluster import PC_FEATURE_SCHEMA
        from cluster_energy import validate_settings
        if not (pid and five) or pc.get('version') != 1 or pc.get('feature_schema') != PC_FEATURE_SCHEMA:
            raise ValueError('Unsupported particle cluster checkpoint version or feature schema')
        energy = any(k.startswith('particle_cluster_head.energy_output.') for k in state)
        score = any(k.startswith('particle_cluster_head.score_output.') for k in state)
        if (energy, score) != (pc.get('energy'), pc.get('score')) or not (energy or score):
            raise ValueError('Particle cluster output metadata disagrees with its weights')
        validate_settings(pc.get('source'), pc.get('tbeta'), pc.get('td'))
    return config


def load_training_checkpoint(model, state, config):
    """Initialize new heads explicitly; never silently reinterpret a five-head file."""
    config = validate_checkpoint_config(state, config)
    old_five = config.get('five_particle_energy_heads', False)
    old_pid = config.get('pid_head', False)
    if old_five and not getattr(model, 'five_particle_energy_heads', False):
        raise ValueError('Checkpoint requires --five-particle-energy-heads')
    if old_pid and not getattr(model, 'pid_head', False):
        raise ValueError('Checkpoint requires --pid-head')
    old_pc = config.get('particle_cluster_config', {})
    new_energy = bool(getattr(model, 'particle_cluster_energy', False))
    new_score = bool(getattr(model, 'particle_cluster_score', False))
    if old_pc.get('energy') and not new_energy:
        raise ValueError('Checkpoint requires --particle-cluster-energy')
    if old_pc.get('score') and not new_score:
        raise ValueError('Checkpoint requires --particle-cluster-score')
    mapped = dict(state)
    allowed_missing = []
    if (new_energy or new_score) and not old_pc:
        allowed_missing.append('particle_cluster_head.')
    else:
        if new_energy and not old_pc.get('energy'):
            allowed_missing.append('particle_cluster_head.energy_output.')
        if new_score and not old_pc.get('score'):
            allowed_missing.append('particle_cluster_head.score_output.')
    if model.pid_head and not old_pid:
        allowed_missing += ['pid_postgn_dense.', 'pid_output.']
    if model.five_particle_energy_heads and not old_five:
        import re
        old_indices = {int(m.group(1)) for k in state
                       for m in [re.match(r'head_output\.(\d+)\.', k)] if m}
        if not old_indices or max(old_indices) > 2 or any(k.startswith('cluster_energy_head.') for k in state):
            raise ValueError('Five-head initialization requires a legacy multihead checkpoint with at most two energy heads')
        prefixes = ('head_postgn_dense', 'head_output', 'interaction_blocks')
        mapped = {k: v for k, v in state.items() if not
                  (k.startswith(('head_postgn_dense.', 'head_output.')) and k.split('.')[1] != '0')
                  and not k.startswith('interaction_blocks.')}
        for new_index in range(1, 6):
            old_index = 1 if new_index <= 3 else 2
            for prefix in prefixes:
                ni = new_index - 1 if prefix == 'interaction_blocks' else new_index
                oi = old_index - 1 if prefix == 'interaction_blocks' else old_index
                target_prefix, source_prefix = f'{prefix}.{ni}.', f'{prefix}.{oi}.'
                copied = {target_prefix + k[len(source_prefix):]: v for k, v in state.items() if k.startswith(source_prefix)}
                mapped.update(copied)
                if not copied:
                    allowed_missing.append(target_prefix)
        allowed_missing.append('_particle_energy_version')
        print('Initialize five species heads: copy legacy track head to charged/electron/muon, calo head to photon/neutral; retain shared encoder')
    loaded = model.load_state_dict(mapped, strict=False)
    missing = [k for k in loaded.missing_keys if not any(k.startswith(p) for p in allowed_missing)]
    if missing or loaded.unexpected_keys:
        raise ValueError(f'Incompatible particle checkpoint: missing={missing}, unexpected={loaded.unexpected_keys}')
    if loaded.missing_keys:
        print('Initialized new parameters: ' + ', '.join(sorted(set(k.split('.')[0] for k in loaded.missing_keys))))
    return model


def decode_clusters(assignments, beta, track, energies=None, pid_logits=None,
                    truth_species=None, *, track_momenta=None):
    """CPU readout, one class per cluster. Truth PID is an explicit diagnostic only.

Five-head energy readout requires measured track momenta at the calorimeter
entry, in GeV (raw feature columns 7:10). A charged prediction with exactly
one track uses sqrt(p^2 + m_PID^2), with pion/electron/muon mass hypotheses.
Multiple-track charged predictions retain the highest-beta track's regressed
energy; neutral sums and trackless charged fallback are unchanged.
The returned PID is never replaced by the fallback energy-head choice.
No unknown-truth filtering is applied to predicted cluster membership.
"""
    import numpy as np
    assignments, beta, track = np.asarray(assignments), np.asarray(beta), np.asarray(track, dtype=bool)
    if truth_species is None and pid_logits is None:
        raise ValueError('Normal five-head inference requires a PID head; use explicit truth-PID diagnostics')
    momenta = None
    if energies is not None:
        if track_momenta is None:
            raise ValueError('Five-head cluster energy readout requires measured track_momenta')
        momenta = np.asarray(track_momenta, dtype=np.float64)
        if momenta.shape != (len(assignments), 3):
            raise ValueError('Measured track_momenta must have shape (number of nodes, 3)')
    result = []
    for ci in np.unique(assignments):
        if ci <= 0:
            continue
        members = np.flatnonzero(assignments == ci)
        tracks = members[track[members]]
        candidates = tracks if len(tracks) else members
        seed = candidates[np.argmax(beta[candidates])]
        probs = None
        if pid_logits is not None:
            logits = np.asarray(pid_logits[seed], dtype=np.float64)
            if logits.shape != (5,) or not np.isfinite(logits).all():
                raise ValueError('PID logits must have five finite values')
            probs = np.exp(logits - logits.max()); probs /= probs.sum()
        if truth_species is not None:
            types = np.unique(np.asarray(truth_species)[members])
            if len(types) != 1 or types[0] < 0:
                raise ValueError('Truth-PID diagnosis requires pure, supported truth clusters')
            pid = int(types[0])
        else:
            pid = int(probs.argmax())
        used = pid
        fallback = int(pid < 3 and not len(tracks))
        if fallback:
            if probs is None:
                raise ValueError('Trackless charged truth-PID object has no PID probabilities for neutral fallback')
            used = 3 + int(probs[3:].argmax())
        energy = None
        if energies is not None:
            values = np.asarray(energies)
            if values.shape != (len(assignments), 5):
                raise ValueError('Five energy values per node are required')
            if used < 3:
                if len(tracks) == 1:
                    momentum = momenta[seed]
                    if not np.isfinite(momentum).all():
                        raise ValueError('Single-track charged energy requires finite measured momentum')
                    energy = float(np.sqrt(np.sum(momentum ** 2) + CHARGED_MASSES_GEV[pid] ** 2))
                else:
                    energy = float(values[seed, used])
            else:
                energy = float(values[members[~track[members]], used].sum(dtype=np.float64))
        result.append(dict(cluster=int(ci), seed=int(seed), pid=pid, energy_head=used,
                           energy_fallback=fallback, energy=energy,
                           probabilities=probs))
    return result
