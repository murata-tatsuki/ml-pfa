"""Collect the unchanged JER/readout definitions from one ROOT per epoch.

The original H5 boundaries are recorded inside the ROOT. They define the
comparison-event digest independently of its new global ROOT event numbers.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile

import collect_jer as core
import epoch_jobs
from scan_layout import summary_directory


EPOCH_LAYOUT = 'epoch-file-v1'
EPOCH_METADATA = 'epoch_scan_metadata'
EPOCH_SCHEMA = 'epoch-root-1'
GROUP_FIELDS = ('model', 'run_id', 'epoch', 'energy', 'method')


def _group(job):
    return {name: job[name] for name in GROUP_FIELDS}


def _cache_key(job):
    identity = _group(job)
    identity.update({name: job[name] for name in (
        'checkpoint_sha256', 'nstart', 'nend', 'input_dim', 'output_dim', 'writer_args')})
    identity['inputs'] = [core._identity(item) for item in job['inputs']]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()


def _root_identity(path):
    info = Path(path).stat()
    return dict(device=info.st_dev, inode=info.st_ino, size=info.st_size, mtime_ns=info.st_mtime_ns)


def _source_ranges(metadata, identities, all_ids, np):
    """Validate a complete, ordered partition of the saved global event IDs."""
    if metadata.get('schema') != EPOCH_SCHEMA:
        raise ValueError('Missing supported epoch ROOT completion metadata')
    ranges = metadata.get('source_ranges')
    if not isinstance(ranges, list) or len(ranges) != len(identities):
        raise ValueError('Epoch ROOT source ranges differ from planned H5 inputs')
    offset = 0
    checked = []
    for item, identity in zip(ranges, identities):
        if not isinstance(item, dict) or core._identity(item['input']) != identity:
            raise ValueError('Epoch ROOT source identity differs from planned H5')
        begin, count = item.get('event_offset'), item.get('event_count')
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in (begin, count)) or begin != offset or count < 0:
            raise ValueError('Epoch ROOT source ranges are not a contiguous partition')
        checked.append(dict(input=identity, event_offset=begin, event_count=count))
        offset += count
    if offset > 2147483647:
        raise ValueError('Epoch ROOT event numbers exceed the supported int32 range')
    if len(all_ids) != offset or not np.array_equal(all_ids, np.arange(offset, dtype=np.int64)):
        raise ValueError('Epoch ROOT event IDs differ from their saved source ranges')
    return checked


def _selection_digest(rows, ranges):
    """Retain collect_jer's H5-path/local-event/truth/thrust digest exactly."""
    digest = hashlib.sha256()
    by_event = {row['event']: row for row in rows}
    for item in sorted(ranges, key=lambda value: value['input']['path']):
        digest.update(item['input']['path'].encode('utf-8') + b'\0')
        offset = item['event_offset']
        for event in range(offset, offset + item['event_count']):
            row = by_event.get(event)
            if row is None:
                # The legacy collector excludes nonpositive/nonfinite truth.
                continue
            angle = row['thrust']
            digest.update(struct.pack('<qdd', event - offset, row['truth_energy'],
                                      angle if math.isfinite(angle) else float('nan')))
    return digest.hexdigest()


def read_epoch_data(plan, job, beta_threshold=0.0):
    """Read and validate one combined ROOT without publishing any files.

    ``rows`` uses the unique saved event number. ``modes`` maps each of saved,
    raw_head and track_replaced to ``{event: energy}`` when PID is enabled.
    ``source_ranges`` recovers each original H5 and its local event number,
    enabling exact before/after migration comparisons without inference.
    """
    if plan.get('layout') != EPOCH_LAYOUT:
        raise ValueError('Combined ROOT collection requires an epoch-file-v1 plan')
    if not math.isfinite(beta_threshold) or not 0 <= beta_threshold <= 1:
        raise ValueError('beta threshold must be finite and in [0, 1]')
    if job['method'] not in ('reco', 'truth_clustering'):
        raise ValueError('Unsupported evaluation method')
    if not job.get('inputs'):
        raise ValueError('Epoch job has no H5 inputs')
    epoch_jobs.verify_implementation(plan)
    deps = core._dependencies(plan['repo'])
    np, _, root_io, helper, _, _ = deps
    identities = [core._check_identity(item) for item in job['inputs']]
    if len({item['path'] for item in identities}) != len(identities):
        raise ValueError('Input H5 occurs more than once in the epoch job')
    if core._sha256(job['checkpoint']) != job['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since planning: ' + job['checkpoint'])
    root_path = Path(job['output_file'])
    root_identity = _root_identity(root_path)
    with root_io.open_root(root_path) as source:
        metadata = helper._json_object(source, 'save_root_metadata')
        signature = metadata.get('signature', {})
        if signature.get('version') != 'cedric-unified-1':
            raise ValueError('Missing supported writer completion identity')
        if signature.get('checkpoint_sha256') != job['checkpoint_sha256']:
            raise ValueError('ROOT checkpoint hash differs from plan')
        if [core._identity(item) for item in signature.get('inputs', [])] != identities:
            raise ValueError('ROOT input identities differ from planned H5 files')
        settings_job = dict(job, output_dir=str(root_path.parent))
        settings = core._settings(settings_job, signature.get('settings', {}), deps)
        if bool(settings['truth_clustering']) != (job['method'] == 'truth_clustering'):
            raise ValueError('ROOT clustering method differs from group')
        implementation = signature.get('implementation', {})
        if not implementation or any(plan.get('implementation', {}).get(name) != digest
                                     for name, digest in implementation.items()):
            raise ValueError('ROOT writer source hashes differ from plan')
        entries = {name: int(root_io.get_tree(source, name).GetEntries()) for name in core.TREE_NAMES}
        if metadata.get('tree_entries') != entries:
            raise ValueError('ROOT trees differ from writer completion identity')
        builder = helper._arrays(root_io.get_tree(source, 'event'), ('event', 'MC_dijet_energy'))
        all_ids = builder['event'].astype(np.int64)
        epoch_metadata = helper._json_object(source, EPOCH_METADATA)
        ranges = _source_ranges(epoch_metadata, identities, all_ids, np)
        allowed_ids = set(map(int, all_ids))
        jets = helper._arrays(root_io.get_tree(source, 'jet'), ('event', 'n_jets', 'jet_p4'))
        if len(np.unique(jets['event'])) != len(jets['event']):
            raise ValueError('Duplicate event IDs in source jet tree')
        if not set(map(int, jets['event'])).issubset(allowed_ids):
            raise ValueError('Jet rows reference absent source events')
        angles = {int(event): helper._thrust(np.asarray(p4).reshape(-1))
                  if int(count) == 2 else float('nan')
                  for event, count, p4 in zip(jets['event'], jets['n_jets'], jets['jet_p4'])}
        valid = np.isfinite(builder['MC_dijet_energy']) & (builder['MC_dijet_energy'] > 0)
        event_ids, truth = all_ids[valid], builder['MC_dijet_energy'][valid]
        pid = root_io.has_branch(root_io.get_tree(source, 'reco'), 'energy_pid')
        if pid != (settings['pid_readout'] == 'representative'):
            raise ValueError('Saved PID branches disagree with writer settings')
        fields = ('event', 'cluster', 'cond_beta', 'energy_pid') if pid else (
            'event', 'cluster', 'cond_beta', 'pred_edep', 'pred_edep_cluster', 'ntrack_hits')
        reco = helper._arrays(root_io.get_tree(source, 'reco'), fields)
        cluster_ids = np.rec.fromarrays((reco['event'], reco['cluster']))
        if len(np.unique(cluster_ids)) != len(cluster_ids):
            raise ValueError('Duplicate event/cluster IDs in source reco tree')
        if not set(map(int, reco['event'])).issubset(allowed_ids):
            raise ValueError('Reco rows reference absent source events')
        energies = (reco['energy_pid'].astype(np.float64) if pid else
                    np.where(reco['ntrack_hits'] > 0, reco['pred_edep'],
                             reco['pred_edep_cluster']).astype(np.float64))
        particle = helper._json_object(source, 'particle_eval_metadata', required=pid)
    selected = ~(reco['cond_beta'] < beta_threshold)
    saved = helper._totals(reco['event'], energies, selected, event_ids)
    rows = [dict(event=int(event), truth_energy=float(value),
                 thrust=angles.get(int(event), float('nan')), reco_energy=saved[int(event)])
            for event, value in zip(event_ids, truth)]
    modes = {'saved': saved}
    if pid:
        alternate, _, _ = helper._file_readouts(root_path, particle, rows, beta_threshold)
        modes.update(alternate)
    if _root_identity(root_path) != root_identity:
        raise ValueError('Epoch ROOT changed while its analysis data was being read')
    return dict(root=str(root_path), settings=settings, pid=pid, particle=particle,
                events=event_ids, truth=truth, angles=angles,
                rows=rows, modes=modes, source_ranges=ranges,
                event_selection_sha256=_selection_digest(rows, ranges),
                _root_identity=root_identity, _cache_key=_cache_key(job),
                _beta_threshold=beta_threshold)


def collect_epoch_group(plan, jobs, beta_threshold=0.0, allow_incomplete=False,
                        output_directory=None, cached_data=None):
    """Return the same metric rows as collect_jer.collect_group for epoch ROOTs."""
    if len(jobs) != 1:
        raise ValueError('An epoch-file-v1 group must contain exactly one job')
    job = jobs[0]
    group = _group(job)
    if plan.get('layout') != EPOCH_LAYOUT or group['method'] not in ('reco', 'truth_clustering'):
        raise ValueError('Unsupported epoch ROOT layout or evaluation method')
    if not math.isfinite(beta_threshold) or not 0 <= beta_threshold <= 1:
        raise ValueError('beta threshold must be finite and in [0, 1]')
    if not job.get('inputs'):
        raise ValueError('Epoch job has no H5 inputs')
    # Code/checkpoint changes invalidate the entire frozen comparison, including
    # --allow-incomplete, just as in the original per-H5 collector.
    epoch_jobs.verify_implementation(plan)
    if core._sha256(job['checkpoint']) != job['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since planning: ' + job['checkpoint'])
    expected_files = len(job['inputs'])
    if output_directory is None:
        output_directory = summary_directory(plan, group)
        if output_directory is None:
            raise ValueError('Epoch plan has no supported summary directory')
    output_directory = Path(output_directory)
    deps = core._dependencies(plan['repo'])
    _, ROOT, root_io, _, _, _ = deps
    summaries, loaded, failures = {}, [], []
    event_digest = hashlib.sha256().hexdigest()
    total_events = 0
    try:
        data = read_epoch_data(plan, job, beta_threshold) if cached_data is None else cached_data
        if (data.get('_cache_key') != _cache_key(job)
                or data.get('_beta_threshold') != beta_threshold
                or data.get('_root_identity') != _root_identity(job['output_file'])):
            raise ValueError('Cached epoch analysis does not match this ROOT/job/beta threshold')
        if cached_data is not None:
            for item in job['inputs']:
                core._check_identity(item)
        rows = data['rows']
        root_path = str(Path(job['output_file']).resolve())
        particle_text = json.dumps(data['particle'], sort_keys=True) if data['pid'] else ''
        for mode, energies in data['modes'].items():
            summary = ROOT.pid_jer.Summary(int(group['energy']), group['method'], beta_threshold, mode)
            summary.addInput(0, root_path, data['pid'], particle_text)
            for row in rows:
                summary.observe(0, row['event'], row['thrust'],
                                energies[row['event']], row['truth_energy'])
            summaries[mode] = summary
        event_digest = data['event_selection_sha256']
        total_events = len(rows)
        for file_index, item in enumerate(data['source_ranges']):
            offset, count = item['event_offset'], item['event_count']
            count_valid = sum(offset <= row['event'] < offset + count for row in rows)
            loaded.append(dict(job_id=job['id'], input=item['input']['path'], root=root_path,
                               file_index=file_index, event_offset=offset,
                               event_count=count, events=count_valid))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        if not allow_incomplete:
            raise ValueError(f"Cannot collect {job['output_file']}: {error}") from error
        # A combined file is published atomically. A missing/invalid file
        # contributes no H5 subset and can never yield a rankable partial result.
        summaries.clear()
        loaded.clear()
        total_events = 0
        event_digest = hashlib.sha256().hexdigest()
        for item in job['inputs']:
            failures.append(dict(job_id=job['id'], input=item['path'],
                                 root=job['output_file'], reason=str(error)))
    complete = not failures
    coverage = dict(schema='root-epoch-scan-coverage-1', group=group,
                    complete=complete, eligible_for_ranking=complete,
                    expected_files=expected_files, collected_files=len(loaded), events=total_events,
                    event_selection_sha256=event_digest, beta_threshold=beta_threshold,
                    loaded=loaded, failures=failures)
    metrics = []
    with tempfile.TemporaryDirectory(prefix='epoch-file-jer-') as temporary:
        staging = Path(temporary)
        for mode, summary in summaries.items():
            mode_directory = staging / mode
            summary.write(str(mode_directory), False)
            with root_io.open_root(mode_directory / 'jet_energy_resolution_summary.root') as source:
                result = root_io.arrays(root_io.get_tree(source, 'resolution'))
                for index in range(len(result['angle_bin'])):
                    angle_bin = int(result['angle_bin'][index])
                    region = 'central' if angle_bin == -1 else 'forward' if angle_bin == -2 else 'angle'
                    metrics.append(dict(group, readout=mode, region=region,
                        **{name: values[index].item() for name, values in result.items()},
                        complete=complete, eligible_for_ranking=complete,
                        expected_files=expected_files, collected_files=len(loaded), events=total_events,
                        event_selection_sha256=event_digest, beta_threshold=beta_threshold,
                        summary_root=str(output_directory / mode / 'jet_energy_resolution_summary.root')))
            for name in ('jet_energy_resolution_summary.root', 'jet_energy_resolution_summary.txt'):
                core._publish(mode_directory / name, output_directory / mode / name)
        if not metrics:
            metrics.append(dict(group, readout='saved', region='unavailable', valid=False,
                complete=False, eligible_for_ranking=False, expected_files=expected_files,
                collected_files=0, events=0, event_selection_sha256=event_digest,
                beta_threshold=beta_threshold))
        coverage_path = staging / 'coverage.json'
        coverage_path.write_text(json.dumps(coverage, indent=2, sort_keys=True) + '\n')
        core._publish(coverage_path, output_directory / 'coverage.json')
    return metrics
