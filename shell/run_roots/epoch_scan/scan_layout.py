"""Paths for one model/run/energy, without repeating its identity below it."""
from __future__ import annotations

import copy
from pathlib import Path

COMPACT_LAYOUT = 'model-energy-v1'
EPOCH_LAYOUT = 'epoch-file-v1'


def compact_plan(plan, filelists):
    """Arrange a newly built single-model/energy plan before it is published."""
    jobs = plan['jobs']
    identity = {key: jobs[0][key] for key in ('model', 'run_id', 'energy')}
    if any(any(job[key] != value for key, value in identity.items()) for job in jobs):
        raise ValueError('Compact layout requires exactly one model, run and energy')
    scan_dir = Path(plan['scan_dir'])
    rewritten_lists = {}
    destinations = set()
    for job in jobs:
        contents = filelists[job['input_list']]
        input_list = scan_dir / 'filelists' / f"shard_{job['shard']:03d}.txt"
        previous = rewritten_lists.setdefault(str(input_list), contents)
        if previous != contents:
            raise ValueError('Different input lists share one compact shard path')
        output_dir = (scan_dir / 'roots' / f"epoch_{job['epoch']:03d}" / job['method'])
        for item in job['inputs']:
            destination = output_dir / (Path(item['path']).stem + '.root')
            if destination in destinations:
                raise ValueError('Duplicate compact ROOT destination: ' + str(destination))
            destinations.add(destination)
        job['input_list'] = str(input_list)
        job['output_dir'] = str(output_dir)
    plan['layout'] = COMPACT_LAYOUT
    plan['layout_identity'] = identity
    return plan, rewritten_lists


def epoch_plan(plan, result_dir, filelists):
    """Return a single-model/energy plan with all H5s in each epoch/method job.

    The original plan is never modified. Keeping the first shard's stable ID
    also lets a one-shard legacy scan retain its job identities when converted.
    """
    plan = copy.deepcopy(plan)
    jobs = plan['jobs']
    if not jobs:
        raise ValueError('Epoch-file layout requires jobs')
    identity = {key: jobs[0][key] for key in ('model', 'run_id', 'energy')}
    if any(any(job[key] != value for key, value in identity.items()) for job in jobs):
        raise ValueError('Epoch-file layout requires exactly one model, run and energy')
    result = Path(result_dir)
    if not result.is_absolute():
        raise ValueError('Epoch result directory must be absolute')
    management = result / '_scan'
    input_list = management / 'filelists' / 'inputs.txt'
    groups = {}
    for job in jobs:
        if job['method'] not in ('reco', 'truth_clustering'):
            raise ValueError('Unsupported epoch-file method: ' + job['method'])
        contents = filelists[job['input_list']]
        listed = [line.strip() for line in contents.splitlines()
                  if line.strip() and not line.lstrip().startswith('#')]
        if listed != [item['path'] for item in job['inputs']]:
            raise ValueError('Input list differs from planned H5 inputs')
        groups.setdefault((int(job['epoch']), job['method']), []).append(job)
    merged = []
    selection = None
    varying = {'id', 'shard', 'inputs', 'input_list', 'output_dir', 'output_file'}
    for key, group in sorted(groups.items()):
        group.sort(key=lambda job: (job['shard'], job['id']))
        first = group[0]
        fixed = {name: value for name, value in first.items() if name not in varying}
        if any({name: value for name, value in job.items() if name not in varying} != fixed
               for job in group):
            raise ValueError('Incompatible shards in one epoch/method')
        inputs = sorted((item for job in group for item in job['inputs']),
                        key=lambda item: item['path'])
        if len({item['path'] for item in inputs}) != len(inputs):
            raise ValueError('Duplicate H5 inputs in one epoch/method')
        if selection is None:
            selection = inputs
        elif selection != inputs:
            raise ValueError('All epoch/method jobs must use the same frozen H5 selection')
        folder = 'truth' if first['method'] == 'truth_clustering' else 'reco'
        job = copy.deepcopy(first)
        job.update(shard=0, inputs=inputs, input_list=str(input_list),
                   output_dir=str(result / folder),
                   output_file=str(result / folder / f"epoch{int(job['epoch']):03d}.root"))
        merged.append(job)
    if len({job['id'] for job in merged}) != len(merged):
        raise ValueError('Duplicate epoch-file job IDs')
    if len({job['output_file'] for job in merged}) != len(merged):
        raise ValueError('Duplicate epoch-file ROOT destinations')
    plan.update(layout=EPOCH_LAYOUT, layout_identity=identity,
                scan_dir=str(management), result_dir=str(result), jobs=merged)
    plan['selection']['files_per_job'] = len(selection)
    plan['bundle_count'] = len({job['epoch'] for job in merged})
    return plan, {str(input_list): ''.join(item['path'] + '\n' for item in selection)}


def summary_directory(plan, group):
    """Return compact summary paths, or keep the core collector's legacy paths."""
    if plan.get('layout') not in (COMPACT_LAYOUT, EPOCH_LAYOUT):
        return None
    identity = plan['layout_identity']
    if any(group[key] != value for key, value in identity.items()):
        raise ValueError('Summary group differs from the compact plan identity')
    if plan.get('layout') == EPOCH_LAYOUT:
        method = 'truth' if group['method'] == 'truth_clustering' else 'reco'
        return (Path(plan['scan_dir']) / 'summaries' /
                f"epoch{int(group['epoch']):03d}" / method)
    return (Path(plan['scan_dir']) / 'summaries' /
            f"epoch_{int(group['epoch']):03d}" / group['method'])
