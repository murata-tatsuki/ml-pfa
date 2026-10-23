"""Paths for one model/run/energy, without repeating its identity below it."""
from __future__ import annotations

from pathlib import Path

COMPACT_LAYOUT = 'model-energy-v1'


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


def summary_directory(plan, group):
    """Return compact summary paths, or keep the core collector's legacy paths."""
    if plan.get('layout') != COMPACT_LAYOUT:
        return None
    identity = plan['layout_identity']
    if any(group[key] != value for key, value in identity.items()):
        raise ValueError('Summary group differs from the compact plan identity')
    return (Path(plan['scan_dir']) / 'summaries' /
            f"epoch_{int(group['epoch']):03d}" / group['method'])
