"""Adapters for one ROOT per epoch, leaving legacy scan implementations intact."""
from __future__ import annotations

from pathlib import Path
import sys

import scan_common as common
import run_server as legacy

EPOCH_LAYOUT = 'epoch-file-v1'
EXECUTION_SOURCES = tuple('shell/run_roots/epoch_scan/' + name for name in (
    'epoch_jobs.py', 'scan_layout.py', 'epoch_writer.py', 'epoch_root.py',
    'epoch_collect.py', 'run_parallel.py', 'run_persistent.py',
    'persistent_worker.py', 'persistent_model.py', 'collect_scan.py',
))

_legacy_outputs = common.expected_outputs
_legacy_command = legacy.command_for
_legacy_verify = common.verify_implementation
_installed = False


def expected_outputs(job):
    if job.get('output_file') is not None:
        return [Path(job['output_file'])]
    return _legacy_outputs(job)


def command_for(plan, job, args, gpu):
    if job.get('output_file') is None:
        return _legacy_command(plan, job, args, gpu)
    command = [args.python, '-B',
               str(Path(plan['repo']) / 'shell/run_roots/epoch_scan/epoch_writer.py'),
               '@' + job['input_list'], job['checkpoint'], job['output_file'],
               str(job['nstart']), str(job['nend']), 'False',
               str(job['input_dim']), str(job['output_dim'])]
    command += job['writer_args']
    command += ['--device', 'cpu' if gpu == 'cpu' else 'cuda:0',
                '--pid-threads', str(args.pid_threads),
                '--root-chunk-events', str(args.root_chunk_events),
                '--pipeline-depth', str(args.pipeline_depth)]
    if args.replace:
        command.append('--replace')
    return command


def freeze_execution(plan):
    """Record this adapter/writer without changing the frozen legacy code list."""
    repo = Path(plan['repo'])
    plan['execution_implementation'] = {
        name: common.sha256_file(repo / name) for name in EXECUTION_SOURCES
        if (repo / name).is_file()}
    return plan


def verify_implementation(plan):
    _legacy_verify(plan)
    repo = Path(plan['repo'])
    expected = plan.get('execution_implementation', {})
    if plan.get('layout') == EPOCH_LAYOUT and not expected:
        raise ValueError('Epoch-file plan has no frozen adapter/writer identity')
    changed = [name for name, digest in expected.items()
               if not (repo / name).is_file() or common.sha256_file(repo / name) != digest]
    if changed:
        raise ValueError('Epoch writer code changed since planning: ' + ', '.join(changed)
                         + '. Restore that version or use a new scan directory.')


def install():
    """Dispatch new jobs through adapters; ordinary jobs retain their old path."""
    global _installed
    if _installed:
        return
    _installed = True
    common.expected_outputs = expected_outputs
    common.verify_implementation = verify_implementation
    legacy.expected_outputs = expected_outputs
    legacy.command_for = command_for
    legacy.verify_implementation = verify_implementation
    # Some entry points import these helpers before installing the adapter.
    for name in ('run_parallel', 'run_persistent', 'collect_scan'):
        module = sys.modules.get(name)
        if module is not None:
            if hasattr(module, 'expected_outputs'):
                module.expected_outputs = expected_outputs
            if hasattr(module, 'verify_implementation'):
                module.verify_implementation = verify_implementation
