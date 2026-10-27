"""Dependency-free helpers for immutable multi-server ROOT epoch scans."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

SCHEMA = 'root-epoch-scan-1'
WRITER_SOURCES = (
    'save_root.py', 'root_export/cedric.py', 'root_export/runtime.py',
    'root_export/scan.py', 'root_export/clustering.py', 'model.py',
    'gravnet_model.py', 'gravnet_conv.py', 'dataset.py', 'event.py',
    'knn_event_parallel/__init__.py', 'knn_event_parallel/_C.so',
    'prediction.py', 'test_yielder_edit.py', 'clustering.py', 'clustering_torch.py', 'matching.py',
    'objectcondensation.py', 'particle_heads.py', 'cluster_energy.py',
    'particle_cluster.py', 'extended_h5.py', 'pandora_eval_data.py',
    'pandora_eval_reconstruction.py', 'sed.py', 'tools/load_awkward.py',
    'scan_fast_root_writer.cxx', 'macro/save_pid_energy_readouts.py',
    'macro/pyroot_io.py',
    'macro/src/pid_jet_resolution_summary.h',
    'shell/run_roots/epoch_scan/scan_common.py',
    'shell/run_roots/epoch_scan/plan_scan.py',
    'shell/run_roots/epoch_scan/run_server.py',
    'shell/run_roots/epoch_scan/collect_jer.py',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def implementation_hashes(repo):
    repo = Path(repo)
    return {name: sha256_file(repo / name) for name in WRITER_SOURCES
            if (repo / name).is_file()}


def verify_implementation(plan):
    expected = plan.get('implementation', {})
    if 'save_root.py' not in expected:
        raise ValueError('Plan has no frozen save_root.py identity')
    repo = Path(plan['repo'])
    changed = [name for name, digest in expected.items()
               if not (repo / name).is_file() or sha256_file(repo / name) != digest]
    if changed:
        raise ValueError('Code changed since planning: ' + ', '.join(changed)
                         + '. Restore that version or create a new scan directory.')


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8',
                                         dir=path.parent, prefix='.' + path.name + '.',
                                         suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_plan(path):
    with open(path, encoding='utf-8') as stream:
        plan = json.load(stream)
    if plan.get('schema') != SCHEMA:
        raise ValueError('Unsupported epoch-scan plan schema')
    jobs = plan.get('jobs')
    if not isinstance(jobs, list) or not jobs:
        raise ValueError('Plan has no jobs')
    ids = [job['id'] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate job IDs in plan')
    for job_id in ids:
        if not isinstance(job_id, str) or not job_id or any(
                c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in job_id):
            raise ValueError('Invalid job ID')
    if not Path(plan['repo']).is_absolute() or not Path(plan['scan_dir']).is_absolute():
        raise ValueError('Plan repo and scan_dir must be absolute paths')
    return plan


def file_identity(path):
    stat = Path(path).stat()
    return dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def job_digest(job):
    return hashlib.sha256(json.dumps(job, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def expected_outputs(job):
    return [Path(job['output_dir']) / (Path(item['path']).stem + '.root')
            for item in job['inputs']]


def state_path(plan, job):
    return Path(plan['scan_dir']) / 'state' / (job['id'] + '.json')


def completed_state(plan, job):
    """Fast resume marker; --recheck delegates ROOT validation to the writer."""
    try:
        state = json.loads(state_path(plan, job).read_text())
        if state.get('status') != 'completed' or state.get('job_digest') != job_digest(job):
            return False
        outputs = expected_outputs(job)
        return len(outputs) == len(state['outputs']) and all(
            recorded['path'] == str(path) and recorded['identity'] == file_identity(path)
            for path, recorded in zip(outputs, state['outputs']))
    except (OSError, ValueError, KeyError, TypeError):
        return False
