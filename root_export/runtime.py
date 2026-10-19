"""Output identity and atomic-publication helpers for the unified ROOT exporter.

Only validated outputs with the same input, checkpoint and physics settings are
reused. Worker counts, compression chunks and GPU placement do not change the
identity of an analysis result.
"""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METADATA_KEY = 'save_root_metadata'
FORMAT_VERSION = 'cedric-unified-1'
PHYSICS_OPTIONS = (
    'nstart', 'nend', 'timingCut', 'input_dim', 'output_dim', 'pandora',
    'event_total_energy', 'energy_regression', 'energy_regression_cluster',
    'energy_regression_weight', 'momentum', 'momentum_amp', 'mctpe',
    'energy_branch', 'tbeta', 'td', 'model_variant', 'truth_clustering',
    'pid_readout', 'exclude_gap_hits', 'detector_one_hot', 'pid_detail',
    'pid_input_format', 'cluster_energy_source', 'cluster_energy_tbeta',
    'cluster_energy_td', 'energy_heads', 'output_schema',
    'particle_cluster_score_threshold',
)


@lru_cache(maxsize=64)
def _hash_file(path, size, modified_ns):
    del size, modified_ns  # They invalidate cached hashes when a file changes.
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha256(path):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return _hash_file(str(path), stat.st_size, stat.st_mtime_ns)


def _input_identity(input_path):
    from .cedric import _resolve_multi_h5

    result = []
    for path in _resolve_multi_h5(str(input_path)):
        path = path.resolve()
        stat = path.stat()
        result.append(dict(path=str(path), size=stat.st_size,
                           modified_ns=stat.st_mtime_ns))
    return result


def output_signature(args, input_path):
    """Describe result-affecting settings; deliberately omit execution tuning."""
    settings = {name: getattr(args, name, None) for name in PHYSICS_OPTIONS}
    timing_cut = settings['timingCut']
    if isinstance(timing_cut, str):
        timing_cut = timing_cut.lower() in ('1', 'true', 'yes', 'on')
    settings['timingCut'] = bool(timing_cut)
    sources = (
        'save_root.py', 'root_export/cedric.py', 'root_export/runtime.py',
        'root_export/scan.py', 'root_export/clustering.py', 'model.py',
        'gravnet_model.py', 'gravnet_conv.py', 'dataset.py', 'event.py',
        'prediction.py', 'test_yielder_edit.py', 'clustering.py',
        'matching.py', 'objectcondensation.py', 'particle_heads.py',
        'cluster_energy.py', 'particle_cluster.py', 'extended_h5.py', 'pandora_eval_data.py',
        'pandora_eval_reconstruction.py', 'sed.py', 'tools/load_awkward.py',
        'scan_fast_root_writer.cxx',
    )
    return dict(
        version=FORMAT_VERSION,
        inputs=_input_identity(input_path),
        checkpoint_sha256=None if args.pandora else sha256(args.ckpt),
        settings=settings,
        implementation={name: sha256(PROJECT_ROOT / name) for name in sources
                        if (PROJECT_ROOT / name).is_file()},
    )


def read_metadata(path):
    import ROOT

    handle = ROOT.TFile.Open(str(path), 'READ')
    if not handle or handle.IsZombie():
        if handle:
            handle.Close()
        return None
    try:
        item = handle.Get(METADATA_KEY)
        if not item or not item.InheritsFrom('TObjString'):
            return None
        metadata = json.loads(str(item.GetString()))
        if not isinstance(metadata, dict):
            return None
        actual_entries = {}
        for name in ('t', 'reco', 'prediction', 'event', 'jet'):
            tree = handle.Get(name)
            if not tree or not tree.InheritsFrom('TTree'):
                return None
            actual_entries[name] = int(tree.GetEntries())
        if metadata.get('tree_entries') != actual_entries:
            return None
        return metadata
    finally:
        handle.Close()


def validate_existing(path, args, input_path, expected_counts=None, pid=False):
    """Return True only for a structurally complete, identical analysis result."""
    from . import cedric

    path = Path(path)
    if not path.is_file():
        return False
    try:
        metadata = read_metadata(path)
        if metadata is None or metadata.get('signature') != output_signature(args, input_path):
            return False
        if expected_counts is None:
            entries = metadata['tree_entries']
            expected_counts = tuple(entries[name] for name in ('event', 'prediction', 'jet'))
        result = cedric._validate_root(path, expected_counts, pid=pid)
        return result.valid
    except (OSError, ValueError, RuntimeError, TypeError, KeyError, AttributeError):
        return False


def write_metadata(path, args, input_path):
    """Append identity after the writer closes, before atomic publication."""
    import ROOT

    handle = ROOT.TFile.Open(str(path), 'UPDATE')
    if not handle or handle.IsZombie():
        if handle:
            handle.Close()
        raise OSError('Cannot add output identity to ' + str(path))
    try:
        entries = {name: int(handle.Get(name).GetEntries())
                   for name in ('t', 'reco', 'prediction', 'event', 'jet')}
        payload = dict(signature=output_signature(args, input_path), tree_entries=entries)
        handle.cd()
        result = ROOT.TObjString(json.dumps(payload, sort_keys=True)).Write(
            METADATA_KEY, ROOT.TObject.kOverwrite)
        if result <= 0:
            raise OSError('Failed to write ' + METADATA_KEY)
    finally:
        handle.Close()
