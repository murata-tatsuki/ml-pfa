"""Event-preserving reader for KEK pandora-eval-1 H5 files.

The complete PFO tables stay outside the model's input mask. No generator
installation or pyLCIO is needed on the analysis server.
"""
import glob
import json
from pathlib import Path
from types import SimpleNamespace
import awkward as ak
import h5py
import numpy as np
from dataset import ILCDataset, checked_int64_ids

WIDTHS = dict(feature=13, label=9, pandora=18, row_info=11, pfo=9,
              pfo_links=5, truth_particles=7, cluster=2, event=10, event_eval=23)
GROUPS = set(WIDTHS) | {'source', 'collections'}
COLUMNS = {
    'row_info': ['kind','collection','element','detector','legacy_row','truth_valid','feature_valid',
                 'pfo_constituent','legacy_model_domain','signed_object_id','mc_object_id'],
    'pfo': ['index','object_id','energy','type','px','py','pz','n_tracks','n_clusters'],
    'pfo_links': ['pfo_index','kind','object_id','input_row','cluster_id'],
    'truth_particles': ['label_id','object_id','pdg','energy','px','py','pz'],
    'event_eval': ['pfo_energy_sum','n_pfos','mc_energy_enu','thrust','q_pdg','pfo_valid','mc_valid',
                   'quark_valid','legacy_index','source_index','run_number','event_number',
                   'legacy_event_valid','legacy_hits','added_hits','legacy_tracks','added_tracks',
                   'pfo_energy_sum_f32','mc_energy_enu_f32','h5_index','event_valid',
                   'legacy_columns_checked','raw_id_offset'],
}

def input_paths(pattern):
    p = Path(pattern).expanduser()
    paths = sorted(p.glob('*.h5')) if p.is_dir() else [Path(x) for x in sorted(glob.glob(str(p)))]
    if not paths:
        raise ValueError(f'No H5 inputs: {pattern}')
    return paths

def load_file(path):
    with h5py.File(path, 'r') as f:
        if f.attrs.get('schema_version') != 'pandora-eval-1':
            raise ValueError(f'{path}: requires pandora-eval-1 (old H5 cannot supply full PFOs)')
        if set(f) != GROUPS:
            raise ValueError(f'{path}: unexpected/missing builders {set(f) ^ GROUPS}')
        metadata = json.loads(f.attrs['metadata'])
        for name, columns in COLUMNS.items():
            if metadata.get('columns', {}).get(name) != columns:
                raise ValueError(f'{path}: unsupported {name} column definitions')
        arrays = {name: ak.from_buffers(json.loads(g.attrs['form']), int(g.attrs['length']),
                  {k: v[:] for k, v in g.items()}) for name, g in f.items()}
    if len({len(x) for x in arrays.values()}) != 1:
        raise ValueError('Event counts differ between builders')
    return arrays, metadata

def validate_event(e):
    for k, width in WIDTHS.items():
        expected = (width,) if k in ('event', 'event_eval') else (len(e[k]), width)
        if e[k].shape != expected:
            raise ValueError(f'{k}: shape {e[k].shape}, expected {expected}')
    if len({len(e[k]) for k in ('feature', 'label', 'pandora', 'row_info')}) != 1:
        raise ValueError('Hit/track builder lengths differ')
    r, p, ev = e['row_info'], e['pfo'], e['event_eval']
    if not np.isin(r[:, 0], [0, 1]).all() or not np.isin(r[:, 5:9], [0, 1]).all():
        raise ValueError('Invalid input kind/validity flag')
    if np.any((r[:, 1] < 0) | (r[:, 1] >= len(e['collections']))):
        raise ValueError('Unknown input collection')
    if np.any(e['label'][r[:, 5] == 0, 1] != -1):
        raise ValueError('Unknown truth must use label -1')
    for k, cols in [('row_info', range(11)), ('pfo', [0, 1, 3, 7, 8]),
                    ('pfo_links', range(5)), ('label', [0, 1])]:
        checked_int64_ids(e[k][:, cols], k)
    if not np.array_equal(r[:, 0], e['feature'][:, 5]) or not np.array_equal(r[:, 9], e['pandora'][:, 9]):
        raise ValueError('Input row identity mismatch')
    if not np.array_equal(r[:, 5] > 0, e['label'][:, 1] >= 0):
        raise ValueError('Truth validity mismatch')
    if len(set(map(tuple, r[:, :3]))) != len(r):
        raise ValueError('Duplicate input collection/element identity')
    if not np.array_equal(p[:, 0], np.arange(len(p))) or len(p) != int(ev[1]):
        raise ValueError('PFO index/count mismatch')
    if not np.isclose(p[:, 2].sum(), ev[0], rtol=1e-12, atol=1e-10):
        raise ValueError('PFO energy sum mismatch')
    truth_ids = set(e['truth_particles'][:, 0])
    if len(truth_ids) != len(e['truth_particles']) or not np.array_equal(e['cluster'][:, 0], e['truth_particles'][:, 0]):
        raise ValueError('Truth-particle/cluster identity mismatch')
    if not set(e['label'][r[:, 5] > 0, 1]).issubset(truth_ids):
        raise ValueError('Unknown truth label reference')
    for pi, kind, obj, row, ci in e['pfo_links'].astype(np.int64):
        if not (0 <= pi < len(p) and 0 <= row < len(r)):
            raise ValueError('PFO link out of bounds')
        if r[row, 0] != kind or r[row, 9] != (-obj if kind else obj):
            raise ValueError('PFO link row identity mismatch')

def iter_events(pattern, start=0, stop=-1):
    if start < 0 or (stop != -1 and stop < start):
        raise ValueError('Require 0 <= start <= stop; stop=-1 means all events')
    seen, index = set(), 0
    baseline_settings = None
    for path in input_paths(pattern):
        arrays, metadata = load_file(path)
        settings = (metadata['analysis_settings'], metadata['input_inventory'])
        if baseline_settings is not None and settings != baseline_settings:
            raise ValueError('Cannot mix analysis settings/input inventories in one output')
        baseline_settings = settings
        for i in range(len(arrays['event_eval'])):
            if stop >= 0 and index >= stop:
                return
            global_index = index
            index += 1
            if global_index < start:
                continue
            event = {k: ak.to_numpy(arrays[k][i]) for k in WIDTHS}
            event.update(source=ak.to_list(arrays['source'][i]),
                         collections=ak.to_list(arrays['collections'][i]),
                         metadata=metadata, input_path=str(path.resolve()), input_index=i,
                         global_index=global_index)
            validate_event(event)
            if event['event_eval'][19] != i:
                raise ValueError('Stored H5 event index does not match row position')
            key = (event['source']['id'], int(event['event_eval'][9]))
            if key in seen:
                raise ValueError(f'Duplicate source event: {key}')
            seen.add(key)
            yield event

ECAL_GAP_COLLECTIONS = frozenset(('EcalBarrelCollectionGapHits', 'EcalEndcapsCollectionGapHits'))

def gap_hit_mask(event):
    """Identify gap calorimeter hits by collection, independently of MC truth."""
    ids = [i for i, name in enumerate(event['collections']) if name in ECAL_GAP_COLLECTIONS]
    return (event['row_info'][:, 0] == 0) & np.isin(event['row_info'][:, 1], ids)

def model_data(event, input_dim=7, momentum=True, momentum_amp=True, exclude_gap_hits=False):
    if input_dim not in (5, 7):
        raise ValueError('Base input_dim must be 5 or 7, before momentum features')
    settings = SimpleNamespace(thetaphi=input_dim == 7, momentum=momentum,
        momentumAmp=momentum_amp, max_momentum=3. if momentum else 1.,
        mctpe=False, test_mode=True, pandora=False, event_energy=False,
        noise_index=-1)
    # Bind the legacy shaper without constructing/loading a legacy dataset.
    settings.shaper_tanh = lambda x, a, b, c, d: a * np.tanh(b * (x-c)) + d
    rows = np.flatnonzero(~gap_hit_mask(event)) if exclude_gap_hits else np.arange(len(event['feature']))
    data = ILCDataset.featurize_from_numpy(event['feature'][rows].copy(), event['label'][rows].copy(),
        None, None, None, event['global_index'], settings, row_info=event['row_info'][rows])
    # Remap after the validity mask and truth-label ordering to original H5 rows.
    data.input_row = data.input_row.new_tensor(rows[data.input_row.numpy()])
    return data
