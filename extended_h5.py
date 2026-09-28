"""Read-only training input policy for pandora-eval-1 H5 files.

Keep detector inputs independently of truth availability. PFO tables are never
modified; only feature/label/row_info/event/collections are read for training.
"""
import json

import awkward as ak
import h5py
import numpy as np


ECAL_GAP_COLLECTIONS = frozenset((
    'EcalBarrelCollectionGapHits', 'EcalEndcapsCollectionGapHits'))
ROW_COLUMNS = ['kind', 'collection', 'element', 'detector', 'legacy_row',
               'truth_valid', 'feature_valid', 'pfo_constituent',
               'legacy_model_domain', 'signed_object_id', 'mc_object_id']


def gap_mask(row_info, collections):
    ids = [i for i, name in enumerate(collections) if name in ECAL_GAP_COLLECTIONS]
    return (row_info[:, 0] == 0) & np.isin(row_info[:, 1], ids)


def read_training_bundle(path):
    """Load only the builders needed for supervised training, without LCIO."""
    with h5py.File(path, 'r') as f:
        if f.attrs.get('schema_version') != 'pandora-eval-1':
            raise ValueError(f'{path}: --extended-h5-input requires pandora-eval-1')
        metadata = json.loads(f.attrs['metadata'])
        if metadata.get('columns', {}).get('row_info') != ROW_COLUMNS:
            raise ValueError(f'{path}: unsupported row_info column definitions')
        arrays = {}
        for name in ('feature', 'label', 'row_info', 'collections', 'event'):
            g = f[name]
            arrays[name] = ak.from_buffers(json.loads(g.attrs['form']), int(g.attrs['length']),
                                          {k: v[:] for k, v in g.items()})
    if len({len(a) for a in arrays.values()}) != 1:
        raise ValueError(f'{path}: event counts differ between training builders')
    return arrays


def validate_rows(feat, label, row_info, collections):
    from dataset import checked_int64_ids
    n = len(feat)
    if feat.shape != (n, 13) or label.shape != (n, 9) or row_info.shape != (n, 11):
        raise ValueError('Extended H5 feature/label/row_info shapes or row counts differ')
    checked_int64_ids(row_info, 'row_info')
    checked_int64_ids(label[:, :2], 'label IDs')
    if not np.isin(row_info[:, 0], [0, 1]).all() or not np.isin(row_info[:, 5:9], [0, 1]).all():
        raise ValueError('Invalid input kind/validity flags')
    if np.any((row_info[:, 1] < 0) | (row_info[:, 1] >= len(collections))):
        raise ValueError('Unknown input collection')
    if not np.array_equal(row_info[:, 0], feat[:, 5]):
        raise ValueError('Input kind differs between feature and row_info')
    known = row_info[:, 5] > 0
    if not np.array_equal(known, label[:, 1] >= 0) or np.any(label[~known, 1] != -1):
        raise ValueError('Truth validity and label IDs disagree')
    # A corrupt teacher must not silently become a background label.
    if not np.isfinite(label[known]).all():
        raise ValueError('Non-finite known truth label')


def training_event(bundle, index, ds, event_index):
    """Apply collection selection, then the same valid-input policy as inference."""
    from dataset import ILCDataset
    def matrix(name, width):
        values = np.array(ak.to_numpy(bundle[name][index]), copy=True)
        return values.reshape(0, width) if values.size == 0 else values
    feat, label, info = matrix('feature', 13), matrix('label', 9), matrix('row_info', 11)
    collections = ak.to_list(bundle['collections'][index])
    validate_rows(feat, label, info, collections)
    gap = gap_mask(info, collections) if ds.exclude_gap_hits else np.zeros(len(feat), dtype=bool)
    rows = np.flatnonzero(~gap)
    event_e = jet_e = None
    if ds.event_energy:
        event_e, jet_e = ILCDataset.decode_event_kinematics(bundle['event'][index])
    data = ILCDataset.featurize_from_numpy(feat[rows], label[rows], None,
        event_e, jet_e, event_index, ds, row_info=info[rows])
    data.input_row = data.input_row.new_tensor(rows[data.input_row.numpy()])
    # Integer diagnostics collate per event; no changes to the stored H5.
    data.n_stored_inputs = data.input_row.new_tensor([len(feat)])
    data.n_excluded_gap_hits = data.input_row.new_tensor([int(gap.sum())])
    data.n_invalid_inputs = data.input_row.new_tensor([len(rows) - len(data.x)])
    return data


def validate_dataset_options(extended, exclude_gap, timing_cut=False, mctpe=False, pandora=False,
                             test_mode=True):
    if exclude_gap and not extended:
        raise ValueError('--exclude-gap-hits requires --extended-h5-input for training')
    if extended and (timing_cut or mctpe or pandora or not test_mode):
        raise ValueError('Extended training uses test_mode=True and reconstructed inputs; '
                         'timing-cut, mctpe and legacy pandora output are unsupported')


def add_training_arguments(parser):
    parser.add_argument('--extended-h5-input', action='store_true',
        help='Read pandora-eval-1 row_info; retain valid unlabelled inputs and mask their supervision')
    parser.add_argument('--exclude-gap-hits', action='store_true',
        help='Exclude the two ECAL GapHits collections from model inputs; keep H5 unchanged')


def validate_training_arguments(args):
    extended = getattr(args, 'extended_h5_input', False)
    validate_dataset_options(extended, getattr(args, 'exclude_gap_hits', False),
        args.timing_cut, args.mctpe)
    if extended and (args.jit or args.dp or args.energy_branch or args.energy_regression_weight):
        raise ValueError('Extended H5 training supports eager single-device/DDP; '
                         'jit, dp, energy-branch and energy-regression-weight are unsupported')
