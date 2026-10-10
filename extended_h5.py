"""Read-only training input policy for supported extended H5 schemas.

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
TRAINING_SCHEMAS = frozenset(('pandora-eval-1', 'nnqq-2m-eval-1', 'single-particle-eval-1'))
NNQQ_EVENT_DEFINITION = 'higgs-direct-qq-terminal-nu-v1'
SINGLE_PARTICLE_EVENT_DEFINITION = 'single-primary-terminal-nu-v1'

# Fixed column order shared by every file/split and recorded in checkpoints.
DETECTOR_CATEGORIES = ((0, 'unknown'), (1, 'ECAL'), (2, 'HCAL'), (3, 'LCAL'),
                       (4, 'LHCAL'), (5, 'MUON'), (7, 'track'))
REGION_CATEGORIES = ('unspecified', 'barrel', 'endcap', 'endcap_ring')
COLLECTION_REGIONS = {
    'EcalBarrelCollectionRec': 1, 'EcalBarrelCollectionGapHits': 1,
    'HcalBarrelCollectionRec': 1,
    'EcalEndcapsCollectionRec': 2, 'EcalEndcapsCollectionGapHits': 2,
    'HcalEndcapsCollectionRec': 2,
    'EcalEndcapRingCollectionRec': 3, 'HcalEndcapRingCollectionRec': 3,
}
DETECTOR_ONE_HOT_DIM = len(DETECTOR_CATEGORIES) + len(REGION_CATEGORIES)


def detector_one_hot_config():
    return dict(version=1, detector_categories=[list(x) for x in DETECTOR_CATEGORIES],
                region_categories=list(REGION_CATEGORIES),
                collection_regions=dict(COLLECTION_REGIONS),
                added_input_dim=DETECTOR_ONE_HOT_DIM)


def detector_one_hot_features(row_info, collections):
    """Encode stored detector identity and explicit collection regions, never truth."""
    from dataset import checked_int64_ids
    if row_info is None or collections is None:
        raise ValueError('--detector-one-hot requires row_info and collections in extended H5')
    ids = checked_int64_ids(row_info[:, [1, 3]], 'collection/detector IDs')
    if np.any((ids[:, 0] < 0) | (ids[:, 0] >= len(collections))):
        raise ValueError('Unknown input collection for --detector-one-hot')
    detector_ids = np.asarray([x[0] for x in DETECTOR_CATEGORIES])
    detector = ids[:, 1, None] == detector_ids[None, :]
    if not detector.any(axis=1).all():
        raise ValueError(f'Unsupported detector IDs for --detector-one-hot: '
                         f'{np.unique(ids[~detector.any(axis=1), 1]).tolist()}')
    # MUON/LCAL/LHCAL/track collections do not encode barrel/endcap. Do not
    # guess their region from coordinates or from their detector category.
    collection_regions = np.asarray([COLLECTION_REGIONS.get(name, 0)
                                     for name in collections], dtype=np.int64)
    region = np.eye(len(REGION_CATEGORIES), dtype=np.float32)[collection_regions[ids[:, 0]]]
    return np.concatenate((detector.astype(np.float32), region), axis=1)


def validate_one_hot_checkpoint(checkpoint, enabled):
    config = checkpoint.get('training_input_config', {})
    saved = config.get('detector_one_hot', False)
    if bool(saved) != bool(enabled):
        raise ValueError('Checkpoint --detector-one-hot setting differs from requested input; '
                         'use matching options or train a new model without --model-ckpt')
    if enabled and config.get('detector_one_hot_config') != detector_one_hot_config():
        raise ValueError('Checkpoint detector one-hot category mapping is incompatible')


def validate_training_schema(handle):
    """Check explicit schema contracts without relabelling the source file.

    nnqq shares the detector/truth rows with fixed uds, but its event energies
    describe the selected Higgs daughters. Preserve that distinction in metadata.
    These event energies are not the per-particle energy targets used by train.py.
    Single-particle event rows store the primary four-vector, four zero padding
    values, and two visible energies. Preserve their own schema and definition.
    """
    schema = handle.attrs.get('schema_version')
    if schema not in TRAINING_SCHEMAS:
        raise ValueError(f'{handle.filename}: unsupported training H5 schema {schema!r}; '
                         f'expected one of {sorted(TRAINING_SCHEMAS)}')
    metadata = json.loads(handle.attrs['metadata'])
    if metadata.get('columns', {}).get('row_info') != ROW_COLUMNS:
        raise ValueError(f'{handle.filename}: unsupported row_info column definitions')
    if schema == 'nnqq-2m-eval-1':
        if metadata.get('event_definition', {}).get('version') != NNQQ_EVENT_DEFINITION:
            raise ValueError(f'{handle.filename}: unsupported nnqq event_definition')
    if schema == 'single-particle-eval-1':
        if metadata.get('event_definition', {}).get('version') != SINGLE_PARTICLE_EVENT_DEFINITION:
            raise ValueError(f'{handle.filename}: unsupported single-particle event_definition')
    return schema


def gap_mask(row_info, collections):
    ids = [i for i, name in enumerate(collections) if name in ECAL_GAP_COLLECTIONS]
    return (row_info[:, 0] == 0) & np.isin(row_info[:, 1], ids)


def read_training_bundle(path):
    """Load only the builders needed for supervised training, without LCIO."""
    with h5py.File(path, 'r') as f:
        schema = validate_training_schema(f)
        arrays = {}
        for name in ('feature', 'label', 'row_info', 'collections', 'event'):
            g = f[name]
            arrays[name] = ak.from_buffers(json.loads(g.attrs['form']), int(g.attrs['length']),
                                          {k: v[:] for k, v in g.items()})
    if len({len(a) for a in arrays.values()}) != 1:
        raise ValueError(f'{path}: event counts differ between training builders')
    if schema == 'nnqq-2m-eval-1' and not ak.all(ak.num(arrays['event'], axis=1) == 10):
        raise ValueError(f'{path}: nnqq event rows must contain 10 values')
    if schema == 'single-particle-eval-1':
        if not ak.all(ak.num(arrays['event'], axis=1) == 10):
            raise ValueError(f'{path}: single-particle event rows must contain 10 values')
        if not ak.all(arrays['event'][:, 4:8] == 0):
            raise ValueError(f'{path}: single-particle event padding must be zero')
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
        event_e, jet_e, event_index, ds, row_info=info[rows], collections=collections)
    data.input_row = data.input_row.new_tensor(rows[data.input_row.numpy()])
    # Integer diagnostics collate per event; no changes to the stored H5.
    data.n_stored_inputs = data.input_row.new_tensor([len(feat)])
    data.n_excluded_gap_hits = data.input_row.new_tensor([int(gap.sum())])
    data.n_invalid_inputs = data.input_row.new_tensor([len(rows) - len(data.x)])
    return data


def validate_dataset_options(extended, exclude_gap, timing_cut=False, mctpe=False, pandora=False,
                             test_mode=True, detector_one_hot=False):
    if detector_one_hot and not extended:
        raise ValueError('--detector-one-hot requires --extended-h5-input for training')
    if exclude_gap and not extended:
        raise ValueError('--exclude-gap-hits requires --extended-h5-input for training')
    if extended and (timing_cut or mctpe or pandora or not test_mode):
        raise ValueError('Extended training uses test_mode=True and reconstructed inputs; '
                         'timing-cut, mctpe and legacy pandora output are unsupported')


def add_training_arguments(parser):
    add_detector_one_hot_argument(parser)
    parser.add_argument('--extended-h5-input', action='store_true',
        help='Read pandora-eval-1, nnqq-2m-eval-1 or single-particle-eval-1 row_info; retain valid unlabelled inputs and mask their supervision')
    parser.add_argument('--exclude-gap-hits', action='store_true',
        help='Exclude the two ECAL GapHits collections from model inputs; keep H5 unchanged')


def add_detector_one_hot_argument(parser):
    parser.add_argument('--detector-one-hot', action='store_true',
        help='Append 7 detector and 4 collection-region one-hot features from stored H5 metadata')


def validate_training_arguments(args):
    extended = getattr(args, 'extended_h5_input', False)
    validate_dataset_options(extended, getattr(args, 'exclude_gap_hits', False),
        args.timing_cut, args.mctpe, detector_one_hot=getattr(args, 'detector_one_hot', False))
    if extended and (args.jit or args.dp or args.energy_branch or args.energy_regression_weight):
        raise ValueError('Extended H5 training supports eager single-device/DDP; '
                         'jit, dp, energy-branch and energy-regression-weight are unsupported')
