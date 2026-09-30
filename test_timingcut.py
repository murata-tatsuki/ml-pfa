"""Regression tests for compact, training-only timing-cut files."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import awkward as ak
import h5py
import numpy as np

from dataset import ILCDataset
from dataset_ilc_sharded import ILCDatasetSharded
from dataset_ilc_streaming import ILCStreamingDataset
from extended_h5 import read_training_bundle
from test_extended_training import fixture, write_fixture
from timingcut import timing_cut_file


class TimingCutTests(unittest.TestCase):
    schema = 'pandora-eval-1'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='test-timingcut-')
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'input.h5'
        self.output = Path(self.tmp.name) / 'output.h5'

    def make_input(self):
        first = fixture()
        first['feature'][:, 4] = 1
        first['feature'][1:4, 4] = [14, 15, 13.999]
        first['feature'][0, 7:9] = [.3, 0]  # boundary track must be removed
        first['feature'][32, 7:9] = [.30001, 0]  # unknown track survives
        first['feature'][-1, 4] = 15  # remove invalid row for exact comparisons
        empty = fixture()
        empty['feature'][:, 4] = 20
        write_fixture(self.source, [first, empty, fixture(True)])
        return [first, empty, fixture(True)]

    def test_extended_exact_legacy_cuts_alignment_and_compact_groups(self):
        self.make_input()
        original = hashlib.sha256(self.source.read_bytes()).hexdigest()
        raw = read_training_bundle(self.source)
        with contextlib.redirect_stdout(io.StringIO()):
            expected_f, expected_l = ILCDataset.timingCut(raw['feature'], raw['label'])
        report = timing_cut_file(self.source, self.output)
        actual = read_training_bundle(self.output)
        with h5py.File(self.output, 'r') as handle:
            self.assertEqual(set(handle), {'feature', 'label', 'row_info', 'collections', 'event'})
            self.assertTrue(handle.attrs['training_only'])
            self.assertEqual(handle.attrs['schema_version'], self.schema)
            self.assertEqual(json.loads(handle.attrs['timing_cut'])['maximum_time'], 14)
        for i in range(3):
            np.testing.assert_equal(ak.to_numpy(actual['feature'][i]), ak.to_numpy(expected_f[i]))
            np.testing.assert_equal(ak.to_numpy(actual['label'][i]), ak.to_numpy(expected_l[i]))
            self.assertEqual(ak.to_list(actual['collections'][i]), ak.to_list(raw['collections'][i]))
            np.testing.assert_equal(ak.to_numpy(actual['event'][i]), ak.to_numpy(raw['event'][i]))
            f = ak.to_numpy(raw['feature'][i])
            mask = (f[:, 4] < 14) & ((f[:, 5] == 0) | (np.linalg.norm(f[:, 7:9], axis=1) > .3))
            np.testing.assert_equal(ak.to_numpy(actual['row_info'][i]).reshape(-1, 11), ak.to_numpy(raw['row_info'][i])[mask])
        self.assertEqual(len(actual['feature'][1]), 0)
        self.assertTrue(ak.any(actual['row_info'][0][:, 5] == 0))
        self.assertTrue(ak.any(actual['row_info'][0][:, 1] == 2))  # gap retained
        self.assertEqual(original, hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertLess(report['rows_after'], report['rows_before'])
        self.assertEqual(list(Path(self.tmp.name).glob('.timingcut-*')), [])

    def test_sharded_and_streaming_training_read_output(self):
        self.make_input()
        timing_cut_file(self.source, self.output)
        kw = dict(thetaphi=True, momentum=True, momentumAmp=True, test_mode=True,
                  extended_h5_input=True, exclude_gap_hits=True)
        sharded = ILCDatasetSharded(str(self.output), **kw)
        streaming = list(ILCStreamingDataset(str(self.output), shuffle=False, **kw))
        self.assertEqual(len(sharded), 2)
        self.assertEqual(len(streaming), 2)
        for a, b in zip(sharded, streaming):
            np.testing.assert_equal(a.x.numpy(), b.x.numpy())
            np.testing.assert_equal(a.truth_valid.numpy(), b.truth_valid.numpy())
            self.assertGreater(int((~a.truth_valid).sum()), 0)
            self.assertEqual(int(a.n_excluded_gap_hits), 2)

    def test_event_slice_empty_outputs_and_no_stale_buffers(self):
        self.make_input()
        for start, stop, count in [(1, 2, 1), (2, 3, 1), (0, 0, 0), (9, -1, 0)]:
            timing_cut_file(self.source, self.output, nstart=start, nend=stop)
            arrays = read_training_bundle(self.output)
            self.assertTrue(all(len(a) == count for a in arrays.values()))
            if start != 2:
                self.assertEqual(int(ak.sum(ak.num(arrays['feature'], axis=1))), 0)
                with h5py.File(self.output, 'r') as handle:
                    data_buffers = [value for key, value in handle['feature'].items() if key.endswith('-data')]
                    self.assertTrue(all(value.size == 0 for value in data_buffers))

    def test_legacy_input_preserves_event_but_drops_unused_groups(self):
        self.make_input()
        with h5py.File(self.source, 'a') as handle:
            del handle.attrs['schema_version']
            del handle.attrs['metadata']
            del handle['row_info']
            del handle['collections']
            handle.create_group('pandora')  # invalid unused builder must not be read
        timing_cut_file(self.source, self.output, maximum_time=13, minimum_pt=.1)
        with h5py.File(self.output, 'r') as handle:
            self.assertEqual(set(handle), {'feature', 'label', 'event'})
        from tools.load_awkward import load_awkward2
        arrays = load_awkward2(self.output)
        self.assertEqual(len(arrays[0]), 3)
        self.assertIsNotNone(arrays[6])

    def test_reject_same_file_and_bad_alignment_without_touching_output(self):
        self.make_input()
        original = self.source.read_bytes()
        with self.assertRaises(ValueError):
            timing_cut_file(self.source, self.source)
        self.assertEqual(self.source.read_bytes(), original)
        self.output.write_bytes(b'existing output')
        with h5py.File(self.source, 'a') as handle:
            handle['row_info'].attrs['length'] = '2'
        with self.assertRaises(ValueError):
            timing_cut_file(self.source, self.output)
        self.assertEqual(self.output.read_bytes(), b'existing output')

    def test_no_event_legacy_input_and_invalid_parameters(self):
        self.make_input()
        with h5py.File(self.source, 'a') as handle:
            for name in ('event', 'row_info', 'collections'):
                del handle[name]
            del handle.attrs['schema_version']
            del handle.attrs['metadata']
        timing_cut_file(self.source, self.output)
        with h5py.File(self.output, 'r') as handle:
            self.assertEqual(set(handle), {'feature', 'label'})
        for kwargs in (dict(nstart=-1), dict(nstart=2, nend=1),
                       dict(maximum_time=float('nan')), dict(minimum_pt=-1)):
            with self.assertRaises(ValueError):
                timing_cut_file(self.source, self.output, **kwargs)


class NnqqTimingCutTests(TimingCutTests):
    """Run the cut/loader/legacy regression contract for the nnqq schema too."""
    schema = 'nnqq-2m-eval-1'

    def make_input(self):
        events = super().make_input()
        with h5py.File(self.source, 'a') as handle:
            handle.attrs['schema_version'] = self.schema
            metadata = json.loads(handle.attrs['metadata'])
            metadata['event_definition'] = {
                'version': 'higgs-direct-qq-terminal-nu-v1',
                'event_8': 'Q - N_H_all + N_H_sim', 'event_9': 'Q - N_H_all'}
            handle.attrs['metadata'] = json.dumps(metadata)
            del handle['event']
            values = np.arange(30, dtype=float).reshape(3, 10) + 100
            form, length, buffers = ak.to_buffers(ak.Array(values))
            group = handle.create_group('event')
            group.attrs['form'] = form.to_json()
            group.attrs['length'] = json.dumps(length)
            for key, value in buffers.items():
                group.create_dataset(key, data=value)
        return events

    def test_preserve_nnqq_metadata_and_event_energy_definitions(self):
        self.make_input()
        timing_cut_file(self.source, self.output)
        with h5py.File(self.source) as src, h5py.File(self.output) as dst:
            self.assertEqual(src.attrs['metadata'], dst.attrs['metadata'])
        ds = ILCDatasetSharded(str(self.output), test_mode=True,
            extended_h5_input=True, exclude_gap_hits=True, event_energy=True)
        # Event energy decoding must preserve nnqq's supplied values, not
        # replace them with fixed-uds or sqrt(s)/2 assumptions.
        bundle = read_training_bundle(self.output)
        visible, partons = ILCDataset.decode_event_kinematics(bundle['event'][0])
        np.testing.assert_equal(visible, [108., 109.])
        np.testing.assert_equal(partons, np.arange(100., 108.).reshape(2, 4))
        self.assertGreater(len(ds[0].x), 0)

    def test_reject_unknown_schema_and_incompatible_nnqq_metadata(self):
        for change, pattern in [('schema', 'schema'), ('event', 'event_definition'),
                                ('columns', 'row_info')]:
            with self.subTest(change=change):
                self.make_input()
                with h5py.File(self.source, 'a') as handle:
                    metadata = json.loads(handle.attrs['metadata'])
                    if change == 'schema':
                        handle.attrs['schema_version'] = 'nnqq-2m-eval-999'
                    elif change == 'event':
                        metadata['event_definition']['version'] = 'unknown'
                    else:
                        metadata['columns']['row_info'] = ['wrong']
                    handle.attrs['metadata'] = json.dumps(metadata)
                self.output.write_bytes(b'unchanged')
                with self.assertRaisesRegex(ValueError, pattern):
                    timing_cut_file(self.source, self.output)
                with self.assertRaisesRegex(ValueError, pattern):
                    read_training_bundle(self.source)
                self.assertEqual(self.output.read_bytes(), b'unchanged')

    def test_reject_wrong_nnqq_event_width(self):
        self.make_input()
        with h5py.File(self.source, 'a') as handle:
            form = json.loads(handle['event'].attrs['form'])
            form['inner_shape'] = [9]
            handle['event'].attrs['form'] = json.dumps(form)
        with self.assertRaisesRegex(ValueError, '10 values'):
            timing_cut_file(self.source, self.output)


if __name__ == '__main__':
    unittest.main()
