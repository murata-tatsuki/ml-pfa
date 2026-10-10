"""Detector features: row alignment, loader parity and checkpoint compatibility."""
import argparse
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from cluster_energy import checkpoint_payload
from dataset_ilc_sharded import ILCDatasetSharded
from dataset_ilc_streaming import ILCStreamingDataset
from extended_h5 import (DETECTOR_ONE_HOT_DIM, add_training_arguments,
                         detector_one_hot_features, validate_dataset_options,
                         validate_one_hot_checkpoint)
from model import get_model
from pandora_eval_data import model_data
from test_extended_training import fixture, write_fixture
from train import build_model, index_setup, load_checkpoint_state, make_ilc_dataset


def training_args(**changes):
    values = dict(timing_cut=False, thetaphi=True, momentum=True, momentum_amp=True,
                  mctpe=False, extended_h5_input=True, exclude_gap_hits=True,
                  detector_one_hot=True, ilc_streaming=False, output_dimension=3,
                  use_charged_cluster_loss=False, energy_regression=False,
                  energy_regression_cluster=False, energy_regression_weight=False,
                  use_multihead_model=False, model_ckpt='', energy_branch=False,
                  multihead_regression_heads=2, multihead_interaction_start_epoch=0,
                  multihead_interaction_mode='none')
    return SimpleNamespace(**dict(values, **changes))


class DetectorOneHotTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)

    def test_fixed_categories_and_collection_lookup(self):
        names = ['unused', 'HcalEndcapRingCollectionRec', 'MUON', 'LCAL',
                 'EcalBarrelCollectionRec', 'EcalEndcapsCollectionGapHits',
                 'LHCAL', 'MarlinTrkTracks', 'UnspecifiedCollection']
        rows = np.zeros((7, 11))
        rows[:, 1] = [8, 4, 1, 3, 6, 2, 7]
        rows[:, 3] = [0, 1, 2, 3, 4, 5, 7]
        encoded = detector_one_hot_features(rows, names)
        np.testing.assert_array_equal(encoded[:, :7], np.eye(7))
        self.assertEqual(encoded[:, 7:].argmax(1).tolist(), [0, 1, 3, 0, 0, 0, 0])
        rows[1, 1] = 5
        self.assertEqual(detector_one_hot_features(rows, names)[1, 7:].argmax(), 2)
        # Collection IDs are event-local; changing their order must not change features.
        reordered = rows.copy(); reordered[:, 1] = len(names) - 1 - rows[:, 1]
        np.testing.assert_array_equal(detector_one_hot_features(rows, names),
                                      detector_one_hot_features(reordered, names[::-1]))
        self.assertEqual(detector_one_hot_features(rows[:0], names).shape, (0, 11))
        with self.assertRaisesRegex(ValueError, 'requires row_info'):
            detector_one_hot_features(None, names)
        rows[0, 3] = 6
        with self.assertRaisesRegex(ValueError, 'Unsupported detector'):
            detector_one_hot_features(rows, names)

    def test_loader_train_validation_split_worker_and_inference_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(2):
                write_fixture(Path(tmp)/f'{i}.h5', [fixture(), fixture(True)])
            hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(tmp).glob('*.h5')}
            expected = model_data(fixture(), exclude_gap_hits=True, detector_one_hot=True)
            baseline = model_data(fixture(), exclude_gap_hits=True)
            torch.testing.assert_close(expected.x[:, :-11], baseline.x, rtol=0, atol=0)
            self.assertEqual(expected.x.shape[1], 22)
            self.assertTrue(torch.all(expected.x[:, -11:-4].sum(1) == 1))
            self.assertTrue(torch.all(expected.x[:, -4:].sum(1) == 1))
            # Check encoding by the *original* row after gap/invalid cuts and label sorting.
            e = fixture()
            for j, original_row in enumerate(expected.input_row.tolist()):
                track = e['feature'][original_row, 5] == 1
                self.assertEqual(int(expected.x[j, -11:-4].argmax()), 6 if track else 1)
                self.assertEqual(int(expected.x[j, -4:].argmax()), 0 if track else 1)
            for streaming in (False, True):
                ds = make_ilc_dataset(training_args(ilc_streaming=streaming), tmp)
                left, right = ds.split(.5)
                for child in (left, right):
                    self.assertTrue(child.detector_one_hot)
                    batches = list(DataLoader(child, batch_size=1, num_workers=1, timeout=30))
                    self.assertEqual(len(batches), 2)
                    known = next(b for b in batches if b.truth_valid.any())
                    for field in ('x', 'y', 'input_row', 'truth_valid', 'feat', 'label'):
                        torch.testing.assert_close(known[field], expected[field], rtol=0, atol=0)
            for p, before in hashes.items():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), before)

    def test_options_require_extended_and_default_is_off(self):
        parser = argparse.ArgumentParser(); add_training_arguments(parser)
        self.assertFalse(parser.parse_args([]).detector_one_hot)
        self.assertTrue(parser.parse_args(['--detector-one-hot']).detector_one_hot)
        with self.assertRaisesRegex(ValueError, 'requires --extended-h5-input'):
            validate_dataset_options(False, False, detector_one_hot=True)
        for cls in (ILCDatasetSharded, ILCStreamingDataset):
            with self.assertRaisesRegex(ValueError, 'requires --extended-h5-input'):
                cls('/nonexistent', detector_one_hot=True)

    def test_forward_backward_and_checkpoint_roundtrip_both_models(self):
        data = Batch.from_data_list([model_data(fixture(), detector_one_hot=True)])
        for multihead in (False, True):
            args = training_args(use_multihead_model=multihead)
            output_dim, _, _, _, extra = index_setup(args)
            self.assertEqual(extra, 4 + DETECTOR_ONE_HOT_DIM)
            model = build_model(args, 7 + extra, output_dim)
            out = (model(data.x, data.batch, return_dict=True)['clustering'] if multihead
                   else model(data.x, data.batch))
            self.assertTrue(torch.isfinite(out).all())
            out.square().mean().backward()
            self.assertTrue(torch.isfinite(model.input.weight.grad).all())
            model.eval()
            with torch.no_grad():
                expected = model(data.x, data.batch, return_dict=True)['clustering'] if multihead else model(data.x, data.batch)
            payload = checkpoint_payload(model, args=args)
            validate_one_hot_checkpoint(payload, True)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'model.pt'; torch.save(payload, path)
                restored = get_model(str(path), jit=False, input_dim=22, output_dim=3,
                                     detector_one_hot=True).eval()
                with torch.no_grad():
                    torch.testing.assert_close(restored(data.x, data.batch), expected)
                with self.assertRaisesRegex(ValueError, 'setting differs'):
                    get_model(str(path), jit=False, input_dim=22, output_dim=3)
                with self.assertRaisesRegex(ValueError, 'setting differs'):
                    load_checkpoint_state(model, str(path))
                load_checkpoint_state(model, str(path), detector_one_hot=True)
            with self.assertRaisesRegex(ValueError, 'setting differs'):
                validate_one_hot_checkpoint({'model': model.state_dict()}, True)
            bad = copy.deepcopy(payload)
            bad['training_input_config']['detector_one_hot_config']['version'] = 999
            with self.assertRaisesRegex(ValueError, 'mapping is incompatible'):
                validate_one_hot_checkpoint(bad, True)
        self.assertEqual(index_setup(training_args(detector_one_hot=False))[-1], 4)
        validate_one_hot_checkpoint({}, False)


if __name__ == '__main__':
    unittest.main()
