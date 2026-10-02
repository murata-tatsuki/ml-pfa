"""CPU tests for unlabelled inputs, gap selection and supervised-loss masks."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import awkward as ak
import h5py
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

import objectcondensation as oc
from cluster_energy import ClusterEnergyHead, pooled_energy_loss, pooled_supervision_mask
from dataset import ILCDataset
from dataset_ilc_sharded import ILCDatasetSharded
from dataset_ilc_streaming import ILCStreamingDataset
from extended_h5 import ROW_COLUMNS, read_training_bundle, training_event
from pandora_eval_data import model_data
from supervised_loss import calc_supervised_loss


def fixture(unknown_only=False):
    rng = np.random.default_rng(2)
    n = 64
    feat = np.zeros((n, 13)); feat[:, :4] = rng.uniform(.01, .1, (n, 4))
    feat[[0, 32], 5] = 1; feat[[0, 32], 0] = 0; feat[[0, 32], 7:10] = [1., 2., 3.]
    label = np.zeros((n, 9)); label[:, 1] = -1
    label[:16, 1] = 0; label[16:32, 1] = 1
    label[:32, 0] = np.arange(1, 33); label[0, 0] = -1
    label[:32, 2] = 211; label[:16, 7] = 10; label[16:32, 7] = 20
    row = np.zeros((n, 11)); row[:, 0] = feat[:, 5]
    row[:, 1] = np.where(feat[:, 5] > .5, 1, 0); row[-3:-1, 1] = [2, 3]
    row[:, 2] = np.arange(n); row[:, 3] = np.where(feat[:, 5] > .5, 7, 1)
    row[:, 4] = -1; row[:, 5] = label[:, 1] >= 0; row[:, 6] = 1
    row[:, 9] = np.where(feat[:, 5] > .5, -1, 1) * np.arange(1, n + 1)
    row[:, 10] = np.where(row[:, 5] > 0, label[:, 1] + 100, -1)
    feat[-1, 2] = np.nan; row[-1, 6] = 0
    if unknown_only:
        label[:] = 0; label[:, 1] = -1; row[:, 5] = 0; row[:, 10] = -1
    return dict(feature=feat, label=label, row_info=row, event=np.zeros(10),
        collections=['EcalBarrelCollectionRec', 'MarlinTrkTracks',
                     'EcalBarrelCollectionGapHits', 'EcalEndcapsCollectionGapHits'], global_index=0)


def write_fixture(path, events, schema='pandora-eval-1'):
    with h5py.File(path, 'w') as f:
        f.attrs['schema_version'] = schema
        metadata = dict(columns=dict(row_info=ROW_COLUMNS))
        definitions = {'nnqq-2m-eval-1': 'higgs-direct-qq-terminal-nu-v1',
                       'single-particle-eval-1': 'single-primary-terminal-nu-v1'}
        if schema in definitions:
            metadata['event_definition'] = dict(version=definitions[schema])
        f.attrs['metadata'] = json.dumps(metadata)
        for name in ('feature', 'label', 'row_info', 'collections', 'event'):
            rows = [e[name].tolist() if isinstance(e[name], np.ndarray) else e[name] for e in events]
            form, length, buffers = ak.to_buffers(ak.Array(rows))
            g = f.create_group(name); g.attrs['form'] = form.to_json(); g.attrs['length'] = json.dumps(length)
            for k, v in buffers.items():
                g.create_dataset(k, data=v)
        # Independent reference data must survive training reads byte-for-byte.
        f.create_dataset('pfo_reference_sentinel', data=[123.5, 99.])


def settings():
    s = SimpleNamespace(thetaphi=True, momentum=True, momentumAmp=True,
        max_momentum=3., mctpe=False, test_mode=True, pandora=False, event_energy=False,
        noise_index=-1, exclude_gap_hits=True)
    s.shaper_tanh = lambda x, a, b, c, d: a * np.tanh(b * (x-c)) + d
    return s


class ExtendedTrainingTests(unittest.TestCase):
    def test_nnqq_training_schema_uses_same_row_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'nnqq.h5'
            write_fixture(path, [fixture()], schema='nnqq-2m-eval-1')
            bundle = read_training_bundle(path)
            self.assertEqual(len(bundle['feature']), 1)
            self.assertEqual(len(training_event(bundle, 0, settings(), 0).x), 61)
            with h5py.File(path, 'r+') as f:
                f.attrs['metadata'] = json.dumps(dict(columns=dict(row_info=[])))
            with self.assertRaisesRegex(ValueError, 'row_info'):
                read_training_bundle(path)

    def test_supported_schemas_preserve_training_inputs_and_file(self):
        expected = model_data(fixture(), exclude_gap_hits=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.h5'
            for schema in ('pandora-eval-1', 'nnqq-2m-eval-1', 'single-particle-eval-1'):
                with self.subTest(schema=schema):
                    write_fixture(path, [fixture()], schema=schema)
                    before = hashlib.sha256(path.read_bytes()).hexdigest()
                    actual = training_event(read_training_bundle(path), 0, settings(), 0)
                    for key in ('x', 'y', 'feat', 'label', 'input_row', 'truth_valid', 'hitid'):
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                    self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_extended_event_definition_is_required_and_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.h5'
            for schema in ('nnqq-2m-eval-1', 'single-particle-eval-1'):
                for definition in (None, {}, {'version': 'incorrect-v1'}):
                    with self.subTest(schema=schema, definition=definition):
                        write_fixture(path, [fixture()], schema=schema)
                        with h5py.File(path, 'r+') as f:
                            metadata = json.loads(f.attrs['metadata'])
                            if definition is None:
                                del metadata['event_definition']
                            else:
                                metadata['event_definition'] = definition
                            f.attrs['metadata'] = json.dumps(metadata)
                        with self.assertRaisesRegex(ValueError, 'event_definition'):
                            read_training_bundle(path)

    def test_extended_event_length_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.h5'
            for schema in ('nnqq-2m-eval-1', 'single-particle-eval-1'):
                for width in (9, 11):
                    with self.subTest(schema=schema, width=width):
                        event = fixture(); event['event'] = np.zeros(width)
                        write_fixture(path, [event], schema=schema)
                        with self.assertRaisesRegex(ValueError, 'must contain 10 values'):
                            read_training_bundle(path)

    def test_single_particle_padding_is_checked_without_restricting_nnqq(self):
        event = fixture(); event['event'][4] = 1.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.h5'
            write_fixture(path, [event], schema='single-particle-eval-1')
            with self.assertRaisesRegex(ValueError, 'padding must be zero'):
                read_training_bundle(path)
            write_fixture(path, [event], schema='nnqq-2m-eval-1')
            self.assertEqual(ak.to_list(read_training_bundle(path)['event'])[0][4], 1.)

    def test_unknown_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.h5'
            write_fixture(path, [fixture()], schema='unknown-eval-1')
            with self.assertRaisesRegex(ValueError, 'unsupported training H5 schema'):
                read_training_bundle(path)

    def setUp(self):
        torch.manual_seed(3); torch.set_num_threads(1)

    def test_streaming_and_sharded_match_inference_preserve_unknowns_and_file(self):
        e = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'a.h5'; write_fixture(path, [e, fixture(True)])
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            kw = dict(thetaphi=True, momentum=True, momentumAmp=True, test_mode=True,
                      extended_h5_input=True, exclude_gap_hits=True)
            sharded = ILCDatasetSharded(str(path), **kw)
            streaming = list(ILCStreamingDataset(str(path), shuffle=False, **kw))
            expected = model_data(e, exclude_gap_hits=True)
            for actual in (sharded[0], streaming[0]):
                for key in ('x', 'y', 'feat', 'label', 'input_row', 'truth_valid', 'hitid'):
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                self.assertEqual(len(actual.x), 61)
                self.assertEqual(int((~actual.truth_valid).sum()), 29)
                self.assertEqual(actual.n_excluded_gap_hits.item(), 2)
                self.assertEqual(actual.n_invalid_inputs.item(), 1)
                self.assertTrue(torch.any((~actual.truth_valid) & (actual.x[:, 4] == 1)))
            self.assertEqual(len(streaming[1].x), 61)
            self.assertFalse(streaming[1].truth_valid.any())
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_only_input_validity_filters_not_truth_or_pfo_membership(self):
        e = fixture(); a = model_data(e, exclude_gap_hits=True)
        changed = copy.deepcopy(e)
        changed['row_info'][:32, 5] = 0; changed['label'][:32, 1] = -1
        changed['row_info'][:, 7] = 1
        b = model_data(changed, exclude_gap_hits=True)
        torch.testing.assert_close(a.x[a.input_row.argsort()], b.x[b.input_row.argsort()])
        self.assertEqual(set(a.input_row.tolist()), set(b.input_row.tolist()))
        for invalid in (np.inf, np.nan, 1e100):
            bad = copy.deepcopy(e); bad['feature'][10, 0] = invalid
            # Energy tanh maps Inf to a finite value: raw validity must still reject it.
            self.assertNotIn(10, model_data(bad, exclude_gap_hits=True).input_row.tolist())

    def loss_inputs(self):
        b = torch.tensor([0]*4 + [1]*3 + [2]*4)
        known = torch.tensor([1,1,1,0, 0,0,0, 1,1,1,0], dtype=torch.bool)
        ids = torch.tensor([1,1,2,0, 0,0,0, 1,1,2,0])
        beta = torch.rand(11, requires_grad=True)
        coords = torch.randn(11, 2, requires_grad=True)
        energy = torch.full((11,), 10.); energy[~known] = float('nan')
        kw = dict(cluster_track_index=torch.tensor([1,0,0,0, 0,0,1, 1,0,0,0]),
            tracker_energy=torch.rand(11, requires_grad=True), detected_energy=torch.rand(11),
            pred_cluster_energy=torch.rand(11, requires_grad=True),
            mcpdg=torch.ones(11)*211, mccharge=torch.ones(11),
            force_track_alpha=True, beta_track_term=True, beta_term_option='short-range-potential',
            LE_track='alpha_tracker_diff_log_perCluster', LE_cluster='sum_log_perCluster',
            Ecl_regression=True, epoch=4)
        return beta, coords, ids, energy, b, known, kw

    def test_loss_matches_labelled_only_and_ignores_unknown_outputs(self):
        beta, coords, ids, energy, b, known, kw = self.loss_inputs()
        actual = calc_supervised_loss(beta, coords, None, ids, energy, b, truth_valid=known, **kw)
        filtered = {k: v[known] if torch.is_tensor(v) else v for k, v in kw.items()}
        remap = torch.tensor([0,0,0,1,1,1])
        expected = oc.calc_LV_Lbeta(beta[known], coords[known], None, ids[known], energy[known], remap, **filtered)
        for x, y in zip(actual[:4], expected[:4]):
            torch.testing.assert_close(x, y * (2/3))
        self.assertEqual(set(actual[4]), set(expected[4]))
        for k in actual[4]:
            torch.testing.assert_close(actual[4][k], expected[4][k] * (2/3))
        sum(actual[:4]).backward()
        for t in (beta, coords, kw['tracker_energy'], kw['pred_cluster_energy']):
            self.assertEqual(t.grad[~known].abs().sum().item(), 0)
        with torch.no_grad():
            beta[~known] = .999; coords[~known] = 999; kw['tracker_energy'][~known] = 1e6
        changed = calc_supervised_loss(beta, coords, None, ids, energy, b, truth_valid=known, **kw)
        for x, y in zip(actual[:4], changed[:4]):
            torch.testing.assert_close(x, y)

    def test_all_unknown_batch_has_differentiable_zero_and_same_component_keys(self):
        beta, coords, ids, energy, b, known, kw = self.loss_inputs()
        baseline = calc_supervised_loss(beta, coords, None, ids, energy, b, truth_valid=known, **kw)
        result = calc_supervised_loss(beta, coords, None, ids*0, energy, b, truth_valid=known & False, **kw)
        self.assertEqual(set(baseline[4]), set(result[4]))
        self.assertTrue(all(v.item() == 0 for v in result[:4]))
        sum(result[:4]).backward()
        for t in (beta, coords, kw['tracker_energy'], kw['pred_cluster_energy']):
            self.assertIsNotNone(t.grad); self.assertEqual(t.grad.abs().sum().item(), 0)

    def test_incomplete_predicted_cluster_is_not_trained_toward_zero(self):
        b = torch.zeros(6, dtype=torch.long); ids = torch.tensor([1,1,0,2,2,0])
        known = ids > 0; tracks = torch.tensor([0,0,0,0,0,1], dtype=torch.bool)
        assignments = torch.tensor([1,1,1,2,2,2])
        e = torch.ones(6); truth = torch.tensor([10.,10.,float('nan'),20.,20.,float('nan')])
        p = ClusterEnergyHead(3)(torch.randn(6,3), b, assignments, tracks, e)
        torch.testing.assert_close(pooled_supervision_mask(p, known), torch.tensor([False,True]))
        p['energy'].retain_grad()
        loss = pooled_energy_loss(p, ids, truth, e, b, tracks, 'sum_log_perCluster', truth_valid=known)
        torch.testing.assert_close(loss, (p['energy'][1]-20).abs().log1p())
        loss.backward(); self.assertEqual(p['energy'].grad[0].item(), 0)
        loss = pooled_energy_loss(p, ids, truth, e, b, tracks, 'log_ratio_mse', truth_valid=known & False)
        self.assertEqual(loss.item(), 0)

    def test_legacy_loss_without_mask_is_unchanged(self):
        beta, coords, ids, energy, b, known, kw = self.loss_inputs()
        kw = {k: v[known] if torch.is_tensor(v) else v for k, v in kw.items()}
        batch = torch.tensor([0,0,0,1,1,1])
        a = oc.calc_LV_Lbeta(beta[known], coords[known], None, ids[known], energy[known], batch, **kw)
        c = calc_supervised_loss(beta[known], coords[known], None, ids[known], energy[known], batch, **kw)
        for x,y in zip(a[:4],c[:4]): torch.testing.assert_close(x,y,rtol=0,atol=0)

    def test_malformed_truth_metadata_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = fixture(); e['row_info'][1,5] = 0
            p = Path(tmp)/'bad.h5'; write_fixture(p,[e])
            with self.assertRaisesRegex(ValueError, 'Truth validity'):
                training_event(read_training_bundle(p),0,settings(),'bad')

    def test_streaming_split_and_multiple_files_preserve_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(4): write_fixture(Path(tmp)/f'{i}.h5',[fixture()])
            ds = ILCStreamingDataset(tmp, test_mode=True, momentum=True,
                extended_h5_input=True, exclude_gap_hits=True, files_per_chunk=2, shuffle=False)
            a,b = ds.split(.5)
            for child in (a,b):
                self.assertTrue(child.extended_h5_input and child.exclude_gap_hits)
                events = list(child); self.assertEqual(len(events),2)
                batch = next(iter(DataLoader(events,batch_size=2)))
                self.assertEqual(len(batch.x),122)
                self.assertEqual(int((~batch.truth_valid).sum()),58)
                self.assertEqual(batch.n_excluded_gap_hits.tolist(),[2,2])

    def test_pretraining_metrics_allow_unlabelled_events_between_labelled_events(self):
        from train import collect_pretraining_responses
        events = [model_data(fixture(True), exclude_gap_hits=True),
                  model_data(fixture(), exclude_gap_hits=True),
                  model_data(fixture(True), exclude_gap_hits=True),
                  model_data(fixture(), exclude_gap_hits=True)]
        data = Batch.from_data_list(events)
        out = torch.zeros(len(data.x), 3)
        regressions = [torch.ones(len(data.x), 1), torch.ones(len(data.x), 1)]
        args = SimpleNamespace(energy_regression=True, use_multihead_model=True,
                               energy_regression_cluster=True)
        result = collect_pretraining_responses(out, regressions, data, args)
        self.assertEqual(len(result['ratio']), 4)
        self.assertTrue(np.isfinite(result['ratio']).all())


if __name__ == '__main__':
    unittest.main()
