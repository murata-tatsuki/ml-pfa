"""Run with python -m unittest test_cluster_energy -v (CPU is sufficient)."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch_geometric.data import Data

from cluster_energy import (ClusterEnergyHead, cluster_membership, deposited_energy_targets,
    pooled_energy_loss, checkpoint_payload, inference_forward, configure_inference,
    validate_training_arguments, POOLED_LOSSES)
from gravnet_model import GravNetModelMultiHead, GravnetModel
from model import get_model


class ClusterEnergyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        torch.set_num_threads(1)

    def test_targets_split_merge_and_event_boundaries(self):
        # Event 0: truth A=10 (deposits 1,3), B=20 (2,2), plus track and noise.
        # Event 1 reuses the SAME truth ID but has a different energy, 100.
        batch = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1])
        truth = torch.tensor([1, 1, 2, 2, 1, 0, 1, 1])
        e = torch.tensor([1., 3., 2., 2., 999., 8., 1., 1.])
        target_e = torch.tensor([10., 10., 20., 20., 10., 0., 100., 100.])
        track = torch.tensor([False, False, False, False, True, False, False, False])
        assigned = torch.tensor([1, 2, 1, 2, 1, 1, 1, 1])
        h = torch.randn(8, 4, requires_grad=True)
        head = ClusterEnergyHead(4)
        pooled = head(h, batch, assigned, track, e)
        target = deposited_energy_targets(pooled, truth, target_e, e, batch, track)
        torch.testing.assert_close(target, torch.tensor([12.5, 17.5, 100.]))
        # Noise energy is never a truth-energy contribution; track deposit is excluded.
        self.assertAlmostEqual(target.sum().item(), 130.)
        assigned[1] = 0  # Missing fragment: do not redistribute its 7.5 elsewhere.
        missing = head(h, batch, assigned, track, e)
        target = deposited_energy_targets(missing, truth, target_e, e, batch, track)
        self.assertAlmostEqual(target.sum().item(), 122.5)

    def test_truth_pooling_target_and_compatibility_sum(self):
        batch = torch.tensor([0, 0, 0, 0])
        ids = torch.tensor([1, 1, 1, 0])
        tracks = torch.tensor([False, False, True, False])
        e = torch.tensor([1., 3., 100., 9.])
        h = torch.randn(4, 6, requires_grad=True)
        pooled = ClusterEnergyHead(6)(h, batch, ids, tracks, e)
        target = deposited_energy_targets(pooled, ids, torch.tensor([8., 8., 8., 0.]), e, batch, tracks)
        torch.testing.assert_close(target, torch.tensor([8.]))
        torch.testing.assert_close(pooled['hit_energy'].sum(), pooled['energy'].sum())
        self.assertEqual(pooled['hit_energy'][2:].tolist(), [0., 0.])
        torch.testing.assert_close(pooled['hit_energy'][1], 3 * pooled['hit_energy'][0])
        pooled['energy'].sum().backward()
        self.assertGreater(h.grad[:2].abs().sum().item(), 0)
        self.assertEqual(h.grad[2:].abs().sum().item(), 0)

    def test_permutation_invariance(self):
        n = 12
        h = torch.randn(n, 5)
        batch = torch.tensor([0] * 6 + [1] * 6)
        ids = torch.tensor([2, 2, 4, 4, 4, 0] * 2)
        tracks = torch.zeros(n, dtype=torch.bool)
        e = torch.rand(n)
        head = ClusterEnergyHead(5).eval()
        before = head(h, batch, ids, tracks, e)
        order = torch.randperm(n)
        after = head(h[order], batch[order], ids[order], tracks[order], e[order])
        torch.testing.assert_close(before['energy'], after['energy'])
        torch.testing.assert_close(before['hit_energy'][order], after['hit_energy'])

    def test_zero_deposits_and_empty_calo_backward(self):
        h = torch.randn(3, 4, requires_grad=True)
        head = ClusterEnergyHead(4)
        b = torch.zeros(3, dtype=torch.long)
        ids = torch.ones(3, dtype=torch.long)
        e = torch.zeros(3)
        track = torch.zeros(3, dtype=torch.bool)
        p = head(h, b, ids, track, e)
        torch.testing.assert_close(p['hit_energy'], p['energy'].expand(3) / 3)
        torch.testing.assert_close(deposited_energy_targets(p, ids, e + 6, e, b, track), torch.tensor([6.]))
        p = head(h, b, ids, ~track, e)
        self.assertEqual(p['energy'].numel(), 0)
        loss = pooled_energy_loss(p, ids, e + 6, e, b, ~track, 'sum_log_perCluster')
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in head.parameters()))

    def test_supported_losses_backward(self):
        for name in POOLED_LOSSES:
            h = torch.randn(6, 3, requires_grad=True)
            b = torch.tensor([0, 0, 0, 1, 1, 1]); ids = torch.ones(6, dtype=torch.long)
            track = torch.zeros(6, dtype=torch.bool); e = torch.ones(6)
            p = ClusterEnergyHead(3)(h, b, ids, track, e)
            loss = pooled_energy_loss(p, ids, e * 5, e, b, track, name)
            self.assertTrue(torch.isfinite(loss), name)
            loss.backward()
            self.assertTrue(torch.isfinite(h.grad).all(), name)

    def test_tiny_positive_deposits_and_loss_normalization(self):
        b = torch.tensor([0, 0, 1, 1])
        ids = torch.ones(4, dtype=torch.long)
        tracks = torch.zeros(4, dtype=torch.bool)
        e = torch.tensor([1e-15, 3e-15, 2e-15, 2e-15])
        p = ClusterEnergyHead(3)(torch.randn(4, 3), b, ids, tracks, e)
        truth_e = torch.tensor([8., 8., 4., 4.])
        target = deposited_energy_targets(p, ids, truth_e, e, b, tracks)
        torch.testing.assert_close(target, torch.tensor([8., 4.]))
        torch.testing.assert_close(p['hit_energy'].sum(), p['energy'].sum())
        torch.testing.assert_close(
            pooled_energy_loss(p, ids, truth_e, e, b, tracks, 'sum'),
            5 * (p['energy'] - target).square().sum() / 2)

    def test_option_validation_leaves_legacy_options_unchanged(self):
        validate_training_arguments(SimpleNamespace(cluster_energy_pooling=False))
        args = SimpleNamespace(cluster_energy_pooling=True, use_multihead_model=True,
            energy_regression=True, energy_regression_cluster=True, LE_cluster='sum_log_perCluster',
            jit=False, dp=False, energy_branch=False, energy_regression_weight=False,
            cluster_energy_source='truth', cluster_energy_tbeta=.7, cluster_energy_td=.5)
        validate_training_arguments(args)
        args.LE_cluster = 'distribution'
        with self.assertRaisesRegex(ValueError, 'LE-cluster'):
            validate_training_arguments(args)
        args.cluster_energy_pooling = False
        validate_training_arguments(args)

    def test_checkpoint_preserves_interaction_schedule(self):
        data = self.make_data()
        model = GravNetModelMultiHead(input_dim=5, output_dim=3, n_heads=3,
            interaction_mode='concat', interaction_start_epoch=10, cluster_energy_pooling=True).eval()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'pooled.pt'
            for epoch in [0, 10]:
                torch.save(checkpoint_payload(model, epoch=epoch), p)
                loaded = get_model(str(p), jit=False, input_dim=5, output_dim=5,
                    energy_regression=True, energy_regression_cluster=True,
                    cluster_energy_source='truth').eval()
                expected = model(data.x, data.batch, epoch=epoch,
                    truth_cluster_index=data.y[:, 0], detected_energy=data.feat[:, 0])
                actual = inference_forward(loaded, data)
                torch.testing.assert_close(actual[:, 1], expected['regressions'][0].flatten())
                torch.testing.assert_close(actual[:, 2], expected['regressions'][1].flatten())

    def test_predicted_membership_uses_existing_algorithm_and_no_truth(self):
        from objectcondensation import get_clustering_np_new
        output = torch.tensor([[3., 0., 0.], [-1., .1, 0.], [2., 2., 2.], [-2., 2.1, 2.]])
        b = torch.zeros(4, dtype=torch.long); track = torch.zeros(4, dtype=torch.bool)
        a = cluster_membership(output, b, track, 'predicted', tbeta=.7, td=.5)
        expected, _ = get_clustering_np_new(None, output[:, 0].sigmoid().numpy(),
                                            output[:, 1:].numpy(), track.numpy(), .7, .5)
        np.testing.assert_array_equal(a.numpy(), expected + 1)
        other_truth = torch.tensor([100, 200, 300, 400])
        torch.testing.assert_close(a, cluster_membership(output, b, track, 'predicted', other_truth))
        with self.assertRaises(ValueError):
            cluster_membership(output, b, track, 'truth')
        self.assertEqual(cluster_membership(output[:1], b[:1], ~track[:1], 'predicted').item(), 1)

    def make_data(self):
        n = 96
        x = torch.randn(n, 5); x[:, 4] = 0; x[[0, 48], 4] = 1
        b = torch.tensor([0] * 48 + [1] * 48)
        ids = torch.tensor(([1] * 24 + [2] * 24) * 2)
        feat = torch.zeros(n, 13); feat[:, 0] = torch.rand(n); feat[:, 5] = x[:, 4]
        return Data(x=x, batch=b, y=torch.stack((ids, x[:, 4].long()), dim=1), feat=feat)

    def test_model_checkpoint_and_inference_truth_is_explicit(self):
        data = self.make_data()
        model = GravNetModelMultiHead(input_dim=5, output_dim=3, n_heads=3,
            interaction_mode='none', cluster_energy_pooling=True).eval()
        result = model(data.x, data.batch, truth_cluster_index=data.y[:, 0], detected_energy=data.feat[:, 0])
        self.assertEqual(result['cluster_energy']['energy'].numel(), 4)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'pooled.pt'; torch.save(checkpoint_payload(model), p)
            loaded = get_model(str(p), jit=False, input_dim=5, output_dim=5,
                               energy_regression=True, energy_regression_cluster=True).eval()
            self.assertEqual(loaded.model.cluster_energy_source, 'predicted')
            # No truth field is required for predicted inference.
            unlabelled = Data(x=data.x, batch=data.batch, feat=data.feat)
            self.assertEqual(inference_forward(loaded, unlabelled).shape, (96, 5))
            configure_inference(loaded, source='truth')
            actual = inference_forward(loaded, data)
            torch.testing.assert_close(actual[:, 1], result['regressions'][0].squeeze(-1))
            torch.testing.assert_close(actual[:, 2], result['regressions'][1].squeeze(-1))
            with self.assertRaises(ValueError):
                inference_forward(loaded, unlabelled)

    def test_legacy_checkpoint_still_loads(self):
        data = self.make_data()
        with tempfile.TemporaryDirectory() as tmp:
            for multi in [False, True]:
                model = (GravNetModelMultiHead(input_dim=5, output_dim=3, n_heads=3, interaction_mode='none')
                         if multi else GravnetModel(input_dim=5, output_dim=5)).eval()
                p = Path(tmp) / f'legacy_{multi}.pt'; torch.save(checkpoint_payload(model), p)
                loaded = get_model(str(p), jit=False, input_dim=5, output_dim=5,
                                   energy_regression=True, energy_regression_cluster=True).eval()
                actual = inference_forward(loaded, data)
                expected = model(data.x, data.batch)
                if multi:
                    expected = torch.cat((expected['clustering'][:, :1], *expected['regressions'],
                                          expected['clustering'][:, 1:]), dim=1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_legacy_to_pooled_bootstrap_preserves_clustering_and_track(self):
        from train import load_checkpoint_state
        data = self.make_data()
        legacy = GravNetModelMultiHead(input_dim=5, output_dim=3, n_heads=3, interaction_mode='none').eval()
        pooled = GravNetModelMultiHead(input_dim=5, output_dim=3, n_heads=3,
                                      interaction_mode='none', cluster_energy_pooling=True).eval()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'legacy.pt'; torch.save(checkpoint_payload(legacy), p)
            load_checkpoint_state(pooled, str(p))
            expected = legacy(data.x, data.batch)
            actual = pooled(data.x, data.batch, truth_cluster_index=data.y[:, 0], detected_energy=data.feat[:, 0])
            for a, b in [(expected['clustering'], actual['clustering']),
                         (expected['regressions'][0], actual['regressions'][0])]:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.save(checkpoint_payload(pooled), p)
            with self.assertRaisesRegex(ValueError, 'enable --cluster-energy-pooling'):
                load_checkpoint_state(legacy, str(p))


if __name__ == '__main__':
    unittest.main()
