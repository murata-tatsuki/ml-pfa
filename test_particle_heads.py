"""CPU checks: python -m unittest test_particle_heads -v."""
import argparse
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch_geometric.data import Data
import particle_heads as ph
from gravnet_model import GravNetModelMultiHead
from cluster_energy import checkpoint_payload
from model import get_model
from pandora_eval_reconstruction import energy_clusters, predict


def settings(**kwargs):
    values = dict(pid_head=True, five_particle_energy_heads=True, pid_loss_weight=2., epochs_noPID=-1,
        epochs_nobeta=-1, epochs_noLE=-1, regression_coefficinet=1., LE_gradually=False,
        use_charged_cluster_loss=False, beta_track=True, beta_track_beginning=True,
        force_track_alpha=True, l_beta_suppression=False, epsilon=1e-3,
        use_multihead_model=True, energy_regression=True, energy_regression_cluster=True,
        LE_track='alpha_tracker_diff_log_perCluster', LE_cluster='sum_log_perCluster',
        multihead_regression_heads=2)
    return SimpleNamespace(**dict(values, **kwargs))


def fixture():
    # Each species has two calo hits and two tracks. Highest beta is a calo;
    # the second track must nevertheless supervise charged energy and PID.
    n = 20
    y = torch.tensor([[i+1, track] for i in range(5) for track in (0, 0, 1, 1)])
    label = torch.zeros(n, 9)
    for i, (pdg, charge) in enumerate(((211, 1), (-11, 1), (13, -1), (22, 0), (2112, 0))):
        label[i*4:(i+1)*4, 2] = pdg
        label[i*4:(i+1)*4, 3] = charge
        label[i*4:(i+1)*4, 7] = 12. if i == 4 else 10.
    data = Data(x=torch.zeros(n, 5), y=y, label=label, feat=torch.ones(n, 13),
                batch=torch.zeros(n, dtype=torch.long), truth_valid=torch.ones(n, dtype=torch.bool),
                input_row=torch.arange(n))
    data.x[:, 4] = y[:, 1]
    out = torch.tensor([[b, .1, .2] for i in range(5) for b in (5., 0., 1., 2.)], requires_grad=True)
    heads = [torch.full((n, 1), float(i+1), requires_grad=True) for i in range(5)]
    logits = torch.zeros(n, 5, requires_grad=True)
    return data, out, heads, logits


class ParticleHeadsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        torch.set_num_threads(1)

    def test_exact_targets_gradients_and_pid_alpha(self):
        data, out, heads, logits = fixture()
        objects = ph.truth_objects(data, out[:, 0].sigmoid())
        self.assertEqual(objects['alpha'].tolist(), [3, 7, 11, 15, 19])
        self.assertEqual(objects['species'].tolist(), list(range(5)))
        torch.testing.assert_close(ph.object_energies(heads, objects), torch.tensor([1., 2., 3., 8., 10.]))
        le, lp, metrics = ph.auxiliary_losses(out, heads, logits, data, settings(), 0, 3.)
        expected = 3 * sum(math.log1p(abs(x-y)) for x, y in zip([1, 2, 3, 8, 10], [10, 10, 10, 10, 12])) / 5
        self.assertAlmostEqual(le.item(), expected, places=6)
        self.assertAlmostEqual(lp.item(), 2 * math.log(5), places=6)
        self.assertAlmostEqual(sum(float(metrics['L_E_'+s]) for s in ph.SPECIES), le.item(), places=6)
        (le + lp).backward()
        for i, h in enumerate(heads):
            expected_rows = [i*4+3] if i < 3 else [i*4, i*4+1]
            self.assertEqual(h.grad.flatten().nonzero().flatten().tolist(), expected_rows)
        self.assertEqual((logits.grad.abs().sum(1) > 0).nonzero().flatten().tolist(), [3, 7, 11, 15, 19])
        # Alpha is a discrete selection; auxiliary losses do not train beta directly.
        self.assertEqual(out.grad.abs().sum().item(), 0.)

    def test_missing_tracks_unknowns_and_empty_event_normalization(self):
        data, out, heads, logits = fixture()
        data.y[:4, 1] = 0  # charged truth without track: PID yes, energy no
        data.truth_valid[4:8] = False
        data.batch[16:] = 1; data.truth_valid[16:] = False  # empty-labelled event still counts
        le, lp, m = ph.auxiliary_losses(out, heads, logits, data, settings(), 0, 1.)
        self.assertEqual(m['N_E_missing_track'], 1.)
        self.assertEqual(m['N_PID'], 3.)
        self.assertAlmostEqual(le.item(), (math.log(8) + math.log(3)) / 4, places=6)
        self.assertAlmostEqual(lp.item(), math.log(5), places=6)
        (le + lp).backward()
        self.assertEqual(heads[0].grad.abs().sum(), 0.)
        self.assertGreater(logits.grad[0].abs().sum(), 0.)
        self.assertEqual(logits.grad[4:8].abs().sum(), 0.)

    def test_empty_rank_connected_zero_and_species_mapping(self):
        data, out, heads, logits = fixture()
        data.truth_valid[:] = False
        le, lp, m = ph.auxiliary_losses(out, heads, logits, data, settings(), 0, 1.)
        self.assertEqual(float(le + lp), 0.)
        (le + lp).backward()
        for t in [out, *heads, logits]:
            self.assertIsNotNone(t.grad)
            self.assertTrue(torch.isfinite(t.grad).all())
            self.assertEqual(t.grad.abs().sum(), 0.)
        self.assertEqual(ph.species_ids(torch.tensor([211., -211., 11., -13., 22., 130., 12., 0., float('nan')]),
                         torch.tensor([1., -1., -1., 1., 0., 0., 0., 0., 1.])).tolist(), [0, 0, 1, 2, 3, 4, -1, -1, -1])

    def test_same_truth_id_in_two_events_and_scheduling(self):
        data, out, heads, logits = fixture()
        data.batch[4:] = 1
        data.y[4:8, 0] = 1
        o = ph.truth_objects(data, out[:, 0].sigmoid())
        self.assertEqual(len(o['alpha']), 5)
        _, lp, _ = ph.auxiliary_losses(out, heads, logits, data, settings(epochs_noPID=0), 0, 1.)
        self.assertEqual(lp.item(), 0.)
        a = settings(epochs_nobeta=2, epochs_noLE=4)
        c = dict(L_V=1., L_beta=10., L_E=100., L_PID=1000.)
        self.assertEqual(ph.compose_loss(c, a, 2), 1002.)
        self.assertEqual(ph.compose_loss(c, a, 3), 1012.)
        self.assertEqual(ph.compose_loss(c, a, 5), 1112.)

    def test_options_default_and_incompatible_combinations(self):
        p = argparse.ArgumentParser(); ph.add_arguments(p)
        self.assertFalse(ph.enabled(p.parse_args([])))
        ph.validate_arguments(p.parse_args([]))
        a = settings(); ph.validate_arguments(a)
        self.assertEqual(a.multihead_regression_heads, 5)
        for kw in [dict(use_multihead_model=False), dict(LE_track='log_ratio_mse'), dict(cluster_energy_pooling=True),
                   dict(pid_loss_weight=-1), dict(energy_regression=False), dict(jit=True)]:
            with self.assertRaises(ValueError): ph.validate_arguments(settings(**kw))

    def test_all_five_pid_and_trackless_fallback(self):
        ids = np.array([1, 1, 2, 2, 3, 3]); beta = np.array([.8, .9, .1, .9, .8, .9])
        track = np.array([1, 0, 0, 0, 0, 0])
        energies = np.arange(30).reshape(6, 5) + 1.
        logits = np.array([[0, 0, 0, 9, 0], [9, 0, 0, 0, 0], [0]*5, [0, 9, 0, 2, 4], [0]*5, [0, 0, 9, 5, 4]])
        got = ph.decode_clusters(ids, beta, track, energies, logits)
        self.assertEqual([c['pid'] for c in got], [3, 1, 2])
        self.assertEqual([c['energy_head'] for c in got], [3, 4, 3])
        self.assertEqual([c['energy_fallback'] for c in got], [0, 1, 1])
        self.assertEqual([c['energy'] for c in got], [energies[1, 3], energies[2:4, 4].sum(), energies[4:, 3].sum()])
        data = Data(x=torch.zeros(6,5), input_row=torch.arange(6)+100)
        data.x[:,4] = torch.tensor(track)
        pred = dict(assignments=ids, beta=beta, tracker=np.ones(6)*3, calo=np.ones(6)*2, pid_logits=logits)
        legacy = energy_clusters(data, {k:v for k,v in pred.items() if k != 'pid_logits'})
        pid_only = energy_clusters(data, pred)
        self.assertEqual([c['energy'] for c in legacy], [c['energy'] for c in pid_only])
        self.assertTrue(all(c['energy_head'] == -1 and not c['energy_fallback'] for c in pid_only))
        with self.assertRaisesRegex(ValueError, 'requires a PID'): ph.decode_clusters(ids, beta, track, energies)

    def test_checkpoint_mapping_roundtrip_and_no_legacy_misread(self):
        old = GravNetModelMultiHead(input_dim=5, output_dim=3, n_heads=3, interaction_mode='concat')
        new = GravNetModelMultiHead(input_dim=5, output_dim=3, n_heads=6, interaction_mode='concat',
                                   pid_head=True, five_particle_energy_heads=True, interaction_start_epoch=5)
        ph.load_training_checkpoint(new, old.state_dict(), None)
        for ni in range(1, 6):
            oi = 1 if ni <= 3 else 2
            for k, v in new.head_output[ni].state_dict().items():
                torch.testing.assert_close(v, old.head_output[oi].state_dict()[k], rtol=0, atol=0)
        x = torch.randn(64, 5); b = torch.zeros(64, dtype=torch.long)
        new.eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'new.pt'
            torch.save(checkpoint_payload(new, epoch=2, args=settings()), path)
            loaded = get_model(str(path), jit=False, input_dim=5, output_dim=5,
                               energy_regression=True, energy_regression_cluster=True).eval()
            a = new(x, b, epoch=2); c = loaded(x, b, return_dict=True)
            for k in ('clustering', 'pid_logits'):
                torch.testing.assert_close(a[k], c[k], rtol=0, atol=0)
            for h1, h2 in zip(a['regressions'], c['regressions']): torch.testing.assert_close(h1, h2, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'structured readout'): loaded(x, b)
            with self.assertRaisesRegex(ValueError, 'missing particle_heads_config'):
                ph.validate_checkpoint_config(new.state_dict(), None)
            with self.assertRaisesRegex(ValueError, 'requires --five'):
                ph.load_training_checkpoint(old, new.state_dict(), ph.checkpoint_config(new, 2))

    def test_forward_optimizer_and_oc_loss_for_both_heads(self):
        # Same graph size as k=40 production model, with unsupported/noise rows masked.
        data, out, heads, logits = fixture()
        idx = torch.arange(64) % 20
        data = Data(x=torch.randn(64,5), y=data.y[idx], label=data.label[idx], feat=data.feat[idx],
                    batch=torch.zeros(64,dtype=torch.long), truth_valid=torch.ones(64,dtype=torch.bool))
        data.x[:,4] = data.y[:,1]
        for unknown in (False, True):
            if unknown: data.truth_valid[:] = False
            model = GravNetModelMultiHead(input_dim=5, output_dim=3, n_heads=6, pid_head=True,
                                         five_particle_energy_heads=True, interaction_mode='none')
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            for step in range(2):
                optimizer.zero_grad()
                result = model(data.x, data.batch)
                loss, metrics = ph.five_head_training_loss(result['clustering'], result['regressions'],
                      result['pid_logits'], data, settings(), step, 1.)
                self.assertTrue(torch.isfinite(loss))
                loss.backward(); optimizer.step()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                components = ph.five_head_training_loss(result['clustering'], result['regressions'],
                      result['pid_logits'], data, settings(), step, 1., return_components=True)
                torch.testing.assert_close(loss.detach(), ph.compose_loss(components, settings(), step))
        model.eval(); model.pid_head = False
        with self.assertRaisesRegex(ValueError, 'requires --pid-head'):
            predict(model, data, 'cpu', .7, .5)


if __name__ == '__main__':
    unittest.main()
