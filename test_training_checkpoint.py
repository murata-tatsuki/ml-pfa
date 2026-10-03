"""CPU regression tests: interrupted epochs and cosineReduce restarts."""
import argparse
import copy
import os
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from lrscheduler import CyclicLRWithRestarts
from training_checkpoint import TrainingCheckpoint


def payload(model, epoch, args):
    return {'model': model.state_dict(), 'inference_epoch': epoch}


def build(warmup=2, ddp=False):
    model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.25),
                                torch.nn.Linear(4, 1))
    model.register_buffer('rank_buffer', torch.tensor(0.))
    if ddp:
        model = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=.01)
    scheduler = CyclicLRWithRestarts(optimizer, batch_size=2, epoch_size=6,
                                    restart_period=2, t_mult=1.1, policy='cosineReduce',
                                    nrestart_cosreduce=3, warmup_epochs=warmup)
    args = SimpleNamespace(batch_size=2, lr_policy='cosineReduce', epochs=25,
                           resume='', ckptdir='unused')
    cp = TrainingCheckpoint(model, optimizer, scheduler, None, args, payload)
    return model, optimizer, scheduler, cp


def epoch(objects, steps=3):
    model, opt, sched, _ = objects
    sched.step()
    lrs = []
    for _ in range(steps):
        x = torch.randn(2, 3) + random.random() + np.random.rand()
        lrs.append(opt.param_groups[0]['lr'])
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step()
        sched.batch_step()
    return lrs


def seed():
    random.seed(134)
    np.random.seed(234)
    torch.manual_seed(334)


def ddp_roundtrip(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + directory + '/rendezvous',
                            rank=rank, world_size=2)
    try:
        seed()
        obj = build(ddp=True)
        torch.manual_seed(100 + rank)
        epoch(obj)
        obj[0].module.rank_buffer.fill_(rank + 3)
        expected_opt = copy.deepcopy(obj[1].state_dict())
        obj[3].save(directory, 0, {'test': True})
        expected_random = torch.rand(3)
        expected_lrs = epoch(obj)
        expected_weights = copy.deepcopy(obj[0].module.state_dict())
        resumed = build(ddp=True)
        start, history = resumed[3].load(directory + '/last.pth.tar')
        assert start == 1 and history['test']
        assert resumed[0].module.rank_buffer.item() == rank + 3
        assert torch.equal(torch.rand(3), expected_random)
        assert resumed[1].state_dict()['param_groups'] == expected_opt['param_groups']
        assert epoch(resumed) == expected_lrs
        for key, value in expected_weights.items():
            assert torch.equal(value, resumed[0].module.state_dict()[key]), key
    finally:
        dist.destroy_process_group()


class TrainingCheckpointTests(unittest.TestCase):
    def assert_tree_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for k in a:
                self.assert_tree_equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_rerun_interrupted_epoch_matches_uninterrupted_cosine_reduce(self):
        torch.set_num_threads(1)
        seed()
        continuous = build()
        reference_lrs = [epoch(continuous) for _ in range(25)]
        reference_model = copy.deepcopy(continuous[0].state_dict())
        reference_optimizer = copy.deepcopy(continuous[1].state_dict())
        self.assertGreaterEqual(continuous[2].restarts, 3)
        self.assertLess(continuous[2].eta_max, 1.)
        # Before the first epoch, during warmup, around a restart, and after
        # cosineReduce has actually reduced the maximum learning rate.
        for interrupted in (0, 1, 4, 15):
            with self.subTest(interrupted=interrupted), tempfile.TemporaryDirectory() as directory:
                seed()
                original = build()
                for _ in range(interrupted):
                    epoch(original)
                original[3].save(directory, interrupted - 1, {'saved': interrupted})
                # Work in the interrupted epoch must be discarded on resume.
                epoch(original, steps=2)
                resumed = build()
                resumed[3].args.epochs = 30  # total target can change
                start, history = resumed[3].load(directory + '/last.pth.tar')
                self.assertEqual(start, interrupted)
                self.assertEqual(history, {'saved': interrupted})
                actual_lrs = [epoch(resumed) for _ in range(start, 25)]
                self.assertEqual(actual_lrs, reference_lrs[start:])
                self.assert_tree_equal(resumed[0].state_dict(), reference_model)
                self.assert_tree_equal(resumed[1].state_dict(), reference_optimizer)
                for field in ('restarts', 'restart_period', 'eta_max', 't_epoch',
                              'total_iterations', 'iteration', 'last_epoch'):
                    self.assertEqual(getattr(resumed[2], field), getattr(continuous[2], field))

    def test_failed_save_keeps_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = build()
            obj[3].save(directory, -1, {})
            epoch(obj)
            with patch('training_checkpoint.torch.save', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(RuntimeError, 'disk full'):
                    obj[3].save(directory, 0, {})
            self.assertEqual(build()[3].load(directory + '/last.pth.tar')[0], 0)

    def test_legacy_checkpoint_with_new_default_arguments_preserves_next_update(self):
        from particle_heads import add_arguments
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        defaults = vars(parser.parse_args([]))
        with tempfile.TemporaryDirectory() as directory:
            seed()
            original = build()
            epoch(original)
            original[3].save(directory, 0, {'saved': 1})
            expected_lrs = epoch(original)
            expected_model = copy.deepcopy(original[0].state_dict())
            expected_optimizer = copy.deepcopy(original[1].state_dict())
            resumed = build()
            resumed[3].config.update(defaults)
            start, history = resumed[3].load(directory + '/last.pth.tar')
            self.assertEqual((start, history), (1, {'saved': 1}))
            self.assertEqual(epoch(resumed), expected_lrs)
            self.assert_tree_equal(resumed[0].state_dict(), expected_model)
            self.assert_tree_equal(resumed[1].state_dict(), expected_optimizer)

    def test_legacy_checkpoint_rejects_enabled_heads_or_changed_new_options(self):
        from particle_heads import add_arguments
        parser = argparse.ArgumentParser()
        add_arguments(parser)
        defaults = vars(parser.parse_args([]))
        with tempfile.TemporaryDirectory() as directory:
            build()[3].save(directory, 0, {})
            for key, value in (('pid_head', True), ('five_particle_energy_heads', True),
                               ('pid_loss_weight', 2.), ('epochs_noPID', 3),
                               ('unknown_future_option', False)):
                with self.subTest(option=key):
                    resumed = build()
                    resumed[3].config.update(defaults)
                    resumed[3].config[key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        resumed[3].load(directory + '/last.pth.tar')

    def test_saved_particle_options_cannot_be_removed_on_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            original = build()
            original[3].config['pid_head'] = True
            original[3].save(directory, 0, {})
            with self.assertRaisesRegex(ValueError, 'pid_head'):
                build()[3].load(directory + '/last.pth.tar')

    def test_knn_backend_switch_preserves_checkpoint_resume(self):
        # KNN scheduling adds no parameters and must not invalidate old checkpoints.
        with tempfile.TemporaryDirectory() as directory:
            seed()
            original = build()
            epoch(original)
            original[3].save(directory, 0, {})
            expected_lrs = epoch(original)
            expected_weights = copy.deepcopy(original[0].state_dict())
            resumed = build()
            resumed[3].args.knn_backend = 'event-parallel'
            resumed_cp = TrainingCheckpoint(resumed[0], resumed[1], resumed[2], None,
                                            resumed[3].args, payload)
            self.assertNotIn('knn_backend', resumed_cp.config)
            self.assertEqual(resumed_cp.load(directory + '/last.pth.tar')[0], 1)
            self.assertEqual(epoch(resumed), expected_lrs)
            self.assert_tree_equal(resumed[0].state_dict(), expected_weights)

    def test_reject_weights_only_and_changed_batch_or_world_size(self):
        with tempfile.TemporaryDirectory() as directory:
            path = directory + '/old.pth'
            obj = build()
            torch.save({'model': obj[0].state_dict()}, path)
            with self.assertRaisesRegex(ValueError, 'weights-only'):
                obj[3].load(path)
            obj[3].save(directory, -1, {})
            path = directory + '/last.pth.tar'
            different = build()
            different[3].config['batch_size'] = 4
            with self.assertRaisesRegex(ValueError, 'batch_size'):
                different[3].load(path)
            different = build()
            different[3].world_size = 2
            with self.assertRaisesRegex(ValueError, 'GPU/rank count'):
                different[3].load(path)

    def test_plateau_state_survives_validation_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            obj = build()
            obj[3].scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(obj[1], patience=1)
            for loss in (1., 2., 3.):
                obj[3].scheduler.step(loss)
            obj[3].save(directory, 2, {})
            resumed = build()
            resumed[3].scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(resumed[1], patience=1)
            resumed[3].load(directory + '/last.pth.tar')
            self.assertEqual(resumed[3].scheduler.state_dict(), obj[3].scheduler.state_dict())
            self.assertEqual(resumed[1].param_groups[0]['lr'], obj[1].param_groups[0]['lr'])

    def test_two_cpu_ddp_ranks_resume_rng_optimizer_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(ddp_roundtrip, args=(directory,), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
