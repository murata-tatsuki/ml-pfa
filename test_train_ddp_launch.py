"""Check distributed launch routing without reserving GPUs or loading data."""
import os
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import train


class DDPLaunchTests(unittest.TestCase):
    def test_explicit_optimizer_lr_is_independent_of_gpu_count(self):
        args = SimpleNamespace(learning_rate=4e-4, ddp_lr_scaling='none')
        self.assertEqual([train.ddp_learning_rate(args, n) for n in (1, 2, 4)],
                         [4e-4, 4e-4, 4e-4])
        args.ddp_lr_scaling = 'linear'
        self.assertEqual(train.ddp_learning_rate(args, 4), 1.6e-3)

    def test_loader_sweep_avoids_empty_validation_workers(self):
        args = SimpleNamespace(stream_files_per_chunk=32, stream_shuffle_buffer=256,
                               num_workers=2, benchmark_loader_sweep=True, ilc_streaming=True)
        self.assertEqual(train.benchmark_loader_configs(args, 4, 20),
                         [(32, 256, 2), (8, 256, 2), (8, 256, 4), (8, 128, 4)])
        self.assertEqual(train.benchmark_loader_configs(args, 4, 8),
                         [(32, 256, 2), (8, 256, 2)])
        args.benchmark_loader_sweep = False
        self.assertEqual(train.benchmark_loader_configs(args, 4, 20), [(32, 256, 2)])

    def test_benchmark_counts_only_measured_batches_and_stops(self):
        args = SimpleNamespace(benchmark_batches=2, benchmark_warmup=1)
        batches = [SimpleNamespace(num_graphs=32) for _ in range(10)]
        out = io.StringIO()
        with patch.object(train.torch.cuda, 'synchronize'), \
             patch.object(train.torch.cuda, 'reset_peak_memory_stats'), \
             patch.object(train.torch.cuda, 'max_memory_allocated', return_value=1048576), \
             patch.object(train.torch.cuda, 'max_memory_reserved', return_value=2097152), \
             patch.object(train.dist, 'get_rank', return_value=3), \
             patch.object(train, 'perf_counter', side_effect=[0., 5., 7., 10.]), \
             patch('sys.stdout', out):
            result = list(train.benchmark_batches(batches, args, 'cuda:0', 'train'))
        self.assertEqual(len(result), 3)
        self.assertIn('measured_steps=2 events=64 seconds=5.000000', out.getvalue())

    def test_benchmark_disabled_does_not_synchronize(self):
        args = SimpleNamespace(benchmark_batches=0)
        with patch.object(train.torch.cuda, 'synchronize') as sync:
            self.assertEqual(list(train.benchmark_batches([1, 2], args, 'cuda:0', 'train')), [1, 2])
            sync.assert_not_called()

    def test_torchrun_rank_three_uses_local_gpu_zero_and_preserves_master(self):
        env = dict(RANK="3", WORLD_SIZE="4", LOCAL_RANK="0",
                   MASTER_ADDR="compute0", MASTER_PORT="29517")
        with patch.dict(os.environ, env, clear=True), \
             patch("sys.argv", ["train.py", "-i", "unused", "--ddp"]), \
             patch.object(train.torch.cuda, "device_count", return_value=1), \
             patch.object(train, "run_ddp_training") as run, \
             patch.object(train.mp, "spawn") as spawn:
            train.main()
            self.assertEqual(run.call_args.args[:2], (3, 4))
            self.assertEqual(run.call_args.kwargs, dict(local_rank=0))
            spawn.assert_not_called()
            self.assertEqual(os.environ["MASTER_ADDR"], "compute0")
            self.assertEqual(os.environ["MASTER_PORT"], "29517")

    def test_single_node_spawn_remains_available(self):
        with patch.dict(os.environ, {}, clear=True), \
             patch("sys.argv", ["train.py", "-i", "unused", "--ddp",
                                "--gpus", "0,1,2,3", "--master-port", "29518"]), \
             patch.object(train.torch.cuda, "device_count", return_value=4), \
             patch.object(train.mp, "spawn") as spawn:
            with self.assertRaises(SystemExit) as result:
                train.main()
            self.assertIsNone(result.exception.code)
            self.assertEqual(spawn.call_args.kwargs["nprocs"], 4)
            self.assertEqual(spawn.call_args.kwargs["args"][0], 4)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0,1,2,3")

    def test_cuda_selected_before_process_group_initialization(self):
        calls = []
        class StopBeforeData(Exception):
            pass

        with patch.object(train, "configure_ddp_rank_logging"), \
             patch.object(train.torch.cuda, "set_device",
                          side_effect=lambda gpu: calls.append(("gpu", gpu))), \
             patch.object(train, "setup_ddp",
                          side_effect=lambda rank, size: calls.append(("group", rank, size))), \
             patch.object(train, "run_requirements", side_effect=StopBeforeData):
            with self.assertRaises(StopBeforeData):
                train.run_ddp_training(3, 4, SimpleNamespace(), local_rank=0)
        self.assertEqual(calls, [("gpu", 0), ("group", 3, 4)])


if __name__ == "__main__":
    unittest.main()
