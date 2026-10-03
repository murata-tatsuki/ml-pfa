"""No GPU allocation or real qsub calls: test the automatic job controller."""
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools import miyabi_auto_resume as auto


DETAILS = '''Job Id: 42.opbs
    Resource_List.walltime = 48:00:00
    resources_used.walltime = 00:02:00
    Resource_List.nodect = 4
'''


def checkpoint(directory, epoch):
    root = Path(directory) / 'checkpoints'
    root.mkdir(exist_ok=True)
    path = root / f'ckpt_{epoch}_1.pth.tar'
    path.write_text('mock checkpoint')
    latest = root / 'last.pth.tar'
    latest.unlink(missing_ok=True)
    latest.symlink_to(path.name)


class AutoResumeTests(unittest.TestCase):
    def scenario(self, outcome='time_limit', code=None, advance=True, stop=False,
                 extra=None, expect_error=None, submit_failure=False, boundary=None):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(PBS_JOBID='42.opbs', PBS_NODEFILE='/unused', RUN_DIR=directory,
                       LR_POLICY='cosineReduce')
            env.update(extra or {})
            checkpoint(directory, 2)
            calls = []

            def command(args, **kwargs):
                calls.append(args)
                if args[0] == 'qstat':
                    return subprocess.CompletedProcess(args, 0, DETAILS, '')
                self.assertEqual(args[0], 'qsub')
                if submit_failure:
                    raise subprocess.TimeoutExpired(args, 60)
                return subprocess.CompletedProcess(args, 0, '43.opbs\n', '')

            def train(args, passed_env, budget, stop_file):
                self.assertEqual(passed_env['BENCHMARK_BATCHES'], '0')
                self.assertEqual(passed_env['EXPECTED_NODES'], '4')
                self.assertEqual(passed_env['BATCH_SIZE'], '32')
                self.assertEqual(passed_env['RESUME_CHECKPOINT'], directory + '/checkpoints/last.pth.tar')
                self.assertEqual(budget, 48 * 3600 - 120 - 600)
                if advance:
                    checkpoint(directory, 3)
                if stop:
                    stop_file.touch()
                if boundary is not None:
                    Path(passed_env['AUTO_EPOCH_STOP_FILE']).write_text(json.dumps({'next_epoch': boundary}))
                return outcome, code

            with patch.object(auto.subprocess, 'run', side_effect=command), \
                 patch.object(auto, 'supervise', side_effect=train):
                if expect_error:
                    with self.assertRaises(expect_error):
                        auto.run(env)
                else:
                    auto.run(env)
            return [cmd for cmd in calls if cmd[0] == 'qsub']

    def test_planned_limit_submits_one_dependent_successor_with_same_settings(self):
        calls = self.scenario()
        self.assertEqual(len(calls), 1)
        command = calls[0]
        self.assertIn('depend=afterany:42.opbs', command)
        self.assertIn('walltime=48:00:00', command)
        self.assertIn('select=4:mpiprocs=1', command)
        variables = command[command.index('-v') + 1]
        self.assertIn('AUTO_JOB_INDEX=2', variables)
        self.assertIn('BATCH_SIZE=32', variables)
        self.assertIn('EXPECTED_NODES=4', variables)
        self.assertIn('LR_POLICY=cosineReduce', variables)
        self.assertNotIn('PBS_JOBID', variables)

    def test_normal_completion_never_submits(self):
        self.assertEqual(self.scenario('complete', 0), [])

    def test_knn_backend_survives_automatic_resubmission(self):
        command = self.scenario(extra={'KNN_BACKEND': 'event-parallel'})[0]
        self.assertIn('KNN_BACKEND=event-parallel', command[command.index('-v') + 1])

    def test_truth_pooling_and_norm_clipping_survive_automatic_resubmission(self):
        command = self.scenario(extra=dict(CLUSTER_ENERGY_POOLING='1',
            CLUSTER_ENERGY_SOURCE='truth', CLIP_MODE='norm'))[0]
        variables = command[command.index('-v') + 1]
        for setting in ('CLUSTER_ENERGY_POOLING=1', 'CLUSTER_ENERGY_SOURCE=truth', 'CLIP_MODE=norm'):
            self.assertIn(setting, variables)

    def test_successful_predicted_boundary_submits_successor(self):
        self.assertEqual(len(self.scenario('complete', 0, boundary=4)), 1)

    def test_boundary_signal_does_not_hide_failure_or_bad_checkpoint(self):
        self.assertEqual(self.scenario('error', 1, boundary=4), [])
        self.assertEqual(self.scenario('complete', 0, boundary=99, expect_error=RuntimeError), [])

    def test_application_errors_including_124_and_143_never_submit(self):
        for code in (1, 124, 137, 143):
            with self.subTest(code=code):
                self.assertEqual(self.scenario('error', code), [])

    def test_stop_file_suppresses_continuation(self):
        self.assertEqual(self.scenario(stop=True), [])

    def test_stalled_epoch_stops_instead_of_spending_more_gpu_time(self):
        self.assertEqual(self.scenario(advance=False, expect_error=RuntimeError), [])

    def test_epoch_target_and_job_cap_stop(self):
        self.assertEqual(self.scenario(extra={'EPOCHS': '4'}), [])
        self.assertEqual(self.scenario(extra={'AUTO_MAX_JOBS': '1'}), [])

    def test_ambiguous_submission_is_not_retried(self):
        calls = self.scenario(submit_failure=True, expect_error=subprocess.TimeoutExpired)
        self.assertEqual(len(calls), 1)

    def test_stop_file_before_training_prevents_launch_or_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'STOP_AUTO').touch()
            with patch.object(auto.subprocess, 'run') as cmd, patch.object(auto, 'supervise') as train:
                auto.run(dict(PBS_JOBID='42.opbs', PBS_NODEFILE='/unused', RUN_DIR=directory))
                cmd.assert_not_called()
                train.assert_not_called()

    def test_successor_refuses_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(auto.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, DETAILS)), \
                 patch.object(auto, 'supervise') as train:
                with self.assertRaisesRegex(RuntimeError, 'checkpoint is missing'):
                    auto.run(dict(PBS_JOBID='42.opbs', PBS_NODEFILE='/unused',
                                  RUN_DIR=directory, AUTO_JOB_INDEX='2'))
                train.assert_not_called()

    def test_supervisor_uses_actual_timer_not_application_returncode(self):
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / 'STOP_AUTO'
            self.assertEqual(auto.supervise(
                [sys.executable, '-c', 'raise SystemExit(124)'], dict(os.environ), 10, stop),
                ('error', 124))
            self.assertEqual(auto.supervise(
                [sys.executable, '-c', 'import time; time.sleep(60)'], dict(os.environ), .1, stop),
                ('time_limit', None))


if __name__ == '__main__':
    unittest.main()
