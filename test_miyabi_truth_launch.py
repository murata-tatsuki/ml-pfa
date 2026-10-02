"""Execute launcher shells with fake PBS/MPI tools; never allocate a GPU."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT = Path(__file__).resolve().parent


class TruthLaunchTests(unittest.TestCase):
    def test_truth_profile_reaches_training_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindir = root / 'bin'
            bindir.mkdir()
            # Capture the environment entering the automatic controller.
            capture = bindir / 'python3'
            capture.write_text(f'#!{sys.executable}\nimport json, os\n'
                               'print(json.dumps(dict(os.environ)))\n')
            capture.chmod(0o755)
            env = dict(os.environ, PATH=str(bindir) + ':' + os.environ['PATH'],
                       PBS_JOBID='test.opbs', RUN_DIR=str(root / 'run'))
            for key in ('CLIP_MODE', 'EXPECTED_NODES', 'BATCH_SIZE',
                        'CLUSTER_ENERGY_POOLING', 'CLUSTER_ENERGY_SOURCE'):
                env.pop(key, None)
            result = subprocess.run(['bash', str(PROJECT / 'shell/miyabi_train_energy_truth.sh')],
                                    env=env, check=True, capture_output=True, text=True)
            profile = json.loads(result.stdout)
            self.assertEqual(profile['CLUSTER_ENERGY_POOLING'], '1')
            self.assertEqual(profile['CLUSTER_ENERGY_SOURCE'], 'truth')
            self.assertEqual(profile['CLIP_MODE'], 'norm')
            self.assertEqual(profile['EXPECTED_NODES'], '4')
            self.assertEqual(profile['BATCH_SIZE'], '32')

            # Capture the fully expanded MPI command that would launch train.py.
            (bindir / 'mpiexec').write_text(f'#!{sys.executable}\nimport json, sys\n'
                                          'print("CAPTURE=" + json.dumps(sys.argv[1:]))\n')
            (bindir / 'mpiexec').chmod(0o755)
            nodefile = root / 'nodes'
            nodefile.write_text('node1\nnode2\nnode3\nnode4\n')
            bash_env = root / 'bash_env'
            bash_env.write_text('module() { :; }\n')
            profile.update(PBS_NODEFILE=str(nodefile), BASH_ENV=str(bash_env),
                           TRAIN_DIR=str(root), VALID_DIR=str(root), BENCHMARK_BATCHES='0',
                           AUTO_TRAIN_DEADLINE='2000000000', AUTO_EPOCH_STOP_FILE=str(root / 'stop.json'))
            result = subprocess.run(['bash', str(PROJECT / 'shell/miyabi_train_energy_fast.sh')],
                                    env=profile, check=True, capture_output=True, text=True)
            argv = json.loads(next(line[8:] for line in result.stdout.splitlines()
                                   if line.startswith('CAPTURE=')))
            argv = argv[argv.index('train.py') + 1:]
            self.assertIn('--cluster-energy-pooling', argv)
            self.assertNotIn('--benchmark-batches', argv)
            for flag, value in (('--cluster-energy-source', 'truth'), ('--clip-mode', 'norm'),
                                ('--batch-size', '32'), ('--learning-rate', '4e-4'),
                                ('--ddp-lr-scaling', 'none'), ('--epoch-budget-deadline', '2000000000')):
                self.assertEqual(argv[argv.index(flag) + 1], value)


if __name__ == '__main__':
    unittest.main()
