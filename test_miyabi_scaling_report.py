"""Reject incomplete timings and keep different local batch sizes separate."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent/'tools'))
from watch_miyabi_scaling import completed_cases


def case(rank, batch=16, validation=True):
    text = (f'BENCH_CONFIG chunk=8 buffer=256 workers=2\n'
            f'BENCH_RUN gpus=2 batch_per_gpu={batch} global_batch={2*batch} '
            f'optimizer_lr=0.0004 precision=fp32\n'
            f'[rank {rank}] epoch=0 train_batch=12/100 loss=1.0\n'
            f'BENCH_RESULT phase=train rank={rank} measured_steps=10 events={10*batch} '
            'seconds=20.0 total_seconds=30.0 host_tree_rss_mib=1024 '
            'gpu_peak_reserved_mib=2048\n')
    if validation:
        text += (f'[rank {rank}] epoch=0 valid_batch=12/20\n'
                 f'BENCH_RESULT phase=validation rank={rank} measured_steps=10 events={10*batch} '
                 'seconds=10.0 total_seconds=15.0\n')
    return text


class ScalingReportTests(unittest.TestCase):
    def test_partial_rank_or_phase_never_selected(self):
        self.assertEqual(completed_cases([case(0), ''], 2), [])
        self.assertEqual(completed_cases([case(0), case(1, validation=False)], 2), [])

    def test_batch_sizes_are_not_overwritten(self):
        rows = completed_cases([case(r,16)+case(r,32) for r in range(2)], 2)
        self.assertEqual([r['batch'] for r in rows], [16,32])
        self.assertEqual(rows[0]['events_per_second'],16)
        self.assertAlmostEqual(rows[0]['gpu_hours'],220*2/3600)

    def test_nonfinite_loss_is_not_a_candidate(self):
        self.assertEqual(completed_cases([case(0).replace('loss=1.0','loss=nan'),case(1)],2),[])


if __name__ == '__main__':
    unittest.main()
