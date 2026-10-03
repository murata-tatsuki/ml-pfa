"""Launch coverage, model integration, and bitwise GPU ablations for KNN 1/3/5."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from knn_event_parallel import (prepare_plan, plan_from_batch, prepare_training_batch,
                                select_knn, select_knn_baseline, knn_graph, BLOCK_SIZES)


class LaunchPlanTests(unittest.TestCase):
    def test_each_query_covered_exactly_once_in_correct_event(self):
        # Empty events, undersized neighborhoods, and block-boundary tails.
        sizes = torch.tensor([0, 1, 29, 30, 40, 41, 127, 128, 129,
                              255, 256, 257, 1023, 1024, 1025, 3000, 0])
        ptr = torch.cat((torch.zeros(1, dtype=torch.long), sizes.cumsum(0)))
        for block_size in BLOCK_SIZES:
            rows, tasks, block, maximum, n = prepare_plan(ptr, block_size=block_size)
            torch.testing.assert_close(rows.long(), ptr, rtol=0, atol=0)
            counts = torch.zeros(n, dtype=torch.long)
            for event, start in tasks.T.tolist():
                self.assertGreaterEqual(start, ptr[event])
                self.assertLess(start, ptr[event+1])
                counts[start:min(start+block, int(ptr[event+1]))] += 1
            self.assertTrue(torch.equal(counts, torch.ones_like(counts)))
            self.assertEqual(tasks.size(1), int(torch.div(sizes+block-1, block, rounding_mode='floor').sum()))
            self.assertEqual(maximum, int(torch.div(sizes+block-1, block, rounding_mode='floor').max()))

    def test_empty_and_invalid_ptr(self):
        p = prepare_plan(torch.tensor([0, 0, 0]))
        self.assertEqual(p[1].shape, (2, 0))
        self.assertEqual(p[3:], (0, 0))
        for ptr in [torch.tensor([1, 2]), torch.tensor([0, 3, 2]),
                    torch.tensor([0]), torch.tensor([0., 1.]),
                    torch.tensor([0, 2**31]), torch.tensor([[0, 2]])]:
            with self.subTest(ptr=ptr), self.assertRaises(ValueError):
                prepare_plan(ptr)
        with self.assertRaises(ValueError):
            prepare_plan(torch.tensor([0, 2]), block_size=64)

    def test_fallback_preserves_missing_event_ids(self):
        batch = torch.tensor([1, 1, 3, 3, 3])
        p = plan_from_batch(torch.empty(5, 4), batch)
        self.assertEqual(p[0].tolist(), [0, 0, 2, 2, 5])
        for invalid in [torch.tensor([1, 0]), torch.tensor([-1, 0])]:
            with self.assertRaises(ValueError):
                plan_from_batch(torch.empty(2, 4), invalid)

    def test_training_handoff_and_once_per_forward(self):
        from torch_geometric.data import Batch, Data
        from gravnet_model import GravNetModelMultiHead
        from cluster_energy import training_forward
        torch.set_num_threads(1)
        torch.manual_seed(1004)
        data = Batch.from_data_list([Data(x=torch.randn(n, 5)) for n in (41, 7)])
        args = SimpleNamespace(knn_backend='event-parallel', knn_block_size=256,
                               use_multihead_model=True, dp=False)
        model = GravNetModelMultiHead(input_dim=5, output_dim=5, n_heads=2,
                                     regression_output_dims=[1]).eval()
        keys = tuple(model.state_dict())
        with torch.no_grad():
            reference = model(data.x, data.batch, epoch=27)
        for block in model.gravnet_blocks:
            block.gravnet_layer.knn_backend = 'event-parallel'
            block.gravnet_layer.knn_block_size = 256
        import knn_event_parallel as ep
        with patch.object(ep, 'plan_from_batch', wraps=ep.plan_from_batch) as fallback:
            with torch.no_grad():
                implicit = model(data.x, data.batch, epoch=27)
            self.assertEqual(fallback.call_count, 1)
        data = prepare_training_batch(data, args).to('cpu')
        seen = []
        original = ep.knn_graph
        def capture(*a, **kw):
            seen.append(kw['plan'])
            return original(*a, **kw)
        with patch.object(ep, 'plan_from_batch', side_effect=AssertionError('unexpected fallback')):
            with patch.object(ep, 'knn_graph', side_effect=capture), torch.no_grad():
                explicit = training_forward(model, data, args, epoch=27)
        self.assertEqual(len(seen), 4)
        self.assertTrue(all(p is seen[0] for p in seen))
        self.assertEqual(seen[0][2], 256)
        for output in (implicit, explicit):
            for a, b in zip(reference['all_heads'], output['all_heads']):
                self.assertTrue(torch.equal(a, b))
        self.assertEqual(keys, tuple(model.state_dict()))


@unittest.skipUnless(torch.cuda.is_available(), 'GPU unavailable: bitwise checks must run on training GPU')
class PlannedGPUChecks(unittest.TestCase):
    def check(self, x, batch, ptr, k, radius):
        expected_i, expected_d = select_knn_baseline(x, k, batch, max_radius=radius)
        bits = torch.int32 if x.dtype == torch.float32 else torch.int64
        for block in BLOCK_SIZES:
            plan = prepare_plan(ptr, x.device, block)
            for packed, zero_init in ((False, True), (True, True), (False, False), (True, False)):
                i, d = select_knn(x, k, batch, max_radius=radius, plan=plan,
                                  packed=packed, zero_init=zero_init)
                self.assertTrue(torch.equal(expected_i, i), (block, packed, zero_init, 'indices'))
                self.assertTrue(torch.equal(expected_d.view(bits), d.view(bits)),
                                (block, packed, zero_init, 'distance bits'))

    def test_all_modes_ties_padding_and_empty_events(self):
        torch.manual_seed(1004)
        ptr = torch.tensor([0, 0, 1, 30, 70, 111, 240, 1265, 1265])
        batch = torch.arange(len(ptr)-1, device='cuda').repeat_interleave((ptr[1:]-ptr[:-1]).cuda())
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(int(ptr[-1]), 4, device='cuda', dtype=dtype)
            for values in (x, x.round(), torch.zeros_like(x)):
                for k, radius in ((1, 1e9), (41, 1e9), (41, .5), (5, -1.)):
                    self.check(values, batch, ptr, k, radius)
        self.check(x[:0], None, torch.tensor([0, 0]), 41, 1e9)
        plan = prepare_plan(ptr, 'cuda')
        for loop in (False, True):
            for flow in ('source_to_target', 'target_to_source'):
                old = knn_graph(x, 40, batch, loop=loop, flow=flow, baseline=True)
                new = knn_graph(x, 40, batch, loop=loop, flow=flow, plan=plan)
                self.assertTrue(torch.equal(old, new))

    def test_plan_upload_and_use_on_current_stream(self):
        ptr = torch.tensor([0, 29, 512])
        batch = torch.arange(2, device='cuda').repeat_interleave((ptr[1:]-ptr[:-1]).cuda())
        x = torch.randn(512, 4, device='cuda')
        expected = select_knn_baseline(x, 41, batch)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            plan = prepare_plan(ptr, 'cuda', 256)
            actual = select_knn(x + 0., 41, batch, plan=plan)
        stream.synchronize()
        for a, b in zip(expected, actual):
            self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)))


if __name__ == '__main__':
    unittest.main()
