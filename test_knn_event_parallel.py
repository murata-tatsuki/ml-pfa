"""Exact legacy-vs-parallel GPU checks; CPU checks do not need the new library."""
import unittest
import torch
from torch_cmspepr import select_knn as legacy_select, knn_graph as legacy_graph
from knn_event_parallel import select_knn, knn_graph
from gravnet_conv import GravNetConv, configure_knn_backend


def assert_exact(test, x, batch, k, radius=1e9):
    old_i, old_d = legacy_select(x, k, batch, max_radius=radius)
    new_i, new_d = select_knn(x, k, batch, max_radius=radius)
    test.assertTrue(torch.equal(old_i, new_i), 'neighbor indices differ')
    bits = torch.int32 if x.dtype == torch.float32 else torch.int64
    test.assertTrue(torch.equal(old_d.view(bits), new_d.view(bits)), 'distance bits differ')


class BackendCPUChecks(unittest.TestCase):
    def test_cpu_fallback_and_legacy_default(self):
        x = torch.randn(20, 4)
        batch = torch.arange(2).repeat_interleave(10)
        assert_exact(self, x, batch, 5)
        layer = GravNetConv(4, 8, 4, 6, 5)
        self.assertEqual(layer.knn_backend, 'legacy')
        state = layer.state_dict()
        configure_knn_backend(layer, 'legacy')
        self.assertEqual(state.keys(), layer.state_dict().keys())
        with self.assertRaises(ValueError):
            configure_knn_backend(layer, 'invalid')

@unittest.skipUnless(torch.cuda.is_available(), 'CUDA test; run inside interactive allocation')
class ParallelGPUChecks(unittest.TestCase):
    def test_exact_search_and_edge_order(self):
        torch.manual_seed(1009)
        # 32 events, unequal sizes, including singleton and fewer-than-k events.
        sizes = torch.tensor([1, 3, 7, 40, 41, 65, 129, 257] * 4, device='cuda')
        batch = torch.arange(32, device='cuda').repeat_interleave(sizes)
        for dtype in (torch.float32, torch.float64):
            for dimensions in (1, 4):
                random = torch.randn(len(batch), dimensions, device='cuda', dtype=dtype)
                for kind, x in (('random', random), ('ties', random.round()),
                                ('duplicates', torch.zeros_like(random))):
                    for k in (1, 5, 41):
                        for radius in (-1., .5, 1e9):
                            with self.subTest(dtype=dtype, dimensions=dimensions, kind=kind, k=k, radius=radius):
                                assert_exact(self, x, batch, k, radius)
        for loop in (True, False):
            for flow in ('source_to_target', 'target_to_source'):
                old = legacy_graph(random, 40, batch, loop=loop, flow=flow, max_radius=.5)
                new = knn_graph(random, 40, batch, loop=loop, flow=flow, max_radius=.5)
                self.assertTrue(torch.equal(old, new))
        assert_exact(self, random[:65].T.contiguous().T, None, 41)

    def test_current_stream(self):
        x = torch.randn(512, 4, device='cuda')
        reference = legacy_select(x, 41)
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            # Produce coordinates on the non-default stream before KNN consumes them.
            actual = select_knn(x + 0., 41)
        stream.synchronize()
        for a, b in zip(reference, actual):
            self.assertTrue(torch.equal(a, b))

    def test_gravnet_outputs_gradients_and_state_dict(self):
        torch.manual_seed(1009)
        layer = GravNetConv(5, 8, 4, 6, 40).cuda()
        x = torch.randn(32 * 65, 5, device='cuda', requires_grad=True)
        batch = torch.arange(32, device='cuda').repeat_interleave(65)
        state_keys = layer.state_dict().keys()
        outputs, gradients = [], []
        for backend in ('legacy', 'event-parallel'):
            configure_knn_backend(layer, backend)
            layer.zero_grad(set_to_none=True)
            x.grad = None
            y = layer(x, batch)
            y.square().mean().backward()
            outputs.append(y.detach())
            gradients.append([x.grad.clone()] + [p.grad.clone() for p in layer.parameters()])
        self.assertEqual(state_keys, layer.state_dict().keys())
        torch.testing.assert_close(*outputs, atol=2e-6, rtol=2e-5)
        for a, b in zip(*gradients):
            self.assertTrue(torch.isfinite(b).all())
            torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main()
