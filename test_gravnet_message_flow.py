"""Regression test for homogeneous features with target_to_source flow."""
import unittest
import torch
from torch_scatter import scatter
from gravnet_conv import GravNetConv, knn_graph


@unittest.skipIf(knn_graph is None, 'torch_cmspepr not installed')
class GravNetFlowTests(unittest.TestCase):
    def test_matches_direct_aggregation_and_has_gradients(self):
        torch.manual_seed(1009)
        layer = GravNetConv(5, 8, 4, 6, 2)
        x = torch.randn(12, 5, requires_grad=True)
        batch = torch.tensor([0]*6+[1]*6)
        actual = layer(x, batch)
        coordinates, features = layer.lin_s(x), layer.lin_h(x)
        edges = knn_graph(coordinates.float(), 2, batch)
        weights = torch.exp(-10*(coordinates[edges[1]]-coordinates[edges[0]]).square().sum(-1))
        messages = features[edges[1]]*weights[:, None]
        mean = scatter(messages, edges[0], dim=0, dim_size=12, reduce='mean')
        maximum = scatter(messages, edges[0], dim=0, dim_size=12, reduce='max')
        expected = layer.lin(torch.cat([mean, maximum, x], dim=1))
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all())


if __name__ == '__main__':
    unittest.main()
