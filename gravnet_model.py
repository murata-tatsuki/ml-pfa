import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_min, scatter_max, scatter_mean

# from torch_cmspepr import GravNetConv
from gravnet_conv import GravNetConv
# from torch_cmspepr.objectcondensation import scatter_count
from objectcondensation import scatter_count

from typing import Tuple, Union, List

def global_exchange(x: Tensor, batch: Tensor) -> Tensor:
    """
    Adds columns for the means, mins, and maxs per feature, per batch.
    Assumes x: (n_hits x n_features), batch: (n_hits),
    and that the batches are sorted!
    """
    n_hits_per_event = scatter_count(batch)
    n_hits, n_features = x.size()
    batch_size = int(batch.max()) + 1

    # minmeanmax: (batch_size x 3*n_features)
    meanminmax = torch.cat((
        scatter_mean(x, batch, dim=0),
        scatter_min(x, batch, dim=0)[0],
        scatter_max(x, batch, dim=0)[0]
        ), dim=1)
    assert list(meanminmax.size()) == [batch_size, 3*n_features]

    meanminmax = torch.repeat_interleave(meanminmax, n_hits_per_event, dim=0)
    assert list(meanminmax.size()) == [n_hits, 3*n_features]

    out = torch.cat((meanminmax, x), dim=1)
    assert out.size() == (n_hits, 4*n_features)
    assert out.device == x.device
    return out


# FROM https://link.springer.com/content/pdf/10.1140/epjc/s10052-019-7113-9.pdf:

# GravNet model: The model consists of four blocks. Each
# block starts with concatenating the mean of the vertex
# features to the vertex features, three dense layers with
# 64 nodes and tanh activation, and one GravNet layer
# with S = 4 coordinate dimensions, FLR = 22 features to
# propagate, and FOUT = 48 output nodes per vertex. For
# each vertex, 40 neighbours are considered. The output
# of each block is passed as input to the next block and
# added to a list containing the output of all blocks. This
# determines the full vector of vertex features passed to a
# final dense layer with 128 nodes and ReLU activation

# In all cases, each output vertex of these model building blocks
# is fed through one dense layer with ReLU activation and three
# nodes, followed by a dense layer with two output nodes and
# softmax activation. This last processing step deter- mines the
# energy fraction belonging to each shower. Batch normalisation
# is applied in all models to the input and after each block.

class GravNetBlock(nn.Module):

    def __init__(
        self,
        in_channels: int, out_channels: int = 96,
        space_dimensions: int = 4, propagate_dimensions: int = 22, k: int = 40
        ):
        super(GravNetBlock, self).__init__()
        # Includes all layers up to the global_exchange
        self.gravnet_layer = GravNetConv(
                in_channels, out_channels,
                space_dimensions, propagate_dimensions, k
                ).jittable()
        self.post_gravnet = nn.Sequential(
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, 128),
            nn.Tanh(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 96),
            nn.Tanh(),
            )
        self.output = nn.Sequential(
            nn.Linear(4*96, 96),
            nn.Tanh(),
            nn.BatchNorm1d(96)
            )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        x = self.gravnet_layer(x, batch)
        x = self.post_gravnet(x)
        assert x.size(1) == 96
        x = global_exchange(x, batch)
        x = self.output(x)
        assert x.size(1) == 96
        return x


class GravnetModel(nn.Module):

    def __init__(
        self, 
        input_dim: int=5,
        output_dim: int=2,
        n_gravnet_blocks: int=4,
        n_postgn_dense_blocks: int=4,
        k: Union[List[int], int] = 40,
        ):
        super(GravnetModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gravnet_blocks = n_gravnet_blocks
        self.n_postgn_dense_blocks = n_postgn_dense_blocks

        self.batchnorm1 = nn.BatchNorm1d(self.input_dim)
        self.input = nn.Linear(4*input_dim, 64)

        print("Hello")

        if isinstance(k, int):
            k = n_gravnet_blocks*[k]

        assert len(k) == n_gravnet_blocks
        
        # Note: out_channels of the internal gravnet layer
        # not clearly specified in paper
        self.gravnet_blocks = nn.ModuleList([
            GravNetBlock(64 if i==0 else 96, k=k[i]) for i in range(self.n_gravnet_blocks)
            ])

        # Post-GravNet dense layers
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend([
                nn.Linear(4*96 if i==0 else 128, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                ])
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)
        
        # Output block
        self.output = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
            )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        device = x.device
        # print('forward called on device', device)
        x = self.batchnorm1(x)
        x = global_exchange(x, batch)
        x = self.input(x)
        assert x.device == device

        x_gravnet_per_block = [] # To store intermediate outputs
        for gravnet_block in self.gravnet_blocks:
            x = gravnet_block(x, batch)
            x_gravnet_per_block.append(x)
        x = torch.cat(x_gravnet_per_block, dim=-1)
        assert x.size() == (x.size(0), 4*96)
        assert x.device == device

        x = self.postgn_dense(x)
        x = self.output(x)
        assert x.device == device
        return x

class GravNetModelBranch(nn.Module):

    def __init__(
        self, 
        input_dim: int=5,
        output_dim: int=2,
        n_gravnet_blocks: int=4,
        n_postgn_dense_blocks: int=4,
        k: Union[List[int], int] = 40,
        b_energy_branch: bool = True,
        ):
        super(GravNetModelBranch, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gravnet_blocks = n_gravnet_blocks
        self.n_postgn_dense_blocks = n_postgn_dense_blocks

        self.batchnorm1 = nn.BatchNorm1d(self.input_dim)
        self.input = nn.Linear(4*input_dim, 64)

        self.dense_nord = 128
        self.b_energy_branch = b_energy_branch

        print("Hello")
        if self.b_energy_branch:
            print("!!!!energy branch separateed!!!!")

        if isinstance(k, int):
            k = n_gravnet_blocks*[k]

        assert len(k) == n_gravnet_blocks
        
        # Note: out_channels of the internal gravnet layer
        # not clearly specified in paper
        self.gravnet_blocks = nn.ModuleList([
            GravNetBlock(64 if i==0 else 96, k=k[i]) for i in range(self.n_gravnet_blocks)
            ])

        # Post-GravNet dense layers
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend([
                nn.Linear(4*96 if i==0 else self.dense_nord, self.dense_nord),
                nn.ReLU(),
                nn.BatchNorm1d(self.dense_nord),
                ])
        self.postgn_dense = nn.Sequential(*postgn_dense_modules)

        # energy regression branch
        self.energy_nord = 5    ## momentum, momentum norm, track bit
        self.energy_branch = nn.Sequential(
            nn.Linear(5, 5)
        )
        outblock_inNord = self.dense_nord + self.energy_nord if self.b_energy_branch else self.dense_nord
        
        # Output block
        self.output = nn.Sequential(
            nn.Linear(outblock_inNord, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
            )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        device = x.device
        energy_var = x[:,-5:-1]
        trackbit_var = x[:,4]
        trackbit_var = trackbit_var.view(trackbit_var.size()[0],1)
        energy_var = torch.cat([energy_var,trackbit_var], dim=-1)
        device = energy_var.device
        # print('forward called on device', device)
        x = self.batchnorm1(x)
        x = global_exchange(x, batch)
        x = self.input(x)
        assert x.device == device

        x_gravnet_per_block = [] # To store intermediate outputs
        for gravnet_block in self.gravnet_blocks:
            x = gravnet_block(x, batch)
            x_gravnet_per_block.append(x)
        x = torch.cat(x_gravnet_per_block, dim=-1)
        assert x.size() == (x.size(0), 4*96)
        assert x.device == device

        x = self.postgn_dense(x)
        energy_var = self.energy_branch(energy_var)
        x = torch.cat([x, energy_var], dim=-1)
        x = self.output(x)
        assert x.device == device
        return x

class GravNetModelMultiHead(nn.Module):
    """
    Multi-head GravNet model.

    - Trunk (before concatenating GravNet block outputs) is identical to the
      baseline GravNet implementation.
    - After concatenation, each head owns the full post-concat stack
      (post-GravNet dense blocks + output block).
    - Head-0 is intended for clustering (beta / virtual coordinate / etc.).
    - Head-1..N are intended for regression tasks (1 head = 1 regression by design).
    """

    def __init__(
        self,
        input_dim: int = 5,
        output_dim: int = 2,
        n_gravnet_blocks: int = 4,
        n_postgn_dense_blocks: int = 4,
        k: Union[List[int], int] = 40,
        n_heads: int = 1,
        regression_output_dims: Union[int, List[int]] = 1,
        interaction_start_epoch: int = 0,
        interaction_mode: str = "concat",
    ):
        super(GravNetModelMultiHead, self).__init__()
        if n_heads < 1:
            raise ValueError("n_heads must be >= 1")
        if interaction_mode not in ["none", "concat", "add", "gate"]:
            raise ValueError("interaction_mode must be one of: none, concat, add, gate")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_gravnet_blocks = n_gravnet_blocks
        self.n_postgn_dense_blocks = n_postgn_dense_blocks
        self.n_heads = n_heads
        self.dense_nord = 128
        self.interaction_start_epoch = interaction_start_epoch
        self.interaction_mode = interaction_mode

        # Keep the pre-concat trunk unchanged.
        self.batchnorm1 = nn.BatchNorm1d(self.input_dim)
        self.input = nn.Linear(4 * input_dim, 64)

        if isinstance(k, int):
            k = n_gravnet_blocks * [k]
        assert len(k) == n_gravnet_blocks

        self.gravnet_blocks = nn.ModuleList([
            GravNetBlock(64 if i == 0 else 96, k=k[i]) for i in range(self.n_gravnet_blocks)
        ])

        self.head_output_dims = self._build_head_output_dims(output_dim, n_heads, regression_output_dims)

        # One full post-concat head per task.
        self.head_postgn_dense = nn.ModuleList([
            self._make_postgn_dense() for _ in range(self.n_heads)
        ])
        self.head_output = nn.ModuleList([
            self._make_output_block(self.head_output_dims[i]) for i in range(self.n_heads)
        ])

        # Interaction layers: from clustering head feature -> regression head feature.
        self.interaction_layers = nn.ModuleList()
        for _ in range(max(0, self.n_heads - 1)):
            if self.interaction_mode == "concat":
                self.interaction_layers.append(nn.Sequential(
                    nn.Linear(2 * self.dense_nord, self.dense_nord),
                    nn.ReLU(),
                    nn.BatchNorm1d(self.dense_nord),
                ))
            elif self.interaction_mode in ["add", "gate"]:
                self.interaction_layers.append(nn.Linear(self.dense_nord, self.dense_nord))

    @staticmethod
    def _build_head_output_dims(
        clustering_output_dim: int,
        n_heads: int,
        regression_output_dims: Union[int, List[int]],
    ) -> List[int]:
        if n_heads == 1:
            return [clustering_output_dim]

        if isinstance(regression_output_dims, int):
            regression_dims = [regression_output_dims] * (n_heads - 1)
        else:
            regression_dims = list(regression_output_dims)
            if len(regression_dims) != (n_heads - 1):
                raise ValueError("len(regression_output_dims) must be n_heads - 1")

        return [clustering_output_dim] + regression_dims

    def _make_postgn_dense(self) -> nn.Sequential:
        postgn_dense_modules = nn.ModuleList()
        for i in range(self.n_postgn_dense_blocks):
            postgn_dense_modules.extend([
                nn.Linear(4 * 96 if i == 0 else self.dense_nord, self.dense_nord),
                nn.ReLU(),
                nn.BatchNorm1d(self.dense_nord),
            ])
        return nn.Sequential(*postgn_dense_modules)

    @staticmethod
    def _make_output_block(out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def _is_interaction_active(self, epoch: Union[int, None]) -> bool:
        if self.n_heads <= 1:
            return False
        if self.interaction_mode == "none":
            return False
        if epoch is None:
            return False
        return epoch >= self.interaction_start_epoch

    def forward(
        self,
        x: Tensor,
        batch: Tensor,
        epoch: Union[int, None] = None,
        return_dict: bool = True,
    ):
        device = x.device

        # Unchanged trunk before concatenating block outputs.
        x = self.batchnorm1(x)
        x = global_exchange(x, batch)
        x = self.input(x)
        assert x.device == device

        x_gravnet_per_block = []
        for gravnet_block in self.gravnet_blocks:
            x = gravnet_block(x, batch)
            x_gravnet_per_block.append(x)
        x = torch.cat(x_gravnet_per_block, dim=-1)
        assert x.size() == (x.size(0), 4 * 96)
        assert x.device == device

        head_features = [head_dense(x) for head_dense in self.head_postgn_dense]
        clustering_feature = head_features[0]
        interaction_active = self._is_interaction_active(epoch)

        outputs = []
        # for i, feat in enumerate(head_features):
        #     if i > 0 and interaction_active:
        #         layer = self.interaction_layers[i - 1]
        #         if self.interaction_mode == "concat":
        #             feat = layer(torch.cat([feat, clustering_feature], dim=-1))
        #         elif self.interaction_mode == "add":
        #             feat = feat + layer(clustering_feature)
        #         elif self.interaction_mode == "gate":
        #             gate = torch.sigmoid(layer(clustering_feature))
        #             feat = feat * gate
        # 
        #     out = self.head_output[i](feat)
        #     outputs.append(out)
        for i, feat in enumerate(head_features):
            if i > 0 and interaction_active:
                # head_0 (clustering) からの入力を計算
                context = self.interaction_layers[i - 1](clustering_feature)

                if self.interaction_mode == "gate":
                    # 0~1のゲートを生成して要素ごとにスケーリング
                    gate = torch.sigmoid(context)
                    feat = feat * gate 
                elif self.interaction_mode == "concat":
                    # 既存実装通り
                    feat = self.interaction_layers[i - 1](torch.cat([feat, clustering_feature], dim=-1))
            out = self.head_output[i](feat)
            outputs.append(out)

        if not return_dict:
            if self.n_heads == 1:
                return outputs[0]
            return tuple(outputs)

        return {
            "clustering": outputs[0],
            "regressions": outputs[1:],
            "all_heads": outputs,
            "interaction_active": interaction_active,
        }


class NoiseFilterModel(nn.Module):

    def __init__(
        self, 
        input_dim: int=5,
        output_dim: int=2,
        ):
        super(NoiseFilterModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.BatchNorm1d(self.input_dim),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
            nn.LogSoftmax()
            )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)

class GravnetModelWithNoiseFilter(nn.Module):

    def __init__(self, *args, **kwargs):
        super(GravnetModelWithNoiseFilter, self).__init__()
        self.signal_threshold = kwargs.pop('signal_threshold', .5)
        self.gravnet = GravnetModel(*args, **kwargs)
        self.noise_filter = NoiseFilterModel(input_dim=self.gravnet.input_dim)

    def forward(self, x: Tensor, batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        out_noise_filter = self.noise_filter(x)
        pass_noise_filter = torch.exp(out_noise_filter[:,1]) > self.signal_threshold
        # Get the GravNet model output on only hits that pass the noise threshold
        out_gravnet = self.gravnet(x[pass_noise_filter], batch[pass_noise_filter])
        return out_noise_filter, pass_noise_filter, out_gravnet
