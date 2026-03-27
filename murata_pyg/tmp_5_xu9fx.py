import typing
from typing import *
from torch_geometric.typing import *

import torch
from torch import Tensor
import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.message_passing import *
from gravnet_conv import *


class Propagate_03b629(NamedTuple):
    x: OptPairTensor
    edge_weight: OptTensor



class Collect_03b629(NamedTuple):
    x_j: torch.Tensor
    edge_weight: torch.Tensor
    index: torch.Tensor
    dim_size: Optional[int]



class GravNetConvJittable_03b629(GravNetConv):

    @torch.jit._overload_method
    def __check_input__(self, edge_index, size):
        # type: (Tensor, Size) -> List[Optional[int]]
        pass

    @torch.jit._overload_method
    def __check_input__(self, edge_index, size):
        # type: (SparseTensor, Size) -> List[Optional[int]]
        pass

    def __check_input__(self, edge_index, size):
        the_size: List[Optional[int]] = [None, None]

        if isinstance(edge_index, Tensor):
            if not edge_index.dtype == torch.long:
                raise ValueError(f"Expected 'edge_index' to be of type "
                                 f"'torch.long' (got '{edge_index.dtype}')")
            if edge_index.dim() != 2:
                raise ValueError(f"Expected 'edge_index' to be two-dimensional"
                                 f" (got {edge_index.dim()} dimensions)")
            if edge_index.size(0) != 2:
                raise ValueError(f"Expected 'edge_index' to have size '2' in "
                                 f"the first dimension (got "
                                 f"'{edge_index.size(0)}')")
            if size is not None:
                the_size[0] = size[0]
                the_size[1] = size[1]
            return the_size

        elif isinstance(edge_index, SparseTensor):
            if self.flow == 'target_to_source':
                raise ValueError(
                    ('Flow direction "target_to_source" is invalid for '
                     'message propagation via `torch_sparse.SparseTensor`. If '
                     'you really want to make use of a reverse message '
                     'passing flow, pass in the transposed sparse tensor to '
                     'the message passing module, e.g., `adj_t.t()`.'))
            the_size[0] = edge_index.sparse_size(1)
            the_size[1] = edge_index.sparse_size(0)
            return the_size

        raise ValueError(
            ('`MessagePassing.propagate` only supports `torch.LongTensor` of '
             'shape `[2, num_messages]` or `torch_sparse.SparseTensor` for '
             'argument `edge_index`.'))

    @torch.jit._overload_method
    def __lift__(self, src, edge_index, dim):
        # type: (Tensor, Tensor, int) -> Tensor
        pass

    @torch.jit._overload_method
    def __lift__(self, src, edge_index, dim):
        # type: (Tensor, SparseTensor, int) -> Tensor
        pass

    def __lift__(self, src, edge_index, dim):
        if isinstance(edge_index, Tensor):
            index = edge_index[dim]
            return src.index_select(self.node_dim, index)
        elif isinstance(edge_index, SparseTensor):
            if dim == 1:
                rowptr = edge_index.storage.rowptr()
                rowptr = expand_left(rowptr, dim=self.node_dim, dims=src.dim())
                return gather_csr(src, rowptr)
            elif dim == 0:
                col = edge_index.storage.col()
                return src.index_select(self.node_dim, col)
        raise ValueError

    @torch.jit._overload_method
    def __collect__(self, edge_def, size, kwargs):
        # type: (Tensor, List[Optional[int]], Propagate_03b629) -> Collect_03b629
        pass

    @torch.jit._overload_method
    def __collect__(self, edge_def, size, kwargs):
        # type: (SparseTensor, List[Optional[int]], Propagate_03b629) -> Collect_03b629
        pass

    def __collect__(self, edge_def, size, kwargs):
        init = torch.tensor(0.)
        i, j = (1, 0) if self.flow == 'source_to_target' else (0, 1)

        edge_weight = kwargs.edge_weight
        x_j: torch.Tensor = init
        data = kwargs.x
        if isinstance(data, (tuple, list)):
            assert len(data) == 2
            tmp = data[1]
            if isinstance(tmp, Tensor):
                self.__set_size__(size, 1, tmp)
            x_j = data[0]
        else:
            x_j = data
        if isinstance(x_j, Tensor):
            self.__set_size__(size, 0, x_j)
            x_j = self.__lift__(x_j, edge_def, j)

        edge_index: Optional[Tensor] = None
        adj_t: Optional[SparseTensor] = None
        edge_index_i: torch.Tensor = init
        edge_index_j: torch.Tensor = init
        ptr: Optional[Tensor] = None
        if isinstance(edge_def, Tensor):
            edge_index = edge_def
            edge_index_i = edge_def[i]
            edge_index_j = edge_def[j]
        elif isinstance(edge_def, SparseTensor):
            adj_t = edge_def
            edge_index_i = edge_def.storage.row()
            edge_index_j = edge_def.storage.col()
            ptr = edge_def.storage.rowptr()
            
            if edge_weight is None:
                edge_weight = edge_def.storage.value()
            
            
            

        assert edge_weight is not None
        
        

        index = edge_index_i
        size_i = size[1] if size[1] is not None else size[0]
        size_j = size[0] if size[0] is not None else size[1]
        dim_size = size_i

        return Collect_03b629(x_j=x_j, edge_weight=edge_weight, index=index, dim_size=dim_size)



    @torch.jit._overload_method
    def propagate(self, edge_index, x, edge_weight, size=None):
        # type: (Tensor, OptPairTensor, OptTensor, Size) -> Tensor
        pass

    @torch.jit._overload_method
    def propagate(self, edge_index, x, edge_weight, size=None):
        # type: (SparseTensor, OptPairTensor, OptTensor, Size) -> Tensor
        pass

    def propagate(self, edge_index, x, edge_weight, size=None):
        the_size = self.__check_input__(edge_index, size)
        in_kwargs = Propagate_03b629(x=x, edge_weight=edge_weight)

        

        kwargs = self.__collect__(edge_index, the_size, in_kwargs)
        out = self.message(edge_weight=kwargs.edge_weight, x_j=kwargs.x_j)
        out = self.aggregate(out, dim_size=kwargs.dim_size, index=kwargs.index)
        return self.update(out)


    def edge_updater(self):
        pass


    @property
    def explain(self) -> bool:
        return self._explain

    @explain.setter
    def explain(self, explain: bool):
        raise ValueError("Explainability of message passing modules "
                         "is only supported on the Python module")

    def forward(
            self, x: Tensor,
            batch: OptTensor = None) -> Tensor:
        """"""

        assert x.dim() == 2, 'Static graphs not supported in `GravNetConv`.'

        b: OptTensor = None
        if isinstance(batch, Tensor):
            b = batch

        h_l: Tensor = self.lin_h(x)

        s_l: Tensor = self.lin_s(x)

        # print("GravnetConv: space coordinate shape:", s_l.shape)
        # print("GravnetConv: space coordinate:", s_l)

        edge_index = knn_graph(s_l, self.k, b)

        edge_weight = (s_l[edge_index[1]] - s_l[edge_index[0]]).pow(2).sum(-1)
        edge_weight = torch.exp(-10. * edge_weight)  # 10 gives a better spread

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=(h_l, None),
                             edge_weight=edge_weight,
                             size=(s_l.size(0), s_l.size(0)))

        return self.lin(torch.cat([out, x], dim=-1))
