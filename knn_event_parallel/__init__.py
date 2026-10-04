"""Opt-in event-parallel CUDA KNN, retaining the legacy candidate search.

Build from the repository root with:
    TORCH_CUDA_ARCH_LIST=9.0 python knn_event_parallel/build.py build_ext --inplace
Use --knn-backend event-parallel in train.py; the default remains legacy.
CPU calls use the installed legacy CPU implementation. Batches must be sorted.
Derived wrappers: cms-pepr/pytorch_cmspepr e94c49b; see LICENSE (BSD-3-Clause).
"""
from pathlib import Path
from typing import Optional, Tuple
import torch

_loaded = False


def load_extension():
    """Load an explicitly built local library, never compile during training."""
    global _loaded
    if not _loaded:
        library = Path(__file__).resolve().parent / '_C.so'
        if not library.is_file():
            raise RuntimeError('Build the optional KNN extension first: '
                               'python knn_event_parallel/build.py build_ext --inplace '
                               '(set TORCH_CUDA_ARCH_LIST for your GPU).')
        torch.ops.load_library(str(library))
        if not hasattr(torch.ops.pfa_knn_event_parallel, "select_knn_planned_cuda"):
            raise RuntimeError("Rebuild the KNN extension after updating its sources")
        _loaded = True


def select_knn_baseline(
    x: torch.Tensor,
    k: int,
    batch_x: Optional[torch.Tensor] = None,
    inmask: Optional[torch.Tensor] = None,
    max_radius: float = 1e9,
    mask_mode: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Finds for each element in :obj:`x` the :obj:`k` nearest points in
    :obj:`x`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch_x` needs to be sorted.
            (default: :obj:`None`)
        max_radius (float): Maximum distance to nearest neighbours. (default: :obj:`1e9`)
        mask_mode (int): ??? (default: :obj:`1`)

    :rtype: :class:`Tuple`[`LongTensor`,`FloatTensor`]

    .. code-block:: python

        import torch
        from torch_cmspepr import select_knn

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch_x = torch.tensor([0, 0, 0, 0])
        assign_index = select_knn(x, 2, batch_x)
    """
    x = x.view(-1, 1) if x.dim() == 1 else x
    x = x.contiguous()

    mask: torch.Tensor = torch.ones(x.shape[0], dtype=torch.int32, device=x.device)
    if inmask is not None:
        mask = inmask

    # Compute row_splits
    if batch_x is None:
        row_splits: torch.Tensor = torch.tensor(
            [0, x.shape[0]], dtype=torch.int32, device=x.device
        )
    else:
        assert x.size(0) == batch_x.size(0)
        batch_size = int(batch_x.max()) + 1

        # Get number of hits per event
        counts = torch.zeros(batch_size, dtype=torch.int32, device=x.device)
        counts.scatter_add_(0, batch_x, torch.ones_like(batch_x, dtype=torch.int32))

        # Convert counts to row_splits by using cumsum.
        # row_splits must start with 0 and end with x.size(0), and has length +1 w.r.t.
        # batch_size.
        # e.g. for 2 events with 5 and 4 hits, row_splits would be [0, 5, 9]
        row_splits = torch.zeros(batch_size + 1, dtype=torch.int32, device=x.device)
        torch.cumsum(counts, 0, out=row_splits[1:])

    if x.device == torch.device('cpu'):
        import torch_cmspepr  # Register the unchanged CPU operator lazily.
        return torch.ops.select_knn_cpu.select_knn_cpu(
            x,
            row_splits,
            mask,
            k,
            max_radius,
            mask_mode,
        )
    else:
        load_extension()
        return torch.ops.pfa_knn_event_parallel.select_knn_cuda(
            x,
            row_splits,
            mask,
            k,
            max_radius,
            mask_mode,
        )


# rows, packed (event, first-query) tasks, block size, rectangular width, N.
# Plans contain no learned coordinates/neighbor indices and are batch-local.
KNNPlan = Tuple[torch.Tensor, torch.Tensor, int, int, int]
BLOCK_SIZES = (128, 256, 512, 1024)


def prepare_plan(ptr: torch.Tensor, device=None, block_size: int = 128) -> KNNPlan:
    """Validate CPU Batch.ptr and upload launch metadata once for all layers.

    Empty events are allowed. Do not mutate or reuse a plan for another batch.
    CPU-only construction is intentional: never read a GPU scalar to launch KNN.
    """
    if ptr.device.type != 'cpu':
        raise ValueError('prepare_plan requires CPU ptr; prepare before data.to(device)')
    if ptr.dtype not in (torch.int32, torch.int64) or ptr.dim() != 1:
        raise ValueError('ptr must be a one-dimensional integer tensor')
    if block_size not in BLOCK_SIZES:
        raise ValueError('block_size must be one of ' + str(BLOCK_SIZES))
    rows = ptr.tolist()
    if not 2 <= len(rows) <= 65536 or rows[0] != 0:
        raise ValueError('ptr must start at zero and describe 1..65535 events')
    if any(a > b for a, b in zip(rows, rows[1:])) or rows[-1] > 2**31 - 1:
        raise ValueError('ptr must be nondecreasing and fit int32')
    events, starts = [], []
    max_blocks = 0
    for event, (start, end) in enumerate(zip(rows, rows[1:])):
        count = (end - start + block_size - 1) // block_size
        max_blocks = max(max_blocks, count)
        events.extend([event] * count)
        starts.extend(range(start, end, block_size))
    rows_tensor = torch.tensor(rows, dtype=torch.int32, device=device)
    tasks = torch.tensor([events, starts], dtype=torch.int32, device=device)
    return rows_tensor, tasks, block_size, max_blocks, rows[-1]


def plan_from_batch(x: torch.Tensor, batch: Optional[torch.Tensor],
                    block_size: int = 128) -> KNNPlan:
    """Compatibility fallback for callers without CPU ptr (one CPU copy).

    Training supplies a prebuilt plan instead. Model forward calls this fallback
    only once, then passes the same plan through every GravNet block.
    """
    if batch is None:
        ptr = torch.tensor([0, x.size(0)], dtype=torch.int64)
    else:
        if batch.dim() != 1 or batch.numel() != x.size(0) or batch.dtype != torch.int64:
            raise ValueError('batch must be int64 with one entry per coordinate')
        ids = batch.detach().cpu()
        if ids.numel() and (int(ids[0]) < 0 or bool((ids[1:] < ids[:-1]).any())):
            raise ValueError('batch must be sorted and nonnegative')
        if ids.numel() and int(ids[-1]) >= 65535:
            raise ValueError('batch describes too many events')
        counts = torch.bincount(ids, minlength=1)
        ptr = torch.cat((torch.zeros(1, dtype=torch.int64), counts.cumsum(0)))
    return prepare_plan(ptr, x.device, block_size)


def select_knn(x: torch.Tensor, k: int, batch_x: Optional[torch.Tensor] = None,
               inmask: Optional[torch.Tensor] = None, max_radius: float = 1e9,
               mask_mode: int = 1, plan: Optional[KNNPlan] = None,
               packed: bool = True, zero_init: bool = False):
    """Same search/order as the original event-parallel operator.

    packed=False and zero_init=True isolate changes 3 and 5 in benchmarks.
    The unchanged select_knn_baseline isolates change 1 as well.
    """
    if x.device.type == 'cpu':
        return select_knn_baseline(x, k, batch_x, inmask, max_radius, mask_mode)
    x = (x.view(-1, 1) if x.dim() == 1 else x).contiguous()
    if plan is None:
        plan = plan_from_batch(x, batch_x)
    rows, tasks, block_size, max_blocks, n_vert = plan
    if n_vert != x.size(0) or rows.device != x.device or tasks.device != x.device:
        raise ValueError('KNN plan does not match coordinate shape/device')
    mask = torch.ones(x.size(0), dtype=torch.int32, device=x.device) if inmask is None else inmask
    load_extension()
    return torch.ops.pfa_knn_event_parallel.select_knn_planned_cuda(
        x, rows, mask, k, max_radius, mask_mode, tasks, block_size,
        max_blocks, packed, zero_init)


def prepare_training_batch(data, args):
    """Called before H2D; PyG transfers plan tensors with the other batch fields."""
    # Batch.ptr is already on the CPU before H2D. Reuse these exact counts
    # in all global exchanges; DataParallel replicas compute their own counts.
    ptr = getattr(data, 'ptr', None)
    if ptr is not None and not getattr(args, 'dp', False):
        data.event_counts = (ptr[1:] - ptr[:-1]).long()
    if getattr(args, 'knn_backend', 'legacy') == 'event-parallel':
        if getattr(args, 'dp', False):
            raise ValueError('event-parallel launch plans support single-device/DDP, not DataParallel')
        ptr = getattr(data, 'ptr', None)
        if ptr is None:
            raise ValueError('event-parallel training requires Batch.ptr')
        data.knn_plan = prepare_plan(ptr, block_size=getattr(args, 'knn_block_size', 128))
    return data


def knn_graph(
    x: torch.Tensor,
    k: int,
    batch: Optional[torch.Tensor] = None,
    loop: bool = False,
    flow: str = 'source_to_target',
    cosine: bool = False,
    num_workers: int = 1,
    max_radius: float = 1e9,
    plan: Optional[KNNPlan] = None,
    packed: bool = True,
    zero_init: bool = False,
    baseline: bool = False,
) -> torch.Tensor:
    r"""Computes graph edges to the nearest :obj:`k` points.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. :obj:`batch` needs to be sorted.
            (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        flow (string, optional): The flow direction when used in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)
        cosine (boolean, optional): If :obj:`True`, will use the Cosine
            distance instead of Euclidean distance to find nearest neighbors.
            (default: :obj:`False`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch` is not :obj:`None`, or the input lies
            on the GPU. (default: :obj:`1`)

    :rtype: :class:`LongTensor`

    .. code-block:: python

        import torch
        from torch_cluster import knn_graph

        x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        batch = torch.tensor([0, 0, 0, 0])
        edge_index = knn_graph(x, k=2, batch=batch, loop=False)
    """
    assert flow in ['source_to_target', 'target_to_source']
    # Ask for k+1 neighbors if loop=True, since select_knn will always contain the self-loop
    K = k if loop else k + 1
    if baseline:
        neighbours, edge_dists = select_knn_baseline(x, K, batch, max_radius=max_radius)
    else:
        neighbours, edge_dists = select_knn(x, K, batch, max_radius=max_radius,
                                          plan=plan, packed=packed, zero_init=zero_init)

    # neighbours has the following (n_neigh x k) structure:
    # [[0,  1,  3, ...],  <-- node 0 connected with 0, 1, 3, ...
    #  [1,  0,  -1, ...]   <-- node 1 connected with 1 and 0
    #  [2, -1, -1, ...]   <-- node 2 connected with 2 and nothing else
    #  ...]
    # Flatten it to a 1-dim tensor; Drop first column if not doing the self loop
    if loop:
        targets = neighbours.flatten()
    else:
        targets = neighbours[:, 1:].flatten()

    # Create sources:
    #   <--k--> <--k-->
    # [ 0 0 0 0 1 1 1 1 ... n_nodes n_nodes n_nodes n_nodes]
    sources = torch.repeat_interleave(torch.arange(x.size(0), device=x.device), k)

    if flow == 'source_to_target':
        edge_index = torch.stack((sources, targets))
    else:
        edge_index = torch.stack((targets, sources))

    # Filter out non-edges (target is -1)
    edge_index = edge_index[:, (targets >= 0)]

    return edge_index
