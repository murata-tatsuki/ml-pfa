"""Device-resident membership for the predicted particle-cluster energy head.

The rules match ``objectcondensation.get_clustering_np_new``: first attach to
nearest high-beta seeds, then to the first low-beta seed within the radius,
then merge singleton track clusters without a distance cut.  Low-beta seeds
remain eligible even when they already belong to another cluster.

Equal beta values are ordered by original hit index.  The NumPy implementation
uses an unstable quicksort, so its order among equal values is not guaranteed
and can differ.  Floating-point distances at the radius boundary can also
round differently between NumPy/CPU and CUDA.  Membership remains discrete;
this module intentionally does not construct an autograd graph.
"""

from typing import Optional

import torch


def _distance_rows(points: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    # Match the explicit coordinate-difference norm used by NumPy.  In-place
    # squaring avoids retaining two coordinate-sized temporary tensors.
    difference = points[:, None, :] - centers[None, :, :]
    difference.square_()
    return difference.sum(dim=-1).sqrt_()


def _chunk_rows(n_centers: int, n_coordinates: int, element_size: int,
                max_temporary_bytes: int) -> int:
    # Allow for coordinate differences, distances, masks and reduction inputs.
    bytes_per_row = max(1, n_centers * (n_coordinates + 5) * element_size)
    return max(1, max_temporary_bytes // bytes_per_row)


def _merge_singleton_tracks(labels: torch.Tensor, coordinates: torch.Tensor,
                            track_mask: torch.Tensor,
                            max_temporary_bytes: int) -> torch.Tensor:
    n_points = labels.numel()
    if n_points <= 1:
        return labels

    # Index 0 represents the NumPy background cluster (-1).  Its representative
    # uses coordinates[-1], as in the existing implementation.
    counts = torch.zeros(n_points + 1, dtype=torch.long, device=labels.device)
    counts.scatter_add_(0, labels + 1, torch.ones_like(labels))
    # Preserve the legacy astype(int) == 1 rule for non-boolean masks too.
    is_track = track_mask if track_mask.dtype == torch.bool else track_mask.long() == 1
    track_singleton = is_track & (labels >= 0) & (counts[labels + 1] == 1)
    sources = labels[track_singleton]
    n_sources = sources.numel()
    if not n_sources:
        return labels

    # Sources occur in input track order.  Unique cluster labels are sorted;
    # this also preserves NumPy's choice when distances tie.
    clusters = torch.unique(labels, sorted=True)
    if clusters.numel() <= 1:
        # This covers degenerate input without asking for an empty argmin.
        return labels
    centers = coordinates[clusters]
    targets = torch.empty_like(sources)
    chunk_rows = _chunk_rows(clusters.numel(), coordinates.shape[1],
                             coordinates.element_size(), max_temporary_bytes)
    for begin in range(0, n_sources, chunk_rows):
        end = min(begin + chunk_rows, n_sources)
        distances = _distance_rows(coordinates[sources[begin:end]], centers)
        distances.masked_fill_(sources[begin:end, None] == clusters[None, :],
                               float('inf'))
        targets[begin:end] = clusters[distances.argmin(dim=1)]

    # The original code performs substitutions in track order.  An earlier
    # source follows another source's substitution only when that other source
    # is processed later.  Those increasing-rank edges form an acyclic graph;
    # pointer jumping reproduces ordered substitution in logarithmic rounds,
    # including mutual nearest-neighbor cycles, without a per-track GPU sync.
    ranks = torch.arange(n_sources, device=labels.device, dtype=torch.long)
    source_rank = torch.full((n_points + 1,), -1, device=labels.device,
                             dtype=torch.long)
    source_rank[sources + 1] = ranks
    target_rank = source_rank[targets + 1]
    successors = torch.where(target_rank > ranks, target_rank, ranks)
    for _ in range(n_sources.bit_length()):
        targets = targets[successors]
        successors = successors[successors]

    remap = torch.arange(-1, n_points, device=labels.device, dtype=torch.long)
    remap[sources + 1] = targets
    return remap[labels + 1]


def _event_membership(betas: torch.Tensor, coordinates: torch.Tensor,
                      track_mask: torch.Tensor, tbeta: float, td: float,
                      max_temporary_bytes: int) -> torch.Tensor:
    n_points = betas.numel()
    if n_points <= 1:
        # The existing training wrapper assigns the sole hit to cluster 0.
        return torch.zeros(n_points, dtype=torch.long, device=betas.device)

    # Ascending negative beta has the same NaN-last convention as NumPy.
    order = torch.sort(-betas, stable=True)[1]
    centers = coordinates[order]
    high_beta = betas[order] > tbeta
    low_beta = ~high_beta
    labels = torch.full((n_points,), -1, dtype=torch.long, device=betas.device)
    chunk_rows = _chunk_rows(n_points, coordinates.shape[1],
                             coordinates.element_size(), max_temporary_bytes)
    for begin in range(0, n_points, chunk_rows):
        end = min(begin + chunk_rows, n_points)
        distances = _distance_rows(coordinates[begin:end], centers)

        # Every low-beta seed remains eligible, so the ordered legacy loop is
        # equivalent to selecting the first in-radius low-beta column per hit.
        low_valid = (distances < td) & low_beta[None, :]
        low_found, low_nearest = low_valid.to(torch.int8).max(dim=1)
        low_labels = order[low_nearest]
        low_labels = torch.where(low_found.bool(), low_labels,
                                 torch.full_like(low_labels, -1))

        distances.masked_fill_(~high_beta[None, :], float('inf'))
        high_distance, high_nearest = distances.min(dim=1)
        labels[begin:end] = torch.where(high_distance < td, order[high_nearest],
                                       low_labels)

    return _merge_singleton_tracks(labels, coordinates, track_mask,
                                   max_temporary_bytes)


@torch.no_grad()
def cluster_membership_torch(betas: torch.Tensor, coordinates: torch.Tensor,
                              batch: torch.Tensor, track_mask: Optional[torch.Tensor],
                              tbeta: float = 0.9, td: float = 0.5, *,
                              max_temporary_bytes: int = 64 << 20) -> torch.Tensor:
    """Return event-local seed labels + 1 (zero denotes unassigned hits).

    All tensors, distance calculations and the returned membership remain on
    ``coordinates.device``.  No NumPy conversion or device-to-host tensor copy
    is used.  Per-event index gathering has a dynamic output shape, which can
    synchronize CUDA once per event; the distance and seed loops themselves do
    not read tensor values on the host.  Temporary distance computation is
    chunked to approximately ``max_temporary_bytes`` (64 MiB by default).

    Arbitrary/noncontiguous event IDs and interleaved input events are accepted.
    Labels refer to each hit's position within its own event, as in the existing
    NumPy training path, and are not globally unique across events.
    """
    if coordinates.ndim != 2:
        raise ValueError('coordinates must have shape [N, D]')
    if betas.ndim != 1 or batch.ndim != 1:
        raise ValueError('betas and batch must have shape [N]')
    if betas.numel() != coordinates.shape[0] or batch.numel() != betas.numel():
        raise ValueError('betas, coordinates and batch must have the same hit count')
    if not coordinates.is_floating_point():
        raise ValueError('coordinates must have a floating-point dtype')
    if betas.device != coordinates.device or batch.device != coordinates.device:
        raise ValueError('betas, coordinates and batch must use the same device')
    if max_temporary_bytes <= 0:
        raise ValueError('max_temporary_bytes must be positive')
    if track_mask is None:
        track_mask = torch.zeros_like(batch, dtype=torch.bool)
    if track_mask.ndim != 1 or track_mask.numel() != betas.numel():
        raise ValueError('track_mask must have shape [N]')
    if track_mask.device != coordinates.device:
        raise ValueError('track_mask must use the same device as coordinates')

    result = torch.zeros_like(batch, dtype=torch.long)
    for event_id in torch.unique(batch, sorted=True):
        indices = torch.nonzero(batch == event_id, as_tuple=False).flatten()
        labels = _event_membership(betas[indices], coordinates[indices],
                                   track_mask[indices], tbeta, td,
                                   max_temporary_bytes)
        result[indices] = labels + 1
    return result
