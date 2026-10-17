"""Reusable beta/distance clustering caches.

The numerical algorithm is retained from the original fast scan writer.  A
beta worker reuses candidate ordering and hit distances for every td value.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch

import objectcondensation as _oc


class _PreparedClustering:
    """One event/beta cache shared by all distance thresholds in a worker."""

    def __init__(self, prediction, tbeta: float):
        self.betas = np.asarray(prediction.pred_betas)
        self.x = np.asarray(prediction.pred_cluster_space_coords)
        self.charged_hits = np.asarray(prediction.charged_hits)
        self.n_points = self.betas.shape[0]

        selected = self.betas > tbeta
        selected_indices = np.nonzero(selected)[0]
        rejected_indices = np.nonzero(~selected)[0]
        self.high_beta = selected_indices[
            np.argsort(-self.betas[selected])
        ]
        self.low_beta = rejected_indices[
            np.argsort(-self.betas[~selected])
        ]

        self._nearest_high_index: Optional[np.ndarray] = None
        self._nearest_high_distance: Optional[np.ndarray] = None
        self._distance_from_low: Dict[int, np.ndarray] = {}

    def _prepare_nearest_high(self) -> None:
        if self._nearest_high_index is not None or self.high_beta.size == 0:
            return

        # Keep the same per-hit norm and candidate order as the legacy code.
        nearest_index = np.empty(self.n_points, dtype=np.int64)
        nearest_distance = np.empty(self.n_points, dtype=self.x.dtype)
        high_x = self.x[self.high_beta]
        for point in range(self.n_points):
            distances = np.linalg.norm(self.x[point] - high_x, axis=-1)
            argmin = np.argmin(distances, axis=-1)
            nearest_index[point] = self.high_beta[argmin]
            nearest_distance[point] = distances[argmin]
        self._nearest_high_index = nearest_index
        self._nearest_high_distance = nearest_distance

    def _low_seed_distances(self, seed: int) -> np.ndarray:
        key = int(seed)
        distances = self._distance_from_low.get(key)
        if distances is None:
            distances = np.linalg.norm(self.x - self.x[seed], axis=-1)
            self._distance_from_low[key] = distances
        return distances

    def cluster(self, td: float) -> Tuple[np.ndarray, np.ndarray]:
        condensation_points = np.zeros(self.n_points, dtype=np.int32)
        clustering = -1 * np.ones(self.n_points, dtype=np.int32)

        if self.high_beta.size > 0:
            self._prepare_nearest_high()
            assigned = self._nearest_high_distance < td
            clustering[assigned] = self._nearest_high_index[assigned]
            unassigned = np.nonzero(~assigned)[0]
        else:
            unassigned = np.arange(self.n_points)

        for seed in self.low_beta:
            if unassigned.size == 0:
                break
            distances = self._low_seed_distances(int(seed))[unassigned]
            selected = distances < td
            assigned_to_seed = unassigned[selected]
            clustering[assigned_to_seed] = seed
            unassigned = unassigned[~selected]

        clustering_indices, clustering_frequency = np.unique(
            clustering, return_counts=True
        )
        clustering_count = dict(zip(clustering_indices, clustering_frequency))
        charged_hits = self.charged_hits.astype(int)
        charged_index = clustering[charged_hits == 1]

        charged_is_cluster = np.isin(charged_index, clustering_indices)
        if not np.all(charged_is_cluster):
            print("charge index NOT found in cluster")

        remap_charged = {}
        for charged_i in charged_index:
            if charged_i == -1:
                print("track -1 skipped")
                continue
            if clustering_count[charged_i] == 1:
                charged_cluster_distance = {}
                for clustering_i in clustering_indices:
                    if clustering_i == charged_i:
                        continue
                    distance = np.linalg.norm(
                        self.x[charged_i] - self.x[clustering_i], axis=-1
                    )
                    charged_cluster_distance[clustering_i] = distance
                index_min = min(
                    charged_cluster_distance,
                    key=charged_cluster_distance.get,
                )
                remap_charged[charged_i] = index_min

        for old_id, new_id in remap_charged.items():
            clustering[clustering == old_id] = new_id

        _, condensation_indices = _oc.scatter_max(
            torch.from_numpy(self.betas.astype(np.int64)),
            torch.from_numpy(clustering.astype(np.int64)),
        )
        condensation_indices = condensation_indices.detach().numpy()
        valid = condensation_indices[condensation_indices < self.n_points]
        condensation_points[valid] = 1
        return clustering + 1, condensation_points


class _BetaClusterEngine:
    def __init__(self, cache: Sequence, tbeta: float):
        self.tbeta = float(tbeta)
        self.prepared = [
            None if prediction.pred_betas is None else _PreparedClustering(prediction, tbeta)
            for _, prediction in cache
        ]

    def cluster(self, event_index: int, td: float):
        prepared = self.prepared[event_index]
        if prepared is None:
            raise RuntimeError("Fast beta scan cannot cluster a Pandora prediction")
        return prepared.cluster(td)
