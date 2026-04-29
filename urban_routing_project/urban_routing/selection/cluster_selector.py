"""
selection/cluster_selector.py — Clustering-based Top-K selection.

Strategy
--------
  1. Represent each Pareto path as a point in normalized objective space.
  2. Cluster the points into K clusters (k-means).
  3. For each cluster, select the path whose objective vector is closest
     to the cluster centroid (medoid-style).

This guarantees coverage of different "regions" of the Pareto front but
does NOT enforce structural (edge-set) diversity like DiversitySelector.

Used both as a standalone selector and as a comparison baseline.
"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.cluster import KMeans

from data.schema import ParetoPath
import config as cfg


def _normalize_objectives(paths: List[ParetoPath]) -> np.ndarray:
    vecs  = np.array([p.total_weight.as_tuple() for p in paths], dtype=float)
    mins  = vecs.min(axis=0)
    maxs  = vecs.max(axis=0)
    rngs  = maxs - mins
    rngs[rngs == 0] = 1.0
    return (vecs - mins) / rngs


class ClusterSelector:
    """
    Selects K representative paths via k-means clustering in objective space.
    """

    def __init__(
        self,
        k:    int = cfg.DEFAULT_TOP_K,
        seed: int = 42,
    ):
        self.k    = k
        self.seed = seed

    def select(self, pareto_paths: List[ParetoPath]) -> List[ParetoPath]:
        """
        Cluster paths in normalized objective space; return the medoid of each cluster.
        """
        if not pareto_paths:
            return []
        if len(pareto_paths) <= self.k:
            return list(pareto_paths)

        norm_vecs = _normalize_objectives(pareto_paths)
        k_actual  = min(self.k, len(pareto_paths))

        km = KMeans(n_clusters=k_actual, random_state=self.seed, n_init="auto")
        labels = km.fit_predict(norm_vecs)

        selected: List[ParetoPath] = []
        for cluster_id in range(k_actual):
            cluster_mask = labels == cluster_id
            cluster_idxs = np.where(cluster_mask)[0]
            if len(cluster_idxs) == 0:
                continue
            centroid     = km.cluster_centers_[cluster_id]
            # Pick the path closest to centroid
            dists        = np.linalg.norm(norm_vecs[cluster_idxs] - centroid, axis=1)
            best_in_cluster = cluster_idxs[int(np.argmin(dists))]
            selected.append(pareto_paths[best_in_cluster])

        return selected
