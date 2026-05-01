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

from data.schema import ParetoPath
import config as cfg


def _normalize_objectives(paths: List[ParetoPath]) -> np.ndarray:
    vecs  = np.array([p.total_weight.as_tuple() for p in paths], dtype=float)
    mins  = vecs.min(axis=0)
    maxs  = vecs.max(axis=0)
    rngs  = maxs - mins
    rngs[rngs == 0] = 1.0
    return (vecs - mins) / rngs


def _kmeans(points: np.ndarray, k: int, seed: int, max_iter: int = 100):
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    first = int(rng.integers(n))
    centers = [points[first]]

    while len(centers) < k:
        center_mat = np.array(centers)
        dists = np.min(
            np.linalg.norm(points[:, None, :] - center_mat[None, :, :], axis=2),
            axis=1,
        )
        next_idx = int(np.argmax(dists))
        centers.append(points[next_idx])

    centers = np.array(centers)
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        dmat = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dmat, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for cluster_id in range(k):
            members = points[labels == cluster_id]
            if len(members):
                centers[cluster_id] = members.mean(axis=0)

    return labels, centers


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

        labels, centers = _kmeans(norm_vecs, k_actual, self.seed)

        selected: List[ParetoPath] = []
        for cluster_id in range(k_actual):
            cluster_mask = labels == cluster_id
            cluster_idxs = np.where(cluster_mask)[0]
            if len(cluster_idxs) == 0:
                continue
            centroid     = centers[cluster_id]
            # Pick the path closest to centroid
            dists        = np.linalg.norm(norm_vecs[cluster_idxs] - centroid, axis=1)
            best_in_cluster = cluster_idxs[int(np.argmin(dists))]
            selected.append(pareto_paths[best_in_cluster])

        return selected
