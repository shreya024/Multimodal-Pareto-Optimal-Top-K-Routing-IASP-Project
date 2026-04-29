"""
selection/diversity_selector.py — Top-K selection with diversity constraints.

Strategy
--------
Select K paths from the Pareto frontier that:
  1. Maximise spread across the objective space (cover extremes and interior).
  2. Enforce minimum pairwise Jaccard dissimilarity on edge sets, preventing
     the selection of K nearly-identical routes.

Algorithm (greedy farthest-point selection with diversity filter)
-----------------------------------------------------------------
  1. Normalize objective vectors to [0, 1].
  2. Seed with the path that minimises the primary objective (fastest).
  3. At each subsequent step, score each candidate by:
         score = min_dist_to_selected_set (in normalized objective space)
     subject to: min Jaccard dissimilarity to all selected paths ≥ threshold.
  4. Select the highest-scoring feasible candidate.
  5. If no candidate satisfies the dissimilarity constraint, relax it and retry.

Jaccard Dissimilarity
---------------------
  d_J(A, B) = 1 - |A ∩ B| / |A ∪ B|
where A, B are edge sets (frozenset of (src, dst, route_id) tuples).
A value of 0 means identical routes; 1 means completely disjoint.
"""
from __future__ import annotations

import math
from typing import List, Optional, Set

import numpy as np

from data.schema import ParetoPath
import config as cfg


def jaccard_dissimilarity(a: ParetoPath, b: ParetoPath) -> float:
    """Return Jaccard distance between two paths' edge sets."""
    ea, eb = a.edge_set, b.edge_set
    if not ea and not eb:
        return 0.0
    intersection = len(ea & eb)
    union        = len(ea | eb)
    return 1.0 - intersection / union if union > 0 else 0.0


def _normalize_objectives(paths: List[ParetoPath]) -> np.ndarray:
    """
    Return an (n, d) array of normalized objective vectors.
    Each dimension is scaled to [0, 1] by its min/max across paths.
    """
    vecs = np.array([p.total_weight.as_tuple() for p in paths], dtype=float)
    mins = vecs.min(axis=0)
    maxs = vecs.max(axis=0)
    ranges = maxs - mins
    # Avoid division by zero for constant objectives
    ranges[ranges == 0] = 1.0
    return (vecs - mins) / ranges


def _min_obj_distance(
    candidate_vec: np.ndarray,
    selected_vecs: np.ndarray,
) -> float:
    """L2 distance from candidate to nearest selected point in objective space."""
    dists = np.linalg.norm(selected_vecs - candidate_vec, axis=1)
    return float(dists.min())


class DiversitySelector:
    """
    Selects Top-K paths using greedy farthest-point sampling in objective space
    with a Jaccard dissimilarity constraint on route structure.
    """

    def __init__(
        self,
        k:                      int   = cfg.DEFAULT_TOP_K,
        min_jaccard_dissim:     float = cfg.MIN_JACCARD_DISSIMILARITY,
        primary_objective_idx:  int   = cfg.OBJ_TIME,
    ):
        self.k                    = k
        self.min_jaccard_dissim   = min_jaccard_dissim
        self.primary_objective_idx = primary_objective_idx

    def select(self, pareto_paths: List[ParetoPath]) -> List[ParetoPath]:
        """
        Select up to self.k diverse, representative paths from pareto_paths.

        Returns
        -------
        List of selected ParetoPath objects (length ≤ k).
        """
        if not pareto_paths:
            return []
        if len(pareto_paths) <= self.k:
            return list(pareto_paths)

        norm_vecs = _normalize_objectives(pareto_paths)  # (n, d)
        n = len(pareto_paths)

        selected_indices: List[int] = []
        selected_vecs:    List[np.ndarray] = []

        # ── Seed: fastest path (min primary objective) ──
        seed_idx = int(np.argmin(norm_vecs[:, self.primary_objective_idx]))
        selected_indices.append(seed_idx)
        selected_vecs.append(norm_vecs[seed_idx])

        remaining = set(range(n)) - {seed_idx}

        for _ in range(self.k - 1):
            if not remaining:
                break

            sel_mat = np.array(selected_vecs)  # shape (|S|, d)

            # Score each candidate
            best_score = -1.0
            best_idx:  Optional[int] = None
            relaxed_best_score = -1.0
            relaxed_best_idx:  Optional[int] = None

            for idx in remaining:
                dist  = _min_obj_distance(norm_vecs[idx], sel_mat)
                jacc  = min(
                    jaccard_dissimilarity(pareto_paths[idx], pareto_paths[s])
                    for s in selected_indices
                )

                if jacc >= self.min_jaccard_dissim:
                    if dist > best_score:
                        best_score = dist
                        best_idx   = idx

                # Track relaxed best (ignoring Jaccard) as fallback
                if dist > relaxed_best_score:
                    relaxed_best_score = dist
                    relaxed_best_idx   = idx

            chosen = best_idx if best_idx is not None else relaxed_best_idx
            if chosen is None:
                break

            selected_indices.append(chosen)
            selected_vecs.append(norm_vecs[chosen])
            remaining.discard(chosen)

        return [pareto_paths[i] for i in selected_indices]
