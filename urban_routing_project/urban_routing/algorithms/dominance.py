"""
algorithms/dominance.py — Dominance relation utilities and Pareto frontier pruning.

Used both during the Dijkstra expansion (per-node) and for final frontier
post-processing.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from data.schema import EdgeWeight


# ── Vector-level dominance ────────────────────────────────────────────────────

def dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
    """
    Return True if vector a Pareto-dominates vector b.
    a dominates b ⟺ a[i] ≤ b[i] ∀i  AND  a[i] < b[i] for some i.
    """
    at_least = all(x <= y for x, y in zip(a, b))
    strictly = any(x <  y for x, y in zip(a, b))
    return at_least and strictly


def is_non_dominated(vec: Tuple[float, ...], frontier: List[Tuple[float, ...]]) -> bool:
    """Return True if vec is not dominated by any vector in frontier."""
    return not any(dominates(f, vec) for f in frontier)


# ── Frontier pruning on a list of cost vectors ────────────────────────────────

def prune_to_pareto_front(
    vectors: List[Tuple[float, ...]],
) -> List[int]:
    """
    Given a list of cost tuples, return the indices of non-dominated vectors.
    O(n² · d) naive implementation (sufficient for typical frontier sizes ≤ 500).
    """
    n = len(vectors)
    dominated = [False] * n
    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            if dominates(vectors[i], vectors[j]):
                dominated[j] = True
    return [i for i in range(n) if not dominated[i]]


def fast_pareto_filter(matrix: np.ndarray) -> np.ndarray:
    """
    Efficient Pareto filtering using numpy.
    matrix: shape (n, d)  — each row is a cost vector (minimize all objectives).
    Returns boolean mask of non-dominated rows.
    """
    n = matrix.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        # Check if row i is dominated by any other currently-pareto row
        others = is_pareto.copy()
        others[i] = False
        if np.any(
            np.all(matrix[others] <= matrix[i], axis=1) &
            np.any(matrix[others] <  matrix[i], axis=1)
        ):
            is_pareto[i] = False
    return is_pareto


# ── Hypervolume indicator (2-D exact, higher-D Monte-Carlo) ───────────────────

def hypervolume_2d(pareto_costs: List[Tuple[float, float]], ref: Tuple[float, float]) -> float:
    """
    Exact hypervolume for 2-D Pareto fronts (time vs cost as default projection).
    ref should be a point dominated by all frontier points (i.e. worst case + ε).
    """
    pts = sorted(pareto_costs, key=lambda p: p[0])  # sort by first objective
    hv = 0.0
    prev_x = ref[0]
    for x, y in reversed(pts):
        if y < ref[1]:
            hv += (prev_x - x) * (ref[1] - y)
            prev_x = x
    return hv


def hypervolume_mc(
    pareto_costs: List[Tuple[float, ...]],
    ref: Tuple[float, ...],
    n_samples: int = 50_000,
    rng_seed: int = 0,
) -> float:
    """
    Monte-Carlo hypervolume estimate for d ≥ 3 objectives.
    Samples uniformly in [0, ref[i]] and estimates the dominated volume fraction.
    """
    rng = np.random.default_rng(rng_seed)
    ref_arr = np.array(ref, dtype=float)
    front   = np.array(pareto_costs, dtype=float)
    samples = rng.uniform(low=0.0, high=ref_arr, size=(n_samples, len(ref)))

    # A sample is dominated if at least one front point dominates it
    dominated = np.zeros(n_samples, dtype=bool)
    for fp in front:
        dominated |= np.all(samples >= fp, axis=1)

    box_vol = float(np.prod(ref_arr))
    return box_vol * dominated.mean()
