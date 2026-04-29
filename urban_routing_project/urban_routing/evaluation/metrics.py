"""
evaluation/metrics.py — Quality metrics for comparing routing methods.

Metrics
-------
  hypervolume       — Volume of objective space dominated by the Pareto front (higher = better)
  spread (Δ)        — Uniformity of distribution along the Pareto front
  generational_dist — Average distance of found front to true/reference front
  diversity_score   — Average pairwise Jaccard distance among selected paths
  objective_range   — Per-objective range covered (max - min across paths)
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from algorithms.dominance import hypervolume_mc, hypervolume_2d
from data.schema import ParetoPath
from selection.diversity_selector import jaccard_dissimilarity
import config as cfg


def compute_hypervolume(
    paths: List[ParetoPath],
    ref_point: Tuple[float, ...] = None,
) -> float:
    """
    Hypervolume indicator (Monte-Carlo for d ≥ 3).
    A larger hypervolume indicates a better-spread Pareto front.
    ref_point defaults to 1.1 × worst objective values in the set.
    """
    if not paths:
        return 0.0

    vecs = np.array([p.total_weight.as_tuple() for p in paths])
    if ref_point is None:
        ref_point = tuple(vecs.max(axis=0) * 1.1)

    return hypervolume_mc(
        [tuple(v) for v in vecs],
        ref_point,
        n_samples=30_000,
    )


def spread_metric(paths: List[ParetoPath], obj_idx: int = cfg.OBJ_TIME) -> float:
    """
    Δ-spread: measures extent and uniformity of distribution along one objective.
    Returns the range (max - min) normalized by the number of paths.
    """
    if len(paths) < 2:
        return 0.0
    vals = sorted(p.total_weight.as_tuple()[obj_idx] for p in paths)
    total_range = vals[-1] - vals[0]
    gaps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    mean_gap = total_range / (len(paths) - 1)
    if mean_gap == 0:
        return 0.0
    # Coefficient of variation of gaps (lower = more uniform)
    cv = np.std(gaps) / mean_gap
    return float(1.0 / (1.0 + cv))          # 1 = perfectly uniform


def diversity_score(paths: List[ParetoPath]) -> float:
    """
    Mean pairwise Jaccard dissimilarity among selected paths.
    Range [0, 1]; higher = more structurally diverse.
    """
    if len(paths) < 2:
        return 0.0
    scores = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            scores.append(jaccard_dissimilarity(paths[i], paths[j]))
    return float(np.mean(scores))


def objective_range(paths: List[ParetoPath]) -> Dict[str, float]:
    """
    Range covered per objective (max - min) across all selected paths.
    """
    if not paths:
        return {name: 0.0 for name in cfg.OBJECTIVE_NAMES}
    vecs = np.array([p.total_weight.as_tuple() for p in paths])
    ranges = vecs.max(axis=0) - vecs.min(axis=0)
    return {cfg.OBJECTIVE_NAMES[i]: float(ranges[i]) for i in range(cfg.N_OBJECTIVES)}


def summarize_paths(paths: List[ParetoPath]) -> Dict:
    """Return a summary dict of all metrics for a set of selected paths."""
    return {
        "n_paths":         len(paths),
        "hypervolume":     compute_hypervolume(paths),
        "spread_time":     spread_metric(paths, cfg.OBJ_TIME),
        "spread_cost":     spread_metric(paths, cfg.OBJ_COST),
        "diversity_score": diversity_score(paths),
        "objective_range": objective_range(paths),
    }
