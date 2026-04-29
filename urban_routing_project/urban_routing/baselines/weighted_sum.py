"""
baselines/weighted_sum.py — Weighted-sum scalarization baseline.

Converts the multi-objective problem into a single-objective one by computing
a weighted linear combination of normalized objectives, then runs standard
Dijkstra on that scalar weight.

Limitations (discussed in evaluation)
--------------------------------------
  • Collapses the entire Pareto structure to a single point.
  • Cannot discover non-convex Pareto front regions.
  • Sensitive to weight specification; small changes can flip the optimal route.
  • Returns exactly one route, offering no alternatives.
"""
from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.graph import MultimodalGraph
from data.schema import EdgeWeight, ParetoPath
import config as cfg


class WeightedSumRouter:
    """
    Single-objective Dijkstra using a weighted-sum scalarization of objectives.
    """

    def __init__(
        self,
        graph:   MultimodalGraph,
        weights: Tuple[float, ...] = cfg.DEFAULT_WEIGHTS,
    ):
        assert len(weights) == cfg.N_OBJECTIVES, "weights must have N_OBJECTIVES elements"
        assert abs(sum(weights) - 1.0) < 1e-6,  "weights must sum to 1.0"
        self.graph   = graph
        self.weights = np.array(weights, dtype=float)

        # Precompute normalization constants from edge weights in graph
        # (Approximate: use heuristic max values for Bengaluru scale)
        self._norm_maxs = np.array([
            120.0,      # time: 2 hours max
            200.0,      # cost: ₹200 max
            5.0,        # transfers: 5 max
            5000.0,     # walking: 5km max
            10_000.0,   # CO2: 10kg max
        ], dtype=float)

    def _scalar_cost(self, ew: EdgeWeight) -> float:
        vec = np.array(ew.as_tuple(), dtype=float)
        normalized = vec / self._norm_maxs
        return float(np.dot(self.weights, normalized))

    def run(self, origin: str, destination: str) -> Optional[ParetoPath]:
        """
        Run weighted-sum Dijkstra. Returns the single best path or None.
        """
        if not self.graph.has_node(origin) or not self.graph.has_node(destination):
            return None

        dist: Dict[str, float] = {origin: 0.0}
        cost: Dict[str, EdgeWeight] = {origin: EdgeWeight.zero()}
        prev: Dict[str, Optional[str]] = {origin: None}
        prev_edge: Dict[str, Optional[object]] = {origin: None}

        counter = 0
        heap    = [(0.0, counter, origin)]

        while heap:
            d, _, u = heapq.heappop(heap)
            if u == destination:
                break
            if d > dist.get(u, float("inf")):
                continue

            for edge in self.graph.neighbors_with_edges(u):
                edge_scalar = self._scalar_cost(edge.weight)
                new_d       = dist[u] + edge_scalar
                new_cost    = cost[u] + edge.weight

                if new_d < dist.get(edge.dst, float("inf")):
                    dist[edge.dst]      = new_d
                    cost[edge.dst]      = new_cost
                    prev[edge.dst]      = u
                    prev_edge[edge.dst] = edge
                    counter += 1
                    heapq.heappush(heap, (new_d, counter, edge.dst))

        if destination not in prev:
            return None

        # Reconstruct
        nodes, edges = [], []
        cur = destination
        while cur is not None:
            nodes.append(cur)
            e = prev_edge.get(cur)
            if e is not None:
                edges.append(e)
            cur = prev.get(cur)
        nodes.reverse()
        edges.reverse()

        return ParetoPath(nodes=nodes, edges=edges, total_weight=cost[destination])

    def run_with_stats(self, origin: str, destination: str) -> Dict:
        t0   = time.perf_counter()
        path = self.run(origin, destination)
        elapsed = time.perf_counter() - t0
        return {
            "paths":       [path] if path else [],
            "frontier_size": 1 if path else 0,
            "runtime_s":   elapsed,
            "method":      "weighted_sum",
            "weights":     list(self.weights),
        }
