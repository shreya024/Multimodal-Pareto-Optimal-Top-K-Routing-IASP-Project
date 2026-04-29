"""
baselines/lexicographic.py — Lexicographic ordering baseline.

Objectives are prioritised in a strict order (e.g. minimise time first,
then cost, then transfers, …). The algorithm runs sequential Dijkstra passes:

  Pass 1: Find the optimal value for objective 0 (e.g. minimum time).
  Pass 2: Among all paths that achieve the pass-1 optimum (within tolerance),
          find the one minimising objective 1.
  Pass k: Continue until all objectives are lexicographically resolved.

Limitations
-----------
  • Extremely brittle: changing the priority order yields a completely different path.
  • Collapses all objectives into a single path; no alternatives presented.
  • The tie-breaking tolerance (EPS) is heuristic.
"""
from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from core.graph import MultimodalGraph
from data.schema import EdgeWeight, ParetoPath
import config as cfg

EPS = 1e-6   # tolerance for "achieving the same value" in tie-breaking passes


class LexicographicRouter:
    """
    Lexicographic Dijkstra: optimises objectives one at a time in priority order.
    """

    def __init__(
        self,
        graph:     MultimodalGraph,
        lex_order: List[int] = None,
    ):
        self.graph     = graph
        self.lex_order = lex_order if lex_order is not None else cfg.DEFAULT_LEX_ORDER

    # ── Single-objective Dijkstra ─────────────────────────────────────────────

    def _dijkstra_on_objective(
        self,
        origin:      str,
        destination: str,
        obj_idx:     int,
        upper_bounds: Optional[Dict[int, float]] = None,
    ) -> Optional[Dict[str, object]]:
        """
        Run Dijkstra minimising objective obj_idx.
        If upper_bounds is provided, prune any path that violates
        upper_bounds[dim] for any dimension dim.
        Returns dict with dist/cost/prev/prev_edge at all nodes.
        """
        dist:      Dict[str, float]      = {origin: 0.0}
        cost:      Dict[str, EdgeWeight] = {origin: EdgeWeight.zero()}
        prev:      Dict[str, Optional[str]] = {origin: None}
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
                new_cost_ew = cost[u] + edge.weight
                new_vec     = new_cost_ew.as_tuple()

                # Prune if violates any upper bound
                if upper_bounds:
                    violates = any(
                        new_vec[dim] > upper_bounds[dim] + EPS
                        for dim in upper_bounds
                        if dim != obj_idx
                    )
                    if violates:
                        continue

                edge_val = new_vec[obj_idx]
                new_d    = dist.get(u, 0.0) + edge.weight.as_tuple()[obj_idx]

                if new_d < dist.get(edge.dst, float("inf")):
                    dist[edge.dst]      = new_d
                    cost[edge.dst]      = new_cost_ew
                    prev[edge.dst]      = u
                    prev_edge[edge.dst] = edge
                    counter += 1
                    heapq.heappush(heap, (new_d, counter, edge.dst))

        return {
            "dist": dist, "cost": cost, "prev": prev, "prev_edge": prev_edge
        }

    # ── Lexicographic solve ───────────────────────────────────────────────────

    def run(self, origin: str, destination: str) -> Optional[ParetoPath]:
        if not self.graph.has_node(origin) or not self.graph.has_node(destination):
            return None

        upper_bounds: Dict[int, float] = {}

        final_result = None
        for obj_idx in self.lex_order:
            result = self._dijkstra_on_objective(
                origin, destination, obj_idx, upper_bounds if upper_bounds else None
            )
            if destination not in result["dist"]:
                break
            # Fix the achieved value as an upper bound for subsequent passes
            optimal_val = result["cost"][destination].as_tuple()[obj_idx]
            upper_bounds[obj_idx] = optimal_val
            final_result = result

        if final_result is None or destination not in final_result["prev"]:
            return None

        # Reconstruct path
        nodes, edges = [], []
        cur = destination
        while cur is not None:
            nodes.append(cur)
            e = final_result["prev_edge"].get(cur)
            if e is not None:
                edges.append(e)
            cur = final_result["prev"].get(cur)
        nodes.reverse()
        edges.reverse()

        return ParetoPath(
            nodes=nodes,
            edges=edges,
            total_weight=final_result["cost"][destination],
        )

    def run_with_stats(self, origin: str, destination: str) -> Dict:
        t0   = time.perf_counter()
        path = self.run(origin, destination)
        elapsed = time.perf_counter() - t0
        return {
            "paths":         [path] if path else [],
            "frontier_size": 1 if path else 0,
            "runtime_s":     elapsed,
            "method":        "lexicographic",
            "lex_order":     self.lex_order,
        }
