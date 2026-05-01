"""
algorithms/pareto_dijkstra.py — Multi-criteria Dijkstra for Pareto frontier generation.

Algorithm
---------
Standard Dijkstra is extended to maintain a set of non-dominated *label vectors*
at each node rather than a single scalar distance.

  Open set: min-heap ordered by primary objective (time by default).
  Each element: (priority, Label).

  At each expansion:
    1. Pop label L from heap.
    2. For each outgoing edge (u → v, weight w):
         new_cost = L.cost + w
         new_label = Label(new_cost, v, prev=L, edge=e)
         Try to add new_label to frontier[v].
         If accepted (non-dominated and frontier not full) → push to heap.

  Termination:
    When the heap is empty. With max_labels=None, this is an exact
    label-setting search over the constructed graph.

Complexity
----------
  O(|E| · |F|_max · log(|open|))
  where |F|_max is the maximum frontier size per node.

Returns
-------
  List[ParetoPath] — the complete Pareto-optimal path set to destination.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional, Tuple

from core.graph import MultimodalGraph
from core.label import Label, LabelFrontier
from data.schema import EdgeWeight, ParetoPath
import config as cfg


class ParetoDijkstra:

    def __init__(self, graph: MultimodalGraph):
        self.graph = graph

    def run(
        self,
        origin:      str,
        destination: str,
        max_labels:  int | None = cfg.MAX_LABELS_PER_NODE,
    ) -> List[ParetoPath]:
        """
        Compute the Pareto-optimal frontier of paths from origin to destination.

        Parameters
        ----------
        origin, destination : stop_id strings
        max_labels          : optional per-node frontier cap; None is exact

        Returns
        -------
        List of ParetoPath objects (non-dominated w.r.t. all objectives).
        Empty list if destination is unreachable.
        """
        if not self.graph.has_node(origin):
            raise ValueError(f"Origin node '{origin}' not in graph.")
        if not self.graph.has_node(destination):
            raise ValueError(f"Destination node '{destination}' not in graph.")

        lower_bounds = self._objective_lower_bounds(destination)

        # Per-node Pareto frontier
        frontiers: Dict[str, LabelFrontier] = {
            nid: LabelFrontier(max_labels=max_labels) for nid in self.graph.stops
        }

        # Initial label at origin
        start = Label(cost=EdgeWeight.zero(), node=origin, prev=None, edge=None)
        frontiers[origin].try_add(start)

        # Min-heap: (priority, counter, Label)
        # Counter breaks ties to avoid comparing Label objects directly
        counter = 0
        heap = [(start.priority(), counter, start)]

        while heap:
            _, _, current = heapq.heappop(heap)

            node_frontier = frontiers.get(current.node)
            if node_frontier and any(lb.dominates(current) for lb in node_frontier.labels()):
                continue

            dest_frontier = frontiers[destination]
            if dest_frontier and self._dominated_by_destination(
                current.cost.as_tuple(), current.node, dest_frontier, lower_bounds
            ):
                continue

            for edge in self.graph.neighbors_with_edges(current.node):
                new_cost  = current.cost + edge.weight
                if dest_frontier and self._dominated_by_destination(
                    new_cost.as_tuple(), edge.dst, dest_frontier, lower_bounds
                ):
                    continue
                new_label = Label(
                    cost=new_cost,
                    node=edge.dst,
                    prev=current,
                    edge=edge,
                )

                frontier = frontiers.get(edge.dst)
                if frontier is None:
                    frontier = LabelFrontier(max_labels=max_labels)
                    frontiers[edge.dst] = frontier

                if frontier.try_add(new_label):
                    counter += 1
                    heapq.heappush(heap, (new_label.priority(), counter, new_label))

        # ── Collect results at destination
        dest_frontier = frontiers.get(destination)
        if dest_frontier is None or len(dest_frontier) == 0:
            return []

        paths: List[ParetoPath] = []
        for label in dest_frontier.labels():
            nodes, edges = label.reconstruct_path()
            paths.append(ParetoPath(
                nodes=nodes,
                edges=edges,
                total_weight=label.cost,
            ))

        return paths

    def _objective_lower_bounds(self, destination: str) -> Dict[str, Tuple[float, ...]]:
        """Single-objective reverse Dijkstra lower bounds for each objective."""
        objective_dists = []
        for obj_idx in range(cfg.N_OBJECTIVES):
            dist = {destination: 0.0}
            heap = [(0.0, destination)]
            while heap:
                d, node = heapq.heappop(heap)
                if d > dist.get(node, float("inf")):
                    continue
                for pred, _, _, data in self.graph.G.in_edges(node, keys=True, data=True):
                    edge = data.get("edge")
                    if edge is None:
                        continue
                    nd = d + edge.weight.as_tuple()[obj_idx]
                    if nd < dist.get(pred, float("inf")):
                        dist[pred] = nd
                        heapq.heappush(heap, (nd, pred))
            objective_dists.append(dist)

        bounds = {}
        for node in self.graph.stops:
            bounds[node] = tuple(
                objective_dists[obj_idx].get(node, float("inf"))
                for obj_idx in range(cfg.N_OBJECTIVES)
            )
        return bounds

    def _dominated_by_destination(
        self,
        cost_vec: Tuple[float, ...],
        node: str,
        dest_frontier: LabelFrontier,
        lower_bounds: Dict[str, Tuple[float, ...]],
    ) -> bool:
        lb_vec = lower_bounds.get(node)
        if lb_vec is None or any(v == float("inf") for v in lb_vec):
            return False
        optimistic = tuple(cost_vec[i] + lb_vec[i] for i in range(cfg.N_OBJECTIVES))
        optimistic_label = Label(cost=EdgeWeight(*optimistic), node=node)
        return any(dest_label.dominates(optimistic_label) for dest_label in dest_frontier.labels())

    def run_with_stats(
        self,
        origin:      str,
        destination: str,
    ) -> Dict:
        """Run and return paths plus algorithm statistics."""
        import time
        t0 = time.perf_counter()
        paths = self.run(origin, destination)
        elapsed = time.perf_counter() - t0

        return {
            "paths":        paths,
            "frontier_size": len(paths),
            "runtime_s":    elapsed,
            "origin":       origin,
            "destination":  destination,
        }
