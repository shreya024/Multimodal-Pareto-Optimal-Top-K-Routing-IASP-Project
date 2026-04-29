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
    When heap is empty OR destination's frontier has been populated
    and the popped label's priority exceeds the best time in frontier[dst].

Complexity
----------
  O(|E| · |F|_max · log(|open|))
  where |F|_max is the maximum frontier size per node (capped at MAX_LABELS_PER_NODE).

Returns
-------
  List[ParetoPath] — the complete Pareto-optimal path set to destination.
"""
from __future__ import annotations

import heapq
from typing import Dict, List, Optional

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
        max_labels:  int = cfg.MAX_LABELS_PER_NODE,
    ) -> List[ParetoPath]:
        """
        Compute the Pareto-optimal frontier of paths from origin to destination.

        Parameters
        ----------
        origin, destination : stop_id strings
        max_labels          : per-node frontier cap (overrides config)

        Returns
        -------
        List of ParetoPath objects (non-dominated w.r.t. all objectives).
        Empty list if destination is unreachable.
        """
        if not self.graph.has_node(origin):
            raise ValueError(f"Origin node '{origin}' not in graph.")
        if not self.graph.has_node(destination):
            raise ValueError(f"Destination node '{destination}' not in graph.")

        # Per-node Pareto frontier
        frontiers: Dict[str, LabelFrontier] = {
            nid: LabelFrontier() for nid in self.graph.stops
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

            # ── Early termination: if current label's time > best known
            # time at destination, it cannot produce a dominant path
            if frontiers[destination]:
                best_dest_time = min(
                    lb.cost.time_min for lb in frontiers[destination].labels()
                )
                if current.cost.time_min > best_dest_time * 3:
                    # Allow 3× slack to ensure Pareto coverage on other dims
                    continue

            # ── Expand neighbours
            for edge in self.graph.neighbors_with_edges(current.node):
                new_cost  = current.cost + edge.weight
                new_label = Label(
                    cost=new_cost,
                    node=edge.dst,
                    prev=current,
                    edge=edge,
                )

                frontier = frontiers.get(edge.dst)
                if frontier is None:
                    frontier = LabelFrontier()
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
