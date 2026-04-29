"""
core/graph.py — Builds the multi-layer directed multigraph from BMTC data.

Layers
------
  BUS   — one directed edge per consecutive stop pair in every route
  WALK  — short walking edges between geographically close stops (≤ MAX_WALK_TRANSFER_M)
  (METRO layer uses the same structure; included if metro routes exist in data)

Intermodal transfer nodes
-------------------------
When a stop appears in two different routes (or modes), a transfer edge is
inserted with the appropriate penalty (time + +1 transfer counter).

Graph representation: NetworkX DiGraph with parallel edges stored as lists
on each (u, v) pair via the 'edges' attribute of a MultiDiGraph.
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from data.loader import haversine_m
from data.schema import GraphEdge, Route, Stop, TransportMode
from core.edge_weights import mode_edge_weight, transfer_penalty, walk_edge_weight
import config as cfg


class MultimodalGraph:
    """
    Fused directed multigraph over all transit layers.

    Attributes
    ----------
    G           : nx.MultiDiGraph  (nodes = stop_id strings)
    stops       : Dict[str, Stop]
    routes      : Dict[str, Route]
    edge_data   : Dict[(u,v,key), GraphEdge]  — rich edge objects
    """

    def __init__(self):
        self.G: nx.MultiDiGraph = nx.MultiDiGraph()
        self.stops:  Dict[str, Stop]  = {}
        self.routes: Dict[str, Route] = {}
        self.edge_data: Dict[Tuple, GraphEdge] = {}

    # ── Builder ───────────────────────────────────────────────────────────────

    def build(self, stops: Dict[str, Stop], routes: Dict[str, Route]) -> "MultimodalGraph":
        self.stops  = stops
        self.routes = routes

        # Add all stop nodes
        for sid, stop in stops.items():
            self.G.add_node(sid, stop=stop)

        # Add transit edges (bus / metro)
        for route in routes.values():
            self._add_route_edges(route)

        # Add walking transfer edges
        self._add_walk_edges()

        print(
            f"[graph] Built: {self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges"
        )
        return self

    # ── Internal builders ──────────────────────────────────────────────────────

    def _add_route_edges(self, route: Route):
        stops = route.stops
        for i in range(len(stops) - 1):
            src = stops[i]
            dst = stops[i + 1]
            dist = haversine_m(src.lat, src.lon, dst.lat, dst.lon)

            weight = mode_edge_weight(route.mode, dist, route.headway_min)

            edge = GraphEdge(
                src=src.stop_id,
                dst=dst.stop_id,
                route_id=route.route_id,
                mode=route.mode,
                weight=weight,
                distance_m=dist,
            )

            key = self.G.add_edge(src.stop_id, dst.stop_id, edge=edge, weight=weight.time_min)
            self.edge_data[(src.stop_id, dst.stop_id, key)] = edge

    def _add_walk_edges(self):
        """
        Connect nearby stops (across any mode) with walking edges.
        Uses a spatial sweep to avoid O(n²) haversine calls where n is large.
        """
        stop_list = list(self.stops.values())
        n = len(stop_list)

        # Sort by latitude for a 1-D sweep
        stop_list.sort(key=lambda s: s.lat)

        # Approx lat degrees for MAX_WALK_TRANSFER_M
        lat_deg_threshold = cfg.MAX_WALK_TRANSFER_M / 111_000.0

        for i, src in enumerate(stop_list):
            for j in range(i + 1, n):
                dst = stop_list[j]
                if dst.lat - src.lat > lat_deg_threshold:
                    break  # all further stops are too far north
                dist = haversine_m(src.lat, src.lon, dst.lat, dst.lon)
                if dist > cfg.MAX_WALK_TRANSFER_M:
                    continue

                weight = walk_edge_weight(dist)
                # Transfer penalty only if stops serve different routes/modes
                src_routes = self._stop_routes(src.stop_id)
                dst_routes = self._stop_routes(dst.stop_id)
                if src_routes.isdisjoint(dst_routes):
                    transfer_w = transfer_penalty(TransportMode.WALK, TransportMode.WALK)
                    weight = weight + transfer_w

                for (s_id, d_id) in [(src.stop_id, dst.stop_id), (dst.stop_id, src.stop_id)]:
                    dist_dir = dist
                    w = walk_edge_weight(dist_dir)
                    if src_routes.isdisjoint(dst_routes):
                        w = w + transfer_w
                    edge = GraphEdge(
                        src=s_id, dst=d_id,
                        route_id=None,
                        mode=TransportMode.WALK,
                        weight=w,
                        distance_m=dist_dir,
                    )
                    key = self.G.add_edge(s_id, d_id, edge=edge, weight=w.time_min)
                    self.edge_data[(s_id, d_id, key)] = edge

    def _stop_routes(self, stop_id: str) -> Set[str]:
        """Return the set of route_ids that serve this stop."""
        return {
            r.route_id
            for r in self.routes.values()
            if any(s.stop_id == stop_id for s in r.stops)
        }

    # ── Accessors ─────────────────────────────────────────────────────────────

    def neighbors_with_edges(self, node: str) -> List[GraphEdge]:
        """Return all outgoing GraphEdge objects from node."""
        result = []
        for _, dst, key, data in self.G.out_edges(node, keys=True, data=True):
            edge = data.get("edge")
            if edge is not None:
                result.append(edge)
        return result

    def has_node(self, node: str) -> bool:
        return self.G.has_node(node)

    def stop_name(self, stop_id: str) -> str:
        return self.stops.get(stop_id, Stop(stop_id, stop_id, 0, 0)).name

    def node_count(self) -> int:
        return self.G.number_of_nodes()

    def edge_count(self) -> int:
        return self.G.number_of_edges()
