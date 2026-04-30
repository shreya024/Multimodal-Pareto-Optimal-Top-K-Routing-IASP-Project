"""
core/graph.py — Multi-layer directed multigraph for multimodal routing.

Layers injected (in build order):
  1. Bus edges      — from BMTC routes (speed-estimated weights)
  2. Metro edges    — from GTFS segment_edges (real travel times)
  3. Transfer edges — Bus↔Metro, Metro↔Metro interchange (from fuser)
  4. Walk edges     — short straight-line walks between nearby stops of
                      DIFFERENT routes (fallback when OSM not available)

All edges are stored as GraphEdge objects in self.edge_data for path
reconstruction. The underlying nx.MultiDiGraph uses edge attribute 'edge'
to hold the GraphEdge.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Set, Tuple

import networkx as nx

from data.loader import haversine_m
from data.schema import GraphEdge, Route, Stop, TransportMode
from core.edge_weights import mode_edge_weight, transfer_penalty, walk_edge_weight
import config as cfg


class MultimodalGraph:

    def __init__(self):
        self.G:         nx.MultiDiGraph         = nx.MultiDiGraph()
        self.stops:     Dict[str, Stop]          = {}
        self.routes:    Dict[str, Route]         = {}
        self.edge_data: Dict[Tuple, GraphEdge]   = {}

    # ── Main builder ──────────────────────────────────────────────────────────

    def build(
        self,
        stops:          Dict[str, Stop],
        routes:         Dict[str, Route],
        extra_edges:    List[GraphEdge] = None,
        transfer_edges: List[GraphEdge] = None,
    ) -> "MultimodalGraph":
        """
        Parameters
        ----------
        stops           : unified stop dict (bus + metro + walk nodes)
        routes          : unified route dict (bus + metro lines)
        extra_edges     : pre-built edges (metro GTFS segments, interchange)
        transfer_edges  : intermodal connector edges (bus↔metro walks)
        """
        self.stops  = stops
        self.routes = routes

        # Add all nodes
        for sid, stop in stops.items():
            self.G.add_node(sid, stop=stop)

        # Layer 1: Bus route edges (speed-estimated weights)
        n_bus = 0
        for route in routes.values():
            if route.mode != TransportMode.BUS:
                continue
            for i in range(len(route.stops) - 1):
                src = route.stops[i]
                dst = route.stops[i + 1]
                dist = haversine_m(src.lat, src.lon, dst.lat, dst.lon)
                weight = mode_edge_weight(route.mode, dist, route.headway_min)
                edge = GraphEdge(
                    src=src.stop_id, dst=dst.stop_id,
                    route_id=route.route_id, mode=route.mode,
                    weight=weight, distance_m=dist,
                )
                key = self.G.add_edge(src.stop_id, dst.stop_id,
                                      edge=edge, weight=weight.time_min)
                self.edge_data[(src.stop_id, dst.stop_id, key)] = edge
                n_bus += 1

        # Layer 2: Pre-built edges (metro GTFS segments + interchange)
        n_extra = 0
        for edge in (extra_edges or []):
            if not self.G.has_node(edge.src) or not self.G.has_node(edge.dst):
                continue
            key = self.G.add_edge(edge.src, edge.dst,
                                  edge=edge, weight=edge.weight.time_min)
            self.edge_data[(edge.src, edge.dst, key)] = edge
            n_extra += 1

        # Layer 3: Intermodal transfer edges (bus↔metro)
        n_xfer = 0
        for edge in (transfer_edges or []):
            if not self.G.has_node(edge.src) or not self.G.has_node(edge.dst):
                continue
            key = self.G.add_edge(edge.src, edge.dst,
                                  edge=edge, weight=edge.weight.time_min)
            self.edge_data[(edge.src, edge.dst, key)] = edge
            n_xfer += 1

        # Layer 4: Straight-line walk edges between nearby stops of different routes
        n_walk = self._add_walk_edges()

        print(
            f"[graph] Built: {self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges "
            f"(bus={n_bus}, metro={n_extra}, xfer={n_xfer}, walk={n_walk})"
        )
        return self

    # ── Walk edge layer ───────────────────────────────────────────────────────

    def _add_walk_edges(self) -> int:
        """
        Connect nearby stops (≤ MAX_WALK_TRANSFER_M) that belong to
        different routes/modes with straight-line walk edges.
        Uses a latitude-sorted sweep to avoid O(n²) full comparison.
        """
        stop_list = list(self.stops.values())
        stop_list.sort(key=lambda s: s.lat)
        n = len(stop_list)
        lat_thresh = cfg.MAX_WALK_TRANSFER_M / 111_000.0

        stop_routes = self._build_stop_route_index()
        n_walk = 0

        for i, src in enumerate(stop_list):
            for j in range(i + 1, n):
                dst = stop_list[j]
                if dst.lat - src.lat > lat_thresh:
                    break
                dist = haversine_m(src.lat, src.lon, dst.lat, dst.lon)
                if dist > cfg.MAX_WALK_TRANSFER_M:
                    continue

                walk_w = walk_edge_weight(dist)

                # Add transfer penalty if different route/mode
                src_routes = stop_routes.get(src.stop_id, set())
                dst_routes = stop_routes.get(dst.stop_id, set())
                if src_routes.isdisjoint(dst_routes):
                    xfer_w = transfer_penalty(src.mode, dst.mode)
                    walk_w = walk_w + xfer_w

                for s, d in [(src.stop_id, dst.stop_id), (dst.stop_id, src.stop_id)]:
                    edge = GraphEdge(
                        src=s, dst=d, route_id=None,
                        mode=TransportMode.WALK,
                        weight=walk_w, distance_m=dist,
                    )
                    key = self.G.add_edge(s, d, edge=edge, weight=walk_w.time_min)
                    self.edge_data[(s, d, key)] = edge
                n_walk += 2

        return n_walk

    def _build_stop_route_index(self) -> Dict[str, Set[str]]:
        idx: Dict[str, Set[str]] = defaultdict(set)
        for route in self.routes.values():
            for stop in route.stops:
                idx[stop.stop_id].add(route.route_id)
        return idx

    # ── Accessors ─────────────────────────────────────────────────────────────

    def neighbors_with_edges(self, node: str) -> List[GraphEdge]:
        result = []
        for _, dst, key, data in self.G.out_edges(node, keys=True, data=True):
            edge = data.get("edge")
            if edge is not None:
                result.append(edge)
        return result

    def has_node(self, node: str) -> bool:
        return self.G.has_node(node)

    def stop_name(self, stop_id: str) -> str:
        s = self.stops.get(stop_id)
        return s.name if s else stop_id

    def stop_mode(self, stop_id: str) -> str:
        s = self.stops.get(stop_id)
        return s.mode.value if s else "?"

    def node_count(self) -> int:
        return self.G.number_of_nodes()

    def edge_count(self) -> int:
        return self.G.number_of_edges()

    def find_stop_by_name(self, query: str) -> List[str]:
        """Fuzzy name search — returns list of matching stop_ids."""
        q = query.strip().lower()
        return [
            sid for sid, stop in self.stops.items()
            if q in stop.name.lower()
        ]
