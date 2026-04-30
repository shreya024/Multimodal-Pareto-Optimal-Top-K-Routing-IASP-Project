"""
data/osm_walk.py — OpenStreetMap pedestrian network for Bengaluru.

Uses osmnx to fetch the full walking graph from OSM, then:
  1. Simplifies to a manageable node set by sampling nodes near
     transit stops (bus + metro) within MAX_WALK_TRANSFER_M.
  2. Exposes these as Stop objects (mode=WALK) and walk-edge GraphEdges.

This replaces the earlier synthetic "straight-line walk edges" with real
street-network walking paths that respect actual pedestrian infrastructure.

Caching
-------
The OSM graph is cached to data/raw/osm_walk_cache.graphml so subsequent
runs don't re-download. Delete the file to refresh.

Network size
------------
Full Bengaluru walking graph ≈ 200k nodes. We prune to nodes within
WALK_BUFFER_M of any transit stop, yielding ~2k–5k walk nodes —
manageable for the label-setting Dijkstra.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np

from data.loader import haversine_m
from data.schema import GraphEdge, Route, Stop, TransportMode
import config as cfg

CACHE_PATH = Path(__file__).parent / "raw" / "osm_walk_cache.pkl"

# Only keep OSM walk nodes within this distance of a transit stop
WALK_BUFFER_M    = cfg.MAX_WALK_TRANSFER_M   # default 600m
# Max walking edge length to include
MAX_EDGE_M       = 800.0
# Bengaluru bounding box for OSM query
BLR_BBOX         = (12.83, 77.43, 13.18, 77.78)   # (south, west, north, east)


def _osm_node_id(nid: int) -> str:
    return f"OSM_{nid}"


def _stable_id(name: str) -> str:
    return hashlib.md5(name.strip().lower().encode()).hexdigest()[:10]


class OSMWalkLayer:
    """
    Fetches and wraps the OSM pedestrian graph for Bengaluru.
    After load(), provides:
      - walk_stops : Dict[str, Stop]   — OSM nodes near transit as Stop objects
      - walk_edges : List[GraphEdge]   — street-network walking edges
    """

    def __init__(self):
        self.walk_stops: Dict[str, Stop]    = {}
        self.walk_edges: List[GraphEdge]    = []
        self._osm_graph: Optional[nx.MultiDiGraph] = None

    def load(
        self,
        transit_stops: Dict[str, Stop],
        use_cache: bool = True,
    ) -> "OSMWalkLayer":
        """
        Parameters
        ----------
        transit_stops : all bus + metro stops (used to filter relevant OSM nodes)
        use_cache     : load from disk cache if available
        """
        self._osm_graph = self._fetch_osm(use_cache)

        if self._osm_graph is None:
            print("[osm] OSM fetch failed — using straight-line walk edges only")
            return self

        self._build_walk_layer(transit_stops)
        print(
            f"[osm] Walk layer: {len(self.walk_stops)} nodes, "
            f"{len(self.walk_edges)} edges"
        )
        return self

    # ── OSM fetch ─────────────────────────────────────────────────────────────

    def _fetch_osm(self, use_cache: bool) -> Optional[nx.MultiDiGraph]:
        if use_cache and CACHE_PATH.exists():
            try:
                print("[osm] Loading walk graph from cache...")
                with open(CACHE_PATH, "rb") as f:
                    G = pickle.load(f)
                print(f"[osm] Cache loaded: {G.number_of_nodes()} nodes")
                return G
            except Exception as e:
                print(f"[osm] Cache load failed ({e}), re-fetching...")

        try:
            import osmnx as ox
            print("[osm] Fetching Bengaluru walk graph from OSM (this may take ~60s)...")
            ox.settings.log_console = False

            G = ox.graph_from_bbox(
                bbox=BLR_BBOX,
                network_type="walk",
                simplify=True,
                retain_all=False,
            )
            G = ox.convert.to_digraph(G, weight="length")

            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(G, f)
            print(f"[osm] Fetched & cached: {G.number_of_nodes()} nodes")
            return G

        except Exception as e:
            print(f"[osm] OSM fetch failed: {e}")
            return None

    # ── Walk layer construction ───────────────────────────────────────────────

    def _build_walk_layer(self, transit_stops: Dict[str, Stop]):
        """
        Filter OSM nodes to those within WALK_BUFFER_M of any transit stop,
        then build GraphEdge objects for each OSM street segment.
        """
        G = self._osm_graph

        # Build array of transit stop coordinates for fast spatial query
        t_stops  = list(transit_stops.values())
        t_lats   = np.array([s.lat for s in t_stops])
        t_lons   = np.array([s.lon for s in t_stops])

        # For each OSM node, check if it's close enough to any transit stop
        relevant_nodes: Set[int] = set()

        print("[osm] Filtering walk nodes near transit stops...")
        node_data = list(G.nodes(data=True))
        n_total   = len(node_data)

        for i, (nid, data) in enumerate(node_data):
            if i % 10_000 == 0:
                print(f"[osm]   {i}/{n_total} nodes checked...", end="\r")
            lat = data.get("y", 0)
            lon = data.get("x", 0)

            # Quick bounding box pre-filter
            lat_diff = np.abs(t_lats - lat)
            lon_diff = np.abs(t_lons - lon)
            candidates = np.where(
                (lat_diff < WALK_BUFFER_M / 111_000) &
                (lon_diff < WALK_BUFFER_M / 85_000)
            )[0]

            for ci in candidates:
                d = haversine_m(lat, lon, t_lats[ci], t_lons[ci])
                if d <= WALK_BUFFER_M:
                    relevant_nodes.add(nid)
                    break

        print(f"\n[osm] {len(relevant_nodes)} relevant walk nodes found")

        # Create Stop objects for relevant OSM nodes
        for nid in relevant_nodes:
            data = G.nodes[nid]
            lat  = data.get("y", 0)
            lon  = data.get("x", 0)
            name = data.get("name") or f"Walk Node {nid}"
            sid  = _osm_node_id(nid)
            self.walk_stops[sid] = Stop(
                stop_id=sid, name=name, lat=lat, lon=lon,
                mode=TransportMode.WALK,
            )

        # Create GraphEdge objects for OSM edges between relevant nodes
        for u, v, data in G.edges(data=True):
            if u not in relevant_nodes or v not in relevant_nodes:
                continue
            length_m = data.get("length", 0)
            if length_m <= 0 or length_m > MAX_EDGE_M:
                continue

            from core.edge_weights import walk_edge_weight
            weight = walk_edge_weight(length_m)
            self.walk_edges.append(GraphEdge(
                src=_osm_node_id(u),
                dst=_osm_node_id(v),
                route_id=None,
                mode=TransportMode.WALK,
                weight=weight,
                distance_m=length_m,
            ))

    def nearest_walk_node(self, lat: float, lon: float) -> Optional[str]:
        """Return the nearest OSM walk node stop_id to given coordinates."""
        if not self.walk_stops:
            return None
        best_id, best_d = None, float("inf")
        for sid, stop in self.walk_stops.items():
            d = haversine_m(lat, lon, stop.lat, stop.lon)
            if d < best_d:
                best_d = d
                best_id = sid
        return best_id if best_d <= WALK_BUFFER_M else None
