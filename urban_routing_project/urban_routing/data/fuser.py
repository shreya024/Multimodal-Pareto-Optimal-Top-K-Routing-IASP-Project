"""
data/fuser.py — Fuses BMTC bus + Namma Metro (real GTFS) into one multimodal universe.

Layers
------
  1. Bus   — BMTC synthetic / real CSV (stop IDs: plain integers as strings)
  2. Metro — BMRCL GTFS real data      (stop IDs: "M_<gtfs_id>")

Intermodal transfer edges
-------------------------
  Bus ↔ Metro  : for every (bus_stop, metro_station) pair within
                 BUS_METRO_TRANSFER_RADIUS_M, create a bidirectional
                 walking transfer edge with:
                   weight = walk_edge_weight(distance) + transfer_penalty()
                 so the router pays time + 1 transfer for each modal switch.

  Metro ↔ Metro  : Majestic intra-station interchange (injected by MetroLoader).

OSM walk
--------
  If use_osm=True (default False — requires internet) the OSM pedestrian
  graph is fetched and nodes near transit stops are added as a third layer.
  Set use_osm=False for offline operation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

from data.loader import BMTCLoader, haversine_m
from data.metro import MetroLoader
from data.schema import GraphEdge, Route, Stop, TransportMode, EdgeWeight
from core.edge_weights import walk_edge_weight, transfer_penalty
import config as cfg

# Max distance for a bus↔metro transfer walk link
BUS_METRO_TRANSFER_RADIUS_M = 500.0


class MultimodalFuser:
    """
    Loads all transit layers and fuses them into:
      all_stops          : Dict[str, Stop]
      all_routes         : Dict[str, Route]
      transfer_edges     : List[GraphEdge]   — intermodal connectors
      extra_edges        : List[GraphEdge]   — metro segment + interchange edges
    """

    def __init__(self, raw_dir=None, use_osm: bool = False, osm_cache: bool = True):
        self.use_osm   = use_osm
        self.osm_cache = osm_cache
        self._raw_dir  = raw_dir

        self.all_stops:      Dict[str, Stop]   = {}
        self.all_routes:     Dict[str, Route]  = {}
        self.transfer_edges: List[GraphEdge]   = []
        self.extra_edges:    List[GraphEdge]   = []   # metro segments + interchange

        self.bus_loader:   Optional[BMTCLoader]  = None
        self.metro_loader: Optional[MetroLoader] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def load(self) -> "MultimodalFuser":
        # 1. Bus
        kwargs = {"raw_dir": Path(self._raw_dir)} if self._raw_dir else {}
        self.bus_loader = BMTCLoader(**kwargs)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.bus_loader.load()
            for warning in w:
                print(f"  [bus] {warning.message}")

        bus_stops  = self.bus_loader.get_stops()
        bus_routes = self.bus_loader.get_routes()
        self.all_stops.update(bus_stops)
        self.all_routes.update(bus_routes)
        print(f"[fuser] Bus   : {len(bus_stops):4d} stops, {len(bus_routes):3d} routes")

        # 2. Metro (real GTFS)
        self.metro_loader = MetroLoader()
        self.metro_loader.load()
        metro_stops  = self.metro_loader.stops
        metro_routes = self.metro_loader.routes

        self.all_stops.update(metro_stops)
        self.all_routes.update({f"metro:{rid}": route for rid, route in metro_routes.items()})

        # Metro segment edges and Majestic interchange
        self.extra_edges.extend(self.metro_loader.segment_edges)
        self.extra_edges.extend(self.metro_loader.interchange_edges)
        print(f"[fuser] Metro : {len(metro_stops):4d} stations, {len(metro_routes):3d} lines, "
              f"{len(self.metro_loader.segment_edges)} seg edges")

        # 3. Bus ↔ Metro transfer edges
        n_xfer = self._build_bus_metro_transfers(bus_stops, metro_stops)
        print(f"[fuser] Bus<->Metro transfer edges : {n_xfer}")

        # 4. Optional OSM walk layer
        if self.use_osm:
            self._add_osm_layer(bus_stops, metro_stops)

        print(f"[fuser] Total : {len(self.all_stops)} nodes, "
              f"{len(self.all_routes)} routes, "
              f"{len(self.transfer_edges)} transfer edges, "
              f"{len(self.extra_edges)} extra edges")
        return self

    # ── Transfer builders ─────────────────────────────────────────────────────

    def _build_bus_metro_transfers(
        self,
        bus_stops:   Dict[str, Stop],
        metro_stops: Dict[str, Stop],
    ) -> int:
        """
        Bidirectional walking transfer edges between every bus stop and
        every metro station within BUS_METRO_TRANSFER_RADIUS_M.
        Weight = walk_edge_weight(d) + transfer_penalty().
        """
        n = 0
        metro_list = list(metro_stops.items())
        bus_list   = list(bus_stops.items())

        for msid, mstop in metro_list:
            for bsid, bstop in bus_list:
                d = haversine_m(mstop.lat, mstop.lon, bstop.lat, bstop.lon)
                if d > BUS_METRO_TRANSFER_RADIUS_M:
                    continue

                walk_w  = walk_edge_weight(d)
                xfer_w  = transfer_penalty(TransportMode.BUS, TransportMode.METRO)
                combined = walk_w + xfer_w

                for src, dst in [(bsid, msid), (msid, bsid)]:
                    self.transfer_edges.append(GraphEdge(
                        src=src, dst=dst, route_id=None,
                        mode=TransportMode.TRANSFER,
                        weight=combined, distance_m=d,
                    ))
                n += 2
        return n

    def _add_osm_layer(self, bus_stops, metro_stops):
        try:
            from data.osm_walk import OSMWalkLayer
            osm = OSMWalkLayer()
            osm.load(self.all_stops, use_cache=self.osm_cache)
            for sid, stop in osm.walk_stops.items():
                self.all_stops[sid] = stop
            self.extra_edges.extend(osm.walk_edges)

            # Connect each transit stop to nearest walk node
            all_transit = list(bus_stops.items()) + list(metro_stops.items())
            from core.edge_weights import walk_edge_weight as wew
            from data.loader import haversine_m as hav
            n_conn = 0
            for tsid, tstop in all_transit:
                nearest = osm.nearest_walk_node(tstop.lat, tstop.lon)
                if nearest is None:
                    continue
                wstop = osm.walk_stops[nearest]
                d = hav(tstop.lat, tstop.lon, wstop.lat, wstop.lon)
                ww = wew(d)
                xw = transfer_penalty(TransportMode.WALK, TransportMode.BUS)
                for src, dst, w in [(tsid, nearest, ww), (nearest, tsid, ww + xw)]:
                    self.transfer_edges.append(GraphEdge(
                        src=src, dst=dst, route_id=None,
                        mode=TransportMode.WALK, weight=w, distance_m=d,
                    ))
                n_conn += 2
            print(f"[fuser] OSM walk: {len(osm.walk_stops)} nodes, "
                  f"{len(osm.walk_edges)} edges, {n_conn} transit connectors")
        except Exception as e:
            print(f"[fuser] OSM layer skipped: {e}")
