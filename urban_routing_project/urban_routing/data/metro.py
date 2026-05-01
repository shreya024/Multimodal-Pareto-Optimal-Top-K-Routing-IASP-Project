"""
data/metro.py — Namma Metro (BMRCL) loader using the real GTFS feed.

GTFS files (data/raw/metro/*.txt):
    stops.txt       — 68 stations with real lat/lon
    routes.txt      — 4 routes (Purple E→W, Purple W→E, Green N→S, Green S→N)
    trips.txt       — 524 trips (one per scheduled departure)
    stop_times.txt  — 18 078 records with real per-segment travel times

Key extractions
---------------
  • Per-segment travel times  → used as edge weight time_min (not speed-estimated)
  • Real headways computed from trip departure times  → wait time = headway/2
  • BMRCL slab fare table  → cost_inr per OD pair
  • Majestic appears as stop 122 (Purple) and 153 (Green) — both kept as
    separate nodes; a zero-cost intra-station transfer edge connects them.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import pandas as pd
import numpy as np

from data.schema import Route, Stop, TransportMode, GraphEdge, EdgeWeight
from data.loader import haversine_m

METRO_RAW_DIR = Path(__file__).parent / "raw" / "metro"

# ── BMRCL slab fare (2024) ────────────────────────────────────────────────────
_FARE_SLABS: List[Tuple[float, float]] = [
    (2.0,  10.0),
    (4.0,  20.0),
    (6.0,  30.0),
    (8.0,  40.0),
    (12.0, 50.0),
    (18.0, 60.0),
    (1e9,  60.0),
]

def metro_slab_fare(distance_km: float) -> float:
    for max_km, fare in _FARE_SLABS:
        if distance_km <= max_km:
            return fare
    return 60.0

# CO2 g/passenger-km for metro (electrical, Bengaluru grid factor)
METRO_CO2_G_PKM = 41.0


def _to_min(t: str) -> float:
    """Parse HH:MM:SS (may exceed 24h) → float minutes."""
    h, m, s = t.strip().split(":")
    return int(h) * 60 + int(m) + int(s) / 60


# ── Named prefix so metro stop IDs never clash with bus stop IDs ──────────────
def _msid(raw_id) -> str:
    return f"M_{raw_id}"


class MetroLoader:
    """
    Parses the BMRCL GTFS feed and exposes:
      stops        : Dict[str, Stop]
      routes       : Dict[str, Route]
      segment_edges: List[GraphEdge]  — one per consecutive stop pair, with
                                        real travel times and slab fares
      interchange_edges: List[GraphEdge]  — Majestic intra-station transfer
    """

    def __init__(self):
        self.stops:            Dict[str, Stop]     = {}
        self.routes:           Dict[str, Route]    = {}
        self.segment_edges:    List[GraphEdge]     = []
        self.interchange_edges: List[GraphEdge]    = []
        self._headways:        Dict[str, float]    = {}   # route_id → avg headway

    def load(self) -> "MetroLoader":
        txt_files = list(METRO_RAW_DIR.glob("*.txt")) if METRO_RAW_DIR.exists() else []
        if not txt_files:
            warnings.warn(
                "[metro] No GTFS files found in data/raw/metro/ — "
                "metro layer will be empty. Place GTFS .txt files there.",
                stacklevel=2,
            )
            return self

        self._parse_gtfs()
        print(
            f"[metro] Loaded {len(self.stops)} stations, "
            f"{len(self.routes)} lines, "
            f"{len(self.segment_edges)} segment edges"
        )
        return self

    # ── GTFS parse ────────────────────────────────────────────────────────────

    def _parse_gtfs(self):
        stops_df      = pd.read_csv(METRO_RAW_DIR / "stops.txt")
        routes_df     = pd.read_csv(METRO_RAW_DIR / "routes.txt")
        trips_df      = pd.read_csv(METRO_RAW_DIR / "trips.txt")
        stop_times_df = pd.read_csv(METRO_RAW_DIR / "stop_times.txt")

        # ── Stops ──────────────────────────────────────────────────────────
        for _, row in stops_df.iterrows():
            sid  = _msid(row["stop_id"])
            self.stops[sid] = Stop(
                stop_id = sid,
                name    = str(row["stop_name"]).strip(),
                lat     = float(row["stop_lat"]),
                lon     = float(row["stop_lon"]),
                mode    = TransportMode.METRO,
            )

        # ── Headways per route from real departure times ───────────────────
        # Use the minimum stop_sequence departure across all trips for each route
        min_seq = stop_times_df.groupby("trip_id")["stop_sequence"].min().reset_index()
        first   = stop_times_df.merge(min_seq, on=["trip_id", "stop_sequence"])
        t2r     = dict(zip(trips_df["trip_id"], trips_df["route_id"]))
        first["route_id"]  = first["trip_id"].map(t2r)
        first["dep_min"]   = first["departure_time"].apply(_to_min)

        for route_id, grp in first.groupby("route_id"):
            sorted_deps = grp["dep_min"].sort_values().tolist()
            if len(sorted_deps) > 1:
                diffs = [sorted_deps[i+1] - sorted_deps[i]
                         for i in range(len(sorted_deps)-1)
                         if sorted_deps[i+1] - sorted_deps[i] > 0]
                self._headways[str(route_id)] = float(np.median(diffs)) if diffs else 10.0
            else:
                self._headways[str(route_id)] = 10.0

        # ── One canonical stop sequence per route (longest trip) ──────────
        stop_times_df["arr_min"] = stop_times_df["arrival_time"].apply(_to_min)

        route_canonical: Dict[str, List[int]] = {}   # route_id → [stop_id, ...]
        route_trip_used: Dict[str, int]        = {}

        for trip_id, grp in stop_times_df.groupby("trip_id"):
            rid = str(t2r.get(trip_id, ""))
            if not rid:
                continue
            seq = grp.sort_values("stop_sequence")["stop_id"].tolist()
            if rid not in route_canonical or len(seq) > len(route_canonical[rid]):
                route_canonical[rid] = seq
                route_trip_used[rid] = trip_id

        # ── Per-segment travel times from the canonical trip ───────────────
        # Build a (trip_id, stop_id) → arr_min lookup
        arr_lookup: Dict[Tuple[int,int], float] = {
            (int(row["trip_id"]), int(row["stop_id"])): row["arr_min"]
            for _, row in stop_times_df.iterrows()
        }

        for route_id_str, stop_id_list in route_canonical.items():
            trip_id     = route_trip_used[route_id_str]
            headway     = self._headways.get(route_id_str, 10.0)
            wait_min    = headway / 2.0

            route_stop_objs: List[Stop] = []
            for raw_sid in stop_id_list:
                sid = _msid(raw_sid)
                if sid in self.stops:
                    route_stop_objs.append(self.stops[sid])

            if len(route_stop_objs) < 2:
                continue

            # Route metadata
            route_row = routes_df[routes_df["route_id"] == int(route_id_str)]
            rname = route_row["route_long_name"].values[0] if len(route_row) else route_id_str

            self.routes[route_id_str] = Route(
                route_id    = route_id_str,
                route_name  = rname,
                mode        = TransportMode.METRO,
                stops       = route_stop_objs,
                headway_min = headway,
            )

            # ── Segment edges with real travel times ───────────────────────
            cumulative_dist_km = 0.0
            for i in range(len(route_stop_objs) - 1):
                src = route_stop_objs[i]
                dst = route_stop_objs[i + 1]

                # Real travel time from GTFS
                t_src = arr_lookup.get((trip_id, int(stop_id_list[i])),   None)
                t_dst = arr_lookup.get((trip_id, int(stop_id_list[i+1])), None)
                if t_src is not None and t_dst is not None and t_dst > t_src:
                    seg_time_min = t_dst - t_src
                else:
                    # Fallback: speed-estimated
                    dist_m = haversine_m(src.lat, src.lon, dst.lat, dst.lon)
                    seg_time_min = (dist_m / (35_000 / 60))   # 35 km/h

                dist_m = haversine_m(src.lat, src.lon, dst.lat, dst.lon)
                cumulative_dist_km += dist_m / 1000.0

                # Segment-proportional fare keeps mid-line boardings from becoming free.
                fare = 2.5 * (dist_m / 1000.0)
                co2  = METRO_CO2_G_PKM * (dist_m / 1000.0)

                # Only first boarding incurs wait; model as half-headway on first seg
                time_total = seg_time_min + (wait_min if i == 0 else 0.0)

                weight = EdgeWeight(
                    time_min  = time_total,
                    cost_inr  = fare,
                    transfers = 0.0,
                    walking_m = 0.0,
                    co2_g     = co2,
                )

                self.segment_edges.append(GraphEdge(
                    src        = src.stop_id,
                    dst        = dst.stop_id,
                    route_id   = route_id_str,
                    mode       = TransportMode.METRO,
                    weight     = weight,
                    distance_m = dist_m,
                ))

        # ── Majestic intra-station interchange (122 ↔ 153) ────────────────
        self._build_majestic_interchange()

    def _build_majestic_interchange(self):
        """
        stop 122 = Majestic on Purple Line
        stop 153 = Majestic on Green Line
        Same physical station — concourse walk ~80m, no extra fare,
        but costs 1 transfer unit and ~3 min concourse time.
        """
        sid_purple = _msid(122)
        sid_green  = _msid(153)
        if sid_purple not in self.stops or sid_green not in self.stops:
            return

        concourse_m   = 80.0
        concourse_min = 3.0    # published BMRCL interchange time

        weight = EdgeWeight(
            time_min  = concourse_min,
            cost_inr  = 0.0,
            transfers = 1.0,
            walking_m = concourse_m,
            co2_g     = 0.0,
        )
        for src, dst in [(sid_purple, sid_green), (sid_green, sid_purple)]:
            self.interchange_edges.append(GraphEdge(
                src=src, dst=dst, route_id="MAJESTIC_XFER",
                mode=TransportMode.TRANSFER,
                weight=weight, distance_m=concourse_m,
            ))

    def get_all_extra_edges(self) -> List[GraphEdge]:
        """Returns interchange edges to be injected into the graph."""
        return self.interchange_edges
