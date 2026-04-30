"""
data/loader.py — Ingests and cleans the BMTC Kaggle dataset.

The BMTC Kaggle dataset (shivamishra2112) is NOT standard GTFS.
Supported layouts are auto-detected:

  Layout A — route-level with From/To stop terminals
    e.g. Route_No, Route_Name, From_Stop, To_Stop, ...

  Layout B — full stop sequence per route
    e.g. Route_No, Stop_Name, Stop_Sequence, Latitude, Longitude, ...

  Layout C — Standard GTFS (stops.csv / routes.csv / trips.csv / stop_times.csv)

Falls back to a Bengaluru-realistic synthetic dataset when no CSVs are found.
"""
from __future__ import annotations

import math
import hashlib
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.schema import GraphEdge, Route, Stop, TransportMode
import config as cfg

RAW_DIR = Path(__file__).parent / "raw"


# ─── Haversine ────────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _stable_id(name: str) -> str:
    return hashlib.md5(name.strip().lower().encode()).hexdigest()[:10]


def _infer_coords(name: str) -> Tuple[float, float]:
    """Deterministic pseudo-random Bengaluru coordinate from stop name."""
    seed = int(hashlib.md5(name.strip().lower().encode()).hexdigest(), 16) % (2 ** 31)
    r = random.Random(seed)
    return r.uniform(12.85, 13.08), r.uniform(77.45, 77.75)


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        c.strip().lower().replace(" ", "_").replace("/", "_").replace("-", "_")
        for c in df.columns
    ]
    return df


def _find_col(df: pd.DataFrame, *candidates) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:           # partial match
        for col in cols:
            if c in col:
                return col
    return None


# ─── Loader ───────────────────────────────────────────────────────────────────

class BMTCLoader:

    def __init__(self, raw_dir: Path = RAW_DIR):
        self.raw_dir = raw_dir
        self.stops:  Dict[str, Stop]  = {}
        self.routes: Dict[str, Route] = {}
        self._loaded = False

    def load(self) -> "BMTCLoader":
        csv_files = list(self.raw_dir.rglob("*.csv")) if self.raw_dir.exists() else []
        if not csv_files:
            warnings.warn(
                f"[loader] data/raw/ not found or empty — using synthetic dataset.\n"
                f"  Download: https://www.kaggle.com/datasets/shivamishra2112/bmtc-bus-transportation-dataset\n"
                f"  Extract all CSVs into: {self.raw_dir}",
                stacklevel=2,
            )
            self._generate_synthetic()
        else:
            try:
                self._load_bmtc(csv_files)
            except Exception as exc:
                warnings.warn(f"[loader] BMTC load failed ({exc}) — using synthetic.", stacklevel=2)
                import traceback; traceback.print_exc()
                self.stops = {}; self.routes = {}
                self._generate_synthetic()
        self._loaded = True
        return self

    def get_stops(self)  -> Dict[str, Stop]:  self._ensure_loaded(); return self.stops
    def get_routes(self) -> Dict[str, Route]: self._ensure_loaded(); return self.routes

    # ─── Auto-detect and dispatch ─────────────────────────────────────────────

    def _load_bmtc(self, csv_files: List[Path]):
        print(f"[loader] Found CSVs: {[f.name for f in csv_files]}")
        by_name = {f.name.lower(): f for f in csv_files}

        # Standard GTFS takes priority — detected by presence of stops.csv + trips.csv
        if "stops.csv" in by_name and "trips.csv" in by_name:
            try:
                self._load_gtfs(by_name)
                if self.stops and self.routes:
                    print(f"[loader] GTFS OK: {len(self.stops)} stops, {len(self.routes)} routes")
                    return
            except Exception as e:
                print(f"[loader] GTFS attempt failed: {e}")
                self.stops = {}; self.routes = {}

        # Fallback: try each CSV as a route-stop table
        for fpath in csv_files:
            try:
                df = _norm_cols(pd.read_csv(fpath, low_memory=False))
                seq_col = _find_col(df, "stop_sequence", "sequence", "seq",
                                    "stop_order", "order", "serial")
                if seq_col is not None:
                    self._load_layout_b(df)
                else:
                    self._load_layout_a(df)

                if self.stops and self.routes:
                    layout = "B (stop-sequence)" if seq_col else "A (from/to)"
                    print(f"[loader] Layout {layout} ({fpath.name}): "
                        f"{len(self.stops)} stops, {len(self.routes)} routes")
                    return
            except Exception as e:
                print(f"[loader] Skipping {fpath.name}: {e}")
                self.stops = {}; self.routes = {}

        raise RuntimeError("No CSV yielded a valid stops+routes dataset.")
    # ─── Layout A: From / To terminals ───────────────────────────────────────

    def _load_layout_a(self, df: pd.DataFrame):
        route_col = _find_col(df, "route_no", "route_id", "routeno", "route_number", "route")
        name_col  = _find_col(df, "route_name", "routename", "name")
        from_col  = _find_col(df, "from_stop", "from", "origin", "source",
                               "start", "from_station", "fromstop", "from_stop_name")
        to_col    = _find_col(df, "to_stop", "to", "destination", "dest",
                               "end", "to_station", "tostop", "to_stop_name")
        stops_col = _find_col(df, "stops", "stop_list", "all_stops", "halt")

        if from_col is None or to_col is None:
            raise ValueError(f"No From/To columns. Available: {list(df.columns)}")

        seen: Dict[str, Route] = {}
        for idx, row in df.iterrows():
            rid       = str(row[route_col]).strip() if route_col else f"R{idx}"
            rname     = str(row[name_col]).strip()  if name_col  else rid
            from_name = str(row[from_col]).strip()
            to_name   = str(row[to_col]).strip()

            if from_name in ("nan", "") or to_name in ("nan", ""):
                continue

            stop_names: List[str] = []
            if stops_col and not pd.isna(row.get(stops_col)):
                raw = str(row[stops_col])
                stop_names = [s.strip() for s in raw.replace(";", ",").split(",") if s.strip()]
            if not stop_names:
                stop_names = [from_name, to_name]

            route_stops = [self._get_or_create_stop(n) for n in stop_names]
            if len(route_stops) < 2:
                continue
            if rid not in seen or len(route_stops) > len(seen[rid].stops):
                seen[rid] = Route(route_id=rid, route_name=rname,
                                  mode=TransportMode.BUS, stops=route_stops, headway_min=10.0)
        self.routes = seen

    # ─── Layout B: full stop sequence ────────────────────────────────────────

    def _load_layout_b(self, df: pd.DataFrame):
        route_col = _find_col(df, "route_no", "route_id", "routeno", "route_number", "route")
        stop_col  = _find_col(df, "stop_name", "stopname", "stop", "station", "halt", "name")
        seq_col   = _find_col(df, "stop_sequence", "sequence", "seq",
                               "stop_order", "order", "serial")
        lat_col   = _find_col(df, "latitude", "lat", "stop_lat")
        lon_col   = _find_col(df, "longitude", "lon", "lng", "stop_lon")

        if route_col is None:
            raise ValueError(f"No route column. Cols: {list(df.columns)}")
        if stop_col is None:
            raise ValueError(f"No stop-name column. Cols: {list(df.columns)}")

        # Force numeric for coordinate and sequence columns
        for col in (lat_col, lon_col, seq_col):
            if col:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for route_id, grp in df.groupby(route_col):
            rid = str(route_id).strip()
            if seq_col:
                grp = grp.sort_values(seq_col)

            route_stops = []
            for _, row in grp.iterrows():
                sname = str(row[stop_col]).strip()
                if not sname or sname == "nan":
                    continue

                # Prefer real coordinates if valid for Karnataka
                lat = float(row[lat_col]) if lat_col and not pd.isna(row.get(lat_col)) else None
                lon = float(row[lon_col]) if lon_col and not pd.isna(row.get(lon_col)) else None
                if lat and lon and not (12.0 <= lat <= 13.5 and 76.5 <= lon <= 78.5):
                    lat = lon = None

                sid = _stable_id(sname)
                if sid not in self.stops:
                    if lat is None or lon is None:
                        lat, lon = _infer_coords(sname)
                    self.stops[sid] = Stop(stop_id=sid, name=sname, lat=lat, lon=lon)
                route_stops.append(self.stops[sid])

            if len(route_stops) >= 2:
                self.routes[rid] = Route(route_id=rid, route_name=rid,
                                         mode=TransportMode.BUS, stops=route_stops,
                                         headway_min=10.0)

    # ─── GTFS ─────────────────────────────────────────────────────────────────

    def _load_gtfs(self, by_name: Dict[str, Path]):
        stops_df = _norm_cols(pd.read_csv(by_name["stops.csv"], low_memory=False))
        sid_col  = _find_col(stops_df, "stop_id", "id")
        sname_col= _find_col(stops_df, "stop_name", "name")
        slat_col = _find_col(stops_df, "stop_lat", "latitude", "lat")
        slon_col = _find_col(stops_df, "stop_lon", "longitude", "lon", "lng")

        for col in (slat_col, slon_col):
            if col:
                stops_df[col] = pd.to_numeric(stops_df[col], errors="coerce")

        for _, row in stops_df.iterrows():
            sid  = str(row[sid_col]).strip()
            name = str(row[sname_col]).strip() if sname_col else sid
            lat  = float(row[slat_col]) if slat_col and not pd.isna(row[slat_col]) else None
            lon  = float(row[slon_col]) if slon_col and not pd.isna(row[slon_col]) else None
            if lat is None or lon is None:
                lat, lon = _infer_coords(name)
            self.stops[sid] = Stop(stop_id=sid, name=name, lat=lat, lon=lon)

        # Build routes from stop_times if present
        if "stop_times.csv" in by_name and "trips.csv" in by_name:
            st_df = _norm_cols(pd.read_csv(by_name["stop_times.csv"], low_memory=False))
            tr_df = _norm_cols(pd.read_csv(by_name["trips.csv"],      low_memory=False))

            trip_c  = _find_col(tr_df, "trip_id", "trip")
            route_c = _find_col(tr_df, "route_id", "route")
            st_tc   = _find_col(st_df, "trip_id",  "trip")
            st_sc   = _find_col(st_df, "stop_id",  "stop")
            st_seq  = _find_col(st_df, "stop_sequence", "seq", "sequence")

            if None not in (trip_c, route_c, st_tc, st_sc, st_seq):
                t2r = dict(zip(tr_df[trip_c].astype(str), tr_df[route_c].astype(str)))
                st_df[st_seq] = pd.to_numeric(st_df[st_seq], errors="coerce")
                st_df[st_sc]  = st_df[st_sc].astype(str)

                route_seqs: Dict[str, List] = {}
                for tid, grp in st_df.groupby(st_tc):
                    rid = t2r.get(str(tid), "UNK")
                    seq = grp.sort_values(st_seq)[st_sc].tolist()
                    if rid not in route_seqs or len(seq) > len(route_seqs[rid]):
                        route_seqs[rid] = seq

                for rid, sids in route_seqs.items():
                    sl = [self.stops[s] for s in sids if s in self.stops]
                    if len(sl) >= 2:
                        self.routes[rid] = Route(route_id=rid, route_name=rid,
                                                 mode=TransportMode.BUS, stops=sl)

    # ─── Shared helper ────────────────────────────────────────────────────────

    def _get_or_create_stop(self, name: str) -> Stop:
        sid = _stable_id(name)
        if sid not in self.stops:
            lat, lon = _infer_coords(name)
            self.stops[sid] = Stop(stop_id=sid, name=name, lat=lat, lon=lon)
        return self.stops[sid]

    # ─── Synthetic fallback ───────────────────────────────────────────────────

    def _generate_synthetic(self):
        areas = [
            "Majestic", "Whitefield", "Koramangala", "Indiranagar", "Jayanagar",
            "BTM Layout", "HSR Layout", "Marathahalli", "Bellandur", "Sarjapur",
            "Electronic City", "Bannerghatta", "JP Nagar", "Banashankari",
            "Yeshwanthpur", "Rajajinagar", "Malleswaram", "Hebbal", "Yelahanka",
            "Devanahalli", "KR Puram", "Hoodi", "Brookefield", "Varthur",
            "Bommanahalli", "Silk Board", "KR Market", "Richmond Town",
            "Shivajinagar", "MG Road", "Brigade Road", "Commercial Street",
            "Ulsoor", "HAL", "Domlur", "Ejipura", "Vivek Nagar",
            "CV Raman Nagar", "Kadugodi", "Kundalahalli", "Nagarbhavi",
            "Vijayanagar", "Basaveshwara Nagar", "Peenya", "Jalahalli",
            "Kengeri", "Palace Ground", "Seshadripuram", "Sadashivanagar", "Vasanth Nagar",
        ]
        area_coords = {
            "Majestic":        (12.9767, 77.5713), "Whitefield":    (12.9698, 77.7499),
            "Koramangala":     (12.9352, 77.6244), "Indiranagar":   (12.9784, 77.6408),
            "Jayanagar":       (12.9299, 77.5826), "BTM Layout":    (12.9166, 77.6101),
            "HSR Layout":      (12.9116, 77.6389), "Marathahalli":  (12.9591, 77.6974),
            "Electronic City": (12.8399, 77.6770), "Silk Board":    (12.9172, 77.6235),
            "MG Road":         (12.9756, 77.6072), "Yeshwanthpur":  (13.0231, 77.5390),
            "Hebbal":          (13.0352, 77.5912), "KR Puram":      (13.0054, 77.6956),
            "Banashankari":    (12.9258, 77.5470), "Malleswaram":   (13.0035, 77.5660),
            "Rajajinagar":     (12.9919, 77.5529), "JP Nagar":      (12.9077, 77.5857),
            "Bellandur":       (12.9256, 77.6767), "KR Market":     (12.9700, 77.5750),
            "Shivajinagar":    (12.9847, 77.6006), "HAL":           (12.9591, 77.6665),
            "Palace Ground":   (13.0057, 77.5909), "Peenya":        (13.0283, 77.5204),
            "Kengeri":         (12.9133, 77.4830), "Nagarbhavi":    (12.9580, 77.5028),
            "Vijayanagar":     (12.9716, 77.5263), "Yelahanka":     (13.1005, 77.5963),
        }

        rng = random.Random(cfg.SYNTHETIC_SEED)
        np.random.seed(cfg.SYNTHETIC_SEED)

        stop_ids = []
        for i in range(cfg.SYNTHETIC_N_STOPS):
            base = areas[i % len(areas)]
            suffix = f" Gate {i // len(areas) + 1}" if i >= len(areas) else ""
            name = base + suffix
            if base in area_coords:
                blat, blon = area_coords[base]
                lat = blat + rng.uniform(-0.006, 0.006)
                lon = blon + rng.uniform(-0.006, 0.006)
            else:
                lat, lon = rng.uniform(12.85, 13.08), rng.uniform(77.45, 77.75)
            sid = str(i)
            self.stops[sid] = Stop(stop_id=sid, name=name, lat=lat, lon=lon)
            stop_ids.append(sid)

        # ── Backbone routes: guarantee full city connectivity ──────────────
        # Overlapping corridor routes ensure every stop is reachable from
        # every other via at most 2 transfers.
        n_stops = len(stop_ids)
        step    = n_stops // 3
        overlap = max(5, step // 5)
        by_lon  = sorted(stop_ids, key=lambda s: self.stops[s].lon)
        by_lat  = sorted(stop_ids, key=lambda s: self.stops[s].lat)
        backbones = [
            by_lon[: step + overlap],
            by_lon[step - overlap: 2 * step + overlap],
            by_lon[2 * step - overlap:],
            by_lat[: step + overlap],
            by_lat[step - overlap: 2 * step + overlap],
            by_lat[2 * step - overlap:],
        ]
        for b_idx, b_stops in enumerate(backbones):
            sl = [self.stops[s] for s in b_stops]
            rid = f"BACKBONE_{b_idx}"
            self.routes[rid] = Route(
                route_id=rid,
                route_name=f"Corridor {b_idx} ({sl[0].name} – {sl[-1].name})",
                mode=TransportMode.BUS, stops=sl, headway_min=8.0,
            )

        # ── Random feeder routes ──────────────────────────────────────────────
        for r in range(cfg.SYNTHETIC_N_ROUTES):
            n = rng.randint(5, 15)
            chosen = rng.sample(stop_ids, min(n, len(stop_ids)))
            chosen.sort(key=lambda s: (self.stops[s].lat, self.stops[s].lon))
            s_list = [self.stops[s] for s in chosen]
            rid = f"R{r:03d}"
            self.routes[rid] = Route(
                route_id=rid,
                route_name=f"Route {r} ({s_list[0].name} – {s_list[-1].name})",
                mode=TransportMode.BUS, stops=s_list,
                headway_min=rng.uniform(5, 20),
            )

        print(f"[loader] Synthetic dataset: {len(self.stops)} stops, {len(self.routes)} routes")
        print(f"[loader] Named stops include: {[s.name for s in list(self.stops.values())[:6]]}")

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call .load() first.")
