"""
data/loader.py — BMTC bus data loader with robust GTFS handling.
"""
from __future__ import annotations

import hashlib
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.schema import Route, Stop, TransportMode
import config as cfg

RAW_DIR = Path(__file__).parent / "raw"


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a  = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [
        c.strip().lower().replace(" ","_").replace("/","_").replace("-","_")
        for c in df.columns
    ]
    return df


def _find_col(df: pd.DataFrame, *candidates) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    for c in candidates:
        for col in cols:
            if c in col:
                return col
    return None


def _stable_id(name: str) -> str:
    return hashlib.md5(name.strip().lower().encode()).hexdigest()[:10]


def _infer_coords(name: str) -> Tuple[float, float]:
    seed = int(hashlib.md5(name.strip().lower().encode()).hexdigest(), 16) % (2**31)
    r = random.Random(seed)
    return r.uniform(12.85, 13.08), r.uniform(77.45, 77.75)


def _looks_like_coordinate(series: pd.Series) -> bool:
    """Return True if most non-null values look like lat/lon coordinates."""
    try:
        nums = pd.to_numeric(series.dropna().head(20), errors="coerce").dropna()
        if len(nums) == 0:
            return False
        return bool(((nums > 10) & (nums < 100)).mean() > 0.7)
    except Exception:
        return False


class BMTCLoader:

    def __init__(self, raw_dir: Path = RAW_DIR):
        self.raw_dir = raw_dir
        self.stops:  Dict[str, Stop]  = {}
        self.routes: Dict[str, Route] = {}
        self._loaded = False

    def load(self) -> "BMTCLoader":
        csv_files = [
            f for f in self.raw_dir.rglob("*.csv")
            if "metro" not in str(f).lower()
        ] if self.raw_dir.exists() else []

        if not csv_files:
            warnings.warn(
                f"[loader] data/raw/ empty — using synthetic dataset.\n"
                f"  Place BMTC CSVs in: {self.raw_dir}",
                stacklevel=2,
            )
            self._generate_synthetic()
        else:
            try:
                self._load_bmtc(csv_files)
            except Exception as exc:
                warnings.warn(
                    f"[loader] BMTC load failed ({exc}) — using synthetic.",
                    stacklevel=2,
                )
                self.stops = {}; self.routes = {}
                self._generate_synthetic()
        self._loaded = True
        return self

    def get_stops(self)  -> Dict[str, Stop]:  self._ensure_loaded(); return self.stops
    def get_routes(self) -> Dict[str, Route]: self._ensure_loaded(); return self.routes

    def _load_bmtc(self, csv_files: List[Path]):
        print(f"[loader] Found CSVs: {[f.name for f in csv_files]}")
        by_name = {f.name.lower(): f for f in csv_files}

        # GTFS path: needs stops + stop_times + trips
        if all(k in by_name for k in ("stops.csv","stop_times.csv","trips.csv")):
            try:
                self._load_gtfs(by_name)
                if self.stops and self.routes:
                    print(f"[loader] GTFS OK: {len(self.stops)} stops, {len(self.routes)} routes")
                    return
                print("[loader] GTFS produced no routes, trying layout fallback...")
                self.stops = {}; self.routes = {}
            except Exception as e:
                print(f"[loader] GTFS failed ({e}), trying layout fallback...")
                self.stops = {}; self.routes = {}

        # Layout A/B fallback
        for fpath in csv_files:
            try:
                df = _norm_cols(pd.read_csv(fpath, low_memory=False))
                seq_col = _find_col(df, "stop_sequence","sequence","seq",
                                    "stop_order","order","serial")
                if seq_col is not None:
                    self._load_layout_b(df)
                else:
                    self._load_layout_a(df)
                if self.stops and self.routes:
                    print(f"[loader] Layout {'B' if seq_col else 'A'} ({fpath.name}): "
                          f"{len(self.stops)} stops, {len(self.routes)} routes")
                    return
            except Exception as e:
                print(f"[loader] Skipping {fpath.name}: {e}")
            self.stops = {}; self.routes = {}

        raise RuntimeError("No CSV yielded a valid stops+routes dataset.")

    def _load_gtfs(self, by_name: Dict[str, Path]):
        stops_df = _norm_cols(pd.read_csv(by_name["stops.csv"], low_memory=False))
        st_df    = _norm_cols(pd.read_csv(by_name["stop_times.csv"], low_memory=False))
        tr_df    = _norm_cols(pd.read_csv(by_name["trips.csv"], low_memory=False))

        # ── Find the correct stop_id column ──────────────────────────────────
        # In the BMTC GTFS, stop_id in stops.csv sometimes contains lat values
        # while the real integer IDs are in stop_code.
        # Strategy: find which stops.csv column contains values that
        # actually match the stop_ids in stop_times.csv.
        st_sc = _find_col(st_df, "stop_id")
        if st_sc is None:
            raise ValueError("stop_times.csv has no stop_id column")

        sample_st_ids = set(st_df[st_sc].dropna().astype(str).str.strip().head(50).tolist())

        # Candidate ID columns in stops.csv
        candidate_id_cols = [c for c in stops_df.columns
                             if any(k in c for k in ("id","code","no","num"))]
        # Also always try "stop_id" and "stop_code" explicitly
        for c in ("stop_id","stop_code"):
            if c in stops_df.columns and c not in candidate_id_cols:
                candidate_id_cols.insert(0, c)

        best_id_col = None
        best_match  = 0
        for col in candidate_id_cols:
            col_vals = set(stops_df[col].dropna().astype(str).str.strip().head(200).tolist())
            overlap  = len(sample_st_ids & col_vals)
            if overlap > best_match:
                best_match  = overlap
                best_id_col = col

        if best_id_col is None or best_match == 0:
            # Last resort: try ALL string-ish columns
            for col in stops_df.columns:
                if _looks_like_coordinate(stops_df[col]):
                    continue
                col_vals = set(stops_df[col].dropna().astype(str).str.strip().head(200).tolist())
                overlap  = len(sample_st_ids & col_vals)
                if overlap > best_match:
                    best_match  = overlap
                    best_id_col = col

        if best_id_col is None or best_match == 0:
            raise RuntimeError(
                f"Cannot match stop_times stop_ids to any stops.csv column.\n"
                f"  stop_times sample IDs : {list(sample_st_ids)[:5]}\n"
                f"  stops.csv columns     : {list(stops_df.columns)}"
            )

        print(f"[loader] Using stops.csv column '{best_id_col}' as stop_id "
              f"({best_match} matches with stop_times)")

        # ── Parse stops ───────────────────────────────────────────────────────
        sname_c = _find_col(stops_df, "stop_name","name")
        slat_c  = _find_col(stops_df, "stop_lat","lat","latitude")
        slon_c  = _find_col(stops_df, "stop_lon","lon","longitude","lng")
        loc_c   = _find_col(stops_df, "location_type")

        if slat_c: stops_df[slat_c] = pd.to_numeric(stops_df[slat_c], errors="coerce")
        if slon_c: stops_df[slon_c] = pd.to_numeric(stops_df[slon_c], errors="coerce")

        # Filter to boarding stops only
        if loc_c:
            stops_df = stops_df[stops_df[loc_c].isna() | (stops_df[loc_c] == 0)]

        for _, row in stops_df.iterrows():
            sid  = str(row[best_id_col]).strip()
            name = str(row[sname_c]).strip() if sname_c else sid
            lat  = float(row[slat_c]) if slat_c and not pd.isna(row[slat_c]) else None
            lon  = float(row[slon_c]) if slon_c and not pd.isna(row[slon_c]) else None
            if lat is not None and lon is not None:
                if not (12.0 <= lat <= 13.5 and 76.5 <= lon <= 78.5):
                    lat = lon = None
            if lat is None or lon is None:
                lat, lon = _infer_coords(name)
            self.stops[sid] = Stop(stop_id=sid, name=name, lat=lat, lon=lon)

        print(f"[loader] GTFS stops loaded: {len(self.stops)}")

        # ── Build routes from trips + stop_times ──────────────────────────────
        trip_c  = _find_col(tr_df, "trip_id")
        route_c = _find_col(tr_df, "route_id")
        st_tc   = _find_col(st_df, "trip_id")
        st_seq  = _find_col(st_df, "stop_sequence")

        if None in (trip_c, route_c, st_tc, st_sc, st_seq):
            raise ValueError("Missing required GTFS columns in trips/stop_times")

        t2r = dict(zip(
            tr_df[trip_c].astype(str).str.strip(),
            tr_df[route_c].astype(str).str.strip()
        ))

        st_df[st_seq] = pd.to_numeric(st_df[st_seq], errors="coerce")
        st_df[st_sc]  = st_df[st_sc].astype(str).str.strip()
        st_df[st_tc]  = st_df[st_tc].astype(str).str.strip()

        print(f"[loader] Processing {len(st_df):,} stop_time records, "
              f"{st_df[st_tc].nunique()} trips, {len(t2r)} route mappings...")

        # One representative trip per route (first encountered)
        route_rep: Dict[str, str] = {}
        for tid, rid in t2r.items():
            if rid not in route_rep:
                route_rep[rid] = tid

        rep_trips = set(route_rep.values())
        st_rep    = st_df[st_df[st_tc].isin(rep_trips)]

        print(f"[loader] Building sequences for {len(route_rep)} routes...")
        built = 0
        for rid, tid in route_rep.items():
            grp  = st_rep[st_rep[st_tc] == tid].sort_values(st_seq)
            sids = grp[st_sc].tolist()
            sl   = [self.stops[s] for s in sids if s in self.stops]
            if len(sl) < 2:
                continue
            self.routes[rid] = Route(
                route_id=rid, route_name=rid,
                mode=TransportMode.BUS, stops=sl, headway_min=10.0,
            )
            built += 1

        if built == 0:
            sample_st  = st_df[st_sc].head(5).tolist()
            sample_sid = list(self.stops.keys())[:5]
            raise RuntimeError(
                f"0 routes built — stop_id mismatch?\n"
                f"  stop_times IDs : {sample_st}\n"
                f"  stops dict keys: {sample_sid}"
            )
        print(f"[loader] Built {built} routes from GTFS")

    def _load_layout_a(self, df: pd.DataFrame):
        route_c = _find_col(df,"route_no","route_id","routeno","route_number","route")
        name_c  = _find_col(df,"route_name","routename","name")
        from_c  = _find_col(df,"from_stop","from","origin","source","start","fromstop","from_stop_name")
        to_c    = _find_col(df,"to_stop","to","destination","dest","end","tostop","to_stop_name")
        stops_c = _find_col(df,"stops","stop_list","all_stops","halt")

        if from_c is None or to_c is None:
            raise ValueError(f"No From/To columns. Available: {list(df.columns)}")

        seen: Dict[str, Route] = {}
        for idx, row in df.iterrows():
            rid       = str(row[route_c]).strip() if route_c else f"R{idx}"
            rname     = str(row[name_c]).strip()  if name_c  else rid
            from_name = str(row[from_c]).strip()
            to_name   = str(row[to_c]).strip()
            if from_name in ("nan","") or to_name in ("nan",""):
                continue
            names: List[str] = []
            if stops_c and not pd.isna(row.get(stops_c, float("nan"))):
                names = [s.strip() for s in str(row[stops_c]).replace(";",",").split(",") if s.strip()]
            if not names:
                names = [from_name, to_name]
            sl = [self._get_or_create(n) for n in names]
            if len(sl) < 2:
                continue
            if rid not in seen or len(sl) > len(seen[rid].stops):
                seen[rid] = Route(route_id=rid, route_name=rname,
                                  mode=TransportMode.BUS, stops=sl, headway_min=10.0)
        self.routes = seen

    def _load_layout_b(self, df: pd.DataFrame):
        route_c = _find_col(df,"route_no","route_id","routeno","route_number","route")
        stop_c  = _find_col(df,"stop_name","stopname","stop","station","halt","name")
        seq_c   = _find_col(df,"stop_sequence","sequence","seq","stop_order","order","serial")
        lat_c   = _find_col(df,"latitude","lat","stop_lat")
        lon_c   = _find_col(df,"longitude","lon","lng","stop_lon")

        if route_c is None: raise ValueError(f"No route column. Cols: {list(df.columns)}")
        if stop_c  is None: raise ValueError(f"No stop-name column. Cols: {list(df.columns)}")

        for col in (lat_c, lon_c, seq_c):
            if col: df[col] = pd.to_numeric(df[col], errors="coerce")

        for route_id, grp in df.groupby(route_c):
            rid = str(route_id).strip()
            if seq_c: grp = grp.sort_values(seq_c)
            sl = []
            for _, row in grp.iterrows():
                sname = str(row[stop_c]).strip()
                if not sname or sname == "nan": continue
                sid = _stable_id(sname)
                if sid not in self.stops:
                    lat = float(row[lat_c]) if lat_c and not pd.isna(row.get(lat_c)) else None
                    lon = float(row[lon_c]) if lon_c and not pd.isna(row.get(lon_c)) else None
                    if lat and lon and not (12.0 <= lat <= 13.5 and 76.5 <= lon <= 78.5):
                        lat = lon = None
                    if lat is None or lon is None:
                        lat, lon = _infer_coords(sname)
                    self.stops[sid] = Stop(stop_id=sid, name=sname, lat=lat, lon=lon)
                sl.append(self.stops[sid])
            if len(sl) >= 2:
                self.routes[rid] = Route(route_id=rid, route_name=rid,
                                         mode=TransportMode.BUS, stops=sl, headway_min=10.0)

    def _get_or_create(self, name: str) -> Stop:
        sid = _stable_id(name)
        if sid not in self.stops:
            lat, lon = _infer_coords(name)
            self.stops[sid] = Stop(stop_id=sid, name=name, lat=lat, lon=lon)
        return self.stops[sid]

    def _generate_synthetic(self):
        areas = [
            "Majestic","Whitefield","Koramangala","Indiranagar","Jayanagar",
            "BTM Layout","HSR Layout","Marathahalli","Bellandur","Sarjapur",
            "Electronic City","Bannerghatta","JP Nagar","Banashankari",
            "Yeshwanthpur","Rajajinagar","Malleswaram","Hebbal","Yelahanka",
            "Devanahalli","KR Puram","Hoodi","Brookefield","Varthur",
            "Bommanahalli","Silk Board","KR Market","Richmond Town",
            "Shivajinagar","MG Road","Brigade Road","Commercial Street",
            "Ulsoor","HAL","Domlur","Ejipura","Vivek Nagar",
            "CV Raman Nagar","Kadugodi","Kundalahalli","Nagarbhavi",
            "Vijayanagar","Basaveshwara Nagar","Peenya","Jalahalli",
            "Kengeri","Palace Ground","Seshadripuram","Sadashivanagar","Vasanth Nagar",
        ]
        area_coords = {
            "Majestic":(12.9767,77.5713),"Whitefield":(12.9698,77.7499),
            "Koramangala":(12.9352,77.6244),"Indiranagar":(12.9784,77.6408),
            "Jayanagar":(12.9299,77.5826),"BTM Layout":(12.9166,77.6101),
            "HSR Layout":(12.9116,77.6389),"Marathahalli":(12.9591,77.6974),
            "Electronic City":(12.8399,77.6770),"Silk Board":(12.9172,77.6235),
            "MG Road":(12.9756,77.6072),"Yeshwanthpur":(13.0231,77.5390),
            "Hebbal":(13.0352,77.5912),"KR Puram":(13.0054,77.6956),
            "Banashankari":(12.9258,77.5470),"Malleswaram":(13.0035,77.5660),
            "Rajajinagar":(12.9919,77.5529),"JP Nagar":(12.9077,77.5857),
            "Bellandur":(12.9256,77.6767),"KR Market":(12.9700,77.5750),
            "Shivajinagar":(12.9847,77.6006),"HAL":(12.9591,77.6665),
            "Palace Ground":(13.0057,77.5909),"Peenya":(13.0283,77.5204),
            "Kengeri":(12.9133,77.4830),"Nagarbhavi":(12.9580,77.5028),
            "Vijayanagar":(12.9716,77.5263),"Yelahanka":(13.1005,77.5963),
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
                lat, lon = rng.uniform(12.85,13.08), rng.uniform(77.45,77.75)
            sid = str(i)
            self.stops[sid] = Stop(stop_id=sid, name=name, lat=lat, lon=lon)
            stop_ids.append(sid)

        # Backbone corridor routes
        n = len(stop_ids); step = n//3; ovlp = max(5, step//5)
        by_lon = sorted(stop_ids, key=lambda s: self.stops[s].lon)
        by_lat = sorted(stop_ids, key=lambda s: self.stops[s].lat)
        for b_idx, b_stops in enumerate([
            by_lon[:step+ovlp], by_lon[step-ovlp:2*step+ovlp], by_lon[2*step-ovlp:],
            by_lat[:step+ovlp], by_lat[step-ovlp:2*step+ovlp], by_lat[2*step-ovlp:],
        ]):
            sl = [self.stops[s] for s in b_stops]
            self.routes[f"BACKBONE_{b_idx}"] = Route(
                route_id=f"BACKBONE_{b_idx}",
                route_name=f"Corridor {b_idx} ({sl[0].name} – {sl[-1].name})",
                mode=TransportMode.BUS, stops=sl, headway_min=8.0)

        for r in range(cfg.SYNTHETIC_N_ROUTES):
            n_in = rng.randint(5,15)
            chosen = rng.sample(stop_ids, min(n_in, len(stop_ids)))
            chosen.sort(key=lambda s: (self.stops[s].lat, self.stops[s].lon))
            sl = [self.stops[s] for s in chosen]
            rid = f"R{r:03d}"
            self.routes[rid] = Route(route_id=rid,
                route_name=f"Route {r} ({sl[0].name} – {sl[-1].name})",
                mode=TransportMode.BUS, stops=sl, headway_min=rng.uniform(5,20))

        print(f"[loader] Synthetic: {len(self.stops)} stops, {len(self.routes)} routes")
        print(f"[loader] Named stops include: {[s.name for s in list(self.stops.values())[:6]]}")

    def _ensure_loaded(self):
        if not self._loaded:
            raise RuntimeError("Call .load() first.")