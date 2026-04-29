"""
config.py — Global configuration for the Urban Multimodal Routing system.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


# ── Objective indices ──────────────────────────────────────────────────────────
OBJ_TIME      = 0   # minutes
OBJ_COST      = 1   # INR
OBJ_TRANSFERS = 2   # count
OBJ_WALKING   = 3   # metres
OBJ_CO2       = 4   # grams

N_OBJECTIVES = 5
OBJECTIVE_NAMES = ["Time (min)", "Cost (INR)", "Transfers", "Walking (m)", "CO₂ (g)"]
OBJECTIVE_UNITS = ["min", "₹", "#", "m", "g"]


# ── Edge / transfer costs ──────────────────────────────────────────────────────
WALK_SPEED_MPS      = 1.4          # metres per second
BUS_AVG_SPEED_KMPH  = 18.0
METRO_AVG_SPEED_KMPH = 35.0

BUS_BASE_FARE_INR   = 5.0          # flat base
BUS_FARE_PER_KM     = 0.75
METRO_FARE_PER_KM   = 2.5
WALK_FARE           = 0.0

TRANSFER_TIME_PENALTY_MIN = 5.0    # minutes added per transfer
TRANSFER_CO2_OVERHEAD     = 0.0    # grams (accounted in idle time below)

# CO2 grams / passenger-km
CO2_BUS_G_PKM    = 68.0
CO2_METRO_G_PKM  = 41.0
CO2_WALK_G_PKM   = 0.0

# Maximum walking edge distance
MAX_WALK_TRANSFER_M = 600          # metres between stops for a walk link


# ── Algorithm ──────────────────────────────────────────────────────────────────
PARETO_HEAP_TIE_BREAKER = OBJ_TIME   # primary sort key in priority queue
MAX_LABELS_PER_NODE     = 500        # hard cap to prevent label explosion


# ── Top-K selection ────────────────────────────────────────────────────────────
DEFAULT_TOP_K           = 5
MIN_JACCARD_DISSIMILARITY = 0.25     # diversity constraint threshold


# ── Weighted-sum baseline default weights ─────────────────────────────────────
# Must sum to 1.0; order: time, cost, transfers, walking, co2
DEFAULT_WEIGHTS: Tuple[float, ...] = (0.40, 0.25, 0.15, 0.10, 0.10)


# ── Lexicographic baseline default priority ────────────────────────────────────
# List of objective indices in priority order
DEFAULT_LEX_ORDER: List[int] = [OBJ_TIME, OBJ_COST, OBJ_TRANSFERS, OBJ_WALKING, OBJ_CO2]


# ── Synthetic graph (used when no dataset is available) ───────────────────────
SYNTHETIC_N_STOPS   = 120
SYNTHETIC_N_ROUTES  = 30
SYNTHETIC_SEED      = 42
