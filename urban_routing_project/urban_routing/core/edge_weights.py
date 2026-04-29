"""
core/edge_weights.py — Computes multi-dimensional EdgeWeight vectors for each graph edge.

Weight dimensions:
  [0] time_min   — travel time including any wait / headway
  [1] cost_inr   — monetary fare
  [2] transfers  — 1 if this edge is an intermodal transfer, else 0
  [3] walking_m  — walking distance (non-zero only for walk edges)
  [4] co2_g      — CO₂ emissions in grams
"""
from __future__ import annotations

import math
from data.schema import EdgeWeight, TransportMode
import config as cfg


def bus_edge_weight(distance_m: float, headway_min: float = 10.0) -> EdgeWeight:
    """Weight for a bus segment of given distance_m."""
    speed_mps   = cfg.BUS_AVG_SPEED_KMPH * 1000 / 3600
    travel_min  = (distance_m / speed_mps) / 60.0
    wait_min    = headway_min / 2.0          # expected wait = half headway
    time_min    = travel_min + wait_min

    dist_km     = distance_m / 1000.0
    cost_inr    = cfg.BUS_BASE_FARE_INR + cfg.BUS_FARE_PER_KM * dist_km

    co2_g       = cfg.CO2_BUS_G_PKM * dist_km

    return EdgeWeight(
        time_min  = time_min,
        cost_inr  = cost_inr,
        transfers = 0.0,
        walking_m = 0.0,
        co2_g     = co2_g,
    )


def metro_edge_weight(distance_m: float, headway_min: float = 5.0) -> EdgeWeight:
    speed_mps  = cfg.METRO_AVG_SPEED_KMPH * 1000 / 3600
    travel_min = (distance_m / speed_mps) / 60.0
    wait_min   = headway_min / 2.0
    time_min   = travel_min + wait_min

    dist_km    = distance_m / 1000.0
    cost_inr   = cfg.METRO_FARE_PER_KM * dist_km

    co2_g      = cfg.CO2_METRO_G_PKM * dist_km

    return EdgeWeight(
        time_min  = time_min,
        cost_inr  = cost_inr,
        transfers = 0.0,
        walking_m = 0.0,
        co2_g     = co2_g,
    )


def walk_edge_weight(distance_m: float) -> EdgeWeight:
    time_min = (distance_m / cfg.WALK_SPEED_MPS) / 60.0
    return EdgeWeight(
        time_min  = time_min,
        cost_inr  = 0.0,
        transfers = 0.0,
        walking_m = distance_m,
        co2_g     = 0.0,
    )


def transfer_penalty(from_mode: TransportMode, to_mode: TransportMode) -> EdgeWeight:
    """
    Intermodal transfer node weight.
    Adds time penalty and increments the transfer counter.
    """
    return EdgeWeight(
        time_min  = cfg.TRANSFER_TIME_PENALTY_MIN,
        cost_inr  = 0.0,
        transfers = 1.0,
        walking_m = 0.0,
        co2_g     = 0.0,
    )


def mode_edge_weight(
    mode: TransportMode,
    distance_m: float,
    headway_min: float = 10.0,
) -> EdgeWeight:
    """Dispatch to the appropriate weight function by mode."""
    if mode == TransportMode.BUS:
        return bus_edge_weight(distance_m, headway_min)
    elif mode == TransportMode.METRO:
        return metro_edge_weight(distance_m, headway_min)
    elif mode == TransportMode.WALK:
        return walk_edge_weight(distance_m)
    else:
        raise ValueError(f"Unknown mode: {mode}")
