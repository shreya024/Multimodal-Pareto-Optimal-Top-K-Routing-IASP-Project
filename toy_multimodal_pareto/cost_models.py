"""Mode-specific cost and time models for the synthetic network."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoadCostConfig:
    """Parameters for private road travel cost."""

    speed_kmph: float = 28.0
    mileage_kmpl: float = 15.0
    fuel_price_per_liter: float = 100.0
    time_value_per_minute: float = 2.0
    congestion_cost_weight: float = 1.0


@dataclass(frozen=True)
class MetroCostConfig:
    """Parameters for metro fare and travel time."""

    speed_kmph: float = 48.0
    base_fare: float = 10.0
    fare_per_km: float = 2.5


@dataclass(frozen=True)
class WalkingCostConfig:
    """Parameters for walking links."""

    speed_kmph: float = 5.0


def travel_time_minutes(distance_km: float, speed_kmph: float) -> float:
    """Return travel time in minutes for a distance and speed."""

    if speed_kmph <= 0:
        raise ValueError("speed_kmph must be positive")
    return (distance_km / speed_kmph) * 60.0


def road_time(distance_km: float, congestion_factor: float, config: RoadCostConfig) -> float:
    """Road travel time, inflated by congestion."""

    return travel_time_minutes(distance_km, config.speed_kmph) * congestion_factor


def road_cost(distance_km: float, time_minutes: float, congestion_factor: float, config: RoadCostConfig) -> float:
    """Road cost = fuel cost + value of time + congestion penalty."""

    fuel_cost = (distance_km / config.mileage_kmpl) * config.fuel_price_per_liter
    time_cost = config.time_value_per_minute * time_minutes
    congestion_penalty = config.congestion_cost_weight * max(0.0, congestion_factor - 1.0) * fuel_cost
    return fuel_cost + time_cost + congestion_penalty


def metro_time(distance_km: float, config: MetroCostConfig) -> float:
    """Metro travel time in minutes."""

    return travel_time_minutes(distance_km, config.speed_kmph)


def metro_cost(distance_km: float, config: MetroCostConfig) -> float:
    """Metro fare with base and distance components."""

    return config.base_fare + config.fare_per_km * distance_km


def walking_time(distance_km: float, config: WalkingCostConfig) -> float:
    """Walking travel time in minutes."""

    return travel_time_minutes(distance_km, config.speed_kmph)


def walking_cost() -> float:
    """Walking has no monetary cost in this model."""

    return 0.0
