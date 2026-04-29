"""
data/schema.py — Typed data structures for stops, routes, and edges.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


class TransportMode(Enum):
    BUS    = "bus"
    METRO  = "metro"
    WALK   = "walk"


@dataclass(frozen=True)
class Stop:
    """A physical transit stop / station."""
    stop_id:   str
    name:      str
    lat:       float
    lon:       float
    mode:      TransportMode = TransportMode.BUS

    def __hash__(self):
        return hash(self.stop_id)


@dataclass
class Route:
    """A single transit route (bus line, metro line)."""
    route_id:    str
    route_name:  str
    mode:        TransportMode
    stops:       List[Stop]         = field(default_factory=list)
    headway_min: float              = 10.0   # average frequency in minutes


@dataclass(frozen=True)
class EdgeWeight:
    """
    Multi-dimensional weight vector on a graph edge.
    Objectives: [time_min, cost_inr, transfers, walking_m, co2_g]
    """
    time_min:   float
    cost_inr:   float
    transfers:  float           # fractional allowed (0 or 1 per edge; summed on path)
    walking_m:  float
    co2_g:      float

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.time_min, self.cost_inr, self.transfers, self.walking_m, self.co2_g)

    def __add__(self, other: "EdgeWeight") -> "EdgeWeight":
        return EdgeWeight(
            time_min  = self.time_min  + other.time_min,
            cost_inr  = self.cost_inr  + other.cost_inr,
            transfers = self.transfers + other.transfers,
            walking_m = self.walking_m + other.walking_m,
            co2_g     = self.co2_g     + other.co2_g,
        )

    @staticmethod
    def zero() -> "EdgeWeight":
        return EdgeWeight(0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class GraphEdge:
    """A directed edge in the multigraph."""
    src:        str              # stop_id
    dst:        str              # stop_id
    route_id:   Optional[str]   # None for walk edges
    mode:       TransportMode
    weight:     EdgeWeight
    distance_m: float = 0.0


@dataclass
class ParetoPath:
    """One path in the Pareto-optimal frontier."""
    nodes:       List[str]          # sequence of stop_ids
    edges:       List[GraphEdge]
    total_weight: EdgeWeight

    @property
    def edge_set(self) -> frozenset:
        """Frozenset of (src, dst, route_id) tuples for Jaccard distance."""
        return frozenset((e.src, e.dst, e.route_id) for e in self.edges)

    def summary(self) -> str:
        w = self.total_weight
        return (
            f"Time={w.time_min:.1f}min  Cost=₹{w.cost_inr:.1f}  "
            f"Transfers={int(w.transfers)}  Walk={w.walking_m:.0f}m  "
            f"CO₂={w.co2_g:.0f}g  Hops={len(self.nodes)-1}"
        )
