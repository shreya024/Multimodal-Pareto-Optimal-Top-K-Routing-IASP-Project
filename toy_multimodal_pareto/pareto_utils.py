"""Utilities for Pareto dominance and path reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


CostVector = tuple[float, float, int, float]


@dataclass
class Label:
    """A multi-objective routing label."""

    node: int
    costs: CostVector
    prev: Optional["Label"] = None
    previous_mode: Optional[str] = None
    route_id: Optional[str] = None
    active: bool = True


@dataclass(frozen=True)
class PathResult:
    """A reconstructed route and its objective vector."""

    nodes: list[int]
    modes: list[str]
    route_ids: list[str]
    costs: CostVector


def dominates(left: CostVector, right: CostVector) -> bool:
    """Return True when left strictly Pareto-dominates right."""

    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def is_dominated(candidate: CostVector, existing: list[Label]) -> bool:
    """Return True when any active existing label dominates or duplicates candidate."""

    return any(label.active and (label.costs == candidate or dominates(label.costs, candidate)) for label in existing)


def deactivate_dominated(candidate: CostVector, existing: list[Label]) -> None:
    """Mark labels dominated by candidate as inactive."""

    for label in existing:
        if label.active and dominates(candidate, label.costs):
            label.active = False


def reconstruct_path(label: Label) -> PathResult:
    """Reconstruct node, mode, route, and objective sequences from a destination label."""

    nodes: list[int] = []
    modes: list[str] = []
    route_ids: list[str] = []
    cursor: Optional[Label] = label

    while cursor is not None:
        nodes.append(cursor.node)
        if cursor.previous_mode is not None:
            modes.append(cursor.previous_mode)
        if cursor.route_id is not None:
            route_ids.append(cursor.route_id)
        cursor = cursor.prev

    nodes.reverse()
    modes.reverse()
    route_ids.reverse()
    return PathResult(nodes=nodes, modes=modes, route_ids=route_ids, costs=label.costs)


def pareto_filter_paths(paths: list[PathResult]) -> list[PathResult]:
    """Remove dominated duplicate paths from an already reconstructed path list."""

    result: list[PathResult] = []
    seen_costs: set[CostVector] = set()
    for i, path in enumerate(paths):
        if any(i != j and dominates(other.costs, path.costs) for j, other in enumerate(paths)):
            continue
        if path.costs in seen_costs:
            continue
        seen_costs.add(path.costs)
        result.append(path)
    return result
